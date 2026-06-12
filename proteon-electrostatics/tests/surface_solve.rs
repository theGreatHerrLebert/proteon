//! Direct gate on the pure `solve_surface` orchestration (shared by the PyO3
//! connector and the `proteon electrostatics` CLI). Confirms the mesh-in /
//! potential-out front-end converges, reports correct topology, and lands on the
//! analytic Born energy — independent of either presentation layer.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, solve_surface, Charge, Locality, Params,
    SurfaceSolveError, SurfaceSolveOptions,
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn solve_surface_central_charge_matches_born() {
    let (radius, eo, es) = (2.0, 1.0, 78.0);
    let mesh = analytic_sphere_mesh(radius, 4);
    let nv = mesh.verts.len();
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let params = Params {
        eps_omega: eo,
        eps_sigma: es,
        eps_inf: 1.8,
        lambda: 20.0,
    };
    let opts = SurfaceSolveOptions {
        params,
        ..Default::default()
    };

    let sol = solve_surface(mesh, &charges, &opts).result.expect("solve");

    assert!(sol.converged, "did not converge");
    assert!(sol.topology.watertight && sol.topology.consistently_oriented);
    assert_eq!(sol.potential.len(), nv, "one potential value per vertex");
    assert_eq!(sol.charge_components.len(), 1);
    assert!(sol.rfenergy < 0.0, "solvation energy should be negative");

    let born = born_rfenergy(1.0, radius, &params, Locality::Local);
    let r = rel(sol.rfenergy, born);
    eprintln!(
        "solve_surface {:.4} vs Born {:.4} (rel {r:.4})",
        sol.rfenergy, born
    );
    assert!(r < 0.02, "BEM off Born by {r:.4} (> 2%)");
}

#[test]
fn solve_surface_empty_inputs_error() {
    let mesh = analytic_sphere_mesh(2.0, 2);
    let opts = SurfaceSolveOptions::default();
    // No charges → Empty. (matches! avoids requiring SurfaceSolution: Debug.)
    assert!(matches!(
        solve_surface(mesh, &[], &opts).result,
        Err(SurfaceSolveError::Empty)
    ));
}

#[test]
fn solve_surface_coarse_mesh_under_budget_ok() {
    // A coarse sphere is well under the dense-matrix budget — the happy path must
    // not be spuriously refused by the memory guard.
    let mesh = analytic_sphere_mesh(2.0, 3);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let opts = SurfaceSolveOptions::default();
    assert!(solve_surface(mesh, &charges, &opts).result.is_ok());
}

#[test]
fn solve_surface_invalid_params_error() {
    // A non-positive dielectric must be rejected by the shared validation (so the CLI,
    // which doesn't pre-validate, never reaches Params::yukawa's debug_assert).
    let mesh = analytic_sphere_mesh(2.0, 2);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let opts = SurfaceSolveOptions {
        params: Params {
            eps_omega: 1.0,
            eps_sigma: 78.0,
            eps_inf: 1.8,
            lambda: 0.0,
        },
        ..Default::default()
    };
    assert!(matches!(
        solve_surface(mesh, &charges, &opts).result,
        Err(SurfaceSolveError::InvalidParams(_))
    ));
}
