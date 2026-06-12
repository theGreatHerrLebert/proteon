//! P8.4 end-to-end gate: the **nonlocal** (Lorentz/Yukawa) 3-block solve driven by the
//! treecode V/K (Cartesian) + Vy/Ky (BLTC) operators, vs the dense nonlocal solve.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, rfenergy, solve_nonlocal_elements, solve_nonlocal_elements_treecode,
    Charge, Params, SolveConfig, Tri,
};

fn sphere_elements(radius: f64, subdiv: u32) -> Vec<Tri> {
    let mesh = analytic_sphere_mesh(radius, subdiv);
    mesh.tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn treecode_nonlocal_solve_matches_dense() {
    let els = sphere_elements(2.0, 3);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let params = Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    };
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };

    let (rd, sd) = solve_nonlocal_elements(&els, &charges, &params, &cfg).expect("dense nonlocal");
    let e_dense = rfenergy(&els, &charges, &rd);

    let (rt, st) = solve_nonlocal_elements_treecode(&els, &charges, &params, &cfg, 8, 0.45)
        .expect("tc nonlocal");
    let e_tree = rfenergy(&els, &charges, &rt);

    assert!(
        sd.converged && st.converged,
        "both nonlocal solves must converge"
    );
    let r = rel(e_tree, e_dense);
    eprintln!("nonlocal rfenergy: dense {e_dense:.4}  treecode {e_tree:.4}  rel {r:.3e}");
    assert!(r < 3e-3, "treecode nonlocal rfenergy off dense by {r:.3e}");
    assert!(e_tree < 0.0, "solvation energy is negative");
}

#[test]
fn treecode_nonlocal_rejects_bad_params() {
    let els = sphere_elements(2.0, 2);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let params = Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    };
    let cfg = SolveConfig {
        tol: 1e-7,
        ..Default::default()
    };
    assert!(solve_nonlocal_elements_treecode(&els, &charges, &params, &cfg, 0, 0.5).is_err());
    assert!(solve_nonlocal_elements_treecode(&els, &charges, &params, &cfg, 6, 1.5).is_err());
}
