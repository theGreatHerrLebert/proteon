//! P5 science gate (part 3): the BEM energy converges to the analytic Born energy.
//!
//! The strongest, NESSie-independent result: solve the *local BEM* on a triangulated
//! sphere with a central unit charge and read off the reaction-field energy; it must
//! converge to the closed-form Born energy as the mesh refines. This exercises the
//! whole local stack (collocation → assembly → GMRES → post) against analytic physics,
//! not against another implementation.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, rfenergy, solve_local_elements, Charge, Locality, Params,
    SolveConfig, Tri,
};

fn params() -> Params {
    Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

/// Local BEM reaction-field energy of a unit charge at the centre of a radius-`R`
/// icosphere with `subdiv` subdivisions.
fn sphere_bem_energy(radius: f64, subdiv: u32) -> f64 {
    let mesh = analytic_sphere_mesh(radius, subdiv);
    let elements: Vec<Tri> = mesh
        .tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect();
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&elements, &charges, &params(), &cfg).expect("solve");
    rfenergy(&elements, &charges, &res)
}

#[test]
fn bem_energy_converges_to_born() {
    let radius = 2.0;
    let born = born_rfenergy(1.0, radius, &params(), Locality::Local);
    assert!(
        born < 0.0,
        "Born energy should be negative solvation: {born}"
    );

    // Three refinement levels: icosphere subdivisions 1 → 2 → 3 (80 → 320 → 1280
    // triangles). Require the error to fall *monotonically* toward the analytic value
    // at every step — two levels alone could improve by accident or limit to a wrong
    // value within a loose band.
    let rels: Vec<f64> = [1u32, 2, 3]
        .iter()
        .map(|&s| {
            let e = sphere_bem_energy(radius, s);
            assert!(e < 0.0, "BEM energy sign wrong at subdiv {s}: {e}");
            (e - born).abs() / born.abs()
        })
        .collect();

    assert!(
        rels[0] > rels[1] && rels[1] > rels[2],
        "error not monotonically decreasing: {rels:?}"
    );
    // Each refinement must cut the error appreciably — a method limiting to the wrong
    // value would stall instead of keep shrinking.
    assert!(
        rels[2] / rels[1] < 0.8,
        "refinement stalled (ratio {:.3}): {rels:?}",
        rels[2] / rels[1]
    );
    assert!(
        rels[2] < 0.03,
        "subdivision 3 not within 3% of Born: {:.4} (rels={rels:?}, born={born})",
        rels[2]
    );
}
