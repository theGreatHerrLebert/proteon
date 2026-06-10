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

    // Refine the icosphere: subdivision 2 (320 triangles) → 3 (1280).
    let e2 = sphere_bem_energy(radius, 2);
    let e3 = sphere_bem_energy(radius, 3);

    let rel2 = (e2 - born).abs() / born.abs();
    let rel3 = (e3 - born).abs() / born.abs();

    // Right sign + ballpark already at the coarse mesh.
    assert!(e2 < 0.0 && e3 < 0.0, "BEM energy sign wrong: {e2}, {e3}");
    assert!(rel2 < 0.10, "coarse mesh too far from Born: {rel2:.3}");
    // Refinement must move toward the analytic value, and the fine mesh be close.
    assert!(rel3 < rel2, "not converging: {rel2:.4} → {rel3:.4}");
    assert!(
        rel3 < 0.03,
        "subdivision 3 not within 3% of Born: {rel3:.4} (e3={e3}, born={born})"
    );
}
