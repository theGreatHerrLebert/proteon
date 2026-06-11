//! Off-centre (l>0) science gate: the single-region BEM vs the Kirkwood series.
//!
//! The Born gate uses a CENTRAL charge — only the `l=0` monopole. An off-centre charge
//! excites higher multipoles, so matching the Kirkwood closed form validates the BEM's
//! `l>0` behaviour (non-constant surface data), independent of any other implementation.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, kirkwood_rfenergy, rfenergy, solve_local_elements, Charge,
    Locality, Params, SolveConfig, Tri,
};

const EPS_OMEGA: f64 = 1.0;
const EPS_SIGMA: f64 = 78.0;

fn params() -> Params {
    Params {
        eps_omega: EPS_OMEGA,
        eps_sigma: EPS_SIGMA,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

fn sphere_bem_energy(radius: f64, subdiv: u32, offset: f64) -> f64 {
    let mesh = analytic_sphere_mesh(radius, subdiv);
    let els: Vec<Tri> = mesh
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
        pos: Vec3::new(offset, 0.0, 0.0),
        val: 1.0,
    }];
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&els, &charges, &params(), &cfg).expect("solve");
    rfenergy(&els, &charges, &res)
}

#[test]
fn kirkwood_offset_zero_is_born() {
    // Sanity: the n=0 Kirkwood term IS Born, so offset=0 reproduces born_rfenergy.
    for &r in &[1.5_f64, 2.0, 3.0] {
        let k = kirkwood_rfenergy(1.0, r, 0.0, EPS_OMEGA, EPS_SIGMA, 40);
        let b = born_rfenergy(1.0, r, &params(), Locality::Local);
        assert!((k - b).abs() / b.abs() < 1e-12, "r={r}: {k} vs {b}");
    }
}

#[test]
fn offcenter_bem_matches_kirkwood() {
    let radius = 2.0;
    let subdiv = 3; // 1280 triangles
    // As the charge moves off-centre the reaction field strengthens (more negative); the
    // BEM must track the Kirkwood series at each offset, not just the l=0 value.
    for &offset in &[0.0_f64, 0.7, 1.2, 1.5] {
        let bem = sphere_bem_energy(radius, subdiv, offset);
        let kirk = kirkwood_rfenergy(1.0, radius, offset, EPS_OMEGA, EPS_SIGMA, 60);
        let rel = (bem - kirk).abs() / kirk.abs();
        eprintln!("offset {offset}: BEM {bem:.3} vs Kirkwood {kirk:.3} (rel {rel:.3})");
        assert!(bem < 0.0 && kirk < 0.0, "offset {offset}: both negative");
        assert!(rel < 0.04, "offset {offset}: BEM off Kirkwood by {rel:.3} (> 4%)");
    }

    // The off-centre energy must be meaningfully MORE negative than the central (l>0
    // contribution is real, not negligible) — otherwise the test could pass on l=0 alone.
    let central = kirkwood_rfenergy(1.0, radius, 0.0, EPS_OMEGA, EPS_SIGMA, 60);
    let edge = kirkwood_rfenergy(1.0, radius, 1.5, EPS_OMEGA, EPS_SIGMA, 60);
    assert!(edge < 1.15 * central, "off-centre must add real l>0 reaction: {edge} vs {central}");
}
