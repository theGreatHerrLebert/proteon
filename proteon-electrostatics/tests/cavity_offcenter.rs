//! Off-centre cavity science gate (codex Layer 2): an OFF-CENTRE charge in a buried
//! cavity, BEM vs the multi-shell Kirkwood series (cross-interface multipole coupling).

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    concentric_kirkwood_rfenergy, concentric_shell_rfenergy, kirkwood_rfenergy, rfenergy,
    solve_local_elements, Charge, Params, SolveConfig, Tri,
};
use proteon_core::surface::mesh::icosphere;

const EO: f64 = 1.0;
const ES: f64 = 78.0;

fn params() -> Params {
    Params {
        eps_omega: EO,
        eps_sigma: ES,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

#[test]
fn concentric_kirkwood_reductions() {
    // One interface ⇒ single-sphere Kirkwood, at several offsets (incl. l>0).
    for &s in &[0.0_f64, 0.5, 1.0] {
        let multi = concentric_kirkwood_rfenergy(1.0, s, EO, &[(2.0, ES)], 60);
        let single = kirkwood_rfenergy(1.0, 2.0, s, EO, ES, 60);
        assert!((multi - single).abs() / single.abs() < 1e-12, "s={s}: {multi} vs {single}");
    }
    // offset = 0 ⇒ the l=0 concentric-shell sum (multi-interface).
    let shells = [(1.0, ES), (2.0, EO), (3.0, ES)];
    let multi0 = concentric_kirkwood_rfenergy(1.0, 0.0, EO, &shells, 60);
    let l0 = concentric_shell_rfenergy(
        1.0,
        &[(1.0, EO, ES), (2.0, ES, EO), (3.0, EO, ES)],
    );
    assert!((multi0 - l0).abs() / l0.abs() < 1e-12, "offset 0: {multi0} vs {l0}");
}

#[test]
fn offcenter_cavity_bem_matches_kirkwood() {
    // island (r<1, εΩ) / cavity (1<r<2, εΣ) / body (2<r<3, εΩ) / exterior (εΣ); a unit
    // charge OFF-CENTRE in the island.
    let subdiv = 3;
    let mut mesh = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, subdiv);
    mesh.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, subdiv));
    mesh.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, subdiv));
    assert!(mesh.orient_by_nesting());

    let els: Vec<Tri> = mesh
        .tris
        .iter()
        .map(|t| Tri::new(mesh.verts[t[0] as usize], mesh.verts[t[1] as usize], mesh.verts[t[2] as usize]))
        .collect();
    let shells = [(1.0, ES), (2.0, EO), (3.0, ES)];

    for &offset in &[0.0_f64, 0.4, 0.7] {
        let charges = [Charge {
            pos: Vec3::new(offset, 0.0, 0.0),
            val: 1.0,
        }];
        let cfg = SolveConfig {
            tol: 1e-9,
            ..Default::default()
        };
        let (res, _) = solve_local_elements(&els, &charges, &params(), &cfg).expect("solve");
        let bem = rfenergy(&els, &charges, &res);
        let kirk = concentric_kirkwood_rfenergy(1.0, offset, EO, &shells, 60);
        let rel = (bem - kirk).abs() / kirk.abs();
        eprintln!("offset {offset}: cavity BEM {bem:.3} vs Kirkwood {kirk:.3} (rel {rel:.3})");
        assert!(bem < 0.0 && kirk < 0.0);
        assert!(rel < 0.04, "offset {offset}: off Kirkwood by {rel:.3} (> 4%)");
    }
}
