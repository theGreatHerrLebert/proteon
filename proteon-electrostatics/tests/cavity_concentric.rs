//! Multi-region (cavity) science gate — the codex-corrected formulation.
//!
//! Codex round 1 established that a buried solvent cavity needs NO operator change: with
//! every component oriented *outward-from-solute* (sign `(−1)^nesting_depth` — body +,
//! cavity −, island +), the scalar `f = εΩ/εΣ` single-region solve is already correct.
//! This gates exactly that: a central charge in three concentric shells (solute island /
//! solvent cavity / solute body) solved by the EXISTING `solve_local_elements` on the
//! nesting-oriented mesh must match the closed-form `l=0` concentric reaction energy.

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::{icosphere, Mesh};
use proteon_electrostatics::{
    born_rfenergy, concentric_shell_rfenergy, rfenergy, solve_local_elements, Charge, Locality,
    Params, SolveConfig, Tri,
};

const EPS_OMEGA: f64 = 1.0; // vacuum solute (matches the validated Born convention)
const EPS_SIGMA: f64 = 78.0;

fn params() -> Params {
    Params {
        eps_omega: EPS_OMEGA,
        eps_sigma: EPS_SIGMA,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

fn elements(mesh: &Mesh) -> Vec<Tri> {
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

fn solve_energy(mesh: &Mesh, charges: &[Charge]) -> f64 {
    let els = elements(mesh);
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&els, charges, &params(), &cfg).expect("solve");
    rfenergy(&els, charges, &res)
}

#[test]
fn concentric_single_interface_is_born() {
    // Sanity: one interface ⇒ the concentric formula IS Born.
    for &r in &[1.5_f64, 2.0, 3.0] {
        let conc = concentric_shell_rfenergy(1.0, &[(r, EPS_OMEGA, EPS_SIGMA)]);
        let born = born_rfenergy(1.0, r, &params(), Locality::Local);
        assert!(
            (conc - born).abs() / born.abs() < 1e-12,
            "r={r}: {conc} vs {born}"
        );
    }
}

#[test]
fn cavity_bem_matches_concentric_analytic() {
    // Geometry: solute island (r<1), solvent cavity (1<r<2), solute body (2<r<3),
    // exterior solvent (r>3); a unit charge at the centre (in the island, solute).
    let subdiv = 3;
    let mut mesh = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, subdiv); // body
    mesh.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, subdiv)); // cavity
    mesh.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, subdiv)); // island

    // Orient outward-from-solute: body +, cavity −, island +.
    assert!(mesh.orient_by_nesting());
    assert_eq!(mesh.component_nesting_depths(), vec![0, 1, 2]);

    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let bem = solve_energy(&mesh, &charges);

    // Interfaces (radius, ε_inside, ε_outside): island/cavity at 1, cavity/body at 2,
    // body/exterior at 3 — alternating solute/solvent.
    let analytic = concentric_shell_rfenergy(
        1.0,
        &[
            (1.0, EPS_OMEGA, EPS_SIGMA),
            (2.0, EPS_SIGMA, EPS_OMEGA),
            (3.0, EPS_OMEGA, EPS_SIGMA),
        ],
    );
    let rel = (bem - analytic).abs() / analytic.abs();
    eprintln!("cavity BEM {bem:.4} vs concentric analytic {analytic:.4} (rel {rel:.3})");
    assert!(bem < 0.0 && analytic < 0.0, "both energies negative");
    // Discretisation floor like the single-sphere Born gate (observed ~0.4%).
    assert!(
        rel < 0.02,
        "cavity BEM off concentric analytic by {rel:.3} (> 2%)"
    );

    // NEGATIVE CONTROL: orientation matters. Solve the SAME geometry with every shell
    // left +volume (no nesting orientation) — it must disagree with the analytic, proving
    // the result is not orientation-insensitive (a shared-mistake guard).
    let mut wrong = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, subdiv);
    wrong.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, subdiv));
    wrong.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, subdiv));
    // (icospheres are all built +volume; no orient_by_nesting → the cavity is mis-oriented)
    let bem_wrong = solve_energy(&wrong, &charges);
    let rel_wrong = (bem_wrong - analytic).abs() / analytic.abs();
    eprintln!("mis-oriented BEM {bem_wrong:.4} (rel {rel_wrong:.3})");
    assert!(
        rel_wrong > 0.1,
        "mis-oriented solve must DISAGREE (orientation is load-bearing)"
    );
}
