//! P5 science gate (part 3): the BEM energy converges to the analytic Born energy.
//!
//! The strongest, NESSie-independent result: solve the *local BEM* on a triangulated
//! sphere with a central unit charge and read off the reaction-field energy; it must
//! converge to the closed-form Born energy as the mesh refines. This exercises the
//! whole local stack (collocation → assembly → GMRES → post) against analytic physics,
//! not against another implementation.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, rfenergy, solve_local_elements, solve_nonlocal_elements,
    solve_nonlocal_elements_q, AdaptiveConfig, Charge, Locality, Params, Quadrature, SolveConfig,
    Tri,
};

/// Nonlocal BEM energy with a selectable regular-Yukawa quadrature.
fn sphere_bem_energy_nonlocal_q(radius: f64, subdiv: u32, quad: Quadrature) -> (f64, usize) {
    let elements = sphere_elements(radius, subdiv);
    let charges = central_charge();
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, stats) =
        solve_nonlocal_elements_q(&elements, &charges, &params(), &cfg, quad).expect("solve");
    (rfenergy(&elements, &charges, &res), stats.capped_panels)
}

#[test]
fn nonlocal_adaptive_matches_fixed_on_convex_sphere() {
    // Honest gate, NOT "adaptive tightens the sphere Born": a convex sphere has no
    // opposing/cleft surfaces, so it has essentially no near-singular element pairs —
    // its few-percent nonlocal floor is *general discretisation error*, not the
    // near-singular artifact. The near-singular remediation therefore (correctly) leaves
    // the sphere energy ~unchanged. The proof that adaptive lifts the actual near-field
    // error is the opposing-panel cleft gate in `adaptive.rs::cleft_opposing_panels_*`,
    // where the fixed rule is off by 0.3%/3% per cross-entry and adaptive by ~1e-7.
    //
    // This asserts (a) adaptive ≈ fixed here (no spurious change on a clean convex mesh)
    // and (b) the self/diagonal panels do NOT spuriously cap (the on-panel term uses the
    // analytic limit, not subdivision).
    let radius = 2.0;
    let born = born_rfenergy(1.0, radius, &params(), Locality::Nonlocal);
    for s in [1u32, 2, 3] {
        let (ef, _) = sphere_bem_energy_nonlocal_q(radius, s, Quadrature::Fixed);
        let (ea, capped) = sphere_bem_energy_nonlocal_q(
            radius,
            s,
            Quadrature::Adaptive(AdaptiveConfig::default()),
        );
        let rf = (ef - born).abs() / born.abs();
        let ra = (ea - born).abs() / born.abs();
        eprintln!("subdiv {s}: fixed rel {rf:.4}  adaptive rel {ra:.4}  (capped {capped})");
        assert_eq!(capped, 0, "subdiv {s}: self panels must not cap");
        // Convex sphere: adaptive and fixed agree closely (no near-singular regime).
        assert!(
            (ea - ef).abs() / born.abs() < 5e-3,
            "subdiv {s}: adaptive {ea} vs fixed {ef} diverge unexpectedly on a convex mesh"
        );
    }
}

fn params() -> Params {
    Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

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

fn central_charge() -> [Charge; 1] {
    [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }]
}

/// Local BEM reaction-field energy of a central unit charge in a radius-`R` icosphere.
fn sphere_bem_energy(radius: f64, subdiv: u32) -> f64 {
    let elements = sphere_elements(radius, subdiv);
    let charges = central_charge();
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&elements, &charges, &params(), &cfg).expect("solve");
    rfenergy(&elements, &charges, &res)
}

/// Nonlocal BEM reaction-field energy of a central unit charge (uses `u`/`q`, same
/// `rfenergy` formula as local — the nonlocal third block `w` enters the *solve*).
fn sphere_bem_energy_nonlocal(radius: f64, subdiv: u32) -> f64 {
    let elements = sphere_elements(radius, subdiv);
    let charges = central_charge();
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, _) = solve_nonlocal_elements(&elements, &charges, &params(), &cfg).expect("solve");
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

#[test]
fn nonlocal_bem_energy_matches_born_within_radon_floor() {
    // P6 physics sanity gate: the *nonlocal* BEM energy agrees with the closed-form
    // nonlocal Born energy to a few percent across mesh resolutions — a broad check
    // that the whole nonlocal stack (Laplace + regular-Yukawa kernels → 3-block
    // assembly → GMRES → post) produces a physically right energy. It is NOT evidence
    // of discretisation convergence (see the note below) and covers only a centred,
    // highly symmetric charge; the entrywise assembly + Cauchy-data parity gates are
    // what pin the operator/RHS. A gross block-placement or sign error would land far
    // outside the few-percent band, but a subtle one need not.
    //
    // NOTE — unlike the local case (analytic Laplace collocation, monotone
    // convergence), the nonlocal energy does NOT converge monotonically: the fixed
    // 7-point Radon cubature for the regular-Yukawa kernel loses accuracy for
    // near-neighbour elements as the mesh refines (the documented P6.5 near-singular
    // limitation), so agreement plateaus at the few-percent level. Tight nonlocal
    // convergence needs the P6.5 adaptive-quadrature work; this gate pins the physics,
    // not the convergence rate.
    let radius = 2.0;
    let born = born_rfenergy(1.0, radius, &params(), Locality::Nonlocal);
    assert!(
        born < 0.0,
        "nonlocal Born energy should be negative: {born}"
    );

    for s in [1u32, 2, 3] {
        let e = sphere_bem_energy_nonlocal(radius, s);
        let rel = (e - born).abs() / born.abs();
        assert!(e < 0.0, "nonlocal BEM energy sign wrong at subdiv {s}: {e}");
        assert!(
            rel < 0.05,
            "subdiv {s}: nonlocal BEM {e} vs Born {born} off by {rel:.3} (> 5%)"
        );
    }
}
