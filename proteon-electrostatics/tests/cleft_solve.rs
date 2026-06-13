//! P6.5 solve-level near-singular demonstration on a **non-convex** geometry.
//!
//! The convex sphere has no opposing surfaces, so adaptive ≈ fixed there (its floor is
//! general discretisation, gated in `born_convergence.rs`). Two close icospheres create
//! a genuine **cleft** — facing caps across a narrow gap — so the near-singular
//! regular-Yukawa cross-entries are now a real contribution to the assembled system.
//! This is where the adaptive remediation must visibly change (and improve) the solve.
//!
//! There is no closed form for two spheres, so the metric is **mesh self-consistency**:
//! a fine adaptive solve is the reference; coarse adaptive must approach it while fixed
//! retains a near-singular gap.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, rfenergy, solve_nonlocal_elements_q, AdaptiveConfig, Charge, Params,
    Quadrature, SolveConfig, Tri,
};

fn params() -> Params {
    Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

/// Two radius-`R` icospheres centred at `±(R + gap/2)` on z, so their facing surfaces
/// are `gap` apart — a cleft when `gap` < element size. Each sphere carries a unit
/// charge at height `charge_depth` *inside* its facing cap (`charge_depth` measured from
/// the cap toward the centre): small ⇒ the reaction field is dominated by the
/// near-singular cleft region; `= R` ⇒ the charge sits at the centre.
fn two_sphere(radius: f64, gap: f64, subdiv: u32, charge_depth: f64) -> (Vec<Tri>, Vec<Charge>) {
    let cz = radius + gap / 2.0;
    let mut elements = Vec::new();
    for sign in [1.0_f64, -1.0] {
        let mesh = analytic_sphere_mesh(radius, subdiv);
        let off = Vec3::new(0.0, 0.0, sign * cz);
        for t in &mesh.tris {
            let v = |k: u32| mesh.verts[k as usize] + off;
            elements.push(Tri::new(v(t[0]), v(t[1]), v(t[2])));
        }
    }
    // Facing cap of the +z sphere is at z = gap/2; place the charge `charge_depth` above
    // it (toward the centre) — i.e. near the cleft when charge_depth is small.
    let qz = gap / 2.0 + charge_depth;
    let charges = vec![
        Charge {
            pos: Vec3::new(0.0, 0.0, qz),
            val: 1.0,
        },
        Charge {
            pos: Vec3::new(0.0, 0.0, -qz),
            val: 1.0,
        },
    ];
    (elements, charges)
}

fn nonlocal_energy(elements: &[Tri], charges: &[Charge], quad: Quadrature) -> (f64, usize) {
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };
    let (res, stats) =
        solve_nonlocal_elements_q(elements, charges, &params(), &cfg, quad).expect("solve");
    (rfenergy(elements, charges, &res), stats.capped_panels)
}

/// The honest solve-level result (this surprised us — record it as a gate, not a
/// manufactured win):
///
/// Adaptive **fixes** the near-singular regular-Yukawa cross-entries — proven at the
/// kernel level, where the fixed 7-point rule is off 0.3%/3% per opposing-panel entry
/// and adaptive ~1e-7 (`adaptive.rs::cleft_opposing_panels_beats_fixed`). But that
/// per-entry error does **not** propagate to the integrated reaction-field energy: even
/// with the charges sitting *at* the cleft and the gap squeezed to 0.1, the solved
/// nonlocal energy moves by only ~1e-6 (fixed vs adaptive). The global GMRES solution
/// averages out a handful of slightly-wrong entries, and the energy is dominated by each
/// component's self-solvation, which both methods compute identically.
///
/// Consequence (design `devdocs/NEAR_SINGULAR_QUADRATURE.md` §5): **do not flip the
/// default to adaptive** — it would impose adaptive's subdivision cost on every nonlocal
/// solve for ~1e-6 of energy accuracy on every testable geometry. Adaptive stays opt-in
/// and correct; whether near-singular ever materially moves an energy awaits a true SES
/// re-entrant (toric/saddle) mesh, which is not yet generable analytically in-tree.
#[test]
fn cleft_solve_energy_insensitive_to_near_singular() {
    let radius = 2.0;
    let subdiv = 1; // 160 triangles → 480 unknowns; coarse + tight gap = worst case
    let adaptive = Quadrature::Adaptive(AdaptiveConfig::default());
    // charge at centre (weakly coupled) and at the cleft (max coupling), tight gaps.
    for charge_depth in [radius, 0.5] {
        for gap in [0.5_f64, 0.25, 0.1] {
            let (elements, charges) = two_sphere(radius, gap, subdiv, charge_depth);
            let (ef, _) = nonlocal_energy(&elements, &charges, Quadrature::Fixed);
            let (ea, capped) = nonlocal_energy(&elements, &charges, adaptive);
            let rel = (ea - ef).abs() / ea.abs();
            eprintln!(
                "depth {charge_depth} gap {gap}: fixed {ef:.4} adaptive {ea:.4} Δ {rel:.2e} (capped {capped})"
            );
            assert_eq!(capped, 0, "no panel should cap on this geometry");
            // The remediation does not materially change the energy here (observed ~3e-6;
            // a 1e-4 bound is the documented "energy-insensitive" finding, not a target).
            assert!(
                rel < 1e-4,
                "depth {charge_depth} gap {gap}: adaptive moved the energy by {rel:.2e} \
                 (>1e-4) — re-examine the near-singular energy sensitivity claim"
            );
        }
    }
}
