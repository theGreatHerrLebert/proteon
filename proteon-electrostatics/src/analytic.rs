//! Closed-form analytical models + the analytic sphere mesh (L4 ground truth).
//!
//! Port of NESSie's `TestModel` (`src/testmodel/`): the Born ion and Xie
//! multi-charge-sphere models give **exact** reaction-field energies/potentials for
//! spherically-symmetric systems — independent of any BEM path, the strongest gate.
//!
//! Gate these against **externally-generated** high-precision fixtures (CAS/mpmath),
//! not only against NESSie — porting the formulas into Rust re-introduces shared
//! transcription + Bessel-function risk ("an oracle for the oracle", plan §6).

use crate::model::{Locality, Params};
use crate::post::{ENERGY_FACTOR, POTPREFACTOR};
use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::{icosphere, Mesh};

/// Closed-form Born solvation energy (kJ/mol) of a single ion of charge `ζ` and
/// radius `R`. NESSie `rfenergy(LocalES|NonlocalES, ion)` = `ζ·rfpotential(:Ω)·energy_factor`:
///
/// ```text
/// local:    potprefactor·energy_factor · ζ²/R · (1/εΣ − 1)
/// nonlocal: potprefactor·energy_factor · ζ²/(R·εΣ) · (1 − εΣ + (εΣ−ε∞)/ε∞ · sinh(ν)/ν · e^(−ν))
///           with ν = √(εΣ/ε∞)·R/λ
/// ```
///
/// NESSie's Born formula assumes a **vacuum solute** (`εΩ = 1`): the local term is
/// `(1/εΣ − 1)`, not `(1/εΣ − 1/εΩ)`. `eps_omega` is therefore ignored here; pass a
/// vacuum-solute ion (the model defaults do).
#[must_use]
pub fn born_rfenergy(charge: f64, radius: f64, params: &Params, locality: Locality) -> f64 {
    // The Born model is derived for a vacuum solute; the local term is (1/εΣ − 1),
    // not (1/εΣ − 1/εΩ). `eps_omega` is deliberately unused — flag a non-vacuum value
    // rather than silently returning an unvalidated generalization.
    debug_assert!(
        (params.eps_omega - 1.0).abs() < 1e-9,
        "Born model assumes a vacuum solute (eps_omega = 1)"
    );
    let pre = POTPREFACTOR * ENERGY_FACTOR * charge * charge / radius;
    match locality {
        Locality::Local => pre * (1.0 / params.eps_sigma - 1.0),
        Locality::Nonlocal => {
            let nu = (params.eps_sigma / params.eps_inf).sqrt() * radius / params.lambda;
            let bulk = (params.eps_sigma - params.eps_inf) / params.eps_inf;
            let factor = 1.0 - params.eps_sigma + bulk * nu.sinh() / nu * (-nu).exp();
            pre / params.eps_sigma * factor
        }
    }
}

/// Closed-form `l=0` reaction-field energy (kJ/mol) of a **central** point charge `q` in
/// concentric dielectric shells (codex multi-region oracle). `interfaces` are the
/// `(radius, ε_inside, ε_outside)` triples in any order. The monopole reaction potential
/// at the centre is `φ = q · Σ_k (1/ε_out − 1/ε_in)/r_k`, and `W* = ½ q φ · prefactor`
/// (the ½ lives in [`ENERGY_FACTOR`]). The single-interface case is exactly the Born
/// energy, so this gates the multi-region (cavity) BEM the way Born gates the single
/// region.
#[must_use]
pub fn concentric_shell_rfenergy(charge: f64, interfaces: &[(f64, f64, f64)]) -> f64 {
    let sum: f64 = interfaces
        .iter()
        .map(|&(r, ein, eout)| (1.0 / eout - 1.0 / ein) / r)
        .sum();
    POTPREFACTOR * ENERGY_FACTOR * charge * charge * sum
}

/// Kirkwood (1934) reaction-field energy (kJ/mol) of a point charge `q` at distance
/// `offset < radius` from the centre of a dielectric sphere (interior `eps_in`, radius
/// `radius`, exterior `eps_out`). Unlike Born (a central charge, `l=0` only), an
/// off-centre charge excites higher multipoles, so this gates the BEM at `l>0`:
///
/// ```text
/// W* = prefactor · q²/a · Σ_{n≥0} (n+1)(ε_in − ε_out) / (ε_in·(n·ε_in + (n+1)·ε_out)) · (s/a)^{2n}
/// ```
///
/// The `n=0` term is `(1/ε_out − 1/ε_in)` — exactly Born — so `offset = 0` reproduces
/// [`born_rfenergy`]. `n_terms` truncates the series (it converges geometrically in
/// `(s/a)²`).
#[must_use]
pub fn kirkwood_rfenergy(
    charge: f64,
    radius: f64,
    offset: f64,
    eps_in: f64,
    eps_out: f64,
    n_terms: usize,
) -> f64 {
    let x2 = (offset / radius).powi(2);
    let mut series = 0.0;
    let mut pow = 1.0; // (s/a)^{2n}
    for n in 0..n_terms {
        let nf = n as f64;
        let coeff =
            (nf + 1.0) * (eps_in - eps_out) / (eps_in * (nf * eps_in + (nf + 1.0) * eps_out));
        series += coeff * pow;
        pow *= x2;
    }
    POTPREFACTOR * ENERGY_FACTOR * charge * charge / radius * series
}

/// Generalized (multi-shell) Kirkwood reaction-field energy (kJ/mol) of a point charge
/// `q` at `offset` from the centre, **in the innermost region** of concentric dielectric
/// shells. `eps_inner` is the innermost dielectric (the charge's region); `shells` are the
/// `(interface_radius, dielectric_outside)` triples in ASCENDING radius. This is the
/// off-centre cavity oracle: it couples all multipoles across every interface.
///
/// Implemented by the numerically **stable** layered-media reflection recursion (Möbius
/// maps + multiplicative `(r_k/r_{k+1})^{2l+1}` propagation), not the ill-conditioned
/// transfer-matrix product. Reductions (both gated): one interface ⇒ [`kirkwood_rfenergy`];
/// `offset = 0` ⇒ [`concentric_shell_rfenergy`] (the `l=0` term).
#[must_use]
pub fn concentric_kirkwood_rfenergy(
    charge: f64,
    offset: f64,
    eps_inner: f64,
    shells: &[(f64, f64)],
    n_terms: usize,
) -> f64 {
    assert!(!shells.is_empty(), "need at least one interface");
    let s = offset;
    let i1 = shells[0].0; // innermost interface radius
    let mut phi = 0.0; // Σ_l (R₀/ε_inner)·(s/I₁)^{2l} / I₁
    for l in 0..n_terms {
        let lf = l as f64;
        // Reflection ratio R = (growing value)/(decaying value) at an interface; the
        // exterior region is purely decaying, so R = 0 there. Propagate inward.
        let mut r = 0.0;
        for k in (0..shells.len()).rev() {
            let eps_above = shells[k].1;
            let eps_below = if k == 0 { eps_inner } else { shells[k - 1].1 };
            // Interface map (P and ε∂φ continuity), solved for the below-region ratio.
            let kk = eps_above * (lf * r - (lf + 1.0)) / (eps_below * (r + 1.0));
            r = ((lf + 1.0) + kk) / (lf - kk);
            // Propagate down through the below-region to the next interface (≤1 factor).
            if k > 0 {
                r *= (shells[k - 1].0 / shells[k].0).powf(2.0 * lf + 1.0);
            }
        }
        // r is now the reaction ratio of the innermost region at I₁.
        phi += (r / eps_inner) * (s / i1).powi(2 * l as i32) / i1;
    }
    POTPREFACTOR * ENERGY_FACTOR * charge * charge * phi
}

/// Exact triangulated sphere for the convergence theorem — vertices lie **on** the
/// analytic sphere, decoupling BEM convergence from SES geometry (plan §3, Q1).
///
/// Reuses proteon-core's `icosphere` (already produces vertices on the sphere), so
/// proteon and the NESSie oracle consume the **same** mesh → tight parity. Emit it
/// as OFF/JSON for the Julia harness to read back via `readoff`.
#[must_use]
pub fn analytic_sphere_mesh(radius: f64, subdivisions: u32) -> Mesh {
    debug_assert!(
        radius > 0.0,
        "sphere radius must be > 0 (else degenerate/inside-out)"
    );
    icosphere(Vec3::new(0.0, 0.0, 0.0), radius, subdivisions)
}
