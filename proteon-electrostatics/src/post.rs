//! Post-processing: reaction-field energy and electrostatic potentials (L4).
//!
//! Port of NESSie's `bem/post.jl` (local). Consumes solved Cauchy data + the geometry
//! and charges to produce the scientific outputs: the reaction-field energy `W*`
//! (kJ/mol) and the electrostatic potential at observation points in each domain.
//!
//! **Energy alone is not enough** (plan §4 L4): a scalar `rfenergy` hides
//! compensating potential errors, so the potential is gated too. On Γ the potential
//! uses the jump-corrected limiting trace (`u` at the closest element), not a raw
//! on-surface kernel value.
//!
//! # API note (P5, codex review)
//! These take `elements`/`charges`/`params` + `cauchy` independently, so a caller can
//! pair Cauchy data with the wrong geometry. A future `SolvedBem` (P7) should bind
//! them; for now [`crate::solve::solve_local`] returns a `LocalResult` that the caller
//! must feed back with the same elements it solved on.
//!
//! # Gates (P5)
//! - vs NESSie `post_dump` (`rfenergy` + `espotential` over Ω/Σ/Γ) at a fixed mesh.
//! - vs the closed-form Born energy on analytic sphere meshes — the science gate,
//!   independent of any BEM path ([`crate::analytic`]).

use crate::laplace::laplace_collocation;
use crate::model::{Charge, Domain, Params, PotentialKind, Tri};
use crate::solve::CauchyData;
use crate::yukawa::regular_yukawa_collocation;
use proteon_core::surface::geom::Vec3;

/// `10¹⁰·e` (Å→m folded in), NESSie `ec` (C).
const EC: f64 = 1.602_176e-9;
/// Vacuum permittivity `1/(4π·1e-7·c²)` (F/m), NESSie `ε0`.
const EPS0: f64 = 1.0 / (4.0 * std::f64::consts::PI * 1e-7 * 299_792_458.0 * 299_792_458.0);
/// `4π`.
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// Common potential prefactor `ec/(4π·ε0)` (NESSie `potprefactor`). Converts the
/// `4π·ε0`-premultiplied Cauchy data to volts.
pub const POTPREFACTOR: f64 = EC / FOUR_PI / EPS0;

/// Energy conversion `ec·Nₐ·1e-3 / 2` = `ec·6.022140857e10 / 2` — Joule→kJ/mol with
/// the ½ for double-counted interactions (NESSie `rfenergy`).
pub const ENERGY_FACTOR: f64 = EC * 6.022_140_857e10 / 2.0;

/// `Σ_j collocation(kind, ξ_c, elem_j)·fvals_j` for each observation point `ξ_c`
/// (NESSie `laplacecoll!` vector form). Result premultiplied by 4π.
fn collocate_contract(
    kind: PotentialKind,
    obs: &[Vec3],
    elements: &[Tri],
    fvals: &[f64],
) -> Vec<f64> {
    obs.iter()
        .map(|&xi| {
            elements
                .iter()
                .zip(fvals)
                .map(|(e, &f)| laplace_collocation(kind, xi, e) * f)
                .sum()
        })
        .collect()
}

/// `Σ_j regular_yukawa(kind, ξ_c, elem_j, yuk)·fvals_j` per observation point.
fn yukawa_contract(
    kind: PotentialKind,
    obs: &[Vec3],
    elements: &[Tri],
    fvals: &[f64],
    yuk: f64,
) -> Vec<f64> {
    obs.iter()
        .map(|&xi| {
            elements
                .iter()
                .zip(fvals)
                .map(|(e, &f)| regular_yukawa_collocation(kind, xi, e, yuk) * f)
                .sum()
        })
        .collect()
}

/// Molecular potential at ξ (volts), NESSie `molpotential(ξ, model)`:
/// `(Σ_c q_c / max(|ξ−r_c|, tol)) / εΩ · potprefactor`.
fn molpotential(xi: Vec3, charges: &[Charge], eps_omega: f64) -> f64 {
    const TOL: f64 = 1e-10;
    let raw: f64 = charges
        .iter()
        .map(|c| c.val / (xi - c.pos).norm().max(TOL))
        .sum();
    raw / eps_omega * POTPREFACTOR
}

/// Index of the element whose centroid is closest to ξ (NESSie `_closest_element_id`).
fn closest_element(xi: Vec3, elements: &[Tri]) -> usize {
    let mut best = (0usize, f64::INFINITY);
    for (i, e) in elements.iter().enumerate() {
        let c = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
        let d = (xi - c).norm();
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// Reaction-field energy `W*` (kJ/mol). NESSie `rfenergy(bem)`:
/// `W*_c = −[K(q_pos)·u]_c + [V(q_pos)·q]_c`, `W* = (W*·qval)/4π · potprefactor · energy_factor`.
#[must_use]
pub fn rfenergy(elements: &[Tri], charges: &[Charge], cauchy: &dyn CauchyData) -> f64 {
    let qpos: Vec<Vec3> = charges.iter().map(|c| c.pos).collect();
    // wstar = −K·u + V·q, both collocated at the charge positions.
    let kd = collocate_contract(PotentialKind::Double, &qpos, elements, cauchy.u());
    let vq = collocate_contract(PotentialKind::Single, &qpos, elements, cauchy.q());
    let dot: f64 = charges
        .iter()
        .enumerate()
        .map(|(c, ch)| (-kd[c] + vq[c]) * ch.val)
        .sum();
    dot / FOUR_PI * POTPREFACTOR * ENERGY_FACTOR
}

/// Electrostatic potential at ξ in `domain` (volts). NESSie `espotential(:Ω/:Σ/:Γ, ξ, bem)`
/// for a local result.
#[must_use]
pub fn espotential(
    domain: Domain,
    xi: Vec3,
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cauchy: &dyn CauchyData,
) -> f64 {
    let obs = [xi];
    match domain {
        // Γ: rf = u[closest]·potprefactor; + molecular potential (jump-corrected trace).
        Domain::Gamma => {
            let i = closest_element(xi, elements);
            cauchy.u()[i] * POTPREFACTOR + molpotential(xi, charges, params.eps_omega)
        }
        // Ω: rf = (−K·u + V·q)/4π · potprefactor; + molecular potential.
        Domain::Omega => {
            let ku = collocate_contract(PotentialKind::Double, &obs, elements, cauchy.u())[0];
            let vq = collocate_contract(PotentialKind::Single, &obs, elements, cauchy.q())[0];
            (-ku + vq) / FOUR_PI * POTPREFACTOR + molpotential(xi, charges, params.eps_omega)
        }
        // Σ (solvent): local and nonlocal differ. Local result (no `w`):
        //   (−εΩ/εΣ·V·(q+qmol) + K·(u+umol)) · potprefactor/4π.
        // Nonlocal result (with `w`): adds the regular-Yukawa terms (NESSie
        //   `_espotential_Σ(::NonlocalBEMResult)`).
        Domain::Sigma => {
            let n = elements.len();
            let qq: Vec<f64> = (0..n).map(|i| cauchy.q()[i] + cauchy.qmol()[i]).collect();
            let uu: Vec<f64> = (0..n).map(|i| cauchy.u()[i] + cauchy.umol()[i]).collect();
            let vqq = collocate_contract(PotentialKind::Single, &obs, elements, &qq)[0];
            let kuu = collocate_contract(PotentialKind::Double, &obs, elements, &uu)[0];
            let (eo, es, ei) = (params.eps_omega, params.eps_sigma, params.eps_inf);
            match cauchy.w() {
                None => (-eo / es * vqq + kuu) * POTPREFACTOR / FOUR_PI,
                Some(w) => {
                    let yuk = params.yukawa();
                    // Ky argument: u + (1−εΩ/εΣ)·umol − ε∞/εΣ·w.
                    let ky_arg: Vec<f64> = (0..n)
                        .map(|i| {
                            cauchy.u()[i] + (1.0 - eo / es) * cauchy.umol()[i] - (ei / es) * w[i]
                        })
                        .collect();
                    let vy_qq = yukawa_contract(PotentialKind::Single, &obs, elements, &qq, yuk)[0];
                    let ky =
                        yukawa_contract(PotentialKind::Double, &obs, elements, &ky_arg, yuk)[0];
                    // −εΩ/ε∞·V·(q+qmol) + εΩ(1/εΣ−1/ε∞)·Vy·(q+qmol) + K·(u+umol) + Ky·(…).
                    (-eo / ei * vqq + eo * (1.0 / es - 1.0 / ei) * vy_qq + kuu + ky) * POTPREFACTOR
                        / FOUR_PI
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefactors_match_independent_si_derivation() {
        // The crate folds Å→m / J→kJ into ec and the Avogadro literal; re-derive the
        // SAME prefactors from base SI constants stated independently so a wrong
        // exponent or a missing ½ cannot hide (every oracle gate shares the crate's
        // own constants, so this is the only place that pins them).
        let pi = std::f64::consts::PI;
        let e = 1.602_176e-19; // elementary charge, C
        let na = 6.022_140_857e23; // Avogadro, 1/mol
        let c = 299_792_458.0; // speed of light, m/s
        let eps0 = 1.0 / (4.0 * pi * 1e-7 * c * c); // F/m

        // energy_factor = e·Nₐ / 2 / 1000  (J→kJ, ½ for double-counted interactions).
        let ef = e * na / 2.0 / 1000.0;
        assert!(
            (ENERGY_FACTOR - ef).abs() / ef < 1e-12,
            "energy_factor {ENERGY_FACTOR} vs SI-derived {ef}"
        );
        // potprefactor = e / (4π·ε0) · 1e10  (Coulomb prefactor in V·Å per e).
        let pp = e / (4.0 * pi * eps0) * 1e10;
        assert!(
            (POTPREFACTOR - pp).abs() / pp < 1e-12,
            "potprefactor {POTPREFACTOR} vs SI-derived {pp}"
        );
        // Sanity anchors to the documented magnitudes (potprefactor ≈ 1.145·4π;
        // energy_factor ≈ ½·Faraday in kJ/mol·V⁻¹).
        assert!((POTPREFACTOR - 14.3997).abs() < 1e-3, "{POTPREFACTOR}");
        assert!((ENERGY_FACTOR - 48.2426).abs() < 1e-3, "{ENERGY_FACTOR}");
    }
}
