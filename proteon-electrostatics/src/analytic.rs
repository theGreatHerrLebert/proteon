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
