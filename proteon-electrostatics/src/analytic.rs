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
use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::{icosphere, Mesh};

/// Closed-form Born solvation energy (kJ/mol) of a single ion. NESSie: `BornIon` +
/// `rfenergy(LocalES|NonlocalES, ion)`.
///
/// NESSie's Born formula assumes a **vacuum solute** (`εΩ = 1`): the local energy
/// carries `(1/εΣ − 1)`, not `(1/εΣ − 1/εΩ)`. Reject (or special-case) `eps_omega
/// != 1` rather than silently evaluating an unvalidated generalization (plan §6).
///
/// TODO(P5): port the Born closed form (local); P6 for nonlocal.
#[must_use]
pub fn born_rfenergy(charge: f64, radius: f64, params: &Params, locality: Locality) -> f64 {
    unimplemented!("P5/P6: port Born closed-form energy (src/testmodel/born/)")
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
