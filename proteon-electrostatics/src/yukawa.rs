//! Radon regular-Yukawa collocation (L2) — Yukawa minus Laplace.
//!
//! Port of NESSie's `Radon` module (`src/Radon.jl`): the **regular** part of the
//! Yukawa potential (Yukawa − Laplace), integrated over each triangle with the
//! 7-point Radon cubature (`quadrature.rs`). The singular Laplace part is handled
//! analytically in `laplace.rs`; subtracting it leaves a smooth integrand.
//!
//! The cancellation guard is the fragile bit: for small `yukawa·|x−ξ|` NESSie uses
//! an alternating-series expansion (threshold `0.1`) instead of the closed form.
//! Limits: SL → `−yukawa`, DL → `yukawa²/(2√3)` as `|x−ξ| → 0`.
//!
//! # Gates (P3)
//! - per-element value vs NESSie `yukawa_dump` **and** numerical quadrature, swept
//!   across the series/closed-form branch boundary (≈0.1) and the `|x−ξ|→0` limit.
//! - near-singular: the fixed 7-point rule is **not** accurate for nearly-touching
//!   non-self elements — gate against high-precision quadrature and document the
//!   floor; adaptive subdivision is the mandatory P6.5 remediation.

use crate::model::PotentialKind;
use proteon_core::surface::geom::Vec3;

/// Series/closed-form branch threshold on `yukawa·|x−ξ|` (NESSie uses `0.1`).
pub const SERIES_THRESHOLD: f64 = 0.1;

/// Regular-Yukawa collocation of a triangle at ξ (result premultiplied by 4π).
///
/// `yukawa` is the exponent `√(εΣ/ε∞)/λ` ([`crate::model::Params::yukawa`]).
///
/// TODO(P3): port `Radon._regularyukawapot` (SL/DL, series guard) +
/// `regularyukawacoll` (7-point cubature accumulation).
#[must_use]
pub fn regular_yukawa_collocation(
    kind: PotentialKind,
    xi: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    yukawa: f64,
) -> f64 {
    unimplemented!("P3: port Radon regular-Yukawa collocation (src/Radon.jl)")
}
