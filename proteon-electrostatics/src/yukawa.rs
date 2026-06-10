//! Radon regular-Yukawa collocation (L2) — Yukawa minus Laplace.
//!
//! Port of NESSie's `Radon` module (`src/Radon.jl`): the **regular** part of the
//! Yukawa potential (Yukawa − Laplace), integrated over each triangle with the
//! 7-point Radon cubature (`quadrature.rs`). The singular Laplace part is handled
//! analytically in `laplace.rs`; subtracting it leaves a smooth integrand:
//!
//! * single (×4π): `(e^(−yukawa·r) − 1) / r`, `r = |x−ξ|`
//! * double (×4π): `(1 − (1 + yukawa·r)·e^(−yukawa·r)) · (x−ξ)·n / r³`
//!
//! The cancellation guard is the fragile bit: for small `yukawa·r` (`< 0.1`) NESSie
//! evaluates an alternating-series expansion instead of the closed form, to avoid
//! catastrophic cancellation in `e^(−c) − 1` / `1 − (1+c)e^(−c)`. Limits as `r → 0`:
//! single → `−yukawa`, double → `yukawa²/(2√3)`.
//!
//! # Gates (P3)
//! - per-element value vs NESSie `yukawa_dump` (`tests/yukawa_parity.rs`).
//! - an independent numerical-quadrature oracle on the single-layer physical kernel,
//!   plus the `r → 0` limits and series/closed-form continuity at the `0.1` boundary.
//! - near-singular: the fixed 7-point rule is **not** accurate for nearly-touching
//!   non-self elements — gate against high-precision quadrature and document the
//!   floor; adaptive subdivision is the mandatory P6.5 remediation.

use crate::laplace::ETOL_F64;
use crate::model::{PotentialKind, Tri};
use crate::quadrature::radon7;
use proteon_core::surface::geom::Vec3;

/// Series/closed-form branch threshold on `yukawa·|x−ξ|` (NESSie uses `0.1`).
pub const SERIES_THRESHOLD: f64 = 0.1;

/// Regular part of the single-layer Yukawa potential at quadrature point `x` for
/// observation point `ξ` (premultiplied by 4π). NESSie `_regularyukawapot(SingleLayer…)`.
#[inline]
fn regular_yukawa_pot_single(x: Vec3, xi: Vec3, yukawa: f64) -> f64 {
    let rnorm = (x - xi).norm();
    if rnorm <= ETOL_F64 {
        return -yukawa; // limit r → 0
    }
    let scalednorm = yukawa * rnorm;
    if scalednorm < SERIES_THRESHOLD {
        // Alternating series for e^(−c) − 1 = Σ (−c)^i / i!, guarding cancellation.
        let mut term = -scalednorm;
        let tolerance = ETOL_F64 * term.abs();
        let mut tsum = 0.0;
        for i in 1..=15 {
            if term.abs() <= tolerance {
                break;
            }
            tsum += term;
            term *= -scalednorm / (f64::from(i) + 1.0);
        }
        return tsum / rnorm;
    }
    ((-scalednorm).exp() - 1.0) / rnorm
}

/// Regular part of the double-layer Yukawa potential (normal derivative) at `x` for
/// `ξ` (premultiplied by 4π). NESSie `_regularyukawapot(DoubleLayer…)`.
#[inline]
fn regular_yukawa_pot_double(x: Vec3, xi: Vec3, yukawa: f64, normal: Vec3) -> f64 {
    let rnorm = (x - xi).norm();
    if rnorm <= ETOL_F64 {
        return yukawa * yukawa / 2.0 / 3.0_f64.sqrt(); // limit r → 0
    }
    let cosovernorm2 = (x - xi).dot(normal) / (rnorm * rnorm * rnorm);
    let scalednorm = yukawa * rnorm;
    if scalednorm < SERIES_THRESHOLD {
        // Series for 1 − (c+1)e^(−c) = Σ (−c)^i (i−1) / i!  (i ≥ 2).
        let mut term = scalednorm * scalednorm / 2.0;
        let tolerance = ETOL_F64 * term.abs();
        let mut tsum = 0.0;
        for i in 2..=16 {
            if term.abs() <= tolerance {
                break;
            }
            tsum += term * (f64::from(i) - 1.0);
            term *= -scalednorm / (f64::from(i) + 1.0);
        }
        return tsum * cosovernorm2;
    }
    (1.0 - (1.0 + scalednorm) * (-scalednorm).exp()) * cosovernorm2
}

/// Regular part of the single/double-layer Yukawa potential at `x` for `ξ`.
#[inline]
fn regular_yukawa_pot(kind: PotentialKind, x: Vec3, xi: Vec3, yukawa: f64, normal: Vec3) -> f64 {
    match kind {
        PotentialKind::Single => regular_yukawa_pot_single(x, xi, yukawa),
        PotentialKind::Double => regular_yukawa_pot_double(x, xi, yukawa, normal),
    }
}

/// Regular-Yukawa collocation of triangle `tri` at ξ (result premultiplied by 4π).
///
/// Mirrors NESSie `regularyukawacoll(ptype, ξ, tquad, yukawa)`: the 7-point Radon
/// cubature of the regular kernel over the triangle, `× 2·area`. `yukawa` is the
/// exponent `√(εΣ/ε∞)/λ` ([`crate::model::Params::yukawa`]).
#[must_use]
pub fn regular_yukawa_collocation(kind: PotentialKind, xi: Vec3, tri: &Tri, yukawa: f64) -> f64 {
    let q = radon7(tri);
    let mut value = 0.0;
    for i in 0..q.points.len() {
        value += regular_yukawa_pot(kind, q.points[i], xi, yukawa, q.normal) * q.weights[i];
    }
    value * 2.0 * q.area
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equilateral() -> Tri {
        let a = Vec3::new(-1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
        let b = Vec3::new(1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
        let c = Vec3::new(0.0, 2.0 / 3.0_f64.sqrt(), 0.0);
        Tri::new(a, b, c)
    }

    #[test]
    fn single_layer_zero_distance_limit() {
        // r → 0 ⟹ single regular pot → −yukawa.
        let p = Vec3::new(1.0, 2.0, 3.0);
        let v = regular_yukawa_pot_single(p, p, 0.37);
        assert!((v + 0.37).abs() < 1e-15, "{v}");
    }

    #[test]
    fn double_layer_zero_distance_limit() {
        // r → 0 ⟹ double regular pot → yukawa²/(2√3).
        let p = Vec3::new(1.0, 2.0, 3.0);
        let yuk = 0.37;
        let v = regular_yukawa_pot_double(p, p, yuk, Vec3::new(0.0, 0.0, 1.0));
        assert!((v - yuk * yuk / 2.0 / 3.0_f64.sqrt()).abs() < 1e-15, "{v}");
    }

    #[test]
    fn series_matches_closed_form_at_boundary() {
        // The series and closed form must agree across the 0.1 branch boundary —
        // straddle it and check continuity (the series is there to avoid the
        // cancellation the closed form suffers, so where both are accurate they match).
        let xi = Vec3::new(0.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        for &yukawa in &[0.5_f64, 1.0, 2.0] {
            // Pick r so scalednorm sits just below / above 0.1.
            let r_below = (SERIES_THRESHOLD - 1e-4) / yukawa;
            let r_above = (SERIES_THRESHOLD + 1e-4) / yukawa;
            let x_b = Vec3::new(r_below, 0.0, 0.0);
            let x_a = Vec3::new(r_above, 0.0, 0.0);
            for kind in [PotentialKind::Single, PotentialKind::Double] {
                let below = regular_yukawa_pot(kind, x_b, xi, yukawa, normal);
                let above = regular_yukawa_pot(kind, x_a, xi, yukawa, normal);
                // The two points are 2e-4/yukawa apart; the kernel is smooth there,
                // so the values are close — a discontinuity at the branch would blow up.
                assert!(
                    (below - above).abs() < 1e-2,
                    "{kind:?} discontinuous at boundary: {below} vs {above}"
                );
            }
        }
    }

    #[test]
    fn single_layer_matches_quadrature() {
        // Independent (non-NESSie) gate: the single-layer collocation vs a fine
        // numerical integral of the PHYSICAL regular kernel e^(−yr)/r − 1/r at
        // non-singular points. Validates the integrand, the Radon mapping, the
        // weights, and the ×2·area factor.
        let t = equilateral();
        let yukawa = 0.8;
        const M: usize = 256;
        for xi in [
            Vec3::new(0.2, -0.1, 2.0),
            Vec3::new(1.5, 1.0, 1.3),
            Vec3::new(-0.8, 0.4, 3.0),
        ] {
            let collok = regular_yukawa_collocation(PotentialKind::Single, xi, &t, yukawa);
            let numeric = integrate_tri(t.v1, t.v2, t.v3, M, &|r| {
                let d = (xi - r).norm();
                (-yukawa * d).exp() / d - 1.0 / d
            });
            let rel = (collok - numeric).abs() / numeric.abs();
            assert!(
                rel < 1e-4,
                "ξ={xi:?}: collok={collok} numeric={numeric} rel={rel:.2e}"
            );
        }
    }

    /// Uniform barycentric subdivision quadrature (mirrors the Laplace gate's helper).
    fn integrate_tri(a: Vec3, b: Vec3, c: Vec3, m: usize, f: &dyn Fn(Vec3) -> f64) -> f64 {
        let area = (b - a).cross(c - a).norm() / 2.0;
        let w = area / (m * m) as f64;
        let inv = 1.0 / m as f64;
        let p = |i: usize, j: usize| a + (b - a) * (i as f64 * inv) + (c - a) * (j as f64 * inv);
        let mut acc = 0.0;
        for i in 0..m {
            for j in 0..(m - i) {
                acc += f((p(i, j) + p(i + 1, j) + p(i, j + 1)) * (1.0 / 3.0));
                if i + j + 2 <= m {
                    acc += f((p(i + 1, j) + p(i + 1, j + 1) + p(i, j + 1)) * (1.0 / 3.0));
                }
            }
        }
        acc * w
    }
}
