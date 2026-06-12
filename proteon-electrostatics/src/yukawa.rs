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

/// Stable scalar `(e^{−κr} − 1)/r` (the single-layer regular-Yukawa kernel value), with
/// the alternating-series branch for small `κr` (`< 0.1`) to guard the `e^{−c} − 1`
/// cancellation. Assumes `r > 0`. Shared by the dense collocation and the P8 treecode
/// far-field eval so both use the *same* numerically-stable kernel.
#[must_use]
pub(crate) fn regyuk_single_scalar(rnorm: f64, yukawa: f64) -> f64 {
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

/// Stable scalar coefficient `1 − (1 + κr)e^{−κr}` (the double-layer regular-Yukawa
/// magnitude, geometry factored out), with the series branch for small `κr`. Its true
/// value is `O((κr)²)`, so the closed form catastrophically cancels for small `κr` — the
/// series recovers it. Assumes `r > 0`.
#[must_use]
pub(crate) fn regyuk_double_coef(rnorm: f64, yukawa: f64) -> f64 {
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
        return tsum;
    }
    1.0 - (1.0 + scalednorm) * (-scalednorm).exp()
}

/// Regular part of the single-layer Yukawa potential at quadrature point `x` for
/// observation point `ξ` (premultiplied by 4π). NESSie `_regularyukawapot(SingleLayer…)`.
#[inline]
fn regular_yukawa_pot_single(x: Vec3, xi: Vec3, yukawa: f64) -> f64 {
    let rnorm = (x - xi).norm();
    if rnorm <= ETOL_F64 {
        return -yukawa; // limit r → 0
    }
    regyuk_single_scalar(rnorm, yukawa)
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
    regyuk_double_coef(rnorm, yukawa) * cosovernorm2
}

/// Regular part of the single/double-layer Yukawa potential at `x` for `ξ` (×4π). Used
/// by the fixed Radon collocation and by the adaptive/Duffy rules ([`crate::adaptive`]).
#[inline]
pub(crate) fn regular_yukawa_pot(
    kind: PotentialKind,
    x: Vec3,
    xi: Vec3,
    yukawa: f64,
    normal: Vec3,
) -> f64 {
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
    regular_yukawa_collocation_parts(kind, xi, tri, yukawa).0
}

/// Like [`regular_yukawa_collocation`], but also returns the **non-cancelling local
/// magnitude** `Σ_i |w_i · pot_i| · 2·area` alongside the signed value. The adaptive
/// near-singular estimator ([`crate::adaptive`]) scales its tolerance to this magnitude
/// so a double-layer collocation passing through zero does not destabilise a purely
/// relative test (review [R5]).
#[must_use]
pub fn regular_yukawa_collocation_parts(
    kind: PotentialKind,
    xi: Vec3,
    tri: &Tri,
    yukawa: f64,
) -> (f64, f64) {
    let q = radon7(tri);
    let mut value = 0.0;
    let mut mag = 0.0;
    for i in 0..q.points.len() {
        let term = regular_yukawa_pot(kind, q.points[i], xi, yukawa, q.normal) * q.weights[i];
        value += term;
        mag += term.abs();
    }
    let scale = 2.0 * q.area;
    (value * scale, mag * scale)
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

    // Closed-form regular kernels (no series guard) — the reference for the in-guard
    // series test, valid where the closed form still keeps enough f64 precision
    // (c ≳ 0.03 loses only ~3 of ~16 digits to cancellation).
    fn closed_single(x: Vec3, xi: Vec3, yukawa: f64) -> f64 {
        let r = (x - xi).norm();
        ((-yukawa * r).exp() - 1.0) / r
    }
    fn closed_double(x: Vec3, xi: Vec3, yukawa: f64, n: Vec3) -> f64 {
        let r = (x - xi).norm();
        let c = yukawa * r;
        (1.0 - (1.0 + c) * (-c).exp()) * (x - xi).dot(n) / (r * r * r)
    }

    #[test]
    fn series_matches_closed_form_in_guard_region() {
        // In the cancellation-guard region (scalednorm < 0.1) the kernel takes the
        // alternating series. Validate it against the directly-evaluated closed form
        // at c values where the closed form is still accurate. The observation point
        // is OFF the element plane (nonzero z), so the double-layer (x−ξ)·n term is
        // genuinely nonzero — an in-plane point would make it 0 and test nothing.
        let xi = Vec3::new(0.0, 0.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let yukawa = 1.3;
        for &c in &[0.03_f64, 0.05, 0.09] {
            let r = c / yukawa;
            // |x| = r with a nonzero normal projection (0.8·r).
            let x = Vec3::new(0.6 * r, 0.0, 0.8 * r);
            assert!(
                yukawa * x.norm() < SERIES_THRESHOLD,
                "must be in the series branch"
            );

            // The series truncates at NESSie's `etol·|term|` tolerance, so its
            // intrinsic accuracy vs the (here-still-accurate) closed form is ~etol
            // ≈ 1.5e-8 relative, not full f64 — a 1e-7 band confirms the series has
            // the right coefficients/recurrence without tripping on its own tail.
            let s_series = regular_yukawa_pot(PotentialKind::Single, x, xi, yukawa, n);
            let s_closed = closed_single(x, xi, yukawa);
            assert!(
                (s_series - s_closed).abs() / s_closed.abs() < 1e-7,
                "single c={c}: series={s_series} closed={s_closed}"
            );

            let d_series = regular_yukawa_pot(PotentialKind::Double, x, xi, yukawa, n);
            let d_closed = closed_double(x, xi, yukawa, n);
            assert!(
                d_closed.abs() > 1e-6,
                "double-layer reference must be nonzero"
            );
            assert!(
                (d_series - d_closed).abs() / d_closed.abs() < 1e-7,
                "double c={c}: series={d_series} closed={d_closed}"
            );
        }
    }

    #[test]
    fn near_zero_series_is_well_behaved() {
        // Just ABOVE ETOL (deep in the series, NOT the exact r=0 branch). The series
        // must stay finite where the closed form would catastrophically cancel.
        //
        // Single → −yukawa as r→0. Double: approaching along the normal, the *pointwise*
        // value is yukawa²/2 (from (1−(1+c)e^(−c)) ≈ c²/2 times (x·n)/r³ = 1/r²). Note
        // this is NOT NESSie's exact-coincidence branch value yukawa²/(2√3) — that
        // constant is a self-element regularization for when a quadrature point lands
        // exactly on ξ (tested separately in double_layer_zero_distance_limit).
        let xi = Vec3::new(0.0, 0.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let yukawa = 1.3;
        let r = 1e-7; // > ETOL_F64 = 1.45e-8

        let s = regular_yukawa_pot_single(Vec3::new(r, 0.0, 0.0), xi, yukawa);
        assert!(
            s.is_finite() && (s + yukawa).abs() < 1e-5,
            "single near 0: {s}"
        );

        let d = regular_yukawa_pot_double(Vec3::new(0.0, 0.0, r), xi, yukawa, n);
        let pointwise = yukawa * yukawa / 2.0;
        assert!(
            d.is_finite() && (d - pointwise).abs() / pointwise < 1e-3,
            "double near 0 (pointwise along normal): {d} vs {pointwise}"
        );
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
