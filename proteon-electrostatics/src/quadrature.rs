//! Radon 7-point triangle cubature (L0).
//!
//! Mirrors NESSie's `TriangleQuad` / `quadraturepoints` (`src/base/quadrature.jl`).
//! The degree-5 7-point Radon rule integrates the regular Yukawa kernel over each
//! triangle (`yukawa.rs`). The points/weights are **ported verbatim** from NESSie's
//! source (faithful-port discipline) — computed from `√15` the same way, so the
//! mapped cubature is bit-identical to NESSie's.
//!
//! Note the weights sum to **½** (the area of the reference triangle in barycentric
//! coordinates), not 1; the `× 2·area` factor in [`crate::yukawa`]'s collocation
//! converts to the actual triangle. NESSie does the same.
//!
//! L0 gate (no oracle): weights sum to ½, points lie inside the reference triangle,
//! and monomials up to total degree 5 integrate exactly.

use crate::model::Tri;
use proteon_core::surface::geom::Vec3;

/// Number of Radon cubature points.
pub const RADON7_N: usize = 7;

/// The 7-point Radon rule in barycentric `(x, y)` coordinates + weights, computed
/// exactly as NESSie's `triquadpts_Float64` (`base/quadrature.jl`). A point in space
/// is `v1 + x·(v2−v1) + y·(v3−v1)`.
fn radon7_rule() -> ([f64; RADON7_N], [f64; RADON7_N], [f64; RADON7_N]) {
    let s = 15.0_f64.sqrt();
    let x = [
        1.0 / 3.0,
        (6.0 + s) / 21.0,
        (9.0 - 2.0 * s) / 21.0,
        (6.0 + s) / 21.0,
        (6.0 - s) / 21.0,
        (9.0 + 2.0 * s) / 21.0,
        (6.0 - s) / 21.0,
    ];
    let y = [
        1.0 / 3.0,
        (9.0 - 2.0 * s) / 21.0,
        (6.0 + s) / 21.0,
        (6.0 + s) / 21.0,
        (9.0 + 2.0 * s) / 21.0,
        (6.0 - s) / 21.0,
        (6.0 - s) / 21.0,
    ];
    let w = [
        9.0 / 80.0,
        (155.0 + s) / 2400.0,
        (155.0 + s) / 2400.0,
        (155.0 + s) / 2400.0,
        (155.0 - s) / 2400.0,
        (155.0 - s) / 2400.0,
        (155.0 - s) / 2400.0,
    ];
    (x, y, w)
}

/// Precomputed cubature for one triangle: world-space points + weights, plus the
/// element's normal and area (NESSie's `TriangleQuad` caches these on the element).
#[derive(Debug, Clone)]
pub struct TriangleQuad {
    /// Cubature points in world space (7 for the Radon rule).
    pub points: [Vec3; RADON7_N],
    /// Cubature weights (sum to ½ — the reference-triangle area).
    pub weights: [f64; RADON7_N],
    /// Unit outward normal of the triangle (for the double-layer kernel).
    pub normal: Vec3,
    /// Triangle area (the `× 2·area` collocation factor).
    pub area: f64,
}

/// Build the 7-point Radon cubature for triangle `tri`, mapping the barycentric rule
/// into world space exactly as NESSie's `TriangleQuad(elem)`:
/// `point_j = v1 + x_j·(v2−v1) + y_j·(v3−v1)`. Carries the element's `normal`/`area`.
#[must_use]
pub fn radon7(tri: &Tri) -> TriangleQuad {
    let (x, y, w) = radon7_rule();
    let e1 = tri.v2 - tri.v1;
    let e2 = tri.v3 - tri.v1;
    let mut points = [tri.v1; RADON7_N];
    for j in 0..RADON7_N {
        // NESSie order `x·e1 + y·e2 + v1` (left-assoc) — matched exactly so the
        // mapped points are bit-identical, not just ULP-close.
        points[j] = e1 * x[j] + e2 * y[j] + tri.v1;
    }
    TriangleQuad {
        points,
        weights: w,
        normal: tri.normal,
        area: tri.area,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_half() {
        // Reference-triangle area in barycentric coordinates.
        let (_, _, w) = radon7_rule();
        let sum: f64 = w.iter().sum();
        assert!((sum - 0.5).abs() < 1e-15, "weights sum = {sum}");
    }

    #[test]
    fn points_inside_reference_triangle() {
        let (x, y, _) = radon7_rule();
        for j in 0..RADON7_N {
            assert!(x[j] >= 0.0 && y[j] >= 0.0 && x[j] + y[j] <= 1.0 + 1e-15);
        }
    }

    #[test]
    fn integrates_monomials_to_degree_5() {
        // Exact reference-triangle integral of x^a y^b is a! b! / (a+b+2)!.
        let (x, y, w) = radon7_rule();
        let fact = |n: u32| (1..=u64::from(n)).product::<u64>().max(1) as f64;
        let exact = |a: u32, b: u32| fact(a) * fact(b) / fact(a + b + 2);
        for a in 0..=5u32 {
            for b in 0..=(5 - a) {
                let quad: f64 = (0..RADON7_N)
                    .map(|j| w[j] * x[j].powi(a as i32) * y[j].powi(b as i32))
                    .sum();
                let want = exact(a, b);
                assert!(
                    (quad - want).abs() < 1e-13,
                    "monomial x^{a} y^{b}: quad={quad} exact={want}"
                );
            }
        }
    }

    #[test]
    fn maps_to_world_and_caches_props() {
        // Centroid (barycentric 1/3,1/3) is the first Radon point; it maps to the
        // triangle centroid, and normal/area are carried from the Tri.
        let t = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let q = radon7(&t);
        let centroid = (t.v1 + t.v2 + t.v3) * (1.0 / 3.0);
        assert!((q.points[0] - centroid).norm() < 1e-14);
        assert!((q.area - 2.0).abs() < 1e-14); // ½·|2×2|
        assert!((q.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-14);
    }
}
