//! P2 independent gate: analytic Laplace collocation vs direct numerical quadrature.
//!
//! NESSie fixture parity (`laplace_parity.rs`) cannot catch a transcription error
//! shared by both implementations of the *same* closed form. This gate integrates the
//! defining surface integrals numerically — independent of NESSie and of the
//! Rjasanow algebra — at **non-singular** observation points:
//!
//! * single layer: `∫_T 1/|ξ−r'| dA`
//! * double layer: `∫_T (ξ−r')·n / |ξ−r'|³ dA`
//!
//! (Both premultiplied by 4π, the value [`laplace_collocation`] returns.) The
//! near-singular / self quadrature (where centroid subdivision converges slowly) is a
//! deliberate follow-up — see the module note in `laplace.rs`.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::laplace::{laplace_collocation, Tri};
use proteon_electrostatics::model::PotentialKind;

/// Uniform barycentric subdivision quadrature of `f` over triangle `(a,b,c)`:
/// `m²` congruent sub-triangles, centroid sampling, area weight `A/m²`. 2nd-order,
/// so it converges fast where `f` is smooth (ξ well off the element).
fn integrate_tri(a: Vec3, b: Vec3, c: Vec3, m: usize, f: &dyn Fn(Vec3) -> f64) -> f64 {
    let area = (b - a).cross(c - a).norm() / 2.0;
    let w = area / (m * m) as f64;
    let inv = 1.0 / m as f64;
    // Grid point p(i,j) = a + (i/m)(b−a) + (j/m)(c−a), i+j ≤ m.
    let p = |i: usize, j: usize| a + (b - a) * (i as f64 * inv) + (c - a) * (j as f64 * inv);

    let mut acc = 0.0;
    for i in 0..m {
        for j in 0..(m - i) {
            // Upward sub-triangle.
            let up = (p(i, j) + p(i + 1, j) + p(i, j + 1)) * (1.0 / 3.0);
            acc += f(up);
            // Downward sub-triangle (exists when i+j ≤ m−2).
            if i + j + 2 <= m {
                let dn = (p(i + 1, j) + p(i + 1, j + 1) + p(i, j + 1)) * (1.0 / 3.0);
                acc += f(dn);
            }
        }
    }
    acc * w
}

fn equilateral() -> Tri {
    // Side 2, z = 0 plane, centroid at origin, +z normal.
    let a = Vec3::new(-1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
    let b = Vec3::new(1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
    let c = Vec3::new(0.0, 2.0 / 3.0_f64.sqrt(), 0.0);
    Tri::new(a, b, c)
}

/// Observation points well off the element (smooth integrand → fast convergence).
fn far_points() -> Vec<Vec3> {
    vec![
        Vec3::new(0.2, -0.1, 2.0),
        Vec3::new(0.2, -0.1, -2.0), // opposite side ⟹ double layer flips sign
        Vec3::new(1.5, 1.0, 1.3),
        Vec3::new(-0.8, 0.4, 3.0),
    ]
}

#[test]
fn single_layer_matches_quadrature() {
    let t = equilateral();
    const M: usize = 256;
    for xi in far_points() {
        let analytic = laplace_collocation(PotentialKind::Single, xi, &t);
        let numeric = integrate_tri(t.v1, t.v2, t.v3, M, &|r| 1.0 / (xi - r).norm());
        let rel = (analytic - numeric).abs() / numeric.abs();
        assert!(
            rel < 1e-4,
            "single layer ξ={xi:?}: analytic={analytic} numeric={numeric} rel={rel:.2e}"
        );
    }
}

#[test]
fn double_layer_matches_quadrature() {
    let t = equilateral();
    let n = t.normal;
    const M: usize = 256;
    for xi in far_points() {
        let analytic = laplace_collocation(PotentialKind::Double, xi, &t);
        let numeric = integrate_tri(t.v1, t.v2, t.v3, M, &|r| {
            let d = xi - r;
            d.dot(n) / d.norm().powi(3)
        });
        let rel = (analytic - numeric).abs() / numeric.abs().max(1e-12);
        assert!(
            rel < 1e-4,
            "double layer ξ={xi:?}: analytic={analytic} numeric={numeric} rel={rel:.2e}"
        );
    }
}
