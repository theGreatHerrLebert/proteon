//! Gauss–Legendre and triangle cubature for the treecode **panel integrals**.
//!
//! The treecode moments are `Q_K = Σ_j w_j ∫_{T_j} L_K(y) dS_y` — each panel's
//! contribution is the cluster basis `L_K` (a polynomial) integrated over the
//! triangle. `L_K` restricted to a planar triangle is a 2-D polynomial of total
//! degree `≤ 3p`, so the panel rule must be exact to that degree. A collapsed
//! (Duffy) tensor Gauss–Legendre rule of order `n` integrates polynomials of total
//! degree `2n − 2` exactly over the triangle (one degree is spent on the collapse
//! Jacobian), so `n = ⌈(3p + 2)/2⌉` suffices.
//!
//! Gauss–Legendre nodes are generated at runtime (Newton on the Legendre polynomial)
//! so any order is available, unlike the tabulated `{8,16,32}` rules in `adaptive`.

use proteon_core::surface::geom::Vec3;

use crate::model::Tri;

/// Gauss–Legendre nodes/weights on `[-1, 1]` for an `n`-point rule (exact to degree
/// `2n − 1`). Computed by Newton's method on `P_n` with the standard weight formula.
#[must_use]
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    assert!(n >= 1, "Gauss–Legendre needs n ≥ 1");
    if n == 1 {
        return vec![(0.0, 2.0)];
    }
    let mut out = Vec::with_capacity(n);
    let nf = n as f64;
    // Roots are symmetric; compute the lower half and mirror.
    for i in 0..n {
        // Initial guess (Chebyshev-like asymptotic for the i-th root).
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            // Evaluate P_n and P_n' by the recurrence.
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            // P_n = p1; derivative via P_n' = n (x P_n − P_{n-1}) / (x² − 1).
            dp = nf * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        out.push((x, w));
    }
    out
}

/// A cubature point on a triangle: an absolute position and its weight (weights sum to
/// the triangle area).
#[derive(Clone, Copy)]
pub struct CubPoint {
    /// Absolute position in 3-D.
    pub pos: Vec3,
    /// Quadrature weight (`Σ = area`).
    pub w: f64,
}

/// Gauss–Legendre order to integrate a polynomial of total degree `deg` over a panel
/// exactly: the collapsed rule loses one degree to the Jacobian, so `2n − 2 ≥ deg`,
/// i.e. `n = ⌈(deg + 2)/2⌉`.
#[must_use]
pub fn panel_order_for_total_degree(deg: usize) -> usize {
    (deg + 2).div_ceil(2)
}

/// Order for the **tensor BLTC** cluster basis (degree `p` per axis ⇒ total degree
/// `≤ 3p` on the planar panel).
#[must_use]
pub fn panel_order_for_degree(p: usize) -> usize {
    panel_order_for_total_degree(3 * p)
}

/// Order for the **Cartesian** monomial moments `u^k`, `|k| ≤ p` (total degree `p` —
/// a third of BLTC's, so a much cheaper panel rule).
#[must_use]
pub fn panel_order_for_cartesian(p: usize) -> usize {
    panel_order_for_total_degree(p)
}

/// Triangle cubature by the collapsed (Duffy) tensor Gauss–Legendre map of order `n`.
///
/// `λ1 = a`, `λ2 = b(1 − a)`, `λ3 = 1 − λ1 − λ2` with `(a, b)` GL nodes on `[0,1]²`;
/// the Jacobian `(1 − a)` and `2·area` fold into the weight. Weights sum to the
/// triangle area; exact for 2-D polynomials of total degree `≤ 2n − 2`.
#[must_use]
pub fn triangle_cubature(tri: &Tri, n: usize) -> Vec<CubPoint> {
    let gl = gauss_legendre(n);
    // Map [-1,1] → [0,1].
    let gl01: Vec<(f64, f64)> = gl
        .iter()
        .map(|&(x, w)| ((x + 1.0) * 0.5, w * 0.5))
        .collect();
    let mut pts = Vec::with_capacity(gl01.len() * gl01.len());
    let two_area = 2.0 * tri.area;
    for &(a, wa) in &gl01 {
        for &(b, wb) in &gl01 {
            let l1 = a;
            let l2 = b * (1.0 - a);
            let l3 = 1.0 - l1 - l2;
            let pos = tri.v1 * l1 + tri.v2 * l2 + tri.v3 * l3;
            let w = wa * wb * (1.0 - a) * two_area;
            pts.push(CubPoint { pos, w });
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gl_integral(n: usize, f: impl Fn(f64) -> f64) -> f64 {
        gauss_legendre(n).iter().map(|&(x, w)| w * f(x)).sum()
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // n-point GL is exact to degree 2n-1. ∫_{-1}^1 x^k dx = 0 (odd) or 2/(k+1) (even).
        let n = 6; // exact to degree 11
        for k in 0..=11u32 {
            let got = gl_integral(n, |x| x.powi(k as i32));
            let want = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert!((got - want).abs() < 1e-12, "∫x^{k}: {got} vs {want}");
        }
    }

    #[test]
    fn gauss_legendre_weights_sum_to_two() {
        for n in [1, 2, 5, 9, 16, 23] {
            let s: f64 = gauss_legendre(n).iter().map(|&(_, w)| w).sum();
            assert!((s - 2.0).abs() < 1e-12, "n={n} weights sum {s}");
        }
    }

    #[test]
    fn triangle_weights_sum_to_area() {
        let tri = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        );
        let pts = triangle_cubature(&tri, 5);
        let s: f64 = pts.iter().map(|p| p.w).sum();
        assert!(
            (s - tri.area).abs() < 1e-12,
            "Σw = {s}, area = {}",
            tri.area
        );
        assert!((tri.area - 3.0).abs() < 1e-12, "area sanity");
    }

    #[test]
    fn triangle_cubature_integrates_planar_polynomials() {
        // ∫_T x·y dA over the reference triangle (0,0),(1,0),(0,1) = 1/24.
        let tri = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let n = panel_order_for_degree(2); // ≥ degree 6, plenty for x·y
        let pts = triangle_cubature(&tri, n);
        let got: f64 = pts.iter().map(|p| p.w * p.pos.x * p.pos.y).sum();
        assert!(
            (got - 1.0 / 24.0).abs() < 1e-12,
            "∫xy dA = {got}, want {}",
            1.0 / 24.0
        );

        // ∫_T x^3 dA = 1/20 on the reference triangle.
        let got3: f64 = pts.iter().map(|p| p.w * p.pos.x.powi(3)).sum();
        assert!(
            (got3 - 1.0 / 20.0).abs() < 1e-12,
            "∫x^3 dA = {got3}, want {}",
            1.0 / 20.0
        );
    }
}
