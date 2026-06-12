//! Cartesian-multipole panel-aware cluster expansion — the bake-off alternative to
//! the barycentric-Lagrange treecode ([`super::expansion`]).
//!
//! The far field expands the kernel in a Taylor series about the cluster center
//! `y_c` in `u = y − y_c`:
//!
//! ```text
//! 1/|ξ − y| = 1/|R − u| = Σ_{|k|≤p} a_k(R) · u^k ,   R = ξ − y_c
//! single layer:  Σ_j x_j ∫_{T_j} 1/|ξ−y| dS  ≈  Σ_{|k|≤p} a_k(R) · M_k
//!                                        M_k = Σ_j x_j ∫_{T_j} (y − y_c)^k dS
//! ```
//!
//! with `a_k` the Coulomb Taylor coefficients (Lindsay–Krasny recurrence) and `M_k`
//! the **panel-aware** monomial moments (monomial integrals over each panel — exact
//! by cubature). A total-degree-`p` expansion has `C(p+3,3) = (p+1)(p+2)(p+3)/6`
//! terms vs the tensor BLTC's `(p+1)³` — the cost edge the bake-off measures. The
//! recurrence is Coulomb-specific (a different one is needed for Yukawa), unlike the
//! kernel-independent BLTC.

use proteon_core::surface::geom::Vec3;

use super::cubature::{panel_order_for_degree, triangle_cubature};
use crate::model::Tri;

/// Flat index of multi-index `(i, j, k)` in a dense `(p+1)³` cube.
#[inline]
fn cidx(i: usize, j: usize, k: usize, p: usize) -> usize {
    (i * (p + 1) + j) * (p + 1) + k
}

/// Coulomb Taylor coefficients `a_k(R) = (1/k!) ∂_u^k [1/|R − u|]|_{u=0}` for all
/// `|k| ≤ p`, by the Lindsay–Krasny recurrence
/// `|k| a_k = (1/r²)[(2|k|−1) Σ_i R_i a_{k−e_i} − (|k|−1) Σ_i a_{k−2e_i}]`,
/// `a_0 = 1/r`. Returned as a dense `(p+1)³` cube (entries with `i+j+k > p` are 0).
#[must_use]
pub fn coulomb_taylor_coeffs(r_vec: Vec3, p: usize) -> Vec<f64> {
    let n = p + 1;
    let mut a = vec![0.0; n * n * n];
    let r2 = r_vec.dot(r_vec);
    let r = r2.sqrt();
    let rc = [r_vec.x, r_vec.y, r_vec.z];
    a[cidx(0, 0, 0, p)] = 1.0 / r;
    // Fill in order of increasing total degree so dependencies are ready.
    for deg in 1..=p {
        for i in 0..=deg {
            for j in 0..=(deg - i) {
                let k = deg - i - j;
                let idx = [i, j, k];
                let mut s1 = 0.0; // Σ_i R_i a_{k−e_i}
                let mut s2 = 0.0; // Σ_i a_{k−2e_i}
                for d in 0..3 {
                    if idx[d] >= 1 {
                        let mut m = idx;
                        m[d] -= 1;
                        s1 += rc[d] * a[cidx(m[0], m[1], m[2], p)];
                    }
                    if idx[d] >= 2 {
                        let mut m = idx;
                        m[d] -= 2;
                        s2 += a[cidx(m[0], m[1], m[2], p)];
                    }
                }
                let degf = deg as f64;
                a[cidx(i, j, k, p)] =
                    ((2.0 * degf - 1.0) * s1 - (degf - 1.0) * s2) / (degf * r2);
            }
        }
    }
    a
}

/// Panel-aware single-layer Cartesian moments `M_k = Σ_j x_j ∫_{T_j} (y − y_c)^k dS`,
/// dense `(p+1)³` (entries with `i+j+k > p` unused / 0).
#[must_use]
pub fn single_layer_moments(center: Vec3, panels: &[(Tri, f64)], p: usize) -> Vec<f64> {
    let n = p + 1;
    let mut m = vec![0.0; n * n * n];
    let cub_order = panel_order_for_degree(p);
    for (tri, x) in panels {
        for cp in triangle_cubature(tri, cub_order) {
            let u = cp.pos - center;
            // Powers of each component up to p.
            let mut px = vec![1.0; n];
            let mut py = vec![1.0; n];
            let mut pz = vec![1.0; n];
            for t in 1..n {
                px[t] = px[t - 1] * u.x;
                py[t] = py[t - 1] * u.y;
                pz[t] = pz[t - 1] * u.z;
            }
            let scale = x * cp.w;
            for i in 0..n {
                for j in 0..(n - i) {
                    let pij = scale * px[i] * py[j];
                    for k in 0..(n - i - j) {
                        m[cidx(i, j, k, p)] += pij * pz[k];
                    }
                }
            }
        }
    }
    m
}

/// Evaluate the single-layer Cartesian far field: `Σ_{|k|≤p} a_k(ξ − y_c) · M_k`.
#[must_use]
pub fn eval_single_layer(center: Vec3, moments: &[f64], xi: Vec3, p: usize) -> f64 {
    let a = coulomb_taylor_coeffs(xi - center, p);
    let n = p + 1;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..(n - i) {
            for k in 0..(n - i - j) {
                let id = cidx(i, j, k, p);
                acc += a[id] * moments[id];
            }
        }
    }
    acc
}

/// Number of terms in a total-degree-`p` Cartesian expansion, `C(p+3, 3)`.
#[must_use]
pub fn n_terms(p: usize) -> usize {
    (p + 1) * (p + 2) * (p + 3) / 6
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::expansion::{direct_single_layer, tri_bbox, Cluster};
    use crate::fastsum::expansion::{eval_single_layer as bltc_eval, single_layer_moments as bltc_moments};

    fn tilted_tri() -> Tri {
        Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.3, 0.2),
            Vec3::new(0.2, 1.0, 0.5),
        )
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    #[test]
    fn taylor_coeffs_match_closed_forms() {
        // a_100 = R1/r³ and a_200 = (3R1²−r²)/(2r⁵) (derived analytically).
        let r = Vec3::new(1.0, 2.0, -0.5);
        let a = coulomb_taylor_coeffs(r, 3);
        let r2 = r.dot(r);
        let rn = r2.sqrt();
        assert!(rel(a[cidx(0, 0, 0, 3)], 1.0 / rn) < 1e-13);
        assert!(rel(a[cidx(1, 0, 0, 3)], r.x / rn.powi(3)) < 1e-12, "a_100");
        let a200 = (3.0 * r.x * r.x - r2) / (2.0 * rn.powi(5));
        assert!(rel(a[cidx(2, 0, 0, 3)], a200) < 1e-12, "a_200");
    }

    #[test]
    fn single_panel_far_field_converges_in_p() {
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let center = (lo + hi) * 0.5;
        let xi = Vec3::new(5.0, 4.0, 6.0);
        let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);
        let err = |p: usize| {
            let m = single_layer_moments(center, &[(tri, 1.0)], p);
            rel(eval_single_layer(center, &m, xi, p), reference)
        };
        let e2 = err(2);
        let e8 = err(8);
        assert!(e8 < e2, "Cartesian error should fall with p: {e2:.3e} -> {e8:.3e}");
        assert!(e8 < 1e-9, "p=8 Cartesian far field should be tight, got {e8:.3e}");
    }

    #[test]
    fn cartesian_and_bltc_agree_in_far_field() {
        // Both methods approximate the same integral; in the converged regime they must
        // agree with each other (and the direct reference) to high accuracy.
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let center = (lo + hi) * 0.5;
        let xi = Vec3::new(6.0, 5.0, 7.0);
        let p = 8;
        let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);

        let cm = single_layer_moments(center, &[(tri, 1.0)], p);
        let cart = eval_single_layer(center, &cm, xi, p);

        let c = Cluster::new(lo, hi, p);
        let bm = bltc_moments(&c, &[(tri, 1.0)], p);
        let bltc = bltc_eval(&c, &bm, xi);

        assert!(rel(cart, reference) < 1e-9, "cartesian vs ref");
        assert!(rel(bltc, reference) < 1e-9, "bltc vs ref");
        assert!(rel(cart, bltc) < 1e-8, "cartesian {cart} vs bltc {bltc}");
    }

    #[test]
    fn cartesian_has_fewer_terms_than_bltc() {
        // The cost edge the bake-off weighs: total-degree-p Cartesian vs tensor BLTC.
        for p in [2, 4, 6, 8] {
            let cart = n_terms(p);
            let bltc = (p + 1).pow(3);
            assert!(cart < bltc, "p={p}: cartesian {cart} should be < bltc {bltc}");
        }
        assert_eq!(n_terms(4), 35);
        assert_eq!((4 + 1usize).pow(3), 125);
    }
}
