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

use super::cubature::{panel_order_for_cartesian, triangle_cubature};
use crate::model::Tri;

/// Lower bound on a cluster radius used for normalization (a genuinely zero-radius
/// cluster — all panel points coincident — cannot arise from non-degenerate triangles).
const MIN_RADIUS: f64 = 1e-300;

/// The effective normalization radius — clamped so a (degenerate) zero/underflowed radius
/// cannot divide-by-zero. Used by every consumer (leaf moments, eval, **and** the M2M
/// upward pass) so child and parent frames stay consistent.
#[must_use]
pub fn eff_radius(r: f64) -> f64 {
    r.max(MIN_RADIUS)
}

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

/// Panel-aware single-layer Cartesian moments in **radius-normalized** coordinates
/// `û = (y − y_c)/R`: `M̂_k = Σ_j x_j ∫_{T_j} û^k dS`, dense `(p+1)³`. Normalizing by
/// the cluster radius `R` keeps `û ∈ [−1,1]` (panel points lie inside the box), so the
/// monomial moments stay O(area) instead of growing like `R^|k|` — the eval rescales by
/// `R^|k|` to recover the physical value. This removes the over/underflow path the raw
/// `u^k · a_k` product risks on scaled meshes / larger `p`.
#[must_use]
pub fn single_layer_moments(center: Vec3, radius: f64, panels: &[(Tri, f64)], p: usize) -> Vec<f64> {
    let n = p + 1;
    let mut m = vec![0.0; n * n * n];
    let cub_order = panel_order_for_cartesian(p);
    let inv_r = 1.0 / radius.max(MIN_RADIUS);
    for (tri, x) in panels {
        for cp in triangle_cubature(tri, cub_order) {
            let u = (cp.pos - center) * inv_r; // normalized
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

/// Evaluate the single-layer Cartesian far field: `Σ_{|k|≤p} [a_k(ξ−y_c)·R^|k|] · M̂_k`,
/// the radius-scaled Taylor coefficients contracted with the normalized moments. With
/// `R/dist ≤ θ < 1` (the MAC), the scaled coefficients are bounded by `θ^|k|` — no
/// overflow regardless of mesh scale.
#[must_use]
pub fn eval_single_layer(center: Vec3, radius: f64, moments: &[f64], xi: Vec3, p: usize) -> f64 {
    let a = coulomb_taylor_coeffs(xi - center, p);
    let r = radius.max(MIN_RADIUS);
    let mut rpow = vec![1.0; p + 1];
    for d in 1..=p {
        rpow[d] = rpow[d - 1] * r;
    }
    let n = p + 1;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..(n - i) {
            for k in 0..(n - i - j) {
                let id = cidx(i, j, k, p);
                acc += a[id] * rpow[i + j + k] * moments[id];
            }
        }
    }
    acc
}

/// Panel-aware double-layer **vector** moments in radius-normalized coordinates
/// `û = (y−y_c)/R`: `Ŵ_m = Σ_j x_j n_j ∫_{T_j} û^m dS`, dense `(p+1)³` of `Vec3`
/// (the dipole eval uses degrees `|m| ≤ p−1`). Normalization as in
/// [`single_layer_moments`].
#[must_use]
pub fn double_layer_moments(center: Vec3, radius: f64, panels: &[(Tri, f64)], p: usize) -> Vec<Vec3> {
    let n = p + 1;
    let mut w = vec![Vec3::new(0.0, 0.0, 0.0); n * n * n];
    let cub_order = panel_order_for_cartesian(p);
    let inv_r = 1.0 / radius.max(MIN_RADIUS);
    for (tri, x) in panels {
        let nrm = tri.normal;
        for cp in triangle_cubature(tri, cub_order) {
            let u = (cp.pos - center) * inv_r; // normalized
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
                        let id = cidx(i, j, k, p);
                        w[id] = w[id] + nrm * (pij * pz[k]);
                    }
                }
            }
        }
    }
    w
}

/// Evaluate the double-layer Cartesian far field: `n·∇_y G` expanded gives
/// `Σ_{|k|≤p} a_k Σ_i k_i (W_{k−e_i})_i`. In normalized coordinates the moment of
/// degree `|k|−1` carries `R^{|k|−1}` and the `∇_y` (acting on `û = u/R`) carries an
/// extra `1/R`, so the term scale is `a_k · R^{|k|−1}` — applied here as the
/// radius-scaled coefficient against the normalized vector moment.
#[must_use]
pub fn eval_double_layer(center: Vec3, radius: f64, w_moments: &[Vec3], xi: Vec3, p: usize) -> f64 {
    let a = coulomb_taylor_coeffs(xi - center, p);
    let r = radius.max(MIN_RADIUS);
    let mut rpow = vec![1.0; p + 1];
    for d in 1..=p {
        rpow[d] = rpow[d - 1] * r;
    }
    let n = p + 1;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..(n - i) {
            for k in 0..(n - i - j) {
                let deg = i + j + k;
                if deg == 0 {
                    continue; // k_i ≥ 1 required for the degree-lowered moment
                }
                let ak = a[cidx(i, j, k, p)] * rpow[deg - 1];
                let idx = [i, j, k];
                let mut s = 0.0;
                for (d, &kd) in idx.iter().enumerate() {
                    if kd >= 1 {
                        let mut m = idx;
                        m[d] -= 1;
                        let wm = w_moments[cidx(m[0], m[1], m[2], p)];
                        let wcomp = [wm.x, wm.y, wm.z][d];
                        s += kd as f64 * wcomp;
                    }
                }
                acc += ak * s;
            }
        }
    }
    acc
}

/// 1-D translation matrix `T[k][m] = C(k,m)·s^m·t^{k−m}` (`m ≤ k`, else 0), flattened
/// row-major `(p+1)²`. Re-expresses normalized monomials of a child cluster
/// (`û_child = (y−c_child)/R_child`) in the parent's normalized coordinate
/// (`û_parent = (y−c_parent)/R_parent`) along one axis, with `s = R_child/R_parent` and
/// `t = (c_child−c_parent)/R_parent`.
fn trans_matrix_1d(s: f64, t: f64, p: usize) -> Vec<f64> {
    let n = p + 1;
    // Pascal triangle for C(k,m).
    let mut binom = vec![0.0; n * n];
    for k in 0..n {
        binom[k * n] = 1.0;
        for m in 1..=k {
            binom[k * n + m] = binom[(k - 1) * n + m - 1] + binom[(k - 1) * n + m];
        }
    }
    let mut spow = vec![1.0; n];
    let mut tpow = vec![1.0; n];
    for i in 1..n {
        spow[i] = spow[i - 1] * s;
        tpow[i] = tpow[i - 1] * t;
    }
    let mut tm = vec![0.0; n * n];
    for k in 0..n {
        for m in 0..=k {
            tm[k * n + m] = binom[k * n + m] * spow[m] * tpow[k - m];
        }
    }
    tm
}

/// Translate a child cluster's **normalized** scalar moments to the parent center/radius
/// (Cartesian M2M). `s = R_child/R_parent`, `t = (c_child − c_parent)/R_parent`. Separable
/// per axis (`O(p⁴)`), so the upward pass that builds all node moments is `O(N·p⁴)`
/// instead of the direct `O(N·depth·p³·cub)` rebuild. Returns the parent-frame
/// contribution of this child (sum over a node's children to get its moments).
#[must_use]
pub fn m2m_single(child: &[f64], s: f64, t: Vec3, p: usize) -> Vec<f64> {
    let n = p + 1;
    let tx = trans_matrix_1d(s, t.x, p);
    let ty = trans_matrix_1d(s, t.y, p);
    let tz = trans_matrix_1d(s, t.z, p);
    // Step z: B[mx][my][kz] = Σ_{mz≤kz} Tz[kz][mz] · child[mx][my][mz].
    let mut b = vec![0.0; n * n * n];
    for mx in 0..n {
        for my in 0..n {
            let base = (mx * n + my) * n;
            for kz in 0..n {
                let mut acc = 0.0;
                for mz in 0..=kz {
                    acc += tz[kz * n + mz] * child[base + mz];
                }
                b[base + kz] = acc;
            }
        }
    }
    // Step y: C[mx][ky][kz] = Σ_{my≤ky} Ty[ky][my] · B[mx][my][kz].
    let mut c = vec![0.0; n * n * n];
    for mx in 0..n {
        for ky in 0..n {
            for kz in 0..n {
                let mut acc = 0.0;
                for my in 0..=ky {
                    acc += ty[ky * n + my] * b[(mx * n + my) * n + kz];
                }
                c[(mx * n + ky) * n + kz] = acc;
            }
        }
    }
    // Step x: P[kx][ky][kz] = Σ_{mx≤kx} Tx[kx][mx] · C[mx][ky][kz].
    let mut out = vec![0.0; n * n * n];
    for kx in 0..n {
        for ky in 0..n {
            for kz in 0..n {
                let mut acc = 0.0;
                for mx in 0..=kx {
                    acc += tx[kx * n + mx] * c[(mx * n + ky) * n + kz];
                }
                out[(kx * n + ky) * n + kz] = acc;
            }
        }
    }
    out
}

/// Cartesian M2M for the double-layer **vector** moments — the same scalar translation
/// applied to each component independently (the shift acts on the polynomial index
/// structure, not the vector component).
#[must_use]
pub fn m2m_double(child: &[Vec3], s: f64, t: Vec3, p: usize) -> Vec<Vec3> {
    let len = child.len();
    let cx: Vec<f64> = child.iter().map(|v| v.x).collect();
    let cy: Vec<f64> = child.iter().map(|v| v.y).collect();
    let cz: Vec<f64> = child.iter().map(|v| v.z).collect();
    let px = m2m_single(&cx, s, t, p);
    let py = m2m_single(&cy, s, t, p);
    let pz = m2m_single(&cz, s, t, p);
    (0..len).map(|i| Vec3::new(px[i], py[i], pz[i])).collect()
}

/// **M2L** (multipole→local): translate a source cluster's normalized multipole moments
/// `M̂_k` (about `c_s`, radius `R_s`) into a **local expansion** `L_m` about the target
/// center `c_t` (radius `R_t`), valid for targets near `c_t` and well separated from the
/// source. Derivation (re-expanding the multipole potential `Σ_k a_k(t−c_s)R_s^|k| M̂_k`
/// as a Taylor series in `v = (t−c_t)/R_t`):
///
/// ```text
/// L_m = (−1)^|m| R_t^|m| Σ_k R_s^|k| · C(k+m, m) · a_{k+m}(D) · M̂_k ,   D = c_t − c_s
/// ```
///
/// with `a_n(D)` the Coulomb Taylor coefficients to order `2p` and `C(k+m,m)` the
/// per-axis multi-index binomial. This is the FMM operation the §5.3 measurement
/// justifies: computed **once per source/target-cluster pair**, then shared by all
/// targets in the target cluster via [`eval_local_single`]. `O(p⁶)` per pair (the
/// classic dense Cartesian M2L cost).
///
/// `pub(crate)`: an internal FMM building block, not public API — the full downward pass
/// (interaction lists, L2L, admissibility, accumulation) is deferred (plan §5.4).
///
/// **Scaling assumption:** `R_s^|k|`, `R_t^|m|`, and the dimensional `a_{k+m}(D)` are
/// formed separately, so although their product is well-scaled for a well-separated pair,
/// the intermediates can over/underflow on *extreme* coordinate scales. Adequate for
/// molecular meshes (Å, well-separated by the MAC); a nondimensional rewrite
/// (`R_s/|D|`, `R_t/|D|`, one final `1/|D|`) is the hardening for the production FMM.
///
/// # Panics (debug)
/// On non-finite / non-positive radii, a non-finite or coincident (`D = 0`) center pair,
/// or a moment slice of the wrong length.
#[must_use]
pub(crate) fn m2l_single(
    src_moments: &[f64],
    r_s: f64,
    c_s: Vec3,
    r_t: f64,
    c_t: Vec3,
    p: usize,
) -> Vec<f64> {
    let n = p + 1;
    debug_assert_eq!(src_moments.len(), n * n * n, "moment slice must be (p+1)³");
    debug_assert!(r_s.is_finite() && r_s > 0.0 && r_t.is_finite() && r_t > 0.0, "radii > 0");
    let d_vec = c_t - c_s;
    debug_assert!(d_vec.norm() > 0.0 && d_vec.norm().is_finite(), "centers must differ + be finite");
    let a = coulomb_taylor_coeffs(d_vec, 2 * p); // (2p+1)³ Taylor cube
    let bcap = 2 * p + 1;
    // Pascal triangle C(i,j) up to 2p.
    let mut binom = vec![0.0; bcap * bcap];
    for i in 0..bcap {
        binom[i * bcap] = 1.0;
        for j in 1..=i {
            binom[i * bcap + j] = binom[(i - 1) * bcap + j - 1] + binom[(i - 1) * bcap + j];
        }
    }
    let pow = |base: f64, up: usize| {
        let mut v = vec![1.0; up + 1];
        for t in 1..=up {
            v[t] = v[t - 1] * base;
        }
        v
    };
    let rs_pow = pow(r_s.max(MIN_RADIUS), p);
    let rt_pow = pow(r_t.max(MIN_RADIUS), p);

    let mut l = vec![0.0; n * n * n];
    for mi in 0..n {
        for mj in 0..(n - mi) {
            for mk in 0..(n - mi - mj) {
                let mdeg = mi + mj + mk;
                let mut acc = 0.0;
                for ki in 0..n {
                    let bx = binom[(ki + mi) * bcap + mi];
                    for kj in 0..(n - ki) {
                        let bxy = bx * binom[(kj + mj) * bcap + mj];
                        for kk in 0..(n - ki - kj) {
                            let kdeg = ki + kj + kk;
                            let cbin = bxy * binom[(kk + mk) * bcap + mk];
                            let an = a[cidx(ki + mi, kj + mj, kk + mk, 2 * p)];
                            acc += rs_pow[kdeg] * cbin * an * src_moments[cidx(ki, kj, kk, p)];
                        }
                    }
                }
                let sign = if mdeg % 2 == 0 { 1.0 } else { -1.0 };
                l[cidx(mi, mj, mk, p)] = sign * rt_pow[mdeg] * acc;
            }
        }
    }
    l
}

/// Evaluate a single-layer **local** expansion `Σ_m L_m·((t−c_t)/R_t)^m` at target `t`
/// (the FMM L2P step — cheap, no per-source work). `pub(crate)`: internal FMM building
/// block (see [`m2l_single`]).
#[must_use]
pub(crate) fn eval_local_single(local: &[f64], r_t: f64, c_t: Vec3, t: Vec3, p: usize) -> f64 {
    let n = p + 1;
    let v = (t - c_t) * (1.0 / r_t.max(MIN_RADIUS));
    let mut vx = vec![1.0; n];
    let mut vy = vec![1.0; n];
    let mut vz = vec![1.0; n];
    for i in 1..n {
        vx[i] = vx[i - 1] * v.x;
        vy[i] = vy[i - 1] * v.y;
        vz[i] = vz[i - 1] * v.z;
    }
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..(n - i) {
            for k in 0..(n - i - j) {
                acc += local[cidx(i, j, k, p)] * vx[i] * vy[j] * vz[k];
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
        let radius = (hi - lo).norm() * 0.5;
        let xi = Vec3::new(5.0, 4.0, 6.0);
        let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);
        let err = |p: usize| {
            let m = single_layer_moments(center, radius, &[(tri, 1.0)], p);
            rel(eval_single_layer(center, radius, &m, xi, p), reference)
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
        let radius = (hi - lo).norm() * 0.5;
        let xi = Vec3::new(6.0, 5.0, 7.0);
        let p = 8;
        let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);

        let cm = single_layer_moments(center, radius, &[(tri, 1.0)], p);
        let cart = eval_single_layer(center, radius, &cm, xi, p);

        let c = Cluster::new(lo, hi, p);
        let bm = bltc_moments(&c, &[(tri, 1.0)], p);
        let bltc = bltc_eval(&c, &bm, xi);

        assert!(rel(cart, reference) < 1e-9, "cartesian vs ref");
        assert!(rel(bltc, reference) < 1e-9, "bltc vs ref");
        assert!(rel(cart, bltc) < 1e-8, "cartesian {cart} vs bltc {bltc}");
    }

    #[test]
    fn cartesian_double_layer_converges_and_matches() {
        use crate::fastsum::expansion::direct_double_layer;
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let center = (lo + hi) * 0.5;
        let radius = (hi - lo).norm() * 0.5;
        let xi = Vec3::new(5.0, 4.0, 6.0);
        let reference = direct_double_layer(&[(tri, 1.0)], xi, 24);
        assert!(reference.abs() > 1e-6, "non-trivial reference: {reference}");
        let err = |p: usize| {
            let w = double_layer_moments(center, radius, &[(tri, 1.0)], p);
            rel(eval_double_layer(center, radius, &w, xi, p), reference)
        };
        let e3 = err(3);
        let e8 = err(8);
        assert!(e8 < e3, "Cartesian dipole should converge: {e3:.3e} -> {e8:.3e}");
        assert!(e8 < 1e-8, "p=8 Cartesian dipole should be tight, got {e8:.3e}");
    }

    #[test]
    fn cartesian_dipole_flips_with_normal() {
        use crate::fastsum::expansion::direct_double_layer;
        let tri = tilted_tri();
        let xi = Vec3::new(4.0, 5.0, 3.0);
        let bare = direct_double_layer(&[(tri, 1.0)], xi, 24);
        let flipped = Tri::new(tri.v1, tri.v3, tri.v2);
        let (lo, hi) = tri_bbox(&flipped);
        let center = (lo + hi) * 0.5;
        let radius = (hi - lo).norm() * 0.5;
        let w = double_layer_moments(center, radius, &[(flipped, 1.0)], 8);
        let got = eval_double_layer(center, radius, &w, xi, 8);
        assert!(rel(got, -bare) < 1e-6, "flipped {got} vs -bare {}", -bare);
    }

    #[test]
    fn m2m_single_matches_direct_parent_moments() {
        // A panel's moments translated from its own (child) frame to a parent frame must
        // equal the panel's moments computed directly about the parent — the M2M identity
        // the upward pass relies on.
        let tri = tilted_tri();
        let p = 7;
        let (lo, hi) = tri_bbox(&tri);
        let c_child = (lo + hi) * 0.5;
        let r_child = (hi - lo).norm() * 0.5;
        // A larger parent box that contains the panel, with a shifted center.
        let c_parent = c_child + Vec3::new(0.4, -0.3, 0.2);
        let r_parent = r_child * 2.5;

        let child = single_layer_moments(c_child, r_child, &[(tri, 1.3)], p);
        let translated = m2m_single(&child, r_child / r_parent, (c_child - c_parent) * (1.0 / r_parent), p);
        let direct = single_layer_moments(c_parent, r_parent, &[(tri, 1.3)], p);

        let n = p + 1;
        let mut maxerr = 0.0_f64;
        for i in 0..n {
            for j in 0..(n - i) {
                for k in 0..(n - i - j) {
                    let id = cidx(i, j, k, p);
                    maxerr = maxerr.max((translated[id] - direct[id]).abs());
                }
            }
        }
        assert!(maxerr < 1e-12, "M2M vs direct parent moments max abs err {maxerr:.3e}");
    }

    #[test]
    fn m2m_double_matches_direct_parent_moments() {
        let tri = tilted_tri();
        let p = 6;
        let (lo, hi) = tri_bbox(&tri);
        let c_child = (lo + hi) * 0.5;
        let r_child = (hi - lo).norm() * 0.5;
        let c_parent = c_child + Vec3::new(-0.2, 0.5, 0.1);
        let r_parent = r_child * 3.0;

        let child = double_layer_moments(c_child, r_child, &[(tri, 0.7)], p);
        let translated = m2m_double(&child, r_child / r_parent, (c_child - c_parent) * (1.0 / r_parent), p);
        let direct = double_layer_moments(c_parent, r_parent, &[(tri, 0.7)], p);

        let n = p + 1;
        let mut maxerr = 0.0_f64;
        for i in 0..n {
            for j in 0..(n - i) {
                for k in 0..(n - i - j) {
                    let id = cidx(i, j, k, p);
                    let d = translated[id] - direct[id];
                    maxerr = maxerr.max(d.x.abs().max(d.y.abs()).max(d.z.abs()));
                }
            }
        }
        assert!(maxerr < 1e-12, "M2M dipole vs direct max abs err {maxerr:.3e}");
    }

    #[test]
    fn m2l_then_local_eval_matches_direct_multipole() {
        // FMM self-consistency: M2L (source multipole → target local) + L2P must reproduce
        // the direct multipole eval at a target inside a well-separated target cluster — to
        // machine precision (this isolates sign/index errors), across radius ratios.
        let tri = tilted_tri();
        let p = 8;
        let (lo, hi) = tri_bbox(&tri);
        let c_s = (lo + hi) * 0.5;
        let r_s = (hi - lo).norm() * 0.5;
        let m_src = single_layer_moments(c_s, r_s, &[(tri, 1.0)], p);

        for (c_t, r_t) in [
            (Vec3::new(6.0, 5.0, 7.0), 0.4),
            (Vec3::new(6.0, 5.0, 7.0), 1.2),  // r_t ≠ r_s (radius-ratio coverage)
            (Vec3::new(-5.0, 6.0, -4.0), 0.7), // different direction
        ] {
            let local = m2l_single(&m_src, r_s, c_s, r_t, c_t, p);
            for off in [Vec3::new(0.1, -0.05, 0.08), Vec3::new(-0.12, 0.2, -0.07), Vec3::new(0.0, 0.0, 0.0)] {
                let t = c_t + off * (r_t / 0.4); // stay inside the target ball
                let via_local = eval_local_single(&local, r_t, c_t, t, p);
                let direct = eval_single_layer(c_s, r_s, &m_src, t, p);
                assert!(rel(via_local, direct) < 1e-9, "M2L+L2P {via_local} vs multipole {direct}");
            }
        }
    }

    #[test]
    fn m2l_then_local_eval_matches_true_panel_integral() {
        // Stronger gate (codex): M2L + L2P vs the DIRECT panel integral — the real far
        // field, not just the truncated multipole. Tests multipole truncation + M2L + L2P
        // end to end, well separated (tight) and near the MAC boundary (converges in p).
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let c_s = (lo + hi) * 0.5;
        let r_s = (hi - lo).norm() * 0.5;

        // Well separated (|D| ≈ 8·r_s): p=10 reaches near machine precision.
        let c_t = Vec3::new(6.0, 5.0, 7.0);
        let r_t = 0.4;
        let t = c_t + Vec3::new(0.1, -0.05, 0.08);
        let reference = direct_single_layer(&[(tri, 1.0)], t, 24);
        let err = |p: usize| {
            let m = single_layer_moments(c_s, r_s, &[(tri, 1.0)], p);
            let local = m2l_single(&m, r_s, c_s, r_t, c_t, p);
            rel(eval_local_single(&local, r_t, c_t, t, p), reference)
        };
        assert!(err(10) < 1e-9, "well-separated M2L+L2P vs panel integral: {:.2e}", err(10));

        // Near the admissibility boundary (target cluster only ~2.5·r_s away): larger
        // error, but it must still CONVERGE with p (the expansion is valid, just slower).
        let c_t2 = c_s + Vec3::new(1.6, 0.0, 0.0);
        let r_t2 = 0.2;
        let t2 = c_t2 + Vec3::new(0.05, 0.03, -0.04);
        let ref2 = direct_single_layer(&[(tri, 1.0)], t2, 24);
        let err2 = |p: usize| {
            let m = single_layer_moments(c_s, r_s, &[(tri, 1.0)], p);
            let local = m2l_single(&m, r_s, c_s, r_t2, c_t2, p);
            rel(eval_local_single(&local, r_t2, c_t2, t2, p), ref2)
        };
        assert!(err2(10) < err2(4), "near-boundary M2L must converge in p: {:.2e}->{:.2e}", err2(4), err2(10));
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
