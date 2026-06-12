//! Panel-aware barycentric cluster expansions (the BLTC far field).
//!
//! A *cluster* is an axis-aligned box holding some source panels. Its far-field
//! potential at a target `ξ` outside the box is approximated by interpolating the
//! kernel `G(ξ, ·)` on a tensor Chebyshev grid over the box, then integrating that
//! polynomial interpolant **over each panel exactly** (panel-aware moments — not a
//! centroid collapse, whose `O(h²/r³)` error no expansion order can remove):
//!
//! ```text
//! single layer:  Σ_j x_j ∫_{T_j} G(ξ,y) dS_y  ≈  Σ_K G(ξ, s_K) · Q_K
//!                                          Q_K = Σ_j x_j ∫_{T_j} L_K(y) dS_y
//! ```
//!
//! with `s_K` the 3-D Chebyshev proxy points and `L_K` the tensor barycentric
//! Lagrange basis. Accuracy is controlled by the expansion order `p`; the panel
//! integrals use [`cubature`](super::cubature) (exact for the polynomial basis).
//!
//! This is the isolated harness (P8.1): no octree, no operator. The double-layer
//! (vector-moment) expansion and a Cartesian-multipole alternative land alongside,
//! for the bake-off the plan calls for.

use proteon_core::surface::geom::Vec3;

use super::cheb;
use super::cubature::{panel_order_for_degree, triangle_cubature};
use crate::model::Tri;

/// Below this box-extent (in a single axis) the axis is treated as **flat**: one
/// Chebyshev node instead of `p+1`, since a planar panel's bounding box collapses in
/// its normal direction and a multi-node grid there would be singular.
const FLAT_AXIS_EPS: f64 = 1e-12;

/// A cluster's Chebyshev tensor grid over its bounding box. Per-axis node counts may
/// differ (a flat axis collapses to one node), so this stores each axis independently.
pub struct Cluster {
    nodes: [Vec<f64>; 3],
    weights: [Vec<f64>; 3],
}

impl Cluster {
    /// Build the grid for box `[lo, hi]` at expansion order `p` (per axis, dropping to
    /// order 0 on a flat axis).
    #[must_use]
    pub fn new(lo: Vec3, hi: Vec3, p: usize) -> Self {
        let lo = [lo.x, lo.y, lo.z];
        let hi = [hi.x, hi.y, hi.z];
        let mut nodes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut weights: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for d in 0..3 {
            let flat = (hi[d] - lo[d]).abs() < FLAT_AXIS_EPS;
            let pd = if flat { 0 } else { p };
            nodes[d] = cheb::nodes(pd, lo[d], hi[d]);
            weights[d] = cheb::bary_weights(pd);
        }
        Self { nodes, weights }
    }

    /// `(nx, ny, nz)` node counts per axis.
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nodes[0].len(), self.nodes[1].len(), self.nodes[2].len())
    }

    /// Number of proxy points `(nx·ny·nz)`.
    #[must_use]
    pub fn n_proxy(&self) -> usize {
        let (a, b, c) = self.dims();
        a * b * c
    }

    /// The proxy point at flat index `idx = (i·ny + j)·nz + k`.
    #[must_use]
    fn proxy(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(self.nodes[0][i], self.nodes[1][j], self.nodes[2][k])
    }

    /// Per-axis Lagrange basis values at `y`: `(Lx, Ly, Lz)`.
    fn basis_at(&self, y: Vec3) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            cheb::lagrange_basis(&self.nodes[0], &self.weights[0], y.x),
            cheb::lagrange_basis(&self.nodes[1], &self.weights[1], y.y),
            cheb::lagrange_basis(&self.nodes[2], &self.weights[2], y.z),
        )
    }
}

/// Single-layer panel-aware moments `Q_K = Σ_j x_j ∫_{T_j} L_K(y) dS_y`, flattened as
/// `(i·ny + j)·nz + k`. `panels` is `(triangle, source weight x_j)`; `p` sets the panel
/// cubature order (high enough to integrate the degree-`p` basis exactly).
#[must_use]
pub fn single_layer_moments(cluster: &Cluster, panels: &[(Tri, f64)], p: usize) -> Vec<f64> {
    let (nx, ny, nz) = cluster.dims();
    let mut q = vec![0.0; nx * ny * nz];
    let cub_order = panel_order_for_degree(p);
    for (tri, x) in panels {
        for cp in triangle_cubature(tri, cub_order) {
            let (lx, ly, lz) = cluster.basis_at(cp.pos);
            let scale = x * cp.w;
            for i in 0..nx {
                let li = scale * lx[i];
                for j in 0..ny {
                    let lij = li * ly[j];
                    let base = (i * ny + j) * nz;
                    for k in 0..nz {
                        q[base + k] += lij * lz[k];
                    }
                }
            }
        }
    }
    q
}

/// Double-layer **vector** moments `Q_K = Σ_j x_j n_j ∫_{T_j} L_K(y) dS_y` (one `Vec3`
/// per proxy point), flattened as `(i·ny + j)·nz + k`. The dipole far field interpolates
/// the vector field `∇_y G(ξ,·)` over the cluster and contracts it with these moments —
/// the explicit vector-moment data model (not "carry n into a scalar weight"), so the
/// normal orientation is faithfully represented and the double-layer sign is correct.
#[must_use]
pub fn double_layer_moments(cluster: &Cluster, panels: &[(Tri, f64)], p: usize) -> Vec<Vec3> {
    let (nx, ny, nz) = cluster.dims();
    let mut q = vec![Vec3::new(0.0, 0.0, 0.0); nx * ny * nz];
    let cub_order = panel_order_for_degree(p);
    for (tri, x) in panels {
        let n = tri.normal;
        for cp in triangle_cubature(tri, cub_order) {
            let (lx, ly, lz) = cluster.basis_at(cp.pos);
            let scale = x * cp.w;
            for i in 0..nx {
                let li = scale * lx[i];
                for j in 0..ny {
                    let lij = li * ly[j];
                    let base = (i * ny + j) * nz;
                    for k in 0..nz {
                        q[base + k] = q[base + k] + n * (lij * lz[k]);
                    }
                }
            }
        }
    }
    q
}

/// Evaluate the double-layer far field at `xi`: `Σ_K ∇_y G(xi, s_K) · Q_K`, where
/// `∇_y G(xi, s) = (xi − s)/|xi − s|³`.
#[must_use]
pub fn eval_double_layer(cluster: &Cluster, moments: &[Vec3], xi: Vec3) -> f64 {
    let (nx, ny, nz) = cluster.dims();
    let mut acc = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let base = (i * ny + j) * nz;
            for k in 0..nz {
                let s = cluster.proxy(i, j, k);
                let d = xi - s;
                let r = d.norm();
                let grad = d * (1.0 / (r * r * r));
                acc += grad.dot(moments[base + k]);
            }
        }
    }
    acc
}

/// Direct reference: `Σ_j x_j ∫_{T_j} (xi − y)·n_j / |xi − y|³ dS_y` by high-order
/// cubature — the bare double-layer integral, which (confirmed by test) equals
/// `laplace_collocation(Double, …)`.
#[must_use]
pub fn direct_double_layer(panels: &[(Tri, f64)], xi: Vec3, cub_order: usize) -> f64 {
    let mut acc = 0.0;
    for (tri, x) in panels {
        let n = tri.normal;
        for cp in triangle_cubature(tri, cub_order) {
            let d = xi - cp.pos;
            let r = d.norm();
            acc += x * cp.w * d.dot(n) / (r * r * r);
        }
    }
    acc
}

/// Evaluate the single-layer far field at target `xi`: `Σ_K (1/|xi − s_K|) · Q_K`.
#[must_use]
pub fn eval_single_layer(cluster: &Cluster, moments: &[f64], xi: Vec3) -> f64 {
    let (nx, ny, nz) = cluster.dims();
    let mut acc = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let base = (i * ny + j) * nz;
            for k in 0..nz {
                let s = cluster.proxy(i, j, k);
                let r = (xi - s).norm();
                acc += moments[base + k] / r;
            }
        }
    }
    acc
}

/// Direct reference: `Σ_j x_j ∫_{T_j} 1/|xi − y| dS_y` by high-order triangle cubature.
/// The isolated convergence gate compares the cluster expansion against this — the
/// single-layer integral the expansion approximates, which (confirmed by test) is
/// exactly what `laplace_collocation(Single, …)` returns, so the treecode far field
/// needs no rescaling to match the dense operator.
#[must_use]
pub fn direct_single_layer(panels: &[(Tri, f64)], xi: Vec3, cub_order: usize) -> f64 {
    let mut acc = 0.0;
    for (tri, x) in panels {
        for cp in triangle_cubature(tri, cub_order) {
            acc += x * cp.w / (xi - cp.pos).norm();
        }
    }
    acc
}

/// Axis-aligned bounding box of a triangle's vertices.
#[must_use]
pub fn tri_bbox(tri: &Tri) -> (Vec3, Vec3) {
    let xs = [tri.v1.x, tri.v2.x, tri.v3.x];
    let ys = [tri.v1.y, tri.v2.y, tri.v3.y];
    let zs = [tri.v1.z, tri.v2.z, tri.v3.z];
    let mn = |a: [f64; 3]| a.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = |a: [f64; 3]| a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (Vec3::new(mn(xs), mn(ys), mn(zs)), Vec3::new(mx(xs), mx(ys), mx(zs)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplace::laplace_collocation;
    use crate::model::PotentialKind;

    fn tilted_tri() -> Tri {
        // A generic tilted triangle whose bbox is non-degenerate in all 3 axes.
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
    fn single_panel_far_field_converges_in_p() {
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let xi = Vec3::new(5.0, 4.0, 6.0); // far from the unit-ish panel
        // High-order direct reference for the bare integral.
        let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);
        let err = |p: usize| {
            let c = Cluster::new(lo, hi, p);
            let q = single_layer_moments(&c, &[(tri, 1.0)], p);
            rel(eval_single_layer(&c, &q, xi), reference)
        };
        let e2 = err(2);
        let e8 = err(8);
        assert!(e8 < e2, "error should fall with p: {e2:.3e} -> {e8:.3e}");
        assert!(e8 < 1e-9, "p=8 far field should be tight, got {e8:.3e}");
    }

    #[test]
    fn far_field_error_grows_as_target_approaches() {
        // The expansion is a far-field approximation: fixed p, error worsens as ξ nears
        // the box (still converges in p, but the constant grows) — sanity on the MAC.
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let c = Cluster::new(lo, hi, 6);
        let q = single_layer_moments(&c, &[(tri, 1.0)], 6);
        let err_at = |d: f64| {
            let xi = Vec3::new(d, d, d);
            let reference = direct_single_layer(&[(tri, 1.0)], xi, 24);
            rel(eval_single_layer(&c, &q, xi), reference)
        };
        let far = err_at(8.0);
        let near = err_at(2.0);
        assert!(far < near, "nearer target = larger error: near {near:.3e} far {far:.3e}");
        assert!(far < 1e-8, "well-separated target should be accurate: {far:.3e}");
    }

    #[test]
    fn double_layer_far_field_converges_in_p() {
        let tri = tilted_tri();
        let (lo, hi) = tri_bbox(&tri);
        let xi = Vec3::new(5.0, 4.0, 6.0);
        let reference = direct_double_layer(&[(tri, 1.0)], xi, 24);
        assert!(reference.abs() > 1e-6, "reference should be non-trivial: {reference}");
        let err = |p: usize| {
            let c = Cluster::new(lo, hi, p);
            let q = double_layer_moments(&c, &[(tri, 1.0)], p);
            rel(eval_double_layer(&c, &q, xi), reference)
        };
        let e2 = err(2);
        let e8 = err(8);
        assert!(e8 < e2, "double-layer error should fall with p: {e2:.3e} -> {e8:.3e}");
        assert!(e8 < 1e-8, "p=8 double-layer far field should be tight, got {e8:.3e}");
    }

    #[test]
    fn double_layer_matches_collocation_and_flips_with_normal() {
        let tri = tilted_tri();
        let xi = Vec3::new(4.0, 5.0, 3.0);
        // Convention: bare double-layer integral = laplace_collocation(Double).
        let bare = direct_double_layer(&[(tri, 1.0)], xi, 24);
        let coll = laplace_collocation(PotentialKind::Double, xi, &tri);
        assert!(rel(bare, coll) < 1e-9, "double bare {bare} vs collocation {coll}");

        // Sign: reversing the panel orientation (swap two vertices) flips the normal and
        // must flip the double-layer sign — the expansion tracks it through the vector
        // moment, not just the direct reference.
        let flipped = Tri::new(tri.v1, tri.v3, tri.v2);
        let (lo, hi) = tri_bbox(&flipped);
        let c = Cluster::new(lo, hi, 8);
        let q = double_layer_moments(&c, &[(flipped, 1.0)], 8);
        let expansion_flipped = eval_double_layer(&c, &q, xi);
        assert!(rel(expansion_flipped, -bare) < 1e-6, "flipped {expansion_flipped} vs -bare {}", -bare);
    }

    #[test]
    fn matches_laplace_collocation_convention() {
        // Pin the convention for wiring: the bare single-layer integral ∫_T 1/r dA IS
        // what laplace_collocation(Single) returns — NESSie's 1/(4π·r) kernel
        // premultiplied by 4π is exactly 1/r, so the treecode far field needs no extra
        // factor to match the dense operator.
        let tri = tilted_tri();
        let xi = Vec3::new(4.0, 5.0, 3.0);
        let bare = direct_single_layer(&[(tri, 1.0)], xi, 24);
        let coll = laplace_collocation(PotentialKind::Single, xi, &tri);
        assert!(rel(bare, coll) < 1e-9, "bare {bare} vs collocation {coll}");
    }
}

