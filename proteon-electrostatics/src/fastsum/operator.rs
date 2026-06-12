//! `CollocationTreecode` — a [`LinearOperator`] that applies a Laplace collocation
//! matrix (`V` single layer or `K` double layer) by **fast summation**, a drop-in for
//! the dense `DenseOperator` inside the local/nonlocal system operators.
//!
//! Per matvec `y_i = Σ_j M[i][j] x_j`, `M[i][j] = laplace_collocation(kind, c_i, T_j)`:
//! - **far field** — a cluster well separated from the target `c_i` (box-separation MAC
//!   `radius/dist ≤ θ` on the vertex-enclosing box) contributes via the panel-aware
//!   Cartesian expansion ([`super::cartesian`]); accuracy is set by `(p, θ)`.
//! - **near field** — non-separated leaves contribute the **exact** analytic
//!   `laplace_collocation` per panel (this is where the self / near-singular entries
//!   stay exact, preserving the reference-tier accuracy).
//!
//! Because the panel-aware moments integrate the basis over the real panel and the
//! boxes enclose vertices, box separation (MAC) is the *only* admissibility condition
//! needed — a panel always lies inside its node's box, so there is no separate
//! per-panel distance test to get wrong (unlike a centroid-collapse far field).
//!
//! v1 cost (measured, not assumed — plan §3.3): per matvec, node moments are rebuilt
//! from `x` directly from each node's subtree panels (`O(N·depth·p³·cub)`); the M2M
//! upward pass that would make this linear is deferred. The octree + panel geometry are
//! built once at construction.

use proteon_core::surface::geom::Vec3;
use rayon::prelude::*;

use super::cartesian;
use super::octree::Octree;
use crate::laplace::laplace_collocation;
use crate::model::{PotentialKind, Tri};
use crate::system::LinearOperator;

/// Default leaf capacity and depth cap for the octree.
pub const DEFAULT_N_LEAF: usize = 32;
/// Octree depth cap (a backstop; real meshes stop on `n_leaf` far sooner).
pub const DEFAULT_MAX_DEPTH: usize = 24;

/// A collocation matrix applied by treecode fast summation.
pub struct CollocationTreecode {
    elements: Vec<Tri>,
    centroids: Vec<Vec3>,
    kind: PotentialKind,
    p: usize,
    theta: f64,
    tree: Octree,
}

impl CollocationTreecode {
    /// Build the treecode operator for `kind` (`Single`/`Double`) at expansion order
    /// `p` and MAC ratio `theta`.
    #[must_use]
    pub fn new(elements: &[Tri], kind: PotentialKind, p: usize, theta: f64) -> Self {
        Self::with_params(elements, kind, p, theta, DEFAULT_N_LEAF, DEFAULT_MAX_DEPTH)
    }

    /// Build with explicit octree parameters.
    ///
    /// # Panics
    /// If `theta` is not in `(0, 1)`. The enclosing-box MAC `radius/dist ≤ θ` is only
    /// sound for `0 < θ < 1`: a panel point satisfies `|u| ≤ radius`, so `θ < 1`
    /// guarantees the target is strictly outside the box and the Taylor expansion
    /// converges. With `θ ≥ 1` a target inside the box could pass the MAC and hit a
    /// singular/divergent expansion instead of the exact near field.
    #[must_use]
    pub fn with_params(
        elements: &[Tri],
        kind: PotentialKind,
        p: usize,
        theta: f64,
        n_leaf: usize,
        max_depth: usize,
    ) -> Self {
        assert!(
            theta.is_finite() && theta > 0.0 && theta < 1.0,
            "treecode MAC ratio theta must be in (0, 1), got {theta}"
        );
        let centroids: Vec<Vec3> = elements
            .iter()
            .map(|t| (t.v1 + t.v2 + t.v3) * (1.0 / 3.0))
            .collect();
        let tree = Octree::build(elements, &centroids, n_leaf, max_depth);
        Self {
            elements: elements.to_vec(),
            centroids,
            kind,
            p,
            theta,
            tree,
        }
    }

    /// Per-node single-layer moments via the **M2M upward pass**: leaf moments are built
    /// from their panels (parallel, the only cubature), then each internal node's moments
    /// are the M2M-translated sum of its children's — `O(N·p³)` leaf work + `O(N·p⁴)`
    /// translations, instead of the direct `O(N·depth·p³·cub)` per-node rebuild. (`build`
    /// guarantees a parent's index is below its children's, so a reverse pass has every
    /// child ready.)
    fn single_moments(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let nodes = &self.tree.nodes;
        let nn = nodes.len();
        let sz = (self.p + 1).pow(3);
        let leaves: Vec<(usize, Vec<f64>)> = nodes
            .par_iter()
            .enumerate()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(i, node)| {
                let panels: Vec<(Tri, f64)> =
                    node.panels.iter().map(|&j| (self.elements[j], x[j])).collect();
                (i, cartesian::single_layer_moments(node.center, node.radius, &panels, self.p))
            })
            .collect();
        let mut moments: Vec<Vec<f64>> = vec![Vec::new(); nn];
        for (i, m) in leaves {
            moments[i] = m;
        }
        for i in (0..nn).rev() {
            let node = &nodes[i];
            if node.children.is_empty() {
                continue;
            }
            let mut acc = vec![0.0; sz];
            for &c in &node.children {
                let s = nodes[c].radius / node.radius;
                let t = (nodes[c].center - node.center) * (1.0 / node.radius);
                let tc = cartesian::m2m_single(&moments[c], s, t, self.p);
                for (a, b) in acc.iter_mut().zip(&tc) {
                    *a += *b;
                }
            }
            moments[i] = acc;
        }
        moments
    }

    /// Per-node double-layer vector moments via the M2M upward pass (as [`single_moments`]).
    fn double_moments(&self, x: &[f64]) -> Vec<Vec<Vec3>> {
        let nodes = &self.tree.nodes;
        let nn = nodes.len();
        let sz = (self.p + 1).pow(3);
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let leaves: Vec<(usize, Vec<Vec3>)> = nodes
            .par_iter()
            .enumerate()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(i, node)| {
                let panels: Vec<(Tri, f64)> =
                    node.panels.iter().map(|&j| (self.elements[j], x[j])).collect();
                (i, cartesian::double_layer_moments(node.center, node.radius, &panels, self.p))
            })
            .collect();
        let mut moments: Vec<Vec<Vec3>> = vec![Vec::new(); nn];
        for (i, m) in leaves {
            moments[i] = m;
        }
        for i in (0..nn).rev() {
            let node = &nodes[i];
            if node.children.is_empty() {
                continue;
            }
            let mut acc = vec![zero; sz];
            for &c in &node.children {
                let s = nodes[c].radius / node.radius;
                let t = (nodes[c].center - node.center) * (1.0 / node.radius);
                let tc = cartesian::m2m_double(&moments[c], s, t, self.p);
                for (a, b) in acc.iter_mut().zip(&tc) {
                    *a = *a + *b;
                }
            }
            moments[i] = acc;
        }
        moments
    }

    /// Evaluate `y_i` at target `xi` by traversing the tree (serial per target, so the
    /// summation order is fixed and the result is deterministic).
    fn eval_target(
        &self,
        xi: Vec3,
        x: &[f64],
        sl: &[Vec<f64>],
        dl: &[Vec<Vec3>],
        node_idx: usize,
    ) -> f64 {
        let node = &self.tree.nodes[node_idx];
        let d = (xi - node.center).norm();
        // Box-separation MAC on the vertex-enclosing box.
        if d > 0.0 && node.radius <= self.theta * d {
            return match self.kind {
                PotentialKind::Single => {
                    cartesian::eval_single_layer(node.center, node.radius, &sl[node_idx], xi, self.p)
                }
                PotentialKind::Double => {
                    cartesian::eval_double_layer(node.center, node.radius, &dl[node_idx], xi, self.p)
                }
            };
        }
        if node.children.is_empty() {
            // Exact near field.
            return node
                .panels
                .iter()
                .map(|&j| laplace_collocation(self.kind, xi, &self.elements[j]) * x[j])
                .sum();
        }
        node.children
            .iter()
            .map(|&c| self.eval_target(xi, x, sl, dl, c))
            .sum()
    }
}

impl CollocationTreecode {
    /// Benchmark hook: run only the per-matvec moment rebuild (no traversal), so a caller
    /// can measure the rebuild's share of the matvec. Not part of the solve path.
    #[doc(hidden)]
    pub fn bench_rebuild(&self, x: &[f64]) {
        match self.kind {
            PotentialKind::Single => {
                std::hint::black_box(self.single_moments(x));
            }
            PotentialKind::Double => {
                std::hint::black_box(self.double_moments(x));
            }
        }
    }
}

impl LinearOperator for CollocationTreecode {
    fn dim(&self) -> usize {
        self.elements.len()
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        // Rebuild moments from x (once per matvec), then evaluate each target.
        let (sl, dl) = match self.kind {
            PotentialKind::Single => (self.single_moments(x), Vec::new()),
            PotentialKind::Double => (Vec::new(), self.double_moments(x)),
        };
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = self.eval_target(self.centroids[i], x, &sl, &dl, 0);
        });
    }

    fn diagonal(&self) -> Vec<f64> {
        // Self-interaction M[i][i] — always the exact near-field collocation.
        (0..self.elements.len())
            .map(|i| laplace_collocation(self.kind, self.centroids[i], &self.elements[i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic::analytic_sphere_mesh;
    use crate::system::laplace_matrices;

    fn sphere_elements(subdiv: u32) -> Vec<Tri> {
        let mesh = analytic_sphere_mesh(2.0, subdiv);
        mesh.tris
            .iter()
            .map(|t| {
                Tri::new(
                    mesh.verts[t[0] as usize],
                    mesh.verts[t[1] as usize],
                    mesh.verts[t[2] as usize],
                )
            })
            .collect()
    }

    fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
        let den: f64 = b.iter().map(|q| q * q).sum::<f64>().sqrt().max(1e-300);
        num / den
    }

    #[test]
    fn single_layer_matvec_matches_dense() {
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, _k) = laplace_matrices(&els);
        let tree = CollocationTreecode::new(&els, PotentialKind::Single, 6, 0.5);

        // A representative right-hand side.
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let mut y_dense = vec![0.0; n];
        let mut y_tree = vec![0.0; n];
        v_dense.matvec(&x, &mut y_dense);
        tree.matvec(&x, &mut y_tree);

        let e = rel_l2(&y_tree, &y_dense);
        eprintln!("single-layer matvec rel L2 (p=6, θ=0.5): {e:.3e}");
        assert!(e < 1e-4, "single-layer treecode matvec off dense by {e:.3e}");
    }

    #[test]
    fn double_layer_matvec_matches_dense() {
        // The double-layer kernel (∝ 1/r²) is more singular than the single layer, so it
        // needs a higher order / tighter MAC for the same matvec accuracy — p=8, θ=0.45
        // here vs p=6, θ=0.5 for the single layer.
        let els = sphere_elements(3);
        let n = els.len();
        let (_v, k_dense) = laplace_matrices(&els);
        let tree = CollocationTreecode::new(&els, PotentialKind::Double, 8, 0.45);

        let x: Vec<f64> = (0..n).map(|i| (((i * 5) % 11) as f64 - 5.0) * 0.1).collect();
        let mut y_dense = vec![0.0; n];
        let mut y_tree = vec![0.0; n];
        k_dense.matvec(&x, &mut y_dense);
        tree.matvec(&x, &mut y_tree);

        let e = rel_l2(&y_tree, &y_dense);
        eprintln!("double-layer matvec rel L2 (p=8, θ=0.45): {e:.3e}");
        assert!(e < 1e-4, "double-layer treecode matvec off dense by {e:.3e}");
    }

    #[test]
    fn accuracy_improves_with_p() {
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, _k) = laplace_matrices(&els);
        let x: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();
        let mut y_dense = vec![0.0; n];
        v_dense.matvec(&x, &mut y_dense);

        let err_p = |p: usize| {
            let tree = CollocationTreecode::new(&els, PotentialKind::Single, p, 0.5);
            let mut y = vec![0.0; n];
            tree.matvec(&x, &mut y);
            rel_l2(&y, &y_dense)
        };
        let e3 = err_p(3);
        let e8 = err_p(8);
        assert!(e8 < e3, "matvec error should fall with p: {e3:.3e} -> {e8:.3e}");
    }

    /// Max over rows of `|y_tree[i] − y_dense[i]| / (|y_dense[i]| + scale)` — catches a
    /// single bad row that an L2 norm could average away.
    fn rowwise_max(a: &[f64], b: &[f64]) -> f64 {
        let scale = b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-30);
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs() / (q.abs() + scale))
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn matvec_rowwise_max_matches_dense() {
        // L2 can hide a localized bad row (especially the double layer) — gate the worst
        // row directly, for both layers.
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, k_dense) = laplace_matrices(&els);
        let x: Vec<f64> = (0..n).map(|i| ((i * 3 % 17) as f64 - 8.0) * 0.25).collect();

        for (kind, dense) in [
            (PotentialKind::Single, &v_dense),
            (PotentialKind::Double, &k_dense),
        ] {
            let (p, theta) = match kind {
                PotentialKind::Single => (7, 0.5),
                PotentialKind::Double => (9, 0.45),
            };
            let tree = CollocationTreecode::new(&els, kind, p, theta);
            let mut yd = vec![0.0; n];
            let mut yt = vec![0.0; n];
            dense.matvec(&x, &mut yd);
            tree.matvec(&x, &mut yt);
            let e = rowwise_max(&yt, &yd);
            eprintln!("{kind:?} rowwise-max: {e:.3e}");
            assert!(e < 1e-3, "{kind:?} worst-row error {e:.3e} too large");
        }
    }

    #[test]
    fn basis_vector_rhs_recovers_dense_column() {
        // x = e_j isolates column j: y_i must reproduce M[i][j] = collocation(c_i, T_j).
        // Tests an individual source's far field to every target (not just an aggregate).
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, _k) = laplace_matrices(&els);
        let tree = CollocationTreecode::new(&els, PotentialKind::Single, 7, 0.5);
        for &j in &[0usize, n / 3, n / 2, n - 1] {
            let mut x = vec![0.0; n];
            x[j] = 1.0;
            let mut y = vec![0.0; n];
            tree.matvec(&x, &mut y);
            let col: Vec<f64> = (0..n).map(|i| v_dense.get(i, j)).collect();
            let e = rowwise_max(&y, &col);
            assert!(e < 1e-3, "column {j} off dense by {e:.3e}");
        }
    }

    #[test]
    fn matvec_reproducible_across_thread_counts() {
        // Determinism: parallel-over-targets + serial per-target traversal ⇒ a fixed
        // summation order, so the result is bit-identical regardless of thread count.
        let els = sphere_elements(3);
        let n = els.len();
        let tree = CollocationTreecode::new(&els, PotentialKind::Double, 6, 0.5);
        let x: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut y = vec![0.0; n];
            pool.install(|| tree.matvec(&x, &mut y));
            y
        };
        let y1 = run(1);
        let y4 = run(4);
        assert_eq!(y1, y4, "matvec must be bit-identical across thread counts");
    }

    #[test]
    #[should_panic(expected = "theta must be in (0, 1)")]
    fn invalid_theta_rejected() {
        let els = sphere_elements(2);
        let _ = CollocationTreecode::new(&els, PotentialKind::Single, 4, 1.0);
    }

    #[test]
    fn diagonal_matches_dense() {
        let els = sphere_elements(2);
        let (v_dense, _k) = laplace_matrices(&els);
        let tree = CollocationTreecode::new(&els, PotentialKind::Single, 4, 0.5);
        let dd = v_dense.diagonal();
        let td = tree.diagonal();
        for (a, b) in td.iter().zip(&dd) {
            assert!((a - b).abs() < 1e-12, "diagonal {a} vs {b}");
        }
    }
}
