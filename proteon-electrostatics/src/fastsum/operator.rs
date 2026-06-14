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
use super::expansion::{
    double_layer_moments as bltc_double_moments, eval_double_layer_yukawa,
    eval_single_layer_yukawa, single_layer_moments as bltc_single_moments, Cluster,
};
use super::octree::Octree;
use crate::laplace::laplace_collocation;
use crate::model::{PotentialKind, Tri};
use crate::system::LinearOperator;
use crate::yukawa::regular_yukawa_collocation;

/// Default leaf capacity and depth cap for the octree.
pub const DEFAULT_N_LEAF: usize = 32;
/// Octree depth cap (a backstop; real meshes stop on `n_leaf` far sooner).
pub const DEFAULT_MAX_DEPTH: usize = 24;
/// Maximum supported Cartesian expansion order. Caps the per-node working set at
/// `(p+1)³` (≈ 9.3k coefficients at p=20) so a hostile / fat-fingered order cannot
/// overflow the size arithmetic or request an unreasonable allocation — and so the
/// "treecode memory is O(N)" argument (linear only for *bounded* p) holds. Far beyond
/// any useful order (accuracy saturates against mesh discretization well before this).
pub const MAX_FS_ORDER: usize = 20;

/// A collocation matrix applied by treecode fast summation.
pub struct CollocationTreecode {
    elements: Vec<Tri>,
    centroids: Vec<Vec3>,
    kind: PotentialKind,
    p: usize,
    theta: f64,
    tree: Octree,
    /// Geometry-only FMM interaction lists, present iff the **FMM downward-pass**
    /// path is enabled (`with_fmm`). `Some((m2l_pairs, p2p_pairs))`: well-separated
    /// `(target_node, source_node)` pairs handled by M2L, and inadmissible
    /// leaf-leaf pairs handled by exact P2P. `None` ⇒ the default Barnes–Hut M2P
    /// per-target traversal. The FMM path completes the algorithm (gated vs dense)
    /// but, with the dense O(p⁶) M2L, is NOT a speed win — see the plan.
    interactions: Option<(Vec<(usize, usize)>, Vec<(usize, usize)>)>,
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
        assert!(
            (1..=MAX_FS_ORDER).contains(&p),
            "treecode order p must be in 1..={MAX_FS_ORDER}, got {p}"
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
            interactions: None,
        }
    }

    /// Enable the **FMM downward pass** (M2L + L2L + L2P), precomputing the
    /// geometry-only interaction lists once. Without this the operator uses the
    /// default Barnes–Hut M2P traversal. The FMM matvec is gated bit-parity vs the
    /// dense matrix; with the dense M2L it is not faster (see the plan).
    #[must_use]
    pub fn with_fmm(mut self) -> Self {
        self.interactions = Some(self.build_interactions());
        self
    }

    /// Build the FMM interaction lists by a **dual-tree recursion** over the single
    /// shared tree (adaptive-FMM, not a uniform-tree V-list): from `(root, root)`,
    /// admissible `(A,B)` → one M2L pair; both-leaf inadmissible → one P2P pair;
    /// otherwise split the **larger** (by radius) side and recurse. This partitions
    /// every (target-leaf, source-leaf) pair exactly once — no drop / double-count.
    /// Admissibility is the box-pair MAC `(r_A + r_B) ≤ θ·|c_A − c_B|` (the same
    /// `θ`, convergence-safe). Geometry-only ⇒ computed once, reused per matvec.
    fn build_interactions(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let nodes = &self.tree.nodes;
        let mut m2l = Vec::new();
        let mut p2p = Vec::new();
        // Explicit stack; deterministic emission order (fixed traversal).
        let mut stack = vec![(0usize, 0usize)];
        while let Some((a, b)) = stack.pop() {
            let na = &nodes[a];
            let nb = &nodes[b];
            let d = (na.center - nb.center).norm();
            if d > 0.0 && (na.radius + nb.radius) <= self.theta * d {
                m2l.push((a, b)); // well separated → multipole-to-local
            } else if na.children.is_empty() && nb.children.is_empty() {
                p2p.push((a, b)); // both leaves, not separated → exact near field
            } else {
                // Split the larger side; if one is a leaf, split the other.
                let split_a = if nb.children.is_empty() {
                    true
                } else if na.children.is_empty() {
                    false
                } else {
                    na.radius >= nb.radius
                };
                if split_a {
                    for &c in &na.children {
                        stack.push((c, b));
                    }
                } else {
                    for &c in &nb.children {
                        stack.push((a, c));
                    }
                }
            }
        }
        (m2l, p2p)
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
                let panels: Vec<(Tri, f64)> = node
                    .panels
                    .iter()
                    .map(|&j| (self.elements[j], x[j]))
                    .collect();
                (
                    i,
                    cartesian::single_layer_moments(node.center, node.radius, &panels, self.p),
                )
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
            let inv_r = 1.0 / cartesian::eff_radius(node.radius);
            let mut acc = vec![0.0; sz];
            for &c in &node.children {
                let s = cartesian::eff_radius(nodes[c].radius) * inv_r;
                let t = (nodes[c].center - node.center) * inv_r;
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
                let panels: Vec<(Tri, f64)> = node
                    .panels
                    .iter()
                    .map(|&j| (self.elements[j], x[j]))
                    .collect();
                (
                    i,
                    cartesian::double_layer_moments(node.center, node.radius, &panels, self.p),
                )
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
            let inv_r = 1.0 / cartesian::eff_radius(node.radius);
            let mut acc = vec![zero; sz];
            for &c in &node.children {
                let s = cartesian::eff_radius(nodes[c].radius) * inv_r;
                let t = (nodes[c].center - node.center) * inv_r;
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
                PotentialKind::Single => cartesian::eval_single_layer(
                    node.center,
                    node.radius,
                    &sl[node_idx],
                    xi,
                    self.p,
                ),
                PotentialKind::Double => cartesian::eval_double_layer(
                    node.center,
                    node.radius,
                    &dl[node_idx],
                    xi,
                    self.p,
                ),
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

    /// The **FMM downward-pass** matvec: upward M2M (per-node moments) → M2L into
    /// per-node local expansions over the precomputed interaction list → top-down
    /// L2L sweep → L2P at each leaf centroid (far field) + exact P2P over the near
    /// leaf-leaf list. Serial + deterministic (correctness path; not a speed win
    /// with dense M2L). Each layer's local is scalar, so one `l2l_single` serves
    /// both single and double layer.
    fn fmm_matvec(
        &self,
        x: &[f64],
        y: &mut [f64],
        m2l_pairs: &[(usize, usize)],
        p2p_pairs: &[(usize, usize)],
    ) {
        let nodes = &self.tree.nodes;
        let nn = nodes.len();
        let sz = (self.p + 1).pow(3);
        let (sl, dl) = match self.kind {
            PotentialKind::Single => (self.single_moments(x), Vec::new()),
            PotentialKind::Double => (Vec::new(), self.double_moments(x)),
        };
        // M2L: accumulate each well-separated source's local contribution into the
        // target node's local expansion (fixed list order ⇒ deterministic sum).
        let mut local: Vec<Vec<f64>> = vec![vec![0.0; sz]; nn];
        for &(t, s) in m2l_pairs {
            let (nt, ns) = (&nodes[t], &nodes[s]);
            let contrib = match self.kind {
                PotentialKind::Single => {
                    cartesian::m2l_single(&sl[s], ns.radius, ns.center, nt.radius, nt.center, self.p)
                }
                PotentialKind::Double => {
                    cartesian::m2l_double(&dl[s], ns.radius, ns.center, nt.radius, nt.center, self.p)
                }
            };
            for (a, b) in local[t].iter_mut().zip(&contrib) {
                *a += *b;
            }
        }
        // Downward L2L sweep: parent index < child by construction, so a forward
        // pass has each node's local complete before pushing it to its children.
        for i in 0..nn {
            if nodes[i].children.is_empty() {
                continue;
            }
            let parent_local = local[i].clone();
            let inv_r = 1.0 / cartesian::eff_radius(nodes[i].radius);
            for &c in &nodes[i].children {
                let s = cartesian::eff_radius(nodes[c].radius) * inv_r;
                let t0 = (nodes[c].center - nodes[i].center) * inv_r;
                let contrib = cartesian::l2l_single(&parent_local, s, t0, self.p);
                for (a, b) in local[c].iter_mut().zip(&contrib) {
                    *a += *b;
                }
            }
        }
        // Leaf L2P: far field at each target centroid (one leaf per panel).
        for (i, node) in nodes.iter().enumerate() {
            if !node.children.is_empty() {
                continue;
            }
            for &j in &node.panels {
                y[j] = cartesian::eval_local_single(
                    &local[i],
                    node.radius,
                    node.center,
                    self.centroids[j],
                    self.p,
                );
            }
        }
        // Near field: exact analytic collocation over the inadmissible leaf pairs.
        for &(tl, src) in p2p_pairs {
            for &ti in &nodes[tl].panels {
                let ci = self.centroids[ti];
                let acc: f64 = nodes[src]
                    .panels
                    .iter()
                    .map(|&sj| laplace_collocation(self.kind, ci, &self.elements[sj]) * x[sj])
                    .sum();
                y[ti] += acc;
            }
        }
    }
}

impl CollocationTreecode {
    /// Diagnostic (geometry-only, `x`-independent): total **near-field panel collocations**
    /// and **far-field cluster expansions** summed over all targets in one *treecode*
    /// matvec. The near/far cost ratio decides whether an FMM (which only accelerates the
    /// far field) can help — if the exact near-field collocations dominate, it cannot.
    /// (The `far` count is per-target treecode evals, **not** the number of M2L
    /// operations a real FMM would do — an FMM shares one cluster-pair interaction across
    /// all targets in a cluster.)
    #[doc(hidden)]
    #[must_use]
    pub fn traversal_counts(&self) -> (u64, u64) {
        let mut near = 0u64;
        let mut far = 0u64;
        for i in 0..self.centroids.len() {
            self.count_target(self.centroids[i], 0, &mut near, &mut far);
        }
        (near, far)
    }

    fn count_target(&self, xi: Vec3, node_idx: usize, near: &mut u64, far: &mut u64) {
        let node = &self.tree.nodes[node_idx];
        let d = (xi - node.center).norm();
        if d > 0.0 && node.radius <= self.theta * d {
            *far += 1;
            return;
        }
        if node.children.is_empty() {
            *near += node.panels.len() as u64;
            return;
        }
        for &c in &node.children {
            self.count_target(xi, c, near, far);
        }
    }

    /// Benchmark hook: a full matvec doing **only the far-field** cluster expansions
    /// (near leaves skipped) — times the part an FMM would accelerate.
    #[doc(hidden)]
    pub fn bench_far_only(&self, x: &[f64]) {
        let (sl, dl) = match self.kind {
            PotentialKind::Single => (self.single_moments(x), Vec::new()),
            PotentialKind::Double => (Vec::new(), self.double_moments(x)),
        };
        let mut y = vec![0.0; self.elements.len()];
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = self.eval_split(self.centroids[i], x, &sl, &dl, 0, false, true);
        });
        std::hint::black_box(&y);
    }

    /// Benchmark hook: a full matvec doing **only the near-field** exact collocations
    /// (far clusters skipped) — times the part an FMM does *not* change.
    #[doc(hidden)]
    pub fn bench_near_only(&self, x: &[f64]) {
        let empty_sl: Vec<Vec<f64>> = Vec::new();
        let empty_dl: Vec<Vec<Vec3>> = Vec::new();
        let mut y = vec![0.0; self.elements.len()];
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = self.eval_split(self.centroids[i], x, &empty_sl, &empty_dl, 0, true, false);
        });
        std::hint::black_box(&y);
    }

    /// Traversal that conditionally does near and/or far work (for the cost-split bench).
    fn eval_split(
        &self,
        xi: Vec3,
        x: &[f64],
        sl: &[Vec<f64>],
        dl: &[Vec<Vec3>],
        node_idx: usize,
        do_near: bool,
        do_far: bool,
    ) -> f64 {
        let node = &self.tree.nodes[node_idx];
        let d = (xi - node.center).norm();
        if d > 0.0 && node.radius <= self.theta * d {
            if !do_far {
                return 0.0;
            }
            return match self.kind {
                PotentialKind::Single => cartesian::eval_single_layer(
                    node.center,
                    node.radius,
                    &sl[node_idx],
                    xi,
                    self.p,
                ),
                PotentialKind::Double => cartesian::eval_double_layer(
                    node.center,
                    node.radius,
                    &dl[node_idx],
                    xi,
                    self.p,
                ),
            };
        }
        if node.children.is_empty() {
            if !do_near {
                return 0.0;
            }
            return node
                .panels
                .iter()
                .map(|&j| laplace_collocation(self.kind, xi, &self.elements[j]) * x[j])
                .sum();
        }
        node.children
            .iter()
            .map(|&c| self.eval_split(xi, x, sl, dl, c, do_near, do_far))
            .sum()
    }

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
        // FMM downward pass when enabled; otherwise the Barnes–Hut M2P traversal.
        if let Some((m2l_pairs, p2p_pairs)) = &self.interactions {
            self.fmm_matvec(x, y, m2l_pairs, p2p_pairs);
            return;
        }
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

/// A **regular-Yukawa** collocation matrix (`Vy` single or `Ky` double) applied by
/// treecode fast summation, the Yukawa sibling of [`CollocationTreecode`]. Uses the
/// **barycentric-Lagrange** far field (the moments are kernel-independent — identical to
/// the Laplace single/double-layer moments — so the only Yukawa-specific part is the
/// proxy-point kernel eval; plan §3.2 hedge). The near field is the exact
/// `regular_yukawa_collocation`.
///
/// v1 builds node moments directly per matvec (no M2M — the BLTC M2M differs from the
/// Cartesian one, and the matvec is traversal-bound regardless). The per-node Chebyshev
/// proxy grids are built once at construction.
pub struct YukawaTreecode {
    elements: Vec<Tri>,
    centroids: Vec<Vec3>,
    kind: PotentialKind,
    kappa: f64,
    p: usize,
    theta: f64,
    tree: Octree,
    clusters: Vec<Cluster>,
}

impl YukawaTreecode {
    /// Build the regular-Yukawa treecode for `kind` (`Single`=`Vy` / `Double`=`Ky`) with
    /// exponent `kappa`, expansion order `p`, MAC ratio `theta`.
    ///
    /// # Panics
    /// If `theta ∉ (0,1)` or `p ∉ 1..=MAX_FS_ORDER`.
    #[must_use]
    pub fn new(elements: &[Tri], kind: PotentialKind, kappa: f64, p: usize, theta: f64) -> Self {
        assert!(
            theta.is_finite() && theta > 0.0 && theta < 1.0,
            "treecode MAC ratio theta must be in (0, 1), got {theta}"
        );
        assert!(
            (1..=MAX_FS_ORDER).contains(&p),
            "treecode order p must be in 1..={MAX_FS_ORDER}, got {p}"
        );
        assert!(
            kappa.is_finite() && kappa >= 0.0,
            "Yukawa exponent kappa must be finite and >= 0, got {kappa}"
        );
        let centroids: Vec<Vec3> = elements
            .iter()
            .map(|t| (t.v1 + t.v2 + t.v3) * (1.0 / 3.0))
            .collect();
        let tree = Octree::build(elements, &centroids, DEFAULT_N_LEAF, DEFAULT_MAX_DEPTH);
        // Per-node Chebyshev grids (x-independent — built once).
        let clusters: Vec<Cluster> = tree
            .nodes
            .iter()
            .map(|nd| Cluster::new(nd.lo, nd.hi, p))
            .collect();
        Self {
            elements: elements.to_vec(),
            centroids,
            kind,
            kappa,
            p,
            theta,
            tree,
            clusters,
        }
    }

    /// Per-node moments from the current `x` (direct from each node's subtree panels).
    fn single_moments(&self, x: &[f64]) -> Vec<Vec<f64>> {
        self.tree
            .nodes
            .par_iter()
            .enumerate()
            .map(|(i, node)| {
                let panels: Vec<(Tri, f64)> = node
                    .panels
                    .iter()
                    .map(|&j| (self.elements[j], x[j]))
                    .collect();
                bltc_single_moments(&self.clusters[i], &panels, self.p)
            })
            .collect()
    }

    fn double_moments(&self, x: &[f64]) -> Vec<Vec<Vec3>> {
        self.tree
            .nodes
            .par_iter()
            .enumerate()
            .map(|(i, node)| {
                let panels: Vec<(Tri, f64)> = node
                    .panels
                    .iter()
                    .map(|&j| (self.elements[j], x[j]))
                    .collect();
                bltc_double_moments(&self.clusters[i], &panels, self.p)
            })
            .collect()
    }

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
        if d > 0.0 && node.radius <= self.theta * d {
            return match self.kind {
                PotentialKind::Single => eval_single_layer_yukawa(
                    &self.clusters[node_idx],
                    &sl[node_idx],
                    xi,
                    self.kappa,
                ),
                PotentialKind::Double => eval_double_layer_yukawa(
                    &self.clusters[node_idx],
                    &dl[node_idx],
                    xi,
                    self.kappa,
                ),
            };
        }
        if node.children.is_empty() {
            return node
                .panels
                .iter()
                .map(|&j| {
                    regular_yukawa_collocation(self.kind, xi, &self.elements[j], self.kappa) * x[j]
                })
                .sum();
        }
        node.children
            .iter()
            .map(|&c| self.eval_target(xi, x, sl, dl, c))
            .sum()
    }
}

impl LinearOperator for YukawaTreecode {
    fn dim(&self) -> usize {
        self.elements.len()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let (sl, dl) = match self.kind {
            PotentialKind::Single => (self.single_moments(x), Vec::new()),
            PotentialKind::Double => (Vec::new(), self.double_moments(x)),
        };
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            *yi = self.eval_target(self.centroids[i], x, &sl, &dl, 0);
        });
    }
    fn diagonal(&self) -> Vec<f64> {
        (0..self.elements.len())
            .map(|i| {
                regular_yukawa_collocation(
                    self.kind,
                    self.centroids[i],
                    &self.elements[i],
                    self.kappa,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic::analytic_sphere_mesh;
    use crate::system::{laplace_matrices, yukawa_matrices};

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
        let num: f64 = a
            .iter()
            .zip(b)
            .map(|(p, q)| (p - q).powi(2))
            .sum::<f64>()
            .sqrt();
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
        assert!(
            e < 1e-4,
            "single-layer treecode matvec off dense by {e:.3e}"
        );
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

        let x: Vec<f64> = (0..n)
            .map(|i| (((i * 5) % 11) as f64 - 5.0) * 0.1)
            .collect();
        let mut y_dense = vec![0.0; n];
        let mut y_tree = vec![0.0; n];
        k_dense.matvec(&x, &mut y_dense);
        tree.matvec(&x, &mut y_tree);

        let e = rel_l2(&y_tree, &y_dense);
        eprintln!("double-layer matvec rel L2 (p=8, θ=0.45): {e:.3e}");
        assert!(
            e < 1e-4,
            "double-layer treecode matvec off dense by {e:.3e}"
        );
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
        assert!(
            e8 < e3,
            "matvec error should fall with p: {e3:.3e} -> {e8:.3e}"
        );
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
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
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
    fn yukawa_matvec_matches_dense() {
        // Vy (single) and Ky (double) regular-Yukawa treecode matvecs vs the dense
        // yukawa_matrices, on a sphere SES.
        let els = sphere_elements(3);
        let n = els.len();
        let kappa = 0.25;
        let (vy_dense, ky_dense) = yukawa_matrices(&els, kappa);
        let x: Vec<f64> = (0..n)
            .map(|i| (((i * 5) % 11) as f64 - 5.0) * 0.1)
            .collect();

        for (kind, dense) in [
            (PotentialKind::Single, &vy_dense),
            (PotentialKind::Double, &ky_dense),
        ] {
            let (p, theta) = match kind {
                PotentialKind::Single => (7, 0.5),
                PotentialKind::Double => (8, 0.45),
            };
            let tree = YukawaTreecode::new(&els, kind, kappa, p, theta);
            let mut yd = vec![0.0; n];
            let mut yt = vec![0.0; n];
            dense.matvec(&x, &mut yd);
            tree.matvec(&x, &mut yt);
            let e = rel_l2(&yt, &yd);
            eprintln!("Yukawa {kind:?} matvec rel L2: {e:.3e}");
            assert!(e < 1e-3, "Yukawa {kind:?} treecode off dense by {e:.3e}");
            // Diagonal is the exact self-collocation.
            let dd = dense.diagonal();
            let td = tree.diagonal();
            for (a, b) in td.iter().zip(&dd) {
                assert!((a - b).abs() < 1e-12, "Yukawa {kind:?} diagonal {a} vs {b}");
            }
        }
    }

    #[test]
    fn fmm_matvec_matches_dense() {
        // Absolute correctness contract (codex): the FMM downward pass must
        // reproduce the dense matrix for BOTH layers — L2 *and* worst-row — not
        // merely beat Barnes–Hut. Uses the box-pair MAC (stricter than the BH
        // point MAC), so the same (p, θ) is at least as accurate.
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, k_dense) = laplace_matrices(&els);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();

        for (kind, dense) in [
            (PotentialKind::Single, &v_dense),
            (PotentialKind::Double, &k_dense),
        ] {
            let (p, theta) = match kind {
                PotentialKind::Single => (7, 0.5),
                PotentialKind::Double => (9, 0.45),
            };
            let tree = CollocationTreecode::new(&els, kind, p, theta).with_fmm();
            let mut yd = vec![0.0; n];
            let mut yf = vec![0.0; n];
            dense.matvec(&x, &mut yd);
            tree.matvec(&x, &mut yf);
            let e_l2 = rel_l2(&yf, &yd);
            let e_row = rowwise_max(&yf, &yd);
            eprintln!("FMM {kind:?}: L2 {e_l2:.3e}, rowwise-max {e_row:.3e}");
            assert!(e_l2 < 1e-4, "FMM {kind:?} L2 {e_l2:.3e} off dense");
            assert!(e_row < 1e-3, "FMM {kind:?} worst-row {e_row:.3e} off dense");
        }
    }

    #[test]
    fn fmm_basis_vector_recovers_dense_column() {
        // x = e_j isolates column j: the FMM must reproduce M[i][j] for every
        // target i — exercises one source's far field delivered to all targets via
        // M2L→L2L→L2P, catching a dropped or double-counted interaction.
        let els = sphere_elements(3);
        let n = els.len();
        let (v_dense, _k) = laplace_matrices(&els);
        let tree = CollocationTreecode::new(&els, PotentialKind::Single, 7, 0.5).with_fmm();
        for &j in &[0usize, n / 3, n / 2, n - 1] {
            let mut x = vec![0.0; n];
            x[j] = 1.0;
            let mut y = vec![0.0; n];
            tree.matvec(&x, &mut y);
            let col: Vec<f64> = (0..n).map(|i| v_dense.get(i, j)).collect();
            let e = rowwise_max(&y, &col);
            assert!(e < 1e-3, "FMM column {j} off dense by {e:.3e}");
        }
    }

    #[test]
    fn fmm_interaction_lists_partition_all_panel_pairs() {
        // Structural guard against drop/double-count: every (target panel, source
        // panel) pair must be covered EXACTLY ONCE — by a P2P leaf pair, or by an
        // M2L pair (some target-ancestor × source-ancestor that contains it). Check
        // it directly on the small tree.
        let els = sphere_elements(2);
        let n = els.len();
        let tree = CollocationTreecode::new(&els, PotentialKind::Single, 6, 0.5).with_fmm();
        let (m2l, p2p) = tree.interactions.as_ref().unwrap();
        let nodes = &tree.tree.nodes;
        // subtree panel sets per node.
        let mut cover = vec![0u32; n * n];
        for &(t, s) in p2p {
            for &ti in &nodes[t].panels {
                for &sj in &nodes[s].panels {
                    cover[ti * n + sj] += 1;
                }
            }
        }
        for &(t, s) in m2l {
            for &ti in &nodes[t].panels {
                for &sj in &nodes[s].panels {
                    cover[ti * n + sj] += 1;
                }
            }
        }
        let bad = cover.iter().filter(|&&c| c != 1).count();
        assert_eq!(bad, 0, "{bad} (target,source) panel pairs not covered exactly once");
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
