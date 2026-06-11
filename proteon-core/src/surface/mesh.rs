//! Triangle mesh + the invariants the L4 triangulation is gated on.
//!
//! Index-based (`verts` + `tris`), the clean Rust shape — not BALL's
//! pointer-linked `TrianglePoint*`/`Triangle*`. The invariants here are exactly
//! the L4 oracle gates (`TO_SES_TRIANGULATION.md`): surface area (→ analytic
//! SES area as density rises), watertightness / manifoldness (every edge shared
//! by two triangles), signed enclosed volume (orientation — catches inside-out),
//! and Euler characteristic per the topology.
//!
//! Also provides `icosphere` — the contact-face primitive. An isolated atom's
//! SES *is* its sphere, so this alone is gateable against `ses_area` (4πr²).

use super::geom::Vec3;
use std::collections::{HashMap, HashSet};

/// Does segment `p0→p1` pass through the **interior** of triangle `(a,b,c)`?
/// Möller–Trumbore, with a strict-interior margin so a mere boundary/endpoint touch
/// (e.g. a shared edge) is not a crossing — only a clear *transverse* penetration counts.
/// `u`/`v`/`t` are dimensionless (normalized), so their absolute margin is scale-free;
/// the parallel cutoff is taken **relative** to the operand magnitudes so it does not
/// become scale-dependent (review).
fn segment_penetrates_triangle(p0: Vec3, p1: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    const EPS: f64 = 1e-9;
    let dir = p1 - p0;
    let (e1, e2) = (b - a, c - a);
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    // |det| = |e1||pvec|·|cos∠|; compare relative to |e1||pvec| (scale-invariant).
    if det.abs() <= EPS * e1.norm() * pvec.norm() {
        return false; // segment (near-)parallel to the triangle plane
    }
    let inv = 1.0 / det;
    let tvec = p0 - a;
    let u = tvec.dot(pvec) * inv;
    if u <= EPS || u >= 1.0 - EPS {
        return false;
    }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv;
    if v <= EPS || u + v >= 1.0 - EPS {
        return false;
    }
    let t = e2.dot(qvec) * inv; // position along the segment
    t > EPS && t < 1.0 - EPS
}

/// Do two triangles penetrate each other? An interior crossing has the intersection
/// segment ending on an edge of each triangle, so some edge of one passes through the
/// other's interior — test all six. (Coplanar overlap is not detected here; that is the
/// duplicate/overlapping-face defect, handled separately.)
fn triangles_penetrate(t1: (Vec3, Vec3, Vec3), t2: (Vec3, Vec3, Vec3)) -> bool {
    let (a1, b1, c1) = t1;
    let (a2, b2, c2) = t2;
    let e1 = [(a1, b1), (b1, c1), (c1, a1)];
    let e2 = [(a2, b2), (b2, c2), (c2, a2)];
    e1.iter()
        .any(|&(p, q)| segment_penetrates_triangle(p, q, a2, b2, c2))
        || e2
            .iter()
            .any(|&(p, q)| segment_penetrates_triangle(p, q, a1, b1, c1))
}

/// An index-based triangle mesh. `normals` is optional (empty if not computed).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub verts: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn num_vertices(&self) -> usize {
        self.verts.len()
    }
    pub fn num_triangles(&self) -> usize {
        self.tris.len()
    }

    fn tri_points(&self, t: [u32; 3]) -> (Vec3, Vec3, Vec3) {
        (
            self.verts[t[0] as usize],
            self.verts[t[1] as usize],
            self.verts[t[2] as usize],
        )
    }

    /// Total triangle area.
    pub fn surface_area(&self) -> f64 {
        self.tris
            .iter()
            .map(|&t| {
                let (a, b, c) = self.tri_points(t);
                0.5 * (b - a).cross(c - a).norm()
            })
            .sum()
    }

    /// Signed enclosed volume (divergence theorem). **Only meaningful for a
    /// closed mesh** (`is_watertight()`): the sum is translation-invariant only
    /// when every edge is shared, so for an open mesh it shifts with the origin.
    /// Gate it together with `num_nonmanifold_edges() == 0`. Positive ⇒
    /// consistently outward-oriented; a watertight inside-out mesh is negative,
    /// which is why orientation is checked separately (`is_consistently_oriented`).
    pub fn signed_volume(&self) -> f64 {
        self.tris
            .iter()
            .map(|&t| {
                let (a, b, c) = self.tri_points(t);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Undirected edges → how many triangles use each.
    fn edge_use_counts(&self) -> HashMap<(u32, u32), u32> {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for &t in &self.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Edges used by a number of triangles other than two (boundary or
    /// non-manifold). Zero ⇒ closed manifold.
    pub fn num_nonmanifold_edges(&self) -> usize {
        self.edge_use_counts().values().filter(|&&c| c != 2).count()
    }

    /// Closed manifold: every edge shared by exactly two triangles.
    pub fn is_watertight(&self) -> bool {
        self.num_nonmanifold_edges() == 0
    }

    /// Consistent outward winding: every directed edge `(a,b)` is used exactly
    /// once and its reverse `(b,a)` exactly once. Closed + consistently-oriented
    /// is what makes `signed_volume` trustworthy (undirected edge counts alone
    /// miss flipped-winding triangles that still pair up).
    pub fn is_consistently_oriented(&self) -> bool {
        let mut dir: HashMap<(u32, u32), i32> = HashMap::new();
        for &t in &self.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *dir.entry((a, b)).or_insert(0) += 1;
            }
        }
        dir.iter()
            .all(|(&(a, b), &c)| c == 1 && dir.get(&(b, a)) == Some(&1))
    }

    /// Compact connected-component label (`0..k`) per triangle, components being
    /// triangles joined transitively by a **shared edge** (union–find, path halving).
    /// Edge connectivity (not vertex) is the surface notion: two shells touching at a
    /// single vertex stay separate components.
    pub fn component_labels(&self) -> Vec<usize> {
        let n = self.tris.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        }
        // undirected edge → first triangle that owns it; union on the rest.
        let mut edge_owner: HashMap<(u32, u32), usize> = HashMap::new();
        for (ti, &t) in self.tris.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&other) = edge_owner.get(&key) {
                    let (ra, rb) = (find(&mut parent, ti), find(&mut parent, other));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                } else {
                    edge_owner.insert(key, ti);
                }
            }
        }
        // Compact roots → 0..k.
        let mut label = vec![0usize; n];
        let mut next = 0usize;
        let mut root_label: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            label[i] = *root_label.entry(r).or_insert_with(|| {
                let l = next;
                next += 1;
                l
            });
        }
        label
    }

    /// Number of connected surface components — `1` for a single sphere, `>1` for a
    /// multi-body surface (a protein SES: several solute bodies + buried cavities).
    pub fn num_connected_components(&self) -> usize {
        if self.tris.is_empty() {
            return 0;
        }
        self.component_labels().iter().copied().max().map_or(0, |m| m + 1)
    }

    /// `(signed volume, area)` of **each** connected component (divergence theorem +
    /// triangle areas). Aggregate `signed_volume` can mask an inward component behind a
    /// larger outward one, so orientation must be judged per component.
    pub fn component_volumes_areas(&self) -> Vec<(f64, f64)> {
        let labels = self.component_labels();
        let k = labels.iter().copied().max().map_or(0, |m| m + 1);
        let mut va = vec![(0.0_f64, 0.0_f64); k];
        for (ti, &t) in self.tris.iter().enumerate() {
            let (a, b, c) = self.tri_points(t);
            va[labels[ti]].0 += a.dot(b.cross(c)) / 6.0;
            va[labels[ti]].1 += 0.5 * (b - a).cross(c - a).norm();
        }
        va
    }

    /// Flip each connected component whose signed volume is **negative** (inside-out) so
    /// every component is outward-oriented. Returns whether any component was flipped.
    /// Per-component (not global) — a multi-body mesh with mixed orientation is fixed
    /// without disturbing already-outward bodies.
    pub fn orient_outward(&mut self) -> bool {
        let labels = self.component_labels();
        let vols: Vec<f64> = self.component_volumes_areas().iter().map(|&(v, _)| v).collect();
        let mut flipped = false;
        for (ti, t) in self.tris.iter_mut().enumerate() {
            if vols[labels[ti]] < 0.0 {
                t.swap(1, 2);
                flipped = true;
            }
        }
        flipped
    }

    /// Number of **duplicate** faces: triangles sharing the same vertex set (counted
    /// as the surplus beyond one per distinct set). Coincident faces are a common mesh
    /// bug that breaks edge-manifold counts and double-counts the surface.
    pub fn num_duplicate_faces(&self) -> usize {
        let mut seen: HashMap<[u32; 3], u32> = HashMap::new();
        let mut dups = 0;
        for &t in &self.tris {
            let mut key = t;
            key.sort_unstable();
            let c = seen.entry(key).or_insert(0);
            if *c >= 1 {
                dups += 1;
            }
            *c += 1;
        }
        dups
    }

    /// Count **self-intersecting** triangle pairs — triangles that *transversely*
    /// penetrate each other — or `None` if the mesh is too irregular to check within a
    /// bounded budget (degenerate sizing, wildly multi-scale triangles, or pathologically
    /// dense cells). A self-intersecting surface has no well-defined interior/exterior, so
    /// the BEM interior-source model breaks.
    ///
    /// **Honest semantics (review):** this is a *detector*, biased toward false negatives.
    /// `Some(k>0)` is authoritative (those crossings are real); `Some(0)` means "no clear
    /// transverse penetration found", **not** a proof of a clean surface — coplanar
    /// overlaps, exact vertex-on-face / edge-on-edge contacts, and crossings inside the
    /// strict-interior margin are not reported. `None` means "could not verify" → the
    /// caller should surface that, not treat it as clean.
    ///
    /// Uses a uniform **spatial hash** (cell ≈ the median triangle size) so it is ~O(N) on
    /// a clean, roughly-uniform mesh. Adjacent (shared-feature) triangles are *not* skipped
    /// wholesale — the strict-interior test already excludes their shared edge/vertex,
    /// while still catching a pair that shares a vertex *and* also crosses elsewhere.
    pub fn count_self_intersections(&self) -> Option<usize> {
        let n = self.tris.len();
        if n < 2 {
            return Some(0);
        }
        // Per-triangle points must be finite (NaN/inf would corrupt the cell keys).
        let pts: Vec<(Vec3, Vec3, Vec3)> = self.tris.iter().map(|&t| self.tri_points(t)).collect();
        let finite = |v: Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        if !pts.iter().all(|&(a, b, c)| finite(a) && finite(b) && finite(c)) {
            return None;
        }

        // Cell size = median longest edge (typical element size).
        let mut edges: Vec<f64> = pts
            .iter()
            .map(|&(a, b, c)| (b - a).norm().max((c - b).norm()).max((a - c).norm()))
            .collect();
        edges.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let cell = edges[n / 2];
        if !(cell > 0.0 && cell.is_finite()) {
            return None; // degenerate sizing — cannot build a grid; not "clean"
        }

        // Bound the work: cap how many cells one triangle may span (oversized/multi-scale
        // triangle ⇒ bail) and the total exact tests (dense cells ⇒ bail). Both keep this
        // safe on adversarial connector input.
        const SPAN_CAP: i64 = 4096;
        let budget = 64usize.saturating_mul(n);

        let key = |x: f64| (x / cell).floor();
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, &(a, b, c)) in pts.iter().enumerate() {
            let span = |lo: f64, hi: f64| ((hi / cell).floor() - (lo / cell).floor()) + 1.0;
            let dx = span(a.x.min(b.x).min(c.x), a.x.max(b.x).max(c.x));
            let dy = span(a.y.min(b.y).min(c.y), a.y.max(b.y).max(c.y));
            let dz = span(a.z.min(b.z).min(c.z), a.z.max(b.z).max(c.z));
            if dx * dy * dz > SPAN_CAP as f64 {
                return None; // a triangle spans too many cells — multi-scale mesh
            }
            let (xlo, xhi) = (key(a.x.min(b.x).min(c.x)) as i64, key(a.x.max(b.x).max(c.x)) as i64);
            let (ylo, yhi) = (key(a.y.min(b.y).min(c.y)) as i64, key(a.y.max(b.y).max(c.y)) as i64);
            let (zlo, zhi) = (key(a.z.min(b.z).min(c.z)) as i64, key(a.z.max(b.z).max(c.z)) as i64);
            for cx in xlo..=xhi {
                for cy in ylo..=yhi {
                    for cz in zlo..=zhi {
                        grid.entry((cx, cy, cz)).or_default().push(i);
                    }
                }
            }
        }

        // Exact-test candidate pairs (sharing a cell), once each. The `tested` set and the
        // exact tests are both bounded by `budget`.
        let mut tested: HashSet<(usize, usize)> = HashSet::new();
        let mut count = 0;
        for ids in grid.values() {
            for ai in 0..ids.len() {
                for bi in (ai + 1)..ids.len() {
                    let (mut a, mut b) = (ids[ai], ids[bi]);
                    if a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    if !tested.insert((a, b)) {
                        continue; // already tested from another shared cell
                    }
                    if tested.len() > budget {
                        return None; // too many candidate pairs — cannot verify cheaply
                    }
                    if triangles_penetrate(pts[a], pts[b]) {
                        count += 1;
                    }
                }
            }
        }
        Some(count)
    }

    /// Euler characteristic V − E + F, counting only vertices actually used by a
    /// triangle (stray/unreferenced vertices don't change the topology). A
    /// closed sphere-topology mesh gives 2.
    pub fn euler_characteristic(&self) -> i64 {
        let mut used = std::collections::HashSet::new();
        for &t in &self.tris {
            used.insert(t[0]);
            used.insert(t[1]);
            used.insert(t[2]);
        }
        let v = used.len() as i64;
        let e = self.edge_use_counts().len() as i64;
        let f = self.tris.len() as i64;
        v - e + f
    }
}

impl Mesh {
    /// Concatenate `other` onto `self` (triangle indices offset accordingly).
    pub fn append(&mut self, other: &Mesh) {
        let base = self.verts.len() as u32;
        self.verts.extend_from_slice(&other.verts);
        self.normals.extend_from_slice(&other.normals);
        self.tris.extend(
            other
                .tris
                .iter()
                .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
        );
    }

    /// Reverse the winding of every triangle (flips all normals' implied side).
    pub fn flip(&mut self) {
        for t in &mut self.tris {
            t.swap(1, 2);
        }
    }

    /// Make the winding globally consistent by flood-filling orientation across
    /// shared edges (flip a neighbor whenever it traverses the shared edge in the
    /// same direction as the current triangle). Robust regardless of how the
    /// input patches were individually wound — the principled alternative to a
    /// per-curve orientation contract. Propagation crosses only edges shared by
    /// exactly two triangles, so a non-manifold edge (>2 triangles) cannot corrupt
    /// the orientation of unrelated triangles. This does **not** validate the
    /// result: on a non-manifold or non-orientable input the outcome is undefined
    /// — `is_watertight()` (catches >2-triangle edges) and `is_consistently_oriented()`
    /// (catches unresolved conflicts) are the validators, asserted after.
    pub fn orient_consistently(&mut self) {
        // undirected edge → triangle indices sharing it
        let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (ti, t) in self.tris.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let k = if a < b { (a, b) } else { (b, a) };
                edge_tris.entry(k).or_default().push(ti);
            }
        }
        // a triangle "has" directed edge (a,b) if (a,b) appears in its cycle
        let has_directed = |t: &[u32; 3], a: u32, b: u32| {
            (t[0] == a && t[1] == b) || (t[1] == a && t[2] == b) || (t[2] == a && t[0] == b)
        };
        let mut visited = vec![false; self.tris.len()];
        for seed in 0..self.tris.len() {
            if visited[seed] {
                continue;
            }
            visited[seed] = true;
            let mut stack = vec![seed];
            while let Some(ti) = stack.pop() {
                let t = self.tris[ti];
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let k = if a < b { (a, b) } else { (b, a) };
                    // Only cross genuine manifold edges; a non-manifold fan
                    // (>2 triangles) must not propagate orientation.
                    if edge_tris[&k].len() != 2 {
                        continue;
                    }
                    for &nb in &edge_tris[&k] {
                        if nb == ti || visited[nb] {
                            continue;
                        }
                        // consistent ⇔ neighbor traverses this edge the OTHER way
                        // (b,a); if it also has (a,b), it's mis-wound → flip it.
                        if has_directed(&self.tris[nb], a, b) {
                            self.tris[nb].swap(1, 2);
                        }
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
            }
        }
    }

    /// Merge **bit-identical** vertices and remap triangles, dropping any
    /// triangle that collapses. Used to fuse shared boundary rims after `append`:
    /// the adjacent patches are constructed to copy the *exact same* `Vec3` rim
    /// values (not merely close ones), so an exact-coordinate dedup is correct and
    /// — unlike a tolerance/grid-snap weld — can never false-merge distinct
    /// near-degenerate features. Crowded / arrangement stitching uses the explicit
    /// shared-index registry (`TO_SES_STITCHING.md`) and needs no coordinate
    /// matching at all. Keeps the first occurrence's normal: **seam normals are
    /// shading-only after this** (a seam vertex belongs to two patches with
    /// different geometric normals) — the residual/normal gates must use
    /// per-patch normals, not these.
    pub fn welded(&self) -> Mesh {
        let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
        let mut rep: HashMap<(u64, u64, u64), u32> = HashMap::new();
        let mut remap = vec![0u32; self.verts.len()];
        let mut verts = Vec::new();
        let mut normals = Vec::new();
        for (i, &p) in self.verts.iter().enumerate() {
            let idx = *rep.entry(key(p)).or_insert_with(|| {
                verts.push(p);
                normals.push(self.normals.get(i).copied().unwrap_or(p));
                (verts.len() - 1) as u32
            });
            remap[i] = idx;
        }
        let tris = self
            .tris
            .iter()
            .map(|t| {
                [
                    remap[t[0] as usize],
                    remap[t[1] as usize],
                    remap[t[2] as usize],
                ]
            })
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
            .collect();
        Mesh {
            verts,
            normals,
            tris,
        }
    }

    /// Like [`Mesh::welded`] but fuses vertices within Euclidean distance `eps`
    /// (a **tolerance** merge). Use this — not `welded` — when shared seams are
    /// sampled by *different* parameterizations that agree mathematically but
    /// not bit-for-bit. The cleaned SES is exactly that case: a great-circle
    /// seam is sampled as `toric_column_curve` (φ = `dir·cos+tan·sin`) on the
    /// toric side and `arc_on_sphere` (slerp) on the spheric side, and burial
    /// corners are reconstructed independently by `arrange_loops` (direction
    /// space) and the toric φ-trim — coincident points that never share f64
    /// bits.
    ///
    /// `eps` MUST sit **below the minimum genuine feature separation** (the
    /// sample spacing: chart `grid`, the reentrant `n_phi` step, the toric
    /// `n_theta` step) and **above** the largest independent-reconstruction gap
    /// between two samplings of the same corner. Pick it by measurement: the
    /// smallest `eps` that closes the open edges without moving `surface_area`.
    /// Too large false-merges distinct features (collapsing real surface); too
    /// small leaves the seam open. The first vertex of a cluster wins (its
    /// normal is kept — see [`Mesh::welded`] on seam normals).
    ///
    /// Buckets vertices on an `eps`-grid and probes the 27 neighbouring cells,
    /// so a cluster straddling a cell boundary still merges.
    pub fn welded_within(&self, eps: f64) -> Mesh {
        assert!(
            eps > 0.0 && eps.is_finite(),
            "weld tolerance must be positive and finite"
        );
        let inv = 1.0 / eps;
        let cell = |p: Vec3| {
            (
                (p.x * inv).floor() as i64,
                (p.y * inv).floor() as i64,
                (p.z * inv).floor() as i64,
            )
        };
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        let mut remap = vec![0u32; self.verts.len()];
        let mut verts: Vec<Vec3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        for (i, &p) in self.verts.iter().enumerate() {
            let (cx, cy, cz) = cell(p);
            let mut found: Option<u32> = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &r in bucket {
                                if verts[r as usize].distance(p) <= eps {
                                    found = Some(r);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            let idx = found.unwrap_or_else(|| {
                let r = verts.len() as u32;
                verts.push(p);
                normals.push(self.normals.get(i).copied().unwrap_or(p));
                grid.entry((cx, cy, cz)).or_default().push(r);
                r
            });
            remap[i] = idx;
        }
        let tris = self
            .tris
            .iter()
            .map(|t| {
                [
                    remap[t[0] as usize],
                    remap[t[1] as usize],
                    remap[t[2] as usize],
                ]
            })
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
            .collect();
        Mesh {
            verts,
            normals,
            tris,
        }
    }

    /// Remove degenerate (near-zero-area) sliver triangles that the tolerance weld
    /// leaves at **singular vertices** — points where ≥3 analytic patches terminate
    /// and each samples the shared corner a hair (just over `eps`) apart, so the
    /// weld can neither fully fuse the cluster nor be loosened without false-merging
    /// real features. Those slivers carry no area but corrupt the topology (a
    /// zero-area triangle's edges show up as boundary/non-manifold).
    ///
    /// **Guarded**: a sliver is dropped only when *none* of its three edges is
    /// currently shared by exactly two triangles — i.e. all three edges are already
    /// boundary (degree 1) or non-manifold (degree ≥3). Dropping such a triangle
    /// therefore strictly *reduces* the defect count and can never turn a clean
    /// degree-2 edge into a new boundary (it cannot open a hole). Iterated to a
    /// fixpoint. A triangle is "degenerate" when its minimum altitude
    /// (`2·area / longest edge`) falls below `eps` — i.e. it is thinner than the
    /// weld can resolve. Returns the number removed. Vertices are left in place
    /// (orphans are harmless); per-vertex normals stay valid.
    pub fn remove_degenerate_triangles_guarded(&mut self, eps: f64) -> usize {
        let key = |a: u32, b: u32| (a.min(b), a.max(b));
        let mut counts: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &self.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *counts.entry(key(a, b)).or_default() += 1;
            }
        }
        let degenerate = |t: &[u32; 3]| -> bool {
            let p = self.verts[t[0] as usize];
            let q = self.verts[t[1] as usize];
            let r = self.verts[t[2] as usize];
            let twice_area = (q - p).cross(r - p).norm();
            let longest = (q - p).norm().max((r - q).norm()).max((p - r).norm());
            longest > 0.0 && twice_area / longest < eps
        };
        let candidates: Vec<usize> = (0..self.tris.len())
            .filter(|&i| degenerate(&self.tris[i]))
            .collect();
        let mut removed = vec![false; self.tris.len()];
        let mut total = 0usize;
        loop {
            let mut progressed = false;
            for &i in &candidates {
                if removed[i] {
                    continue;
                }
                let t = self.tris[i];
                let edges = [key(t[0], t[1]), key(t[1], t[2]), key(t[2], t[0])];
                // Safe iff no edge is currently degree-2 (removal would open it).
                if edges.iter().all(|e| counts[e] != 2) {
                    removed[i] = true;
                    for e in edges {
                        *counts.get_mut(&e).unwrap() -= 1;
                    }
                    total += 1;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        if total > 0 {
            let mut keep = Vec::with_capacity(self.tris.len() - total);
            for (i, &t) in self.tris.iter().enumerate() {
                if !removed[i] {
                    keep.push(t);
                }
            }
            self.tris = keep;
        }
        total
    }

    /// Area-weighted per-vertex normals (smooth shading). Each triangle adds its
    /// area-scaled face normal to its three vertices; the sum is normalized. Falls
    /// back to a unit +z for any isolated/degenerate vertex.
    pub fn vertex_normals(&self) -> Vec<Vec3> {
        let mut n = vec![Vec3::new(0.0, 0.0, 0.0); self.verts.len()];
        for t in &self.tris {
            let (a, b, c) = self.tri_points(*t);
            let fn_ = (b - a).cross(c - a); // length = 2·area, direction = face normal
            for &v in t {
                n[v as usize] = n[v as usize] + fn_;
            }
        }
        n.iter()
            .map(|&v| v.normalized().unwrap_or(Vec3::new(0.0, 0.0, 1.0)))
            .collect()
    }

    /// Write the mesh as Wavefront **OBJ** (text, universally viewable). Includes
    /// smooth per-vertex normals (`vn`) so viewers shade it nicely; faces are
    /// 1-indexed `v//vn`. `name` becomes the object name.
    pub fn write_obj(&self, mut w: impl std::io::Write, name: &str) -> std::io::Result<()> {
        let normals = if self.normals.len() == self.verts.len() {
            self.normals.clone()
        } else {
            self.vertex_normals()
        };
        writeln!(
            w,
            "# proteon SES mesh: {} verts, {} tris",
            self.verts.len(),
            self.tris.len()
        )?;
        writeln!(w, "o {name}")?;
        for v in &self.verts {
            writeln!(w, "v {} {} {}", v.x, v.y, v.z)?;
        }
        for n in &normals {
            writeln!(w, "vn {} {} {}", n.x, n.y, n.z)?;
        }
        for t in &self.tris {
            let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
            writeln!(w, "f {a}//{a} {b}//{b} {c}//{c}")?;
        }
        Ok(())
    }

    /// Write the mesh as **binary little-endian PLY** with per-vertex normals —
    /// compact for large meshes and read by MeshLab/Blender/PyMOL. Vertices and
    /// normals are `float32`; faces are `uchar 3` + `int32×3`.
    pub fn write_ply(&self, mut w: impl std::io::Write) -> std::io::Result<()> {
        let normals = if self.normals.len() == self.verts.len() {
            self.normals.clone()
        } else {
            self.vertex_normals()
        };
        for line in [
            "ply".to_string(),
            "format binary_little_endian 1.0".to_string(),
            "comment proteon SES mesh".to_string(),
            format!("element vertex {}", self.verts.len()),
            "property float x".to_string(),
            "property float y".to_string(),
            "property float z".to_string(),
            "property float nx".to_string(),
            "property float ny".to_string(),
            "property float nz".to_string(),
            format!("element face {}", self.tris.len()),
            "property list uchar int vertex_indices".to_string(),
            "end_header".to_string(),
        ] {
            writeln!(w, "{line}")?;
        }
        for (v, n) in self.verts.iter().zip(&normals) {
            for c in [v.x, v.y, v.z, n.x, n.y, n.z] {
                w.write_all(&(c as f32).to_le_bytes())?;
            }
        }
        for t in &self.tris {
            w.write_all(&[3u8])?;
            for &i in t {
                w.write_all(&(i as i32).to_le_bytes())?;
            }
        }
        Ok(())
    }
}

/// Triangulated sphere by `subdivisions` levels of icosahedron refinement,
/// projected to the sphere of `radius` at `center`. `subdivisions=0` is the
/// 12-vertex/20-triangle icosahedron; each level quadruples the triangles.
/// Outward-oriented, with per-vertex outward normals.
pub fn icosphere(center: Vec3, radius: f64, subdivisions: u32) -> Mesh {
    // Icosahedron: 12 vertices (golden-ratio rectangles), 20 faces.
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut verts: Vec<Vec3> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalized().unwrap())
    .collect();

    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Loop subdivision on the unit sphere: split each triangle into 4, every new
    // midpoint projected back to the sphere; a midpoint cache keeps the mesh
    // watertight (shared edges share a vertex).
    let mut midpoint: HashMap<(u32, u32), u32> = HashMap::new();
    for _ in 0..subdivisions {
        let mut new_tris = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = get_midpoint(a, b, &mut verts, &mut midpoint);
            let bc = get_midpoint(b, c, &mut verts, &mut midpoint);
            let ca = get_midpoint(c, a, &mut verts, &mut midpoint);
            new_tris.push([a, ab, ca]);
            new_tris.push([b, bc, ab]);
            new_tris.push([c, ca, bc]);
            new_tris.push([ab, bc, ca]);
        }
        tris = new_tris;
        midpoint.clear();
    }

    // Unit sphere → world: normals are the unit directions, verts scaled+shifted.
    let normals = verts.clone();
    let verts = verts.iter().map(|&d| center + d * radius).collect();
    Mesh {
        verts,
        normals,
        tris,
    }
}

fn get_midpoint(
    a: u32,
    b: u32,
    verts: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) {
        return i;
    }
    let mid = (verts[a as usize] + verts[b as usize])
        .normalized()
        .unwrap();
    let i = verts.len() as u32;
    verts.push(mid);
    cache.insert(key, i);
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn icosahedron_is_closed_manifold() {
        let m = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 0);
        assert_eq!(m.num_vertices(), 12);
        assert_eq!(m.num_triangles(), 20);
        assert!(m.is_watertight());
        assert!(m.is_consistently_oriented());
        assert_eq!(m.euler_characteristic(), 2); // sphere topology
        assert!(m.signed_volume() > 0.0, "must be outward-oriented");

        // Euler χ counts only used vertices: a stray vertex must not change it.
        let mut m2 = m.clone();
        m2.verts.push(Vec3::new(9.0, 9.0, 9.0));
        assert_eq!(m2.euler_characteristic(), 2);
    }

    #[test]
    fn connected_components_and_duplicate_faces() {
        let one = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 1);
        assert_eq!(one.num_connected_components(), 1, "a single sphere is one body");
        assert_eq!(one.num_duplicate_faces(), 0);

        // Two disjoint spheres → two components (append offsets indices, so no shared
        // edges between them).
        let mut two = one.clone();
        let mut other = icosphere(Vec3::new(5.0, 0.0, 0.0), 1.0, 1);
        // Detach `other`'s vertices/indices via append.
        two.append(&other);
        assert_eq!(two.num_connected_components(), 2, "two disjoint spheres");

        // Duplicate a face → one duplicate.
        other.tris.push(other.tris[0]);
        assert_eq!(other.num_duplicate_faces(), 1);
    }

    #[test]
    fn self_intersection_detection() {
        // A clean icosphere has no self-intersections (no false positives from the now
        // un-skipped adjacent triangles — the strict-interior test excludes them).
        let clean = icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, 3);
        assert_eq!(clean.count_self_intersections(), Some(0), "a clean sphere is clean");

        let cross_mesh = |s: f64| Mesh {
            verts: vec![
                Vec3::new(-s, 0.0, 0.0),
                Vec3::new(s, 0.0, 0.0),
                Vec3::new(0.0, s, 0.5 * s),
                Vec3::new(0.0, -s, 0.0),
                Vec3::new(0.0, s, 0.0),
                Vec3::new(0.5 * s, 0.0, s),
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        // Crossing pair detected at small AND large scale (scale-invariant det cutoff).
        assert_eq!(cross_mesh(1.0).count_self_intersections(), Some(1));
        assert_eq!(cross_mesh(1e-4).count_self_intersections(), Some(1), "small scale");
        assert_eq!(cross_mesh(1e5).count_self_intersections(), Some(1), "large scale");

        // Two coplanar, non-overlapping triangles → no penetration.
        let apart = Mesh {
            verts: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(5.0, 5.0, 5.0),
                Vec3::new(6.0, 5.0, 5.0),
                Vec3::new(5.0, 6.0, 5.0),
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        assert_eq!(apart.count_self_intersections(), Some(0));

        // Shared-vertex pair that ALSO crosses elsewhere: the old any-shared-vertex skip
        // would miss it; now it must be caught (review). Two triangles share vertex 0;
        // the second folds back through the first's interior.
        let shared_and_crossing = Mesh {
            verts: vec![
                Vec3::new(0.0, 0.0, 0.0), // shared vertex
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0), // tri A in z=0
                Vec3::new(1.0, 0.5, -1.0), // tri B straddles z=0; its far edge punches
                Vec3::new(1.0, 0.5, 1.0),  //   through A's interior at (1, 0.5, 0)
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [0, 3, 4]],
        };
        assert_eq!(
            shared_and_crossing.count_self_intersections(),
            Some(1),
            "a shared-vertex pair that also crosses must be detected"
        );
    }

    #[test]
    fn per_component_orient_outward_fixes_mixed_orientation() {
        // One outward sphere + one INWARD sphere. Aggregate signed_volume can mask the
        // inward one; per-component orientation must flip only the inward body and leave
        // the outward one alone.
        let outward = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 1);
        let mut inward = icosphere(Vec3::new(5.0, 0.0, 0.0), 1.0, 1);
        inward.flip(); // now inside-out
        let mut mixed = outward.clone();
        mixed.append(&inward);

        let vols: Vec<f64> = mixed.component_volumes_areas().iter().map(|&(v, _)| v).collect();
        assert_eq!(vols.len(), 2);
        assert!(
            vols.iter().any(|&v| v > 0.0) && vols.iter().any(|&v| v < 0.0),
            "mixed orientation: {vols:?}"
        );

        assert!(mixed.orient_outward(), "the inward component must be flipped");
        let fixed: Vec<f64> = mixed.component_volumes_areas().iter().map(|&(v, _)| v).collect();
        assert!(fixed.iter().all(|&v| v > 0.0), "both components outward now: {fixed:?}");
        // The already-outward component is untouched (same volume).
        assert!((fixed.iter().cloned().fold(f64::INFINITY, f64::min)
            - vols.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b.abs())))
            .abs()
            < 1e-9);
    }

    #[test]
    fn icosphere_area_and_volume_converge() {
        // The SES of an isolated atom IS its sphere → mesh area must converge to
        // 4πr² and signed volume to (4/3)πr³ (the ses_area / ses_mesh gate).
        let r = 2.0;
        let center = Vec3::new(1.0, -2.0, 3.0);
        let coarse = icosphere(center, r, 2);
        let fine = icosphere(center, r, 4);
        let exact_area = 4.0 * PI * r * r;
        let exact_vol = 4.0 / 3.0 * PI * r.powi(3);

        // Inscribed triangulation under-estimates; finer is closer to exact.
        let ea_c = (coarse.surface_area() - exact_area).abs();
        let ea_f = (fine.surface_area() - exact_area).abs();
        assert!(ea_f < ea_c, "area should converge with subdivision");
        assert!(ea_f / exact_area < 0.01, "fine area within 1% of 4πr²");

        let ev_f = (fine.signed_volume() - exact_vol).abs();
        assert!(ev_f / exact_vol < 0.02, "fine volume within 2% of (4/3)πr³");

        // Stays a watertight, consistently-oriented sphere manifold.
        assert!(fine.is_watertight());
        assert!(fine.is_consistently_oriented());
        assert_eq!(fine.euler_characteristic(), 2);
    }

    /// Two triangles that share an edge sampled with a sub-`eps` mismatch (the
    /// cleaned-SES case: two parameterizations of the same seam) weld into a
    /// closed manifold under `welded_within` but stay open under exact `welded`.
    #[test]
    fn welded_within_fuses_near_coincident_seams() {
        // A unit square split into two triangles, but the shared diagonal is
        // sampled twice with a 1e-7 jitter — bit-different, geometrically equal.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 1.0, 0.0);
        // Patch 1 uses (a, c); patch 2 uses a jittered copy of (a, c).
        let jit = Vec3::new(1e-7, -1e-7, 1e-7);
        let m = Mesh {
            verts: vec![a, b, c, a + jit, c + jit, d],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        // Exact weld leaves the diagonal split (4 + 2 = 6 verts; open).
        let exact = m.welded();
        assert_eq!(exact.num_vertices(), 6);
        assert!(!exact.is_watertight());

        // Tolerance weld fuses the jittered pair → 4 verts, the diagonal shared.
        let tol = m.welded_within(1e-4);
        assert_eq!(tol.num_vertices(), 4, "jittered diagonal endpoints fused");
        // (Two triangles meeting on a shared edge: that edge is now manifold.)
        let mut shared = 0;
        let mut edges: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for t in &tol.tris {
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
        for &c in edges.values() {
            if c == 2 {
                shared += 1;
            }
        }
        assert_eq!(
            shared, 1,
            "exactly the diagonal is now a shared (manifold) edge"
        );

        // Too-tight eps does NOT fuse (1e-9 < the 1e-7 jitter).
        assert_eq!(m.welded_within(1e-9).num_vertices(), 6);
    }

    /// `welded_within` must not collapse genuinely distinct features: vertices
    /// farther apart than `eps` stay separate.
    #[test]
    fn welded_within_preserves_distinct_features() {
        let m = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 2);
        // Min vertex spacing on a level-2 icosphere is ~0.2; eps well below it.
        let w = m.welded_within(1e-3);
        assert_eq!(
            w.num_vertices(),
            m.num_vertices(),
            "no distinct vertex merged"
        );
        assert!(w.is_watertight());
        assert_eq!(w.euler_characteristic(), 2);
    }

    #[test]
    fn guarded_cleanup_removes_a_sliver_at_a_nonmanifold_edge() {
        // Edge A–B shared by two real triangles plus a degenerate sliver A–B–E
        // (E almost on segment AB) → AB is degree 3 (non-manifold), the signature
        // of the 1a7j singular-vertex residual. The guarded cleanup must drop only
        // the sliver, leaving AB a clean degree-2 edge.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.5, 1.0, 0.0);
        let d = Vec3::new(0.5, -1.0, 0.0);
        let e = Vec3::new(0.5, 1e-9, 0.0); // ~on AB → zero-area sliver
        let mut m = Mesh {
            verts: vec![a, b, c, d, e],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]], // last is the sliver
        };
        assert_eq!(m.edge_use_counts()[&(0, 1)], 3, "AB non-manifold before");
        let removed = m.remove_degenerate_triangles_guarded(1e-4);
        assert_eq!(removed, 1, "exactly the sliver is dropped");
        assert_eq!(m.tris.len(), 2);
        assert_eq!(m.edge_use_counts()[&(0, 1)], 2, "AB manifold after");
    }

    #[test]
    fn guarded_cleanup_will_not_open_a_hole() {
        // Edge A–B shared by ONE real triangle and one degenerate sliver → AB is
        // degree 2. Dropping the sliver would make AB a boundary (open a hole), so
        // the guard must refuse even though the sliver is degenerate.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.5, 1.0, 0.0);
        let e = Vec3::new(0.5, 1e-9, 0.0);
        let mut m = Mesh {
            verts: vec![a, b, c, e],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [0, 1, 3]], // real + sliver share AB (degree 2)
        };
        let removed = m.remove_degenerate_triangles_guarded(1e-4);
        assert_eq!(
            removed, 0,
            "guard refuses — removing the sliver would open AB"
        );
        assert_eq!(m.tris.len(), 2);
    }

    #[test]
    fn obj_and_ply_export_have_the_right_shape() {
        let m = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 1);
        let (nv, nt) = (m.num_vertices(), m.num_triangles());

        // OBJ: one `v ` and one `vn ` per vertex, one `f ` per triangle.
        let mut obj = Vec::new();
        m.write_obj(&mut obj, "sphere").unwrap();
        let obj = String::from_utf8(obj).unwrap();
        assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), nv);
        assert_eq!(obj.lines().filter(|l| l.starts_with("vn ")).count(), nv);
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), nt);

        // PLY: header declares the right counts, body is the expected byte length
        // (nv × 6 f32 + nt × (1 byte + 3 i32)).
        let mut ply = Vec::new();
        m.write_ply(&mut ply).unwrap();
        let hdr_end = ply.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = std::str::from_utf8(&ply[..hdr_end]).unwrap();
        assert!(header.contains(&format!("element vertex {nv}")));
        assert!(header.contains(&format!("element face {nt}")));
        let body = ply.len() - hdr_end;
        assert_eq!(body, nv * 6 * 4 + nt * (1 + 3 * 4));
    }
}
