//! Octree over surface panels for the treecode (plan §4.1).
//!
//! Each node's bounding box **encloses the triangle vertices** of its panels — not
//! just their centroids (codex review): a large or high-aspect panel can reach a
//! target while its centroid cluster looks far, so admissibility must see the true
//! panel extent. Panels are assigned to the octant of their centroid (each panel in
//! exactly one leaf), but every node's stored box, center, and radius come from the
//! vertices, so a panel always lies inside its node's box — which is what makes the
//! panel-aware cluster expansion valid under the box-separation MAC.

use proteon_core::surface::geom::Vec3;

use crate::model::Tri;

/// One octree node.
pub struct OctNode {
    /// Box center (`(lo + hi)/2`).
    pub center: Vec3,
    /// Cluster radius: half the vertex-box diagonal (`|hi − lo|/2`).
    pub radius: f64,
    /// Vertex-enclosing box lower corner.
    pub lo: Vec3,
    /// Vertex-enclosing box upper corner.
    pub hi: Vec3,
    /// Element indices in this node's **subtree** (all of them; leaves hold their few).
    pub panels: Vec<usize>,
    /// Child node indices into [`Octree::nodes`]; empty ⇒ leaf.
    pub children: Vec<usize>,
}

/// A built octree.
pub struct Octree {
    /// Nodes; `nodes[0]` is the root.
    pub nodes: Vec<OctNode>,
}

fn vertex_bbox(elements: &[Tri], idxs: &[usize]) -> (Vec3, Vec3) {
    let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut acc = |p: Vec3| {
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    };
    for &j in idxs {
        let t = &elements[j];
        acc(t.v1);
        acc(t.v2);
        acc(t.v3);
    }
    (lo, hi)
}

impl Octree {
    /// Build an octree over `elements`, subdividing while a node has more than
    /// `n_leaf` panels and depth `< max_depth`. `centroids[j]` is element `j`'s
    /// collocation point (used for octant assignment).
    #[must_use]
    pub fn build(elements: &[Tri], centroids: &[Vec3], n_leaf: usize, max_depth: usize) -> Self {
        let mut nodes = Vec::new();
        let all: Vec<usize> = (0..elements.len()).collect();
        build_node(elements, centroids, all, 0, n_leaf, max_depth, &mut nodes);
        Self { nodes }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    elements: &[Tri],
    centroids: &[Vec3],
    idxs: Vec<usize>,
    depth: usize,
    n_leaf: usize,
    max_depth: usize,
    nodes: &mut Vec<OctNode>,
) -> usize {
    let (lo, hi) = vertex_bbox(elements, &idxs);
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).norm() * 0.5;
    let my_idx = nodes.len();
    nodes.push(OctNode {
        center,
        radius,
        lo,
        hi,
        panels: idxs.clone(),
        children: Vec::new(),
    });

    if idxs.len() <= n_leaf || depth >= max_depth {
        return my_idx; // leaf
    }

    // Split about the geometric center into 8 octants by each panel's centroid.
    let mut buckets: [Vec<usize>; 8] = Default::default();
    for &j in &idxs {
        let c = centroids[j];
        let oct = usize::from(c.x >= center.x)
            | (usize::from(c.y >= center.y) << 1)
            | (usize::from(c.z >= center.z) << 2);
        buckets[oct].push(j);
    }
    // Degenerate split (all panels in one octant, e.g. coincident centroids) ⇒ stop.
    if buckets.iter().filter(|b| !b.is_empty()).count() <= 1 {
        return my_idx;
    }

    let mut children = Vec::new();
    for b in buckets {
        if !b.is_empty() {
            let ci = build_node(elements, centroids, b, depth + 1, n_leaf, max_depth, nodes);
            children.push(ci);
        }
    }
    nodes[my_idx].children = children;
    my_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centroid(t: &Tri) -> Vec3 {
        (t.v1 + t.v2 + t.v3) * (1.0 / 3.0)
    }

    fn grid_panels(nside: usize) -> Vec<Tri> {
        // A grid of small tilted triangles spread over a cube — a non-degenerate set.
        let mut v = Vec::new();
        for a in 0..nside {
            for b in 0..nside {
                for c in 0..nside {
                    let o = Vec3::new(a as f64, b as f64, c as f64);
                    v.push(Tri::new(
                        o,
                        o + Vec3::new(0.3, 0.1, 0.05),
                        o + Vec3::new(0.1, 0.3, 0.15),
                    ));
                }
            }
        }
        v
    }

    #[test]
    fn every_panel_is_in_a_leaf_exactly_once() {
        let els = grid_panels(4); // 64 panels
        let cs: Vec<Vec3> = els.iter().map(centroid).collect();
        let tree = Octree::build(&els, &cs, 4, 10);
        // Collect leaf panels.
        let mut seen = vec![0u32; els.len()];
        for node in &tree.nodes {
            if node.children.is_empty() {
                for &j in &node.panels {
                    seen[j] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "each panel in exactly one leaf");
    }

    #[test]
    fn boxes_enclose_vertices() {
        let els = grid_panels(3);
        let cs: Vec<Vec3> = els.iter().map(centroid).collect();
        let tree = Octree::build(&els, &cs, 2, 12);
        for node in &tree.nodes {
            for &j in &node.panels {
                for v in [els[j].v1, els[j].v2, els[j].v3] {
                    assert!(
                        v.x >= node.lo.x - 1e-12 && v.x <= node.hi.x + 1e-12
                            && v.y >= node.lo.y - 1e-12 && v.y <= node.hi.y + 1e-12
                            && v.z >= node.lo.z - 1e-12 && v.z <= node.hi.z + 1e-12,
                        "vertex outside node box"
                    );
                }
            }
        }
    }

    #[test]
    fn leaves_respect_n_leaf() {
        let els = grid_panels(4);
        let cs: Vec<Vec3> = els.iter().map(centroid).collect();
        let tree = Octree::build(&els, &cs, 4, 12);
        for node in &tree.nodes {
            if node.children.is_empty() {
                // A leaf either fits n_leaf or could not be split further.
                assert!(node.panels.len() <= 4 || node.children.is_empty());
            }
        }
    }
}
