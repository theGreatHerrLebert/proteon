//! Constrained Delaunay triangulation of a planar region, over `spade`.
//!
//! Used to mesh the *interior* of an analytic SES contact cap once its exposed
//! region has been projected to a planar chart (`TO_SES_EXACT_COMPLETION.md`):
//! the boundary loops (outer + buried-cap holes) are sampled once into shared
//! registry vertices and passed as **constraints**, so the triangulation keeps
//! the exact boundary (no Steiner points on constraint edges) and the caller can
//! lift the result back onto the atom sphere by index. Triangles outside the
//! region (inside a hole, or outside the outer loop) are dropped by an
//! `in_domain` centroid test.

use anyhow::{bail, ensure, Result};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};
use std::collections::HashMap;

/// Triangulate the planar PSLG given by `points` (vertices; boundary samples plus
/// any interior Steiner samples) and `loops` (closed boundary loops as index
/// lists into `points` — outer first, then holes). Returns the triangles, as
/// index triples into `points`, whose centroid satisfies `in_domain`.
///
/// Boundary loops become constraint edges. The contract this function *enforces*
/// (errors otherwise) — because watertight stitching depends on it — is that
/// **every boundary vertex round-trips by index and every boundary edge survives
/// intact**: no input point is merged, no constraint edge is split by a vertex
/// lying on it, and no two loop edges cross. So a returned triangle's boundary
/// indices are exactly the shared registry vertices the neighbouring toric/
/// spheric patches use. (codex-review: spade silently merges coincident points
/// and may split constraints — both would crack the assembled mesh.)
///
/// The centroid `in_domain` test is sound here precisely because the PSLG is
/// validated to be simple and non-degenerate before classification.
pub fn constrained_triangulate(
    points: &[[f64; 2]],
    loops: &[Vec<usize>],
    in_domain: impl Fn([f64; 2]) -> bool,
) -> Result<Vec<[usize; 3]>> {
    // --- preflight: a well-formed, in-range, finite PSLG ---
    for p in points {
        ensure!(p[0].is_finite() && p[1].is_finite(), "non-finite point");
    }
    for lp in loops {
        ensure!(lp.len() >= 3, "boundary loop has < 3 vertices");
        for &v in lp {
            ensure!(v < points.len(), "loop index {v} out of range");
        }
        // no immediately-repeated vertex (zero-length constraint)
        for w in 0..lp.len() {
            ensure!(
                lp[w] != lp[(w + 1) % lp.len()],
                "loop has a repeated consecutive vertex"
            );
        }
    }

    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    let mut handles = Vec::with_capacity(points.len());
    let mut mine_of: HashMap<usize, usize> = HashMap::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let h = cdt.insert(Point2::new(p[0], p[1]))?;
        handles.push(h);
        mine_of.insert(h.index(), i);
    }
    // Spade overwrites a coincident insert, so a merge shows up as a vertex
    // shortfall — that would alias two registry indices to one vertex.
    ensure!(
        cdt.num_vertices() == points.len(),
        "duplicate/near-coincident points merged ({} vertices for {} points) — \
         would alias boundary vertices",
        cdt.num_vertices(),
        points.len()
    );

    for lp in loops {
        let n = lp.len();
        for w in 0..n {
            let (a, b) = (handles[lp[w]], handles[lp[(w + 1) % n]]);
            // Panic-free: refuse rather than let spade panic on a crossing.
            ensure!(
                cdt.can_add_constraint(a, b),
                "boundary edge {}->{} crosses an existing constraint",
                lp[w],
                lp[(w + 1) % n]
            );
            cdt.add_constraint(a, b);
            // A vertex on the segment would split it: the registry edge is then
            // not preserved as a single edge → reject.
            ensure!(
                cdt.exists_constraint(a, b),
                "boundary edge {}->{} was split (a vertex lies on it)",
                lp[w],
                lp[(w + 1) % n]
            );
        }
    }

    let mut tris = Vec::new();
    for face in cdt.inner_faces() {
        let vs = face.vertices();
        let (a, b, c) = (vs[0].position(), vs[1].position(), vs[2].position());
        let centroid = [(a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0];
        if !in_domain(centroid) {
            continue;
        }
        match (
            mine_of.get(&vs[0].fix().index()),
            mine_of.get(&vs[1].fix().index()),
            mine_of.get(&vs[2].fix().index()),
        ) {
            (Some(&i), Some(&j), Some(&k)) => tris.push([i, j, k]),
            // No Steiner points exist (asserted above), so a miss is a bug.
            _ => bail!("in-domain triangle references an unknown (Steiner) vertex"),
        }
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_area(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
        0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])).abs()
    }

    /// Annulus: a 4×4 outer square with a 2×2 square hole. The constrained
    /// triangulation must keep the hole empty and the filled (domain) triangles
    /// must sum to the annulus area 16 − 4 = 12 — exactly, since the region is
    /// triangulable from the 8 corners alone (no Steiner points needed).
    #[test]
    fn square_with_square_hole_is_an_annulus() {
        let points = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0], // outer
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0], // hole
        ];
        let loops = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let in_hole = |c: [f64; 2]| c[0] > 1.0 && c[0] < 3.0 && c[1] > 1.0 && c[1] < 3.0;
        let tris = constrained_triangulate(&points, &loops, |c| !in_hole(c)).unwrap();

        assert!(!tris.is_empty());
        let area: f64 = tris
            .iter()
            .map(|t| tri_area(points[t[0]], points[t[1]], points[t[2]]))
            .sum();
        assert!((area - 12.0).abs() < 1e-9, "annulus area {area} != 12");
        // No filled triangle sits inside the hole.
        for t in &tris {
            let c = [
                (points[t[0]][0] + points[t[1]][0] + points[t[2]][0]) / 3.0,
                (points[t[0]][1] + points[t[1]][1] + points[t[2]][1]) / 3.0,
            ];
            assert!(!in_hole(c), "a triangle leaked into the hole");
        }
    }

    /// Coincident points would alias two registry indices to one spade vertex
    /// (silent crack); the merge guard must catch it.
    #[test]
    fn coincident_points_are_rejected() {
        let points = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]];
        let loops = vec![vec![0, 1, 2, 3]];
        assert!(constrained_triangulate(&points, &loops, |_| true).is_err());
    }

    /// A vertex lying exactly on a boundary edge would split that edge, so the
    /// registry edge is no longer a single shared edge; reject it.
    #[test]
    fn vertex_on_boundary_edge_is_rejected() {
        // point 4 sits on the bottom edge 0->1.
        let points = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 0.0]];
        let loops = vec![vec![0, 1, 2, 3]];
        assert!(constrained_triangulate(&points, &loops, |_| true).is_err());
    }
}
