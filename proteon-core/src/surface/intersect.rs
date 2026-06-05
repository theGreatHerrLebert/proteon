//! Mesh self-intersection detection — the gate that distinguishes a *valid
//! embedding* from a merely *combinatorially watertight* mesh (codex-review: the
//! general-N assembler can produce a closed-but-self-intersecting SES where the
//! probe self-intersects, and watertightness alone does not catch it).
//!
//! Broad phase: a uniform spatial hash over triangle AABBs. Narrow phase:
//! segment–triangle (Möller–Trumbore) for all six edges of a pair, skipping pairs
//! that share a vertex (adjacent triangles meet legitimately).

use super::geom::Vec3;
use super::mesh::Mesh;
use std::collections::HashMap;

/// Does segment `p→q` cross triangle `(a,b,c)` strictly in its interior range?
fn seg_tri(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    const EPS: f64 = 1e-9;
    let dir = q - p;
    let e1 = b - a;
    let e2 = c - a;
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < EPS {
        return false; // parallel
    }
    let inv = 1.0 / det;
    let s = p - a;
    let u = s.dot(h) * inv;
    if !(EPS..=1.0 - EPS).contains(&u) {
        return false;
    }
    let qv = s.cross(e1);
    let v = dir.dot(qv) * inv;
    if v < EPS || u + v > 1.0 - EPS {
        return false;
    }
    let t = e2.dot(qv) * inv;
    (EPS..=1.0 - EPS).contains(&t) // interior of the segment
}

fn tri_tri(a: [Vec3; 3], b: [Vec3; 3]) -> bool {
    let edges = |t: [Vec3; 3]| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
    edges(a)
        .into_iter()
        .any(|(p, q)| seg_tri(p, q, b[0], b[1], b[2]))
        || edges(b)
            .into_iter()
            .any(|(p, q)| seg_tri(p, q, a[0], a[1], a[2]))
}

/// Count triangle pairs that intersect (excluding vertex-adjacent pairs). `0` ⇒ a
/// valid (non-self-intersecting) embedding. `cell` is the spatial-hash grid size
/// (≈ a few triangle edge lengths). Stops early at `cap` if `cap > 0`.
pub fn self_intersections(mesh: &Mesh, cell: f64, cap: usize) -> usize {
    let inv = 1.0 / cell;
    let key = |v: Vec3| {
        (
            (v.x * inv).floor() as i64,
            (v.y * inv).floor() as i64,
            (v.z * inv).floor() as i64,
        )
    };
    // Bucket each triangle into every cell its AABB touches.
    let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    let tri_v = |t: [u32; 3]| {
        [
            mesh.verts[t[0] as usize],
            mesh.verts[t[1] as usize],
            mesh.verts[t[2] as usize],
        ]
    };
    for (ti, &t) in mesh.tris.iter().enumerate() {
        let v = tri_v(t);
        let lo = key(Vec3::new(
            v[0].x.min(v[1].x).min(v[2].x),
            v[0].y.min(v[1].y).min(v[2].y),
            v[0].z.min(v[1].z).min(v[2].z),
        ));
        let hi = key(Vec3::new(
            v[0].x.max(v[1].x).max(v[2].x),
            v[0].y.max(v[1].y).max(v[2].y),
            v[0].z.max(v[1].z).max(v[2].z),
        ));
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                for cz in lo.2..=hi.2 {
                    buckets.entry((cx, cy, cz)).or_default().push(ti as u32);
                }
            }
        }
    }
    let shares_vertex = |a: [u32; 3], b: [u32; 3]| a.iter().any(|x| b.contains(x));
    let mut hits = 0usize;
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for ids in buckets.values() {
        for (ii, &ai) in ids.iter().enumerate() {
            for &bi in &ids[ii + 1..] {
                let (lo, hi) = (ai.min(bi), ai.max(bi));
                if !seen.insert((lo, hi)) {
                    continue; // pair already tested (shared across cells)
                }
                let (ta, tb) = (mesh.tris[ai as usize], mesh.tris[bi as usize]);
                if shares_vertex(ta, tb) {
                    continue;
                }
                if tri_tri(tri_v(ta), tri_v(tb)) {
                    hits += 1;
                    if cap > 0 && hits >= cap {
                        return hits;
                    }
                }
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::super::mesh::{icosphere, Mesh};
    use super::*;

    #[test]
    fn an_icosphere_does_not_self_intersect() {
        let m = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.5, 3);
        assert_eq!(self_intersections(&m, 0.5, 0), 0);
    }

    #[test]
    fn two_crossing_triangles_are_detected() {
        // A "+" of two triangles whose planes cross through each other's interior.
        let m = Mesh {
            verts: vec![
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 0.5),
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        assert!(self_intersections(&m, 1.0, 0) >= 1);
    }
}
