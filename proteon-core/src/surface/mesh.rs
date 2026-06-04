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
use std::collections::HashMap;

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
    /// per-curve orientation contract. Assumes a manifold (each edge ≤ 2
    /// triangles); leaves orientation arbitrary-but-consistent (call `flip` after
    /// if `signed_volume` should be positive).
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

    /// Merge vertices within `eps` (a grid-snap dedup) and remap triangles,
    /// dropping any triangle that collapses. The pragmatic glue for stitching
    /// **non-degenerate** patches whose only coincident vertices are the intended
    /// shared boundary rims; the explicit shared-index registry
    /// (`TO_SES_STITCHING.md`) replaces this for crowded / arrangement cases where
    /// coordinate welding could merge distinct near-degenerate features. Keeps the
    /// first occurrence's normal (seam normals are an accepted artifact — mesh
    /// invariants use positions + winding, not normals).
    pub fn welded(&self, eps: f64) -> Mesh {
        let key = |p: Vec3| {
            (
                (p.x / eps).round() as i64,
                (p.y / eps).round() as i64,
                (p.z / eps).round() as i64,
            )
        };
        let mut rep: HashMap<(i64, i64, i64), u32> = HashMap::new();
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
}
