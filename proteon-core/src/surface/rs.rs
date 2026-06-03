//! L1 reduced surface (RS).
//!
//! The reduced surface is the combinatorial skeleton the SES is built on
//! (Sanner/Connolly): an **RS face** is an atom triple with a probe sphere
//! resting on all three and intersecting no other atom; an **RS edge** is an
//! atom pair the probe can roll between; an **RS vertex** is a surface atom.
//!
//! This first implementation computes the RS by **enumerate-and-validate**:
//! for every candidate atom triple, place the probe (two positions via
//! `intersect_three_spheres` on probe-inflated atoms) and keep each position
//! that no other atom overlaps. It produces the same RS *face set* BALL's
//! rolling-probe algorithm does — which is exactly what `ball-py
//! reduced_surface_stats` exposes for gating (order-independent atom triples +
//! probe centers). BALL rolls for O(n·k) efficiency; this is O(n³) with a
//! distance prefilter — fine at oracle-corpus scale, and the efficient rolling
//! variant can replace `compute` later behind the same oracle.

use super::geom::{intersect_three_spheres, Sphere, Vec3, EPSILON};

/// One reduced-surface face: a sorted atom triple and the probe-sphere center
/// resting on those three atoms.
#[derive(Clone, Copy, Debug)]
pub struct RsFace {
    pub atoms: [usize; 3],
    pub probe_center: Vec3,
}

/// The reduced surface of a set of atoms for a given probe radius.
#[derive(Clone, Debug)]
pub struct ReducedSurface {
    pub probe_radius: f64,
    pub faces: Vec<RsFace>,
    /// Sorted atom pairs that bound an RS face (the RS edges this layer knows;
    /// free toric edges between exactly two atoms are a later refinement).
    pub edges: Vec<[usize; 2]>,
    /// Indices of atoms that appear on the surface (in any face).
    pub vertices: Vec<usize>,
}

impl ReducedSurface {
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Sorted atom triples, one per RS face (matches `reduced_surface_stats`
    /// `face_atoms` for parity comparison — compare as a multiset).
    pub fn face_atoms(&self) -> Vec<[usize; 3]> {
        self.faces.iter().map(|f| f.atoms).collect()
    }

    pub fn probe_centers(&self) -> Vec<Vec3> {
        self.faces.iter().map(|f| f.probe_center).collect()
    }
}

/// Whether a probe centered at `c` (radius `probe_radius`) clears every atom
/// except the three it rests on — BALL's `checkProbe`: a probe overlapping any
/// other atom is rejected.
fn probe_clear(c: Vec3, probe_radius: f64, atoms: &[Sphere], skip: [usize; 3]) -> bool {
    for (m, atom) in atoms.iter().enumerate() {
        if m == skip[0] || m == skip[1] || m == skip[2] {
            continue;
        }
        let limit = probe_radius + atom.radius;
        // strictly-less ⇒ genuine overlap (touching is allowed); EPSILON guard
        // mirrors BALL's Maths::isLess tolerance.
        if c.square_distance(atom.center) < limit * limit - EPSILON {
            return false;
        }
    }
    true
}

/// Compute the reduced surface. `atoms` are the van-der-Waals spheres; the
/// probe is rolled at `probe_radius`.
pub fn compute(atoms: &[Sphere], probe_radius: f64) -> ReducedSurface {
    let n = atoms.len();
    // Probe-inflated atoms: a probe touches atom i ⇔ its center lies on the
    // sphere (center_i, r_i + probe).
    let inflated: Vec<Sphere> = atoms.iter().map(|a| a.inflated(probe_radius)).collect();

    // Distance prefilter: a probe can touch atoms i and j only if their inflated
    // spheres intersect, i.e. dist(i,j) ≤ (r_i+probe) + (r_j+probe).
    let touch = |i: usize, j: usize| -> bool {
        let reach = inflated[i].radius + inflated[j].radius;
        atoms[i].center.square_distance(atoms[j].center) <= reach * reach
    };

    let mut faces: Vec<RsFace> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if !touch(i, j) {
                continue;
            }
            for k in (j + 1)..n {
                if !touch(i, k) || !touch(j, k) {
                    continue;
                }
                let Some((c1, c2)) = intersect_three_spheres(inflated[i], inflated[j], inflated[k])
                else {
                    continue;
                };
                for c in [c1, c2] {
                    if probe_clear(c, probe_radius, atoms, [i, j, k]) {
                        faces.push(RsFace {
                            atoms: [i, j, k], // already ascending (i<j<k)
                            probe_center: c,
                        });
                    }
                }
            }
        }
    }

    // Edges = sorted atom pairs that bound some face; vertices = atoms in faces.
    let mut edge_set = std::collections::BTreeSet::new();
    let mut vertex_set = std::collections::BTreeSet::new();
    for f in &faces {
        let [a, b, c] = f.atoms;
        edge_set.insert([a, b]);
        edge_set.insert([a, c]);
        edge_set.insert([b, c]);
        vertex_set.insert(a);
        vertex_set.insert(b);
        vertex_set.insert(c);
    }

    ReducedSurface {
        probe_radius,
        faces,
        edges: edge_set.into_iter().collect(),
        vertices: vertex_set.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    #[test]
    fn single_atom_has_no_faces() {
        let rs = compute(&[sph(0.0, 0.0, 0.0, 1.5)], 1.4);
        assert_eq!(rs.num_faces(), 0);
        assert_eq!(rs.num_vertices(), 0);
    }

    #[test]
    fn three_close_atoms_give_two_faces() {
        // Matches the BALL oracle smoke result for three mutually-close atoms:
        // the probe rests above AND below the triangle ⇒ 2 RS faces, both the
        // triple [0,1,2]; 3 edges; 3 vertices.
        let atoms = [
            sph(0.0, 0.0, 0.0, 2.0),
            sph(2.5, 0.0, 0.0, 2.0),
            sph(1.25, 2.0, 0.0, 2.0),
        ];
        let rs = compute(&atoms, 1.4);
        assert_eq!(
            rs.num_faces(),
            2,
            "probe sits on both sides of the triangle"
        );
        for f in &rs.faces {
            assert_eq!(f.atoms, [0, 1, 2]);
        }
        assert_eq!(rs.num_edges(), 3);
        assert_eq!(rs.num_vertices(), 3);
        // The two probe centers are mirror images across the z=0 atom plane.
        let z0 = rs.faces[0].probe_center.z;
        let z1 = rs.faces[1].probe_center.z;
        assert!((z0 + z1).abs() < 1e-9 && z0.abs() > 1e-6);
        // Each probe rests at exactly probe+radius from each of its 3 atoms.
        for f in &rs.faces {
            for &a in &f.atoms {
                let d = f.probe_center.distance(atoms[a].center);
                assert!((d - (1.4 + 2.0)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn buried_atom_does_not_create_faces() {
        // A tiny atom fully inside a large one shares no valid probe triple with
        // a third: the large atom blocks every probe that would touch the small.
        let atoms = [
            sph(0.0, 0.0, 0.0, 5.0),
            sph(0.0, 0.0, 0.0, 0.3), // buried in atom 0
            sph(9.0, 0.0, 0.0, 2.0), // far away
        ];
        let rs = compute(&atoms, 1.4);
        assert_eq!(rs.num_faces(), 0);
    }

    #[test]
    fn far_apart_atoms_no_faces() {
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.5),
            sph(20.0, 0.0, 0.0, 1.5),
            sph(0.0, 20.0, 0.0, 1.5),
        ];
        assert_eq!(compute(&atoms, 1.4).num_faces(), 0);
    }

    /// L1 oracle gate. Expected RS face atom-triple multisets are from
    /// `ball-py 0.1.0a6` `reduced_surface_stats(spheres, probe_radius=1.4)` —
    /// BALL's reduced surface. We gate on the FACE set (the core RS invariant);
    /// edge/vertex parity for face-less configs (free toric edges, isolated
    /// surface atoms) is a later refinement of this layer.
    fn face_multiset(rs: &ReducedSurface) -> std::collections::BTreeMap<[usize; 3], usize> {
        let mut m = std::collections::BTreeMap::new();
        for f in rs.face_atoms() {
            *m.entry(f).or_insert(0) += 1;
        }
        m
    }

    #[test]
    fn rs_face_set_matches_ball_oracle() {
        use std::collections::BTreeMap;

        // triangle3 → oracle: nf=2, {(0,1,2): 2}
        let tri = [
            sph(0.0, 0.0, 0.0, 2.0),
            sph(2.5, 0.0, 0.0, 2.0),
            sph(1.25, 2.0, 0.0, 2.0),
        ];
        assert_eq!(
            face_multiset(&compute(&tri, 1.4)),
            BTreeMap::from([([0, 1, 2], 2)])
        );

        // tetra4 → oracle: nf=4, one face per atom triple
        let tetra = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        assert_eq!(
            face_multiset(&compute(&tetra, 1.4)),
            BTreeMap::from([
                ([0, 1, 2], 1),
                ([0, 1, 3], 1),
                ([0, 2, 3], 1),
                ([1, 2, 3], 1)
            ])
        );

        // chain4 + pair2 → oracle: nf=0 (no 3-atom probe rest)
        let chain = [
            sph(0.0, 0.0, 0.0, 1.5),
            sph(2.6, 0.0, 0.0, 1.5),
            sph(5.2, 0.0, 0.0, 1.5),
            sph(7.8, 0.0, 0.0, 1.5),
        ];
        assert_eq!(compute(&chain, 1.4).num_faces(), 0);
        let pair = [sph(0.0, 0.0, 0.0, 1.8), sph(2.5, 0.0, 0.0, 1.8)];
        assert_eq!(compute(&pair, 1.4).num_faces(), 0);
    }
}
