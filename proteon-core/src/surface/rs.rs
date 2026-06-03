//! L1 reduced surface (RS).
//!
//! The reduced surface is the combinatorial skeleton the SES is built on
//! (Sanner/Connolly): an **RS face** is an atom triple with a probe sphere
//! resting on all three and intersecting no other atom; an **RS edge** is an
//! atom pair the probe can roll between; an **RS vertex** is a surface atom.
//!
//! This first implementation computes the RS by **enumerate-and-validate**:
//! faces from atom triples carrying a non-intersecting probe; edges from atom
//! pairs whose roll-circle has a clear point; vertices from atoms with an
//! exposed inflated cap. It produces the same RS *graph* (faces, edges,
//! vertices) BALL's rolling-probe algorithm does — gated against `ball-py
//! reduced_surface_stats` (counts + atom triples + probe centers). BALL rolls
//! for O(n·k) efficiency; this is O(n³) with a distance prefilter (and circle/
//! sphere sampling for edge/vertex existence) — fine at oracle-corpus scale.
//! The efficient rolling variant and exact (sampling-free) arc analysis can
//! replace `compute` later behind the same oracle.

use super::geom::{intersect_three_spheres, intersect_two_spheres, Sphere, Vec3, EPSILON};

/// Circle / sphere sampling resolution for surface-atom and edge detection.
///
/// LIMITATION (sampling is a stopgap, not exact — to be replaced before
/// production / large molecules): existence of an RS edge/vertex is decided by
/// testing sample points, so it can diverge from BALL in three measure-near-zero
/// regimes — (a) a clear arc narrower than `2π/CIRCLE_SAMPLES` or an exposed cap
/// with solid angle below the sphere-sample spacing is **missed** (false
/// negative); (b) a sample landing just inside a blocker's `EPSILON` clearance
/// band near a tangency reads as clear (false positive); (c) tangent blockers
/// that BALL routes to singular handling. The exact blocked-arc / cap-coverage
/// analysis removes the sampling entirely and is the hardening path; the
/// non-degenerate oracle corpus does not exercise these regimes.
const CIRCLE_SAMPLES: usize = 256;
const SPHERE_SAMPLES: usize = 512;

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
    /// Sorted atom pairs the probe can rest on (face-bounding and free toric
    /// edges), ascending.
    pub edges: Vec<[usize; 2]>,
    /// Indices of the surface atoms (any atom a probe can touch), ascending.
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
/// except those in `skip` — BALL's `checkProbe`: a probe overlapping any other
/// atom is rejected. The comparison is intentionally in **squared** units with
/// an absolute `EPSILON` (`squareDistance < limit² − EPSILON`), mirroring BALL's
/// `Maths::isLess(probe.p.getSquareDistance(atom.p), dist*dist)` exactly — the
/// effective length tolerance is therefore `~EPSILON/(2·limit)`, by design, to
/// match BALL's accept/reject boundary rather than impose a separate one.
fn probe_clear(c: Vec3, probe_radius: f64, atoms: &[Sphere], skip: &[usize]) -> bool {
    for (m, atom) in atoms.iter().enumerate() {
        if skip.contains(&m) {
            continue;
        }
        let limit = probe_radius + atom.radius;
        if c.square_distance(atom.center) < limit * limit - EPSILON {
            return false;
        }
    }
    true
}

/// Two orthonormal vectors spanning the plane normal to `n`.
fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    // Pick the world axis least parallel to n to avoid a degenerate cross.
    let seed = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if n.y.abs() <= n.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = n
        .cross(seed)
        .normalized()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v = n.cross(u);
    (u, v)
}

/// Is atom `i` on the reduced surface — i.e. can a probe touch it without
/// overlapping another atom? Probes touching `i` have their center on `i`'s
/// inflated sphere; `i` is a surface vertex iff any such center clears the rest.
fn atom_exposed(i: usize, atoms: &[Sphere], inflated: &[Sphere], probe_radius: f64) -> bool {
    let ci = atoms[i].center;
    let ri = inflated[i].radius;
    // Fibonacci-sphere directions for near-uniform coverage of the inflated cap.
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for s in 0..SPHERE_SAMPLES {
        let y = 1.0 - 2.0 * (s as f64 + 0.5) / SPHERE_SAMPLES as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * s as f64;
        let dir = Vec3::new(r * theta.cos(), y, r * theta.sin());
        if probe_clear(ci + dir * ri, probe_radius, atoms, &[i]) {
            return true;
        }
    }
    false
}

/// Is the atom pair `(i, j)` an RS edge — i.e. can a probe rest on both without
/// overlapping a third atom? Probes touching both have their center on the
/// roll-circle (the contact circle of the two inflated atoms); the edge exists
/// iff any point on that circle clears the rest.
fn edge_exists(
    i: usize,
    j: usize,
    atoms: &[Sphere],
    inflated: &[Sphere],
    probe_radius: f64,
) -> bool {
    let Some(circle) = intersect_two_spheres(inflated[i], inflated[j]) else {
        return false;
    };
    let (u, v) = plane_basis(circle.normal);
    for s in 0..CIRCLE_SAMPLES {
        let a = std::f64::consts::TAU * s as f64 / CIRCLE_SAMPLES as f64;
        let p = circle.center + u * (circle.radius * a.cos()) + v * (circle.radius * a.sin());
        if probe_clear(p, probe_radius, atoms, &[i, j]) {
            return true;
        }
    }
    false
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
                // A tangent probe (discriminant ≈ 0) yields c1 == c2 — emit that
                // RS face once, not twice. Distinct centers are the usual
                // above/below pair.
                let mut centers = vec![c1];
                if c1.square_distance(c2) > EPSILON * EPSILON {
                    centers.push(c2);
                }
                for c in centers {
                    if probe_clear(c, probe_radius, atoms, &[i, j, k]) {
                        faces.push(RsFace {
                            atoms: [i, j, k], // already ascending (i<j<k)
                            probe_center: c,
                        });
                    }
                }
            }
        }
    }

    // RS edges: every touching pair the probe can rest on without overlapping a
    // third atom (covers both face-bounding and free toric edges).
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if touch(i, j) && edge_exists(i, j, atoms, &inflated, probe_radius) {
                edges.push([i, j]);
            }
        }
    }

    // RS vertices: every atom a probe can touch — i.e. every surface atom.
    let vertices: Vec<usize> = (0..n)
        .filter(|&i| atom_exposed(i, atoms, &inflated, probe_radius))
        .collect();

    ReducedSurface {
        probe_radius,
        faces,
        edges,
        vertices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    #[test]
    fn single_atom_is_a_free_vertex() {
        // A lone atom is entirely solvent-exposed: one free RS vertex, no edges
        // or faces (oracle: nv=1 ne=0 nf=0).
        let rs = compute(&[sph(0.0, 0.0, 0.0, 1.5)], 1.4);
        assert_eq!(
            (rs.num_vertices(), rs.num_edges(), rs.num_faces()),
            (1, 0, 0)
        );
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

    fn face_multiset(rs: &ReducedSurface) -> std::collections::BTreeMap<[usize; 3], usize> {
        let mut m = std::collections::BTreeMap::new();
        for f in rs.face_atoms() {
            *m.entry(f).or_insert(0) += 1;
        }
        m
    }

    /// L1 oracle gate. Expected `(n_vertices, n_edges, n_faces)` and the RS face
    /// atom-triple multisets are from `ball-py 0.1.0a6`
    /// `reduced_surface_stats(spheres, probe_radius=1.4)` — BALL's reduced
    /// surface. Full RS-graph parity (vertices + edges + faces), including the
    /// face-less configs (free toric edges, isolated surface atoms).
    #[test]
    fn rs_graph_matches_ball_oracle() {
        use std::collections::BTreeMap;

        let tri = [
            sph(0.0, 0.0, 0.0, 2.0),
            sph(2.5, 0.0, 0.0, 2.0),
            sph(1.25, 2.0, 0.0, 2.0),
        ];
        let tetra = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        let chain = [
            sph(0.0, 0.0, 0.0, 1.5),
            sph(2.6, 0.0, 0.0, 1.5),
            sph(5.2, 0.0, 0.0, 1.5),
            sph(7.8, 0.0, 0.0, 1.5),
        ];
        let far3 = [
            sph(0.0, 0.0, 0.0, 1.5),
            sph(20.0, 0.0, 0.0, 1.5),
            sph(0.0, 20.0, 0.0, 1.5),
        ];
        let pair = [sph(0.0, 0.0, 0.0, 1.8), sph(2.5, 0.0, 0.0, 1.8)];

        // (config, (nv, ne, nf)) — straight from the oracle.
        let cases: [(&[Sphere], (usize, usize, usize)); 5] = [
            (&tri, (3, 3, 2)),
            (&tetra, (4, 6, 4)),
            (&chain, (4, 3, 0)),
            (&far3, (3, 0, 0)),
            (&pair, (2, 1, 0)),
        ];
        for (atoms, (nv, ne, nf)) in cases {
            let rs = compute(atoms, 1.4);
            assert_eq!(
                (rs.num_vertices(), rs.num_edges(), rs.num_faces()),
                (nv, ne, nf),
                "RS graph counts diverge from BALL"
            );
        }

        // Face atom-triple multisets for the configs that have faces.
        assert_eq!(
            face_multiset(&compute(&tri, 1.4)),
            BTreeMap::from([([0, 1, 2], 2)])
        );
        assert_eq!(
            face_multiset(&compute(&tetra, 1.4)),
            BTreeMap::from([
                ([0, 1, 2], 1),
                ([0, 1, 3], 1),
                ([0, 2, 3], 1),
                ([1, 2, 3], 1)
            ])
        );
    }
}
