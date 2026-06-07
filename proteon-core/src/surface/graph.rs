//! SES element enumeration (general-N) — toric faces from the analytic free
//! intervals, with each interval endpoint identified to the RS face it bounds.
//!
//! For every atom pair with a roll circle, [`intervals::free_intervals`] gives the
//! free θ-arcs (the toric faces). Each non-ring endpoint is a probe position
//! touching a *third* atom; we recover that atom by tangency, so the endpoint is
//! labelled with its RS face (the sorted atom triple). This is the analytic
//! authority for which pairs are RS edges (≥1 free interval) and which triples are
//! RS faces — replacing the sampled detection per the general-N plan.
//!
//! NOTE (codex-review): this labels which RS face an endpoint belongs to (the atom
//! triple), but it is NOT the place to establish shared **SES-vertex identity**.
//! The three pairs of one RS face each rediscover that probe centre independently,
//! so the half-edge build must compute each SES vertex **once** per RS face
//! (canonical probe centre) and intern it — not take it from per-pair recovery.

use super::geom::{intersect_three_spheres, intersect_two_spheres, Sphere, Vec3};
use super::intervals::free_intervals;
use anyhow::{ensure, Context, Result};
use std::f64::consts::TAU;

/// Tangency residual (Å) accepting an atom as touched by the probe at an endpoint.
const TANGENT_TOL: f64 = 1e-6;

/// One reduced-surface face: a probe resting on an atom triple, clear of every
/// other atom. The `probe` centre is computed **once** here (analytically, via
/// `intersect_three_spheres`) so all three incident toric pairs and the contact
/// caps share the *same* SES corner positions (the canonical-vertex identity the
/// half-edge decomposition needs — codex-review).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RsFace {
    pub atoms: [usize; 3],
    pub probe: Vec3,
}

/// Every RS face of `atoms` for the given `probe`: each atom triple carries 0, 1,
/// or 2 probe positions (`intersect_three_spheres`), kept when the probe there
/// clears every *other* atom. The canonical source of SES-corner positions.
pub fn enumerate_rs_faces(atoms: &[Sphere], probe: f64) -> Vec<RsFace> {
    let mut faces = Vec::new();
    let n = atoms.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let Some((p1, p2)) = intersect_three_spheres(
                    atoms[i].inflated(probe),
                    atoms[j].inflated(probe),
                    atoms[k].inflated(probe),
                ) else {
                    continue;
                };
                let mut cand = vec![p1];
                if p1.distance(p2) > 1e-9 {
                    cand.push(p2); // distinct above/below probes (else tangent triple)
                }
                for p in cand {
                    let clear = atoms.iter().enumerate().all(|(m, a)| {
                        m == i
                            || m == j
                            || m == k
                            || p.distance(a.center) >= a.radius + probe - 1e-9
                    });
                    if clear {
                        faces.push(RsFace {
                            atoms: [i, j, k],
                            probe: p,
                        });
                    }
                }
            }
        }
    }
    faces
}

/// One toric face: a free interval of the `[i,j]` roll circle. `ends[e]` is the
/// third atom bounding that endpoint (so the RS face is the sorted triple), or
/// `None` for a full free ring's (absent) ends.
#[derive(Clone, Debug, PartialEq)]
pub struct ToricFace {
    pub edge: [usize; 2],
    pub theta: (f64, f64),
    pub ends: [Option<usize>; 2],
}

/// The point on `roll` at angle `theta`.
fn roll_point(roll: &super::geom::Circle3, theta: f64) -> Vec3 {
    let (u, v) = super::geom::plane_basis(roll.normal);
    roll.center + (u * theta.cos() + v * theta.sin()) * roll.radius
}

/// Every third atom the probe at roll-angle `theta` is tangent to (within
/// [`TANGENT_TOL`]). Generically exactly one (the blocker bounding the endpoint);
/// **more than one means a ≥4-cospherical degeneracy** — a singular SES vertex the
/// non-singular path must reject rather than silently pick one (codex-review).
fn tangent_thirds(
    atoms: &[Sphere],
    i: usize,
    j: usize,
    roll: &super::geom::Circle3,
    theta: f64,
    probe: f64,
) -> Vec<usize> {
    let p = roll_point(roll, theta);
    (0..atoms.len())
        .filter(|&k| k != i && k != j)
        .filter(|&k| (p.distance(atoms[k].center) - (atoms[k].radius + probe)).abs() < TANGENT_TOL)
        .collect()
}

/// Every toric face of `atoms` for the given `probe` (analytic, non-singular).
///
/// **Errors** (rather than emit an ill-identified face) when a bounded interval's
/// endpoint is not a clean generic RS vertex: no tangent third atom (a
/// construction failure) or ≥2 (a cospherical-degeneracy / singular vertex).
pub fn enumerate_toric_faces(atoms: &[Sphere], probe: f64) -> Result<Vec<ToricFace>> {
    let mut faces = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let Some(roll) =
                intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
            else {
                continue;
            };
            let blockers: Vec<Sphere> = atoms
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, &a)| a)
                .collect();
            for (s, e) in free_intervals(&roll, &blockers, probe) {
                let full_ring = s.abs() < 1e-12 && (e - TAU).abs() < 1e-12;
                let ends = if full_ring {
                    [None, None]
                } else {
                    let mut got = [None, None];
                    for (slot, &theta) in got.iter_mut().zip([s, e].iter()) {
                        let owners = tangent_thirds(atoms, i, j, &roll, theta, probe);
                        ensure!(
                            owners.len() == 1,
                            "toric endpoint of pair [{i},{j}] has {} tangent third atoms \
                             (expected 1; ≥2 ⇒ cospherical/singular)",
                            owners.len()
                        );
                        *slot = Some(owners[0]);
                    }
                    got
                };
                faces.push(ToricFace {
                    edge: [i, j],
                    theta: (s, e),
                    ends,
                });
            }
        }
    }
    Ok(faces)
}

/// A toric face wired into the graph: a free interval of the `[i,j]` roll circle
/// whose endpoints reference the **canonical** RS faces (indices into
/// [`SesGraph::rs_faces`]), so every incident patch resolves the same SES corners.
/// `end_faces[e] == None` only for a full free ring.
#[derive(Clone, Debug)]
pub struct ToricArc {
    pub edge: [usize; 2],
    pub theta: (f64, f64),
    pub end_faces: [Option<usize>; 2],
}

/// The SES element graph: canonical RS faces + toric arcs wired to them. The
/// half-edge decomposition the general-N assembler stitches.
#[derive(Clone, Debug)]
pub struct SesGraph {
    pub rs_faces: Vec<RsFace>,
    pub toric: Vec<ToricArc>,
}

/// Build the SES element graph: enumerate canonical RS faces and toric intervals,
/// then **link each toric endpoint to its canonical RS face** (matched by atom
/// triple + nearest probe, validated within tolerance). That shared index is what
/// makes the incident toric pairs, contact caps and spheric faces resolve the same
/// SES-vertex positions — the basis for a watertight general-N stitch.
pub fn build_graph(atoms: &[Sphere], probe: f64) -> Result<SesGraph> {
    let rs_faces = enumerate_rs_faces(atoms, probe);
    let toric_faces = enumerate_toric_faces(atoms, probe)?;
    let mut toric = Vec::with_capacity(toric_faces.len());
    for tf in &toric_faces {
        let [i, j] = tf.edge;
        let roll = intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
            .context("toric pair lost its roll circle")?;
        let thetas = [tf.theta.0, tf.theta.1];
        let mut end_faces = [None, None];
        for e in 0..2 {
            if let Some(k) = tf.ends[e] {
                let p = roll_point(&roll, thetas[e]);
                let mut triple = [i, j, k];
                triple.sort_unstable();
                let idx = rs_faces
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.atoms == triple)
                    .min_by(|(_, x), (_, y)| p.distance(x.probe).total_cmp(&p.distance(y.probe)))
                    .map(|(idx, _)| idx)
                    .context("toric endpoint has no matching RS face")?;
                ensure!(
                    p.distance(rs_faces[idx].probe) < 1e-6,
                    "toric endpoint probe does not match its canonical RS face"
                );
                end_faces[e] = Some(idx);
            }
        }
        toric.push(ToricArc {
            edge: tf.edge,
            theta: tf.theta,
            end_faces,
        });
    }
    Ok(SesGraph { rs_faces, toric })
}

#[cfg(test)]
mod tests {
    use super::super::geom::Vec3;
    use super::*;
    use std::collections::HashSet;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// The RS faces implied by the toric endpoints, as sorted atom triples.
    fn rs_faces(faces: &[ToricFace]) -> HashSet<[usize; 3]> {
        let mut s = HashSet::new();
        for f in faces {
            for end in f.ends.into_iter().flatten() {
                let mut t = [f.edge[0], f.edge[1], end];
                t.sort_unstable();
                s.insert(t);
            }
        }
        s
    }

    #[test]
    fn triangle3_has_three_toric_faces_and_one_rs_face_triple() {
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.7),
            sph(2.5, 0.0, 0.0, 1.7),
            sph(1.25, 2.165, 0.0, 1.7),
        ];
        let faces = enumerate_toric_faces(&atoms, 1.4).unwrap();
        assert_eq!(faces.len(), 3, "3 pairs × 1 free interval");
        for f in &faces {
            // Both ends are bounded by the third atom (the triple's apex).
            assert!(f.ends[0].is_some() && f.ends[1].is_some());
        }
        assert_eq!(
            rs_faces(&faces),
            HashSet::from([[0, 1, 2]]),
            "the only RS-face triple is {{0,1,2}}"
        );
    }

    #[test]
    fn a_free_pair_has_a_full_ring_toric_face() {
        // Two atoms with a distant third that blocks nothing: pair (0,1) is a free
        // ring; the third atom shares no toric face at all.
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.6, 0.0, 0.0, 1.6),
            sph(20.0, 0.0, 0.0, 1.6),
        ];
        let faces = enumerate_toric_faces(&atoms, 1.4).unwrap();
        let pair01: Vec<_> = faces.iter().filter(|f| f.edge == [0, 1]).collect();
        assert_eq!(pair01.len(), 1);
        assert_eq!(pair01[0].ends, [None, None], "free ring, no RS-face ends");
        assert_eq!(pair01[0].theta, (0.0, TAU));
    }

    #[test]
    fn rs_faces_are_canonical_and_clear_of_other_atoms() {
        // triangle3: one triple, two probe positions (above/below the plane).
        let tri = [
            sph(0.0, 0.0, 0.0, 1.7),
            sph(2.5, 0.0, 0.0, 1.7),
            sph(1.25, 2.165, 0.0, 1.7),
        ];
        let f = enumerate_rs_faces(&tri, 1.4);
        assert_eq!(f.len(), 2, "triangle3 → 2 RS faces");
        assert!(f.iter().all(|x| x.atoms == [0, 1, 2]));
        // The two probes are distinct and each tangent to all three atoms.
        assert!(f[0].probe.distance(f[1].probe) > 1e-3);
        for face in &f {
            for &a in &face.atoms {
                assert!((face.probe.distance(tri[a].center) - (tri[a].radius + 1.4)).abs() < 1e-9);
            }
        }
        // tetra: each triple's *inner* probe is buried by the 4th atom → 4 faces.
        let tetra = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        assert_eq!(
            enumerate_rs_faces(&tetra, 1.4).len(),
            4,
            "tetra → 4 RS faces"
        );
    }

    #[test]
    fn graph_shares_each_rs_face_across_its_three_toric_arcs() {
        // triangle3: 2 RS faces, 3 toric arcs; each RS face is an endpoint of all
        // three toric arcs (the half-edge sharing that makes the stitch watertight).
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.7),
            sph(2.5, 0.0, 0.0, 1.7),
            sph(1.25, 2.165, 0.0, 1.7),
        ];
        let g = build_graph(&atoms, 1.4).unwrap();
        assert_eq!(g.rs_faces.len(), 2);
        assert_eq!(g.toric.len(), 3);
        for face_idx in 0..g.rs_faces.len() {
            let refs = g
                .toric
                .iter()
                .filter(|t| t.end_faces.contains(&Some(face_idx)))
                .count();
            assert_eq!(refs, 3, "RS face {face_idx} shared by all 3 toric arcs");
        }
        // tetra: every toric endpoint resolves to a canonical RS face (no None on a
        // bounded arc), so the graph is fully wired.
        let tetra = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        let gt = build_graph(&tetra, 1.4).unwrap();
        assert_eq!(gt.rs_faces.len(), 4);
        for t in &gt.toric {
            assert!(
                t.end_faces.iter().all(Option::is_some),
                "tetra toric arcs bounded"
            );
        }
    }

    #[test]
    fn tetra_enumerates_four_rs_face_triples() {
        // 4 mutually-contacting atoms: every triple carries a probe → 4 RS faces.
        let atoms = [
            sph(0.0, 0.0, 0.0, 1.6),
            sph(2.0, 0.0, 0.0, 1.6),
            sph(1.0, 1.7, 0.0, 1.6),
            sph(1.0, 0.6, 1.6, 1.6),
        ];
        let faces = enumerate_toric_faces(&atoms, 1.4).unwrap();
        assert_eq!(
            rs_faces(&faces),
            HashSet::from([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            "all four triples are RS faces"
        );
    }
}
