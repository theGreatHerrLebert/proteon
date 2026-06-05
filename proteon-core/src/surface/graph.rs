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

use super::geom::{intersect_two_spheres, Sphere, Vec3};
use super::intervals::free_intervals;
use anyhow::{ensure, Result};
use std::f64::consts::TAU;

/// Tangency residual (Å) accepting an atom as touched by the probe at an endpoint.
const TANGENT_TOL: f64 = 1e-6;

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
