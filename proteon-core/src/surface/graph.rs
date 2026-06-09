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

/// A uniform spatial grid over atom centres for neighbour queries, the spatial
/// acceleration that turns the reduced-surface enumeration from O(N³)/O(N⁴) into
/// O(N·k²) (k = average neighbour count). **Cell size = `2·(r_max + probe)`** — the
/// *interaction cutoff*: the three atoms of an RS face are pairwise within that
/// distance (the probe touches all three, so each pair is ≤ 2·(r+probe) apart), and
/// any atom that can block a probe / bury a roll circle lies within `(r_max+probe)`
/// of it — so the 27-cell neighbourhood of any query point captures **every**
/// relevant atom. The grid therefore prunes *only* atoms that provably cannot
/// participate, leaving the enumerated faces bit-identical to the brute-force set.
struct NeighborGrid {
    cell: f64,
    cells: std::collections::HashMap<[i64; 3], Vec<usize>>,
}

impl NeighborGrid {
    fn new(atoms: &[Sphere], probe: f64) -> Self {
        let r_max = atoms.iter().map(|a| a.radius).fold(0.0_f64, f64::max);
        // Cutoff (and cell size) = the max centre-to-centre distance of two atoms
        // that can share a probe, plus headroom so the 27-cell stencil never clips a
        // borderline neighbour. A degenerate cell size is avoided for empty input.
        let cell = (2.0 * (r_max + probe)).max(1e-6);
        let mut cells: std::collections::HashMap<[i64; 3], Vec<usize>> =
            std::collections::HashMap::new();
        for (i, a) in atoms.iter().enumerate() {
            cells.entry(Self::key(a.center, cell)).or_default().push(i);
        }
        NeighborGrid { cell, cells }
    }

    fn key(p: Vec3, cell: f64) -> [i64; 3] {
        [
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        ]
    }

    /// Indices of every atom in the 27 cells around `p` — a superset of all atoms
    /// within `cell` (= the cutoff) of `p`, so any clearance/blocker test run over
    /// this set is exact.
    fn near(&self, p: Vec3) -> Vec<usize> {
        let [cx, cy, cz] = Self::key(p, self.cell);
        let mut out = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.cells.get(&[cx + dx, cy + dy, cz + dz]) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
        out
    }
}

/// Every RS face of `atoms` for the given `probe`: each atom triple carries 0, 1,
/// or 2 probe positions (`intersect_three_spheres`), kept when the probe there
/// clears every *other* atom. The canonical source of SES-corner positions.
///
/// Spatially accelerated via [`NeighborGrid`]: only triples of mutually-near atoms
/// are tested, and clearance is checked against the probe's 27-cell neighbourhood —
/// both provably complete, so the result is identical to the brute-force O(N³) form
/// (asserted by `grid_enumeration_matches_brute_force`).
pub fn enumerate_rs_faces(atoms: &[Sphere], probe: f64) -> Vec<RsFace> {
    let grid = NeighborGrid::new(atoms, probe);
    let cutoff = grid.cell;
    let mut faces = Vec::new();
    for i in 0..atoms.len() {
        // Near atoms with index > i, sorted ascending so the emitted triple stays
        // [i<j<k] like the brute-force loop (downstream matches triples by order).
        let mut near_i: Vec<usize> = grid
            .near(atoms[i].center)
            .into_iter()
            .filter(|&m| m > i && atoms[i].center.distance(atoms[m].center) <= cutoff)
            .collect();
        near_i.sort_unstable();
        for (jpos, &j) in near_i.iter().enumerate() {
            for &k in &near_i[jpos + 1..] {
                // k > j by construction; require j,k also mutually within cutoff.
                if atoms[j].center.distance(atoms[k].center) > cutoff {
                    continue;
                }
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
                    let clear = grid.near(p).into_iter().all(|m| {
                        m == i
                            || m == j
                            || m == k
                            || p.distance(atoms[m].center) >= atoms[m].radius + probe - 1e-9
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
    grid: &NeighborGrid,
) -> Vec<usize> {
    // A third atom tangent to the probe at `p` lies within (r+probe) < cutoff of
    // `p`, so the probe's 27-cell neighbourhood is a complete candidate set.
    let p = roll_point(roll, theta);
    grid.near(p)
        .into_iter()
        .filter(|&k| k != i && k != j)
        .filter(|&k| (p.distance(atoms[k].center) - (atoms[k].radius + probe)).abs() < TANGENT_TOL)
        .collect()
}

/// Every toric face of `atoms` for the given `probe` (analytic, non-singular).
///
/// **Errors** (rather than emit an ill-identified face) when a bounded interval's
/// endpoint is not a clean generic RS vertex: no tangent third atom (a
/// construction failure) or ≥2 (a cospherical-degeneracy / singular vertex).
///
/// Spatially accelerated via [`NeighborGrid`] (identical result to brute force,
/// asserted by `grid_enumeration_matches_brute_force`): only near pairs roll, and a
/// roll circle's blockers lie within the cutoff of its centre.
pub fn enumerate_toric_faces(atoms: &[Sphere], probe: f64) -> Result<Vec<ToricFace>> {
    let grid = NeighborGrid::new(atoms, probe);
    let cutoff = grid.cell;
    let mut faces = Vec::new();
    for i in 0..atoms.len() {
        let mut near_i: Vec<usize> = grid
            .near(atoms[i].center)
            .into_iter()
            .filter(|&m| m > i && atoms[i].center.distance(atoms[m].center) <= cutoff)
            .collect();
        near_i.sort_unstable();
        for &j in &near_i {
            let Some(roll) =
                intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
            else {
                continue;
            };
            // Blockers of the roll circle lie within the cutoff of its centre.
            let blockers: Vec<Sphere> = grid
                .near(roll.center)
                .into_iter()
                .filter(|&k| k != i && k != j)
                .map(|k| atoms[k])
                .collect();
            for (s, e) in free_intervals(&roll, &blockers, probe) {
                let full_ring = s.abs() < 1e-12 && (e - TAU).abs() < 1e-12;
                let ends = if full_ring {
                    [None, None]
                } else {
                    let mut got = [None, None];
                    for (slot, &theta) in got.iter_mut().zip([s, e].iter()) {
                        let owners = tangent_thirds(atoms, i, j, &roll, theta, probe, &grid);
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

    /// Brute-force O(N³) RS-face enumeration — the reference the grid-accelerated
    /// [`enumerate_rs_faces`] must reproduce *exactly* (same faces, same order, same
    /// probe positions). The grid only prunes provably-irrelevant atoms, so any
    /// divergence is a bug in the cutoff/stencil reasoning.
    fn rs_faces_brute(atoms: &[Sphere], probe: f64) -> Vec<RsFace> {
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
                        cand.push(p2);
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

    #[test]
    fn grid_enumeration_matches_brute_force() {
        // A deterministic blob of overlapping atoms spanning several grid cells, so
        // the 27-cell stencil and cross-cell triples are exercised. SplitMix-style
        // jitter (no rand dep), radii varied so r_max sets the cutoff.
        let mut z: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((x ^ (x >> 31)) as f64) / (u64::MAX as f64)
        };
        let mut atoms = Vec::new();
        for _ in 0..120 {
            atoms.push(sph(
                next() * 12.0,
                next() * 12.0,
                next() * 12.0,
                1.4 + next() * 0.6,
            ));
        }
        let probe = 1.4;
        // RS faces: identical sequence (set, order, and probe positions).
        let grid = enumerate_rs_faces(&atoms, probe);
        let brute = rs_faces_brute(&atoms, probe);
        assert_eq!(grid.len(), brute.len(), "RS face count differs");
        for (g, b) in grid.iter().zip(&brute) {
            assert_eq!(g.atoms, b.atoms, "RS face triple/order differs");
            assert!(g.probe.distance(b.probe) < 1e-12, "RS probe position differs");
        }
        // Toric faces: same set of (edge, θ-interval, ends). Order can differ (the
        // grid outer loop is near-sorted), so compare as sets. Brute reference uses
        // all atoms as blockers and all atoms for the tangent third.
        let mut brute_toric: Vec<([usize; 2], [Option<usize>; 2])> = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let Some(roll) =
                    intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
                else {
                    continue;
                };
                let blockers: Vec<Sphere> =
                    (0..atoms.len()).filter(|&k| k != i && k != j).map(|k| atoms[k]).collect();
                for (s, e) in free_intervals(&roll, &blockers, probe) {
                    let full = s.abs() < 1e-12 && (e - TAU).abs() < 1e-12;
                    let ends = if full {
                        [None, None]
                    } else {
                        let mut got = [None, None];
                        for (slot, &th) in got.iter_mut().zip([s, e].iter()) {
                            let p = roll_point(&roll, th);
                            let owners: Vec<usize> = (0..atoms.len())
                                .filter(|&k| k != i && k != j)
                                .filter(|&k| {
                                    (p.distance(atoms[k].center) - (atoms[k].radius + probe)).abs()
                                        < TANGENT_TOL
                                })
                                .collect();
                            assert_eq!(owners.len(), 1, "test blob must be generic");
                            *slot = Some(owners[0]);
                        }
                        got
                    };
                    brute_toric.push(([i, j], ends));
                }
            }
        }
        let tg = enumerate_toric_faces(&atoms, probe).unwrap();
        let mut gset: Vec<_> = tg.iter().map(|f| (f.edge, f.ends)).collect();
        gset.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        brute_toric.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        assert_eq!(gset, brute_toric, "toric (edge, ends) set differs from brute force");
        assert!(!gset.is_empty(), "blob should have toric faces");
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
