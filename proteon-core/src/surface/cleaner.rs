//! SES singularity cleaner (nonradial) — a staged port of BALL's
//! `SESSingularityCleaner` graph rewrite.
//!
//! When two fixed probes sit closer than `2·probe` their reentrant surfaces
//! interpenetrate; the raw analytic mesh is watertight but self-intersects (the
//! dominant defect on real proteins — measured 1575 probe-probe collisions vs 21
//! spindle on crambin). BALL does *not* subtract caps per spheric triangle
//! (codex-review: that spherical-envelope op does not reproduce the cleaned
//! topology). Instead it rewrites the surface graph along **singular edges** that
//! live on the pairwise probe-intersection circles, meeting at **triple-probe
//! vertices**. This module ports that, staged:
//!
//! 1. [`SingularVertices`] — the **unified boundary-event registry**: triple-probe
//!    vertices `(sorted i,j,k, branch)` *and* great-circle corners `(owner i,
//!    probe j, sorted atoms, branch)`, interned in one pool so every incident
//!    patch looks up bit-identical coordinates (the weld guarantee; `welded()`
//!    compares `to_bits()`). The branch bit is part of each key (codex Q2/Q6).
//! 2. singular edges on each `C_ij` split by those vertices + global exposure.
//! 3. spheric + toric face rewrite onto the singular edges (`clip_spheric_face`
//!    is the spheric half; the toric event-aligned trim is next).
//! 4. richer gate vs BALL (volume, Euler, per-face-type area, …).

use super::arrangement::{arrange_loops, sample_loop};
use super::chart::fill_spherical_region;
use super::geom::{plane_basis, Plane3, Vec3};
use super::mesh::Mesh;
use super::nonradial::{
    canonical_burial_circle, circle_plane_intersections, probe_burial_cap, sample_circle_rim,
    spheric_face_caps, triple_sphere_intersections,
};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Unified canonical registry of SES boundary-event vertices (codex review): the
/// single source of truth so every incident patch looks up **bit-identical**
/// coordinates (`Mesh::welded()` compares `f64::to_bits()`, so mathematical
/// equality is not enough). Two event kinds, one shared point pool:
///
/// - **triple-probe vertices** — keyed `(sorted i,j,k, branch)`; where three
///   collision circles / singular edges meet.
/// - **great-circle corners** — keyed `(owner probe i, probe j, sorted atoms a,b,
///   branch)`; where a collision circle `C_ij` crosses a contact great circle
///   (the spheric↔toric boundary on sphere `i`). Both the spheric clip of face
///   `i` and the toric trim along edge `(a,b)` resolve the same corner.
#[derive(Default)]
pub struct SingularVertices {
    points: Vec<Vec3>,
    triple_index: HashMap<([usize; 3], i8), usize>,
    corner_index: HashMap<(usize, usize, [usize; 2], i8), usize>,
}

impl SingularVertices {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn point(&self, id: usize) -> Vec3 {
        self.points[id]
    }

    /// Intern the triple-probe vertices of probes `(i, j, k)` (probe spheres all
    /// of radius `probe`, centres `centers`), returning their registry ids.
    /// Canonical:
    /// the triple is sorted before computing geometry, so `intern_triple` is
    /// invariant to the argument order — `(i,j,k)` and `(k,j,i)` give the same
    /// ids and the same coordinates. Empty if the three spheres share no point.
    pub fn intern_triple(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        centers: &[Vec3],
        probe: f64,
    ) -> Vec<usize> {
        let mut tri = [i, j, k];
        tri.sort_unstable();
        if tri[0] == tri[1] || tri[1] == tri[2] {
            return Vec::new(); // not three distinct probes
        }
        let (a, b, c) = (centers[tri[0]], centers[tri[1]], centers[tri[2]]);
        let mut ids = Vec::new();
        // Branch label comes from the construction order (codex review): robust
        // against near-degenerate triples that a coordinate-derived sign could
        // collapse onto one key.
        for (x, branch) in triple_sphere_intersections(a, b, c, probe) {
            let id = *self.triple_index.entry((tri, branch)).or_insert_with(|| {
                self.points.push(x);
                self.points.len() - 1
            });
            ids.push(id);
        }
        ids
    }

    /// Intern the **great-circle corner(s)** where collision circle `C_ij` crosses
    /// a contact great circle on sphere `i` — the plane through probe `i` and the
    /// toric axis of atoms `(a, b)` (`great_plane`). Keyed `(i, j, sorted{a,b},
    /// branch)` so the spheric clip of face `i` and the toric trim of edge `(a,b)`
    /// resolve the **same** interned point. The key — not the recomputed geometry —
    /// guarantees bit-identity: the first caller fixes the coordinate, the second
    /// looks it up. Empty if the probes do not overlap or the circle misses the
    /// plane.
    pub fn intern_corner(
        &mut self,
        i: usize,
        j: usize,
        atoms: [usize; 2],
        centers: &[Vec3],
        probe: f64,
        great_plane: &Plane3,
    ) -> Vec<usize> {
        let Some(c) = canonical_burial_circle(i, j, centers, probe) else {
            return Vec::new();
        };
        let mut ab = atoms;
        ab.sort_unstable();
        let mut ids = Vec::new();
        for (x, branch) in circle_plane_intersections(&c, great_plane) {
            let id = *self
                .corner_index
                .entry((i, j, ab, branch))
                .or_insert_with(|| {
                    self.points.push(x);
                    self.points.len() - 1
                });
            ids.push(id);
        }
        ids
    }
}

/// A singular edge: an exposed θ-interval of the collision circle `C_ij`, the arc
/// where probes `i` and `j`'s reentrant surfaces meet on the actual SES. Sample
/// it with `nonradial::sample_circle_rim(C_ij, …)` so both incident faces weld.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingularEdge {
    pub theta_start: f64,
    pub theta_end: f64,
}

/// Is a point on a collision circle exposed as a singular edge — i.e. not strictly
/// inside any *other* probe ball (`m ∉ {i, j}`)? (Atom exclusion is a later
/// refinement; this is the probe-probe term that dominates the defect.)
fn singular_exposed(x: Vec3, i: usize, j: usize, centers: &[Vec3], probe: f64) -> bool {
    let inside = probe * (1.0 - 1e-9); // scale-relative inward bias
    !centers
        .iter()
        .enumerate()
        .any(|(m, &c)| m != i && m != j && x.distance(c) < inside)
}

/// Stage 2 — the singular edges on the collision circle `C_ij` of two overlapping
/// probes: split the full circle by every triple-probe vertex `(i, j, k)` (a
/// global event, registered canonically), then keep the arcs whose midpoint is
/// exposed. One global arrangement of the circle (codex Q1), not two per-face
/// DCELs. Empty if the probes do not overlap or `C_ij` is wholly buried.
pub fn singular_edges(
    i: usize,
    j: usize,
    centers: &[Vec3],
    probe: f64,
    reg: &mut SingularVertices,
) -> Vec<SingularEdge> {
    let Some(c) = canonical_burial_circle(i, j, centers, probe) else {
        return Vec::new();
    };
    let (u, v) = plane_basis(c.normal);
    let theta_of = |x: Vec3| -> f64 {
        let r = x - c.center;
        let t = r.dot(v).atan2(r.dot(u));
        if t < 0.0 {
            t + TAU
        } else {
            t
        }
    };
    // Split angles: every triple vertex (i,j,k) lies on C_ij (equidistant `probe`
    // from i and j), so it is a split point where a third ball enters/leaves.
    let mut splits: Vec<f64> = Vec::new();
    for k in 0..centers.len() {
        if k == i || k == j {
            continue;
        }
        for id in reg.intern_triple(i, j, k, centers, probe) {
            splits.push(theta_of(reg.point(id)));
        }
    }
    let sample = |t: f64| c.center + (u * t.cos() + v * t.sin()) * c.radius;
    if splits.is_empty() {
        // No third ball crosses C_ij: it is wholly exposed or wholly buried.
        return if singular_exposed(sample(0.0), i, j, centers, probe) {
            vec![SingularEdge {
                theta_start: 0.0,
                theta_end: TAU,
            }]
        } else {
            Vec::new()
        };
    }
    splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Dedup by *circular* distance, derived from a positional tolerance (raw
    // angular tol scales as radius·Δθ), and merge the 0/2π seam (codex review).
    let ang_tol = (1e-9 / c.radius.max(1e-9)).min(1e-6);
    splits.dedup_by(|a, b| (*a - *b).abs() < ang_tol);
    if splits.len() > 1 && (TAU - splits[splits.len() - 1] + splits[0]) < ang_tol {
        splits.pop(); // first and last coincide across the seam
    }
    // Arcs between consecutive splits (cyclic); keep the exposed ones.
    let mut edges = Vec::new();
    for w in 0..splits.len() {
        let (a, b) = (splits[w], splits[(w + 1) % splits.len()]);
        let end = if b > a { b } else { b + TAU };
        let mid = (a + end) / 2.0;
        if singular_exposed(sample(mid), i, j, centers, probe) {
            edges.push(SingularEdge {
                theta_start: a,
                theta_end: end,
            });
        }
    }
    edges
}

/// Sample one singular edge into world points on the canonical collision circle
/// `C_ij` (`n_interior` interior points plus both corners). Because the edge θ-
/// range and the circle are both canonical (keyed on the unordered pair), the two
/// faces that share this seam call this and get **bit-identical** points — the
/// weld guarantee for the stage-3 face rewrite. Caller passes the same edge from
/// [`singular_edges`].
pub fn sample_singular_edge(
    i: usize,
    j: usize,
    centers: &[Vec3],
    probe: f64,
    edge: SingularEdge,
    n_interior: usize,
) -> Vec<Vec3> {
    let Some(c) = canonical_burial_circle(i, j, centers, probe) else {
        return Vec::new();
    };
    let thetas: Vec<f64> = (0..=n_interior + 1)
        .map(|k| {
            edge.theta_start
                + (edge.theta_end - edge.theta_start) * k as f64 / (n_interior + 1) as f64
        })
        .collect();
    sample_circle_rim(&c, &thetas)
}

/// Stage 3b — rewrite one spheric (reentrant) face, trimmed by the neighbour
/// probes that collide with its probe `p`. The face is `exposed({3 great circles}
/// ∪ {burial caps})`; `arrange_loops` resolves the (possibly multivalent)
/// boundary and `fill_spherical_region` meshes it. The burial portions of the
/// boundary land on the collision circles `C_pj` (shared seams).
///
/// `p` is this RS face's probe centre, `cs` its three contact points, `centers`
/// all RS-face probe centres, `this_idx` the index of `p` within them. With no
/// colliding neighbour the result is the original spheric triangle (regression).
///
/// NOTE: this meshes the patch in `p`'s own frame; the *cross-face* weld (sampling
/// the shared burial seam canonically via [`sample_singular_edge`], and the
/// coordinated toric trim) is the remaining stage-3 wiring.
pub fn clip_spheric_face(
    p: Vec3,
    cs: [Vec3; 3],
    centers: &[Vec3],
    this_idx: usize,
    probe: f64,
    grid: f64,
    n_arc: usize,
) -> Result<Mesh> {
    let dir = |x: Vec3| (x - p).normalized().context("contact at probe centre");
    let dirs = [dir(cs[0])?, dir(cs[1])?, dir(cs[2])?];
    let great = spheric_face_caps(dirs).context("degenerate spheric triple")?;
    let mut caps = great.to_vec();
    for (j, &q) in centers.iter().enumerate() {
        if j != this_idx {
            if let Some(bc) = probe_burial_cap(p, q, probe) {
                caps.push(bc);
            }
        }
    }
    let loops = arrange_loops(&caps)?;
    let pole = ((dirs[0] + dirs[1] + dirs[2]) * (1.0 / 3.0))
        .normalized()
        .context("spheric centroid degenerate")?;
    let world: Vec<Vec<Vec3>> = loops
        .iter()
        .map(|l| {
            sample_loop(l, &caps, n_arc)
                .into_iter()
                .map(|d| p + d * probe)
                .collect()
        })
        .collect();
    fill_spherical_region(p, probe, &world, pole, grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctr(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn interning_is_permutation_invariant_and_deduplicated() {
        let probe = 1.4;
        let centers = [ctr(0.0, 0.0, 0.0), ctr(1.6, 0.0, 0.0), ctr(0.5, 1.5, 0.0)];
        let mut reg = SingularVertices::new();
        let a = reg.intern_triple(0, 1, 2, &centers, probe);
        assert_eq!(a.len(), 2, "generic triple → two branch vertices");
        // Any permutation interns the SAME ids (no new points).
        let b = reg.intern_triple(2, 0, 1, &centers, probe);
        let c = reg.intern_triple(1, 2, 0, &centers, probe);
        let mut a_s = a.clone();
        a_s.sort_unstable();
        let mut b_s = b.clone();
        b_s.sort_unstable();
        let mut c_s = c;
        c_s.sort_unstable();
        assert_eq!(a_s, b_s);
        assert_eq!(a_s, c_s);
        assert_eq!(reg.len(), 2, "no duplicate vertices across permutations");
        // Interned coordinates are exactly equidistant `probe` from all three.
        for &id in &a {
            for ctr in &centers {
                assert!((reg.point(id).distance(*ctr) - probe).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distinct_triples_get_distinct_entries_and_no_phantom_points() {
        let probe = 1.4;
        let centers = [
            ctr(0.0, 0.0, 0.0),
            ctr(1.6, 0.0, 0.0),
            ctr(0.5, 1.5, 0.0),
            ctr(0.8, 0.5, 1.4),
        ];
        let mut reg = SingularVertices::new();
        let t012 = reg.intern_triple(0, 1, 2, &centers, probe);
        let t013 = reg.intern_triple(0, 1, 3, &centers, probe);
        // Different triples → different vertices (no accidental merge).
        for x in &t012 {
            assert!(!t013.contains(x));
        }
        assert_eq!(reg.len(), t012.len() + t013.len());
    }

    // A spheric triangle ~40° around +z (wide enough for a burial cap to bite an
    // edge without swallowing the whole face).
    fn wide_face() -> (Vec3, [Vec3; 3]) {
        let p = ctr(0.0, 0.0, 0.0);
        let probe = 1.4;
        let (s, c) = (40.0_f64.to_radians().sin(), 40.0_f64.to_radians().cos());
        let cs = std::array::from_fn(|k| {
            let phi = TAU * k as f64 / 3.0;
            p + Vec3::new(s * phi.cos(), s * phi.sin(), c) * probe
        });
        (p, cs)
    }

    #[test]
    fn clip_with_no_or_far_neighbour_is_the_original_face() {
        let (p, cs) = wide_face();
        let probe = 1.4;
        let alone = clip_spheric_face(p, cs, &[p], 0, probe, 0.07, 14).unwrap();
        assert_eq!(alone.euler_characteristic(), 1, "open spheric disk");
        let a0 = alone.surface_area();
        assert!(a0 > 0.0);
        // A neighbour beyond 2·probe contributes no burial cap → identical.
        let far = clip_spheric_face(p, cs, &[p, ctr(5.0, 0.0, 0.0)], 0, probe, 0.07, 14).unwrap();
        assert!(
            (far.surface_area() - a0).abs() < 1e-9,
            "far neighbour changes nothing"
        );
    }

    #[test]
    fn clip_with_colliding_neighbour_removes_area_and_stays_a_disk() {
        let (p, cs) = wide_face();
        let probe = 1.4;
        let q = ctr(1.2, 0.0, 0.8); // |p−q| = 1.44 < 2.8, biased toward +x edge
        let without = clip_spheric_face(p, cs, &[p], 0, probe, 0.06, 18).unwrap();
        let with = clip_spheric_face(p, cs, &[p, q], 0, probe, 0.06, 18).unwrap();
        assert_eq!(
            with.euler_characteristic(),
            1,
            "clipped face is still a disk"
        );
        assert!(
            with.surface_area() < without.surface_area() * 0.98,
            "neighbour burial removed area: {} vs {}",
            with.surface_area(),
            without.surface_area()
        );
        // Cross-check the kept area against a grid estimate of the directions that
        // are inside the triangle AND outside q's ball.
        let great = spheric_face_caps([
            (cs[0] - p).normalized().unwrap(),
            (cs[1] - p).normalized().unwrap(),
            (cs[2] - p).normalized().unwrap(),
        ])
        .unwrap();
        let mut hit = 0usize;
        let n = 300_000;
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for sidx in 0..n {
            let y = 1.0 - 2.0 * (sidx as f64 + 0.5) / n as f64;
            let rr = (1.0 - y * y).max(0.0).sqrt();
            let th = golden * sidx as f64;
            let d = Vec3::new(rr * th.cos(), y, rr * th.sin());
            let in_tri = super::super::arrangement::is_exposed(d, &great);
            let outside_q = (p + d * probe).distance(q) >= probe;
            if in_tri && outside_q {
                hit += 1;
            }
        }
        let grid = 4.0 * std::f64::consts::PI * probe * probe * hit as f64 / n as f64;
        assert!(
            (with.surface_area() - grid).abs() / grid < 0.03,
            "clipped area {} within 3% of grid {grid}",
            with.surface_area()
        );
    }

    #[test]
    fn great_circle_corner_is_interned_once_for_both_consumers() {
        // C_01 lives in the plane x=1 around (1,0,0); the contact great plane y=0
        // (through probe 0 at the origin) cuts it at (1,0,±0.98).
        let probe = 1.4;
        let centers = [ctr(0.0, 0.0, 0.0), ctr(2.0, 0.0, 0.0)];
        let plane = Plane3 {
            normal: ctr(0.0, 1.0, 0.0),
            d: 0.0,
        };
        let mut reg = SingularVertices::new();
        // Consumer A = the spheric clip of face 0; consumer B = the toric trim of
        // edge (3,5). Same key → identical interned ids, no new points.
        let a = reg.intern_corner(0, 1, [3, 5], &centers, probe, &plane);
        let b = reg.intern_corner(0, 1, [5, 3], &centers, probe, &plane);
        assert_eq!(a.len(), 2);
        assert_eq!(a, b, "same corner key → same ids for both consumers");
        assert_eq!(reg.len(), 2, "interned once");
        for &id in &a {
            let x = reg.point(id);
            assert!((x.distance(centers[0]) - probe).abs() < 1e-9, "on sphere 0");
            assert!(
                (x.distance(centers[1]) - probe).abs() < 1e-9,
                "on sphere 1 (C_01)"
            );
            assert!(x.dot(plane.normal).abs() < 1e-9, "on the great plane");
        }
        // A different edge is a different event (distinct key).
        let other = reg.intern_corner(0, 1, [3, 9], &centers, probe, &plane);
        assert_eq!(reg.len(), 4);
        assert!(other.iter().all(|x| !a.contains(x)));
    }

    #[test]
    fn the_singular_seam_is_shared_identically_by_both_faces() {
        // The collision seam between probes i and j must be the SAME canonical
        // geometry whichever probe owns the face — the watertight weld of stage 3.
        let probe = 1.4;
        // 0,1 collide; a single third probe buries one stretch, leaving an arc.
        let centers = [ctr(0.0, 0.0, 0.0), ctr(2.0, 0.0, 0.0), ctr(1.0, 0.9, 0.3)];
        let mut r0 = SingularVertices::new();
        let mut r1 = SingularVertices::new();
        let e01 = singular_edges(0, 1, &centers, probe, &mut r0);
        let e10 = singular_edges(1, 0, &centers, probe, &mut r1);
        // Symmetric seam: identical edge intervals regardless of pair order.
        assert_eq!(e01, e10, "singular edges are symmetric in (i,j)");
        assert!(!e01.is_empty());
        for &edge in &e01 {
            let s0 = sample_singular_edge(0, 1, &centers, probe, edge, 12);
            let s1 = sample_singular_edge(1, 0, &centers, probe, edge, 12);
            assert_eq!(s0, s1, "bit-identical samples → weldable seam");
            // Every sample lies on both probe spheres (it is on C_01).
            for x in &s0 {
                assert!((x.distance(centers[0]) - probe).abs() < 1e-9);
                assert!((x.distance(centers[1]) - probe).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn near_degenerate_thin_triple_keeps_two_distinct_branches() {
        // A thin (nearly collinear) triple with circumradius < probe: the two
        // branch vertices are well separated in space, but the plane normal is
        // tiny so a coordinate-recomputed branch sign could merge them. The
        // construction-order labels must keep them distinct (codex #1).
        let probe = 1.4;
        let centers = [ctr(0.0, 0.0, 0.0), ctr(0.3, 0.0, 0.0), ctr(0.15, 0.01, 0.0)];
        let mut reg = SingularVertices::new();
        let ids = reg.intern_triple(0, 1, 2, &centers, probe);
        assert_eq!(ids.len(), 2, "two branches survive a thin triple");
        assert_eq!(reg.len(), 2, "not merged onto one key");
        assert!(
            reg.point(ids[0]).distance(reg.point(ids[1])) > 1e-3,
            "branches are genuinely distinct points"
        );
        for &id in &ids {
            for ctr in &centers {
                assert!((reg.point(id).distance(*ctr) - probe).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn two_isolated_probes_give_one_full_singular_circle() {
        // Just two overlapping probes, nothing else: C_ij is wholly exposed → one
        // singular edge spanning the full circle.
        let probe = 1.4;
        let centers = [ctr(0.0, 0.0, 0.0), ctr(2.0, 0.0, 0.0)];
        let mut reg = SingularVertices::new();
        let edges = singular_edges(0, 1, &centers, probe, &mut reg);
        assert_eq!(edges.len(), 1);
        assert!((edges[0].theta_end - edges[0].theta_start - TAU).abs() < 1e-9);
    }

    #[test]
    fn a_third_probe_splits_the_circle_and_buries_part_of_it() {
        let probe = 1.4;
        // 0,1 collide; probe 2 sits so its ball buries one stretch of C_01.
        let centers = [
            ctr(0.0, 0.0, 0.0),
            ctr(2.0, 0.0, 0.0),
            ctr(1.0, 0.9, 0.0), // near the +y side of C_01
        ];
        let mut reg = SingularVertices::new();
        let edges = singular_edges(0, 1, &centers, probe, &mut reg);
        // Some — but not all — of the circle survives.
        assert!(!edges.is_empty(), "part of C_01 stays exposed");
        let total: f64 = edges.iter().map(|e| e.theta_end - e.theta_start).sum();
        assert!(total < TAU - 1e-3, "probe 2 buried a stretch");
        // Every surviving edge midpoint is genuinely exposed and lies on C_01.
        let c = canonical_burial_circle(0, 1, &centers, probe).unwrap();
        let (u, v) = plane_basis(c.normal);
        for e in &edges {
            let mid = (e.theta_start + e.theta_end) / 2.0;
            let x = c.center + (u * mid.cos() + v * mid.sin()) * c.radius;
            assert!((x.distance(centers[0]) - probe).abs() < 1e-9);
            assert!((x.distance(centers[1]) - probe).abs() < 1e-9);
            assert!(
                x.distance(centers[2]) >= probe - 1e-6,
                "exposed (outside probe 2)"
            );
        }
    }
}
