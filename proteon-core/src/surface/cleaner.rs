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
//! 1. [`SingularVertices`] — the canonical triple-probe vertex registry (this
//!    file): the 0/1/2 points equidistant `probe` from three probe centres,
//!    interned by `(sorted triple, branch)` so every incident edge/face looks up
//!    bit-identical coordinates (the weld guarantee). Three spheres give two
//!    branches, so the branch bit is part of the key (codex Q2).
//! 2. singular edges on each `C_ij` split by those vertices + global exposure.
//! 3. spheric + toric face rewrite onto the singular edges.
//! 4. richer gate vs BALL (volume, Euler, per-face-type area, …).

use super::geom::{plane_basis, Vec3};
use super::nonradial::{canonical_burial_circle, triple_sphere_intersections};
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Canonical registry of triple-probe SES vertices. Keyed by the **sorted** probe
/// triple plus the branch side (`+1`/`-1`), so any permutation of a triple and
/// any incident face resolve the same interned point.
#[derive(Default)]
pub struct SingularVertices {
    points: Vec<Vec3>,
    index: HashMap<([usize; 3], i8), usize>,
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
            let id = *self.index.entry((tri, branch)).or_insert_with(|| {
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
