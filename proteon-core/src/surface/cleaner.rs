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

use super::geom::{Sphere, Vec3};
use super::nonradial::{branch_sign, triple_sphere_intersections};
use std::collections::HashMap;

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

    /// Intern the triple-probe vertices of probes `(i, j, k)` (their probe
    /// spheres all of radius `probe`), returning their registry ids. Canonical:
    /// the triple is sorted before computing geometry, so `intern_triple` is
    /// invariant to the argument order — `(i,j,k)` and `(k,j,i)` give the same
    /// ids and the same coordinates. Empty if the three spheres share no point.
    pub fn intern_triple(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        probes: &[Sphere],
        probe: f64,
    ) -> Vec<usize> {
        let mut tri = [i, j, k];
        tri.sort_unstable();
        if tri[0] == tri[1] || tri[1] == tri[2] {
            return Vec::new(); // not three distinct probes
        }
        let (a, b, c) = (
            probes[tri[0]].center,
            probes[tri[1]].center,
            probes[tri[2]].center,
        );
        let mut ids = Vec::new();
        for x in triple_sphere_intersections(a, b, c, probe) {
            let key = (tri, branch_sign(x, a, b, c));
            let id = *self.index.entry(key).or_insert_with(|| {
                self.points.push(x);
                self.points.len() - 1
            });
            ids.push(id);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sph(x: f64, y: f64, z: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), 1.4)
    }

    #[test]
    fn interning_is_permutation_invariant_and_deduplicated() {
        let probe = 1.4;
        let probes = [sph(0.0, 0.0, 0.0), sph(1.6, 0.0, 0.0), sph(0.5, 1.5, 0.0)];
        let mut reg = SingularVertices::new();
        let a = reg.intern_triple(0, 1, 2, &probes, probe);
        assert_eq!(a.len(), 2, "generic triple → two branch vertices");
        // Any permutation interns the SAME ids (no new points).
        let b = reg.intern_triple(2, 0, 1, &probes, probe);
        let c = reg.intern_triple(1, 2, 0, &probes, probe);
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
            for p in &probes {
                assert!((reg.point(id).distance(p.center) - probe).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distinct_triples_get_distinct_entries_and_no_phantom_points() {
        let probe = 1.4;
        let probes = [
            sph(0.0, 0.0, 0.0),
            sph(1.6, 0.0, 0.0),
            sph(0.5, 1.5, 0.0),
            sph(0.8, 0.5, 1.4),
        ];
        let mut reg = SingularVertices::new();
        let t012 = reg.intern_triple(0, 1, 2, &probes, probe);
        let t013 = reg.intern_triple(0, 1, 3, &probes, probe);
        // Different triples → different vertices (no accidental merge).
        for x in &t012 {
            assert!(!t013.contains(x));
        }
        assert_eq!(reg.len(), t012.len() + t013.len());
    }
}
