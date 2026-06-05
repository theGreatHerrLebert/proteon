//! Roll-circle free intervals — the analytic authority for toric faces.
//!
//! A probe rolling on the pair `(i,j)` rides the roll circle
//! `(i⊕probe) ∩ (j⊕probe)`. Each *other* atom `k` blocks a θ-arc of that circle
//! (where the probe would overlap `k`); the **free** intervals — the complement of
//! the union of all blocked arcs — are exactly the bounded toric faces (or one
//! full ring if nothing blocks). Endpoints are RS faces (the probe touching a
//! third atom) = SES vertices.
//!
//! This is computed analytically (the clearance `|P(θ)−c_k|² − (r_k+probe)²` is a
//! constant plus one sinusoid, so each blocker contributes at most one arc), which
//! per the general-N plan replaces the sampled RS-edge detection that can miss
//! narrow arcs.

use super::geom::{plane_basis, Circle3, Sphere};
use std::f64::consts::TAU;

/// Angular tolerance (rad) for the circular interval union/complement — merging
/// near-coincident endpoints and suppressing measure-zero arcs. Distinct from any
/// length tolerance (codex-review: don't overload one geometric epsilon across
/// dimensions). The `t ≥ 1` (Full) / sub-tolerance-arc suppression below make this
/// a deliberately **non-singular-regularized** complement, not the exact one — a
/// single tangent point or a vanishing arc is dropped, consistent with the
/// non-singular SES contract.
const ANG_EPS: f64 = 1e-9;
/// Length tolerance (Å) for the degenerate "roll centre on the blocker axis" case.
const LEN_EPS: f64 = 1e-9;

/// The θ-arc of `roll` blocked by `blocker` (probe overlaps it).
enum Blocked {
    None,
    Full,
    /// `(start, end)` with `start ∈ [0, TAU)` and `end = start + width` (≤ 2·TAU).
    Arc(f64, f64),
}

fn blocked_arc(roll: &Circle3, blocker: Sphere, probe: f64) -> Blocked {
    let (u, v) = plane_basis(roll.normal);
    let d = roll.center - blocker.center;
    let (a, b) = (d.dot(u), d.dot(v));
    let c = (a * a + b * b).sqrt();
    let rhs = blocker.radius + probe;
    // |P(θ)−c_k|² = |d|² + R² + 2R·c·cos(θ−φ);  blocked where that < rhs².
    let base = d.norm_sq() + roll.radius * roll.radius;
    if c < LEN_EPS {
        // P is equidistant from the blocker for every θ.
        return if base < rhs * rhs {
            Blocked::Full
        } else {
            Blocked::None
        };
    }
    let t = (rhs * rhs - base) / (2.0 * roll.radius * c); // cos(θ−φ) < t ⇒ blocked
    if t >= 1.0 {
        Blocked::Full
    } else if t <= -1.0 {
        Blocked::None
    } else {
        let phi = b.atan2(a);
        let ac = t.acos();
        // cos(θ−φ) < t for θ−φ ∈ (ac, TAU−ac): arc start φ+ac, width TAU−2ac.
        let start = (phi + ac).rem_euclid(TAU);
        Blocked::Arc(start, start + (TAU - 2.0 * ac))
    }
}

/// The free θ-intervals of `roll` (the bounded toric faces) — the complement of
/// the union of every blocker's arc. `[(0, TAU)]` ⇒ a full free ring; `[]` ⇒ the
/// pair is fully buried (no toric face). A wrapping free interval is returned as a
/// single `(start, end)` with `end > TAU`.
pub fn free_intervals(roll: &Circle3, blockers: &[Sphere], probe: f64) -> Vec<(f64, f64)> {
    // Collect blocked arcs as non-wrapping pieces in [0, TAU).
    let mut blocked: Vec<(f64, f64)> = Vec::new();
    for &b in blockers {
        match blocked_arc(roll, b, probe) {
            Blocked::Full => return Vec::new(),
            Blocked::None => {}
            Blocked::Arc(s, e) => {
                if e <= TAU + ANG_EPS {
                    blocked.push((s, e.min(TAU)));
                } else {
                    blocked.push((s, TAU));
                    blocked.push((0.0, e - TAU));
                }
            }
        }
    }
    if blocked.is_empty() {
        return vec![(0.0, TAU)];
    }

    // Union on [0, TAU).
    blocked.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (s, e) in blocked {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 + ANG_EPS {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }

    // Free = gaps.
    let mut free: Vec<(f64, f64)> = Vec::new();
    let mut cursor = 0.0;
    for (s, e) in &merged {
        if s - cursor > ANG_EPS {
            free.push((cursor, *s));
        }
        cursor = *e;
    }
    if TAU - cursor > ANG_EPS {
        free.push((cursor, TAU));
    }
    // Merge the wrap seam: a free piece ending at TAU joins one starting at 0.
    if free.len() >= 2 && free[0].0 < ANG_EPS && (free.last().unwrap().1 - TAU).abs() < ANG_EPS {
        let (ls, _) = free.pop().unwrap();
        let e0 = free[0].1;
        free[0] = (ls, e0 + TAU); // wrapping interval
    }
    free
}

#[cfg(test)]
mod tests {
    use super::super::geom::{intersect_two_spheres, Vec3};
    use super::*;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    fn roll(a: Sphere, b: Sphere, probe: f64) -> Circle3 {
        intersect_two_spheres(a.inflated(probe), b.inflated(probe)).unwrap()
    }

    #[test]
    fn no_blockers_gives_a_full_free_ring() {
        let a = sph(0.0, 0.0, 0.0, 1.8);
        let b = sph(2.5, 0.0, 0.0, 1.8);
        let f = free_intervals(&roll(a, b, 1.4), &[], 1.4);
        assert_eq!(f, vec![(0.0, TAU)]);
    }

    #[test]
    fn one_blocker_leaves_one_free_interval() {
        // triangle3 geometry: pair (0,1) blocked by atom 2 → exactly one free arc.
        let a = sph(0.0, 0.0, 0.0, 1.7);
        let b = sph(2.5, 0.0, 0.0, 1.7);
        let c = sph(1.25, 2.165, 0.0, 1.7);
        let probe = 1.4;
        let f = free_intervals(&roll(a, b, probe), &[c], probe);
        assert_eq!(f.len(), 1, "one blocker → one free interval, got {f:?}");
        let (s, e) = f[0];
        assert!(e > s && e - s < TAU, "a proper sub-arc, not the full ring");
        // Its midpoint probe clears c; a point in the blocked complement does not.
        let mid = midpoint_probe(&roll(a, b, probe), s, e);
        assert!(
            mid.distance(c.center) >= c.radius + probe - 1e-9,
            "free mid clears c"
        );
    }

    #[test]
    fn two_opposite_blockers_give_two_free_intervals() {
        // A pair blocked from both sides by different atoms → two disjoint free
        // arcs (the disjoint-interval case general-N must handle).
        let a = sph(0.0, 0.0, 0.0, 1.7);
        let b = sph(2.5, 0.0, 0.0, 1.7);
        let c1 = sph(1.25, 2.0, 0.0, 1.7);
        let c2 = sph(1.25, -2.0, 0.0, 1.7);
        let probe = 1.4;
        let f = free_intervals(&roll(a, b, probe), &[c1, c2], probe);
        assert_eq!(
            f.len(),
            2,
            "two opposite blockers → two free intervals, got {f:?}"
        );
    }

    fn midpoint_probe(roll: &Circle3, s: f64, e: f64) -> Vec3 {
        let (u, v) = plane_basis(roll.normal);
        let m = 0.5 * (s + e);
        roll.center + (u * m.cos() + v * m.sin()) * roll.radius
    }
}
