//! Spherical-circle arrangement — the exposed region of an atom sphere given its
//! incident contact circles.
//!
//! Per the stitching design (`TO_SES_STITCHING.md`), a contact cap is **not** the
//! sphere minus independent holes: the incident contact circles overlap, nest,
//! and touch, so the exposed region is the part of the sphere outside the
//! **union of buried caps**. This module is that arrangement, working in
//! unit-direction space about the atom center (scale-invariant):
//!
//! - [`SphereCircle`] — one contact circle as the *buried cap* it bounds
//!   (axis toward the neighbour atom + angular radius).
//! - [`circle_intersections`] — where two contact circles cross on the sphere.
//! - [`is_buried`] / [`is_exposed`] — pointwise classification.
//! - [`exposed_arcs`] — the θ-intervals of one circle's rim that survive (are
//!   not inside any other buried cap) — the boundary loops the contact-cap mesh
//!   follows and shares with the toric faces.

use super::geom::{intersect_planes, plane_basis, solve_quadratic, Plane3, Vec3};
use std::f64::consts::TAU;

const EPS: f64 = 1e-9;

/// A contact circle on the unit sphere, described by the **buried cap** it bounds:
/// directions within `half_angle` of `axis` (toward the neighbour atom) are
/// buried; the rim is at exactly `half_angle`.
#[derive(Clone, Copy, Debug)]
pub struct SphereCircle {
    pub axis: Vec3,
    pub half_angle: f64,
}

impl SphereCircle {
    pub fn new(axis: Vec3, half_angle: f64) -> Self {
        Self {
            axis: axis.normalized().expect("axis must be nonzero"),
            half_angle,
        }
    }

    fn cos_half(&self) -> f64 {
        self.half_angle.cos()
    }

    /// Unit direction on this circle's rim at parameter `theta`, in the circle's
    /// own `plane_basis`.
    pub fn rim_point(&self, theta: f64) -> Vec3 {
        let (u, v) = plane_basis(self.axis);
        self.axis * self.half_angle.cos()
            + (u * theta.cos() + v * theta.sin()) * self.half_angle.sin()
    }
}

/// Is unit direction `d` buried by `cap` (strictly inside its cap)?
pub fn is_buried(d: Vec3, cap: &SphereCircle) -> bool {
    d.dot(cap.axis) > cap.cos_half() + EPS
}

/// Is unit direction `d` exposed — outside every buried cap?
pub fn is_exposed(d: Vec3, caps: &[SphereCircle]) -> bool {
    !caps.iter().any(|c| is_buried(d, c))
}

/// The (0, 1, or 2) unit directions where two contact circles cross. A point `p`
/// is on both rims iff `p·a.axis = cos αa` and `p·b.axis = cos αb` with `|p|=1` —
/// two planes intersected with the unit sphere.
pub fn circle_intersections(a: SphereCircle, b: SphereCircle) -> Vec<Vec3> {
    // Plane convention n·x + d = 0 ⇒ d = -cos(half_angle) for `n·x = cos α`.
    let pa = Plane3 {
        normal: a.axis,
        d: -a.cos_half(),
    };
    let pb = Plane3 {
        normal: b.axis,
        d: -b.cos_half(),
    };
    let Some(line) = intersect_planes(pa, pb) else {
        return Vec::new();
    };
    // |line.p + t·dir|² = 1
    let Some((t1, t2)) = solve_quadratic(
        line.dir.dot(line.dir),
        2.0 * line.p.dot(line.dir),
        line.p.dot(line.p) - 1.0,
    ) else {
        return Vec::new();
    };
    // Mathematically unit, but renormalize against accumulated round-off so
    // downstream angle/burial tests see exact directions.
    let pt = |t: f64| {
        (line.p + line.dir * t)
            .normalized()
            .unwrap_or(line.p + line.dir * t)
    };
    if (t1 - t2).abs() < EPS {
        vec![pt(t1)]
    } else {
        vec![pt(t1), pt(t2)]
    }
}

/// The θ-intervals of `circle`'s rim that are **exposed** (not buried by any cap
/// in `others`), as `(start, end)` pairs in `[0, TAU)` with `start < end` (a
/// wrapping interval is split at 0). Empty ⇒ the rim is fully buried. One
/// interval `[0, TAU)` ⇒ fully exposed.
pub fn exposed_arcs(circle: &SphereCircle, others: &[SphereCircle]) -> Vec<(f64, f64)> {
    // One basis for the whole rim — `theta_of` must be the inverse of
    // `rim_point`, so both use exactly these (u, v) (plane_basis is a
    // deterministic pure function of the axis, but pinning it here removes the
    // dependence on that and keeps the round-trip exact).
    let (u, v) = plane_basis(circle.axis);
    let theta_of = |p: Vec3| -> f64 {
        let t = p.dot(v).atan2(p.dot(u));
        if t < 0.0 {
            t + TAU
        } else {
            t
        }
    };

    // Collect the θ-intervals removed by each other cap.
    let mut removed: Vec<(f64, f64)> = Vec::new();
    let alpha = circle.half_angle;
    for o in others {
        // Classify the pair by the angle δ between axes vs the two angular
        // radii — robust, unlike sampling a single rim point. A rim point sits
        // at angular distance in [δ−α, δ+α] from o.axis; o's cap reaches to β.
        let beta = o.half_angle;
        let delta = circle.axis.dot(o.axis).clamp(-1.0, 1.0).acos();
        if delta >= alpha + beta - EPS {
            continue; // disjoint: nearest rim point still outside o's cap
        }
        if delta + alpha <= beta + EPS {
            return Vec::new(); // farthest rim point still inside o → fully buried
        }
        // Crossing (|α−β| < δ < α+β): two genuine intersections bound the
        // buried arc. If round-off failed to produce both, the buried arc is a
        // numerically negligible sliver — skip it rather than guess from one
        // sample.
        let pts = circle_intersections(*circle, *o);
        let [p1, p2] = pts.as_slice() else { continue };
        let (mut a, mut b) = (theta_of(*p1), theta_of(*p2));
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        // One of the two arcs (a,b) / (b,a+TAU) is the buried one; pick it by
        // testing the midpoint.
        let mid = (a + b) / 2.0;
        if is_buried(circle.rim_point(mid), o) {
            removed.push((a, b));
        } else {
            // The complementary (wrapping) arc is buried.
            removed.push((b, TAU));
            removed.push((0.0, a));
        }
    }
    complement_intervals(removed)
}

/// `[0, TAU)` minus the union of `removed` intervals.
fn complement_intervals(mut removed: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    removed.retain(|&(a, b)| b - a > EPS);
    if removed.is_empty() {
        return vec![(0.0, TAU)];
    }
    removed.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    // merge overlaps
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in removed {
        if let Some(last) = merged.last_mut() {
            if a <= last.1 + EPS {
                last.1 = last.1.max(b);
                continue;
            }
        }
        merged.push((a, b));
    }
    // gaps between merged removed intervals
    let mut out = Vec::new();
    let mut cursor = 0.0;
    for (a, b) in &merged {
        if a - cursor > EPS {
            out.push((cursor, *a));
        }
        cursor = *b;
    }
    if TAU - cursor > EPS {
        out.push((cursor, TAU));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circ(ax: Vec3, alpha: f64) -> SphereCircle {
        SphereCircle::new(ax, alpha)
    }

    #[test]
    fn two_circles_cross_at_two_points_on_both_rims() {
        let a = circ(Vec3::new(0.0, 0.0, 1.0), 0.9);
        let b = circ(Vec3::new(1.0, 0.0, 0.0), 0.9);
        let pts = circle_intersections(a, b);
        assert_eq!(pts.len(), 2);
        for p in pts {
            assert!((p.norm() - 1.0).abs() < 1e-9, "on the sphere");
            assert!((p.dot(a.axis) - a.cos_half()).abs() < 1e-9, "on rim a");
            assert!((p.dot(b.axis) - b.cos_half()).abs() < 1e-9, "on rim b");
        }
    }

    #[test]
    fn far_apart_caps_dont_cross() {
        let a = circ(Vec3::new(0.0, 0.0, 1.0), 0.3);
        let b = circ(Vec3::new(0.0, 0.0, -1.0), 0.3); // antipodal small caps
        assert!(circle_intersections(a, b).is_empty());
    }

    #[test]
    fn exposure_classification() {
        let cap = circ(Vec3::new(0.0, 0.0, 1.0), 0.5);
        assert!(is_buried(Vec3::new(0.0, 0.0, 1.0), &cap)); // along axis → inside
        assert!(is_exposed(Vec3::new(0.0, 0.0, -1.0), &[cap])); // antipode → outside
    }

    #[test]
    fn exposed_arcs_agree_with_pointwise_classifier() {
        // A circle's surviving rim arcs must contain exactly the rim points that
        // are not buried by any other cap — the cross-check that validates the
        // interval arithmetic independent of parameterization.
        let circle = circ(Vec3::new(0.0, 0.0, 1.0), 1.2);
        let others = [
            circ(Vec3::new(1.0, 0.0, 0.2), 0.8),
            circ(Vec3::new(-0.3, 1.0, 0.4), 0.9),
        ];
        let arcs = exposed_arcs(&circle, &others);
        let in_arcs = |t: f64| arcs.iter().any(|&(a, b)| t >= a - 1e-9 && t <= b + 1e-9);
        for k in 0..2000 {
            let t = TAU * k as f64 / 2000.0;
            let exposed = is_exposed(circle.rim_point(t), &others);
            assert_eq!(in_arcs(t), exposed, "arc/classifier disagree at θ={t}");
        }
    }

    #[test]
    fn fully_buried_and_fully_exposed_rims() {
        let circle = circ(Vec3::new(0.0, 0.0, 1.0), 0.3); // small cap near +z
                                                          // A big cap covering +z buries the whole rim.
        let cover = circ(Vec3::new(0.0, 0.0, 1.0), 1.5);
        assert!(exposed_arcs(&circle, &[cover]).is_empty());
        // A far cap removes nothing → fully exposed.
        let far = circ(Vec3::new(0.0, 0.0, -1.0), 0.3);
        assert_eq!(exposed_arcs(&circle, &[far]), vec![(0.0, TAU)]);
    }

    #[test]
    fn tangent_and_near_tangent_caps_do_not_flip_the_rim() {
        // Two caps that just touch (δ = α + β): the buried arc has zero length,
        // so the rim must survive whole — the angular classifier treats this as
        // disjoint, where the old single-sample test could have flipped it.
        let circle = circ(Vec3::new(0.0, 0.0, 1.0), 0.6);
        let beta = 0.5;
        let delta = circle.half_angle + beta; // exact external tangency
        let other = circ(Vec3::new(delta.sin(), 0.0, delta.cos()), beta);
        assert_eq!(exposed_arcs(&circle, &[other]), vec![(0.0, TAU)]);

        // Nudge to a genuine (tiny) overlap — still a valid arrangement, no panic,
        // and the surviving arcs agree with the pointwise classifier.
        let inside = circ(
            Vec3::new((delta - 0.05).sin(), 0.0, (delta - 0.05).cos()),
            beta,
        );
        let arcs = exposed_arcs(&circle, &[inside]);
        let in_arcs = |t: f64| arcs.iter().any(|&(a, b)| t >= a - 1e-9 && t <= b + 1e-9);
        for k in 0..2000 {
            let t = TAU * k as f64 / 2000.0;
            assert_eq!(in_arcs(t), is_exposed(circle.rim_point(t), &[inside]));
        }
    }

    #[test]
    fn rim_just_inside_a_cap_is_fully_buried() {
        // Concentric caps, the circle's rim a hair inside the other's cap
        // (δ + α < β). The angular test must report fully buried even though no
        // intersection points exist (parallel planes).
        let circle = circ(Vec3::new(0.0, 0.0, 1.0), 0.40);
        let cover = circ(Vec3::new(0.0, 0.0, 1.0), 0.45);
        assert!(exposed_arcs(&circle, &[cover]).is_empty());
    }
}
