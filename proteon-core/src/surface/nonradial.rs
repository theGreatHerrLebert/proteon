//! Nonradial singularity resolution: distinct-probe collisions.
//!
//! The dominant SES self-intersection on real proteins (≈1575 vs 21 spindle on
//! crambin) is *nonradial* — two **different** fixed probes closer than `2·probe`,
//! so part of one probe's reentrant (spheric) face lies buried inside the other.
//! `rs::probe_clear` rejects only atom overlap, never probe overlap, so these
//! survive RS enumeration and double-cover the reentrant surface.
//!
//! Resolution: each spheric face is a region of its probe sphere; a neighbouring
//! probe `q` buries the spherical **cap** of directions pointing toward `q`,
//! bounded by the circle where the two equal-radius probe spheres meet. The kept
//! reentrant region is the spheric triangle minus the union of these burial caps;
//! we feed the caps into the same azimuthal chart / CDT that already meshes the
//! triangle. This module holds the cap geometry; the arrangement against the
//! triangle boundary builds on it.

use super::arrangement::SphereCircle;
use super::geom::{plane_basis, Circle3, Vec3};

/// The cap of probe-sphere `p` that is buried inside neighbouring probe `q` (both
/// radius `probe`). Directions from `p` pointing toward `q` within the returned
/// half-angle are inside `q` (i.e. `is_buried` against this cap). `None` if the
/// probes do not overlap (`|p−q| ≥ 2·probe`) or are coincident.
///
/// Equal radii make `None` unambiguous: *full* burial would need `d + probe <
/// probe` ⇒ `d < 0`, which is impossible — two equal probes can only ever cut a
/// partial cap, never swallow each other. So there is no "fully buried" case to
/// distinguish; `None` always means "no cap to subtract." Coincident duplicate
/// placements (`d ≈ 0`) carry only a zero-measure burial and are an RS-dedup
/// concern handled upstream, not here.
pub fn probe_burial_cap(p: Vec3, q: Vec3, probe: f64) -> Option<SphereCircle> {
    let off = q - p;
    let d = off.norm();
    if d >= 2.0 * probe || d < 1e-9 {
        return None;
    }
    // Equal radii: the intersection circle sits at the midplane, so a point on
    // p's sphere at angle θ from (q−p) is inside q iff cos θ > d/(2·probe).
    let half = (d / (2.0 * probe)).clamp(-1.0, 1.0).acos();
    Some(SphereCircle::new(off.normalized()?, half))
}

/// The collision circle where two equal-radius probe spheres meet, in a form that
/// is **identical regardless of which probe asks** — the shared-seam foundation
/// for watertightness (codex #5/#7). Keyed by the unordered probe pair `{i, j}`:
/// the normal always points from the lower-index probe to the higher, the centre
/// is their midpoint (equal radii), so both faces that border this seam derive
/// bit-identical geometry and a bit-identical [`plane_basis`]. `None` if the
/// probes do not overlap.
///
/// SEAM SCOPE (worked out from the geometry): the part of `C` that is a *shared*
/// seam between faces `i` and `j` is only the arc lying inside **both** spheric
/// triangles and exposed w.r.t. all other spheres. Where `C` is inside `i`'s
/// triangle but not `j`'s, that stretch is bounded by `i`'s own toric great
/// circles, not by `j` — it is not a seam with `j`. The integration must sample
/// the shared stretch from this canonical circle and the non-shared stretches
/// from each face's own arrangement.
pub fn canonical_burial_circle(
    i: usize,
    j: usize,
    centers: &[Vec3],
    probe: f64,
) -> Option<Circle3> {
    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
    let (a, b) = (centers[lo], centers[hi]);
    let off = b - a;
    let d = off.norm();
    if d >= 2.0 * probe || d < 1e-9 {
        return None;
    }
    let normal = off.normalized()?; // canonical: lower index → higher
    let half = d * 0.5;
    Some(Circle3 {
        center: (a + b) * 0.5,
        normal,
        radius: (probe * probe - half * half).max(0.0).sqrt(),
    })
}

/// The 0, 1, or 2 points equidistant `r` from three sphere centres `a, b, c` —
/// the **triple-probe SES vertices** where singular edges meet (BALL caches these
/// by sorted face triple). The two solutions are the **branches**, symmetric
/// about the plane `(a,b,c)`; `branch_sign` tags them by the side of that plane's
/// canonical normal `(b−a)×(c−a)`. Empty if the centres are collinear or the
/// three spheres share no common point (`r` too small for the circumradius).
///
/// To be canonical (frame-independent), pass `a, b, c` in a fixed order — e.g.
/// sorted by probe index — so the normal, and hence the branch labelling, is
/// deterministic for the triple.
pub fn triple_sphere_intersections(a: Vec3, b: Vec3, c: Vec3, r: f64) -> Vec<Vec3> {
    let u = b - a;
    let v = c - a;
    let uxv = u.cross(v);
    let n2 = uxv.norm_sq();
    if n2 < 1e-18 {
        return Vec::new(); // collinear centres
    }
    // Circumcentre of (a,b,c): o − a = ((|u|²v − |v|²u) × (u×v)) / (2|u×v|²).
    let o = a + (v * u.norm_sq() - u * v.norm_sq()).cross(uxv) * (1.0 / (2.0 * n2));
    let Some(nh) = uxv.normalized() else {
        return Vec::new();
    };
    // The equidistant line is o + t·nh (nh ⊥ plane, so ⊥ (o−a)); |x−a|=r ⇒
    // |o−a|² + t² = r².
    let disc = r * r - (o - a).norm_sq();
    if disc < -1e-12 {
        return Vec::new(); // no common point
    }
    let t = disc.max(0.0).sqrt();
    if t < 1e-9 {
        return vec![o]; // tangent: single point on the plane
    }
    vec![o + nh * t, o - nh * t] // branch + then −
}

/// Which side of the canonical plane `(a,b,c)` a triple-vertex `x` is on: `+1` or
/// `-1`. Stable branch label for the triple-sphere vertex registry key.
pub fn branch_sign(x: Vec3, a: Vec3, b: Vec3, c: Vec3) -> i8 {
    let n = (b - a).cross(c - a);
    if (x - a).dot(n) >= 0.0 {
        1
    } else {
        -1
    }
}

/// Sample a circle's rim at the given `thetas` using its canonical
/// [`plane_basis`] — a pure function of the circle, so two callers that pass the
/// same [`Circle3`] and the same `thetas` get bit-identical world points (the
/// weld guarantee).
pub fn sample_circle_rim(c: &Circle3, thetas: &[f64]) -> Vec<Vec3> {
    let (u, v) = plane_basis(c.normal);
    thetas
        .iter()
        .map(|&t| c.center + (u * t.cos() + v * t.sin()) * c.radius)
        .collect()
}

/// The three great-circle caps bounding a spheric (reentrant) face, given the
/// probe centre `p` and the three contact-point directions `dirs` (unit, from `p`
/// toward each atom's contact). Each boundary arc between two contacts is a
/// great-circle arc (its plane passes through `p`, the sphere centre), so the
/// spheric triangle is `exposed({these three caps})`. Each cap's axis points
/// **away** from the triangle (the third contact stays exposed), so burial caps
/// can be unioned into the same arrangement. `None` if any pair is degenerate.
pub fn spheric_face_caps(dirs: [Vec3; 3]) -> Option<[SphereCircle; 3]> {
    let mut caps = [SphereCircle::new(Vec3::new(1.0, 0.0, 0.0), 0.0); 3];
    for e in 0..3 {
        let (x, y, k) = (e, (e + 1) % 3, (e + 2) % 3);
        let n = dirs[x].cross(dirs[y]).normalized()?;
        // Orient so the third contact is on the exposed side (d·axis < 0): a
        // great circle is half_angle π/2, so buried ⇔ d·axis > 0. If the third
        // contact lies on (or numerically near) this edge's great circle the
        // hemisphere choice is undefined — a degenerate (near-coplanar) triple
        // with no well-formed spheric face; reject rather than guess (codex).
        let s = dirs[k].dot(n);
        if s.abs() < 1e-9 {
            return None;
        }
        let n = if s > 0.0 { n * -1.0 } else { n };
        caps[e] = SphereCircle::new(n, std::f64::consts::FRAC_PI_2);
    }
    Some(caps)
}

/// All probes (besides `self_idx`) whose burial cap cuts probe `self_idx`'s
/// sphere, paired with the cap. The broad-phase filter is `|p−q| < 2·probe`.
pub fn burial_caps(self_idx: usize, centers: &[Vec3], probe: f64) -> Vec<(usize, SphereCircle)> {
    let p = centers[self_idx];
    centers
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != self_idx)
        .filter_map(|(k, &q)| probe_burial_cap(p, q, probe).map(|c| (k, c)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::chart::fill_spherical_region;
    use super::*;
    use std::f64::consts::{PI, TAU};

    #[test]
    fn canonical_burial_circle_is_identical_from_either_probe() {
        let probe = 1.4;
        let centers = [Vec3::new(0.3, -0.2, 1.1), Vec3::new(2.0, 0.5, 0.4)];
        let from_0 = canonical_burial_circle(0, 1, &centers, probe).expect("overlap");
        let from_1 = canonical_burial_circle(1, 0, &centers, probe).expect("overlap");
        // Bit-identical circle regardless of which probe asks.
        assert_eq!(from_0, from_1);
        // And the rim it bounds lies exactly `probe` from BOTH probe centres
        // (it is sphere_0 ∩ sphere_1).
        let thetas: Vec<f64> = (0..64).map(|k| TAU * k as f64 / 64.0).collect();
        let rim_a = sample_circle_rim(&from_0, &thetas);
        let rim_b = sample_circle_rim(&from_1, &thetas);
        assert_eq!(rim_a, rim_b, "bit-identical samples → weldable seam");
        for x in &rim_a {
            assert!((x.distance(centers[0]) - probe).abs() < 1e-12);
            assert!((x.distance(centers[1]) - probe).abs() < 1e-12);
        }
    }

    #[test]
    fn triple_sphere_vertices_are_equidistant_and_branch_symmetric() {
        let r = 1.4;
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.6, 0.0, 0.0);
        let c = Vec3::new(0.5, 1.5, 0.0);
        let pts = triple_sphere_intersections(a, b, c, r);
        assert_eq!(pts.len(), 2, "generic triple → two branches");
        for x in &pts {
            for ctr in [a, b, c] {
                assert!((x.distance(ctr) - r).abs() < 1e-12, "equidistant r");
            }
        }
        // Branches are on opposite sides of plane(a,b,c) and mirror images.
        assert_eq!(branch_sign(pts[0], a, b, c), 1);
        assert_eq!(branch_sign(pts[1], a, b, c), -1);
        // Canonical: same triple in the same order → identical points (the weld
        // guarantee for triple vertices).
        let again = triple_sphere_intersections(a, b, c, r);
        assert_eq!(pts, again);
    }

    #[test]
    fn no_triple_vertex_when_spheres_too_far_or_collinear() {
        let r = 1.4;
        // Far apart: circumradius > r → no common point.
        let far = triple_sphere_intersections(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(1.25, 2.5, 0.0),
            r,
        );
        assert!(far.is_empty());
        // Collinear centres → degenerate.
        let line = triple_sphere_intersections(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            r,
        );
        assert!(line.is_empty());
    }

    #[test]
    fn no_canonical_circle_when_probes_disjoint() {
        let centers = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)];
        assert!(canonical_burial_circle(0, 1, &centers, 1.4).is_none());
    }

    #[test]
    fn no_cap_when_probes_too_far() {
        assert!(
            probe_burial_cap(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), 1.4).is_none()
        );
    }

    #[test]
    fn spheric_face_via_caps_matches_the_arc_based_triangle() {
        use super::super::arrangement::{boundary_loops, sample_loop};
        use super::super::elements::arc_on_sphere;
        let p = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.4;
        let dirs = [
            Vec3::new(0.3, 0.0, 1.0).normalized().unwrap(),
            Vec3::new(-0.15, 0.26, 1.0).normalized().unwrap(),
            Vec3::new(-0.15, -0.26, 1.0).normalized().unwrap(),
        ];
        let cs = dirs.map(|d| p + d * r);
        let pole = ((dirs[0] + dirs[1] + dirs[2]) * (1.0 / 3.0))
            .normalized()
            .unwrap();

        // Arc-based (current assemble.rs path): great-circle arcs between contacts.
        let mut arc_loop = Vec::new();
        for e in 0..3 {
            let (x, y) = (e, (e + 1) % 3);
            arc_loop.push(cs[x]);
            arc_loop.extend(arc_on_sphere(p, r, cs[x], cs[y], 24));
        }
        let arc_area = fill_spherical_region(p, r, &[arc_loop], pole, 0.1)
            .unwrap()
            .surface_area();

        // Cap-arrangement: the same triangle as exposed({3 great circles}).
        let caps = spheric_face_caps(dirs).unwrap();
        let loops = boundary_loops(&caps).unwrap();
        let cap_loops: Vec<Vec<Vec3>> = loops
            .iter()
            .map(|lp| {
                sample_loop(lp, &caps, 24)
                    .into_iter()
                    .map(|d| p + d * r)
                    .collect()
            })
            .collect();
        let cap_area = fill_spherical_region(p, r, &cap_loops, pole, 0.1)
            .unwrap()
            .surface_area();

        assert!(
            (arc_area - cap_area).abs() / arc_area < 0.01,
            "cap-arrangement spheric area {cap_area} within 1% of arc-based {arc_area}"
        );
    }

    #[test]
    fn one_burial_cap_cuts_a_spherical_cap_of_the_right_area() {
        let probe = 1.4;
        let p = Vec3::new(0.0, 0.0, 0.0);
        let q = Vec3::new(2.0, 0.0, 0.0);
        let cap = probe_burial_cap(p, q, probe).expect("overlap");
        // Expected: cos(half) = d/(2·probe).
        assert!((cap.half_angle.cos() - 2.0 / (2.0 * probe)).abs() < 1e-12);

        // Mesh the *kept* region = sphere-p minus the cap toward q, bounded by the
        // burial circle, pole away from q. Area must match the closed-form big cap
        // 2π·probe²·(1+cos half).
        let rim: Vec<Vec3> = (0..200)
            .map(|i| {
                let th = 2.0 * PI * i as f64 / 200.0;
                p + cap.rim_point(th) * probe
            })
            .collect();
        let pole = (p - q).normalized().unwrap(); // -x, inside the kept region
        let m = fill_spherical_region(p, probe, &[rim], pole, 0.15).expect("fill");
        let exact = 2.0 * PI * probe * probe * (1.0 + cap.half_angle.cos());
        let got = m.surface_area();
        assert!(
            (got - exact).abs() / exact < 0.01,
            "kept area {got} within 1% of {exact}"
        );
    }
}
