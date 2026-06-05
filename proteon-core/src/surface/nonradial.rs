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
use super::geom::Vec3;

/// The cap of probe-sphere `p` that is buried inside neighbouring probe `q` (both
/// radius `probe`). Directions from `p` pointing toward `q` within the returned
/// half-angle are inside `q` (i.e. `is_buried` against this cap). `None` if the
/// probes do not overlap (`|p−q| ≥ 2·probe`) or are coincident (fully buried —
/// the caller drops the whole face).
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
        let mut n = dirs[x].cross(dirs[y]).normalized()?;
        // Orient so the third contact is on the exposed side (d·axis < 0): a
        // great circle is half_angle π/2, so buried ⇔ d·axis > 0.
        if dirs[k].dot(n) > 0.0 {
            n = n * -1.0;
        }
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
    use std::f64::consts::PI;

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
