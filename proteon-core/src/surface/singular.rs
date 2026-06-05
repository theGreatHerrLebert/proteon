//! Probe self-intersection (singular toric faces) — detection + geometry, the
//! foundation of the geometric singularity resolver.
//!
//! A toric face's probe rolls on the roll circle (radius `R_roll`). When
//! `R_roll < probe` the torus is a **spindle** (not a ring): the reentrant
//! surface reaches past the roll axis and self-intersects, so the raw toric patch
//! overlaps itself — the "closed but wrong" excess the assembler currently leaves
//! on dense regions (≈1% on crambin). The two self-intersection points are the
//! **spindle poles** on the axis, at `roll.center ± n·√(probe² − R_roll²)`; every
//! reentrant arc passes through one of them. The resolver clips each arc at its
//! pole so the surface embeds (each point covered once), turning the poles into
//! singular SES vertices.

use super::geom::{Circle3, Vec3};

/// Is this toric face singular (a self-intersecting spindle)? `true` iff the roll
/// circle is smaller than the probe.
pub fn is_singular(roll: &Circle3, probe: f64) -> bool {
    roll.radius < probe
}

/// The two spindle poles of a singular toric face: the axis points at distance
/// exactly `probe` from the whole roll circle, where the reentrant surface
/// self-intersects. `None` for a non-singular (ring) torus.
pub fn spindle_poles(roll: &Circle3, probe: f64) -> Option<(Vec3, Vec3)> {
    if !is_singular(roll, probe) {
        return None;
    }
    let h = (probe * probe - roll.radius * roll.radius).max(0.0).sqrt();
    let n = roll.normal.normalized()?;
    Some((roll.center + n * h, roll.center - n * h))
}

#[cfg(test)]
mod tests {
    use super::super::geom::{intersect_two_spheres, Sphere};
    use super::*;

    fn sph(x: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, 0.0, 0.0), r)
    }

    fn roll(a: Sphere, b: Sphere, probe: f64) -> Circle3 {
        intersect_two_spheres(a.inflated(probe), b.inflated(probe)).unwrap()
    }

    #[test]
    fn close_pair_is_not_singular_far_pair_is() {
        let probe = 1.4;
        // Snug contact: large roll circle → ring torus, not singular.
        let snug = roll(sph(0.0, 1.5), sph(2.5, 1.5), probe);
        assert!(!is_singular(&snug, probe));
        assert!(spindle_poles(&snug, probe).is_none());

        // Wide gap (still sharing a probe): small roll circle → spindle, singular.
        let wide = roll(sph(0.0, 1.5), sph(5.3, 1.5), probe);
        assert!(wide.radius < probe, "roll radius {} < probe", wide.radius);
        assert!(is_singular(&wide, probe));
        let (p0, p1) = spindle_poles(&wide, probe).unwrap();
        // Each pole is exactly `probe` from every point of the roll circle.
        for pole in [p0, p1] {
            let on_circle =
                wide.center + super::super::geom::plane_basis(wide.normal).0 * wide.radius;
            assert!(
                (pole.distance(on_circle) - probe).abs() < 1e-9,
                "pole on reentrant surface"
            );
        }
        // Distinct poles, symmetric about the roll centre.
        assert!(p0.distance(p1) > 1e-6);
        assert!(((p0 + p1) * 0.5).distance(wide.center) < 1e-9);
    }
}
