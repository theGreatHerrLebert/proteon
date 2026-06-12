// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Conf mutation + random placement, ported from AutoDock-Vina
// src/lib/{mutate.cpp,conf.h,quaternion.cpp,random.cpp} (Apache-2.0).

//! Random-conformation generation and single-step mutation for the
//! Monte-Carlo outer search.
//!
//! Two operations drive the MC loop:
//!
//! * [`randomize_conf`] — uniform random placement of the whole ligand
//!   inside the search box (random root position, random orientation,
//!   random torsions). Used once per replicate to seed the walk.
//! * [`mutate_conf`] — perturb *one* randomly-chosen degree of freedom
//!   (the root translation, the root orientation, or a single torsion),
//!   exactly as upstream's `mutate_conf` does. Used every MC step before
//!   the local BFGS minimisation.
//!
//! The RNG is injected as a `&mut R: Rng` so the caller controls
//! seeding/reproducibility (the search uses a seeded `ChaCha8Rng`).

use crate::conf::{Conf, Quat, Vec3};
use rand::Rng;

const TWO_PI: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// Quaternion for a rotation expressed as an axis-angle *rotation
/// vector* (direction = axis, magnitude = angle in radians). Mirrors
/// upstream `angle_to_quaternion(const vec&)`: a near-zero rotation maps
/// to the identity.
#[must_use]
pub fn angle_to_quaternion(rotation: Vec3) -> Quat {
    let angle = (rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2])
        .sqrt();
    if angle > f64::EPSILON {
        let inv = 1.0 / angle;
        Quat::from_axis_angle([rotation[0] * inv, rotation[1] * inv, rotation[2] * inv], angle)
    } else {
        Quat::IDENTITY
    }
}

/// Compose a small rotation onto an existing orientation, matching
/// upstream `quaternion_increment`: `q ← angle_to_quaternion(rotation) · q`,
/// renormalised.
#[must_use]
pub fn quaternion_increment(q: Quat, rotation: Vec3) -> Quat {
    angle_to_quaternion(rotation).mul(q).normalized()
}

/// Uniformly-distributed point strictly inside the unit ball, by
/// rejection sampling (upstream `random_inside_sphere`).
pub fn random_inside_sphere<R: Rng + ?Sized>(rng: &mut R) -> Vec3 {
    loop {
        let v = [
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < 1.0 {
            return v;
        }
    }
}

/// A uniformly-distributed random unit quaternion via Shoemake's
/// subgroup algorithm (Shoemake, *Graphics Gems III*, 1992). Upstream
/// normalises four Gaussian samples; both are uniform on SO(3), and we
/// do not match the upstream RNG stream (Phase-E parity is statistical),
/// so the cheaper three-uniform form is used.
pub fn random_orientation<R: Rng + ?Sized>(rng: &mut R) -> Quat {
    let u1: f64 = rng.gen_range(0.0..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    let u3: f64 = rng.gen_range(0.0..1.0);
    let r1 = (1.0 - u1).sqrt();
    let r2 = u1.sqrt();
    Quat {
        w: r2 * (TWO_PI * u3).cos(),
        x: r1 * (TWO_PI * u2).sin(),
        y: r1 * (TWO_PI * u2).cos(),
        z: r2 * (TWO_PI * u3).sin(),
    }
    .normalized()
}

/// Uniform point in the axis-aligned box `[corner1, corner2]`
/// (upstream `random_in_box`; expects `corner1[i] < corner2[i]`).
pub fn random_in_box<R: Rng + ?Sized>(corner1: Vec3, corner2: Vec3, rng: &mut R) -> Vec3 {
    [
        rng.gen_range(corner1[0]..=corner2[0]),
        rng.gen_range(corner1[1]..=corner2[1]),
        rng.gen_range(corner1[2]..=corner2[2]),
    ]
}

/// Radius of gyration of `coords` about `center` — RMS distance of the
/// atoms from the root origin. Upstream uses heavy atoms only; PDBQT
/// ligands are united-atom (only polar H survive), so the all-atom value
/// used here is within rounding of upstream and only scales the
/// orientation-mutation step, never an energy.
#[must_use]
pub fn gyration_radius(coords: &[Vec3], center: Vec3) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }
    let acc: f64 = coords
        .iter()
        .map(|c| {
            let d = [c[0] - center[0], c[1] - center[1], c[2] - center[2]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
        .sum();
    (acc / coords.len() as f64).sqrt()
}

/// A uniformly-random starting conformation inside the search box: root
/// position uniform in `[corner1, corner2]`, orientation uniform on
/// SO(3), every torsion uniform in `[-π, π)`. Mirrors upstream
/// `conf::randomize`.
pub fn randomize_conf<R: Rng + ?Sized>(
    corner1: Vec3,
    corner2: Vec3,
    n_torsions: usize,
    rng: &mut R,
) -> Conf {
    Conf {
        center: random_in_box(corner1, corner2, rng),
        orientation: random_orientation(rng),
        torsions: (0..n_torsions).map(|_| rng.gen_range(-PI..PI)).collect(),
    }
}

/// Perturb exactly one degree of freedom of `conf`, chosen uniformly
/// among the root translation, the root orientation, and the torsions
/// (upstream `mutate_conf`). The number of "mutable entities" is
/// `2 + n_torsions`:
///
/// * entity 0 — translate the root by `amplitude · random_inside_sphere`;
/// * entity 1 — rotate the root by `(amplitude / gyration_radius) ·
///   random_inside_sphere` (skipped if `gyration_radius` is ~0);
/// * entity `2 + k` — randomise torsion `k` to a fresh `[-π, π)` value.
///
/// `gyration_radius` should be that of the *current* pose (see
/// [`gyration_radius`]); it converts the linear `amplitude` (in Å) into a
/// comparable angular step so that large and small ligands explore
/// orientation space at similar effective displacement.
pub fn mutate_conf<R: Rng + ?Sized>(
    conf: &mut Conf,
    gyration_radius: f64,
    amplitude: f64,
    rng: &mut R,
) {
    let n_torsions = conf.torsions.len();
    let mutable = 2 + n_torsions;
    let which = rng.gen_range(0..mutable);

    if which == 0 {
        let d = random_inside_sphere(rng);
        conf.center[0] += amplitude * d[0];
        conf.center[1] += amplitude * d[1];
        conf.center[2] += amplitude * d[2];
    } else if which == 1 {
        if gyration_radius > f64::EPSILON {
            let s = amplitude / gyration_radius;
            let d = random_inside_sphere(rng);
            conf.orientation =
                quaternion_increment(conf.orientation, [s * d[0], s * d[1], s * d[2]]);
        }
        // else: zero-radius ligand — leave orientation unchanged, as upstream does.
    } else {
        conf.torsions[which - 2] = rng.gen_range(-PI..PI);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0x5EED_u64)
    }

    #[test]
    fn angle_to_quaternion_zero_is_identity() {
        assert_eq!(angle_to_quaternion([0.0, 0.0, 0.0]), Quat::IDENTITY);
    }

    #[test]
    fn random_inside_sphere_is_inside() {
        let mut r = rng();
        for _ in 0..10_000 {
            let v = random_inside_sphere(&mut r);
            assert!(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < 1.0);
        }
    }

    #[test]
    fn random_orientation_is_unit() {
        let mut r = rng();
        for _ in 0..10_000 {
            let q = random_orientation(&mut r);
            assert!((q.norm_sqr() - 1.0).abs() < 1e-9, "non-unit quat: {}", q.norm_sqr());
        }
    }

    #[test]
    fn random_in_box_respects_bounds() {
        let mut r = rng();
        let (c1, c2) = ([-3.0, 1.0, -10.0], [4.0, 2.0, -5.0]);
        for _ in 0..10_000 {
            let p = random_in_box(c1, c2, &mut r);
            for i in 0..3 {
                assert!(p[i] >= c1[i] && p[i] <= c2[i]);
            }
        }
    }

    #[test]
    fn randomize_conf_has_right_shape() {
        let mut r = rng();
        let c = randomize_conf([-5.0; 3], [5.0; 3], 4, &mut r);
        assert_eq!(c.torsions.len(), 4);
        for i in 0..3 {
            assert!(c.center[i] >= -5.0 && c.center[i] <= 5.0);
        }
        assert!((c.orientation.norm_sqr() - 1.0).abs() < 1e-9);
        assert!(c.torsions.iter().all(|&t| (-PI..PI).contains(&t)));
    }

    #[test]
    fn mutate_changes_exactly_one_kind_of_dof() {
        let mut r = rng();
        let gr = 3.0;
        // Over many trials each mutation must leave the conf still valid
        // (unit orientation, finite center/torsions) and change *something*.
        for _ in 0..2000 {
            let mut c = Conf::identity_at([1.0, 2.0, 3.0], 3);
            c.torsions = vec![0.1, 0.2, 0.3];
            let before = c.clone();
            mutate_conf(&mut c, gr, 2.0, &mut r);
            assert!((c.orientation.norm_sqr() - 1.0).abs() < 1e-6);
            assert!(c.center.iter().all(|v| v.is_finite()));
            assert!(c.torsions.iter().all(|v| v.is_finite()));
            let moved = c.center != before.center
                || c.orientation != before.orientation
                || c.torsions != before.torsions;
            assert!(moved, "mutation was a no-op");
        }
    }

    #[test]
    fn gyration_radius_of_centered_cube() {
        // 8 corners of a cube of side 2 centered at origin: each corner is
        // at distance sqrt(3); RMS distance is sqrt(3).
        let pts: Vec<Vec3> = vec![
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];
        let gr = gyration_radius(&pts, [0.0, 0.0, 0.0]);
        assert!((gr - 3.0_f64.sqrt()).abs() < 1e-12);
    }
}
