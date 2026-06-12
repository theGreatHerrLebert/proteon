// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Conformation types + quaternion helpers, ported from
// AutoDock-Vina src/lib/{conf.h,quaternion.h,tree.h} (Apache-2.0).

//! Ligand conformation and the quaternion math needed to apply it.
//!
//! A [`Conf`] is the minimum state that distinguishes one docked
//! pose from another: the ROOT fragment's rigid-body pose plus one
//! torsion angle per rotatable bond. Everything else — atom
//! positions — is recomputed from the fragment tree on demand.

/// 3-vector alias used throughout scoring and geometry.
pub type Vec3 = [f64; 3];

/// Quaternion in `(w, x, y, z)` order. Upstream stores quaternions
/// in the same layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    /// Identity rotation.
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// Quaternion representing a rotation of `angle` radians around
    /// `axis` (assumed unit-length). Matches upstream
    /// `angle_to_quaternion`.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Quat {
        let half = angle * 0.5;
        let s = half.sin();
        Quat {
            w: half.cos(),
            x: s * axis[0],
            y: s * axis[1],
            z: s * axis[2],
        }
    }

    /// Multiply two quaternions. Convention matches upstream's
    /// boost-quaternion operator* (Hamilton product).
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Quat) -> Quat {
        Quat {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Squared Euclidean norm of the 4-tuple.
    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Renormalize to unit length. Called after each torsion composition
    /// to keep accumulated float drift from degrading the rotation.
    #[must_use]
    pub fn normalized(self) -> Quat {
        let n = self.norm_sqr().sqrt();
        if n < f64::EPSILON {
            Quat::IDENTITY
        } else {
            let inv = 1.0 / n;
            Quat {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        }
    }

    /// Rotate a 3-vector by this quaternion. Uses the 3×3 matrix
    /// form for speed; equivalent to `q * v * q.conj()` but cheaper
    /// to apply repeatedly via the cached matrix below.
    #[must_use]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let m = self.to_matrix();
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// 3×3 rotation matrix for this quaternion. Standard formula
    /// (matches upstream `quaternion_to_r3`).
    #[must_use]
    pub fn to_matrix(self) -> [[f64; 3]; 3] {
        let Quat { w, x, y, z } = self;
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }
}

/// Rigid-body pose in world space. Origin of a fragment plus its
/// orientation relative to the ligand's root frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub origin: Vec3,
    pub orientation: Quat,
    /// Cached rotation matrix derived from `orientation`.
    matrix: [[f64; 3]; 3],
}

impl Frame {
    /// Identity frame at the given origin.
    #[must_use]
    pub fn at(origin: Vec3) -> Frame {
        Frame {
            origin,
            orientation: Quat::IDENTITY,
            matrix: Quat::IDENTITY.to_matrix(),
        }
    }

    /// Build a frame with an explicit orientation.
    #[must_use]
    pub fn new(origin: Vec3, orientation: Quat) -> Frame {
        let q = orientation.normalized();
        Frame { origin, orientation: q, matrix: q.to_matrix() }
    }

    /// Transform a point expressed in the frame's local coords into
    /// world coords: `origin + R · v`.
    #[inline]
    #[must_use]
    pub fn local_to_lab(&self, v: Vec3) -> Vec3 {
        let m = &self.matrix;
        [
            self.origin[0] + m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            self.origin[1] + m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            self.origin[2] + m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// Transform a direction vector (origin ignored): `R · v`.
    #[inline]
    #[must_use]
    pub fn local_to_lab_direction(&self, v: Vec3) -> Vec3 {
        let m = &self.matrix;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }
}

/// Ligand conformation: what a single pose consists of.
///
/// * `center` — world position of the ROOT fragment's origin.
/// * `orientation` — rotation of the ROOT fragment from its base
///   frame. The base frame is whatever was fixed at load time
///   (we pin it as identity, so the file-pose conformation has
///   `orientation == IDENTITY`).
/// * `torsions` — one angle per rotatable bond, indexed by child
///   fragment ID minus one (i.e. `torsions[0]` is the torsion on
///   fragment 1, which is the first `BRANCH` line).
#[derive(Clone, Debug)]
pub struct Conf {
    pub center: Vec3,
    pub orientation: Quat,
    pub torsions: Vec<f64>,
}

impl Conf {
    /// Zero-torsion conformation at the given center and identity
    /// orientation. Used to recover the original file pose (the
    /// torsion tree bakes the file's torsion angles into its base
    /// coordinates).
    #[must_use]
    pub fn identity_at(center: Vec3, n_torsions: usize) -> Conf {
        Conf {
            center,
            orientation: Quat::IDENTITY,
            torsions: vec![0.0; n_torsions],
        }
    }

    /// Total degrees of freedom: 3 translation + 4 quaternion + N torsions.
    #[must_use]
    pub fn num_dof(&self) -> usize {
        7 + self.torsions.len()
    }
}

#[inline]
pub(crate) fn vec_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn vec_add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn vec_scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn vec_norm(a: Vec3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
pub(crate) fn vec_normalize(a: Vec3) -> Vec3 {
    let n = vec_norm(a);
    assert!(n > f64::EPSILON, "cannot normalize zero vector");
    vec_scale(a, 1.0 / n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_quat_rotates_nothing() {
        let v = [1.0, 2.0, 3.0];
        let r = Quat::IDENTITY.rotate(v);
        for (a, b) in v.iter().zip(r.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn rotation_around_z_90deg() {
        // 90° around +z maps (1, 0, 0) → (0, 1, 0).
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let r = q.rotate([1.0, 0.0, 0.0]);
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(r[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn quat_mul_associative_with_axis_angle_composition() {
        // Two consecutive 45° rotations around z should equal one 90°.
        let q45 = Quat::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_4);
        let q90_composed = q45.mul(q45);
        let q90 = Quat::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(q90_composed.w, q90.w, epsilon = 1e-12);
        assert_relative_eq!(q90_composed.x, q90.x, epsilon = 1e-12);
        assert_relative_eq!(q90_composed.y, q90.y, epsilon = 1e-12);
        assert_relative_eq!(q90_composed.z, q90.z, epsilon = 1e-12);
    }

    #[test]
    fn quat_rotate_preserves_length() {
        let q = Quat::from_axis_angle([1.0, 1.0, 1.0_f64].map(|x| x / 3.0_f64.sqrt()), 1.2345);
        for v in [[1.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, -2.0, 3.0]] {
            let before = vec_norm(v);
            let after = vec_norm(q.rotate(v));
            assert_relative_eq!(before, after, epsilon = 1e-12);
        }
    }

    #[test]
    fn frame_composition_is_affine() {
        // Translating the origin then rotating should give the same
        // result as local_to_lab on the sum of the two contributions.
        let f = Frame::new([1.0, 2.0, 3.0], Quat::from_axis_angle([0.0, 0.0, 1.0], 0.3));
        let v = [0.5, 0.1, -0.2];
        let r = f.local_to_lab(v);
        let rot = f.orientation.rotate(v);
        assert_relative_eq!(r[0], 1.0 + rot[0], epsilon = 1e-12);
        assert_relative_eq!(r[1], 2.0 + rot[1], epsilon = 1e-12);
        assert_relative_eq!(r[2], 3.0 + rot[2], epsilon = 1e-12);
    }

    #[test]
    fn conf_identity_has_right_dof_count() {
        let c = Conf::identity_at([0.0, 0.0, 0.0], 7);
        assert_eq!(c.num_dof(), 14);
        assert!(c.torsions.iter().all(|&t| t == 0.0));
        assert_eq!(c.orientation, Quat::IDENTITY);
    }
}
