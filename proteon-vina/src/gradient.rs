// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Ported from AutoDock-Vina src/lib/tree.h::derivative and
// `atom_frame::sum_force_and_torque` (Apache-2.0).

//! Gradient of the scoring function with respect to a [`Conf`].
//!
//! Takes per-atom forces (i.e. `-dE/dr_i`) and projects them back
//! onto the conformation's degrees of freedom using the cached
//! torsion tree. The composition rule mirrors upstream:
//!
//! * Each fragment's own force/torque is the sum over its atoms of
//!   `(f_i, (r_i − origin_F) × f_i)`.
//! * Descendants propagate into ancestors: `force[parent] +=
//!   force[child]`, `torque[parent] += torque[child] +
//!   (origin_child − origin_parent) × force[child]`.
//! * Torsion DoF gradient for fragment F = `torque[F] · axis_F_world`
//!   (the component of torque that rotates about the rotor axis).
//! * ROOT DoF gradients = (`force[ROOT]`, `torque[ROOT]`) — the
//!   translation force and the 3-component rotation torque.
//!
//! A per-atom **force** `f_i = −∂E/∂r_i` feeds in; the output is a
//! gradient in DoF space with the same sign convention (so gradient
//! descent moves DoFs along `-ConfGrad`).

use crate::conf::{Conf, Quat, Vec3};
use crate::torsion::{cross, dot, AppliedConf, TorsionTree};

/// Gradient of the scoring function in conformation space.
///
/// Layout matches upstream `change = (rigid_change, flv torsions)`
/// where `rigid_change = (position: vec, orientation: vec)`.
/// Every field has the SAME SIGN as the corresponding force: moving
/// `conf.center` along `grad.center * h` (for small `h > 0`) should
/// DECREASE the energy at first order, because `grad.center` is
/// `−∂E/∂center` (i.e. it's a force, not a loss gradient).
#[derive(Clone, Debug, PartialEq)]
pub struct ConfGrad {
    pub center: Vec3,
    pub orientation: Vec3,
    pub torsions: Vec<f64>,
}

impl ConfGrad {
    /// Zero gradient with the right torsion-vector length.
    #[must_use]
    pub fn zero(num_torsions: usize) -> ConfGrad {
        ConfGrad {
            center: [0.0; 3],
            orientation: [0.0; 3],
            torsions: vec![0.0; num_torsions],
        }
    }

    /// Total number of scalar components: 3 translation + 3 rotation
    /// + N torsions.
    #[must_use]
    pub fn len(&self) -> usize {
        6 + self.torsions.len()
    }

    /// False for a nonsense zero-length gradient (which would
    /// correspond to a molecule with no atoms — never constructed in
    /// normal use).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// L∞ norm across all DoFs. Useful for convergence checks.
    #[must_use]
    pub fn max_abs(&self) -> f64 {
        let r = self
            .center
            .iter()
            .chain(self.orientation.iter())
            .chain(self.torsions.iter())
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        r
    }

    /// Flat scalar access: `0..3` center, `3..6` orientation, `6..6+N`
    /// torsions. Matches upstream `Change::operator()(sz)`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> f64 {
        match i {
            0..=2 => self.center[i],
            3..=5 => self.orientation[i - 3],
            k => self.torsions[k - 6],
        }
    }

    /// Mutable flat access.
    #[inline]
    pub fn set(&mut self, i: usize, v: f64) {
        match i {
            0..=2 => self.center[i] = v,
            3..=5 => self.orientation[i - 3] = v,
            k => self.torsions[k - 6] = v,
        }
    }

    /// Dot product: `Σ self(i) · other(i)` across all DoFs.
    #[must_use]
    pub fn dot(&self, other: &ConfGrad) -> f64 {
        let mut s = 0.0;
        for k in 0..3 {
            s += self.center[k] * other.center[k];
            s += self.orientation[k] * other.orientation[k];
        }
        for (a, b) in self.torsions.iter().zip(other.torsions.iter()) {
            s += a * b;
        }
        s
    }

    /// L2 norm squared — same as `self.dot(self)`.
    #[must_use]
    pub fn norm_sqr(&self) -> f64 {
        self.dot(self)
    }

    /// Element-wise subtraction: `self -= other`.
    pub fn subtract(&mut self, other: &ConfGrad) {
        for k in 0..3 {
            self.center[k] -= other.center[k];
            self.orientation[k] -= other.orientation[k];
        }
        for (a, b) in self.torsions.iter_mut().zip(other.torsions.iter()) {
            *a -= *b;
        }
    }

    /// Negate into a fresh value: returns `-self` as a new `ConfGrad`.
    #[must_use]
    pub fn negated(&self) -> ConfGrad {
        ConfGrad {
            center: [-self.center[0], -self.center[1], -self.center[2]],
            orientation: [
                -self.orientation[0],
                -self.orientation[1],
                -self.orientation[2],
            ],
            torsions: self.torsions.iter().map(|t| -t).collect(),
        }
    }
}

impl Conf {
    /// In-place step: `self += alpha * direction`. Matches upstream
    /// `Conf::increment`. Used by BFGS line search.
    ///
    /// Translation and torsion components add linearly; orientation
    /// composes via axis-angle as in [`conf_step`].
    pub fn increment(&mut self, direction: &ConfGrad, alpha: f64) {
        *self = conf_step(self, direction, alpha);
    }
}

/// Project per-atom forces back onto a [`ConfGrad`] via the
/// [`TorsionTree`].
///
/// * `tree` — the cached torsion tree built from the ligand.
/// * `applied` — state produced by [`TorsionTree::apply_full`] on
///   the conf whose gradient we want.
/// * `per_atom_force` — `-∂E/∂r_i` for each atom (world-frame
///   vector). Must have length `tree.num_atoms()`.
#[must_use]
pub fn gradient_from_forces(
    tree: &TorsionTree,
    applied: &AppliedConf,
    per_atom_force: &[Vec3],
) -> ConfGrad {
    assert_eq!(per_atom_force.len(), tree.num_atoms(), "force length");
    let n_frags = tree.parents.len();

    // Own force/torque per fragment (force = Σ f_i, torque = Σ
    // (r_i − origin_F) × f_i). Later reduced into the parent.
    let mut frag_force: Vec<Vec3> = vec![[0.0; 3]; n_frags];
    let mut frag_torque: Vec<Vec3> = vec![[0.0; 3]; n_frags];
    for (i, &fid) in tree.atom_fragment.iter().enumerate() {
        let f = fid as usize;
        let r = applied.coords[i];
        let origin = applied.frames[f].origin;
        let arm = [r[0] - origin[0], r[1] - origin[1], r[2] - origin[2]];
        let fi = per_atom_force[i];
        frag_force[f][0] += fi[0];
        frag_force[f][1] += fi[1];
        frag_force[f][2] += fi[2];
        let t = cross(arm, fi);
        frag_torque[f][0] += t[0];
        frag_torque[f][1] += t[1];
        frag_torque[f][2] += t[2];
    }

    // Propagate children → parent in reverse fragment-ID order.
    // Our parser emits DFS pre-order, so parent(f) < f for all f ≥ 1
    // and iterating high-to-low visits every descendant before its
    // ancestor.
    let mut torsion_grad = vec![0.0_f64; tree.num_torsions()];
    for f in (1..n_frags).rev() {
        // Torsion DoF for this branch = torque about its world axis.
        torsion_grad[f - 1] = dot(frag_torque[f], applied.axes_world[f]);

        let parent = tree.parents[f].expect("non-root has parent") as usize;
        let arm = {
            let o_child = applied.frames[f].origin;
            let o_parent = applied.frames[parent].origin;
            [
                o_child[0] - o_parent[0],
                o_child[1] - o_parent[1],
                o_child[2] - o_parent[2],
            ]
        };
        let shifted = cross(arm, frag_force[f]);
        frag_force[parent][0] += frag_force[f][0];
        frag_force[parent][1] += frag_force[f][1];
        frag_force[parent][2] += frag_force[f][2];
        frag_torque[parent][0] += frag_torque[f][0] + shifted[0];
        frag_torque[parent][1] += frag_torque[f][1] + shifted[1];
        frag_torque[parent][2] += frag_torque[f][2] + shifted[2];
    }

    ConfGrad {
        center: frag_force[0],
        orientation: frag_torque[0],
        torsions: torsion_grad,
    }
}

/// Apply a small step `step_size * direction` to `conf`, returning a
/// new `Conf`. Convention: the step moves along the gradient (i.e.
/// `direction = grad` moves the energy DOWN at first order, because
/// `grad` is −∂E/∂conf). Used by finite-difference regression tests
/// and by the BFGS line search.
#[must_use]
pub fn conf_step(conf: &Conf, direction: &ConfGrad, step_size: f64) -> Conf {
    let mut new = conf.clone();
    for i in 0..3 {
        new.center[i] += step_size * direction.center[i];
    }
    // Rotation step: compose a small-angle quaternion about the
    // `orientation` torque direction onto the current orientation.
    // |orientation| = rotation magnitude; normalise to axis.
    let omega = [
        step_size * direction.orientation[0],
        step_size * direction.orientation[1],
        step_size * direction.orientation[2],
    ];
    let ang = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    if ang > f64::EPSILON {
        let inv = 1.0 / ang;
        let axis = [omega[0] * inv, omega[1] * inv, omega[2] * inv];
        let dq = Quat::from_axis_angle(axis, ang);
        new.orientation = dq.mul(conf.orientation).normalized();
    }
    for (i, &d) in direction.torsions.iter().enumerate() {
        new.torsions[i] += step_size * d;
    }
    new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conf::{vec_sub, Conf, Quat};
    use crate::molecule::Molecule;
    use crate::pdbqt::parse_pdbqt;
    use crate::torsion::{rmsd, TorsionTree};

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const LIG_BACE1: &str = include_str!("../tests/fixtures/pairs/bace1/ligand.pdbqt");

    fn load(lig: &str) -> (Molecule, TorsionTree) {
        let mol = Molecule::from_pdbqt_str(lig).unwrap();
        let file = parse_pdbqt(lig).unwrap();
        let tree = TorsionTree::from_molecule(&mol, &file);
        (mol, tree)
    }

    /// Toy quadratic energy `E(coords) = Σ ‖r_i − target_i‖²`. Per-atom
    /// force `f_i = −∂E/∂r_i = −2 (r_i − target_i)`. Analytic and FD
    /// gradients must match to high precision.
    fn quadratic_energy(coords: &[Vec3], target: &[Vec3]) -> f64 {
        coords
            .iter()
            .zip(target.iter())
            .map(|(r, t)| {
                let d = vec_sub(*r, *t);
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            })
            .sum()
    }

    fn quadratic_forces(coords: &[Vec3], target: &[Vec3]) -> Vec<Vec3> {
        coords
            .iter()
            .zip(target.iter())
            .map(|(r, t)| {
                let d = vec_sub(*r, *t);
                [-2.0 * d[0], -2.0 * d[1], -2.0 * d[2]]
            })
            .collect()
    }

    /// Central-difference numerical derivative of a scalar function
    /// along a given direction in Conf space.
    fn fd_energy_directional<F>(
        tree: &TorsionTree,
        conf: &Conf,
        dir: &ConfGrad,
        h: f64,
        energy: F,
    ) -> f64
    where
        F: Fn(&[Vec3]) -> f64,
    {
        let plus = conf_step(conf, dir, h);
        let minus = conf_step(conf, dir, -h);
        let e_plus = energy(&tree.apply(&plus));
        let e_minus = energy(&tree.apply(&minus));
        (e_plus - e_minus) / (2.0 * h)
    }

    #[test]
    fn gradient_of_zero_force_is_zero() {
        let (_mol, tree) = load(LIG_1IEP);
        let conf = tree.identity_conf();
        let applied = tree.apply_full(&conf);
        let zeros = vec![[0.0_f64; 3]; tree.num_atoms()];
        let g = gradient_from_forces(&tree, &applied, &zeros);
        assert_eq!(g.center, [0.0; 3]);
        assert_eq!(g.orientation, [0.0; 3]);
        assert!(g.torsions.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn translation_gradient_equals_total_force() {
        // Only the ROOT translation component should respond to a
        // uniform force. Every other DoF must stay at zero modulo
        // float noise.
        let (_mol, tree) = load(LIG_1IEP);
        let conf = tree.identity_conf();
        let applied = tree.apply_full(&conf);
        let uniform = [0.3, -0.7, 1.1];
        let forces = vec![uniform; tree.num_atoms()];
        let g = gradient_from_forces(&tree, &applied, &forces);

        let n = tree.num_atoms() as f64;
        for i in 0..3 {
            assert!(
                (g.center[i] - n * uniform[i]).abs() < 1e-9,
                "center[{i}]: got {}, want {}",
                g.center[i],
                n * uniform[i]
            );
        }
        // Torque about ROOT origin = Σ (r_i − o) × uniform_force. For
        // a uniform force, this is (Σ r_i − N·o) × uniform_force.
        // Non-zero in general but fully determined; easier invariant:
        // every torsion gradient is zero (no atom-specific asymmetry
        // between a branch and its complement).
        // With uniform forces pointing in an arbitrary direction,
        // the torque about an arbitrary axis is generally non-zero,
        // so we only assert the torsion-space gradient is finite.
        for t in g.torsions {
            assert!(t.is_finite());
        }
    }

    /// Verify the analytic gradient against a central-difference
    /// numerical gradient on a toy quadratic potential, across every
    /// DoF direction.
    fn check_fd_parity(mol: &Molecule, tree: &TorsionTree, conf: &Conf) {
        // Non-trivial target: original pose shifted by a quirky vector
        // so atoms aren't at their minimum (otherwise the gradient
        // would be trivially zero).
        let target: Vec<Vec3> = mol
            .coords
            .iter()
            .map(|&c| [c[0] + 0.1, c[1] - 0.2, c[2] + 0.05])
            .collect();

        let applied = tree.apply_full(conf);
        let forces = quadratic_forces(&applied.coords, &target);
        let g_analytic = gradient_from_forces(tree, &applied, &forces);
        // Note: forces are -∂E/∂r, so gradient_from_forces returns
        // -∂E/∂conf.  For FD we compute ∂E/∂conf (positive sign),
        // so compare |-g_analytic_i - g_fd_i|.

        let h = 1e-5;

        // Sweep unit-vector directions across the whole DoF space.
        let mut probe = ConfGrad::zero(tree.num_torsions());
        let dof = probe.len();
        let energy = |c: &[Vec3]| quadratic_energy(c, &target);

        for k in 0..dof {
            // Zero all directions then light one up.
            probe = ConfGrad::zero(tree.num_torsions());

            let analytic_component = if k < 3 {
                probe.center[k] = 1.0;
                g_analytic.center[k]
            } else if k < 6 {
                probe.orientation[k - 3] = 1.0;
                g_analytic.orientation[k - 3]
            } else {
                let t = k - 6;
                probe.torsions[t] = 1.0;
                g_analytic.torsions[t]
            };

            let fd = fd_energy_directional(tree, conf, &probe, h, energy);
            // analytic is -∂E/∂x, fd is +∂E/∂x. They should differ by
            // sign. Compare |analytic + fd|.
            let err_abs = (analytic_component + fd).abs();
            let scale = analytic_component.abs().max(fd.abs()).max(1.0);
            let err_rel = err_abs / scale;
            assert!(
                err_rel < 1e-4,
                "DoF {k}: analytic={analytic_component:.6e}, fd={fd:.6e}, rel err={err_rel:.3e}"
            );
        }
    }

    #[test]
    fn fd_parity_on_quadratic_potential_1iep() {
        let (mol, tree) = load(LIG_1IEP);
        let conf = tree.identity_conf();
        check_fd_parity(&mol, &tree, &conf);
    }

    #[test]
    fn fd_parity_on_quadratic_potential_with_nonidentity_conf() {
        // Non-trivial starting conformation: small rotation + a few
        // non-zero torsions. Gradients must still match FD.
        let (mol, tree) = load(LIG_1IEP);
        let mut conf = tree.identity_conf();
        conf.orientation = Quat::from_axis_angle([0.0, 0.0, 1.0], 0.25);
        conf.torsions[0] = 0.3;
        if conf.torsions.len() > 1 {
            conf.torsions[1] = -0.2;
        }
        check_fd_parity(&mol, &tree, &conf);
    }

    #[test]
    fn fd_parity_on_bace1_macrocycle() {
        // Macrocycle ligand with 22 torsions — stresses the reverse
        // reduction through deeply nested fragments.
        let (mol, tree) = load(LIG_BACE1);
        let conf = tree.identity_conf();
        check_fd_parity(&mol, &tree, &conf);
    }

    #[test]
    fn conf_step_zero_direction_is_identity() {
        let zero = ConfGrad::zero(3);
        let c = Conf {
            center: [1.0, 2.0, 3.0],
            orientation: Quat::from_axis_angle([0.0, 0.0, 1.0], 0.5),
            torsions: vec![0.1, 0.2, 0.3],
        };
        let c2 = conf_step(&c, &zero, 0.1);
        assert_eq!(c2.center, c.center);
        assert_eq!(c2.orientation, c.orientation);
        assert_eq!(c2.torsions, c.torsions);
    }

    #[test]
    fn conf_step_preserves_ligand_rmsd_under_matched_inverse_step() {
        // Step by +h*dir, then by -h*dir: original conf recovered.
        let (_mol, tree) = load(LIG_1IEP);
        let conf = tree.identity_conf();
        let mut dir = ConfGrad::zero(tree.num_torsions());
        dir.center = [1.0, 0.0, 0.0];
        dir.orientation = [0.0, 0.0, 1.0];
        dir.torsions[0] = 0.7;
        let c1 = conf_step(&conf, &dir, 0.05);
        let c2 = conf_step(&c1, &dir, -0.05);
        // Translation/torsion recover exactly; orientation to float
        // precision since composition is approximate.
        assert!(rmsd(&tree.apply(&conf), &tree.apply(&c2)) < 1e-9);
    }
}
