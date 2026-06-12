// Licensed under the Apache License, Version 2.0. See LICENSE.
//
// Torsion tree forward pass, ported from AutoDock-Vina
// src/lib/{tree.h,conf.h} (Apache-2.0).

//! Cached torsion-tree representation of a ligand.
//!
//! `TorsionTree::from_molecule` captures each fragment's local base
//! coordinates once at load time; `apply` then generates world
//! coordinates from any [`Conf`] by a single root-first traversal.
//!
//! Layout choices mirror upstream:
//!
//! * Fragment ID 0 is the ROOT; its origin is the first-in-file
//!   atom of the ROOT block (upstream `postprocess_ligand` does
//!   `rigid_body(p.atoms[0].a.coords, ...)`). We store the ROOT
//!   origin in `relative_origins[0]` and treat `relative_axes[0]`
//!   as unused.
//! * For every non-root fragment, `relative_origins[f]` and
//!   `relative_axes[f]` are captured in the *parent's* body frame
//!   with parent orientation = identity — which is exactly the
//!   initial state upstream constructs and the case the `segment`
//!   constructor explicitly asserts (`VINA_CHECK(eq(parent.orientation(),
//!   qt_identity))`).

use crate::conf::{vec_add, vec_normalize, vec_sub, Conf, Frame, Quat, Vec3};
use crate::molecule::Molecule;
use crate::pdbqt::PdbqtFile;

/// Cached torsion-tree data for one ligand.
#[derive(Clone, Debug)]
pub struct TorsionTree {
    /// Per-atom (Molecule-indexed) coordinates in the owning
    /// fragment's body frame.
    pub base_atoms: Vec<Vec3>,
    /// Fragment ID per atom (parallel to `base_atoms`).
    pub atom_fragment: Vec<u32>,
    /// Parent fragment index per fragment (None for ROOT).
    pub parents: Vec<Option<u32>>,
    /// World-space origin of the ROOT at load time (element 0);
    /// parent-frame origin offset for every other fragment.
    pub relative_origins: Vec<Vec3>,
    /// Parent-frame axis direction for each non-root fragment
    /// (unused for ROOT, stored as zeros).
    pub relative_axes: Vec<Vec3>,
}

impl TorsionTree {
    /// Build the cached tree from a fully-loaded [`Molecule`] and its
    /// originating [`PdbqtFile`]. The `Molecule` gives typed atoms +
    /// fragment IDs; the `PdbqtFile` gives the tree topology and
    /// axis atom serials.
    ///
    /// # Panics
    /// Panics if the `Molecule` and `PdbqtFile` disagree on atom
    /// count or if a declared axis atom was dropped during typing
    /// (H / W atoms as axis atoms — not expected for real PDBQT).
    pub fn from_molecule(mol: &Molecule, file: &PdbqtFile) -> Self {
        let n_frags = file.fragment_parents.len();

        // Build serial → Molecule-index map for axis-atom lookup.
        let serial_to_mol: std::collections::HashMap<u32, usize> = mol
            .original_serials
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();

        // Step 1: each fragment's world-space origin at load time.
        // ROOT: coords of its first-in-file atom (upstream convention).
        // Non-root F: coords of axis_end_F.
        let mut load_origin = vec![[0.0_f64; 3]; n_frags];
        for (f, axis_end) in file.fragment_axis_end.iter().enumerate() {
            if let Some(serial) = axis_end {
                let mi = serial_to_mol[serial];
                load_origin[f] = mol.coords[mi];
            }
        }
        // ROOT origin: first Molecule atom whose fragment_id is 0.
        let root_first_mi = mol
            .fragment_ids
            .iter()
            .position(|&fid| fid == 0)
            .expect("ligand must have at least one ROOT atom");
        load_origin[0] = mol.coords[root_first_mi];

        // Step 2: per non-root fragment, relative_origin and
        // relative_axis in the parent's body frame (parent
        // orientation = identity at load, so "parent-frame" ==
        // "world-offset from parent origin").
        let mut relative_origins = load_origin.clone();
        let mut relative_axes = vec![[0.0_f64; 3]; n_frags];
        for f in 1..n_frags {
            let parent = file.fragment_parents[f].expect("non-root has parent") as usize;
            // Shift into parent's body frame: (world - parent_world_origin).
            relative_origins[f] = vec_sub(load_origin[f], load_origin[parent]);
            // Axis direction: from axis_begin (in parent) → axis_end (our origin).
            let axis_begin_serial = file
                .fragment_axis_begin[f]
                .expect("non-root has axis_begin");
            let axis_begin_mi = serial_to_mol[&axis_begin_serial];
            let axis_dir = vec_sub(load_origin[f], mol.coords[axis_begin_mi]);
            relative_axes[f] = vec_normalize(axis_dir);
        }

        // Step 3: base coords per atom (world − fragment origin).
        let mut base_atoms = Vec::with_capacity(mol.coords.len());
        for (i, &xyz) in mol.coords.iter().enumerate() {
            let f = mol.fragment_ids[i] as usize;
            base_atoms.push(vec_sub(xyz, load_origin[f]));
        }

        Self {
            base_atoms,
            atom_fragment: mol.fragment_ids.clone(),
            parents: file.fragment_parents.clone(),
            relative_origins,
            relative_axes,
        }
    }

    /// Number of rotatable bonds = number of torsion-angle degrees
    /// of freedom (non-root fragment count).
    #[must_use]
    pub fn num_torsions(&self) -> usize {
        self.parents.len().saturating_sub(1)
    }

    /// Number of atoms represented by the tree.
    #[must_use]
    pub fn num_atoms(&self) -> usize {
        self.base_atoms.len()
    }

    /// The [`Conf`] that reproduces the input pose under `apply`:
    /// identity orientation, zero torsions, ROOT centred at its
    /// load-time origin.
    #[must_use]
    pub fn identity_conf(&self) -> Conf {
        Conf::identity_at(self.relative_origins[0], self.num_torsions())
    }

    /// Apply a conformation to the cached tree and emit world
    /// coordinates. Length matches `num_atoms`.
    ///
    /// Convenience wrapper over [`TorsionTree::apply_full`] that
    /// discards the intermediate frames.
    ///
    /// # Panics
    /// Panics if `conf.torsions.len()` doesn't match `num_torsions()`.
    #[must_use]
    pub fn apply(&self, conf: &Conf) -> Vec<Vec3> {
        self.apply_full(conf).coords
    }

    /// Apply a conformation and return the full forward-pass state
    /// (world coords, per-fragment frames, and per-non-root fragment
    /// world-space rotation axis). The gradient pass needs the
    /// frames to compute fragment origins and the world axis to
    /// project torques onto torsion DoFs, so caching the forward
    /// pass once and reusing it is both faster and avoids float
    /// drift between two separate evaluations of the same conf.
    ///
    /// # Panics
    /// Panics if `conf.torsions.len()` doesn't match `num_torsions()`.
    #[must_use]
    pub fn apply_full(&self, conf: &Conf) -> AppliedConf {
        assert_eq!(
            conf.torsions.len(),
            self.num_torsions(),
            "torsion count mismatch"
        );

        let n_frags = self.parents.len();
        let mut frames = vec![Frame::at([0.0; 3]); n_frags];
        let mut axes_world = vec![[0.0_f64; 3]; n_frags];

        frames[0] = Frame::new(conf.center, conf.orientation);

        for f in 1..n_frags {
            let parent = self.parents[f].expect("non-root has parent") as usize;
            debug_assert!(parent < f);
            let parent_frame = frames[parent];

            let origin_world = parent_frame.local_to_lab(self.relative_origins[f]);
            let axis_world = parent_frame.local_to_lab_direction(self.relative_axes[f]);
            axes_world[f] = axis_world;
            let torsion_q = Quat::from_axis_angle(axis_world, conf.torsions[f - 1]);
            let child_orientation = torsion_q.mul(parent_frame.orientation).normalized();
            frames[f] = Frame::new(origin_world, child_orientation);
        }

        let coords: Vec<Vec3> = self
            .atom_fragment
            .iter()
            .zip(self.base_atoms.iter())
            .map(|(&f, &base)| frames[f as usize].local_to_lab(base))
            .collect();

        AppliedConf { coords, frames, axes_world }
    }
}

/// State produced by a single forward pass of a [`Conf`] through a
/// [`TorsionTree`]. Reused by [`crate::gradient::gradient_from_forces`].
#[derive(Clone, Debug)]
pub struct AppliedConf {
    /// World-space atom coordinates (parallel to `TorsionTree::base_atoms`).
    pub coords: Vec<Vec3>,
    /// One [`Frame`] per fragment, root-first.
    pub frames: Vec<Frame>,
    /// World-space rotation axis for each non-root fragment. Element
    /// 0 is a zero placeholder for ROOT.
    pub axes_world: Vec<Vec3>,
}

/// Translate every coordinate in `xs` by `shift`. Useful helper for
/// round-trip tests.
#[must_use]
pub fn translated(xs: &[Vec3], shift: Vec3) -> Vec<Vec3> {
    xs.iter().map(|&v| vec_add(v, shift)).collect()
}

/// 3-vector cross product. Used by the gradient pass.
#[inline]
#[must_use]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3-vector dot product.
#[inline]
#[must_use]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}


/// Rotate every coordinate in `xs` around `origin` by `q`.
#[must_use]
pub fn rotated(xs: &[Vec3], origin: Vec3, q: Quat) -> Vec<Vec3> {
    xs.iter()
        .map(|&v| vec_add(origin, q.rotate(vec_sub(v, origin))))
        .collect()
}

/// RMSD between two coordinate sequences. Used in tests.
#[must_use]
pub fn rmsd(a: &[Vec3], b: &[Vec3]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(u, v)| {
            let d = vec_sub(*u, *v);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
        .sum();
    (sum / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdbqt::parse_pdbqt;
    use approx::assert_relative_eq;

    const LIG_1IEP: &str = include_str!("../tests/fixtures/pairs/1iep/ligand.pdbqt");
    const LIG_1FPU: &str = include_str!("../tests/fixtures/pairs/1fpu/ligand.pdbqt");
    const LIG_1S63: &str = include_str!("../tests/fixtures/pairs/1s63/ligand.pdbqt");
    const LIG_BACE1: &str = include_str!("../tests/fixtures/pairs/bace1/ligand.pdbqt");

    fn load(name: &str) -> (Molecule, PdbqtFile) {
        let lig = match name {
            "1iep" => LIG_1IEP,
            "1fpu" => LIG_1FPU,
            "1s63" => LIG_1S63,
            "bace1" => LIG_BACE1,
            _ => panic!("unknown fixture {name}"),
        };
        let molecule = Molecule::from_pdbqt_str(lig).unwrap();
        let file = parse_pdbqt(lig).unwrap();
        (molecule, file)
    }

    #[test]
    fn identity_conf_reproduces_input_pose_for_all_fixtures() {
        // The round-trip test: extract the tree, ask for the identity
        // conformation, apply it, compare to the original coords.
        // Bit-level match isn't required — tree construction goes
        // through float arithmetic — but 1e-9 RMSD is a very loose
        // upper bound.
        for name in ["1iep", "1fpu", "1s63", "bace1"] {
            let (mol, file) = load(name);
            let tree = TorsionTree::from_molecule(&mol, &file);
            let conf = tree.identity_conf();
            let world = tree.apply(&conf);
            let r = rmsd(&world, &mol.coords);
            assert!(
                r < 1e-9,
                "{name}: identity-conf RMSD = {r:.3e}; expected ~0"
            );
        }
    }

    #[test]
    fn pure_translation_conf_shifts_every_atom_equally() {
        let (mol, file) = load("1iep");
        let tree = TorsionTree::from_molecule(&mol, &file);
        let mut conf = tree.identity_conf();
        let delta = [3.0, -1.5, 0.7];
        conf.center = [
            conf.center[0] + delta[0],
            conf.center[1] + delta[1],
            conf.center[2] + delta[2],
        ];
        let moved = tree.apply(&conf);
        let expected = translated(&mol.coords, delta);
        assert_relative_eq!(rmsd(&moved, &expected), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn pure_rotation_preserves_internal_distances() {
        // Rotating the ROOT's orientation while keeping every torsion
        // at zero should rigidly rotate the whole ligand. Pairwise
        // distances must be unchanged to float precision.
        let (mol, file) = load("1iep");
        let tree = TorsionTree::from_molecule(&mol, &file);
        let mut conf = tree.identity_conf();
        conf.orientation =
            Quat::from_axis_angle([0.3, 0.7, 0.1_f64].map(|x| x / (0.59_f64).sqrt()), 0.8);
        let rotated_coords = tree.apply(&conf);

        let n = mol.coords.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d_orig = {
                    let d = vec_sub(mol.coords[i], mol.coords[j]);
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                };
                let d_new = {
                    let d = vec_sub(rotated_coords[i], rotated_coords[j]);
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                };
                assert_relative_eq!(d_orig, d_new, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn torsion_change_affects_only_child_subtree() {
        // Rotating a torsion by any angle must leave every atom in
        // the parent subtree (fragment_id <= parent) untouched, and
        // move at least one atom in the child subtree.
        let (mol, file) = load("1iep");
        let tree = TorsionTree::from_molecule(&mol, &file);
        let mut conf = tree.identity_conf();
        // Perturb the torsion on fragment 1 (first BRANCH).
        conf.torsions[0] = 0.42;
        let new_coords = tree.apply(&conf);

        let mut root_max_shift: f64 = 0.0;
        let mut child_subtree_max_shift: f64 = 0.0;
        // Identify fragment-1 subtree as every fragment whose ancestor
        // chain contains 1.
        let mut in_sub = vec![false; tree.parents.len()];
        in_sub[1] = true;
        for f in 2..tree.parents.len() {
            let p = tree.parents[f].unwrap() as usize;
            if in_sub[p] {
                in_sub[f] = true;
            }
        }
        for (i, &fid) in mol.fragment_ids.iter().enumerate() {
            let d = vec_sub(new_coords[i], mol.coords[i]);
            let s = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if in_sub[fid as usize] {
                child_subtree_max_shift = child_subtree_max_shift.max(s);
            } else {
                root_max_shift = root_max_shift.max(s);
            }
        }
        // Parent subtree (ROOT here) must not move.
        assert!(
            root_max_shift < 1e-9,
            "ROOT atoms shifted by {root_max_shift:.3e} after torsion"
        );
        // Child subtree must move perceptibly.
        assert!(
            child_subtree_max_shift > 0.1,
            "child subtree moved only {child_subtree_max_shift:.3e}"
        );
    }

    #[test]
    fn torsion_tree_has_expected_dof_count() {
        let (mol, file) = load("bace1");
        let tree = TorsionTree::from_molecule(&mol, &file);
        // BACE_1 has 22 BRANCH lines.
        assert_eq!(tree.num_torsions(), 22);
        assert_eq!(tree.num_atoms(), mol.coords.len());
    }

    #[test]
    fn forward_apply_is_deterministic_under_repeated_calls() {
        let (mol, file) = load("1s63");
        let tree = TorsionTree::from_molecule(&mol, &file);
        let mut conf = tree.identity_conf();
        conf.orientation = Quat::from_axis_angle([0.0, 0.0, 1.0], 0.5);
        conf.torsions[0] = 0.33;
        let a = tree.apply(&conf);
        let b = tree.apply(&conf);
        assert_eq!(a, b, "apply must be deterministic");
    }

    #[test]
    fn round_trip_rotation_returns_to_origin() {
        // Rotating by q then by q⁻¹ (via an opposite-sign axis-angle)
        // should recover the original pose to float precision.
        let (mol, file) = load("1iep");
        let tree = TorsionTree::from_molecule(&mol, &file);

        let raw = [1.0, 2.0, 3.0_f64];
        let n = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let axis_unit = [raw[0] / n, raw[1] / n, raw[2] / n];
        let angle = 0.7;

        let mut conf = tree.identity_conf();
        conf.orientation = Quat::from_axis_angle(axis_unit, angle);
        let rotated = tree.apply(&conf);
        // Forward then inverse rotation:
        conf.orientation = Quat::from_axis_angle(axis_unit, -angle)
            .mul(conf.orientation)
            .normalized();
        let back = tree.apply(&conf);
        // Rotated should be far from the original.
        assert!(rmsd(&rotated, &mol.coords) > 0.1);
        // Back should match the original.
        assert!(rmsd(&back, &mol.coords) < 1e-9);
    }
}
