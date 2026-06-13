// Licensed under the Apache License, Version 2.0. See LICENSE.

//! End-to-end pipeline: PDBQT string → typed, ready-to-score atoms.
//!
//! `Molecule` holds the subset of atoms that carry an X-Score type —
//! i.e. everything except hydrogens and hydrated-docking waters.
//!
//! For ligands we also retain torsion-tree fragment IDs (from
//! `ROOT`/`BRANCH`) so intra-molecular pair accounting can exclude
//! pairs that sit in the same rigid fragment (upstream's
//! `DISTANCE_FIXED` case).
//!
//! Parallel arrays rather than an `Atom` struct — the hot scoring
//! loop iterates over coordinates, and this layout keeps them
//! contiguous.

use crate::ad_types::AdType;
use crate::atom_types::XsType;
use crate::bonds::{infer_bonds_masked, BondGraph};
use crate::pdbqt::{parse_pdbqt, PdbqtError, PdbqtFile};
use crate::xs_assign::assign_xs_types;

/// Typed, ready-to-score atoms from a single PDBQT file.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// Cartesian coordinates of typed atoms (Å).
    pub coords: Vec<[f64; 3]>,
    /// X-Score atom type of each typed atom.
    pub xs_types: Vec<XsType>,
    /// AutoDock atom type of each typed atom — retained for
    /// upstream-compatible closure-clash / glue-pair checks.
    pub ad_types: Vec<AdType>,
    /// Partial charges carried over from the PDBQT.
    pub partial_charges: Vec<f64>,
    /// Original PDBQT serial for each retained atom.
    pub original_serials: Vec<u32>,
    /// Torsion-tree fragment ID per retained atom. Receptors and
    /// tree-free files have all zeros.
    pub fragment_ids: Vec<u32>,
    /// Adjacency list among retained atoms, inferred from covalent
    /// radii. Symmetric; neighbor lists are sorted.
    pub bonds: BondGraph,
    /// Per-atom bitmask of fragment memberships in upstream's
    /// `DISTANCE_FIXED` sense: atom `i` contributes bit `f` if it
    /// belongs to fragment `f`'s "extended rigid group" (core atoms
    /// of `f` plus axis atoms of `f` plus axis-end atoms of `f`'s
    /// direct children). Two atoms are `DISTANCE_FIXED` iff their
    /// masks share any bit. Ligands with ≥ 64 fragments are
    /// unsupported in v0.
    pub fragment_mask: Vec<u64>,
}

impl Molecule {
    /// Parse a PDBQT string into a typed `Molecule`.
    ///
    /// Pipeline: line-level parse + tree tracking → bond inference →
    /// XS-type assignment → drop untyped atoms (H, HD, W). Fragment
    /// IDs and the bond graph are remapped onto the retained-atom
    /// index space.
    pub fn from_pdbqt_str(text: &str) -> Result<Self, PdbqtError> {
        let file = parse_pdbqt(text)?;
        Ok(Self::from_pdbqt_file(&file))
    }

    /// Build a `Molecule` from an already-parsed `PdbqtFile`. Prefer
    /// [`Molecule::from_pdbqt_str`] for the single-pose case; this
    /// constructor is meant for multi-pose workflows that call
    /// [`crate::pdbqt::parse_pdbqt_models`] once and then iterate
    /// the resulting pose slices.
    #[must_use]
    pub fn from_pdbqt_file(file: &PdbqtFile) -> Self {
        let raw = &file.atoms;
        // Bond inference is filtered by fragment-mask mobility
        // (upstream `model::assign_bonds` skips DISTANCE_VARIABLE
        // pairs when considering candidate bonds). Without this,
        // macrocycle ligands introduce "ghost bonds" between
        // co-located G0/CG0 pseudo-atoms that pollute the 1-2/1-3/1-4
        // exclusion sets.
        let raw_mask = raw_fragment_mask(file);
        let raw_graph = infer_bonds_masked(raw, Some(&raw_mask));
        let xs = assign_xs_types(raw, &raw_graph);

        // Map raw-atom index → molecule-atom index. `None` for atoms
        // dropped during XS-type assignment.
        let mut raw_to_mol: Vec<Option<usize>> = vec![None; raw.len()];
        let mut next = 0_usize;
        for (i, t) in xs.iter().enumerate() {
            if t.is_some() {
                raw_to_mol[i] = Some(next);
                next += 1;
            }
        }
        let n_kept = next;

        let mut coords = Vec::with_capacity(n_kept);
        let mut xs_types = Vec::with_capacity(n_kept);
        let mut ad_types = Vec::with_capacity(n_kept);
        let mut partial_charges = Vec::with_capacity(n_kept);
        let mut original_serials = Vec::with_capacity(n_kept);
        let mut fragment_ids = Vec::with_capacity(n_kept);

        for (i, atom) in raw.iter().enumerate() {
            if let Some(t) = xs[i] {
                coords.push(atom.coords);
                xs_types.push(t);
                ad_types.push(atom.ad_type);
                partial_charges.push(atom.partial_charge);
                original_serials.push(atom.serial);
                fragment_ids.push(file.fragment_ids[i]);
                debug_assert_eq!(raw_to_mol[i], Some(coords.len() - 1));
            }
        }

        // Fragment mask on the MOLECULE-indexed atoms. Derived from
        // the same raw_mask computed above, with indices remapped.
        let mut fragment_mask = vec![0_u64; n_kept];
        for (i_raw, &mi) in raw_to_mol.iter().enumerate() {
            if let Some(mi) = mi {
                fragment_mask[mi] = raw_mask[i_raw];
            }
        }

        // Remap the bond graph through raw_to_mol, preserving sort.
        let mut bonds: BondGraph = vec![Vec::new(); n_kept];
        for (i, nbrs) in raw_graph.iter().enumerate() {
            let Some(mi) = raw_to_mol[i] else { continue };
            for &j in nbrs {
                if let Some(mj) = raw_to_mol[j] {
                    bonds[mi].push(mj);
                }
            }
            // Neighbour list stays sorted because raw_to_mol is
            // monotonic in retained-atom index.
            debug_assert!(bonds[mi].windows(2).all(|w| w[0] < w[1]));
        }

        Self {
            coords,
            xs_types,
            ad_types,
            partial_charges,
            original_serials,
            fragment_ids,
            bonds,
            fragment_mask,
        }
    }

    /// Number of typed atoms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Upper bound on the number of atoms supported per molecule for
    /// the u64 fragment_mask. See `Molecule::fragment_mask`.
    pub const MAX_FRAGMENTS: usize = 64;
}

/// Compute upstream's `DISTANCE_FIXED` extended-group bitmask on the
/// RAW atom set (before H/W filtering). Bit `F` is set on atom `a`
/// iff `a` is in fragment F's effective rigid group:
/// * core atoms of F (atoms whose `fragment_ids[a] == F`), or
/// * `a == fragment_axis_begin[F]` (parent-side axis atom, folded in
///   by `add_bonds(axis_begin, b.node)`), or
/// * `a == fragment_axis_end[C]` for some direct child C of F
///   (the "immobile" atoms upstream inserts into the parent's
///   `b.node` range).
fn raw_fragment_mask(file: &PdbqtFile) -> Vec<u64> {
    let n_frags = file.fragment_parents.len();
    assert!(
        n_frags <= Molecule::MAX_FRAGMENTS,
        "proteon-vina v0 supports up to {} torsion fragments",
        Molecule::MAX_FRAGMENTS
    );
    let mut mask = vec![0_u64; file.atoms.len()];
    // Bit for own fragment.
    for (i, &fid) in file.fragment_ids.iter().enumerate() {
        mask[i] |= 1 << fid;
    }
    let serial_to_idx: std::collections::HashMap<u32, usize> = file
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (a.serial, i))
        .collect();
    for f in 0..n_frags {
        let f_u32 = f as u32;
        if let Some(s) = file.fragment_axis_begin[f] {
            if let Some(&i) = serial_to_idx.get(&s) {
                mask[i] |= 1 << f_u32;
            }
        }
        if let Some(parent) = file.fragment_parents[f] {
            if let Some(s) = file.fragment_axis_end[f] {
                if let Some(&i) = serial_to_idx.get(&s) {
                    mask[i] |= 1 << parent;
                }
            }
        }
    }
    mask
}

impl Molecule {
    /// True when the molecule has no typed atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad_types::AdType;
    use crate::pdbqt::parse_pdbqt_atoms;

    const LIGAND_FIXTURE: &str = include_str!("../tests/fixtures/1iep_ligand.pdbqt");
    const RECEPTOR_FIXTURE: &str = include_str!("../tests/fixtures/1iep_receptor.pdbqt");

    /// Count heavy + polar-metal atoms in a raw PDBQT (anything that
    /// isn't H, HD, or W). Used as the oracle for what survives
    /// `Molecule` construction.
    fn count_typed_atoms(text: &str) -> usize {
        parse_pdbqt_atoms(text)
            .unwrap()
            .iter()
            .filter(|a| !matches!(a.ad_type, AdType::H | AdType::Hd | AdType::W))
            .count()
    }

    #[test]
    fn ligand_drops_hydrogens_and_keeps_heavy_atoms() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).expect("load");
        let expected = count_typed_atoms(LIGAND_FIXTURE);
        assert_eq!(m.len(), expected);
        assert_eq!(m.xs_types.len(), m.len());
        assert_eq!(m.partial_charges.len(), m.len());
        assert_eq!(m.original_serials.len(), m.len());
        assert_eq!(m.fragment_ids.len(), m.len());
        assert_eq!(m.bonds.len(), m.len());
    }

    #[test]
    fn receptor_drops_hydrogens_and_keeps_heavy_atoms() {
        let m = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).expect("load");
        let expected = count_typed_atoms(RECEPTOR_FIXTURE);
        assert_eq!(m.len(), expected);
        assert!(m.len() > 1000);
    }

    #[test]
    fn coords_and_charges_round_trip_from_source() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        assert_eq!(m.original_serials[0], 1);
        assert_eq!(m.coords[0], [16.600, 51.810, 14.798]);
        assert!((m.partial_charges[0] - (-0.322)).abs() < 1e-9);
    }

    #[test]
    fn serials_are_strictly_increasing() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        assert!(m.original_serials.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn receptor_has_ca_and_nd_xs_types_present() {
        let m = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).unwrap();
        let has_nd = m
            .xs_types
            .iter()
            .any(|&t| matches!(t, XsType::ND | XsType::NDA));
        let has_ch = m.xs_types.contains(&XsType::CH);
        let has_oa = m
            .xs_types
            .iter()
            .any(|&t| matches!(t, XsType::OA | XsType::ODA));
        assert!(has_nd, "receptor should contain N_D or N_DA");
        assert!(has_ch, "receptor should contain C_H");
        assert!(has_oa, "receptor should contain O_A or O_DA");
    }

    #[test]
    fn parse_error_propagates_from_bad_input() {
        let text = "FOOBAR garbage\n";
        assert!(Molecule::from_pdbqt_str(text).is_err());
    }

    #[test]
    fn empty_input_yields_empty_molecule() {
        let m = Molecule::from_pdbqt_str("").unwrap();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    // --- new: fragment IDs + remapped bonds ------------------------------

    #[test]
    fn receptor_has_all_zero_fragments() {
        let m = Molecule::from_pdbqt_str(RECEPTOR_FIXTURE).unwrap();
        assert!(m.fragment_ids.iter().all(|&id| id == 0));
    }

    #[test]
    fn ligand_has_multiple_fragments_and_first_atoms_are_fragment_zero() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        let max_id = *m.fragment_ids.iter().max().unwrap();
        assert!(max_id > 0, "expected ≥2 fragments, got max id {max_id}");
        // The first retained atom (serial 1) sits in ROOT → fragment 0.
        assert_eq!(m.fragment_ids[0], 0);
    }

    #[test]
    fn bond_graph_remains_symmetric_after_filtering() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        for (i, nbrs) in m.bonds.iter().enumerate() {
            for &j in nbrs {
                assert!(
                    m.bonds[j].contains(&i),
                    "bond graph asymmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn bond_graph_has_no_references_to_dropped_atoms() {
        let m = Molecule::from_pdbqt_str(LIGAND_FIXTURE).unwrap();
        for nbrs in &m.bonds {
            for &j in nbrs {
                assert!(j < m.len(), "bond index {j} out of Molecule range");
            }
        }
    }
}
