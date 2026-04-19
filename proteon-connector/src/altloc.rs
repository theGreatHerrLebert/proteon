//! Altloc-aware iteration helpers over pdbtbx hierarchies.
//!
//! pdbtbx's parser represents alternate conformations with one
//! [`pdbtbx::Conformer`] per altLoc character ("A", "B", ...). For a
//! residue with altlocs, the parser copies the **non-altloc** backbone
//! atoms (altLoc blank, occupancy 1.00) into *every* altloc conformer,
//! dividing their occupancy across the copies. Calling the standard
//! [`pdbtbx::Residue::atoms`] or [`pdbtbx::PDB::atoms`] iterators then
//! yields the backbone atoms once per altloc. The downstream effect is
//! a flat atom list containing duplicated entries at **identical
//! coordinates**, which detonates anything that builds bonds or non-
//! bonded pair lists by coordinate (topology builders, SASA grids,
//! neighbor lists).
//!
//! This module provides *primary-conformer* iteration: for each residue
//! we pick one conformer — blank altLoc first, then "A", then the first
//! available — and yield only that conformer's atoms. The result is a
//! consistent, non-duplicated view of a single physical realization of
//! the structure. Every proteon code path that needs a flat atom list
//! should use these helpers rather than `pdbtbx::*::atoms` directly.
//!
//! `proteon-io/loader.rs` already does this manually for the alignment
//! path; this module centralizes the same logic for the
//! `proteon-connector` Python/force-field side.

use pdbtbx::{Chain, Conformer, Model, Residue, PDB};

/// Pick the primary conformer of a residue: blank altLoc first, then "A",
/// then the first available conformer. Returns `None` only for an empty
/// residue (no conformers at all).
pub fn primary_conformer(residue: &Residue) -> Option<&Conformer> {
    residue
        .conformers()
        .find(|c| c.alternative_location().is_none())
        .or_else(|| {
            residue
                .conformers()
                .find(|c| c.alternative_location() == Some("A"))
        })
        .or_else(|| residue.conformers().next())
}

/// Atoms of a residue's primary conformer only.
pub fn residue_atoms_primary(residue: &Residue) -> impl Iterator<Item = &pdbtbx::Atom> + '_ {
    primary_conformer(residue)
        .into_iter()
        .flat_map(|c| c.atoms())
}

/// Number of atoms in a residue's primary conformer only.
pub fn residue_atom_count_primary(residue: &Residue) -> usize {
    primary_conformer(residue).map_or(0, |c| c.atom_count())
}

/// Atoms of a chain via primary-conformer-per-residue iteration.
pub fn chain_atoms_primary(chain: &Chain) -> impl Iterator<Item = &pdbtbx::Atom> + '_ {
    chain.residues().flat_map(residue_atoms_primary)
}

/// Atom count for a chain, counting each residue's primary conformer only.
pub fn chain_atom_count_primary(chain: &Chain) -> usize {
    chain.residues().map(residue_atom_count_primary).sum()
}

/// Atoms of a model via primary-conformer-per-residue iteration.
pub fn model_atoms_primary(model: &Model) -> impl Iterator<Item = &pdbtbx::Atom> + '_ {
    model.chains().flat_map(chain_atoms_primary)
}

/// Atom count for a model, counting each residue's primary conformer only.
pub fn model_atom_count_primary(model: &Model) -> usize {
    model.chains().map(chain_atom_count_primary).sum()
}

/// Atoms of a PDB's first model via primary-conformer iteration.
///
/// Returns a boxed iterator because the `None`-model and `Some`-model
/// branches have different concrete types.
pub fn pdb_atoms_primary(pdb: &PDB) -> Box<dyn Iterator<Item = &pdbtbx::Atom> + '_> {
    match pdb.models().next() {
        Some(m) => Box::new(model_atoms_primary(m)),
        None => Box::new(std::iter::empty()),
    }
}

/// Atom count for a PDB's first model, primary conformers only.
pub fn pdb_atom_count_primary(pdb: &PDB) -> usize {
    pdb.models().next().map_or(0, model_atom_count_primary)
}

/// Total atom count across ALL models, primary conformers only.
pub fn pdb_total_atom_count_primary(pdb: &PDB) -> usize {
    pdb.models().map(model_atom_count_primary).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: parse a PDB with altloc residues and verify the
    /// primary-conformer iteration returns fewer atoms than the naive
    /// pdbtbx::PDB::atoms iteration, and that the primary-conformer
    /// iteration yields unique coordinates.
    ///
    /// Requires a test PDB with altlocs — skipped if unavailable.
    #[test]
    fn altloc_iteration_dedupes_backbone() {
        // Try to find a small altloc test file. If none is checked in,
        // this test is a no-op. 1aie.pdb is the canonical reproducer.
        let candidates = [
            "../tests/data/1aie.pdb",
            "tests/data/1aie.pdb",
            "/tmp/bucket_a_pdbs/1aie.pdb",
        ];
        let path = candidates.iter().find(|p| std::path::Path::new(p).exists());
        let Some(path) = path else {
            eprintln!("altloc test: no fixture available, skipping");
            return;
        };

        let (pdb, _warnings) = pdbtbx::ReadOptions::new()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(path)
            .expect("parse test PDB");

        let naive: usize = pdb.atoms().count();
        let primary: usize = pdb_atoms_primary(&pdb).count();
        assert!(
            primary < naive,
            "expected primary count ({}) < naive count ({}) for altloc structure",
            primary,
            naive,
        );

        // No exact-duplicate coordinates in the primary view.
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for atom in pdb_atoms_primary(&pdb) {
            let (x, y, z) = atom.pos();
            let key = (x.to_bits(), y.to_bits(), z.to_bits());
            assert!(
                seen.insert(key),
                "duplicate coord in primary-conformer iteration: ({:.3}, {:.3}, {:.3})",
                x,
                y,
                z,
            );
        }
    }
}
