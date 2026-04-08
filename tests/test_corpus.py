"""Regression tests using the corpus of edge-case PDB fixtures.

RELIABILITY_ROADMAP P2.8-9: Corpus and bug-repro tests.

Each test exercises a specific failure mode that was found during
the 5K validation run or code review.
"""

import os

import numpy as np
import pytest

import ferritin

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def corpus_path(subdir, name):
    return os.path.join(CORPUS, subdir, name)


# =========================================================================
# Insertion codes
# =========================================================================


class TestInsertionCodes:
    """Regression: insertion code interleaving produced garbage omega angles.

    When pdbtbx sorts residues by serial number, insertion-code residues
    (e.g., prosegment "P") get interleaved with main-chain residues.
    CA-CA distances between interleaved pairs are ~30-50 A, producing
    nonsensical dihedral angles.

    Fix: backbone-break detection via CA-CA distance > 4.5 A.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        assert s.atom_count > 0

    def test_dihedrals_break_detected(self):
        s = ferritin.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        phi, psi, omega = ferritin.backbone_dihedrals(s)
        # With backbone-break detection, the interleaved insertion code
        # boundary (res 3 -> 3A, 50 A apart) should produce NaN
        n_nan = np.sum(np.isnan(omega))
        assert n_nan > 0, "Should have NaN at the backbone break"

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        report = ferritin.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, ferritin.PrepReport)


# =========================================================================
# Multi-model
# =========================================================================


class TestMultiModel:
    """Regression: H placement only wrote to model 0.

    Multi-model NMR structures have the same chain in multiple models.
    If H atoms are only placed in model 0, a second call sees models 1+
    still lacking H and adds more — breaking idempotency.

    Fix: scope read pass to first model only.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("multimodel", "two_models.pdb"))
        assert s.model_count == 2

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("multimodel", "two_models.pdb"))
        report = ferritin.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, ferritin.PrepReport)


# =========================================================================
# Alternate conformations
# =========================================================================


class TestAltloc:
    """Edge case: structures with alternate conformations (altloc A/B).

    Atoms with altloc codes should not be double-counted in analysis.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("altloc", "dual_conformer.pdb"))
        assert s.atom_count > 0

    def test_energy_finite(self):
        s = ferritin.load(corpus_path("altloc", "dual_conformer.pdb"))
        e = ferritin.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("altloc", "dual_conformer.pdb"))
        report = ferritin.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, ferritin.PrepReport)


# =========================================================================
# Missing atoms
# =========================================================================


class TestMissingAtoms:
    """Edge case: residues with missing sidechain atoms.

    reconstruct_fragments should be able to add missing atoms.
    Analysis functions should not crash on incomplete residues.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        assert s.atom_count > 0

    def test_energy_finite(self):
        s = ferritin.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        e = ferritin.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_dihedrals_no_crash(self):
        s = ferritin.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        phi, psi, omega = ferritin.backbone_dihedrals(s)
        assert len(phi) > 0

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        report = ferritin.prepare(s, reconstruct=True, minimize=False)
        assert isinstance(report, ferritin.PrepReport)


# =========================================================================
# Chain breaks
# =========================================================================


class TestChainBreaks:
    """Regression: dihedrals computed across sequence gaps.

    When a chain has missing residues (e.g., 1-3, then 7-9), the backbone
    is discontinuous. Dihedrals between residues 3 and 7 should be NaN,
    not computed from distant atoms.

    Fix: CA-CA distance > 4.5 A marks a backbone break.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        assert s.atom_count > 0

    def test_dihedrals_gap_is_nan(self):
        s = ferritin.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        phi, psi, omega = ferritin.backbone_dihedrals(s)
        # There should be NaN values at the gap between residue 3 and 7
        assert np.any(np.isnan(omega)), "Gap should produce NaN omega"

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        report = ferritin.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, ferritin.PrepReport)


# =========================================================================
# Ligands
# =========================================================================


class TestLigands:
    """Edge case: structures with HETATM ligands.

    Ferritin includes HETATM atoms in SASA and energy; FreeSASA does not.
    Selection language should be able to filter protein-only.
    """

    def test_loads(self):
        s = ferritin.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        assert s.atom_count > 0

    def test_has_hetatm(self):
        s = ferritin.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        assert s.atom_count > 9  # 9 protein atoms + 6 ligand atoms

    def test_energy_finite(self):
        s = ferritin.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        e = ferritin.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_prepare_succeeds(self):
        s = ferritin.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        report = ferritin.prepare(s, reconstruct=False, hydrogens="general", minimize=False)
        assert isinstance(report, ferritin.PrepReport)
