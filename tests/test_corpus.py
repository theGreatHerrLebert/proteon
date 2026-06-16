"""Regression tests using the corpus of edge-case PDB fixtures.

RELIABILITY_ROADMAP P2.8-9: Corpus and bug-repro tests.

Each test exercises a specific failure mode that was found during
the 5K validation run or code review.
"""

import os

import numpy as np
import pytest

import proteon

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
        s = proteon.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        assert s.atom_count > 0

    def test_dihedrals_break_detected(self):
        s = proteon.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        phi, psi, omega = proteon.backbone_dihedrals(s)
        # With backbone-break detection, the interleaved insertion code
        # boundary (res 3 -> 3A, 50 A apart) should produce NaN
        n_nan = np.sum(np.isnan(omega))
        assert n_nan > 0, "Should have NaN at the backbone break"

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)

    def test_supervision_torsions_mask_disconnected_insertion(self):
        """AlphaFold-format torsion supervision must mask pre_omega/phi across the
        interleaved insertion boundary (res 3 -> 3A, ~80 A apart). Numbering alone
        treats 3 -> 3A as bonded (insertion-code-adjacent); the CA-CA geometric
        veto catches the physical break, matching backbone_dihedrals' NaN-ing."""
        s = proteon.load(corpus_path("insertion_codes", "icode_interleave.pdb"))
        ex = proteon.build_structure_supervision_example(s)
        m = ex.torsion_angles_mask
        # Residue order [1, 2, 3, 3A, 4]: 3->3A and 3A->4 cross the 80 A break.
        assert m[3, 0] == 0.0 and m[3, 1] == 0.0  # 3A: disconnected from 3
        assert m[4, 0] == 0.0 and m[4, 1] == 0.0  # 4: disconnected from 3A
        # Residues 1->2->3 are genuine ~3 A bonds — must stay unmasked.
        assert m[1, 0] == 1.0 and m[2, 0] == 1.0


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
        s = proteon.load(corpus_path("multimodel", "two_models.pdb"))
        assert s.model_count == 2

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("multimodel", "two_models.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)


# =========================================================================
# Alternate conformations
# =========================================================================


class TestAltloc:
    """Edge case: structures with alternate conformations (altloc A/B).

    Atoms with altloc codes should not be double-counted in analysis.
    """

    def test_loads(self):
        s = proteon.load(corpus_path("altloc", "dual_conformer.pdb"))
        assert s.atom_count > 0

    def test_energy_finite(self):
        s = proteon.load(corpus_path("altloc", "dual_conformer.pdb"))
        e = proteon.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("altloc", "dual_conformer.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)


# =========================================================================
# Missing atoms
# =========================================================================


class TestMissingAtoms:
    """Edge case: residues with missing sidechain atoms.

    reconstruct_fragments should be able to add missing atoms.
    Analysis functions should not crash on incomplete residues.
    """

    def test_loads(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        assert s.atom_count > 0

    def test_energy_finite(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        e = proteon.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_dihedrals_no_crash(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert len(phi) > 0

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_cb.pdb"))
        report = proteon.prepare(s, reconstruct=True, minimize=False)
        assert isinstance(report, proteon.PrepReport)


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
        s = proteon.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        assert s.atom_count > 0

    def test_dihedrals_gap_is_nan(self):
        s = proteon.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        phi, psi, omega = proteon.backbone_dihedrals(s)
        # There should be NaN values at the gap between residue 3 and 7
        assert np.any(np.isnan(omega)), "Gap should produce NaN omega"

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("chain_breaks", "gap_in_chain.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)


# =========================================================================
# Ligands
# =========================================================================


class TestLigands:
    """Edge case: structures with HETATM ligands.

    Proteon includes HETATM atoms in SASA and energy; FreeSASA does not.
    Selection language should be able to filter protein-only.
    """

    def test_loads(self):
        s = proteon.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        assert s.atom_count > 0

    def test_has_hetatm(self):
        s = proteon.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        assert s.atom_count > 9  # 9 protein atoms + 6 ligand atoms

    def test_energy_finite(self):
        s = proteon.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        e = proteon.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("ligands", "protein_with_ligand.pdb"))
        report = proteon.prepare(s, reconstruct=False, hydrogens="general", minimize=False)
        assert isinstance(report, proteon.PrepReport)


# =========================================================================
# Waters
# =========================================================================


class TestWaters:
    """Edge case: structures with crystallographic HOH waters.

    Solvent must load and be distinguishable from protein (so callers can
    strip it), and must not break energy or prepare. The fixture is 2 ALA
    (10 atoms) plus 3 HOH oxygens placed well away from the protein.
    """

    def test_loads(self):
        s = proteon.load(corpus_path("waters", "protein_with_waters.pdb"))
        assert s.atom_count == 13  # 10 protein + 3 water oxygens

    def test_waters_are_distinguishable_from_protein(self):
        s = proteon.load(corpus_path("waters", "protein_with_waters.pdb"))
        # residue_names is per-atom; solvent is labelled HOH, protein ALA.
        names = list(s.residue_names)
        assert names.count("HOH") == 3, "three water oxygens expected"
        assert names.count("ALA") == 10, "ten protein atoms expected"

    def test_energy_finite(self):
        s = proteon.load(corpus_path("waters", "protein_with_waters.pdb"))
        e = proteon.compute_energy(s)
        assert np.isfinite(e["total"]), "waters must not make the energy non-finite"

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("waters", "protein_with_waters.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)


# =========================================================================
# Missing backbone atoms
# =========================================================================


class TestMissingBackbone:
    """Edge case: a residue missing a BACKBONE atom (the carbonyl C of res 2).

    Distinct from TestMissingAtoms (a missing sidechain CB): a missing
    backbone atom breaks the dihedral chain around that residue, so the
    affected phi/psi/omega must come back NaN rather than be computed from
    the wrong atoms or crash. Energy must stay finite and prepare must not
    crash on the incomplete backbone.
    """

    def test_loads(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_backbone_c.pdb"))
        assert s.atom_count == 11  # res 2 is missing its C

    def test_backbone_c_of_residue_2_is_absent(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_backbone_c.pdb"))
        serials = list(s.residue_serial_numbers)
        names = list(s.atom_names)
        res2_atoms = {a for ser, a in zip(serials, names) if ser == 2}
        assert "C" not in res2_atoms, "fixture must omit the backbone C of res 2"
        assert {"N", "CA", "O"} <= res2_atoms, "res 2 keeps its other backbone atoms"

    def test_dihedrals_nan_at_break_no_crash(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_backbone_c.pdb"))
        phi, psi, omega = proteon.backbone_dihedrals(s)
        assert len(phi) > 0
        # The missing carbonyl C breaks psi(res2)/omega(res2-3)/phi(res3).
        assert np.any(np.isnan(psi)) or np.any(np.isnan(omega)), (
            "a missing backbone atom must NaN the affected dihedrals"
        )

    def test_energy_finite(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_backbone_c.pdb"))
        e = proteon.compute_energy(s)
        assert np.isfinite(e["total"])

    def test_prepare_succeeds(self):
        s = proteon.load(corpus_path("missing_atoms", "missing_backbone_c.pdb"))
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)
