"""P2 chemistry hazards: non-standard residues and metals.

These are label hazards that don't corrupt the heavy COORDINATES (the atoms are
observed) but do corrupt sequence-indexed and energy labels:
  - a modified residue (MSE, SEP, ...) is not a canonical sequence token and has
    no force-field typing;
  - a metal's coordination chemistry is not modelled by the protein-only FF.
"""

import os

import pytest

import proteon
from proteon import PrepReport

PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


class TestContractLogic:
    def test_nonstandard_blocks_seq_and_energy_not_coords(self):
        r = PrepReport(hydrogens_added=50, has_nonstandard_residues=True)
        assert "nonstandard_residues" in r.label_hazards
        assert r.label_safe_heavy_coords is True       # atoms are observed
        assert r.label_safe_all_atom_coords is True
        assert r.label_safe_sequence_indexed is False  # not a canonical token
        assert r.label_safe_energy is False            # no FF typing
        assert r.label_safe is False

    def test_metals_block_energy_not_coords_or_sequence(self):
        r = PrepReport(hydrogens_added=50, has_metals=True)
        assert "metals" in r.label_hazards
        assert r.label_safe_heavy_coords is True
        assert r.label_safe_sequence_indexed is True   # metal doesn't shift residue ids
        assert r.label_safe_energy is False            # coordination not modelled
        assert r.label_safe is False


class TestDetection:
    def test_metalloprotein_flagged(self):
        # 4hhb carries heme iron (Fe).
        r = proteon.prepare(proteon.load(os.path.join(PDBS, "4hhb.pdb")))
        assert r.has_metals is True
        assert "metals" in r.label_hazards
        assert r.label_safe_energy is False

    def test_clean_protein_no_chemistry_hazards(self):
        r = proteon.prepare(proteon.load(os.path.join(PDBS, "1crn.pdb")))
        assert r.has_metals is False
        assert r.has_nonstandard_residues is False

    def test_modified_residue_flagged(self):
        # A fixture with one residue renamed to MSE (selenomethionine).
        fixture = os.path.join(CORPUS, "nonstandard", "has_mse.pdb")
        r = proteon.prepare(proteon.load(fixture))
        assert r.has_nonstandard_residues is True
        assert "nonstandard_residues" in r.label_hazards
        assert r.label_safe_sequence_indexed is False
