"""P3 geometry hazards: chain gaps and CA chirality outliers.

  - A chain gap (broken peptide bond / missing residues) creates a FALSE
    sequential edge in graph / (chain, resnum)-indexed labels -> blocks
    sequence_indexed. The present residues' coordinates are still fine.
  - A CA chirality outlier (D-amino acid or modeling error) is a coordinate-
    geometry anomaly -> blocks heavy_coords.
"""

import os

import pytest

import proteon
from proteon import PrepReport

PDBS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


class TestContractLogic:
    def test_chain_gap_blocks_sequence_not_coords(self):
        r = PrepReport(hydrogens_added=50, n_chain_gaps=1)
        assert "chain_gaps" in r.label_hazards
        assert r.label_safe_heavy_coords is True        # present residues are fine
        assert r.label_safe_sequence_indexed is False    # false adjacency
        assert r.label_safe is False

    def test_chirality_outlier_blocks_coords(self):
        r = PrepReport(hydrogens_added=50, n_chirality_outliers=1)
        assert "chirality_outliers" in r.label_hazards
        assert r.label_safe_heavy_coords is False
        assert r.label_safe is False


class TestDetection:
    def test_clean_L_protein_no_outliers(self):
        # crambin is all-L and complete: zero chirality outliers, zero gaps
        # (this is the calibration anchor for the chirality sign).
        r = proteon.prepare(proteon.load(os.path.join(PDBS, "1crn.pdb")))
        assert r.n_chirality_outliers == 0
        assert r.n_chain_gaps == 0
        assert r.has_chirality_outliers is False
        assert r.has_chain_gaps is False

    def test_chain_break_detected(self):
        r = proteon.prepare(proteon.load(os.path.join(CORPUS, "chain_breaks", "gap_in_chain.pdb")))
        assert r.n_chain_gaps >= 1
        assert r.has_chain_gaps is True
        assert r.label_safe_sequence_indexed is False

    @pytest.mark.parametrize("name", ["1ubq", "1bpi", "1enh", "1ake"])
    def test_standard_proteins_have_no_chirality_outliers(self, name):
        # All standard L-proteins: the chirality detector must not false-positive.
        r = proteon.prepare(proteon.load(os.path.join(PDBS, f"{name}.pdb")))
        assert r.n_chirality_outliers == 0
