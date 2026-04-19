"""Tests for hydrogen bond detection."""

import os
import numpy as np
import pytest
import proteon

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")


def load_crambin():
    return proteon.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))


def load_ubiquitin():
    return proteon.load(os.path.join(EXAMPLE_DIR, "1ubq.pdb"))


class TestBackboneHBonds:
    def test_shape(self):
        hb = proteon.backbone_hbonds(load_crambin())
        assert hb.ndim == 2
        assert hb.shape[1] == 4

    def test_finds_hbonds(self):
        hb = proteon.backbone_hbonds(load_crambin())
        assert len(hb) > 10  # crambin should have ~25-30

    def test_energy_negative(self):
        hb = proteon.backbone_hbonds(load_crambin())
        assert np.all(hb[:, 2] < -0.5)  # all below cutoff

    def test_distance_reasonable(self):
        hb = proteon.backbone_hbonds(load_crambin())
        assert np.all(hb[:, 3] > 2.0)  # O-N > 2A
        assert np.all(hb[:, 3] < 5.5)  # O-N < 5.5A

    def test_stricter_cutoff_fewer_bonds(self):
        hb_loose = proteon.backbone_hbonds(load_crambin(), energy_cutoff=-0.5)
        hb_strict = proteon.backbone_hbonds(load_crambin(), energy_cutoff=-2.0)
        assert len(hb_strict) < len(hb_loose)

    def test_ubiquitin_more_than_crambin(self):
        hb_crn = proteon.backbone_hbonds(load_crambin())
        hb_ubq = proteon.backbone_hbonds(load_ubiquitin())
        assert len(hb_ubq) > len(hb_crn)

    def test_helix_hbonds(self):
        """Alpha helix should have i→i+4 H-bonds."""
        hb = proteon.backbone_hbonds(load_crambin())
        # Check for i+4 pattern (alpha helix)
        i4_bonds = [(int(r[0]), int(r[1])) for r in hb if int(r[1]) - int(r[0]) == 4]
        assert len(i4_bonds) > 5  # crambin has a long helix


class TestGeometricHBonds:
    def test_shape(self):
        ghb = proteon.geometric_hbonds(load_crambin())
        assert ghb.ndim == 2
        assert ghb.shape[1] == 3

    def test_finds_contacts(self):
        ghb = proteon.geometric_hbonds(load_crambin())
        assert len(ghb) > 50

    def test_distance_within_cutoff(self):
        ghb = proteon.geometric_hbonds(load_crambin(), dist_cutoff=3.0)
        assert np.all(ghb[:, 2] <= 3.0)

    def test_tighter_cutoff_fewer(self):
        g35 = proteon.geometric_hbonds(load_crambin(), dist_cutoff=3.5)
        g30 = proteon.geometric_hbonds(load_crambin(), dist_cutoff=3.0)
        assert len(g30) < len(g35)


class TestHBondCount:
    def test_shape(self):
        counts = proteon.hbond_count(load_crambin())
        # Should have one count per amino acid residue
        assert len(counts) > 40

    def test_some_nonzero(self):
        counts = proteon.hbond_count(load_crambin())
        assert (counts > 0).sum() > 10

    def test_reasonable_max(self):
        counts = proteon.hbond_count(load_crambin())
        assert counts.max() <= 6  # no residue should have > 6 backbone H-bonds


class TestBatchHBonds:
    def test_batch(self):
        structures = [load_crambin(), load_ubiquitin()]
        results = proteon.batch_backbone_hbonds(structures, n_threads=-1)
        assert len(results) == 2
        assert len(results[0]) > 10
        assert len(results[1]) > 10

    def test_batch_matches_serial(self):
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_backbone_hbonds(structures, n_threads=-1)
        serial = [proteon.backbone_hbonds(s) for s in structures]
        for b, s in zip(batch, serial):
            assert len(b) == len(s)
