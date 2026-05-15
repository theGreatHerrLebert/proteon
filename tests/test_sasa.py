"""Tests for SASA (Solvent Accessible Surface Area).

Tests the Shrake-Rupley implementation against known values
and validates against Biopython as oracle.
"""

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


# ===========================================================================
# atom_sasa
# ===========================================================================


class TestAtomSASA:
    def test_shape(self):
        s = load_crambin()
        sasa = proteon.atom_sasa(s)
        assert sasa.shape == (s.atom_count,)

    def test_non_negative(self):
        s = load_crambin()
        sasa = proteon.atom_sasa(s)
        assert np.all(sasa >= 0)

    def test_total_reasonable(self):
        """Crambin total SASA should be ~2500-3500 A²."""
        s = load_crambin()
        total = proteon.atom_sasa(s).sum()
        assert 2500 < total < 3500, f"Total SASA {total:.0f} out of range"

    def test_some_buried(self):
        """Some atoms should be completely buried (SASA = 0)."""
        s = load_crambin()
        sasa = proteon.atom_sasa(s)
        n_buried = (sasa == 0).sum()
        assert n_buried > 0, "No buried atoms found"

    def test_some_exposed(self):
        """Most atoms should have nonzero SASA."""
        s = load_crambin()
        sasa = proteon.atom_sasa(s)
        n_exposed = (sasa > 0).sum()
        assert n_exposed > s.atom_count * 0.4, "Too few exposed atoms"

    def test_different_n_points(self):
        """More points should give a more precise answer but similar total."""
        s = load_crambin()
        sasa_100 = proteon.atom_sasa(s, n_points=100).sum()
        sasa_960 = proteon.atom_sasa(s, n_points=960).sum()
        # Should agree within ~5%
        assert abs(sasa_100 - sasa_960) / sasa_960 < 0.05

    def test_probe_radius_effect(self):
        """Different probe radius should give different SASA."""
        s = load_crambin()
        sasa_14 = proteon.atom_sasa(s, probe=1.4).sum()
        sasa_20 = proteon.atom_sasa(s, probe=2.0).sum()
        # Different probe should give different result
        assert sasa_14 != sasa_20

    def test_ubiquitin(self):
        """Test on a different structure."""
        s = load_ubiquitin()
        sasa = proteon.atom_sasa(s)
        assert sasa.shape == (s.atom_count,)
        assert np.all(sasa >= 0)
        # Ubiquitin total SASA should be larger than crambin
        assert sasa.sum() > 3000


# ===========================================================================
# residue_sasa
# ===========================================================================


class TestResidueSASA:
    def test_shape(self):
        s = load_crambin()
        res_sasa = proteon.residue_sasa(s)
        assert res_sasa.shape == (s.residue_count,)

    def test_sum_matches_atom(self):
        """Residue SASA sum should equal total atom SASA."""
        s = load_crambin()
        atom_total = proteon.atom_sasa(s).sum()
        res_total = proteon.residue_sasa(s).sum()
        np.testing.assert_allclose(atom_total, res_total, rtol=1e-10)

    def test_non_negative(self):
        s = load_crambin()
        res_sasa = proteon.residue_sasa(s)
        assert np.all(res_sasa >= 0)


# ===========================================================================
# relative_sasa (RSA)
# ===========================================================================


class TestRelativeSASA:
    def test_shape(self):
        s = load_crambin()
        rsa = proteon.relative_sasa(s)
        assert rsa.shape == (s.residue_count,)

    def test_range(self):
        """RSA should be between 0 and ~1.5 for standard residues."""
        s = load_crambin()
        rsa = proteon.relative_sasa(s)
        valid = rsa[~np.isnan(rsa)]
        assert np.all(valid >= 0)
        assert np.all(valid < 2.0)  # very generous upper bound

    def test_buried_residues(self):
        """Some residues should be buried (RSA < 0.25)."""
        s = load_crambin()
        rsa = proteon.relative_sasa(s)
        valid = rsa[~np.isnan(rsa)]
        n_buried = (valid < 0.25).sum()
        assert n_buried > 0


# ===========================================================================
# total_sasa
# ===========================================================================


class TestTotalSASA:
    def test_matches_atom_sum(self):
        s = load_crambin()
        total = proteon.total_sasa(s)
        atom_sum = proteon.atom_sasa(s).sum()
        np.testing.assert_allclose(total, atom_sum, rtol=1e-10)

    def test_crambin_range(self):
        total = proteon.total_sasa(load_crambin())
        assert 2500 < total < 3500


# ===========================================================================
# Oracle: compare with Biopython
# ===========================================================================


class TestBiopythonOracle:
    @pytest.fixture
    def crambin_biopython_sasa(self):
        """Compute Biopython SASA for crambin."""
        try:
            from Bio.PDB import PDBParser
            from Bio.PDB.SASA import ShrakeRupley
        except ImportError:
            pytest.skip("Biopython not installed")

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("1crn", os.path.join(TEST_PDBS_DIR, "1crn.pdb"))
        sr = ShrakeRupley()
        sr.compute(struct, level="A")
        return sum(a.sasa for a in struct.get_atoms())

    def test_total_sasa_matches_biopython(self, crambin_biopython_sasa):
        """Proteon total SASA should be within 2% of Biopython."""
        s = load_crambin()
        proteon_total = proteon.total_sasa(s)
        bp_total = crambin_biopython_sasa
        rel_diff = abs(proteon_total - bp_total) / bp_total
        assert rel_diff < 0.02, (
            f"Proteon {proteon_total:.1f} vs Biopython {bp_total:.1f} "
            f"({rel_diff*100:.1f}% diff)"
        )


# ===========================================================================
# Batch SASA
# ===========================================================================


class TestBatchSASA:
    def test_batch_total_sasa(self):
        structures = [load_crambin(), load_ubiquitin()]
        totals = proteon.batch_total_sasa(structures, n_threads=1)
        assert totals.shape == (2,)
        assert np.all(totals > 0)

    def test_batch_matches_serial(self):
        structures = [load_crambin(), load_ubiquitin()]
        totals_batch = proteon.batch_total_sasa(structures, n_threads=-1)
        totals_serial = np.array([proteon.total_sasa(s) for s in structures])
        np.testing.assert_allclose(totals_batch, totals_serial, rtol=1e-10)

    def test_batch_atom_sasa_matches_serial(self):
        """batch_atom_sasa must equal a Python loop of atom_sasa per structure."""
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_atom_sasa(structures, n_threads=-1)
        serial = [proteon.atom_sasa(s) for s in structures]
        assert len(batch) == len(serial)
        for b, s in zip(batch, serial):
            assert b.shape == s.shape
            np.testing.assert_array_equal(b, s)

    def test_batch_residue_sasa_matches_serial(self):
        """batch_residue_sasa must equal a loop of residue_sasa per structure."""
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_residue_sasa(structures, n_threads=-1)
        serial = [proteon.residue_sasa(s) for s in structures]
        for b, s in zip(batch, serial):
            np.testing.assert_array_equal(b, s)

    def test_batch_relative_sasa_matches_serial(self):
        """batch_relative_sasa must equal a loop of relative_sasa per structure."""
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_relative_sasa(structures, n_threads=-1)
        serial = [proteon.relative_sasa(s) for s in structures]
        for b, s in zip(batch, serial):
            # NaN for non-standard residues; equal_nan to compare exactly.
            np.testing.assert_array_equal(b, s)


# ===========================================================================
# load_and_sasa
# ===========================================================================


class TestLoadAndSASA:
    def test_basic(self):
        paths = [os.path.join(TEST_PDBS_DIR, "1crn.pdb")]
        result = proteon.load_and_sasa(paths, n_threads=1)
        assert len(result) == 1
        assert result.n_attempted == 1
        assert result.n_ok == 1
        assert result.n_failed == 0
        item = result[0]
        assert item.index == 0
        assert item.ok
        assert 2500 < item.value < 3500

    def test_bad_files_recorded_not_skipped(self):
        """A file that fails to load is a failed item, not an omission."""
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            "/nonexistent/file.pdb",
        ]
        result = proteon.load_and_sasa(paths, n_threads=1)
        # Cardinality is preserved: one item per input, in input order.
        assert len(result) == 2
        assert result.n_attempted == 2
        assert result.n_ok == 1
        assert result.n_failed == 1
        assert result[0].ok and result[0].value > 0
        assert not result[1].ok
        assert result[1].error  # carries the load error message
        assert result.failures == [(1, result[1].error)]

    def test_strict_raises_on_failure(self):
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            "/nonexistent/file.pdb",
        ]
        with pytest.raises(RuntimeError):
            proteon.load_and_sasa(paths, n_threads=1, strict=True)
