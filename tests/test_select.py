"""Tests for atom selection language."""

import os
import numpy as np
import pytest
import ferritin

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pdbtbx", "example-pdbs")


def load_crambin():
    return ferritin.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))


def load_ubiquitin():
    return ferritin.load(os.path.join(EXAMPLE_DIR, "1ubq.pdb"))


class TestBasicSelections:
    def test_all(self):
        s = load_crambin()
        mask = ferritin.select(s, "all")
        assert mask.sum() == s.atom_count

    def test_ca(self):
        s = load_crambin()
        mask = ferritin.select(s, "CA")
        assert mask.sum() == 46

    def test_name_ca(self):
        """Explicit 'name CA' should match shorthand 'CA'."""
        s = load_crambin()
        m1 = ferritin.select(s, "CA")
        m2 = ferritin.select(s, "name CA")
        np.testing.assert_array_equal(m1, m2)

    def test_backbone(self):
        s = load_crambin()
        mask = ferritin.select(s, "backbone")
        assert mask.sum() == 184  # 46 residues * 4 backbone atoms

    def test_chain(self):
        s = load_crambin()
        mask = ferritin.select(s, "chain A")
        assert mask.sum() == s.atom_count  # single chain

    def test_resname(self):
        s = load_crambin()
        mask = ferritin.select(s, "resname ALA")
        assert mask.sum() > 0

    def test_resid_single(self):
        s = load_crambin()
        mask = ferritin.select(s, "resid 1")
        assert mask.sum() > 0
        assert mask.sum() < 15  # one residue

    def test_resid_range(self):
        s = load_crambin()
        mask = ferritin.select(s, "resid 1-10")
        assert mask.sum() > 50

    def test_element(self):
        s = load_crambin()
        mask = ferritin.select(s, "element N")
        assert mask.sum() > 40  # at least one N per residue


class TestBooleanLogic:
    def test_and(self):
        s = load_crambin()
        mask = ferritin.select(s, "CA and chain A")
        assert mask.sum() == 46

    def test_or(self):
        s = load_crambin()
        m_ala = ferritin.select(s, "resname ALA")
        m_gly = ferritin.select(s, "resname GLY")
        m_or = ferritin.select(s, "resname ALA or resname GLY")
        assert m_or.sum() == m_ala.sum() + m_gly.sum()

    def test_not(self):
        s = load_crambin()
        m_bb = ferritin.select(s, "backbone")
        m_not_bb = ferritin.select(s, "not backbone")
        assert m_bb.sum() + m_not_bb.sum() == s.atom_count

    def test_parentheses(self):
        s = load_crambin()
        mask = ferritin.select(s, "(resid 1-5 or resid 40-46) and CA")
        assert mask.sum() == 12  # 5 + 7 residues

    def test_complex(self):
        s = load_crambin()
        mask = ferritin.select(s, "backbone and resid 1-10 and not element O")
        assert mask.sum() > 0
        assert mask.sum() < ferritin.select(s, "backbone and resid 1-10").sum()


class TestKeywords:
    def test_protein(self):
        s = load_ubiquitin()
        m_protein = ferritin.select(s, "protein")
        m_water = ferritin.select(s, "water")
        assert m_protein.sum() + m_water.sum() <= s.atom_count
        assert m_water.sum() > 0  # ubiquitin has water

    def test_water(self):
        s = load_ubiquitin()
        mask = ferritin.select(s, "water")
        assert mask.sum() == 40  # known water count

    def test_heavy(self):
        s = load_crambin()
        m_heavy = ferritin.select(s, "heavy")
        m_hydrogen = ferritin.select(s, "hydrogen")
        assert m_heavy.sum() + m_hydrogen.sum() == s.atom_count

    def test_sidechain(self):
        s = load_crambin()
        m_sc = ferritin.select(s, "sidechain")
        m_bb = ferritin.select(s, "backbone")
        # sidechain + backbone + hydrogen should cover most atoms
        assert m_sc.sum() > 0
        assert m_sc.sum() < s.atom_count


class TestWithCoords:
    def test_select_coords(self):
        s = load_crambin()
        mask = ferritin.select(s, "CA")
        ca_coords = s.coords[mask]
        assert ca_coords.shape == (46, 3)

    def test_select_bfactors(self):
        s = load_crambin()
        mask = ferritin.select(s, "backbone")
        bb_bf = s.b_factors[mask]
        assert len(bb_bf) == 184

    def test_matches_extract_ca(self):
        """select('CA') should give same coords as extract_ca_coords."""
        s = load_crambin()
        mask = ferritin.select(s, "CA")
        ca_select = s.coords[mask]
        ca_extract = ferritin.extract_ca_coords(s)
        np.testing.assert_allclose(ca_select, ca_extract)


class TestEdgeCases:
    def test_empty_result(self):
        s = load_crambin()
        mask = ferritin.select(s, "chain Z")
        assert mask.sum() == 0

    def test_case_sensitive_atom_name(self):
        s = load_crambin()
        m1 = ferritin.select(s, "CA")
        m2 = ferritin.select(s, "ca")  # should not match
        assert m1.sum() == 46
        assert m2.sum() == 0

    def test_case_insensitive_keywords(self):
        s = load_crambin()
        m1 = ferritin.select(s, "CA AND chain A")
        m2 = ferritin.select(s, "CA and chain A")
        np.testing.assert_array_equal(m1, m2)

    def test_invalid_syntax(self):
        s = load_crambin()
        with pytest.raises(ValueError):
            ferritin.select(s, "CA and")  # incomplete

    def test_mask_dtype(self):
        s = load_crambin()
        mask = ferritin.select(s, "CA")
        assert mask.dtype == np.bool_
