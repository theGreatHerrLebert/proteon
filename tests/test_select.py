"""Tests for atom selection language."""

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


class TestBasicSelections:
    def test_all(self):
        s = load_crambin()
        mask = proteon.select(s, "all")
        assert mask.sum() == s.atom_count

    def test_ca(self):
        s = load_crambin()
        mask = proteon.select(s, "CA")
        assert mask.sum() == 46

    def test_name_ca(self):
        """Explicit 'name CA' should match shorthand 'CA'."""
        s = load_crambin()
        m1 = proteon.select(s, "CA")
        m2 = proteon.select(s, "name CA")
        np.testing.assert_array_equal(m1, m2)

    def test_backbone(self):
        s = load_crambin()
        mask = proteon.select(s, "backbone")
        assert mask.sum() == 184  # 46 residues * 4 backbone atoms

    def test_chain(self):
        s = load_crambin()
        mask = proteon.select(s, "chain A")
        assert mask.sum() == s.atom_count  # single chain

    def test_resname(self):
        s = load_crambin()
        mask = proteon.select(s, "resname ALA")
        assert mask.sum() > 0

    def test_resid_single(self):
        s = load_crambin()
        mask = proteon.select(s, "resid 1")
        assert mask.sum() > 0
        assert mask.sum() < 15  # one residue

    def test_resid_range(self):
        s = load_crambin()
        mask = proteon.select(s, "resid 1-10")
        assert mask.sum() > 50

    def test_element(self):
        s = load_crambin()
        mask = proteon.select(s, "element N")
        assert mask.sum() > 40  # at least one N per residue


class TestBooleanLogic:
    def test_and(self):
        s = load_crambin()
        mask = proteon.select(s, "CA and chain A")
        assert mask.sum() == 46

    def test_or(self):
        s = load_crambin()
        m_ala = proteon.select(s, "resname ALA")
        m_gly = proteon.select(s, "resname GLY")
        m_or = proteon.select(s, "resname ALA or resname GLY")
        assert m_or.sum() == m_ala.sum() + m_gly.sum()

    def test_not(self):
        s = load_crambin()
        m_bb = proteon.select(s, "backbone")
        m_not_bb = proteon.select(s, "not backbone")
        assert m_bb.sum() + m_not_bb.sum() == s.atom_count

    def test_parentheses(self):
        s = load_crambin()
        mask = proteon.select(s, "(resid 1-5 or resid 40-46) and CA")
        assert mask.sum() == 12  # 5 + 7 residues

    def test_complex(self):
        s = load_crambin()
        mask = proteon.select(s, "backbone and resid 1-10 and not element O")
        assert mask.sum() > 0
        assert mask.sum() < proteon.select(s, "backbone and resid 1-10").sum()


class TestKeywords:
    def test_protein(self):
        s = load_ubiquitin()
        m_protein = proteon.select(s, "protein")
        m_water = proteon.select(s, "water")
        assert m_protein.sum() + m_water.sum() <= s.atom_count
        assert m_water.sum() > 0  # ubiquitin has water

    def test_water(self):
        s = load_ubiquitin()
        mask = proteon.select(s, "water")
        assert mask.sum() == 40  # known water count

    def test_heavy(self):
        s = load_crambin()
        m_heavy = proteon.select(s, "heavy")
        m_hydrogen = proteon.select(s, "hydrogen")
        assert m_heavy.sum() + m_hydrogen.sum() == s.atom_count

    def test_sidechain(self):
        s = load_crambin()
        m_sc = proteon.select(s, "sidechain")
        m_bb = proteon.select(s, "backbone")
        # sidechain + backbone + hydrogen should cover most atoms
        assert m_sc.sum() > 0
        assert m_sc.sum() < s.atom_count


class TestWithCoords:
    def test_select_coords(self):
        s = load_crambin()
        mask = proteon.select(s, "CA")
        ca_coords = s.coords[mask]
        assert ca_coords.shape == (46, 3)

    def test_select_bfactors(self):
        s = load_crambin()
        mask = proteon.select(s, "backbone")
        bb_bf = s.b_factors[mask]
        assert len(bb_bf) == 184

    def test_matches_extract_ca(self):
        """select('CA') should give same coords as extract_ca_coords."""
        s = load_crambin()
        mask = proteon.select(s, "CA")
        ca_select = s.coords[mask]
        ca_extract = proteon.extract_ca_coords(s)
        np.testing.assert_allclose(ca_select, ca_extract)


class TestEdgeCases:
    def test_empty_result(self):
        s = load_crambin()
        mask = proteon.select(s, "chain Z")
        assert mask.sum() == 0

    def test_case_sensitive_atom_name(self):
        s = load_crambin()
        m1 = proteon.select(s, "CA")
        m2 = proteon.select(s, "ca")  # should not match
        assert m1.sum() == 46
        assert m2.sum() == 0

    def test_case_insensitive_keywords(self):
        s = load_crambin()
        m1 = proteon.select(s, "CA AND chain A")
        m2 = proteon.select(s, "CA and chain A")
        np.testing.assert_array_equal(m1, m2)

    def test_invalid_syntax(self):
        s = load_crambin()
        with pytest.raises(ValueError):
            proteon.select(s, "CA and")  # incomplete

    def test_mask_dtype(self):
        s = load_crambin()
        mask = proteon.select(s, "CA")
        assert mask.dtype == np.bool_
