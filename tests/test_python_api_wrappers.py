"""Smoke tests for the Python-wrapper modules.

The existing `test_geometry.py`, `test_analysis.py`, etc. call
`from ferritin_connector import py_geometry` and invoke the PyO3 Rust
classes directly — that exercises the Rust code, but never touches the
one-line Python wrapper functions in `ferritin/geometry.py`,
`ferritin/dssp.py`, `ferritin/hydrogens.py`, etc. The behavior is
safely tested (via the connector path); the shim lines at the Python
public-API boundary are not.

This file closes that gap with a direct smoke-test per public wrapper
function: "if a user calls `ferritin.rmsd(x, y)` from their script,
does the wrapper forward correctly to the Rust backend?"

No attempt at algorithmic depth — those assertions live in the
dedicated `test_<module>.py` files. The invariant being pinned here is
"the Python public API dispatches successfully and returns a shape /
type the user would expect."
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import ferritin

TEST_PDBS = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
CRAMBIN = os.path.join(TEST_PDBS, "1crn.pdb")
UBIQ = os.path.join(TEST_PDBS, "1ubq.pdb")


# ===========================================================================
# geometry.py
# ===========================================================================


class TestGeometryWrappers:
    """Cover ferritin.kabsch_superpose / rmsd / rmsd_no_super /
    apply_transform / assign_secondary_structure / tm_score."""

    @pytest.fixture
    def coords(self):
        return np.asarray(ferritin.load(UBIQ).coords, dtype=np.float64)

    def test_kabsch_superpose_self_rmsd_zero(self, coords):
        rmsd, rot, trans = ferritin.kabsch_superpose(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        # Rotation on identical coords is the identity.
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-8)

    def test_rmsd_self_zero(self, coords):
        assert ferritin.rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_no_super_nonzero_after_translation(self, coords):
        translated = coords + np.array([1.0, 0.0, 0.0])
        # rmsd_no_super does NOT superpose — a pure translation shows up.
        rmsd = ferritin.rmsd_no_super(coords, translated)
        assert rmsd == pytest.approx(1.0, rel=1e-6)
        # rmsd (with superposition) should collapse to zero.
        assert ferritin.rmsd(coords, translated) == pytest.approx(0.0, abs=1e-8)

    def test_apply_transform_identity(self, coords):
        out = ferritin.apply_transform(
            coords, np.eye(3), np.zeros(3),
        )
        np.testing.assert_allclose(out, coords, atol=1e-12)

    def test_apply_transform_known_translation(self, coords):
        delta = np.array([1.5, -2.0, 0.5])
        out = ferritin.apply_transform(coords, np.eye(3), delta)
        # Each row should differ from the corresponding input row by
        # the same translation vector.
        diff = out - coords
        np.testing.assert_allclose(diff, np.broadcast_to(delta, diff.shape), atol=1e-8)

    def test_assign_secondary_structure_returns_string(self, coords):
        ss = ferritin.assign_secondary_structure(coords)
        assert isinstance(ss, str)
        assert len(ss) == coords.shape[0]
        # Characters should all be from the {H, E, T, C} alphabet.
        assert set(ss).issubset(set("HETC"))

    def test_tm_score_self_alignment(self, coords):
        # Trivial self-alignment: invmap[j] = j for every residue.
        n = coords.shape[0]
        invmap = np.arange(n, dtype=np.int32)
        score, n_aligned, rmsd, _rot, _trans = ferritin.tm_score(
            coords, coords, invmap,
        )
        assert score == pytest.approx(1.0, abs=1e-6)
        assert n_aligned == n
        assert rmsd == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# dssp.py
# ===========================================================================


class TestDsspWrappers:
    """Cover ferritin.dssp / dssp_array / batch_dssp / load_and_dssp."""

    def test_dssp_returns_string(self):
        s = ferritin.load(CRAMBIN)
        ss = ferritin.dssp(s)
        assert isinstance(ss, str)
        assert len(ss) == 46  # crambin amino acids
        assert set(ss).issubset(set("HGIEBTSC"))

    def test_dssp_array_matches_string(self):
        s = ferritin.load(CRAMBIN)
        ss_str = ferritin.dssp(s)
        ss_arr = ferritin.dssp_array(s)
        assert ss_arr.shape == (len(ss_str),)
        # ASCII codes per residue.
        assert "".join(chr(c) for c in ss_arr) == ss_str

    def test_batch_dssp_matches_single(self):
        structures = [ferritin.load(CRAMBIN), ferritin.load(UBIQ)]
        batch = ferritin.batch_dssp(structures, n_threads=1)
        singles = [ferritin.dssp(s) for s in structures]
        assert batch == singles

    def test_load_and_dssp_returns_index_string_tuples(self):
        results = ferritin.load_and_dssp([CRAMBIN, UBIQ], n_threads=1)
        assert len(results) == 2
        for idx, ss in results:
            assert isinstance(idx, int)
            assert isinstance(ss, str)
        # The indices should be the original positions (0, 1).
        indices = sorted(idx for idx, _ in results)
        assert indices == [0, 1]


# ===========================================================================
# hydrogens.py
# ===========================================================================


class TestHydrogensWrappers:
    """Cover ferritin.place_peptide_hydrogens / place_all_hydrogens /
    place_general_hydrogens / reconstruct_fragments /
    batch_place_peptide_hydrogens."""

    def test_place_peptide_hydrogens_returns_counts(self):
        s = ferritin.load(CRAMBIN)
        added, skipped = ferritin.place_peptide_hydrogens(s)
        assert isinstance(added, int)
        assert isinstance(skipped, int)
        assert added + skipped > 0

    def test_place_peptide_hydrogens_with_coords_returns_tuple(self):
        s = ferritin.load(CRAMBIN)
        (added, skipped), coords = ferritin.place_peptide_hydrogens(
            s, return_coords=True,
        )
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3
        assert coords.shape[0] == added

    def test_place_peptide_hydrogens_idempotent(self):
        s = ferritin.load(CRAMBIN)
        a1, _ = ferritin.place_peptide_hydrogens(s)
        a2, _ = ferritin.place_peptide_hydrogens(s)
        # Second call adds nothing — contract promised in the docstring.
        assert a2 == 0
        assert a1 > 0

    def test_place_all_hydrogens_adds_more_than_peptide_only(self):
        # place_all = backbone + sidechain. Starting from a fresh load,
        # place_all should add strictly more H than place_peptide would.
        s_all = ferritin.load(CRAMBIN)
        added_all, _ = ferritin.place_all_hydrogens(s_all)

        s_peptide = ferritin.load(CRAMBIN)
        added_peptide, _ = ferritin.place_peptide_hydrogens(s_peptide)

        assert added_all > added_peptide

    def test_place_general_hydrogens_returns_counts(self):
        s = ferritin.load(CRAMBIN)
        added, skipped = ferritin.place_general_hydrogens(s)
        assert isinstance(added, int)
        assert isinstance(skipped, int)
        assert added > 0

    def test_reconstruct_fragments_returns_int(self):
        s = ferritin.load(CRAMBIN)
        n_added = ferritin.reconstruct_fragments(s)
        # Crambin has complete heavy atoms, so likely 0 added — but the
        # return type contract is what we're pinning here.
        assert isinstance(n_added, int)
        assert n_added >= 0

    def test_batch_place_peptide_hydrogens_parallel(self):
        structures = [ferritin.load(CRAMBIN), ferritin.load(UBIQ)]
        results = ferritin.batch_place_peptide_hydrogens(
            structures, n_threads=1,
        )
        assert len(results) == 2
        for added, skipped in results:
            assert isinstance(added, int)
            assert isinstance(skipped, int)
