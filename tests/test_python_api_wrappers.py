"""Smoke tests for the Python-wrapper modules.

The existing `test_geometry.py`, `test_analysis.py`, etc. call
`from proteon_connector import py_geometry` and invoke the PyO3 Rust
classes directly — that exercises the Rust code, but never touches the
one-line Python wrapper functions in `proteon/geometry.py`,
`proteon/dssp.py`, `proteon/hydrogens.py`, etc. The behavior is
safely tested (via the connector path); the shim lines at the Python
public-API boundary are not.

This file closes that gap with a direct smoke-test per public wrapper
function: "if a user calls `proteon.rmsd(x, y)` from their script,
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

import proteon

TEST_PDBS = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
CRAMBIN = os.path.join(TEST_PDBS, "1crn.pdb")
UBIQ = os.path.join(TEST_PDBS, "1ubq.pdb")
# Multi-chain complex for MM-align wrapper tests — lives in the sibling
# test-pdbs/ directory, same layout the existing tests/test_mmalign.py uses.
SHARED_PDBS = os.path.join(os.path.dirname(__file__), "..", "..", "test-pdbs")
HHB = os.path.join(SHARED_PDBS, "4hhb.pdb")


# ===========================================================================
# geometry.py
# ===========================================================================


class TestGeometryWrappers:
    """Cover proteon.kabsch_superpose / rmsd / rmsd_no_super /
    apply_transform / assign_secondary_structure / tm_score."""

    @pytest.fixture
    def coords(self):
        return np.asarray(proteon.load(UBIQ).coords, dtype=np.float64)

    def test_kabsch_superpose_self_rmsd_zero(self, coords):
        rmsd, rot, trans = proteon.kabsch_superpose(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        # Rotation on identical coords is the identity.
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-8)

    def test_rmsd_self_zero(self, coords):
        assert proteon.rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_no_super_nonzero_after_translation(self, coords):
        translated = coords + np.array([1.0, 0.0, 0.0])
        # rmsd_no_super does NOT superpose — a pure translation shows up.
        rmsd = proteon.rmsd_no_super(coords, translated)
        assert rmsd == pytest.approx(1.0, rel=1e-6)
        # rmsd (with superposition) should collapse to zero.
        assert proteon.rmsd(coords, translated) == pytest.approx(0.0, abs=1e-8)

    def test_apply_transform_identity(self, coords):
        out = proteon.apply_transform(
            coords, np.eye(3), np.zeros(3),
        )
        np.testing.assert_allclose(out, coords, atol=1e-12)

    def test_apply_transform_known_translation(self, coords):
        delta = np.array([1.5, -2.0, 0.5])
        out = proteon.apply_transform(coords, np.eye(3), delta)
        # Each row should differ from the corresponding input row by
        # the same translation vector.
        diff = out - coords
        np.testing.assert_allclose(diff, np.broadcast_to(delta, diff.shape), atol=1e-8)

    def test_assign_secondary_structure_returns_string(self, coords):
        ss = proteon.assign_secondary_structure(coords)
        assert isinstance(ss, str)
        assert len(ss) == coords.shape[0]
        # Characters should all be from the {H, E, T, C} alphabet.
        assert set(ss).issubset(set("HETC"))

    def test_tm_score_self_alignment(self, coords):
        # Trivial self-alignment: invmap[j] = j for every residue.
        n = coords.shape[0]
        invmap = np.arange(n, dtype=np.int32)
        score, n_aligned, rmsd, _rot, _trans = proteon.tm_score(
            coords, coords, invmap,
        )
        assert score == pytest.approx(1.0, abs=1e-6)
        assert n_aligned == n
        assert rmsd == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# dssp.py
# ===========================================================================


class TestDsspWrappers:
    """Cover proteon.dssp / dssp_array / batch_dssp / load_and_dssp."""

    def test_dssp_returns_string(self):
        s = proteon.load(CRAMBIN)
        ss = proteon.dssp(s)
        assert isinstance(ss, str)
        assert len(ss) == 46  # crambin amino acids
        assert set(ss).issubset(set("HGIEBTSC"))

    def test_dssp_array_matches_string(self):
        s = proteon.load(CRAMBIN)
        ss_str = proteon.dssp(s)
        ss_arr = proteon.dssp_array(s)
        assert ss_arr.shape == (len(ss_str),)
        # ASCII codes per residue.
        assert "".join(chr(c) for c in ss_arr) == ss_str

    def test_batch_dssp_matches_single(self):
        structures = [proteon.load(CRAMBIN), proteon.load(UBIQ)]
        batch = proteon.batch_dssp(structures, n_threads=1)
        singles = [proteon.dssp(s) for s in structures]
        assert batch == singles

    def test_load_and_dssp_returns_index_string_tuples(self):
        results = proteon.load_and_dssp([CRAMBIN, UBIQ], n_threads=1)
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
    """Cover proteon.place_peptide_hydrogens / place_all_hydrogens /
    place_general_hydrogens / reconstruct_fragments /
    batch_place_peptide_hydrogens."""

    def test_place_peptide_hydrogens_returns_counts(self):
        s = proteon.load(CRAMBIN)
        added, skipped = proteon.place_peptide_hydrogens(s)
        assert isinstance(added, int)
        assert isinstance(skipped, int)
        assert added + skipped > 0

    def test_place_peptide_hydrogens_with_coords_returns_tuple(self):
        s = proteon.load(CRAMBIN)
        (added, skipped), coords = proteon.place_peptide_hydrogens(
            s, return_coords=True,
        )
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3
        assert coords.shape[0] == added

    def test_place_peptide_hydrogens_idempotent(self):
        s = proteon.load(CRAMBIN)
        a1, _ = proteon.place_peptide_hydrogens(s)
        a2, _ = proteon.place_peptide_hydrogens(s)
        # Second call adds nothing — contract promised in the docstring.
        assert a2 == 0
        assert a1 > 0

    def test_place_all_hydrogens_adds_more_than_peptide_only(self):
        # place_all = backbone + sidechain. Starting from a fresh load,
        # place_all should add strictly more H than place_peptide would.
        s_all = proteon.load(CRAMBIN)
        added_all, _ = proteon.place_all_hydrogens(s_all)

        s_peptide = proteon.load(CRAMBIN)
        added_peptide, _ = proteon.place_peptide_hydrogens(s_peptide)

        assert added_all > added_peptide

    def test_place_general_hydrogens_returns_counts(self):
        s = proteon.load(CRAMBIN)
        added, skipped = proteon.place_general_hydrogens(s)
        assert isinstance(added, int)
        assert isinstance(skipped, int)
        assert added > 0

    def test_reconstruct_fragments_returns_int(self):
        s = proteon.load(CRAMBIN)
        n_added = proteon.reconstruct_fragments(s)
        # Crambin has complete heavy atoms, so likely 0 added — but the
        # return type contract is what we're pinning here.
        assert isinstance(n_added, int)
        assert n_added >= 0

    def test_batch_place_peptide_hydrogens_parallel(self):
        structures = [proteon.load(CRAMBIN), proteon.load(UBIQ)]
        results = proteon.batch_place_peptide_hydrogens(
            structures, n_threads=1,
        )
        assert len(results) == 2
        for added, skipped in results:
            assert isinstance(added, int)
            assert isinstance(skipped, int)


# ===========================================================================
# align.py
# ===========================================================================


class TestAlignWrappers:
    """Cover the Python-facing align entry points and result classes.

    `test_alignment.py` / `test_mmalign.py` go through
    `proteon_connector.py_align_funcs` directly — they exercise the
    Rust core but never touch the Python-side `AlignResult` /
    `SoiAlignResult` / `FlexAlignResult` / `MMAlignResult` property
    wrappers or the public `proteon.tm_align` / `soi_align` /
    `flex_align` / `mm_align` dispatch functions.
    """

    @pytest.fixture(scope="class")
    def pair(self):
        return proteon.load(CRAMBIN), proteon.load(UBIQ)

    @pytest.fixture(scope="class")
    def triples(self):
        """Three-structure list for one-to-many / many-to-many tests."""
        return [
            proteon.load(CRAMBIN),
            proteon.load(UBIQ),
            proteon.load(CRAMBIN),
        ]

    # --- TM-align + AlignResult properties --------------------------------

    def test_tm_align_result_properties(self, pair):
        a, b = pair
        r = proteon.tm_align(a, b)
        # Every declared property on AlignResult — touching each one
        # covers the `self._ptr.X` forwarding line in the wrapper.
        assert 0.0 <= r.tm_score_chain1 <= 1.0
        assert 0.0 <= r.tm_score_chain2 <= 1.0
        assert r.rmsd >= 0.0
        assert r.n_aligned > 0
        assert 0.0 <= r.seq_identity <= 1.0
        rot = r.rotation_matrix
        trans = r.translation
        assert rot is not None and trans is not None
        # Aligned sequences are same-length strings (gaps padded).
        assert isinstance(r.aligned_seq_x, str)
        assert isinstance(r.aligned_seq_y, str)
        assert len(r.aligned_seq_x) == len(r.aligned_seq_y)
        # __repr__ path
        assert isinstance(repr(r), str)
        # Wrapper-object contract.
        assert r.get_py_ptr() is not None

    def test_tm_align_fast_mode(self, pair):
        a, b = pair
        r = proteon.tm_align(a, b, fast=True)
        assert r.n_aligned > 0

    def test_tm_align_one_to_many(self, pair, triples):
        a, _ = pair
        results = proteon.tm_align_one_to_many(a, triples, n_threads=1)
        assert len(results) == 3
        for r in results:
            assert r.n_aligned > 0

    def test_tm_align_many_to_many(self, pair):
        a, b = pair
        results = proteon.tm_align_many_to_many([a], [a, b], n_threads=1)
        # Cartesian product: 1 × 2 = 2.
        assert len(results) == 2
        for qi, ti, r in results:
            assert isinstance(qi, int)
            assert isinstance(ti, int)
            assert r.tm_score_chain1 >= 0.0

    # --- SOI-align + SoiAlignResult properties ----------------------------

    def test_soi_align_result_properties(self, pair):
        a, b = pair
        r = proteon.soi_align(a, b)
        assert 0.0 <= r.tm_score_chain1 <= 1.0
        assert 0.0 <= r.tm_score_chain2 <= 1.0
        assert r.rmsd >= 0.0
        assert r.n_aligned > 0
        assert 0.0 <= r.seq_identity <= 1.0
        assert r.rotation_matrix is not None
        assert r.translation is not None
        assert isinstance(r.aligned_seq_x, str)
        assert isinstance(r.aligned_seq_y, str)
        assert isinstance(repr(r), str)
        assert r.get_py_ptr() is not None

    def test_soi_align_one_to_many(self, pair, triples):
        a, _ = pair
        results = proteon.soi_align_one_to_many(a, triples, n_threads=1)
        assert len(results) == 3

    def test_soi_align_many_to_many(self, pair):
        a, b = pair
        results = proteon.soi_align_many_to_many([a], [a, b], n_threads=1)
        assert len(results) == 2

    # --- FlexAlign + FlexAlignResult properties ---------------------------

    def test_flex_align_result_properties(self, pair):
        a, b = pair
        r = proteon.flex_align(a, b)
        assert 0.0 <= r.tm_score_chain1 <= 1.0
        assert 0.0 <= r.tm_score_chain2 <= 1.0
        assert r.rmsd >= 0.0
        assert r.n_aligned > 0
        assert 0.0 <= r.seq_identity <= 1.0
        # FlexAlign-specific: hinge_count + Kx3x3 rotation / Kx3
        # translation arrays.
        assert r.hinge_count >= 0
        assert r.rotation_matrices is not None
        assert r.translations is not None
        assert isinstance(r.aligned_seq_x, str)
        assert isinstance(r.aligned_seq_y, str)
        assert isinstance(repr(r), str)

    def test_flex_align_one_to_many(self, pair, triples):
        a, _ = pair
        results = proteon.flex_align_one_to_many(a, triples, n_threads=1)
        assert len(results) == 3

    def test_flex_align_many_to_many(self, pair):
        a, b = pair
        results = proteon.flex_align_many_to_many([a], [a, b], n_threads=1)
        assert len(results) == 2

    # --- MM-align + MMAlignResult / ChainPairResult -----------------------

    @pytest.fixture(scope="class")
    def hhb(self):
        if not os.path.exists(HHB):
            pytest.skip(f"4hhb.pdb not available at {HHB}")
        return proteon.load(HHB)

    def test_mm_align_result_properties(self, hhb):
        """Self-align hemoglobin to itself; exercises the MMAlignResult +
        ChainPairResult Python wrappers end-to-end."""
        r = proteon.mm_align(hhb, hhb)
        # MMAlignResult properties.
        assert r.total_score >= 0.0
        assignments = r.chain_assignments
        assert isinstance(assignments, list)
        # Every chain-pair carries its per-chain scores — these are the
        # ChainPairResult property accesses we want to cover.
        for pair in r.chain_pairs:
            assert isinstance(pair.query_chain, int)
            assert isinstance(pair.target_chain, int)
            assert pair.tm_score >= 0.0
            assert pair.rmsd >= 0.0
            assert pair.n_aligned >= 0
            assert isinstance(pair.aligned_seq_x, str)
            assert isinstance(pair.aligned_seq_y, str)
            # Wrapper-object contract.
            assert pair.get_py_ptr() is not None
        # __repr__ path on MMAlignResult.
        assert isinstance(repr(r), str)
        assert r.get_py_ptr() is not None

    def test_mm_align_one_to_many(self, hhb):
        results = proteon.mm_align_one_to_many(hhb, [hhb, hhb], n_threads=1)
        assert len(results) == 2
        for r in results:
            assert r.total_score >= 0.0

    def test_mm_align_many_to_many(self, hhb):
        results = proteon.mm_align_many_to_many([hhb], [hhb, hhb], n_threads=1)
        assert len(results) == 2
