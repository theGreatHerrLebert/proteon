"""Tests for TM-align Python bindings: single pair, one-to-many, many-to-many.

Tests cover: correctness, symmetry, self-alignment, batch parallelism,
thread count configuration, error handling.
"""

import os
import time

import numpy as np
import pytest

from proteon_connector import py_align_funcs, py_io

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
# Parent test-pdbs has a larger collection
PARENT_TEST_PDBS = os.path.join(os.path.dirname(__file__), "..", "..", "test-pdbs")

UBIQ = os.path.join(EXAMPLE_DIR, "1ubq.pdb")
CRAMBIN = os.path.join(TEST_PDBS_DIR, "1crn.pdb")


@pytest.fixture
def ubiq():
    return py_io.load(UBIQ)


@pytest.fixture
def crambin():
    return py_io.load(CRAMBIN)


# ---------------------------------------------------------------------------
# Single pair alignment
# ---------------------------------------------------------------------------


class TestTMAlignPair:
    def test_basic_alignment(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        assert 0.0 < result.tm_score_chain1 < 1.0
        assert 0.0 < result.tm_score_chain2 < 1.0
        assert result.rmsd > 0.0
        assert result.n_aligned > 0

    def test_self_alignment(self, ubiq):
        result = py_align_funcs.tm_align_pair(ubiq, ubiq)
        assert result.tm_score_chain1 == pytest.approx(1.0, abs=0.001)
        assert result.tm_score_chain2 == pytest.approx(1.0, abs=0.001)
        assert result.rmsd == pytest.approx(0.0, abs=0.01)

    def test_result_fields(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        assert isinstance(result.tm_score_chain1, float)
        assert isinstance(result.tm_score_chain2, float)
        assert isinstance(result.rmsd, float)
        assert isinstance(result.n_aligned, int)
        assert isinstance(result.seq_identity, float)
        assert isinstance(result.aligned_seq_x, str)
        assert isinstance(result.aligned_seq_y, str)

    def test_rotation_matrix(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        rot = result.rotation_matrix
        assert rot.shape == (3, 3)
        assert rot.dtype == np.float64

    def test_translation(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        t = result.translation
        assert len(t) == 3

    def test_fast_mode(self, ubiq, crambin):
        result_normal = py_align_funcs.tm_align_pair(ubiq, crambin, fast=False)
        result_fast = py_align_funcs.tm_align_pair(ubiq, crambin, fast=True)
        # Fast should give a result (may differ slightly)
        assert result_fast.tm_score_chain1 > 0.0
        assert result_fast.n_aligned > 0

    def test_chain_filter(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin, chain1="A")
        assert result.n_aligned > 0

    def test_repr(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        r = repr(result)
        assert "AlignResult" in r

    def test_alignment_strings(self, ubiq, crambin):
        result = py_align_funcs.tm_align_pair(ubiq, crambin)
        assert len(result.aligned_seq_x) > 0
        assert len(result.aligned_seq_y) > 0
        assert len(result.aligned_seq_x) == len(result.aligned_seq_y)


# ---------------------------------------------------------------------------
# One-to-many
# ---------------------------------------------------------------------------


class TestTMAlignOneToMany:
    def test_basic(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(ubiq, [crambin, ubiq])
        assert len(results) == 2

    def test_results_order(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(ubiq, [crambin, ubiq])
        # Second result is self-alignment
        assert results[1].tm_score_chain1 == pytest.approx(1.0, abs=0.001)
        # First result is cross-alignment
        assert results[0].tm_score_chain1 < 1.0

    def test_matches_single_pair(self, ubiq, crambin):
        single = py_align_funcs.tm_align_pair(ubiq, crambin)
        batch = py_align_funcs.tm_align_one_to_many(ubiq, [crambin])
        assert batch[0].tm_score_chain1 == pytest.approx(
            single.tm_score_chain1, abs=1e-6
        )
        assert batch[0].rmsd == pytest.approx(single.rmsd, abs=1e-6)

    def test_n_threads_1(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(
            ubiq, [crambin, ubiq], n_threads=1
        )
        assert len(results) == 2

    def test_n_threads_all(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(
            ubiq, [crambin], n_threads=-1
        )
        assert len(results) == 1

    def test_n_threads_none(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(
            ubiq, [crambin], n_threads=None
        )
        assert len(results) == 1

    def test_empty_targets(self, ubiq):
        results = py_align_funcs.tm_align_one_to_many(ubiq, [])
        assert len(results) == 0

    def test_fast_mode(self, ubiq, crambin):
        results = py_align_funcs.tm_align_one_to_many(
            ubiq, [crambin], fast=True
        )
        assert len(results) == 1
        assert results[0].n_aligned > 0


# ---------------------------------------------------------------------------
# Many-to-many
# ---------------------------------------------------------------------------


class TestTMAlignManyToMany:
    def test_basic(self, ubiq, crambin):
        results = py_align_funcs.tm_align_many_to_many([ubiq], [crambin])
        assert len(results) == 1
        qi, ti, r = results[0]
        assert qi == 0
        assert ti == 0
        assert r.n_aligned > 0

    def test_cartesian_product_size(self, ubiq, crambin):
        results = py_align_funcs.tm_align_many_to_many(
            [ubiq, crambin], [crambin, ubiq]
        )
        # 2 x 2 = 4 pairs
        assert len(results) == 4

    def test_indices(self, ubiq, crambin):
        results = py_align_funcs.tm_align_many_to_many(
            [ubiq, crambin], [crambin, ubiq]
        )
        indices = [(qi, ti) for qi, ti, _ in results]
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert sorted(indices) == sorted(expected)

    def test_self_alignment_in_batch(self, ubiq, crambin):
        results = py_align_funcs.tm_align_many_to_many(
            [ubiq, crambin], [ubiq, crambin]
        )
        # Find self-alignments (diagonal)
        for qi, ti, r in results:
            if qi == ti:
                assert r.tm_score_chain1 == pytest.approx(1.0, abs=0.001)

    def test_matches_single_pair(self, ubiq, crambin):
        single = py_align_funcs.tm_align_pair(ubiq, crambin)
        batch = py_align_funcs.tm_align_many_to_many([ubiq], [crambin])
        _, _, r = batch[0]
        assert r.tm_score_chain1 == pytest.approx(
            single.tm_score_chain1, abs=1e-6
        )

    def test_n_threads(self, ubiq, crambin):
        results = py_align_funcs.tm_align_many_to_many(
            [ubiq, crambin], [crambin], n_threads=2
        )
        assert len(results) == 2

    def test_empty_queries(self, crambin):
        results = py_align_funcs.tm_align_many_to_many([], [crambin])
        assert len(results) == 0

    def test_empty_targets(self, ubiq):
        results = py_align_funcs.tm_align_many_to_many([ubiq], [])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Batch with multiple PDB files
# ---------------------------------------------------------------------------


class TestBatchMultipleFiles:
    """Test batch alignment with multiple PDB files from test-pdbs/."""

    @pytest.fixture
    def structures(self):
        pdbs = ["1crn.pdb", "1lyz.pdb", "2gb1.pdb", "1tit.pdb", "5pti.pdb"]
        loaded = []
        for name in pdbs:
            for d in [TEST_PDBS_DIR, PARENT_TEST_PDBS]:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    loaded.append(py_io.load(path))
                    break
        if len(loaded) < 2:
            pytest.skip("need at least 2 test PDBs")
        return loaded

    def test_one_to_many_multiple(self, structures):
        query = structures[0]
        targets = structures[1:]
        results = py_align_funcs.tm_align_one_to_many(query, targets, n_threads=2)
        assert len(results) == len(targets)
        for r in results:
            assert r.n_aligned > 0
            assert r.tm_score_chain1 > 0.0

    def test_many_to_many_multiple(self, structures):
        results = py_align_funcs.tm_align_many_to_many(
            structures, structures, n_threads=2
        )
        n = len(structures)
        assert len(results) == n * n
        # Diagonal should be self-alignment
        for qi, ti, r in results:
            if qi == ti:
                assert r.tm_score_chain1 == pytest.approx(1.0, abs=0.001)

    def test_fast_batch(self, structures):
        results = py_align_funcs.tm_align_one_to_many(
            structures[0], structures[1:], fast=True, n_threads=2
        )
        assert len(results) == len(structures) - 1


# ---------------------------------------------------------------------------
# Thread scaling (performance sanity check)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SOI-align
# ---------------------------------------------------------------------------


class TestSoiAlignPair:
    def test_basic(self, ubiq, crambin):
        result = py_align_funcs.soi_align_pair(ubiq, crambin)
        assert 0.0 < result.tm_score_chain1 < 1.0
        assert result.rmsd > 0.0
        assert result.n_aligned > 0

    def test_self_alignment(self, ubiq):
        result = py_align_funcs.soi_align_pair(ubiq, ubiq)
        assert result.tm_score_chain1 == pytest.approx(1.0, abs=0.001)

    def test_result_fields(self, ubiq, crambin):
        result = py_align_funcs.soi_align_pair(ubiq, crambin)
        assert isinstance(result.tm_score_chain1, float)
        assert isinstance(result.tm_score_chain2, float)
        assert isinstance(result.rmsd, float)
        assert isinstance(result.n_aligned, int)
        assert result.rotation_matrix.shape == (3, 3)
        assert len(result.translation) == 3

    def test_alignment_strings(self, ubiq, crambin):
        result = py_align_funcs.soi_align_pair(ubiq, crambin)
        assert len(result.aligned_seq_x) > 0
        assert len(result.aligned_seq_y) > 0


class TestSoiAlignBatch:
    def test_one_to_many(self, ubiq, crambin):
        results = py_align_funcs.soi_align_one_to_many(ubiq, [crambin, ubiq], n_threads=2)
        assert len(results) == 2
        assert results[1].tm_score_chain1 == pytest.approx(1.0, abs=0.001)

    def test_many_to_many(self, ubiq, crambin):
        results = py_align_funcs.soi_align_many_to_many([ubiq], [crambin, ubiq], n_threads=2)
        assert len(results) == 2

    def test_empty(self, ubiq):
        results = py_align_funcs.soi_align_one_to_many(ubiq, [])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# FlexAlign
# ---------------------------------------------------------------------------


class TestFlexAlignPair:
    def test_basic(self, ubiq, crambin):
        result = py_align_funcs.flex_align_pair(ubiq, crambin)
        assert 0.0 < result.tm_score_chain1 < 1.0
        assert result.rmsd > 0.0
        assert result.n_aligned > 0
        assert result.hinge_count >= 0

    def test_self_alignment(self, ubiq):
        result = py_align_funcs.flex_align_pair(ubiq, ubiq)
        assert result.tm_score_chain1 == pytest.approx(1.0, abs=0.01)

    def test_hinge_transforms(self, ubiq, crambin):
        result = py_align_funcs.flex_align_pair(ubiq, crambin)
        k = result.hinge_count + 1  # segments = hinges + 1
        assert result.rotation_matrices.shape == (k, 3, 3)
        assert result.translations.shape == (k, 3)

    def test_alignment_strings(self, ubiq, crambin):
        result = py_align_funcs.flex_align_pair(ubiq, crambin)
        assert len(result.aligned_seq_x) > 0
        assert len(result.aligned_seq_y) > 0

    def test_repr(self, ubiq, crambin):
        result = py_align_funcs.flex_align_pair(ubiq, crambin)
        r = repr(result)
        assert "FlexAlignResult" in r
        assert "hinges=" in r


class TestFlexAlignBatch:
    def test_one_to_many(self, ubiq, crambin):
        results = py_align_funcs.flex_align_one_to_many(ubiq, [crambin, ubiq], n_threads=2)
        assert len(results) == 2

    def test_many_to_many(self, ubiq, crambin):
        results = py_align_funcs.flex_align_many_to_many([ubiq], [crambin], n_threads=1)
        assert len(results) == 1
        qi, ti, r = results[0]
        assert qi == 0 and ti == 0
        assert r.n_aligned > 0

    def test_empty(self, ubiq):
        results = py_align_funcs.flex_align_one_to_many(ubiq, [])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Thread scaling
# ---------------------------------------------------------------------------


class TestThreadScaling:
    """Verify that multi-threading doesn't break correctness."""

    def test_thread_counts_give_same_results(self, ubiq, crambin):
        targets = [crambin] * 4
        r1 = py_align_funcs.tm_align_one_to_many(ubiq, targets, n_threads=1)
        r2 = py_align_funcs.tm_align_one_to_many(ubiq, targets, n_threads=2)
        r4 = py_align_funcs.tm_align_one_to_many(ubiq, targets, n_threads=4)

        for a, b, c in zip(r1, r2, r4):
            assert a.tm_score_chain1 == pytest.approx(b.tm_score_chain1, abs=1e-10)
            assert a.tm_score_chain1 == pytest.approx(c.tm_score_chain1, abs=1e-10)
            assert a.rmsd == pytest.approx(b.rmsd, abs=1e-10)
