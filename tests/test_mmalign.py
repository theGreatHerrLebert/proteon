"""Tests for MM-align (multi-chain complex alignment)."""

import os

import pytest

from proteon_connector import py_align_funcs, py_io

TEST_PDBS = os.path.join(os.path.dirname(__file__), "..", "..", "test-pdbs")
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")

# Multi-chain structures
HHB = os.path.join(TEST_PDBS, "4hhb.pdb")     # hemoglobin, 4 chains
HHO = os.path.join(TEST_PDBS, "1hho.pdb")     # oxy-hemoglobin, 2 chains
SOD = os.path.join(TEST_PDBS, "2sod.pdb")     # superoxide dismutase
UBIQ = os.path.join(EXAMPLE_DIR, "1ubq.pdb")  # single chain (control)


def have_file(path):
    return os.path.exists(path)


@pytest.fixture
def hhb():
    if not have_file(HHB):
        pytest.skip("4hhb.pdb not available")
    return py_io.load(HHB)


@pytest.fixture
def hho():
    if not have_file(HHO):
        pytest.skip("1hho.pdb not available")
    return py_io.load(HHO)


@pytest.fixture
def ubiq():
    return py_io.load(UBIQ)


# ---------------------------------------------------------------------------
# Single pair
# ---------------------------------------------------------------------------


class TestMMAlignPair:
    def test_self_alignment(self, hhb):
        result = py_align_funcs.mm_align_pair(hhb, hhb)
        # All chains should be matched to themselves
        assert len(result.chain_pairs) == 4
        for p in result.chain_pairs:
            assert p.query_chain == p.target_chain
            assert p.tm_score == pytest.approx(1.0, abs=0.001)
            assert p.rmsd == pytest.approx(0.0, abs=0.01)

    def test_cross_alignment(self, hhb, hho):
        result = py_align_funcs.mm_align_pair(hhb, hho)
        assert result.total_score > 0.0
        assert len(result.chain_pairs) >= 1
        assert len(result.chain_assignments) >= 1

    def test_result_fields(self, hhb):
        result = py_align_funcs.mm_align_pair(hhb, hhb)
        assert isinstance(result.total_score, float)
        assert isinstance(result.chain_assignments, list)
        assert isinstance(result.chain_pairs, list)

    def test_chain_pair_fields(self, hhb):
        result = py_align_funcs.mm_align_pair(hhb, hhb)
        p = result.chain_pairs[0]
        assert isinstance(p.query_chain, int)
        assert isinstance(p.target_chain, int)
        assert isinstance(p.tm_score, float)
        assert isinstance(p.rmsd, float)
        assert isinstance(p.n_aligned, int)
        assert isinstance(p.aligned_seq_x, str)
        assert isinstance(p.aligned_seq_y, str)

    def test_single_chain_complex(self, ubiq):
        # Single-chain structure treated as 1-chain complex
        result = py_align_funcs.mm_align_pair(ubiq, ubiq)
        assert len(result.chain_pairs) == 1
        assert result.chain_pairs[0].tm_score == pytest.approx(1.0, abs=0.001)

    def test_repr(self, hhb):
        result = py_align_funcs.mm_align_pair(hhb, hhb)
        r = repr(result)
        assert "MMAlignResult" in r


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestMMAlignBatch:
    def test_one_to_many(self, hhb, hho):
        results = py_align_funcs.mm_align_one_to_many(hhb, [hho, hhb], n_threads=2)
        assert len(results) == 2
        # Second is self-alignment
        assert results[1].total_score > results[0].total_score

    def test_many_to_many(self, hhb, ubiq):
        results = py_align_funcs.mm_align_many_to_many([hhb], [ubiq, hhb], n_threads=2)
        assert len(results) == 2

    def test_empty_targets(self, hhb):
        results = py_align_funcs.mm_align_one_to_many(hhb, [])
        assert len(results) == 0

    def test_n_threads_1(self, hhb, hho):
        results = py_align_funcs.mm_align_one_to_many(hhb, [hho], n_threads=1)
        assert len(results) == 1

    def test_n_threads_all(self, hhb, hho):
        results = py_align_funcs.mm_align_one_to_many(hhb, [hho], n_threads=-1)
        assert len(results) == 1
