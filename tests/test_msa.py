"""Smoke tests for the proteon.msa Pythonic wrapper.

Covers the end-to-end Rust-Python bridge:
  - connector availability
  - engine build over a target corpus
  - search returns ranked hit dicts with expected fields
  - build_msa returns the AF2 tensor dict with expected shapes/dtypes
  - duplicates in the target corpus surface as separate hits

The structure-side `build_sequence_example` integration is exercised
in a separate test file (it requires loading a real PDB).
"""

from __future__ import annotations

import numpy as np
import pytest

from proteon.msa_backend import rust_msa_available

if not rust_msa_available():
    pytest.skip(
        "proteon_connector.py_msa not available; run "
        "`maturin develop` in proteon-connector/ first.",
        allow_module_level=True,
    )

from proteon.msa import MsaSearch  # noqa: E402  (skip-then-import)


# Three distinct proteins; targets 1 and 3 share the same sequence so
# duplicate-handling can be exercised without writing a near-duplicate.
TARGETS = [
    (1, "MNALVVKFGGTSVANAERFLRVADILESNARQGQ"),
    (2, "WVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP"),
    (3, "MNALVVKFGGTSVANAERFLRVADILESNARQGQ"),  # duplicate of 1
]


@pytest.fixture(scope="module")
def engine():
    return MsaSearch.build(TARGETS, k=6, reduce_to=13)


def test_engine_builds_and_reports_target_count(engine):
    assert engine.target_count == 3


def test_self_search_returns_self_at_top(engine):
    query = TARGETS[1][1]  # the unique target_id=2 sequence
    hits = engine.search(query)
    assert len(hits) >= 1, "self-search should return at least one hit"
    assert hits[0]["target_id"] == 2
    # CIGAR for an exact self-match is a single Match op.
    assert hits[0]["cigar"] == f"{len(query)}M"


def test_search_returns_both_duplicate_targets(engine):
    query = TARGETS[0][1]  # matches both target_id 1 and 3 exactly
    hits = engine.search(query)
    seq_ids = {h["target_id"] for h in hits}
    assert {1, 3}.issubset(seq_ids), f"expected both duplicate ids, got {seq_ids}"


def test_hit_dict_has_all_expected_fields(engine):
    hits = engine.search(TARGETS[0][1])
    assert hits, "expected non-empty hit list"
    expected_keys = {
        "target_id",
        "prefilter_score",
        "best_diagonal",
        "ungapped_score",
        "score",
        "query_start",
        "query_end",
        "target_start",
        "target_end",
        "cigar",
    }
    actual_keys = set(hits[0].keys())
    missing = expected_keys - actual_keys
    assert not missing, f"hit dict missing fields: {missing}"


def test_hits_sorted_by_score_descending(engine):
    hits = engine.search(TARGETS[0][1])
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "hits not sorted by score desc"


def test_build_msa_returns_expected_shapes_and_dtypes(engine):
    query = TARGETS[0][1]
    msa = engine.build_msa(query, max_seqs=10)
    # Scalars
    assert msa["query_len"] == len(query)
    assert msa["gap_idx"] == 21
    # Row 0 is the query, plus however many hits we find.
    assert msa["n_seqs"] == msa["msa"].shape[0]
    assert msa["n_seqs"] >= 1, "at least the query row is always present"
    # Shapes
    L = msa["query_len"]
    N = msa["n_seqs"]
    assert msa["aatype"].shape == (L,)
    assert msa["seq_mask"].shape == (L,)
    assert msa["msa"].shape == (N, L)
    assert msa["deletion_matrix"].shape == (N, L)
    assert msa["msa_mask"].shape == (N, L)
    # Dtypes — match Rust connector output (uint8 for indexed data,
    # float32 for masks).
    assert msa["aatype"].dtype == np.uint8
    assert msa["seq_mask"].dtype == np.float32
    assert msa["msa"].dtype == np.uint8
    assert msa["deletion_matrix"].dtype == np.uint8
    assert msa["msa_mask"].dtype == np.float32


def test_msa_row_zero_is_query(engine):
    query = TARGETS[0][1]
    msa = engine.build_msa(query)
    # MSA[0] must equal aatype (both are the query encoded).
    assert np.array_equal(msa["msa"][0], msa["aatype"])
    # Seq mask is all 1.0.
    assert np.all(msa["seq_mask"] == 1.0)


def test_msa_caps_at_max_seqs(engine):
    msa = engine.build_msa(TARGETS[0][1], max_seqs=2)
    assert msa["n_seqs"] == 2  # query + 1 hit


def test_unrelated_query_returns_at_most_some_hits(engine):
    # An all-W query won't match any of our targets meaningfully, so
    # results should be empty or low-scoring noise. We just assert
    # the call doesn't crash and returns a list.
    hits = engine.search("WWWWWWWWWWWWWWWWWWWW")
    assert isinstance(hits, list)
