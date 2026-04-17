"""Unit tests for corpus smoke-release split strategies."""

from __future__ import annotations

from collections import Counter

import pytest

from pathlib import Path
from types import SimpleNamespace

from ferritin.corpus_smoke import (
    _default_split_assignments,
    _expand_chains,
    _hash_split_assignments,
    _format_ratios,
)


def _fake_struct(chain_ids):
    chains = [SimpleNamespace(id=cid) for cid in chain_ids]
    return SimpleNamespace(chains=chains)


def test_hash_split_is_deterministic():
    rids = [f"p{i:04d}" for i in range(50)]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = _hash_split_assignments(rids, ratios)
    b = _hash_split_assignments(rids, ratios)
    assert a == b


def test_hash_split_is_order_invariant():
    rids = [f"p{i:04d}" for i in range(50)]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    forward = _hash_split_assignments(rids, ratios)
    reverse = _hash_split_assignments(list(reversed(rids)), ratios)
    assert forward == reverse


def test_hash_split_distribution_approximates_ratios():
    rids = [f"p{i:05d}" for i in range(2000)]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    counts = Counter(_hash_split_assignments(rids, ratios).values())
    assert abs(counts["train"] / 2000 - 0.8) < 0.03
    assert abs(counts["val"] / 2000 - 0.1) < 0.03
    assert abs(counts["test"] / 2000 - 0.1) < 0.03


def test_hash_split_covers_all_records():
    rids = [f"p{i:04d}" for i in range(100)]
    out = _hash_split_assignments(rids, {"train": 0.5, "val": 0.5})
    assert set(out.keys()) == set(rids)
    assert set(out.values()) <= {"train", "val"}


def test_hash_split_accepts_unnormalized_ratios():
    rids = [f"p{i:04d}" for i in range(500)]
    normalized = _hash_split_assignments(rids, {"train": 0.8, "val": 0.2})
    unnormalized = _hash_split_assignments(rids, {"train": 4.0, "val": 1.0})
    assert normalized == unnormalized


def test_hash_split_rejects_empty_or_negative():
    with pytest.raises(ValueError):
        _hash_split_assignments(["a"], {})
    with pytest.raises(ValueError):
        _hash_split_assignments(["a"], {"train": 0.0})
    with pytest.raises(ValueError):
        _hash_split_assignments(["a"], {"train": 1.0, "val": -0.1})


def test_default_split_is_hash_80_10_10():
    # Default split must produce all three buckets at roughly 80/10/10.
    # Regression for the Apr 17 1K rerun where the old default put
    # everything in train + exactly one record in val, silently killing
    # the test split and making downstream evaluation bogus.
    rids = [f"rec_{i:05d}" for i in range(10_000)]
    out = _default_split_assignments(rids)
    assert set(out.values()) == {"train", "val", "test"}
    counts = Counter(out.values())
    assert 7_500 <= counts["train"] <= 8_500
    assert 800 <= counts["val"] <= 1_200
    assert 800 <= counts["test"] <= 1_200
    # Determinism: same input must yield the same assignment.
    assert _default_split_assignments(rids) == out


def test_expand_chains_passes_through_single_chain():
    s = _fake_struct(["A"])
    structs, preps, rids, srcs, cids, paths = _expand_chains(
        [s], ["prep"], ["1crn"], ["/p/1crn.pdb"], [Path("/p/1crn.pdb")]
    )
    assert rids == ["1crn"]
    assert cids == [None]
    assert structs == [s]


def test_expand_chains_splits_multi_chain():
    s = _fake_struct(["A", "B"])
    structs, preps, rids, srcs, cids, paths = _expand_chains(
        [s], ["prep"], ["1ake"], ["/p/1ake.pdb"], [Path("/p/1ake.pdb")]
    )
    assert rids == ["1ake_A", "1ake_B"]
    assert cids == ["A", "B"]
    assert structs == [s, s]
    assert preps == ["prep", "prep"]


def test_expand_chains_dedupes_multi_model_chain_repeats():
    # pdbtbx's Structure.chains flattens across models, so an NMR
    # structure with 4 models × 3 chains yields 12 chain objects with
    # repeated ids. We want one record per *logical* chain, not one
    # per (model, chain) pair. Regression for the 1K monster3 run
    # where 193d (4 models, 3 chains) produced 4 duplicates of
    # `193d_C` in the training manifest.
    repeated = _fake_struct(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"])
    _, _, rids, _, cids, _ = _expand_chains(
        [repeated], ["prep"], ["193d"], ["/p/193d.pdb"], [Path("/p/193d.pdb")]
    )
    assert rids == ["193d_A", "193d_B", "193d_C"]
    assert cids == ["A", "B", "C"]


def test_expand_chains_preserves_ordering_across_inputs():
    a = _fake_struct(["X"])
    b = _fake_struct(["A", "B", "C"])
    c = _fake_struct(["A"])
    _, _, rids, _, cids, _ = _expand_chains(
        [a, b, c],
        ["pa", "pb", "pc"],
        ["alpha", "beta", "gamma"],
        ["sa", "sb", "sc"],
        [Path("a"), Path("b"), Path("c")],
    )
    assert rids == ["alpha", "beta_A", "beta_B", "beta_C", "gamma"]
    assert cids == [None, "A", "B", "C", None]


def test_format_ratios_is_stable_order():
    s = _format_ratios({"val": 0.1, "train": 0.9})
    assert s == "train=0.900,val=0.100"
