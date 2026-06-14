"""AlphaFold MSA feature pipeline — invariants (pure NumPy, runs in CI).

Gates the transforms by structure/invariant (the OpenFold *bit-parity* oracle is
test_msa_features_openfold.py, which needs torch+openfold). Covers remap
direction, sample/mask/cluster/summarize/crop/assemble shapes + contracts, the
injectable-RNG determinism, edge cases, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteon.msa_features import (
    MSA_FEAT_CHANNELS,
    MSA_ONE_HOT,
    build_msa_features,
    crop_extra_msa,
    make_masked_msa,
    make_msa_feat,
    nearest_neighbor_clusters,
    remap_msa_tokens,
    remap_profile_channels,
    sample_msa,
    summarize_clusters,
)


# --- remapping -------------------------------------------------------------

def test_token_remap_is_a_bijection_with_gap_mask_passthrough():
    out = remap_msa_tokens(np.arange(23))
    assert sorted(out.tolist()) == list(range(23))
    assert out[20] == 20 and out[21] == 21 and out[22] == 22  # X / gap / mask fixed


def test_profile_remap_scatters_channels_not_gathers():
    # A one-hot at proteon channel i must land at OpenFold channel MAP[i] — the
    # scatter direction. proteon 'C' is index 1; OpenFold 'C' is index 4.
    p = np.zeros((1, 22), np.float32)
    p[0, 1] = 1.0  # proteon 'C'
    of = remap_profile_channels(p)
    assert of[0, 4] == 1.0 and of[0, 1] == 0.0
    # Per-row mass is preserved (it's a permutation of residue+gap channels).
    rng = np.random.default_rng(0)
    q = rng.random((5, 22)).astype(np.float32)
    np.testing.assert_allclose(remap_profile_channels(q).sum(-1), q.sum(-1), atol=1e-6)


# --- sample_msa ------------------------------------------------------------

def test_sample_msa_keeps_query_first_and_partitions():
    msa = np.arange(6 * 4).reshape(6, 4)
    dm = msa.astype(np.float32)
    mask = np.ones((6, 4), np.float32)
    clustered, extra = sample_msa(msa, dm, mask, max_seq=4, rng=np.random.default_rng(0))
    assert clustered["msa"].shape == (4, 4) and extra["extra_msa"].shape == (2, 4)
    # Row 0 (the query) is always kept first in the clustered set.
    np.testing.assert_array_equal(clustered["msa"][0], msa[0])
    # The split is a partition of all rows.
    seen = np.concatenate([clustered["msa"][:, 0], extra["extra_msa"][:, 0]])
    assert sorted(seen.tolist()) == sorted(msa[:, 0].tolist())


def test_sample_msa_row_order_injection_is_deterministic():
    msa = np.arange(5 * 3).reshape(5, 3)
    dm = msa.astype(np.float32)
    mask = np.ones((5, 3), np.float32)
    order = np.array([0, 3, 1, 4, 2])
    clustered, extra = sample_msa(msa, dm, mask, max_seq=3, row_order=order)
    np.testing.assert_array_equal(clustered["msa"], msa[[0, 3, 1]])
    np.testing.assert_array_equal(extra["extra_msa"], msa[[4, 2]])


def test_sample_msa_rejects_non_permutation_row_order():
    msa = np.arange(5 * 3).reshape(5, 3)
    dm = msa.astype(np.float32)
    mask = np.ones((5, 3), np.float32)
    with pytest.raises(ValueError, match="permutation"):
        sample_msa(msa, dm, mask, max_seq=3, row_order=np.array([1, 2, 3, 4, 0]))  # row 0 not first
    with pytest.raises(ValueError, match="permutation"):
        sample_msa(msa, dm, mask, max_seq=3, row_order=np.array([0, 1, 1]))  # short + dup


def test_crop_extra_rejects_injection_over_cap():
    extra = {
        "extra_msa": np.arange(6 * 2).reshape(6, 2),
        "extra_deletion_matrix": np.zeros((6, 2), np.float32),
        "extra_msa_mask": np.ones((6, 2), np.float32),
    }
    with pytest.raises(ValueError, match="exceeds max_extra_msa"):
        crop_extra_msa(extra, 2, extra_indices=np.array([0, 1, 2]))  # 3 > cap 2


def test_build_rejects_broadcastable_msa_mask():
    rng = np.random.default_rng(0)
    msa = rng.integers(0, 20, (6, 4)).astype(np.int32)
    with pytest.raises(ValueError, match="msa_mask shape"):
        build_msa_features(msa, np.zeros((6, 4), np.float32), msa[0],
                           msa_mask=np.ones((6, 1), np.float32), seed=0)


def test_sample_msa_max_seq_geq_n_gives_empty_extra():
    msa = np.arange(3 * 2).reshape(3, 2)
    clustered, extra = sample_msa(msa, msa.astype(np.float32), np.ones((3, 2), np.float32),
                                  max_seq=10, rng=np.random.default_rng(1))
    assert clustered["msa"].shape == (3, 2) and extra["extra_msa"].shape == (0, 2)


# --- make_masked_msa -------------------------------------------------------

def test_masked_msa_injection_is_deterministic_and_targets_positions():
    msa = np.array([[0, 1, 2, 3]], dtype=np.int64)  # openfold tokens
    profile = np.zeros((4, 22), np.float32)
    pos = np.array([[True, False, True, False]])
    repl = np.array([[22, 22, 22, 22]], dtype=np.int64)  # all MASK
    masked, true_msa, bert_mask = make_masked_msa(
        msa, profile, mask_position=pos, replacement_tokens=repl,
    )
    # Masked positions take the replacement; others stay original.
    np.testing.assert_array_equal(masked, [[22, 1, 22, 3]])
    np.testing.assert_array_equal(true_msa, msa)
    np.testing.assert_array_equal(bert_mask, [[1.0, 0.0, 1.0, 0.0]])


def test_masked_msa_rejects_bad_probs_and_fraction():
    msa = np.zeros((1, 3), np.int64)
    profile = np.zeros((3, 22), np.float32)
    with pytest.raises(ValueError, match="replace_fraction"):
        make_masked_msa(msa, profile, replace_fraction=1.5, mask_position=np.zeros((1, 3), bool),
                        replacement_tokens=np.zeros((1, 3), np.int64))
    with pytest.raises(ValueError, match="<= 1"):
        make_masked_msa(msa, profile, uniform_prob=0.5, profile_prob=0.5, same_prob=0.5,
                        mask_position=np.zeros((1, 3), bool), replacement_tokens=np.zeros((1, 3), np.int64))


# --- nearest_neighbor_clusters --------------------------------------------

def test_nn_assigns_identical_extra_to_its_center():
    msa = np.array([[0, 1, 2], [5, 6, 7]], dtype=np.int64)
    msa_mask = np.ones((2, 3), np.float32)
    extra = np.array([[5, 6, 7], [0, 1, 2]], dtype=np.int64)  # copies of center 1, then 0
    extra_mask = np.ones((2, 3), np.float32)
    a = nearest_neighbor_clusters(msa, msa_mask, extra, extra_mask)
    np.testing.assert_array_equal(a, [1, 0])


def test_nn_empty_extra_is_shape_zero():
    msa = np.zeros((2, 3), np.int64)
    a = nearest_neighbor_clusters(msa, np.ones((2, 3), np.float32),
                                  np.zeros((0, 3), np.int64), np.zeros((0, 3), np.float32))
    assert a.shape == (0,)


def test_nn_exact_tie_breaks_to_lowest_index():
    # Two identical centers; an extra equidistant — argmax picks index 0.
    msa = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    extra = np.array([[0, 1, 2]], dtype=np.int64)
    a = nearest_neighbor_clusters(msa, np.ones((2, 3), np.float32), extra, np.ones((1, 3), np.float32))
    assert a[0] == 0


# --- summarize_clusters ----------------------------------------------------

def test_summarize_empty_extra_is_center_only():
    msa = np.array([[0, 1, 2]], dtype=np.int64)
    dm = np.array([[2.0, 0.0, 4.0]], np.float32)
    cp, cdm = summarize_clusters(
        msa, np.ones((1, 3), np.float32), dm,
        np.zeros((0, 3), np.int64), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
        np.zeros((0,), np.int64),
    )
    # Profile ≈ the center one-hot (denominator ≈ 1); deletion mean ≈ center.
    assert cp.shape == (1, 3, MSA_ONE_HOT)
    np.testing.assert_allclose(cp[0, np.arange(3), msa[0]], 1.0, atol=1e-5)
    np.testing.assert_allclose(cdm[0], dm[0], atol=1e-5)


def test_summarize_profile_rows_sum_to_one_with_full_mask():
    rng = np.random.default_rng(3)
    msa = rng.integers(0, 20, (3, 5)).astype(np.int64)
    extra = rng.integers(0, 20, (7, 5)).astype(np.int64)
    a = nearest_neighbor_clusters(msa, np.ones((3, 5), np.float32), extra, np.ones((7, 5), np.float32))
    cp, _ = summarize_clusters(msa, np.ones((3, 5), np.float32), np.zeros((3, 5), np.float32),
                               extra, np.ones((7, 5), np.float32), np.zeros((7, 5), np.float32), a)
    np.testing.assert_allclose(cp.sum(-1), 1.0, atol=1e-4)


# --- crop_extra_msa --------------------------------------------------------

def test_crop_extra_caps_and_is_seed_reproducible():
    extra = {
        "extra_msa": np.arange(10 * 4).reshape(10, 4),
        "extra_deletion_matrix": np.zeros((10, 4), np.float32),
        "extra_msa_mask": np.ones((10, 4), np.float32),
    }
    a = crop_extra_msa(extra, 3, rng=np.random.default_rng(5))
    b = crop_extra_msa(extra, 3, rng=np.random.default_rng(5))
    assert a["extra_msa"].shape == (3, 4)
    np.testing.assert_array_equal(a["extra_msa"], b["extra_msa"])  # same seed → same crop
    # Injection is exact.
    c = crop_extra_msa(extra, 2, extra_indices=np.array([7, 2]))
    np.testing.assert_array_equal(c["extra_msa"], extra["extra_msa"][[7, 2]])


# --- make_msa_feat ---------------------------------------------------------

def test_msa_feat_channel_layout():
    N, L = 3, 4
    msa = np.zeros((N, L), np.int64)
    dm = np.zeros((N, L), np.float32)
    cp = np.zeros((N, L, MSA_ONE_HOT), np.float32)
    cdm = np.zeros((N, L), np.float32)
    aatype = np.zeros(L, np.int64)
    msa_feat, target_feat = make_msa_feat(msa, dm, cp, cdm, aatype)
    assert msa_feat.shape == (N, L, MSA_FEAT_CHANNELS)  # 49
    assert target_feat.shape == (L, 22)
    # The first 23 channels are the msa one-hot.
    np.testing.assert_array_equal(msa_feat[..., :MSA_ONE_HOT].argmax(-1), msa)


# --- build_msa_features orchestrator + edges -------------------------------

def _inputs(n, length, seed=0):
    rng = np.random.default_rng(seed)
    msa = rng.integers(0, 22, (n, length)).astype(np.int32)
    msa[0] = rng.integers(0, 20, length)
    dm = rng.integers(0, 5, (n, length)).astype(np.float32)
    return msa, dm, msa[0].copy().astype(np.int32)


def test_build_is_seed_reproducible_and_decorrelated():
    msa, dm, aatype = _inputs(12, 8)
    a = build_msa_features(msa, dm, aatype, max_seq=5, max_extra_msa=4, seed=7)
    b = build_msa_features(msa, dm, aatype, max_seq=5, max_extra_msa=4, seed=7)
    c = build_msa_features(msa, dm, aatype, max_seq=5, max_extra_msa=4, seed=8)
    np.testing.assert_array_equal(a.msa, b.msa)
    np.testing.assert_array_equal(a.bert_mask, b.bert_mask)
    # The clustered row mask is returned (codex) and sampled alongside msa.
    assert a.msa_mask.shape == a.msa.shape
    # Different seed → different sampling/masking (with high probability).
    assert not np.array_equal(a.bert_mask, c.bert_mask)


def test_build_default_call_without_seed_works():
    """The documented default invocation (seed=None) must produce features, not
    crash on a missing RNG (codex P1)."""
    msa, dm, aatype = _inputs(8, 6)
    f = build_msa_features(msa, dm, aatype, max_seq=4, max_extra_msa=3)
    assert f.msa_feat.shape == (4, 6, MSA_FEAT_CHANNELS)


def test_build_accepts_non_uniform_msa_mask():
    """The profile is computed internally (unmasked), so a partial-coverage
    msa_mask (what the canonical Rust backend emits) is accepted, not rejected,
    and flows through to the returned clustered mask (codex P1)."""
    msa, dm, aatype = _inputs(10, 6)
    mask = np.ones((10, 6), np.float32)
    mask[3:, 4:] = 0.0  # partial coverage, like a local-hit MSA
    f = build_msa_features(msa, dm, aatype, max_seq=5, max_extra_msa=4, msa_mask=mask, seed=1)
    assert f.msa_feat.shape == (5, 6, MSA_FEAT_CHANNELS)
    # The clustered mask carries the partial coverage (some zeros survive).
    assert f.msa_mask.shape == (5, 6) and f.msa_mask.min() == 0.0


def test_build_rejects_negative_extra_cap_and_bad_probs():
    msa, dm, aatype = _inputs(8, 6)
    with pytest.raises(ValueError, match="max_extra_msa"):
        build_msa_features(msa, dm, aatype, max_extra_msa=-1, seed=0)
    with pytest.raises(ValueError, match="finite and >= 0"):
        build_msa_features(msa, dm, aatype, uniform_prob=-0.1, seed=0)
    with pytest.raises(ValueError, match="finite and >= 0"):
        build_msa_features(msa, dm, aatype, profile_prob=float("nan"), seed=0)


def test_build_rejects_empty_msa_supports_single_row():
    with pytest.raises(ValueError, match="msa must be"):
        build_msa_features(np.zeros((0, 4), np.int32), np.zeros((0, 4), np.float32),
                           np.zeros(4, np.int32), seed=0)
    # A single-sequence MSA works: extra is empty, cluster_profile is the query.
    f = build_msa_features(np.zeros((1, 4), np.int32), np.zeros((1, 4), np.float32),
                           np.zeros(4, np.int32), seed=0)
    assert f.msa.shape == (1, 4) and f.extra_msa.shape == (0, 4)


def test_build_max_extra_zero_and_validation():
    msa, dm, aatype = _inputs(10, 6)
    f = build_msa_features(msa, dm, aatype, max_seq=3, max_extra_msa=0, seed=0)
    assert f.extra_msa.shape == (0, 6)
    with pytest.raises(ValueError, match="non-negative"):
        build_msa_features(msa, -dm - 1, aatype, seed=0)
    with pytest.raises(ValueError, match="max_seq"):
        build_msa_features(msa, dm, aatype, max_seq=0, seed=0)
