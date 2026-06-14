"""AlphaFold/OpenFold MSA feature pipeline — extra-MSA + clustering + BERT mask.

Framework-neutral NumPy port of OpenFold's MSA transforms
(`openfold/data/data_transforms.py`), gated against OpenFold for parity. Turns a
raw MSA (+ deletion matrix + full-MSA profile) into the model-ready `msa_feat`
(clustered) / `extra_*` (shallow) features, in **OpenFold channel order**.

Transform order matches OpenFold `input_pipeline`:
`sample_msa → make_masked_msa → nearest_neighbor_clusters → summarize_clusters
→ crop_extra_msa → make_msa_feat`. Clustering/summarize therefore run on the
*masked* clustered MSA — replicated exactly, including OpenFold's center-mask
asymmetry in `summarize_clusters`.

Encoding: all transforms run in **OpenFold** token order (residues
`ARNDCQEGHILKMFPSTWYV`, `X=20`, `gap=21`, `mask=22`); `build_msa_features` remaps
proteon-order inputs (`ACDEFGHIKLMNPQRSTVWYX`) once at the top. Tokens remap by
*value* (gather); profile *channels* remap by *scatter* (the inverse permutation).

RNG: `build_msa_features` draws every stochastic choice from a single seeded
`np.random.Generator`, consumed sequentially. Each stochastic choice is also
injectable (`row_order`, `mask_position`, `replacement_tokens`, `extra_indices`)
so the deterministic outputs can be oracle-gated bit-for-bit against OpenFold.

Parity is integer-exact (token/mask outputs) and float-`allclose` (profiles,
deletion means, probabilities) — never bit-exact float (Torch `scatter_add_` vs
NumPy accumulation order differ). Oracle on CPU only. See
`devdocs/SHALLOW_MSA_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# proteon residue order -> OpenFold restype order, by index. proteon
# "ACDEFGHIKLMNPQRSTVWYX" (X=20) -> OpenFold "ARNDCQEGHILKMFPSTWYV" (unk=20).
_PT_TO_OF = np.array(
    [0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20],
    dtype=np.int64,
)
# Token map incl. gap(21)/mask(22) passthrough — both alphabets agree there.
_MSA_TOKEN_MAP = np.concatenate([_PT_TO_OF, np.array([21, 22], dtype=np.int64)])
# Profile channel map (22-wide profiles: residues 0-19, X=20, gap=21).
_PROFILE_CHANNEL_MAP = np.concatenate([_PT_TO_OF, np.array([21], dtype=np.int64)])

GAP_TOKEN = 21
MASK_TOKEN = 22
MSA_ONE_HOT = 23  # residues(20) + X + gap + mask
AATYPE_ONE_HOT = 21
PROFILE_WIDTH = 22
MSA_FEAT_CHANNELS = 49  # 23 + 1 + 1 + 23 + 1


def remap_msa_tokens(tokens: NDArray) -> NDArray[np.int64]:
    """Remap proteon-order MSA/aatype token *values* to OpenFold order (gather)."""
    t = np.asarray(tokens, dtype=np.int64)
    if t.size and (t.min() < 0 or t.max() >= _MSA_TOKEN_MAP.shape[0]):
        raise ValueError(f"token out of range [0, {_MSA_TOKEN_MAP.shape[0]})")
    return _MSA_TOKEN_MAP[t]


def remap_profile_channels(profile: NDArray) -> NDArray[np.float32]:
    """Remap a proteon-order `(L, 22)` profile's *channels* to OpenFold order via
    scatter — `of[..., MAP] = pt` — the inverse of a gather (codex v2 catch)."""
    p = np.asarray(profile, dtype=np.float32)
    if p.shape[-1] != PROFILE_WIDTH:
        raise ValueError(f"profile last dim must be {PROFILE_WIDTH}, got {p.shape[-1]}")
    out = np.zeros_like(p)
    out[..., _PROFILE_CHANNEL_MAP] = p
    return out


def _one_hot(x: NDArray, num_classes: int) -> NDArray[np.float32]:
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros(x.shape + (num_classes,), dtype=np.float32)
    np.put_along_axis(out, x[..., None], 1.0, axis=-1)
    return out


def _unsorted_segment_sum(data: NDArray, segment_ids: NDArray, num_segments: int) -> NDArray:
    """Sum `data` rows into `num_segments` bins by `segment_ids` (≈ tf/torch)."""
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    if data.shape[0]:
        np.add.at(out, np.asarray(segment_ids, dtype=np.int64), data)
    return out


def sample_msa(
    msa: NDArray,
    deletion_matrix: NDArray,
    msa_mask: NDArray,
    max_seq: int,
    *,
    rng: Optional[np.random.Generator] = None,
    row_order: Optional[NDArray] = None,
) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
    """Split the MSA into the clustered set (first `max_seq` of a permutation that
    keeps row 0 first) and the shallow `extra` set (the rest). `row_order`, if
    given, is the full index order (row 0 first); else drawn from `rng`."""
    num_seq = msa.shape[0]
    if max_seq < 1:
        raise ValueError(f"max_seq must be >= 1, got {max_seq}")
    if row_order is not None:
        index_order = np.asarray(row_order, dtype=np.int64)
        # Must be a full permutation with the query (row 0) first — else rows are
        # silently dropped/duplicated rather than partitioned (codex).
        if index_order.shape != (num_seq,) or index_order[0] != 0 or \
                not np.array_equal(np.sort(index_order), np.arange(num_seq)):
            raise ValueError("row_order must be a length-num_seq permutation with row 0 first")
    else:
        if rng is None:
            raise ValueError("sample_msa needs an rng or an explicit row_order")
        index_order = np.concatenate([[0], rng.permutation(num_seq - 1) + 1]).astype(np.int64)
    num_sel = min(max_seq, num_seq)
    sel, not_sel = index_order[:num_sel], index_order[num_sel:]
    clustered = {
        "msa": msa[sel],
        "deletion_matrix": deletion_matrix[sel],
        "msa_mask": msa_mask[sel],
    }
    extra = {
        "extra_msa": msa[not_sel],
        "extra_deletion_matrix": deletion_matrix[not_sel],
        "extra_msa_mask": msa_mask[not_sel],
    }
    return clustered, extra


def _shaped_categorical(probs: NDArray, rng: np.random.Generator) -> NDArray[np.int64]:
    """Sample one token per position. Matches OpenFold `shaped_categorical`: add
    `1e-10`, normalize each per-position vector, then inverse-CDF sample."""
    p = probs.astype(np.float64) + 1e-10
    p = p / p.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(p, axis=-1)
    u = rng.random(probs.shape[:-1] + (1,))
    return np.argmax(u < cdf, axis=-1).astype(np.int64)


def make_masked_msa(
    msa: NDArray,
    profile: NDArray,
    *,
    uniform_prob: float = 0.1,
    profile_prob: float = 0.1,
    same_prob: float = 0.1,
    replace_fraction: float = 0.15,
    rng: Optional[np.random.Generator] = None,
    mask_position: Optional[NDArray] = None,
    replacement_tokens: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """BERT-mask the (OpenFold-encoded) clustered MSA. Returns
    `(masked_msa, true_msa, bert_mask)`. `categorical = uniform·random_aa +
    profile·hhblits_profile + same·onehot(msa)` with a 23rd `[MASK]` column of
    `mask_prob = 1 − those three`. `mask_position` / `replacement_tokens` are
    injectable; the latter must be the *full* sampled-token result."""
    if not (0.0 <= replace_fraction <= 1.0):  # NaN also fails this (raises)
        raise ValueError(f"replace_fraction must be in [0, 1], got {replace_fraction}")
    for name, p in (("uniform_prob", uniform_prob), ("profile_prob", profile_prob), ("same_prob", same_prob)):
        # Each must be finite and non-negative — else the categorical mixture goes
        # negative or NaN and sampling silently corrupts the replacement (codex).
        if not np.isfinite(p) or p < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {p}")
    mask_prob = 1.0 - uniform_prob - profile_prob - same_prob
    if mask_prob < 0.0:
        raise ValueError("uniform_prob + profile_prob + same_prob must be <= 1")

    random_aa = np.array([0.05] * 20 + [0.0, 0.0], dtype=np.float32)  # (22,)
    categorical = (
        uniform_prob * random_aa[None, None, :]
        + profile_prob * profile[None, :, :]
        + same_prob * _one_hot(msa, PROFILE_WIDTH)
    )  # (N, L, 22)
    mask_col = np.full(categorical.shape[:-1] + (1,), mask_prob, dtype=np.float32)
    categorical = np.concatenate([categorical, mask_col], axis=-1)  # (N, L, 23)

    if mask_position is None:
        if rng is None:
            raise ValueError("make_masked_msa needs an rng or explicit mask_position")
        mask_position = rng.random(msa.shape) < replace_fraction
    mask_position = np.asarray(mask_position, dtype=bool)

    if replacement_tokens is None:
        if rng is None:
            raise ValueError("make_masked_msa needs an rng or explicit replacement_tokens")
        replacement_tokens = _shaped_categorical(categorical, rng)
    bert_msa = np.where(mask_position, np.asarray(replacement_tokens, dtype=np.int64), msa)
    return bert_msa.astype(np.int64), np.asarray(msa, dtype=np.int64), mask_position.astype(np.float32)


def nearest_neighbor_clusters(
    msa: NDArray,
    msa_mask: NDArray,
    extra_msa: NDArray,
    extra_msa_mask: NDArray,
    *,
    gap_agreement_weight: float = 0.0,
) -> NDArray[np.int64]:
    """Assign each extra sequence to its closest clustered sequence by weighted
    one-hot Hamming agreement (`argmax`). Empty extra → shape `(0,)`."""
    weights = np.concatenate(
        [np.ones(21, np.float32), np.array([gap_agreement_weight], np.float32), np.zeros(1, np.float32)]
    )  # (23,)
    num_seq, num_res = msa.shape
    extra_num = extra_msa.shape[0]
    if extra_num == 0:
        return np.zeros((0,), dtype=np.int64)
    sample_oh = msa_mask[:, :, None] * _one_hot(msa, MSA_ONE_HOT)
    extra_oh = extra_msa_mask[:, :, None] * _one_hot(extra_msa, MSA_ONE_HOT)
    agreement = extra_oh.reshape(extra_num, num_res * MSA_ONE_HOT) @ (
        sample_oh * weights
    ).reshape(num_seq, num_res * MSA_ONE_HOT).T  # (E, N)
    return np.argmax(agreement, axis=1).astype(np.int64)


def summarize_clusters(
    msa: NDArray,
    msa_mask: NDArray,
    deletion_matrix: NDArray,
    extra_msa: NDArray,
    extra_msa_mask: NDArray,
    extra_deletion_matrix: NDArray,
    extra_cluster_assignment: NDArray,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-cluster `(cluster_profile, cluster_deletion_mean)`. Replicates
    OpenFold's center-mask asymmetry exactly: extra rows are mask-weighted, the
    center one-hot/deletion are added **un-masked**, only the denominator counts
    the center mask (codex v2: do not 'fix' this)."""
    num_seq = msa.shape[0]

    def csum(x):
        return _unsorted_segment_sum(x, extra_cluster_assignment, num_seq)

    mask = extra_msa_mask
    mask_counts = 1e-6 + msa_mask + csum(mask)  # include center in denominator
    msa_sum = csum(mask[:, :, None] * _one_hot(extra_msa, MSA_ONE_HOT))
    msa_sum += _one_hot(msa, MSA_ONE_HOT)  # center, un-masked
    cluster_profile = msa_sum / mask_counts[:, :, None]
    del_sum = csum(mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # center, un-masked
    cluster_deletion_mean = del_sum / mask_counts
    return cluster_profile.astype(np.float32), cluster_deletion_mean.astype(np.float32)


def crop_extra_msa(
    extra: Dict[str, NDArray],
    max_extra_msa: int,
    *,
    rng: Optional[np.random.Generator] = None,
    extra_indices: Optional[NDArray] = None,
) -> Dict[str, NDArray]:
    """Random-subsample the extra MSA to `max_extra_msa` rows. `extra_indices`
    injectable; else drawn from `rng`."""
    if max_extra_msa < 0:
        raise ValueError(f"max_extra_msa must be >= 0, got {max_extra_msa}")
    num = extra["extra_msa"].shape[0]
    num_sel = min(max_extra_msa, num)
    if extra_indices is not None:
        sel = np.asarray(extra_indices, dtype=np.int64)
        # Honour the cap even for injected indices — else max_extra_msa is
        # bypassed and the model gets oversized inputs (codex).
        if sel.shape[0] > num_sel:
            raise ValueError(
                f"extra_indices has {sel.shape[0]} entries, exceeds max_extra_msa cap {num_sel}"
            )
    elif num == 0:
        sel = np.zeros((0,), dtype=np.int64)
    else:
        if rng is None:
            raise ValueError("crop_extra_msa needs an rng or explicit extra_indices")
        sel = rng.permutation(num)[:num_sel]
    return {k: v[sel] for k, v in extra.items()}


def _deletion_value(deletion_matrix: NDArray) -> NDArray[np.float32]:
    return (np.arctan(np.asarray(deletion_matrix, np.float32) / 3.0) * (2.0 / np.pi)).astype(np.float32)


@dataclass
class MsaFeatures:
    """Model-ready MSA features, all in **OpenFold channel order**.

    `msa_feat` `(N_clust, L, 49)` = onehot(23) + has_deletion(1) + deletion_value(1)
    + cluster_profile(23) + cluster_deletion_mean_value(1). `target_feat` `(L, 22)`
    = has_break(1) + aatype_1hot(21). `extra_*` are the shallow-stack inputs."""

    msa_feat: NDArray[np.float32]
    target_feat: NDArray[np.float32]
    msa: NDArray[np.int64]          # masked clustered MSA (OpenFold tokens)
    true_msa: NDArray[np.int64]     # pre-mask clustered MSA (BERT target)
    msa_mask: NDArray[np.float32]   # clustered row mask (sampled alongside msa)
    bert_mask: NDArray[np.float32]
    cluster_profile: NDArray[np.float32]
    cluster_deletion_mean: NDArray[np.float32]
    extra_msa: NDArray[np.int64]
    extra_msa_mask: NDArray[np.float32]
    extra_has_deletion: NDArray[np.float32]
    extra_deletion_value: NDArray[np.float32]


def make_msa_feat(
    msa: NDArray,
    deletion_matrix: NDArray,
    cluster_profile: NDArray,
    cluster_deletion_mean: NDArray,
    aatype: NDArray,
    *,
    between_segment_residues: Optional[NDArray] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """`(msa_feat (N,L,49), target_feat (L,22))` — the OpenFold channel assembly.
    `aatype`/`msa` are OpenFold-encoded; `between_segment_residues` defaults to
    zeros (no domain break for single chains)."""
    num_res = msa.shape[1]
    if between_segment_residues is None:
        has_break = np.zeros((num_res,), dtype=np.float32)
    else:
        has_break = np.clip(np.asarray(between_segment_residues, np.float32), 0.0, 1.0)
    target_feat = np.concatenate(
        [has_break[:, None], _one_hot(aatype, AATYPE_ONE_HOT)], axis=-1
    ).astype(np.float32)  # (L, 22)

    msa_1hot = _one_hot(msa, MSA_ONE_HOT)
    has_deletion = np.clip(np.asarray(deletion_matrix, np.float32), 0.0, 1.0)
    deletion_value = _deletion_value(deletion_matrix)
    deletion_mean_value = _deletion_value(cluster_deletion_mean)
    msa_feat = np.concatenate(
        [
            msa_1hot,
            has_deletion[..., None],
            deletion_value[..., None],
            cluster_profile,
            deletion_mean_value[..., None],
        ],
        axis=-1,
    ).astype(np.float32)  # (N, L, 49)
    return msa_feat, target_feat


def build_msa_features(
    msa: NDArray,
    deletion_matrix: NDArray,
    aatype: NDArray,
    *,
    max_seq: int = 512,
    max_extra_msa: int = 1024,
    uniform_prob: float = 0.1,
    profile_prob: float = 0.1,
    same_prob: float = 0.1,
    replace_fraction: float = 0.15,
    gap_agreement_weight: float = 0.0,
    msa_mask: Optional[NDArray] = None,
    between_segment_residues: Optional[NDArray] = None,
    seed: Optional[int] = None,
    row_order: Optional[NDArray] = None,
    mask_position: Optional[NDArray] = None,
    replacement_tokens: Optional[NDArray] = None,
    extra_indices: Optional[NDArray] = None,
) -> MsaFeatures:
    """Run the full pipeline on **proteon-encoded** inputs.

    Remaps `msa`/`aatype` to OpenFold order once, computes the BERT-masking
    profile as OpenFold's `make_hhblits_profile` does (the **unmasked** mean of
    one-hot rows over the full MSA — so it is mask-independent and the pipeline
    consumes any `msa_mask`, including the canonical Rust backend's partial-
    coverage masks; codex), then sample → mask → cluster → summarize → crop →
    assemble. Every stochastic choice comes from one `np.random.Generator(seed)`
    consumed in order (or the injected `row_order`/`mask_position`/
    `replacement_tokens`/`extra_indices`).

    Rejects an empty MSA (OpenFold assumes row 0 exists); a single-row MSA is
    supported. `msa_mask` defaults to all-ones."""
    msa = np.asarray(msa, dtype=np.int64)
    if msa.ndim != 2 or msa.shape[0] < 1:
        raise ValueError("msa must be (N>=1, L)")
    n_seq, length = msa.shape
    deletion_matrix = np.asarray(deletion_matrix, dtype=np.float32)
    if deletion_matrix.shape != (n_seq, length):
        raise ValueError("deletion_matrix shape must match msa")
    if np.any(deletion_matrix < 0):
        raise ValueError("deletion_matrix must be non-negative")
    if msa_mask is None:
        msa_mask = np.ones((n_seq, length), dtype=np.float32)
    else:
        msa_mask = np.asarray(msa_mask, dtype=np.float32)
        if msa_mask.shape != (n_seq, length):
            raise ValueError(
                f"msa_mask shape {msa_mask.shape} must match msa {(n_seq, length)} "
                "(broadcastable shapes silently corrupt clustering summaries)"
            )

    # Always have a usable Generator (seed=None → OS entropy, stochastic but
    # valid) so the default invocation works; injected draws still take
    # precedence per transform (codex P1).
    rng = np.random.default_rng(seed)

    # Remap tokens to OpenFold order (gather), then derive the OpenFold
    # `hhblits_profile`: the unmasked mean of one-hot(22) over the full MSA. This
    # is what OpenFold uses for masking and is independent of msa_mask, so partial
    # masks are supported with exact parity (codex P1).
    msa_of = remap_msa_tokens(msa)
    aatype_of = remap_msa_tokens(aatype)
    profile_of = _one_hot(msa_of, PROFILE_WIDTH).mean(axis=0).astype(np.float32)

    clustered, extra = sample_msa(
        msa_of, deletion_matrix, msa_mask, max_seq, rng=rng, row_order=row_order
    )
    masked, true_msa, bert_mask = make_masked_msa(
        clustered["msa"], profile_of,
        uniform_prob=uniform_prob, profile_prob=profile_prob, same_prob=same_prob,
        replace_fraction=replace_fraction, rng=rng,
        mask_position=mask_position, replacement_tokens=replacement_tokens,
    )
    clustered["msa"] = masked

    assignment = nearest_neighbor_clusters(
        clustered["msa"], clustered["msa_mask"], extra["extra_msa"], extra["extra_msa_mask"],
        gap_agreement_weight=gap_agreement_weight,
    )
    cluster_profile, cluster_deletion_mean = summarize_clusters(
        clustered["msa"], clustered["msa_mask"], clustered["deletion_matrix"],
        extra["extra_msa"], extra["extra_msa_mask"], extra["extra_deletion_matrix"],
        assignment,
    )
    extra = crop_extra_msa(extra, max_extra_msa, rng=rng, extra_indices=extra_indices)

    msa_feat, target_feat = make_msa_feat(
        clustered["msa"], clustered["deletion_matrix"], cluster_profile, cluster_deletion_mean,
        aatype_of, between_segment_residues=between_segment_residues,
    )
    return MsaFeatures(
        msa_feat=msa_feat,
        target_feat=target_feat,
        msa=clustered["msa"],
        true_msa=true_msa,
        msa_mask=clustered["msa_mask"].astype(np.float32),
        bert_mask=bert_mask,
        cluster_profile=cluster_profile,
        cluster_deletion_mean=cluster_deletion_mean,
        extra_msa=extra["extra_msa"].astype(np.int64),
        extra_msa_mask=extra["extra_msa_mask"].astype(np.float32),
        extra_has_deletion=np.clip(extra["extra_deletion_matrix"], 0.0, 1.0).astype(np.float32),
        extra_deletion_value=_deletion_value(extra["extra_deletion_matrix"]),
    )
