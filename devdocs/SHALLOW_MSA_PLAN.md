# Shallow MSA — extra-MSA + stochastic clustering + BERT masking

## Motivation

proteon's `SequenceExample` carries the **deterministic** MSA features today —
`msa`, `deletion_matrix`, `msa_mask`, `msa_profile`, and the per-(seq,res)
`has_deletion` / `deletion_value`. The `sequence_example.py` source explicitly
notes the gap: *"The cluster_profile / extra-MSA features need the stochastic
clustering pipeline"*. This is the last `make_msa_feat` gap (roadmap §8).

AlphaFold/OpenFold turns a raw MSA into model inputs through a fixed transform
chain (`openfold/data/data_transforms.py`):

1. `sample_msa(max_seq, keep_extra, seed)` — random-permute rows (row 0 = query
   stays first), take the first `max_seq` as the **clustered** MSA; the rest
   become the **shallow `extra_*` MSA**.
2. `make_masked_msa(config, replace_fraction, seed)` — BERT masking on the
   clustered MSA: `categorical = uniform·random_aa + profile·hhblits_profile +
   same·onehot(msa)`, pad a `[MASK]` column with the remaining mass; positions
   where `rand < replace_fraction` get resampled. Sets `true_msa`, `bert_mask`,
   `msa = bert_msa`.
3. `nearest_neighbor_clusters(gap_agreement_weight)` — assign each extra sequence
   to its nearest clustered sequence by weighted one-hot(23) Hamming agreement
   (`argmax`); weights `[1]*21 + [gap_w] + [0]`.
4. `summarize_clusters` — per-cluster `cluster_profile` (one-hot sums incl. the
   center / mask-counts) and `cluster_deletion_mean`, via segment-sum over the
   assignment.
5. `crop_extra_msa(max_extra_msa, seed)` — random-subsample extra to a cap.
6. `make_msa_feat` — assemble `msa_feat` `(N_clust, L, 49)` =
   `[onehot(23), has_deletion(1), deletion_value(1), cluster_profile(23),
   cluster_deletion_mean_value(1)]`, plus `extra_has_deletion` /
   `extra_deletion_value` for the extra stack, and `target_feat`
   `[has_break(1), aatype_1hot(21)]` `(L, 22)`.

This plan ports steps 1–6 as **framework-neutral NumPy** transforms (proteon's
supervision layer is NumPy + oracle-gated vs OpenFold, like the torsions /
deletion features already shipped).

## Scope

**In scope** — a new `msa_features.py`:

- `sample_msa(msa, deletion_matrix, msa_mask, max_seq, *, seed) -> Clustered, Extra`
- `make_masked_msa(msa, profile, *, uniform_prob, profile_prob, same_prob,
  replace_fraction, seed) -> (masked_msa, true_msa, bert_mask)`
- `nearest_neighbor_clusters(msa, msa_mask, extra_msa, extra_msa_mask, *,
  gap_agreement_weight=0.0) -> extra_cluster_assignment`
- `summarize_clusters(msa, msa_mask, deletion_matrix, extra_msa, extra_msa_mask,
  extra_deletion_matrix, extra_cluster_assignment) -> (cluster_profile,
  cluster_deletion_mean)`
- `crop_extra_msa(extra_*, max_extra_msa, *, seed)`
- `make_msa_feat(...) -> MsaFeatures` (msa_feat, target_feat, extra_has_deletion,
  extra_deletion_value, ...)
- `build_msa_features(msa, deletion_matrix, msa_profile, aatype, *, max_seq,
  max_extra_msa, mask config, seed)` — orchestrates 1→6 into a result dataclass.

Wire into `SequenceExample` as **optional** fields (like the existing MSA
features), populated by an opt-in builder; serialization can follow in a
separate PR (these are large, stochastic, regenerated per training step — they
may stay compute-at-load, not persisted).

**Out of scope (deferred)** — `block_delete_msa` (training augmentation, not a
feature); `make_fixed_size` padding (the collator's job); multimer MSA;
persisting `msa_feat` to parquet (decide later — likely recomputed per step).

## The RNG-parity problem (the crux)

OpenFold's **stochastic** steps (`sample_msa` permutation, `make_masked_msa`
sample positions, `crop_extra_msa` subsample) draw from a **torch** `Generator`.
NumPy cannot reproduce torch's RNG stream bit-for-bit, so proteon **cannot**
bit-match the *random choices*. Two-tier oracle strategy:

- **Deterministic transforms** (`nearest_neighbor_clusters`,
  `summarize_clusters`, `make_msa_feat`, and `make_masked_msa`'s
  `categorical_probs` *construction*) are pure functions of their inputs →
  oracle-gate **bit-exact vs OpenFold** by feeding *identical* post-sample state
  (same sampled msa, same cluster assignment, same mask positions) to both.
- **Stochastic choices** (which rows sampled, which positions masked) use a
  seedable NumPy `Generator` (reproducible *within* proteon) and are gated by
  **invariants/distribution**, not OpenFold bit-equality: row 0 always kept
  first; `|selected| == min(max_seq, N)`; selected ∪ extra partition the rows;
  masked fraction ≈ `replace_fraction`; resampled tokens drawn from the right
  categorical. To make even the deterministic oracle feedable, each stochastic
  function also accepts **pre-drawn indices/positions** (so a test can inject
  OpenFold's choices and assert the rest matches bit-exact).

## Test plan

- **Oracle (vs OpenFold, gated; needs torch+openfold venv):** feed identical
  sampled MSA + cluster assignment → `nearest_neighbor_clusters`,
  `summarize_clusters`, `make_msa_feat` bit-match (atol 1e-6). `categorical_probs`
  matches for `make_masked_msa`.
- **Invariants (pure NumPy, CI):** `sample_msa` keeps row 0, partitions rows,
  respects `max_seq`; cluster assignment ∈ `[0, N_clust)`; identical extra rows
  cluster to the right center; `cluster_profile` rows sum to ~1 over non-masked;
  `msa_feat` channel count = 49 and slices equal the documented components;
  `crop_extra_msa` caps and is seed-reproducible; masked fraction ≈ target on a
  large synthetic MSA; an empty / single-sequence MSA degrades gracefully
  (extra is empty, cluster_profile = the query).
- **Seed reproducibility:** same seed → identical outputs; different seed →
  different sampling (within proteon).

## Claudex review outcome (adopted)

1. **Alphabet remap is MANDATORY, at one boundary.** proteon's
   `ACDEFGHIKLMNPQRSTVWYX` ≠ OpenFold restype `ARNDCQEGHILKMFPSTWYV`. Remap `msa`,
   `aatype`, and `msa_profile` to OpenFold order **once** at the top of
   `build_msa_features` (reuse the existing `_PT_TO_OF` table). `MsaFeatures` is
   defined to be in **OpenFold channel order** (documented on the dataclass); raw
   persisted data stays in proteon order. Without this, `msa_feat[...,:23]`,
   `target_feat`, `cluster_profile`, and `make_masked_msa`'s same-token term are
   all wrong (gap/X keep 21/20 but the 20 residue channels do not coincide).
2. **RNG injection must cover ALL stochastic outputs**, not just positions.
   Explicit injectable params, mutually exclusive with `seed`/`rng`: `row_order`
   (sample_msa), `mask_position` **and** `replacement_tokens` (make_masked_msa —
   OpenFold's `shaped_categorical` draws tokens on a *separate* RNG path, so
   positions alone can't oracle-gate the masked MSA), `extra_indices`
   (crop_extra_msa). This makes every deterministic output oracle-feedable.
3. **Parity = integer-exact + float-allclose**, never bit-exact floats. Exact
   equality for msa/mask integer outputs and `bert_mask`; `allclose` (atol 1e-6,
   float32 throughout) for `cluster_profile`, `cluster_deletion_mean`,
   probabilities, `atan`/`deletion_value`. Oracle only on CPU (Torch
   `scatter_add_` vs NumPy reductions differ in accumulation order; GPU worse).
4. **nearest_neighbor_clusters ties:** both `np`/`torch` argmax take the first
   max, but float accumulation can flip a near-tie — so test exact ties (→ lowest
   index), duplicate centers, zero masks, fractional `gap_agreement_weight`;
   oracle the assignment exactly on CPU only.
5. **Profile (`make_hhblits_profile`):** proteon's `msa_profile` is `(L,22)`
   incl. gap, computed pre-sampling — equivalent to OpenFold's `hhblits_profile`
   **only after the same permutation**. Validate `(L,22)` float32 + remap; the
   23rd `[MASK]` channel is added *only* by `make_masked_msa`, never in the input
   profile. Add an oracle test vs `make_hhblits_profile` incl. gap-heavy columns.

**Edge cases (all to be tested):** reject an empty MSA (OpenFold assumes row 0)
but support a single-row MSA; require `max_seq >= 1`; empty extra → assignment
`(0,)`, center-only summaries, `(0,L)` extra features; `max_seq >= N` and
`max_extra_msa == 0`; validate probs finite/nonneg/`sum<=1` and
`replace_fraction ∈ [0,1]` (don't silently clip negative mask mass); validate
token ranges / shapes / nonneg deletions; all-gap columns give gap prob 1 (not
zero-profile); zero-coverage masked columns deliberately match OpenFold's `1e-6`
denominator.

**Newly load-bearing (was going to defer):**
- **Seed derivation.** Don't expose only a caller scalar — provide
  `derive_msa_seed(record_id, epoch, worker, rank)` so sampling is reproducible
  and decorrelated per example/epoch/worker (the data-loader contract). The raw
  `seed` stays accepted for tests.
- **`between_segment_residues`.** Set explicitly to zeros for single chains (so
  `target_feat`'s `has_break` channel is parity-complete), not omitted.

**Confirmed deferrals:** `block_delete_msa`, `make_fixed_size` padding, multimer,
and **persistence** (recompute stochastic features at load; persist only raw MSA
+ deletion + mask + the full-MSA profile; return a transient `MsaFeatures` from
the loader rather than putting sampled/masked outputs on the durable
`SequenceExample`).

## Claudex v2 outcome (implementation-level, adopted)

A second pass on the revised plan caught parity traps that bite at code time:

1. **Remap DIRECTION differs for tokens vs channels.** Token arrays (`msa`,
   `aatype`) remap **by value** (gather): `of_tok = _MSA_PT_TO_OF[pt_tok]` where
   `_MSA_PT_TO_OF = PT_TO_OF + [21, 22]` (gap/mask passthrough). Profile
   *channels* remap by **scatter**, the inverse: `of_profile[..., PT_TO_OF] =
   pt_profile` — NOT `pt_profile[..., PT_TO_OF]`. Getting this backwards
   silently permutes profiles wrong. One helper each, clearly named.
2. **Profile is the FULL-MSA profile, remapped — never recomputed from sampled
   rows.** OpenFold `make_hhblits_profile` is an *unmasked* mean of one-hot rows;
   proteon `compute_msa_profile` weights by `msa_mask`. They coincide only when
   `msa_mask` is all-ones — which it currently is (set to ones in
   `build_sequence_example`). Documented **precondition**: all-ones raw mask;
   otherwise the divergence is deliberate and oracle-defined.
3. **One `np.random.Generator(seed)` in `build_msa_features`, consumed
   sequentially** across the four stochastic draws (row permutation, mask
   positions, categorical tokens, extra subselection) — NOT a fresh re-seed per
   transform (that correlates streams). Injection args (`row_order`,
   `mask_position`, `replacement_tokens`, `extra_indices`) are exposed on the
   orchestrator too, not just the leaf transforms.
4. **`shaped_categorical` parity:** OpenFold adds `1e-10` to every category and
   `Categorical` normalizes. The NumPy port must add `1e-10` then normalize each
   per-position vector before sampling (matters at zero-coverage columns where
   the mixture sums < 1). Oracle the *normalized* effective probabilities, not
   just the pre-normalization mixture.
5. **Don't "fix" `summarize_clusters`' center-mask asymmetry.** Extra rows are
   multiplied by their masks, but the center one-hot / center deletion are added
   **un-masked**; only the denominator (`mask_counts`) includes the center mask.
   Replicate exactly. Restrict the "cluster_profile rows sum to ~1" invariant to
   valid (unpadded) center masks.
6. **Seed derivation is OUT of this PR.** Accept an explicit `seed` only. The
   `(global_seed, record_id, epoch)` derivation belongs at the loader-integration
   boundary later — worker/rank must affect worker stream init, not an example's
   identity. (Revises the v1 "newly load-bearing" item — drop `derive_msa_seed`
   here.)

`replacement_tokens` must be the **full shaped-categorical result** (sampled
token ids), not raw uniforms. The plan is now implementation-ready.
