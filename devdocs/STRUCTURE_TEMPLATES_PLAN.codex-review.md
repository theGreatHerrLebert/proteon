# Codex plan review — STRUCTURE_TEMPLATES_PLAN.md (2026-06-14)

Independent review of the structure-based template-features plan. Verdict: the
core raw-feature construction is sound (template-native frame, aligned residues
mapped onto query rows); several load-bearing fixes folded into the plan.

## Major catches (all incorporated)
1. **Core construction correct** — store template-native-frame coords on aligned
   query rows; do NOT superpose; store only aligned residues, not the full chain.
2. **[CRITICAL] Indexing bug.** `extract_for_alignment()` skips residues lacking
   CA, so walking `aligned_seq_x/_y` yields indices into the *aligner's filtered*
   residue list, not atom37 rows — and it concatenates chains when none is
   selected. Must carry an explicit `alignment_index → atom37_index` map; don't
   infer position from letters. → made `StructuralCorrespondence` (explicit
   original residue indices) the T1 core.
3. **Aligned-pairs contract:** equal-length strings; ungapped lengths == inputs;
   emitted pairs == `n_aligned`; query/template indices monotonic + unique.
   Missing-CA residues can't be aligned even if other atoms exist; aligned
   residues with missing atoms keep per-atom masks.
4. **[CRITICAL] Template discontinuity → false torsions.** Query-adjacent mapped
   rows can map to *nonconsecutive template* residues (template insertions); a
   naive torsion call forges a peptide bond. Pass mapped template residue indices
   as continuity → mask pre_omega/phi on a break (reuses the `residue_index`
   continuity arg already in `compute_torsion_angles_sin_cos`). Query gaps cover
   some cases, template insertions do not.
5. **Downstream contract test** — template_aatype 22-class (incl unknown+gap),
   template identity (not query), and an actual OpenFold preprocessing/model-input
   smoke test incl. zero-template + partial-mask.
6. **[IMPORTANT] `sum_probs` must NOT be calibrated.** HHsearch-specific; no
   justified TM→prob map. Per-result-set max-norm is *wrong* (value changes with
   competitors). v0 = documented raw query-length-normalized TM-score (× aligned
   coverage optionally); expose components separately; calibration is later.
7. **Validation too self-referential.** Self-alignment catches wiring only. Add
   hand-authored cases (terminal offsets, indels, missing-CA, unknown residues,
   chain breaks, multichain, repeated sequences), rigid-transform metamorphic
   tests, a differential vs OpenFold given the *same explicit mapping*, and an
   OpenFold consumer smoke test. Close-pair agreement should be exact (not "small
   shift"). Remote-homolog = a curated benchmark (recall, TM/coverage dists), not
   one anecdote.
8. **[CRITICAL framing] Query structure must already exist.** Structural retrieval
   needs the structure being predicted, so this is NOT de-novo AF templating —
   it's for experimentally-known queries, refinement, or iterative retrieval from
   a preliminary prediction. Different products + a leakage path. State the
   workflow explicitly.

## Open-question dispositions
- Frame = template-native; aatype = template identity (decide now).
- Multichain → single-chain v0, require explicit chain, reject multichain inputs.
- Retrieval = NOT delivered in this feature (featurizer-only is the bounded slice).
- sum_probs = raw documented TM-score for v0.
- Benchmarking remains open; use the one-to-many align API with candidate caps.

## Phasing
T1 = `StructuralCorrespondence` (explicit indices/masks/scores/continuity) →
featurizer → OpenFold consumer test, **with derived pseudo_beta/torsions in the
same slice** (they expose discontinuity bugs). Retrieval DB construction,
metadata, chain/domain selection, date/leakage filtering, query provenance are
SEPARATE substantial projects — not hidden under T3/T4.
