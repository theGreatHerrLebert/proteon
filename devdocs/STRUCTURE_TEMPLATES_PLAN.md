# Structure-based template features (design)

Plan for **structure-based** AlphaFold template features in proteon — using
proteon's own TM-align for the query↔template residue correspondence, instead of
AF's sequence/profile (HHsearch) path. AF templates are limited by *sequence*
homology detection; proteon can template off *structural* similarity, reaching the
remote-homolog regime sequence search misses. (Codex-reviewed; see
`STRUCTURE_TEMPLATES_PLAN.codex-review.md`.)

## 0. Scope & workflow — read first

**This is not de-novo AlphaFold templating.** Structural retrieval needs the query
*structure*, which the model is supposed to predict. So the intended workflows are:
experimentally-known query + structural templates (annotation/feature studies),
**refinement**, or **iterative** retrieval from a preliminary prediction. These are
materially different products from the standard sequence-templated AF path, and
iterative retrieval is a **leakage** risk (the query's own deposition must be
excludable). The deliverable is stated in those terms, not as a drop-in AF
template source.

**This deliverable = the featurizer**, given a query structure + a set of candidate
template structures (atom37 + sequence). Building/curating a template database,
retrieval-at-scale, date/leakage filtering, chain/domain selection, and query
provenance are **separate substantial projects**, explicitly out of scope here
(§6) — not buried in a later phase.

## 1. Where this slots in

`packages/proteon/src/proteon/templates.py` ships **sequence-based** v0:
`TemplateFeatures` (`template_aatype (T,L)`, `template_all_atom_positions (T,L,37,3)`,
`template_all_atom_masks (T,L,37)`, `template_sum_probs (T,)`). Those 4 are AF's
**on-disk** template set; `template_pseudo_beta` / `template_torsion_angles_sin_cos`
are derived (OpenFold `make_pseudo_beta` / `atom37_to_torsion_angles`, template
prefix) — both transforms now live in `supervision_geometry`, verified vs OpenFold.

Reuse the feature schema + top-K/exclusion machinery. **Replace** the engine
(structural candidates) and, critically, the placement: a *structural*
correspondence, not a sequence CIGAR.

## 2. The pipeline

```
query structure ──▶ (candidate template set, provided) ──▶ for each candidate:
   TM-align(query, template) ─▶ AlignResult (aligned_seq_x/_y, tm_score, n_aligned)
        │
        ▼   build an explicit StructuralCorrespondence (NOT inferred from letters)
   gather template atom37[j] onto query row i for each aligned pair (i, j)
        │
        ▼   derive template pseudo_beta + torsions (continuity-masked)
   TemplateFeatures (T, L, …)  + per-template metadata (tm_score, coverage)
```

### 2a. `StructuralCorrespondence` — the load-bearing object (codex catch #2)
**Do not infer atom37 row indices by walking the alignment letters.** proteon's
`extract_for_alignment()` drops residues lacking CA (and concatenates chains when
none is selected), so the aligner's residue list ≠ the atom37 residue list. T1
first builds an explicit object carrying **original residue indices**:

```
StructuralCorrespondence:
  query_idx[k]      # atom37 row in the query   for aligned pair k
  template_idx[k]   # atom37 row in the template for aligned pair k
  n_aligned, tm_score, query_len, template_len
```

It is produced by pairing TM-align's aligned columns back to the *original* atom37
indices (the aligner must expose, or the caller must thread, an
`alignment_index → atom37_index` map per side). Contract checks: aligned strings
equal length; ungapped lengths == the exact alignment inputs; `len(pairs) ==
n_aligned`; `query_idx` and `template_idx` each strictly monotonic + unique.

### 2b. Featurization
For template `t`: `positions[t]=zeros(L,37,3)`, `masks[t]=zeros(L,37)`,
`aatype[t]=<gap/unk token>`. For each pair `(i,j)` in the correspondence:
`positions[t,i]=template.atom37_pos[j]`, `masks[t,i]=template.atom37_mask[j]`
(per-atom — an aligned residue with missing atoms keeps a partial mask),
`aatype[t,i]=aa(template residue j)` — the **template's** identity, AF-encoded
(22-class incl. unknown + gap). Coords stay in the **template's own atom37 frame**
(AF convention; the template stack derives intra-template distances/orientations —
no superposition; `AlignResult.rotation/translation` are *not* applied).

### 2c. Derived geometry — in T1, not later (codex catch #4)
`template_pseudo_beta` / `template_torsion_angles_sin_cos` via
`supervision_geometry`, computed from the gathered template atom37. **Continuity**:
query-adjacent rows can map to *nonconsecutive template* residues (template
insertions) — a naive torsion call forges a peptide bond. Thread `template_idx`
as the `residue_index` continuity arg (already supported by
`compute_torsion_angles_sin_cos`) so pre_omega/phi mask on a template break. This
must be in the first slice — it's where discontinuity bugs surface.

### 2d. `template_sum_probs` — raw, not calibrated (codex catch #6)
Store a **documented raw TM-score** (query-length normalized), optionally × aligned
query coverage, and expose the components (tm_score, n_aligned, coverage)
separately. **Not** a per-result-set max-normalization (a hit's value must not
depend on its competitors). A learned TM→probability calibration needs a downstream
outcome dataset — later research, not this feature.

## 3. Validation (no external bit-oracle for the structural path)

AF's template pipeline is sequence-driven (HHsearch + PDB), so there is no
bit-parity oracle. Gates (codex catch #7 — beyond self-referential):

1. **Correspondence contract** — the §2a checks; per-atom masks correct.
2. **Self-template identity** — template a structure against itself → identity
   correspondence: `positions[0] ==` query atom37, full mask, `tm_score ≈ 1`,
   `aatype == query aatype`. (Necessary, not sufficient.)
3. **Hand-authored correspondence cases** — terminal offsets, insertions/deletions,
   missing-CA residues, unknown residues, chain breaks, multichain rejection,
   repeated-sequence ambiguity. Each pins the explicit index map + masks.
4. **Rigid-transform metamorphic** — rotating/translating the *template* leaves the
   features invariant up to that rigid transform of the stored coords (intra-template
   geometry unchanged), confirming frame handling.
5. **Differential vs OpenFold given the same explicit mapping** — feed an identical
   `(query, template, mapping)` to a transcribed-OpenFold template featurization and
   match the 4 tensors + derived pseudo_beta/torsion exactly. The geometry
   transforms are already OpenFold-gated; this gates the *gather*.
6. **OpenFold consumer smoke test** — the produced features (incl. zero-template and
   partially-masked) survive an actual OpenFold preprocessing/model-input pass.
7. **Close-pair agreement (exact)** — on a high-identity pair the structural
   correspondence equals the sequence-based one **exactly**, except documented
   alternative-alignment columns.
8. **Remote-homolog reach (measured)** — a *curated* fold-superfamily benchmark:
   retrieval recall, TM/coverage distributions, failure cases — not one anecdote.
9. **Determinism** — fixed candidate order + tie-breaks.

## 4. Open questions — dispositions (codex)

- **Frame:** template-native. **Decided.**
- **`template_aatype`:** the template's identity (22-class, unk+gap). **Decided.**
- **Multichain:** single-chain v0; require explicit chain selection; reject
  accidental multichain inputs. **Decided.**
- **Retrieval scope:** featurizer-only; retrieval is **not** delivered here.
  **Decided.**
- **`sum_probs`:** raw documented TM-score for v0; calibration is later research.
  **Decided.**
- **Benchmark corpus** for §3.8 remote-homolog reach — still open (curated set +
  recall/coverage metrics; use the existing one-to-many TM-align API with candidate
  caps).

## 5. Phasing

- **T1 — correspondence + featurizer + derived geometry + consumer test.**
  `StructuralCorrespondence` (explicit indices/masks/scores/continuity) →
  `build_structure_template_features(query, candidates, template_store, *, top_k)`
  → template pseudo_beta/torsions (continuity-masked) → §3.1–3.7 + §3.9 + the
  OpenFold consumer smoke test. This is the bounded deliverable.
- **T2 — retrieval adapter.** Wire proteon's structural search as the candidate
  engine (capped K via the one-to-many path); the §3.8 remote-homolog benchmark.
- **T3 — release/scale.** Template-store format, **date/leakage exclusion** for
  training splits, batch path, wire into `SequenceExample` / the release manifest.
  (Each of these is itself substantial — sized honestly, not "optional polish".)

## 6. Non-goals (v0)

Building/curating a template DB; HHsearch-style profile templating; multichain
templating; superposed-frame coords; learned `sum_probs` calibration; retrieval-at-
scale; date/leakage filtering and query provenance (T3+, separate projects).
