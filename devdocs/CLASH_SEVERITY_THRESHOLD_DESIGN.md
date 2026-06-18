# Clash-severity threshold for the label-safe gate

## Problem

The strict `label_safe` gate currently rejects **100% of real PDB**. The cause
is the `heavy_clashes` hazard: it is binary (`has_heavy_clashes = n_heavy_clashes
> 0`), so a *single* 0.4 Å heavy-atom overlap flips a structure to unsafe — and
99% of deposited structures have at least one such overlap.

Measured on a 1,000-structure random PDB sample (`validation/pdbs_10k`, 974
loaded):

- strictly `label_safe`: **0 / 974**
- `heavy_clashes` flagged: 965 / 974 (99%)
- even reconstruct + altloc/model selection + **heavy relaxation** (median CA
  drift 0.65 Å) failed to drive clashes to exactly zero on 38/39 — so the
  drop-on-clash *and* relax-on-clash repair policies both yield 0%.

A single mild clash in a 300-residue protein is a local artifact, not a poisoned
coordinate label. The all-or-nothing gate is the wrong abstraction for this one
hazard. Every other hazard (missing atoms, altlocs, models, insertion codes,
non-standard residues, metals, chain gaps, chirality, assembly) is genuinely
binary; clashes are a matter of *degree*.

## Proposal

Introduce a **clashscore** severity metric and gate on it instead of on "any
clash". Keep the raw count fully visible — the change is to what *blocks* a
label, not to what is *reported*.

### Metric (two-part, per claudex)

```
clashscore           = 1000 * n_heavy_clashes / n_heavy_atoms   # pervasiveness
max_heavy_overlap    = max heavy-atom overlap depth (Å)         # worst local defect
```

`clashscore` is the heavy-atom serious-overlap (≥0.4 Å) density per 1000 heavy
atoms — the MolProbity clashscore convention, restricted to heavy atoms
(proteon's clash detector is heavy-only, because placed-H positions are
model-dependent). It normalizes by size so a large complex isn't penalized for
having proportionally more atoms.

But density alone **dilutes a catastrophic local defect**: one 2 Å atom-on-atom
interpenetration in a 5,000-atom complex barely moves the clashscore yet is a
toxic coordinate label. So the gate also caps the single worst overlap. The
clash scan already computes per-pair overlap to threshold at 0.4 Å — tracking
the maximum is free.

### Threshold: clashscore ≤ 20 (default)

Calibrated against deposited resolution on the 974-structure sample
(clashscore by resolution band):

(clashscore via the templated-atom denominator — the same one the gate uses):

| Resolution | n | median | p90 | frac ≤10 | frac ≤20 |
|---|---|---|---|---|---|
| ≤1.5 Å | 80 | 7.8 | 11.4 | 74% | **100%** |
| 1.5–2.0 Å | 295 | 9.7 | 17.1 | 54% | 93% |
| 2.0–2.5 Å | 248 | 12.3 | 22.5 | 34% | 84% |
| 2.5–3.0 Å | 157 | 16.8 | 38.5 | 17% | 59% |
| ≥3.0 Å | 142 | 18.9 | 43.8 | 17% | 54% |

`clashscore ≤ 20` passes **100%** of ≤1.5 Å structures and **93%** of 1.5–2.0 Å
structures — essentially all the well-resolved deposits we trust — while
progressively rejecting the low-resolution tail where clashes signal genuinely
unreliable coordinates. A stricter `≤ 10` would reject ~46% of good 1.8 Å
structures (false negatives), so 20 is the elbow. The high-res population's p90
is 11.4, leaving a comfortable margin below the gate.

Clash-only yield across the whole sample (upper bound before other hazards):

| clashscore ≤ | 0 (today) | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| passes (of 974) | 9 | 24 | 96 | 372 | **761** | 912 |

### Report changes (`PrepReport`)

- **New** `clashscore: float` — always reported (0.0 when clash-free).
- **New** `max_heavy_overlap: float` — worst single overlap depth in Å (0.0 when
  clash-free). Emitted from the Rust clash scan.
- **New** `n_heavy_atoms: int` — needed to compute clashscore; cheap to emit
  from the Rust clash scan, which already iterates heavy atoms.
- `n_heavy_clashes` — unchanged, still the raw count.
- `has_heavy_clashes` — unchanged: `n_heavy_clashes > 0` (honest "any clash").
  Stays an **observation**, no longer a label hazard on its own.
- **New** `has_severe_clashes: bool` —
  `clashscore > CLASH_SCORE_THRESHOLD or max_heavy_overlap > MAX_OVERLAP_DEPTH`.
- The gate properties switch their clash term from `not has_heavy_clashes` to
  `not has_severe_clashes`:
  - `label_safe_heavy_coords`, `label_safe_all_atom_coords`, `label_safe_energy`,
    `label_safe` (the strict union).
  - `label_safe_sequence_indexed` never gated on clashes — unchanged.

### Hazard rename (per claudex — avoid a silent semantic break)

The blocking hazard is renamed so a downstream consumer can never mistake the
new meaning for the old:

- `label_hazards` reports **`"severe_heavy_clashes"`** when `has_severe_clashes`.
- `"heavy_clashes"` is **observation-only** — never in `label_hazards`; surfaced
  through `has_heavy_clashes` / `n_heavy_clashes` / `clashscore`. A user who
  keyed on `"heavy_clashes" in label_hazards` gets a clean rename (KeyError-free,
  the name simply isn't a hazard) rather than a quietly-changed truth value.

This is the documented split: `has_heavy_clashes == True` with `label_safe ==
True` is **expected** for a mildly-clashing high-resolution structure — the docs
must call this out explicitly.

### RepairPolicy implications

- The policy hazard key becomes **`"severe_heavy_clashes"`** (was
  `"heavy_clashes"`). `PROFILE_BLOCKERS`, `KNOWN_HAZARDS`, `_ACTION_HAZARDS`,
  and the profile↔property consistency test all rename in lockstep. A policy
  rule on the old `"heavy_clashes"` name now raises "unknown hazard" rather than
  silently changing behavior (the intended, loud migration).
- `relax` still attempts heavy minimization to pull a severe structure below
  threshold; success = post-relax `not has_severe_clashes` (not exactly 0).

### Configurability

`CLASH_SCORE_THRESHOLD = 20.0` and `MAX_OVERLAP_DEPTH = 1.0` (Å) are module-level
constants (matching how `PEPTIDE_GAP_MAX` etc. are constants). Optionally
overridable later via a parameter on `prepare_for_supervision` / the profile
call; not in scope for v1 to keep the contract simple.

Guard: `n_heavy_atoms == 0` ⇒ `clashscore = 0.0` (no atoms ⇒ no clashes; such a
degenerate structure fails other gates anyway). Never divide by zero.

## Migration / test impact

- 1crn (0 clashes → clashscore 0): still `label_safe`. ✓
- 4hhb (clashscore ≈ 25 > 20): still unsafe — was unsafe before. ✓
- Existing tests that construct `PrepReport(n_heavy_clashes=N)` and assert
  unsafe must set `n_heavy_atoms` so clashscore exceeds threshold (or use a new
  helper). Tests asserting "any clash ⇒ unsafe" change to "severe clash ⇒
  unsafe"; a new test pins a mild clash (clashscore < 20) as `label_safe`.
- `clash_count_inferred` (un-templated-residue exclusion) is unchanged — it
  still governs whether the count is trustworthy at all.

## Re-evaluation plan

After implementing, re-run `validation/eval_prepare_diverse.py` and report the
new strict-`label_safe` yield and the heavy_coords repair yield with the
threshold in place — the headline number that was 0% should now reflect the
~80% clash-only ceiling minus the other (binary) hazards.

## Out of scope (claudex follow-ons)

- **Relax drift cap.** Relaxation that pulls a clash below threshold can move
  atoms away from the deposited (experimental) coordinates; if relaxed coords
  become labels, large drift is itself a corruption. v1 *records* `coords_drift`
  (CA-RMSD) on the repair outcome, so a user can gate on it today; a built-in
  max-drift cap on the `relax` action is the follow-on.
- **Tiered / resolution-aware thresholds.** A strict tier (`≤ 10`) vs the
  permissive default (`≤ 20`), or requiring resolution ≤ 2.5 Å when using 20.
  v1 ships one global pair of constants; the default `20` is calibrated to admit
  trusted high-res deposits, and the depth cap catches catastrophic locals — the
  remaining concern (admitting *mediocre* low-res structures) is a quality knob,
  not a corruption guard, so it is deferred.
- Per-residue / per-chain clash localization for masking (interface dilution in
  large complexes).
