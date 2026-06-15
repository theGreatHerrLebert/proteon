# MINIMIZE_RELIABILITY_PLAN — close the two silent-correctness traps in load→fix→relax→DL

Status: DRAFT (pre-implementation). Scope is deliberately narrow: the two
**Tier-1 silent-correctness traps** found in the weak-spot audit that sit directly
on the `load → prepare → minimize → supervision` pipeline. NOT the O(N²) scaling
work (OBC GB / MD neighbor lists), NOT SES, NOT the broader reliability roadmap.

## 1. Problem

An end-to-end run on `1crn.pdb` (`prepare(reconstruct=True)` → `minimize_structure`
→ `build_structure_supervision_example`) exposed two failures that **return a
plausible result with no error raised**:

1. **The minimizer no-ops on a clashing structure.** `minimize_structure(s)` ran
   ~30 s but `final_energy == initial_energy` (121994.04 → 121994.04 kJ/mol) and
   `converged == false`. No exception, no warning.
2. **`quality.prep_success` came back `None`**, so a downstream filter keying on it
   cannot distinguish "prep succeeded" from "no info," and at corpus scale cannot
   cull bad examples.

Both are on the exact relax→supervision path used to build training data, so a
silently un-relaxed or unflagged example can reach a model.

## 2. Root cause

**Correction (post-review):** an earlier draft blamed *catastrophic cancellation*
of two ~1.2e5 totals. That is wrong — f64 spacing near 121,994 is ~1.5e-11, so a
genuine ~1e-4 decrease is easily representable. The real mechanism is one or more of
the candidates below, and **step 0 of implementation is an instrumented repro**, not
a code change.

### 2.0 Instrument first (gating step)

Reproduce the `1crn.pdb` no-op with per-iteration logging of: `cur_energy`,
`prev_energy`, `new_energy`, `max_force`, `g·direction`, component-wise energy
deltas, and accepted/rejected trial counts. Only then commit to the acceptance/
direction change. (The ~30 s wall-clock implies the loop ran ≈`max_steps`=1000
full O(N²) evals — i.e. it did *not* break early on step collapse, which already
argues against the "all-rejected → collapse" story and for the direction/plateau
mechanisms below.)

### 2a. Minimizer no-op — candidate mechanisms (ranked)

- **[primary] SD's step is per-atom unit-normalized, not true −∇E.** Each movable
  atom normalizes *its own* force and moves exactly `step_size`, regardless of force
  magnitude (`minimize.rs:269-287`; `scale = step_size / f_mag`, `:283`). So weak-
  force atoms travel as far as clashing atoms and the move direction is **not** the
  global negative gradient. On a heterogeneous structure this can raise the *total*
  energy for every practical `step_size`, so steps are perpetually rejected or make
  negligible progress. Fix: step along the **raw force** with a **global** max-
  displacement cap, and line-search on the real `g·direction`.
- **[latent bug, found in review] Plateau check can falsely report convergence on
  rejected steps.** `check_energy_plateau` is fed `cur_energy` every iteration
  including rejections — but on a rejection `pos` is unchanged, so `cur_energy` is
  identical, and after 5 such iterations SD sets `converged=true` (`minimize.rs:259-266`)
  **despite a large gradient**. So a stalled SD may exit `converged=true` with energy
  unchanged — worse than the `converged=false` the first draft assumed. Plateau
  detection must consume **accepted iterates only** and must not override a
  still-large gradient.
- **[possible] energy-only vs energy+forces path inconsistency.** `prev_energy`/
  `initial_energy` come from `nbc.energy(...)` (`:225`) while `cur_energy` comes from
  `nbc.energy_and_forces(...)` (`:238`); the accept test compares across the two
  paths (`:292`). Any divergence between them makes acceptance compare against the
  wrong baseline. Must be ruled out by a parity test before trusting the optimizer.
- **Mechanics of the no-op exit:** `pos` is written **only on accept** (`:294`); on
  reject `step_size *= 0.5` (`:301`) until `< 1e-8` → `break` (`:302-304`). Combined
  with the per-atom direction, this yields `final_energy ≈ initial_energy` with no
  exception. SD has no Armijo line search; CG/LBFGS have one (see 2b).

### 2b. LBFGS is not a safe drop-in default — it has its own rollback bug

[latent bug, found in review] `lbfgs` (`minimize.rs:~686`) mutates `pos` **before**
testing fallback acceptance and, on break, can leave the **rejected, higher-energy**
coordinates in place — a transactional bug. So Fix A1 (default `sd`→`lbfgs`) is not a
guaranteed fix and must be preceded by making LBFGS coordinate updates transactional
(only commit accepted iterates). LBFGS also shares the same energy/force contract and
the same per-atom 0.01 Å fallback step, so 2a's direction fix benefits it too.

### 2c. No test guards any of this

The only minimize tests (`minimize.rs:936+`, `plateau_tests`) unit-test the plateau
helper; GPU-parity tests compare GPU vs CPU energy but skip without a GPU and never
assert a decrease. There is **no test that minimization lowers energy or that a
stall is reported as a stall** — which is exactly why both bugs shipped.

### 2b. `quality.prep_success` is not a real signal

- `_quality_from_prep_report(None)` returns `None` for the **entire** quality object
  (`packages/proteon/src/proteon/supervision.py:392-394`). The observed run called
  `build_structure_supervision_example(s)` **without** threading a `PrepReport`, so
  `prep_report[i]` was None ⇒ `quality` was None ⇒ `quality.prep_success` is `None`
  (and `quality.<anything>` would `AttributeError`).
- When a report *is* present, `prep_success = not prep_report.skipped_no_protein`
  (`supervision.py:396`). That only reflects the not-a-protein heuristic
  (`prepare.rs:159`), **never minimizer success/convergence**. A structure that
  minimized but did not converge (or no-op'd per 2a) still reports
  `prep_success=True`. The honest convergence info *is* carried separately
  (`converged`, `minimizer_steps`, `minimized` on both `PrepReport`
  (`prepare.rs:66-80`) and `StructureQualityMetadata` (`supervision.py:397-398`)),
  but `prep_success` conflates "is a protein" with "prep succeeded," and the
  None-wholesale path drops all of it.

## 3. Proposed fixes

Order matters: **2.0 instrument → fix the optimizer correctness (A2/A2′) →
transactional safety (A1′) → status plumbing (A3) → default change (A1) → quality
(B)**. Don't flip the default before the optimizer is correct and transactional.

### Fix A — make the minimizer either work or say it didn't

- **A2 (direction). Use the true descent direction with a global step cap.** Replace
  the per-atom unit-step (`minimize.rs:269-287`) with a step along the **raw force**
  (= −gradient), scaled by a single global factor capped so the **maximum** atomic
  displacement ≤ a cap (e.g. 0.1–0.2 Å). This makes SD an actual descent method, so a
  sufficiently small step is guaranteed to decrease energy on a consistent
  energy/force field.
- **A2′ (acceptance). Standard backtracking Armijo on `g·direction`.** Accept iff
  `E_new ≤ E_old + c₁·α·(g·d)` with `g·d < 0` guaranteed by A2; backtrack `α` on
  failure. Drop the first draft's "relative-epsilon" idea (it both rejects legitimate
  late progress and can drift uphill). **Line-search exhaustion ⇒ `LineSearchFailed`
  status, never convergence.**
- **A2″ (plateau). Only consume accepted iterates; never override a large gradient.**
  Fix the latent bug: feed `check_energy_plateau` only after an accepted step, and
  gate any energy-plateau "converged" on `max_force` already being below tolerance
  (otherwise it's `Stalled`, not converged).
- **A1′ (transactional LBFGS).** Make LBFGS commit coordinates **only** on accepted
  line-search steps; on break it must restore the last accepted `pos`, never leave
  rejected higher-energy coords (`minimize.rs:~686`). Apply the same transactional
  discipline to CG.
- **A3 (status enum). Replace the bare `converged: bool` with a status.** Per review,
  use a richer enum, e.g. `MinimizeStatus { ConvergedGradient, ConvergedEnergy,
  MaxSteps, LineSearchFailed, NumericalFailure, NotRun(reason) }`, plus counts of
  accepted steps and function evaluations. Add it to `MinimizeResult` **additively**
  (keep `converged` as a derived `matches!(status, Converged*)` for compat). Surface
  `status`, `accepted_steps`, `n_evals` in every connector dict
  (`py_forcefield.rs:388-396` and the batch / `load_and_minimize` dicts). "Net zero
  coordinate change" is **not** a reliable stall definition — drive the flag off the
  status, not a coordinate diff.
- **A1 (default). Switch `minimize_structure` default `sd`→`lbfgs`** — but **only
  after A1′** — in **both** the Rust/PyO3 binding (`py_forcefield.rs:306`) **and** the
  Python wrapper default (`forcefield.py:318`), which the first draft missed. *(Minor
  behavior change — see Risks.)*

### Fix B — make the quality signal honest (revised per review)

- **B1. Don't encode "unknown" as a measured failure.** Returning `prep_success=False`
  (and default-zero metrics) when no report was threaded turns *absence of data* into
  an apparent observation — rejected. Instead return a `StructureQualityMetadata` with
  an explicit **`report_present: bool=False`** and **tri-state** outcome fields
  (`relax_ok`, `converged` as `Optional[bool]=None`), so `example.quality.<field>`
  never `AttributeError`s **and** consumers can tell "unknown" from "failed." (Keeping
  `quality=None` is the alternative the Optional contract already allows; the tri-state
  object is preferred because it removes the AttributeError footgun.)
- **B2. Keep `prep_success` for compat, add honest derived fields.** `prep_success =
  not skipped_no_protein` is badly named ("is a protein," not "succeeded") — preserve
  it for backward-compat but **document** it, and add `protein_eligible` (its real
  meaning) plus a status-derived **`relax_ok: Optional[bool]`** =
  `minimized && status∈Converged*` (None when unknown). `relax_ok` is only meaningful
  given A2″ (so "converged" can't mean plateau-with-large-gradient). Corpus filters
  switch to `relax_ok`.
- **B3. Propagate `minimized` + status across the Python boundary.** The Python
  `PrepReport` (`prepare.py:95`) lacks the `minimized` field the Rust side carries
  (`prepare.rs:215`); add it (and the new optimizer `status`) to the Python dataclass
  and the connector conversion, else `relax_ok` can't be computed. Then thread the
  `PrepReport` through `build_structure_supervision_example` / batch callers so the
  documented pipeline populates quality; a caller who omits it gets the explicit
  `report_present=False` object (not a crash).

## 4. Test plan (expanded per review)

**Optimizer correctness (the guards whose absence let the bugs ship):**
- **Directional finite-difference check** on the exact clashing fixture: `−g·d > 0`
  and a small step along `d` decreases `E` (verifies A2 direction is genuinely
  descent).
- **Energy-only vs energy+forces parity**: `nbc.energy` total ≡ `nbc.energy_and_forces`
  total at the same coords (rules out 2a's path-inconsistency candidate).
- **Accepted-step monotonicity**: every accepted iterate strictly lowers energy.
- **Rejected-step rollback**: after a rejected line-search trial, `pos` is byte-for-byte
  the last accepted `pos` (guards A1′ transactional fix for SD/CG/LBFGS).
- **Plateau only after accepted steps**: a run that only rejects must **not** report
  `Converged*` while `max_force > tol` (guards the A2″ latent bug).
- **Clashing-fixture energy drop**: `final < initial` strictly for sd/cg/lbfgs after
  A2; assert on **coordinate displacement and status**, not merely final energy.
- **Edge cases**: NaN/Inf coords → `NumericalFailure` (not silent); exact-overlap atoms;
  all-constrained; zero atoms; `max_steps=0` → `NotRun`/`MaxSteps` with no panic.
- **Cross-path**: AMBER96 / CHARMM19 / AMBER96-OBC × direct / NBL / (GPU when present),
  so the fix holds on every energy path, not just the default.

**Quality signal (Python):**
- `_quality_from_prep_report(None)` returns a non-None object with `report_present=False`
  and `relax_ok is None` (B1) — and `example.quality.relax_ok` does not `AttributeError`.
- `relax_ok` is `False` for a stalled/no-op minimize, `True` for a converged one,
  `None` when no report (B2); `protein_eligible` mirrors old `prep_success`.
- `minimized`/`status` survive the Rust→Python `PrepReport` boundary (B3).
- Full prepare→minimize→supervision smoke: `quality` present, `relax_ok` reflects the
  optimizer status.

**End-to-end regression:** re-run the original `1crn.pdb` chain and confirm
`final_energy < initial_energy`, a non-stall status, and `quality.relax_ok is True`.

## 5. Non-goals (explicit)

- O(N²) nonbonded scaling (OBC GB NBL stub, MD neighbor list) — separate Tier-2 work.
- SHAKE/RATTLE non-convergence swallowing — related but separate.
- Any change to the force-field math or the prepare reconstruction logic.
- Changing `minimize_hydrogens` defaults (its `method="sd"` is constrained-H only and
  far less prone to the all-atom clash stall — but A2/A3 hardening still applies since
  it shares `steepest_descent`).

## 6. Risks / backward-compat

- **A1 default change** (`sd`→`lbfgs`) changes results for callers relying on the SD
  default. LBFGS is strictly better here; document in the changelog. Alternative if
  we want zero default-change: keep `sd` default but rely on A2/A3 to make it correct
  or loudly flagged. *Recommend switching the default.*
- **A3 API change** to `MinimizeResult` ripples to every dict builder in
  `py_forcefield.rs` and any Rust caller — keep it additive (new field, existing
  fields unchanged).
- **B1/B2**: returning a quality object where `None` was returned, and adding
  `relax_ok`, are additive; do **not** repurpose `prep_success`.

## 7. Files in scope

- `proteon-core/src/forcefield/minimize.rs` — SD direction (A2) + Armijo (A2′) +
  plateau fix (A2″) + transactional LBFGS/CG (A1′) + `MinimizeStatus` enum (A3); tests
- `proteon-core/src/prepare.rs` — carry optimizer `status`/`minimized` into `PrepReport`
- `proteon-connector/src/py_forcefield.rs` — default `sd`→`lbfgs` (A1); surface
  `status`/`accepted_steps`/`n_evals` in all minimize dicts
- `packages/proteon/src/proteon/forcefield.py` — Python wrapper default (A1)
- `packages/proteon/src/proteon/prepare.py` — add `minimized`/`status` to `PrepReport`
- `packages/proteon/src/proteon/supervision.py` — tri-state quality, `report_present`,
  `protein_eligible`, `relax_ok` (B1/B2/B3)
- Rust minimize test module + `packages/proteon` quality/pipeline tests

## 8. Review log

Reviewed by codex (claudex). Adopted: cancellation diagnosis was wrong (→ instrument-
first, §2.0); the real primary cause is the per-atom unit-step direction (A2); two
latent bugs surfaced — plateau-false-convergence (A2″) and LBFGS non-transactional
rollback (A1′); status enum over bools (A3); quality must not encode unknown as failure
(B1 tri-state); Python `PrepReport` missing `minimized` (B3); standard Armijo over
relative-epsilon (A2′); expanded test matrix (§4). No findings rejected.
