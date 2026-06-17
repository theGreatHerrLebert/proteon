# Minimize defaults: make `report.converged` meaningful at scale

## Problem (measured)

A 1000-PDB run of `batch_load_and_prepare` (default minimize on) plus a
focused 20-protein lever sweep show the default minimization **never reports
`converged`**, so any caller gating on `report.converged` for energy-grade
work rejects everything. And NoCutoff O(N²) makes it ~0.16 struct/s on CPU
(one 300k-atom assembly hung the whole batch at 35 min).

Sweep on 20 proteins (300–1500 atoms), `batch_prepare(minimize=True)`:

| config (constrain_heavy, steps, tol) | converged | ready | wall |
|---|---|---|---|
| **default** (heavy, 500, 0.1) | **0/20** | 20/20 | 149s |
| H-only (frozen, 500, 0.1) | 19/20 | 20/20 | 36s |
| heavy, 2000 steps, 0.1 | 12/20 | 20/20 | 280s |
| heavy, looser (heavy, 500, 1.0) | 19/20 | 20/20 | 112s |
| H-only, looser (frozen, 500, 1.0) | 20/20 | 20/20 | 26s |
| H-only, 2000 steps, 0.1 | 19/20 | 20/20 | 38s |

Reads:

1. **`gradient_tolerance = 0.1` kcal/mol/Å is unreachable for heavy-atom
   relaxation.** Even at 2000 steps, heavy relax hits `line_search_failed`
   (8/20) — the minimizer is at a numerical local minimum it cannot improve,
   with gradient norm still between 0.1 and 1.0. tol 0.1 is below what L-BFGS
   achieves on these systems.
2. **`constrain_heavy` is the dominant lever.** H-only minimization (freeze
   heavy atoms, relax only H onto the fixed framework) converges ~19–20/20 and
   is ~4× faster.
3. **Inconsistency:** the single-structure `prepare()` path already passes
   `constrain_heavy=True` (H-only) — it converges. `batch_prepare` /
   `batch_load_and_prepare` default to `constrain_heavy=None` → FF-aware →
   **heavy relax for CHARMM** → never converges. Same input, different verdict
   depending on which entry point you call.

## Constraint: these are validated defaults

Per CLAUDE.md, heavy-atom relaxation was *deliberately* chosen for
CHARMM19+EEF1 ("polar-H united-atom with inflated carbon radii needs
heavy-atom relaxation for correctly-signed totals"), and the fold-preservation
benchmark (1000 PDBs, median TM=0.9945, 30× faster than OpenMM) ran under the
current defaults. Any change must not silently regress:

- AMBER96 vs OpenMM ≤0.5% energy parity (NoCutoff).
- CHARMM19+EEF1 correctly-signed energy totals.
- Fold preservation median TM 0.9945.

## Options

- **A. Raise default `gradient_tolerance`** (0.1 → ~0.5–1.0). Makes
  `converged` meaningful (heavy relax then converges 19/20). Risk: it is also a
  *stopping* criterion — the structure stops earlier than the 500-step
  max_steps point the fold benchmark used. Energy delta in the flat basin is
  small, but the benchmark must be re-run to confirm TM unchanged.
- **B. Raise default `minimize_steps`** (500 → 2000+). Does **not** fix it:
  heavy relax still only 12/20 (stalls via line_search), and 2× the wall time.
  Rejected by the data.
- **C. Switch batch default to H-only** (`constrain_heavy=True`), matching
  `prepare()`. Converges 19/20, 4× faster, and removes the entry-point
  inconsistency. Risk: changes CHARMM energy semantics (the reason heavy relax
  was chosen) — `report.final_energy` / components would no longer be the
  heavy-relaxed totals. May regress the fold benchmark.
- **D. Default nonbonded cutoff for large structures.** Orthogonal to
  convergence; fixes the O(N²) perf tail so a giant assembly can't hang a
  batch. CLAUDE.md notes an opt-in `CutoffNonPeriodic` exists (GB) and a 15 Å
  cutoff for AMBER. Lowest-risk perf win; does not touch the convergence
  target.

## Audit: the convergence criterion (codex's key question, answered)

`minimize.rs` compares `gradient_tolerance` against **`max_force`** — the
MAX over atoms of the per-atom force magnitude `sqrt(fx²+fy²+fz²)`
(minimize.rs:226-236, 472-480, 672-674). This is a **size-stable
max-component criterion, NOT a total L2 norm over 3N coordinates.** So
codex's worst case (a scaling bug where 0.1 is impossible by construction)
is **ruled out** — 0.1 kcal/mol/Å is a genuine, interpretable per-atom force
target that is simply *tighter than heavy-atom relaxation of a crystal
structure can reach* (a few strained atoms keep the max force in the
0.1–1.0 band).

There is also already an energy-plateau fallback: a plateau counts as
`ConvergedEnergy` only if `max_grad < 10 × tol`, else it is reported as
`LineSearchFailed` (a stall) — so the report ALREADY separates the raw
optimizer status from `converged`, satisfying codex's "don't launder
line_search_failed into convergence" point. We will NOT change that.

Achievable band (from the sweep): with `tol = 1.0`, heavy relax converges
**by gradient** 19/20 within 500 steps → the per-atom force floor is ~1.0.

## Recommendation (refined after audit)

Criterion is size-stable max-component, so per codex: **raise the default
`gradient_tolerance` to the achievable band and keep heavy relaxation.**

1. **Default `gradient_tolerance` 0.1 → 1.0 kcal/mol/Å.** Makes `converged`
   honest for the validated heavy-relax default without changing the
   minimizer, the criterion, or the FF semantics. Structures already reach the
   0.1–1.0 floor and then grind uselessly; stopping at 1.0 leaves coordinates
   in the same flat basin.
2. **Expose `final_max_force` (per-atom) in the report** — auditability, and
   it lets callers set their own stricter threshold. Pure addition.
3. **Validate before landing:** on a sample, compare `tol=0.1` vs `tol=1.0`
   (TM-score vs input = fold preservation, RMSD, total + component energy,
   converged rate, step count). Confirm median TM and energy sign/magnitude
   hold. Only ship if the fold benchmark is unchanged.

Option **C** (H-only batch default) stays held — it changes CHARMM energy
semantics. Option **D** (cutoff for large structures) is a separate perf PR.

### Prior recommendation (pre-audit, superseded)

Combine the two low-risk, physics-preserving moves and validate:

1. **A — raise `gradient_tolerance` default to the achievable local-minimum
   band** (measure the actual achieved gradient distribution to pick the
   number; ~0.5–1.0). This makes `converged` honest without changing the
   minimizer or the trajectory shape, only the stop/label point. Re-run the
   fold-preservation oracle to confirm median TM holds.
2. **D — apply a default nonbonded cutoff above a size threshold** so batch
   prep is tractable on archive-scale corpora.

Hold **C** (H-only batch default) unless A+D prove insufficient — it changes
validated CHARMM energy semantics and the entry-point inconsistency is better
fixed by making `prepare()` and `batch_prepare` *agree* deliberately than by
flipping the batch path's physics.

## Open questions for review

1. Is raising `gradient_tolerance` the right axis, or should `converged` be
   redefined to also accept "line-search stalled at a local minimum" as a
   success (the minimizer genuinely cannot improve)? That would make the
   *current* tol meaningful without changing the stop point.
2. Should `prepare()` and `batch_prepare` be unified to one default
   `constrain_heavy` policy, and which?
3. What size threshold / cutoff distance for D, and does it interact with the
   EEF1 solvation term?
