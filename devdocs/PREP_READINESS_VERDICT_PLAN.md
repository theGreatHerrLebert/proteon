# PREP_READINESS_VERDICT_PLAN — a trustworthy `PrepReport.ready` (RELIABILITY)

Status: DRAFT (for claudex). The load→prepare loop has good per-step guards, but
**reliability is caller-managed**: to know if a prepared structure is safe to use,
a caller must hand-AND `skipped_no_protein` + `minimizer_status` + `n_unassigned_atoms`
+ `minimized` — and get it subtly wrong. This adds a single first-class verdict so
the loop becomes `structure, report = load_and_prepare(p); if report.ready: use(structure)`.

## 1. API (computed properties on the existing `PrepReport` dataclass)
No change to construction sites — derive from existing fields:
- `class PrepStatus(str, Enum)`: `READY`, `NOT_PROTEIN`, `MINIMIZE_FAILED`,
  `INCOMPLETE_FF`.
- `@property status -> PrepStatus` — the verdict.
- `@property ready -> bool` — `status is PrepStatus.READY`.
- `@property reason -> str` — human-readable "" when ready, else why not.
- `__repr__` gains a `ready=…/status=…` line.

## 2. Verdict logic (the crux — intent-respecting)
```
if skipped_no_protein:                         NOT_PROTEIN     # FF can't process (nucleic/ligand-only/exotic)
elif minimizer_status == "numerical_failure":  MINIMIZE_FAILED # geometry blew up (NaN/Inf) — PR-B makes this fire honestly
elif n_unassigned_atoms > UNASSIGNED_TOL:       INCOMPLETE_FF   # chemistry incomplete → energy/topology unreliable
else:                                           READY
```
`ready` = **"a usable structure was produced and nothing FAILED"** — NOT "fully
minimized to a minimum". Rationale for what does NOT block `ready`:
- **`minimizer_status == "not_run"`** — minimize wasn't requested (`minimize=False`),
  or there were no H to move. Intentional / N/A, not a failure → READY.
- **`max_steps`** — ran out of budget; the structure is *improved* over the input,
  just not at a minimum. Usable for analysis → READY (convergence quality is a
  separate axis; the caller inspects `minimizer_status` if it needs a true minimum).
- **`hydrogens_skipped > 0`** — termini / proline / chain-break residues legitimately
  skip the amide H (PR-A). Expected, not a failure.

So `ready` is a **no-hard-failure** gate; `minimizer_status` remains the
finer convergence signal. Documented explicitly so callers needing a converged
minimum don't over-trust `ready`.

## 3. Thresholds
- `UNASSIGNED_TOL`: the report already warns at `n_unassigned_atoms > 10`. The
  not-a-protein heuristic (`skipped_no_protein`) already catches >50% unassigned.
  `INCOMPLETE_FF` covers the middle: a structure that IS mostly protein but has a
  meaningful chunk of unassigned atoms (a few bad residues) → energy/topology
  partially wrong. Proposed `UNASSIGNED_TOL = 10` (align with the existing warning).

## 4. Tests
- Corpus structures (1crn etc.) prepared with defaults → `ready is True`,
  `status is READY`.
- A nucleic/ligand-only or all-`UNK` structure → `NOT_PROTEIN`, `ready is False`.
- `prepare(minimize=False)` on a normal protein → still `READY` (intent respected).
- A report with `minimizer_status="numerical_failure"` (constructed) → `MINIMIZE_FAILED`.
- A report with `n_unassigned_atoms=50, skipped_no_protein=False` → `INCOMPLETE_FF`.
- `reason` non-empty exactly when not ready.

## 5. Non-goals
- Re-deciding convergence — `minimizer_status` already encodes that; `ready` is a
  coarser safe-to-use gate.
- Per-residue verdicts — structure-level only.

## 6. Open questions (for claudex)
1. **`line_search_failed`** — the minimizer stalled (improved but couldn't find a
   further decreasing step). READY (improved, usable) or a soft not-ready? Lean:
   READY (it's not a blow-up; the structure is better than input), but is there a
   case for a `MINIMIZE_STALLED` non-ready status?
2. **`max_steps` as READY** — agree it shouldn't block `ready`, or should there be a
   `converged` sub-signal exposed on the verdict for callers who need a true minimum?
3. **`UNASSIGNED_TOL = 10` absolute** — robust across structure sizes, or should it
   be a fraction of non-water atoms (e.g., >2%)? A 5000-atom structure with 11
   unassigned is fine; a 50-atom one with 11 is not.
4. **Should `ready` require that minimization was even attempted** when the caller
   passed `minimize=True` but it ended `not_run` (e.g. all atoms constrained, or no
   H added under `hydrogens="none"`)? Currently → READY. Is "asked to minimize, got
   not_run" a silent surprise worth a distinct status?
