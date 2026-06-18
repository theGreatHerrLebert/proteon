# Prepare defaults audit — are they sane for step-0 of a DL pipeline?

Concern: `prepare` / `batch_prepare` / `batch_load_and_prepare` are the FIRST
step of complex downstream pipelines (geometric-DL supervision, MD, energy
scoring). A silently-wrong default here is garbage-in/garbage-out: it corrupts
every label and every training example downstream, invisibly. We want to verify
the DEFAULTS AS A SET are sane and that every foot-gun is FLAGGED (not silent).

## The defaults (current, after the prepare-reliability series)

| param | default | GIGO-relevant behaviour |
|---|---|---|
| `reconstruct` | `True` | fills MISSING heavy atoms from templates — these are GUESSES that enter the structure (and any DL label). |
| `hydrogens` | `"all"` | places all H; irrelevant to heavy-atom supervision. |
| `strip_hydrogens` | `True` | drops experimental/deposited H, replaces with placed H. |
| `minimize` | `True` | runs minimization. |
| `minimize_method` | `"lbfgs"` | — |
| `minimize_steps` | `500` | — |
| `gradient_tolerance` | `1.0` | max per-atom force target (kcal/mol/Å); achievable band. |
| `constrain_heavy` | `True` (H-only) | freezes heavy atoms — preserves the deposited coordinates EXACTLY; only H move. |
| `ff` | `"charmm19_eef1"` | validated production FF. |
| `n_threads` | `None` | all cores. |

## The report signals a consumer can gate on

- `ready` / `status` / `reason` — READY, READY_WITH_LIGANDS (usable, untyped
  cofactors), INCOMPLETE_FF (polymer-chain FF gap, hard), NOT_PROTEIN,
  MINIMIZE_FAILED.
- `fully_typed` — every non-water atom got an FF type (energy-grade).
- `converged` — minimizer reached the gradient tolerance.
- `heavy_relaxed` — did heavy atoms move? False under the H-only default ⇒
  `final_energy` is not an equilibrium quantity.
- `atoms_reconstructed` + a warning when rebuilt heavy atoms are left unrelaxed.
- `untyped_cofactors`, `n_unassigned_nonwater`, `warnings[]`.

## Why the SET looks DL-safe (claim to be checked)

1. The default is **H-only**, so the deposited heavy-atom coordinates are
   preserved exactly — DL labels built on backbone/heavy atoms are faithful to
   the experiment, not perturbed by minimization. This is the single most
   important property for supervision data.
2. The two ways prep can silently corrupt a label are FLAGGED:
   - reconstructed (guessed) heavy atoms → `atoms_reconstructed` > 0 + a warning.
   - incomplete FF coverage / non-protein → `ready` / `fully_typed` / `status`.
3. A careful consumer filters on `ready` (or `fully_typed`) and inspects
   `atoms_reconstructed` / `warnings` — and gets clean data.

## Open questions for review (the GIGO lens)

1. Is `reconstruct=True` the right DEFAULT for a structure that may feed DL
   training? It injects template-guessed heavy atoms into the structure (and
   labels). It is flagged, but the safety depends on the consumer checking
   `atoms_reconstructed`/warnings. Should a supervision-oriented path default
   `reconstruct=False` (drop incomplete residues) instead of guessing — or is
   "flag, don't decide" correct here?
2. Is any silent foot-gun UNFLAGGED — a way the default prep can degrade a
   downstream label with NO signal on the report? (e.g. clashes left by H-only;
   altloc/insertion-code handling; multi-model picking model 0; partial sidechain
   H; reconstructed atoms that ALSO clash.)
3. Is `minimize=True` (H-only) a sane default for DL, or does relaxing H (and
   spending the compute) risk anything for supervision that wants raw structures?
   Should DL paths prefer `minimize=False`?
4. Does `ready=True` (which includes READY_WITH_LIGANDS) admit anything a DL
   consumer would consider garbage, or is the tiered verdict the right gate?
5. Are the warnings discoverable enough — they live on `report.warnings` (a
   list). A batch consumer iterating 100k structures must actively read them. Is
   that the right ergonomics, or should "this structure has guessed atoms / is
   not fully typed" be a first-class boolean the consumer is forced to confront?
6. Overall: can a careful DL pipeline, using ONLY the documented report fields,
   reliably avoid garbage-in — and if not, what's the missing signal?
