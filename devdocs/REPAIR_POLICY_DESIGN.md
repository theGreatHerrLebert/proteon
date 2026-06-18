# RepairPolicy — the remediation layer

## Motivation

P1 gives a **gate**: `label_safe` + `label_hazards` say whether a prepared
structure is safe to use as a training label and why not. But a gate only
filters. In practice a user wants to decide, **per hazard, case by case, in
code**, what to do with the structures that didn't pass: fix the ones that are
fixable, accept the ones whose hazard doesn't matter for their label type, and
drop the rest — then re-verify. This closes the loop:

> detect → decide (rules) → repair → re-verify → report.

`RepairPolicy` is that decision layer. It does NOT invent new chemistry; it
orchestrates capabilities proteon already has (reconstruct, heavy-relax
minimize, conformer/model selection) under a declarative per-hazard policy.

## The action model

For each hazard, the policy assigns one action:

- **`FIX`** — actively repair, then re-verify. Only meaningful for hazards that
  proteon can actually change:
  - `missing_atoms` → `reconstruct=True` (fill from templates). TRADE-OFF: the
    filled atoms become `reconstructed_atoms` (model-derived) — a FIX that
    trades one hazard for a (provenance) one. Re-verify makes this explicit.
  - `heavy_clashes` → heavy-relax minimize (`constrain_heavy=False`). TRADE-OFF:
    lossy — moves the backbone ~0.5 Å off the deposited coordinates and sets
    `heavy_relaxed=True`. Resolves *some* clashes, not guaranteed all.
- **`ACCEPT`** — keep the structure despite the hazard; it doesn't matter for
  this consumer's label type. (e.g. accept `untyped_cofactors` for backbone
  coordinate labels.)
- **`DROP`** — exclude the structure.

A structure **passes the policy** iff, after FIXes, every remaining hazard is
`ACCEPT`-ed (none maps to `DROP` and none is unhandled).

## What is actually fixable (be honest)

Sorting the P1 hazards by what proteon can do:

| Hazard | Fixable? | How / why not |
|--------|----------|---------------|
| `missing_atoms` | **FIX** (lossy→provenance) | `reconstruct=True`; becomes `reconstructed_atoms` |
| `heavy_clashes` | **FIX** (lossy→coords) | heavy-relax minimize; moves backbone |
| `altlocs` | accept-selection | prepare already uses the primary conformer; "fix" = accept that choice (a true highest-occupancy chooser is a future primitive) |
| `multiple_models` | accept-selection | prepare already uses model 0; "fix" = accept model 0 |
| `reconstructed_atoms` | not fixable | it IS the result of a fix; ACCEPT or DROP |
| `untyped_atoms` / cofactors | not fixable | missing FF chemistry; ACCEPT or DROP |
| `insertion_codes` | not fixable (safely) | renumbering changes residue identity; ACCEPT or DROP |
| `numerical_failure` | not fixable here | the geometry blew up; DROP |

So the only **coordinate-altering** fixes are `reconstruct` and `relax`.
`altlocs` / `multiple_models` are *accept-selections* (the choice already
happened in prepare; the policy decides whether to accept it). Everything else
is accept-or-drop. This honesty matters: the layer should not pretend to "fix"
chemistry it can't.

## Proposed API

```python
policy = proteon.RepairPolicy(
    missing_atoms        = "reconstruct",     # FIX
    heavy_clashes        = "relax",           # FIX (lossy)
    altlocs              = "accept",          # accept primary conformer
    multiple_models      = "accept",          # accept model 0
    untyped_atoms        = "accept",          # fine for coordinate labels
    nonstandard_residues = "drop",
    insertion_codes      = "drop",
    default              = "drop",            # any unlisted hazard → drop
)

for res in proteon.prepare_for_supervision(paths, repair=policy):
    if res.passes_policy:
        use(res.structure)           # fixed and/or accepted
    else:
        log(res.path, res.dropped_for, res.actions_taken)
```

Result fields on `LoadPrepResult` (or a `RepairResult`):
- `passes_policy: bool` — safe under the policy after repair.
- `actions_taken: list[str]` — e.g. `["reconstruct(missing_atoms)", "relax(heavy_clashes)"]`.
- `remaining_hazards: list[str]` — hazards still present after fixes.
- `accepted_hazards` / `dropped_for: list[str]` — which remaining hazards were
  accepted vs which triggered the drop.

Convenience presets: `RepairPolicy.strict()` (drop everything), `.coords_only()`
(accept typing/cofactor hazards, fix missing/clashes, drop identity hazards),
`.permissive()` (fix what's fixable, accept the rest).

## The repair loop (semantics)

1. **Order.** Accept-selections (model 0, primary conformer) already happened in
   prepare. The FIX actions are prepare-time options, so a FIX means
   **re-prepare** the structure with the fix flags turned on (`reconstruct=True`
   and/or `constrain_heavy=False`).
2. **Single combined re-prepare**, not iterative: collect all FIX flags implied
   by the active hazards, re-run `prepare` once with them, then re-detect. (A
   clash-relax + reconstruct is one prepare call.)
3. **Re-verify.** Recompute `label_hazards` on the repaired structure. A fix can
   introduce a new hazard (reconstruct → `reconstructed_atoms`); that new hazard
   is then subject to the SAME policy (so a `reconstruct` user typically sets
   `reconstructed_atoms = "accept"`, or the `reconstruct` action implies it).
4. **Verdict.** `passes_policy` = every remaining hazard is ACCEPT-ed.

## Claudex revisions (adopted — these change the API shape)

1. **Profile-targeted (the key correction).** A policy targets a label PROFILE,
   not the strict all-types gate. Hazards only matter relative to the intended
   label, so `accept` is scoped to what that label needs and can't leak into a
   context where it's invalid.
   ```python
   policy = proteon.RepairPolicy.for_profile(
       "heavy_coords",
       missing_atoms="reconstruct",
       reconstructed_atoms="accept",     # explicit (Q1)
       heavy_clashes="drop",             # NOT relax by default (Q3)
       altlocs="accept_selected",
       multiple_models="accept_selected",
   )
   # verdict is computed RELATIVE to the profile:
   res.passes_policy   # == "safe for heavy_coords under this policy"
   ```
2. **Clash-relax is loud / opt-in, never a silent fix (highest-risk).** It moves
   the deposited coordinates off the experiment, so the "fixed" label is no
   longer purely observed. If chosen, it sets a distinct provenance flag and
   records a drift metric (CA-RMSD vs input); the default for `heavy_clashes` is
   `drop`, not `relax`. A consumer must opt into relaxed coordinates explicitly.
3. **Explicit provenance acceptance.** A FIX does NOT implicitly accept the
   hazard it creates. `reconstruct` requires `reconstructed_atoms="accept"`;
   `relax` requires accepting the relaxed-coords provenance. The trade-off is
   never hidden.
4. **altloc / multiple_models are `accept_selected`, not FIX.** The selection
   already happened in prepare (primary conformer, model 0); the policy accepts
   that lossy choice. Real selectors (highest-occupancy, specific model) are a
   later primitive, not the first cut.
5. **Single-pass repair** (collect FIX flags → one re-prepare → re-verify). No
   iterate-to-fixpoint in the first cut.
6. **Result provenance is more precise than the input action**, classified by
   semantic class: selection-acceptance / restorative (reconstruct) /
   coordinate-physicalization (relax) / impossible (untyped, insertion,
   numerical). The user vocabulary stays simple (fix/accept/drop) but the result
   records what really happened.
7. **Corpus summary.** `RepairSummary.from_results(results)` →
   `by_hazard` / `by_action` / `fixed_count` / `accepted_count` /
   `dropped_count` / `provenance_counts`. The real triage decision is
   corpus-level.
8. **Declarative dict is the primitive; callback is an escape hatch**
   (`RepairPolicy.from_callback(fn, name=...)`) whose resolved per-structure
   decisions are recorded (not just "callback accepted") for replayability.

### First-cut scope (safest)

profile-targeted policy + explicit provenance acceptance + single-pass repair +
`reconstruct` FIX + `accept`/`accept_selected`/`drop` + corpus summary.
**`relax` is loud-and-opt-in with a drift metric.** Real altloc/model selectors
and the callback escape hatch are follow-ons.

## Original open questions (answered by claudex above)

1. Does a FIX implicitly ACCEPT the provenance hazard it creates (reconstruct ⇒
   auto-accept `reconstructed_atoms`; relax ⇒ auto-accept the moved-coords
   signal), or must the user accept it explicitly? Implicit is ergonomic but
   hides the trade-off; explicit is honest but verbose.
2. Single-pass re-prepare vs iterate-to-fixpoint — can a fix introduce a hazard
   that another active FIX would resolve, needing a second pass? Is single-pass
   ever wrong?
3. Is "relax heavy_clashes" a defensible FIX for label data at all — it moves
   the deposited coordinates, so the "fixed" label is no longer purely
   experimental. Should clash-relax be opt-in-loud (a distinct provenance flag
   like `coords_relaxed_for_clashes`) rather than a silent FIX?
4. Altloc/model as accept-selections: is "accept model 0 / primary conformer"
   the right default, or should the layer expose real selectors (highest
   occupancy, specific model) — and is that in scope for the first cut?
5. Should the policy operate per-structure only, or also support a corpus-level
   summary (how many fixed / accepted / dropped, by hazard) for triage?
6. Is `RepairPolicy` the right shape, or is a callback (`repair_fn(report) ->
   action_map`) more flexible for genuinely case-by-case logic?
7. Interaction with the label PROFILES: should the policy target a specific
   profile (e.g. "make it `label_safe_heavy_coords`") rather than the strict
   all-types `label_safe`, so accepts are scoped to what the label needs?
