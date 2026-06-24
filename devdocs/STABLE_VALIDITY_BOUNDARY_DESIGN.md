# Stable-tier correctness & validity-boundary suite (readiness item #2)

> Status: **design** (pre-claudex). Follows the stable/experimental tiering
> (#206). Codex's deepest catch on the readiness review: *robustness ≠
> correctness* — "47k structures, zero crashes" proves the pipeline doesn't fall
> over, not that it returns the right scientific object on ugly-but-common
> inputs. And the most dangerous blind spot: *a library that emits plausible
> numbers for unsupported chemistry is more dangerous than one that fails
> loudly.* This item makes the **stable tier** (the 81 symbols we now promise)
> either correct on hard cases or explicit about its validity boundary.

## Grounding: an empirical probe already found real gaps

Before designing, I probed the stable functions on synthesized pathological
inputs (empty, single CA, HETATM-only water, colinear CAs, no-hydrogen residue,
degenerate/NaN coords). Findings:

**Fails loud / correct (no change needed):**
- `load(empty)` → raises `OSError(BreakingError)`.
- `tm_align` on <3 residues / no-CA / empty → raises `RuntimeError` / `ValueError`.
- `dssp` / `backbone_dihedrals` on unassignable input → empty string / empty array.
- `kabsch_superpose(mismatched shapes)` → raises `ValueError`.

**Plausible-but-wrong (the dangerous gaps):**
1. **`compute_energy` returns `total=0.0` for inputs it could not parameterize.**
   A single CA atom and a water-only input both return `total=0.0`. The dict
   *does* carry `n_topo_atoms` (0 for the dropped water) and
   `n_unassigned_atoms`, but `total=0.0` sits at the same level as a real energy
   and reads as "perfectly relaxed." A caller doing
   `compute_energy(s)["total"]` in a batch gets `0.0` with **no signal** that
   nothing was computed. This is the worst case — energy 0 is actively
   misleading.
2. **`total_sasa` returns a number for HETATM-only / single-atom input** (120.8
   Å² for one CA, 107.1 for a lone water) with no signal that the input isn't a
   meaningful protein surface.
3. **`kabsch_superpose` propagates NaN silently** (NaN in → NaN out) and returns
   `0.0` for rank-deficient (single-point / colinear) inputs with no
   degeneracy signal.

The probe is the seed of the test corpus; these three gaps are the first fixes.

## Scope

**Stable tier only.** We just promised these 81 symbols (#206); proving their
validity boundaries is the natural completion of that promise. The experimental
tier (prepare/supervision/search/…) is explicitly out of scope here — its
contract is "may change," so a frozen correctness suite would be premature.

Two deliverables:
1. A **per-function validity-boundary contract** (documented), and
2. a **boundary-assertion test suite** over a curated hard-case fixture corpus
   that asserts each stable function meets its contract: on out-of-domain input
   it must EITHER fail loud OR return a value carrying an explicit degraded/empty
   signal — **never a silent plausible-but-wrong number.**

## The validity-boundary contract (the rule)

For every stable function, define three things:
- **Valid domain** — what inputs it is correct for.
- **Out-of-domain behavior** — exactly one of: (a) raise a specific, documented
  exception; (b) return a value with an explicit emptiness/degraded signal
  (empty collection, documented sentinel, or a validity field). NaN is allowed
  ONLY if documented as the degraded signal and the function is NaN-in/NaN-out
  by contract.
- **The signal** — how a caller detects degraded output without out-of-band
  knowledge.

The test suite encodes this as: for each (function, pathology) pair, assert the
observed behavior matches the declared contract.

## Hard-case taxonomy (the fixture axes)

Curated, small, checked-in fixtures (synthesized PDB text where possible, real
mini-structures where a real feature is needed). Axes:

- **Size degeneracy**: empty, single atom, two atoms, below-method-minimum.
- **Composition**: HETATM-only (water/ligand), metals, no-protein-chain,
  nonstandard/modified residue (MSE), missing backbone atoms.
- **Geometry degeneracy**: colinear CAs, coincident atoms, NaN/Inf coords.
- **Chain topology**: chain break (missing residues), insertion codes, altlocs,
  multi-model.
- **Chemistry completeness**: no-hydrogen structure fed to a forcefield.

Not every axis applies to every function; the suite is the (function × relevant
pathology) matrix, with the irrelevant cells omitted (and that omission logged,
not silent).

## The three fixes the probe already justifies

1. **`compute_energy` validity signal.** Non-breaking: the returned dict gains a
   boolean (e.g. `"is_parameterized"` / `"valid"`) that is false when
   `n_topo_atoms == 0` or `n_unassigned_atoms > 0`, and the docstring documents
   that `total` is only meaningful when it is true. **Open question:** is a dict
   field enough, or should `compute_energy` *warn* (or raise) when
   `n_topo_atoms == 0` (genuinely nothing computed, as opposed to a partial
   typing)? Leaning: warn on `n_topo_atoms == 0`, field for partial typing, never
   silently return a bare 0.0 as if real.
2. **`total_sasa`** — document that it computes the SASA of whatever atoms it is
   given (it is atom-set-correct, not protein-validated); add the
   no-atoms/HETATM-only cases to the boundary tests so the documented behavior is
   pinned. (Likely doc + test, not a behavior change — SASA of arbitrary atoms is
   well-defined.)
3. **`kabsch_superpose`** — document NaN-in/NaN-out and the rank-deficiency
   behavior; decide whether NaN input should raise (leaning: raise on
   non-finite input, since silent NaN propagation is a foot-gun) and whether
   rank-deficient superposition should be flagged.

## Test plan

- `tests/test_stable_validity_boundaries.py`: the (function × pathology) matrix,
  each cell asserting the declared contract (raise-with-type OR explicit
  signal). The three fixes above get direct regression tests.
- `tests/data/adversarial/`: the curated fixtures (small PDB texts).
- A meta-test: every name in `proteon.__stable__` that takes a structure/coords
  is covered by at least one boundary test, or is on an explicit, commented
  exemption list (so coverage gaps are visible, not silent — mirrors the #206
  snapshot-guard ethos).
- Oracle cross-checks where a hard case has a known-correct answer (e.g. a
  structure with MSE: `tm_align` must treat MSE as MET, matching USalign; a
  chain-break structure: dihedral at the break must be the documented sentinel,
  not a bogus angle across the gap).

## Scope boundaries / non-goals

- Not the experimental tier.
- Not a full re-derivation of every algorithm's correctness — oracle parity
  already covers the happy path; this is specifically the *boundary*.
- Not exhaustive fuzzing — curated, named pathologies with expected behavior,
  not random input generation (that can come later).

## Decisions (post-claudex, 2026-06-24)

- **Narrow the invariant** (Codex): "out-of-domain input must fail loud or carry
  an explicit validity signal" applies to functions **whose output can be
  mistaken for a meaningful scientific result** — not to low-level numeric
  utilities where NaN-in/NaN-out or empty output is the conventional signal.
- **`compute_energy`: RAISE on `n_topo_atoms == 0`** with a new
  `ParameterizationError(ValueError)` (subclasses `ValueError`, so existing
  `except ValueError` still catches — non-breaking for catchers). `n_topo_atoms
  == 0` means no atom entered the force-field model, so `total` is nonexistent,
  not merely degraded; returning `0.0` is a correctness bug. For `n_topo_atoms >
  0 and n_unassigned_atoms > 0` add `parameterization_status="partial"` /
  `is_parameterized=False` (additive dict keys). A frozen API does not mean
  preserving misleading behaviour. **Note the precise boundary:** a single CA
  atom has `n_topo_atoms == 1` and `total == 0.0` is *correct* (an isolated atom
  has zero internal energy) — only the genuinely-empty topology raises.
  `ParameterizationError` is a deliberate new STABLE symbol (part of the stable
  `compute_energy` contract) → added to `_FORCEFIELD_API` and the frozen
  snapshot (exercises the #206 guard as intended).
- **`total_sasa`: docs + pinning only** — SASA of arbitrary atoms (water, single
  atom) is well-defined; do NOT make it validate "proteinness." Document it as
  atom-set SASA; pin the HETATM-only/single-atom behaviour in tests.
- **`kabsch_superpose`: docs + pinning, no raise** — it's a low-level utility;
  NaN-in/NaN-out stays (documented). Add an *identifiability* note (rank-
  deficient input → finite RMSD but underdetermined rotation). Add correctness
  oracle tests (identity + known-rigid-transform recovery).
- **`dssp`: document the coarse `""` signal** (conflates empty / too-short /
  no-assignable-SS); pin current behaviour; flag the limitation. No behaviour
  change to the frozen `str` return now.
- **Scope: stable-only, curated fixtures, NOT fuzzing** — right-sized; the one
  required addition is a **meta-coverage test** against `proteon.__stable__` so
  "curated" can't silently become "whatever we remembered." Do NOT test every
  pathology × every function; require each stable structure/coords-facing
  function to have ≥1 boundary test or a commented exemption.
- **First test (highest value):** `compute_energy(water-only)` raises
  `ParameterizationError`, does not return `total == 0.0`.

## Open questions for Codex (answered above)

1. Is "fail loud OR explicit signal, never silent plausible-wrong" the right
   single invariant, or are there stable functions where a silent best-effort
   value is genuinely the right contract?
2. For `compute_energy`: dict validity-field vs warning vs raise on
   `n_topo_atoms == 0` — which is the right call for a *stable* (frozen) API,
   given a raise would be the loudest but is the most breaking?
3. Is the curated-fixture matrix the right scope, or is property-based/fuzz
   testing necessary to claim "correctness" rather than "we checked the cases we
   thought of"?
4. Which hard cases carry a *known-correct oracle answer* (so we test
   correctness, not just non-silence)? MSE→MET, chain-break dihedral sentinel,
   altloc selection — what else has an unambiguous right answer?
5. Anything in the "fails loud / correct" list that is actually wrong — e.g. is
   `dssp` returning `""` for a 1-residue input right, or should it distinguish
   "no SS" from "input too small to assign"?
