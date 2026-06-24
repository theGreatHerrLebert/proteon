# Stable-surface tiering + API-growth guard (readiness item #1)

> Status: **design** (pre-claudex). First concrete step of the product-readiness
> plan. Goal: stop exposing research frontiers at the same endorsement level as
> the oracle-gated, battle-tested core, and make the *stable* surface boringly
> hard to grow by accident. No new algorithms; this is a contract + CI change.

## Problem (from the readiness review)

`proteon.__init__` exports ~250 symbols across ~35 named API groups, all flat in
one namespace, all in one `__all__`. A user cannot tell load-bearing
(`tm_align`, oracle-gated to 0.003 TM drift; `sasa`, 0.17% vs Biopython) from
scaffolding (`born_energy` from a DRAFT BEM formulation; `dock` from an
incomplete Vina port). `test_public_api_surface.py` checks that `__all__`
resolves and is unique — it does **not** prevent the surface from silently
growing, and there is no stable-vs-experimental distinction at all.

Codex's sharpening: *don't audit the sprawling surface — shrink and freeze a
small one, then make CI block accidental expansion.*

## Design

### 1. Two tiers, one source of truth

Introduce an explicit split of the existing per-group tuples into
`_STABLE_GROUPS` and `_EXPERIMENTAL_GROUPS`. `__all__` stays the union (so
nothing is removed — non-breaking), but two new frozensets are exported:

```python
__stable__       = frozenset(...)   # the supported, oracle-/battle-validated contract
__experimental__ = frozenset(...)   # research frontiers + still-churning orchestration
assert __stable__.isdisjoint(__experimental__)
assert set(__all__) - {"__version__"} == __stable__ | __experimental__
```

Rationale for keeping experimental names **flat** (not removed): removing them
(forcing `proteon.experimental.search`) is a breaking change to working user
code and our own examples/tests. The tier is carried by metadata + docs +
(optional) a one-time warning, not by breakage. **Open question for review:** is
a tier that still lets `proteon.dock(...)` work a real signal, or theatre? (See
"Mechanism strength" below.)

### 2. Tier assignment (the actual classification)

**STABLE** — "compute one well-defined quantity from a structure, oracle- or
battle-validated, API not expected to change":

| Group | Evidence |
|---|---|
| `_IO_API` (load/save PDB+mmCIF, tolerant/rescue loaders) | foundation; 96.4% load rate on 47k |
| `_STRUCTURE_API` (Atom/Chain/Model/Residue/Structure) | core data model |
| `_ARROW_API` (Arrow/Parquet atom export) | stable columnar contract |
| `_ANALYSIS_API` (dihedrals, contact/distance maps, Rg, CA, to_dataframe) | pure geometry |
| `_GEOMETRY_API` (kabsch, rmsd, tm_score, apply_transform, SS assign) | pure geometry |
| `_ALIGN_API` (tm/mm/soi/flex align + results) | 0.003 median TM drift vs USAlign, 4,656 pairs |
| `_SASA_API` | 0.17% median vs Biopython, 1,000 PDBs |
| `_DSSP_API` | oracle-gated |
| `_HBOND_API` | oracle-gated geometry |
| `_HYDROGEN_API` (H placement, fragment reconstruct) | part of prepare |
| `_PREPARE_API` (prepare/batch_prepare/coverage/masking) | 47k zero-crash; **schema-versioned, not behaviour-frozen** |
| `compute_energy`, `minimize_*`, `gpu_available/info`, `batch_compute_energy/minimize` | AMBER96 ≤0.5%, OBC ≤1% vs OpenMM |
| `_CORE_API` (RustWrapperObject) | base class |
| `_SELECT_API` (select) | utility |
| `_SUPERVISION_API` + `_SUPERVISION_EXPORT_API` (per-structure example build + parquet) | the canonical battle-tested workflow; schema-versioned |
| `_FAILURE_API` (failure taxonomy, loader-failure analysis) | diagnostic utility |

**EXPERIMENTAL** — research frontiers or multi-stage orchestration still
churning (most is v0.3.0-era, weeks old):

| Group | Why |
|---|---|
| electrostatics (`born_energy`, `surface_potential`, `read_*`/`write_off`) | `ELECTROSTATICS_FORMULATION.md` is **DRAFT** |
| `_VINA_API` (dock/score/local_only + types + `vina` submodule) | Vina port roadmap incomplete |
| `run_md` | MD (SHAKE/RATTLE) far less validated than `minimize` |
| `_SEARCH_API` | already documented "treat as experimental" |
| `_MSA_API` | depends on search |
| `_TEMPLATE_API` (templates, structure templates, retrieval) | DL frontier, recent |
| `_SEQUENCE_API` (sequence example/MSA features) | DL frontier, recent |
| `_SEQUENCE_EXPORT_API`, `_TRAINING_API` | DL release layers, v0.3.0 |
| `_CORPUS_RELEASE_API`, `_CORPUS_VALIDATION_API` | multi-file release orchestration, v0.3.0, actively changing |
| `_CLUSTER_ASSIGNMENTS_API` | consume-only artifact contract, newest code |

**Borderline calls flagged for the reviewer:**
- **`assembly_builder`** (currently inside `_PREPARE_API`: `build_assembly`,
  `prepare_assembly`, `build_assembly_supervision_examples`, `BuiltAssembly`,
  `PreparedAssembly`) was written **this week**. Grouping says stable; age says
  experimental. Proposal: split it out of `_PREPARE_API` into an experimental
  `_ASSEMBLY_API`. Agree?
- **Corpus release/dataset builders** are called "canonical" in the current
  docstring yet are the most-churning code. Proposal: experimental until the
  schema stops moving. Agree, or does "canonical" imply a stability promise we
  must instead honour?
- **`run_md`** splitting it out of the otherwise-stable forcefield group — is a
  per-symbol exception worth the irregularity, or mark the whole FF group by its
  weakest member?

### 3. Mechanism strength (the consequential choice)

Three options, increasing signal / increasing breakage:

- **(A) Metadata only** — `__stable__`/`__experimental__` + docs + a
  `STABILITY.md`. Zero breakage, weakest signal (a user who doesn't read docs
  sees no difference).
- **(B) Metadata + one-time runtime warning** — first access to an experimental
  symbol emits `ProteonExperimentalWarning` once. Requires making experimental
  names lazy via PEP 562 module `__getattr__` (they can't be eagerly bound, or
  `__getattr__` never fires). Risk: interacts with `from proteon import *` and
  static analysers; more machinery.
- **(C) Sub-namespace + deprecation** — canonical path becomes
  `proteon.experimental.search`; flat names kept as deprecated shims for one
  minor version, then removed. Strongest signal, real (eventual) breakage.

**Proposed: (A) now, (C) as the announced direction.** Ship the tiers + guard +
docs immediately (non-breaking), and document that experimental symbols *will*
move under `proteon.experimental.*` in a future minor. Avoids the PEP-562
lazy-binding complexity of (B) while still committing to a real signal.
**Reviewer: is (A)-now/(C)-later too soft — should we just do (C) and eat the
breakage while the user base is still essentially us?**

### 4. The CI growth guard (the point of the exercise)

Extend `test_public_api_surface.py`:

1. **Frozen stable snapshot.** Check `__stable__` against a checked-in literal
   set (in the test, or `tests/data/stable_api_snapshot.txt`). Any add/remove
   fails the test with a message saying "stable API changed — this is a
   contract change; update the snapshot intentionally and note it in the
   changelog." This makes stable-surface growth a *deliberate, reviewed* act,
   not an import side-effect. (Mirrors the existing Version-Sync gate ethos.)
2. **No leaks / no overlap.** `set(__all__) - {"__version__"} == __stable__ |
   __experimental__`; `__stable__.isdisjoint(__experimental__)`. Catches a
   symbol added to a group tuple but not tiered.
3. **Resolvability** (keep existing): every name resolves; uniqueness.
4. **Experimental is allowed to churn** — `__experimental__` is *not* frozen
   (research moves), but it must stay disjoint from the frozen stable set, so a
   frontier symbol can never silently become "stable."

The asymmetry is deliberate: **freeze what we promise, let the frontier move.**

## What this is NOT

- Not removing anything (mechanism A is non-breaking).
- Not a docs rewrite — `STABILITY.md` is a short tier table, not API reference.
- Not the correctness suite (readiness item #2) — that's the next doc.
- Not touching Rust crate publishability (item #3).

## Test plan

- `test_public_api_surface.py`: stable snapshot match; union/disjoint
  invariants; existing resolve+unique tests stay green.
- A deliberately-added dummy export to a stable group fails the snapshot test
  (proves the guard bites).
- `import proteon; proteon.__stable__ & proteon.__experimental__ == set()`.
- `from proteon import *` still binds the stable core (existing test unchanged).
- Round-trip: every `__experimental__` name is still importable from top level
  under mechanism (A) (non-breaking proof).

## Decisions (post-claudex, 2026-06-24)

Codex reviewed; the maintainer chose the **strict pure-compute core**. Final:

- **Stable = "compute one oracle-validated quantity from a structure, fixed
  signature, named oracle+N+tolerance."** Concretely: `_ALIGN_API`,
  `_ANALYSIS_API`, `_GEOMETRY_API`, `_SASA_API`, `_DSSP_API`, `_HBOND_API`,
  `_IO_API`, `_STRUCTURE_API`, and a forcefield-stable subset (`compute_energy`,
  `batch_compute_energy`, `minimize_hydrogens`, `batch_minimize_hydrogens`,
  `minimize_structure`, `load_and_minimize_hydrogens`, `gpu_available`,
  `gpu_info`).
- **Everything else is experimental**, including the 47k-validated flagship
  (`prepare`, supervision build/export, corpus/sequence/training/cluster
  pipeline), `_HYDROGEN_API`, `_ARROW_API`, electrostatics (science *and*
  parsers, for now), `_SELECT_API`, `_FAILURE_API`, `_CORE_API`
  (`RustWrapperObject` — impl artifact, candidate for de-export), `run_md`, and
  Vina. Rationale: all fail the back-compat gate (heuristics/schemas still move)
  or lack a named oracle. `prepare` is robust (no crashes) but that is not
  correctness; its numeric output is schema-versioned, not frozen.
- **The 5 stable gates** (a symbol is stable only if ALL hold): (1) documented
  I/O + units/conventions; (2) defined error semantics; (3) a back-compat
  promise; (4) validation evidence with oracle + N + tolerance named; (5) a
  schema/versioning policy. These live in `STABILITY.md`.
- **Mechanism: C, staged.** PR1 (this one) is **non-breaking**: add
  `proteon.experimental.*` as the canonical path for experimental symbols (built
  programmatically from `__experimental__`, registered in `sys.modules`), keep
  flat names bound, add `__stable__`/`__experimental__` + the frozen snapshot
  guard + `STABILITY.md`, fix the "canonical" docstring language. PR2 flips flat
  experimental access to `DeprecationWarning` and migrates internal callers.
  Splitting avoids coupling the contract to the large caller migration, and
  sidesteps the PEP-562-vs-no-PEP-562 tension until PR2 decides it.
- **No `beta` tier.** Binary only; "experimental: schema-versioned / likely to
  graduate" is prose in `STABILITY.md`.
- **Guard: frozen snapshot, not `@stable` decorator.** Re-exports + bound Rust
  symbols make definition-site harvesting ambiguous; guard the export list.
  `__experimental__` is intentionally NOT frozen (research churns) but MUST stay
  disjoint from the frozen `__stable__`, so nothing silently graduates.

## Open questions for Codex (answered above)

1. Is keeping experimental names flat (mechanism A) a real signal or theatre —
   should we bite the breakage and do (C) now while the blast radius is small?
2. Tier assignments: is `assembly_builder` stable-by-grouping or
   experimental-by-age? Are the corpus release/dataset builders allowed to be
   "experimental" given the docstring currently calls them "canonical"?
3. Is a checked-in frozen snapshot the right guard, or is there a less
   brittle mechanism (e.g. requiring an explicit `@stable` decorator at
   definition site that the test harvests)?
4. Is the stable/experimental binary too coarse — do we need a `beta` middle
   tier for the corpus pipeline (API-stable-ish but schema still moving)?
5. Anything in the proposed STABLE set that does NOT deserve it — i.e. where
   the validation evidence is thinner than the tier implies?
