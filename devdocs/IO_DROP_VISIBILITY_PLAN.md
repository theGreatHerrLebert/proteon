# IO_DROP_VISIBILITY_PLAN — surface the silent supervision I/O drops as structured provenance

Status: DRAFT (pre-implementation). The **visibility half** that #150 explicitly split out
(the corruption half — `residue_index` — shipped in #150). Per codex's #150 review:
"Model and conformer reduction are deliberate normalization decisions, better represented
as **structured provenance** (`models_present`, `model_selected`, `altloc_residues`,
`altloc_policy`) … than free-form strings; thread parser diagnostics onto `PyPDB` from
every loader." This change makes the drops **visible & filterable**; it does NOT change the
selection behaviour (model-1 / primary-conformer stay — occupancy-aware altloc is separate).

## 1. The three silent drops (on the supervision path)

1. **Models 2..N dropped** — a multi-model (NMR) input is reduced to model 1 with no signal
   (`_select_chain` picks model 1's chain). Model 1 is the right AF convention, but silent.
2. **Altloc B+ dropped** — `primary_conformer` keeps blank→A→first per residue; alternates
   vanish with no record (occupancy not consulted — separate follow-up).
3. **Parse warnings discarded** — every connector loader does `let (pdb, _errors) = …` and
   throws away pdbtbx's non-fatal diagnostics (`py_io.rs:37/54/71/142`), and the reserved
   `StructureQualityMetadata.parse_warnings` is never populated.

All three are already representable: `Structure.model_count` and `Residue.conformer_names()`
are exposed in Python; only parse warnings need a small connector addition.

## 2. Fix (revised per claudex)

**[BUG] Scope chain selection to model 1 (Python + Rust).** `PyPDB.chains()` flattens ALL
models, so `_select_chain` / `select_chain` pick the first chain matching by *order* across
all models — if model 1 lacks the requested chain but model 2 has it, supervision silently
selects **model 2's** chain (claudex). Fix: select from model 1 explicitly —
`structure.models[0].chains` (Python) and `pdb.models().next()?.chains()` (Rust) — then
compute provenance from that exact chain. (`structure.models[0]` is already exposed; the
single-model common case is unaffected.)

**Connector — retain parse warnings on `PyPDB` (all FS loaders).**
- Add `parse_warnings: Vec<String>` to `PyPDB` (`py_pdb.rs`). `from_inner(pdb)` keeps empty
  (so Arrow reconstruction etc. correctly get none); add `from_inner_with_warnings`.
- The FS load paths — `load`, `load_pdb`, `load_mmcif`, `load_one`, `batch_load`,
  `batch_load_tolerant` — capture the `_errors` (non-fatal Loose-level diagnostics) and pass
  through. `load_one` returns `(PDB, Vec<String>)` so both batch APIs + rescue flows carry
  them. Preserve pdbtbx severity/category in the string if cheaply available.
- Expose `#[getter] fn parse_warnings(&self)`. These are **immutable initial-parse
  diagnostics** (prepare replaces only `pdb.inner`, so the sibling field survives unchanged —
  not re-validated). Python `Structure.parse_warnings` property over it.

**Connector — a precise altloc signal.** Don't parse `conformer_names` strings. Add
`#[getter] fn conformer_count(&self)` to `PyResidue` (number of conformers from the same
primary-conformer policy). A residue with `conformer_count > 1` is exactly one where primary-
conformer selection dropped an alternate. Python `Residue.conformer_count`.

**Python supervision — structured provenance in `StructureQualityMetadata`** (all
optional/defaulted; **no schema bump** — quality is one `quality_json` blob via `asdict`, so
defaulted fields are backward-compatible):
- `models_present: Optional[int]` (= `structure.model_count`),
  `model_selected_index: Optional[int]` (= 0; unambiguous ordinal, not a MODEL serial),
  `conformer_reduced_residue_count: Optional[int]` (residues on the selected chain with
  `conformer_count > 1`), `altloc_policy: Optional[str]` ("primary"), and populate the
  reserved `parse_warnings`.
- Compute in `build_structure_supervision_example` (single + batch) and **pass into the
  quality at construction** (not post-hoc mutation). Use `getattr(structure, "parse_warnings",
  [])` / `getattr(r, "conformer_count", 1)` fallbacks so existing fake/`SimpleNamespace` test
  structures stay valid. Share via `_io_provenance(structure, chain)`.

## 3. Non-goals
- Changing the model/altloc SELECTION *result* (still model 1 / primary conformer — the #1
  fix only makes selection *correctly* model-1, it doesn't change which conformer is kept).
  Occupancy-aware altloc selection = separate.
- Multi-model / multi-state supervision export; a per-residue altloc map (count + policy
  suffice for filtering).
- Bumping the parquet schema version (not needed — §5/claudex).

## 4. Tests (expanded per claudex)
- **Model-1 selection regression (the bug):** a 2-model fixture where model 1 lacks chain
  `B` but model 2 has it — supervision must select model 1's chain (or raise), NOT model 2's.
  Single AND batch paths.
- **Models provenance:** `tests/corpus/multimodel/two_models.pdb` → `models_present == 2`,
  `model_selected_index == 0`; single-model → `models_present == 1`.
- **Altloc provenance:** `tests/corpus/altloc/dual_conformer.pdb` (VAL 2 has A/B) →
  `conformer_reduced_residue_count == 1`, `altloc_policy == "primary"`; a no-altloc structure
  → `0`. Cover blank+`A` and `A`+`B`.
- **Parse warnings:** a structure loaded with non-fatal warnings → `structure.parse_warnings`
  non-empty, propagated to `quality.parse_warnings`; clean → empty (not None); warnings
  **survive `prepare` and `batch_prepare`**; `from_inner` (Arrow path) → empty.
- **Round-trips:** `quality_json` carries the new fields (asdict→json→reload) on BOTH the
  supervision and training parquet paths; single/batch parity for every new field.

## 5. No schema bump (claudex #5)
Quality is serialized as a single `quality_json` string (`asdict` → JSON → reload), so adding
defaulted dataclass fields is backward-compatible and does NOT change the Arrow schema —
`SUPERVISION_PARQUET_SCHEMA_VERSION` stays v2. (Verify in implementation that an old v2
`quality_json` lacking these keys still reloads via the dataclass defaults.)

## 6. Files
- `proteon-connector/src/py_pdb.rs` (PyPDB.parse_warnings + getter; PyResidue.conformer_count)
- `proteon-connector/src/py_io.rs` (capture `_errors` in all FS loaders; `load_one → (PDB, warnings)`)
- `proteon-connector/src/py_supervision.rs` (Rust `select_chain` → model-1 scope)
- `packages/proteon/src/proteon/structure.py` (`parse_warnings` + `conformer_count` props)
- `packages/proteon/src/proteon/supervision.py` (model-1 `_select_chain`; new quality fields;
  `_io_provenance`)
- tests under `tests/`

## 7. Review log (claudex)
**Bug found:** chain selection flattens all models → silent cross-model selection; fix to
model-1 scope (Python + Rust) + regression. Adopted: precise altloc signal via
`PyResidue.conformer_count` (not `conformer_names` string parsing) →
`conformer_reduced_residue_count`; `model_selected_index=0` (unambiguous); **no schema bump**
(quality is one JSON blob); populate quality at construction, not post-mutation; `getattr`
fallbacks for fake structures; warnings = immutable initial-parse diagnostics surviving
`prepare`; expanded single/batch + round-trip tests. No findings rejected.
