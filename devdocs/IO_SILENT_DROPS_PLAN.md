# IO_SILENT_DROPS_PLAN — stop silently corrupting / dropping supervision data

Status: DRAFT (pre-implementation). Tier-2 follow-up to the weak-spot audit. Scope:
the I/O silent-data issues that reach the **DL supervision tensors**, traced
end-to-end (file → connector → Python `Structure` → `build_structure_supervision_example`).
NOT the TM-align extraction path (`proteon-io/pdb_io.rs`), whose first-model /
primary-conformer behaviour is correct-by-design for alignment.

## 1. Findings (traced, with classification)

**Headline real bug — `residue_index` collapses insertion codes.** Built from raw
`serial_number` on **both** paths:
- Python `packages/proteon/src/proteon/supervision.py:198`
- Rust `proteon-connector/src/py_supervision.rs:435` (record at `:268`)

so residues `10` and `10A` receive the **same** `residue_index` (10) — duplicate,
non-monotonic indices. Any downstream relative-position / pair-bias encoding keyed on
`residue_index` (AF-style relpos: `clip(idx[i]−idx[j], −32, 32)`) treats the two as the
**same chain position**. The icode-aware `continuity_index`
(`supervision_geometry.py`) exists but is a *bond-adjacency* signal (+1=bond, +2=break)
wired only into torsion masking — it is **not** a numbering index and cannot replace
`residue_index` (which the schema, `STRUCTURE_SUPERVISION_SCHEMA.md:161-164`, defines as
"residue numbering in the example coordinate order", i.e. it must preserve gaps).

**By-design-but-silent (should surface, not change behaviour):**
- **Models 2..N dropped.** Connector load never sets `only_first_model`
  (`py_io.rs:17-26`); `_select_chain` / `select_chain` pick model 1's chain
  (`supervision.py:455-465`, `py_supervision.rs:232-242`). Model 1 is the right AF
  convention, but a multi-model (NMR) input is silently reduced with no signal.
- **Altloc B+ dropped, occupancy ignored.** `primary_conformer`
  (`proteon-core/src/altloc.rs:31-41`) picks blank→"A"→first; occupancy is never
  consulted, so a higher-occupancy altloc B loses to A silently.
- **Parse warnings discarded.** `let (pdb, _errors) = …` on every connector load
  (`py_io.rs:37,54,71,142`); the reserved `StructureQualityMetadata.parse_warnings`
  (`supervision.py:86`) is never populated.

## 2. Fix — `residue_index` (primary; the real corruption)

**Revised after claudex (the running-max idea was rejected — see §7).** The supervision
sequence is built from **coordinate-present residues only** (`supervision.py:192-197`,
`sequence = join(present residues)`), so the sequence / atom37 tensors are **gapless**.
The `residue_index` must therefore be the matching **sequence coordinate**, not author
numbering: encoding author-number gaps into a gapless representation is itself a
corruption (relpos would read adjacent rows as far apart), and the existing
`serial_number` code does exactly that (the icode collapse is its most visible symptom).

**Use a 0-based positional index** — the AlphaFold/OpenFold convention for a
resolved-residue sequence:

```
residue_index = arange(N)        # 0 .. N-1, in coordinate order
```

- **Distinct + monotonic** — `10, 10A, 11` → `0, 1, 2`; the collapse is gone.
- **Consistent with the gapless sequence/atom37/atom14 tensors** — row i ↔ index i.
- **AF/OpenFold-correct** for a present-residues-only sequence; relpos
  `clip(idx[i]−idx[j], −32, 32)` sees the actual tensor-row distance.
- **Trivial parity / no overflow** — it is just the row counter (use `i64`/checked in
  Rust regardless; `N` is tiny).

**Preserve author identity separately** (so depositor numbering is not lost — it's
needed for provenance / joining back to the PDB). Carry the insertion code into the Rust
`ResidueRecord` (`py_supervision.rs:266-270`, currently only `serial_number`), and export
two new identity arrays on the example:
- `author_seq_id: int32[N]` (the old `serial_number`),
- `insertion_code: str[N]` (blank when none).

These are **identity metadata, not a relpos coordinate**. (A future, larger change could
derive a true polymer coordinate from mmCIF `label_seq_id` / SEQRES alignment to encode
*unresolved*-residue gaps; out of scope — this pipeline does not track unresolved
residues at all, so positional is the honest choice today.)

**One shared policy across all three builders** (claudex #3): structure supervision
(`supervision.py:198` + batch path), the Rust fast path (`py_supervision.rs:435`), **and
`SequenceExample`** (`sequence_example.py:71`, which independently emits raw serials).
Factor a `positional_residue_index(n)` helper so they cannot drift.

## 3. Deferred to a separate change (claudex #5)

Model / altloc / parse-warning **visibility** is explicitly **out of this PR**. It is a
distinct concern (deliberate normalization, not corruption), needs a structured
provenance channel (`models_present` / `model_selected` / `altloc_residues` /
`altloc_policy`, or `io_warnings` — NOT the parser-diagnostic `parse_warnings` bucket),
and threading real pdbtbx diagnostics requires retaining them on `PyPDB` from every
loader incl. batch. Occupancy-aware altloc selection (a behaviour change) is separate
again. Tracked as a follow-up; this PR is **only** the `residue_index` + author-identity
fix.

## 4. Tests (semantic, per claudex #6)

- **Positional index, structural supervision** on `icode_interleave.pdb`
  (`tests/corpus/insertion_codes/`): `residue_index == [0,1,2,3,4]` (was `[1,2,3,3,4]` —
  the duplicate at the 3/3A pair is gone); strictly increasing, no duplicates.
- **author identity preserved**: `author_seq_id == [1,2,3,3,4]` and
  `insertion_code == ["","","","A",""]` on the same fixture.
- **Synthetic residue lists** (via the `FakeResidue` helper already in
  `tests/test_supervision_torsion_continuity.py`): `10,10A,11` → `[0,1,2]`;
  `10,10A,10B,11` → `[0,1,2,3]`; decreasing/restarted `(5,4,3)` → `[0,1,2]`; negative
  `(-2,-1,0)` → `[0,1,2]`; a HETATM/non-amino-acid interleave is filtered out before
  indexing (only `is_amino_acid` rows counted) and the index stays `0..N-1`.
- **Rust/Python parity**: the supervision parity test (`tests/test_supervision_rust_parity.py`)
  asserts Rust `residue_index` == Python on the icode fixture (add it to the parity set);
  exercise the **batched** Rust path + padding + empty-chain rejection.
- **Cross-artifact equality** (claudex #3): for the same chain, `SequenceExample` and
  structure-supervision `residue_index` are identical.
- mmCIF coverage (auth/label/icode) noted as a nice-to-have if a fixture is cheap;
  otherwise deferred with the polymer-coordinate follow-up.

## 5. Non-goals
- Author-numbering / running-max `residue_index` (rejected, §7).
- A true polymer coordinate from `label_seq_id` / SEQRES to encode unresolved-residue
  gaps — larger change; this pipeline doesn't track unresolved residues.
- Model/altloc/parse-warning visibility and occupancy-aware altloc selection (§3 — a
  separate change).
- Changing model-1 / primary-conformer SELECTION (correct for AF supervision).
- The TM-align extraction path (`proteon-io/pdb_io.rs`) — first-model is correct there.
- The pdbtbx `chain_count()` vs `chains()` model-0-vs-all asymmetry — upstream dep bug.

## 6. Files in scope
- `packages/proteon/src/proteon/supervision.py` (positional index; author-identity export)
- `packages/proteon/src/proteon/supervision_geometry.py` (`positional_residue_index` helper)
- `packages/proteon/src/proteon/sequence_example.py` (same index policy)
- `proteon-connector/src/py_supervision.rs` (positional index parity; carry insertion_code
  into `ResidueRecord`; export author_seq_id/insertion_code)
- `STRUCTURE_SUPERVISION_SCHEMA.md` (update `residue_index` semantics; add author-identity
  fields)
- tests: `tests/test_supervision*.py`, `tests/test_sequence_example.py`

## 7. Review log

claudex (codex) reviewed v1 and **rejected the running-max formula**: author numbering is
depositor metadata, not a polymer coordinate, and `max(serial, prev+1)` makes relpos
distances accidentally depend on later author gaps. Adopted: 0-based **positional**
`residue_index` (AF/OpenFold convention, consistent with the gapless present-residues
sequence) + preserve `(auth_seq_id, insertion_code)` as separate identity metadata;
**one shared policy** incl. `SequenceExample` (#3); model/altloc/parse-warning visibility
**split out** to a separate change with a structured channel (#5); i64/checked parity +
batched/padding/empty tests (#4); semantic test matrix (#6). The headline bug (icode
collapse) is real on both paths (premise verified empirically: SER `3` and VAL `3A` both
report `serial_number=3`).
