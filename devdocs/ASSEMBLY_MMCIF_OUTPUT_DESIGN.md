# mmCIF assembly output — unlock large assemblies

> Status: **design** (pre-claudex). Follows the assembly builder (PR1–3, merged).
> PDB output caps the builder at 62 chains / 99,999 atoms / ±9999.999 Å coords, so
> capsids and large oligomers (1mva → 180 chains, 1z14 → 60-mer) currently fail
> loud as `assembly_too_large_for_pdb` / `assembly_coords_exceed_pdb`. mmCIF has
> none of these limits — this emits mmCIF for the cases PDB can't represent.

## Feasibility (verified)

A hand-written minimal mmCIF `_atom_site` loop with **multi-character chain ids**
(`AAA`, `ZZ9`) loads cleanly via `load_mmcif`. The 15 core columns that work:
`group_PDB, id, type_symbol, label_atom_id, label_comp_id, label_asym_id,
label_seq_id, Cartn_x/y/z, occupancy, B_iso_or_equiv, auth_seq_id, auth_asym_id,
pdbx_PDB_model_num`. So the same hand-emit approach as PR1's PDB text works for
mmCIF, with no chain/serial/coordinate caps.

## Approach

`build_assembly` becomes **format-adaptive**: emit PDB when it fits (cheapest, and
the common dimer/tetramer case), else emit mmCIF — never a size drop reason.

- A `BuiltAssembly.format` field (`"pdb"` | `"mmcif"`) and a generic `text` field;
  `pdb_text` is kept as a back-compat alias (returns `text`; the field name is
  public API and tests use it). `load()` dispatches to `load_pdb` / `load_mmcif`.
- The three PDB-capacity checks (>62 chains, >99,999 atoms, coords beyond
  ±9999.999) **no longer return drop reasons** — they select mmCIF instead. The
  only remaining drops are `no_assembly_metadata` / `biomolecule_not_found` (true
  "can't build") — size is never a reason to drop now.

### mmCIF chain ids

No 62-char alphabet limit, so ids can be multi-char and human-readable. Scheme,
preserving the deposited id for the identity copy:

```
identity copy of source "A"   -> "A"
expansion copies of "A"       -> "A-2", "A-3", … "A-60"
```

Deterministic, unique, and traceable back to (source, operator). The blank-chain
identity → a sentinel like `"."`-free placeholder; mmCIF label_asym_id can't be
empty, so a blank source uses a single legal id (e.g. `"A"`)—the only case where
the deposited id isn't literally preserved (documented).

### `_atom_site` row generation — a REAL serializer (claudex)

mmCIF generation is treated as a proper serializer, not "PDB rows with longer
chain ids." Every string field routes through a single `_cif_token(value)`
(handles `""`→`.`, `?`, embedded spaces/quotes → quoting). The full column set
(beyond the 15 minimal) adds the identity-preserving fields PR2 re-prepare needs:

- `label_alt_id` — the source altloc char, or `.` (else alternate conformers
  collapse / mis-read).
- `pdbx_PDB_ins_code` — the source insertion code, or `?` (else `10A`/`10B`
  collapse to one residue under `auth_seq_id` alone).
- `label_seq_id` — **integer for ATOM polymer rows, `.` for HETATM** (ligands /
  waters have no polymer sequence position; `= resseq` for HETATM is wrong).
- `auth_seq_id` — the original PDB residue number (incl. negatives).
- `type_symbol` — from the **PDB element column (cols 76–78)**; only when blank,
  infer with proper rules (strip digits, two-letter elements, common-atom-name
  special cases) — NOT the atom-name first letter (`CA` = calcium vs Cα).
- `pdbx_PDB_model_num = 1` always (model-1-only build).

Per source ATOM/HETATM line: parse those fields, transform xyz by the operator,
emit a row with the generated `label_asym_id` == `auth_asym_id`, a fresh
sequential `id`, and `%.3f` coords (locale-independent). Missing occupancy/B →
`1.00` / `0.00`.

### `pdb_text` guard (claudex)

`pdb_text` must NOT silently return mmCIF (callers may feed it to a PDB parser).
`BuiltAssembly` (an OUTPUT type — returned by `build_assembly`, not user-
constructed) gains `text` + `format`; `pdb_text` survives as a READ property that
returns `text` only when `format == "pdb"` and raises a clear error for mmCIF
(directing to `.text` / `.load()`). The serialized text moved from the `pdb_text`
field to `text`, so the back-compat is read-access (the realistic surface for a
result object); the lone internal constructor (`build_assembly`) uses `text=`. All
parsing goes through `load()` (dispatches on format).

## What stays the same

Per-block operator applicability, identity-first global dedup, cross-chain
coincident-copy skip, provenance (`AssemblyChain`), and the re-prepare (PR2) /
export (PR3) paths — all unchanged. `prepare_assembly` / `build_assembly_
supervision_examples` work on the mmCIF-backed `BuiltAssembly` transparently
(`load()` dispatches; `prepare` takes the loaded structure either way).

## Test plan

- 1z14 (60-mer) and 1mva (180 chains): now build as mmCIF (`format == "mmcif"`),
  load, and have the expected chain count; previously `assembly_too_large_for_pdb`.
- A large-translation assembly (coords > 9999.999): builds as mmCIF, was
  `assembly_coords_exceed_pdb`.
- A small dimer (1doe): stays PDB (`format == "pdb"`); `pdb_text` alias still works.
- mmCIF round-trip: built assembly's `load()` reproduces the chain count and ~atom
  count; identity copy preserves the source id.
- `prepare_assembly` / `build_assembly_supervision_examples` work on an mmCIF
  assembly (re-prepare + interface gate over the capsid).
- **PDB↔mmCIF parity (claudex):** force-emit a small dimer as BOTH PDB and mmCIF,
  re-prepare each, and assert the prepared structures match — chain ids, residue
  count, `n_heavy_clashes` / `clash_residue_indices`, `clashscore`. Proves the
  large-assembly (mmCIF) path behaves identically to the small (PDB) path through
  PR2/PR3, so a structure with ATOM + HETATM (and an insertion code / altloc if a
  fixture has one) re-loads the same either way.
- `pdb_text` raises on an mmCIF assembly; `text` returns the mmCIF.

## Scope boundaries

- Minimal `_atom_site` loop only — no full mmCIF metadata (entity, struct, etc.);
  enough to re-load + re-prepare, not to be a publication-grade mmCIF.
- The capsid's SIZE is now representable; whether it PASSES the interface gate is
  a separate (coverage/clash) question — a 60-mer with severe inter-copy clashes
  still drops on coverage, correctly.
