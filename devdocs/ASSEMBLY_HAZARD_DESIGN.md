# Biological-assembly label hazard — design

## Why this is the hard frontier

The PDB **asymmetric unit (ASU)** — what you get when you load a file — is not
always the **biological assembly** (the functional oligomer). A homotetramer
may be deposited as one chain + symmetry operators; a monomer may be deposited
with crystal-packing neighbors in the ASU. For label-safety this corrupts a
specific class of labels:

- **interface / contact / neighbor-graph / SASA / pocket** labels depend on the
  *correct* set of neighboring chains. Train on the ASU when the biological unit
  is a tetramer → missing three subunits' worth of interfaces; train on an ASU
  that includes crystal mates → false biological contacts.
- **per-chain backbone coordinate** labels are assembly-INDEPENDENT (a chain's
  coordinates are identical in the ASU and the assembly).

So unlike the other hazards, assembly is not "this structure is broken" — it's
"you may be using the wrong oligomeric state for contact-type labels."

## Feasibility constraint (the crux)

pdbtbx parses `MtriX` (MTRIXn / non-crystallographic symmetry to build the ASU)
and `Symmetry` (space group), but does **NOT** parse `REMARK 350` / `BIOMT`
(PDB) or `_pdbx_struct_assembly` (mmCIF) — the biological-assembly operators.
**The loader discards the assembly definition.** Verified: the raw text is
present (`4hhb` REMARK 350 declares a tetramer over chains A,B,C,D with identity
BIOMT — i.e. the ASU already IS the biological unit), but it is gone after load.

Consequences:
- Proper detection requires parsing the assembly metadata OURSELVES from the
  raw file (PDB `REMARK 350 BIOMT`, and the different mmCIF
  `_pdbx_struct_assembly_gen` + `_pdbx_struct_oper_list`).
- This is **path-based only**: `prepare(structure)` has no file to read, so the
  signal is available only in `prepare_for_supervision` / `batch_load_and_prepare`
  (which take paths). An inherent limitation worth stating.

## What is detectable from REMARK 350 BIOMT

Per BIOMOLECULE: the set of operators (BIOMT matrices) and the chains they apply
to. Then:
- **`biological_assembly_copies`** = number of operators in biomolecule 1 (1 =
  identity-only ⇒ the listed chains already form the assembly).
- **`assembly_is_asu`** = the assembly is exactly the chains present with
  identity transforms (no expansion needed). 4hhb: True (tetramer present,
  identity BIOMT).
- **`assembly_ambiguous` / hazard `assembly_state`** = the assembly needs
  operators the ASU doesn't already realize (expansion required), OR REMARK 350
  is absent (unknown). This is the contact-label hazard.

## Two levels of capability

1. **Detect + flag** (smaller): parse REMARK 350, expose
   `biological_assembly_copies` / `assembly_is_asu`, and a label hazard for
   contact-type labels. Does NOT change coordinates.
2. **Apply the assembly** (larger): a repair-style action that generates the
   symmetry-related chain copies (apply the BIOMT transforms) so the structure
   IS the biological unit. A real structural construction (new chains), like the
   selectors but additive. Needs careful chain-renaming and is a separate, big
   piece.

## The profile question

Assembly affects interface/contact/SASA labels — a label type the current
profiles (`heavy_coords` / `all_atom` / `energy` / `sequence_indexed`) do NOT
cover. So either:
- add a new profile `label_safe_interface` (or `contacts`) that the
  assembly hazard blocks, leaving the existing profiles untouched (assembly does
  not corrupt per-chain coords/energy/sequence); or
- expose the assembly signals as informational fields a contact-label consumer
  checks, without a profile.

`heavy_coords` / `energy` / `sequence_indexed` must NOT be blocked by assembly —
a single chain's coordinates / energy / sequence are correct regardless of
oligomeric state.

## Proposed first cut (detect + flag, PDB REMARK 350)

- A raw `REMARK 350` parser (PDB) → `(biomolecule -> [BIOMT operators], chains)`.
- Report fields (path-based entry points only):
  `biological_assembly_copies: Option<usize>` (None if no REMARK 350),
  `assembly_is_asu: Option<bool>`.
- A new profile `label_safe_interface` that requires `assembly_is_asu == Some(true)`
  (the ASU already is the assembly) — interface labels are safe only then.
- `RepairPolicy` action for the `assembly_state` hazard: `accept` / `drop`
  (apply = a later, larger piece).
- mmCIF assembly parsing and "apply the assembly" are explicit follow-ons.

## Claudex decisions (adopted)

1. **Ship detect-and-flag as a conservative GATE** (prevent unsafe interface
   labels). The risk is OVERCLAIMING what detect-only proves — so the gate is
   strict and honestly scoped. Apply-the-assembly is a separate later piece;
   detect-and-flag does not produce correct oligomers, it prevents bad ones.
2. **Three states**, not boolean:
   `Some(true)` metadata says the chosen biological assembly is identity
   transforms over EXACTLY the present relevant chains; `Some(false)` metadata
   known but ASU is not sufficient/exact (expansion needed, or ASU has crystal
   extras, or multiple differing biomolecules); `None` no usable metadata.
   Absent REMARK 350 is `None` (no evidence) — NOT "monomer".
3. **`label_safe_interface` profile**: passes only on `Some(true)`; `Some(false)`
   and `None` block it (or `None` requires an explicit accept). Per-chain labels
   (`heavy_coords` / `all_atom` / `energy` / `sequence_indexed`) are NEVER
   blocked by assembly — a chain's coords/energy/sequence are oligomer-invariant.
4. **Strict `assembly_is_asu == Some(true)`** requires: single chosen biomolecule,
   identity-only operators, chains-listed == present chains exactly. Anything
   weaker is `Some(false)` or `None`. Do NOT infer from MTRIX / space group.
5. **Isolated parser, path layer.** REMARK 350 is text — parse it in the PYTHON
   path layer (`proteon.assembly`), since Rust `prepare(structure)` has no path
   and proteon-core stays I/O-free. The PrepReport gains optional fields
   (`biological_assembly_copies`, `assembly_is_asu`) set ONLY by the path-based
   `prepare_for_supervision`; `None` everywhere else. Shape it so an upstream
   pdbtbx / mmCIF parser can replace the source later.
6. **PDB REMARK 350 only in the first cut, advertised honestly** (a capability
   note: source = pdb_remark_350). mmCIF `_pdbx_struct_assembly` parity and
   apply-the-assembly are explicit follow-ons.

## First cut (adopted)

- `proteon.assembly`: parse REMARK 350 from a path → biomolecules (operators +
  chain lists); `assembly_metadata(path, present_chains)` → `(copies, is_asu)`
  with the strict three-state semantics.
- `PrepReport.biological_assembly_copies: Optional[int]`,
  `assembly_is_asu: Optional[bool]` (path-layer-set).
- `label_safe_interface` profile + `assembly_state` in `label_hazards` (only
  when not `Some(true)`); existing profiles untouched.
- `prepare_for_supervision` populates the fields from each path + its structure's
  present chains. `RepairPolicy` can `accept`/`drop` `assembly_state`.

## Original open questions (answered by claudex above)

1. Is detect-and-flag the right first cut, or is it near-useless without
   apply-the-assembly (since a contact-label consumer needs the actual oligomer,
   not just a flag)? Where's the value floor?
2. The interface profile: add `label_safe_interface`, or keep assembly as
   informational fields? Is a profile that the existing pipeline never blocks on
   coherent?
3. Detection semantics: is "identity-only BIOMT over all present chains" a
   reliable `assembly_is_asu` test? Edge cases: multiple biomolecules, partial
   chain lists, operators that are crystallographic vs NCS, an ASU containing
   MORE than the assembly (crystal packing — the opposite hazard).
4. REMARK 350 is absent in many files (1ubq has none). Is "absent ⇒ unknown ⇒
   hazard" right, or is absent usually "monomer, ASU == assembly"?
5. Is parsing raw PDB text in proteon (bypassing the pdbtbx loader) acceptable
   architecture, or should REMARK 350 parsing live in pdbtbx (upstream)?
6. mmCIF: most modern depositions are mmCIF-first with a DIFFERENT assembly
   encoding. Is a PDB-only first cut useful, or does it need both to matter?
