# Design note: unified `proteon` Rust CLI for the molecular-mechanics half

## Problem

proteon ships five Rust binaries (`proteon-bin`):

| Binary | Purpose |
|---|---|
| `tmalign` | TM-align structural alignment (release-tested) |
| `usalign` | US-align / MM-align (release-tested) |
| `ingest` | bulk PDB/mmCIF → per-atom Parquet |
| `build_kmi` | external-memory k-mer index builder (search prep) |
| `fasta_to_mmseqs_db` | FASTA → MMseqs2 DB (search prep) |

The entire molecular-mechanics + analysis half of the library is reachable
**only through the Python (PyO3) extension** — there is no command-line door:

- SASA (`proteon-connector/src/sasa.rs`)
- DSSP 3/8-class (`dssp.rs`)
- H-bonds (`hbond.rs`)
- hydrogen placement (`add_hydrogens.rs`)
- force-field energy: CHARMM19+EEF1, AMBER96 (`forcefield/energy.rs`)
- GB/OBC implicit solvent (`forcefield/gb_obc.rs`)
- minimization / MD (`forcefield/minimize.rs`, `md.rs`)
- missing-atom reconstruction (`reconstruct.rs`)
- the `prepare` / `batch_prepare` pipeline (load → reconstruct → protonate → minimize)
- geometry (RMSD, dihedrals, contact/distance maps)

The README pitches these as core capabilities behind "Python **and CLI** entry
points," but release binaries are only `tmalign` + `usalign`. Any infra that
doesn't want a Python runtime (HPC scheduler, Nextflow/Snakemake stage, slim
container, pure-Rust pipeline) cannot currently get a SASA number, a DSSP
string, or a minimized structure. The project's stated design principle is
"proteon is a compute kernel, not a platform; zero friction on any infra" — the
missing CLI directly contradicts it.

There is also a secondary gap **inside search**: the CLI can build the DB
(`fasta_to_mmseqs_db`) and the k-mer index (`build_kmi`), but no binary
actually *runs* a search or emits an MSA — that path (`proteon-search/src/search.rs`,
`msa.rs`) is Python-only too.

## What is already true (verified, not assumed)

The compute logic is **already cleanly separated from PyO3**. The `py_*.rs`
files are thin shims; the algorithms take plain Rust + `pdbtbx` types:

- `dssp::extract_dssp_residues(&pdbtbx::PDB) -> Vec<DsspResidue>`,
  `dssp::assign_dssp(&[DsspResidue]) -> String`, `dssp::dssp_from_pdb(&pdbtbx::PDB) -> String`
- `hbond::backbone_hbonds(&pdbtbx::PDB, f64) -> Vec<BackboneHBond>`
- `add_hydrogens::place_peptide_hydrogens(&mut pdbtbx::PDB) -> AddHydrogensResult`
- `sasa::sasa_from_pdb(...)`, `sasa::shrake_rupley(...)` (`sasa` is already `pub mod`)
- `forcefield::{energy, minimize, params, topology, gb_obc, md}` are all `pub mod`;
  the py wrappers just call `energy::...` / `minimize::...`

So **no PyO3 untangling is needed**. `py_dssp::compute_dssp` is literally
`extract_dssp_residues` + `assign_dssp`.

Two real constraints:

1. **Visibility.** Most compute modules are private (`mod dssp;`, `mod hbond;`,
   `mod reconstruct;`, `mod add_hydrogens;`) and their entry points are
   `pub(crate) fn` — reachable inside `proteon-connector` but not from
   `proteon-bin`. Only `sasa` and `forcefield` are `pub mod`.
2. **The PyO3 dependency.** `proteon-bin` does **not** currently depend on
   `proteon-connector`. The connector is `crate-type = ["cdylib", "rlib"]` with
   a hard `pyo3 = { features = ["extension-module"] }` dependency. On Unix,
   `extension-module` deliberately suppresses the libpython link directive.
   **This is a real build-mode/portability risk, not a guaranteed failure**
   (codex-review correction): whether the unused `#[pyfunction]` code's PyO3
   C-API references reach the final link depends on CGU granularity, rlib
   archive section layout, LTO, debug-vs-release, and platform linker dead-code
   elimination (e.g. macOS `dead_strip`). A plain binary that calls *only* the
   pure-Rust paths *may* link fine — but relying on the linker to GC the dead
   PyO3 symbols is not something to bank as an architectural invariant across
   Linux + macOS + debug/release/LTO. The only way to make this safe is feature
   gating (see Option A), not hoping for dead-code elimination.

   > **Empirical finding (prototype, 2026-06-02, branch
   > `feat/proteon-mm-cli-prototype`).** A plain `proteon` binary that depends
   > on `proteon-connector` (pyo3 `extension-module`) and calls only
   > `sasa::sasa_from_pdb` / `dssp::dssp_from_pdb` **links and runs on Linux —
   > including the release profile with `lto = true`, `codegen-units = 1`,
   > `strip = true`.** `ldd` shows no libpython dependency; the dead pyo3
   > symbols are GC'd even under full LTO. SASA/DSSP output is byte-identical to
   > the Python path. So on Linux the link risk does **not** materialize even in
   > the worst-case codegen config. **macOS is unverified** (different linker +
   > `dead_strip`); that is the only remaining unknown for Option A/C.
3. **Batch orchestration is Python-coupled.** `batch.rs` (`PyBatchItem` /
   `PyBatchResult`) is PyO3. The CLI cannot reuse it; it needs its own rayon
   loop (proteon-bin already depends on rayon).

## The central architecture decision

**Where does the pure-Rust compute live so both the PyO3 shim and the CLI can
call it without dragging pyo3 into a plain binary?**

### Option A — expose `proteon-connector` internals
Flip `mod dssp;` → `pub mod dssp;`, promote the needed `pub(crate) fn` to
`pub fn`, add `proteon-connector` as a dep of `proteon-bin`.

To make this *safe* (not just hope the linker GCs dead PyO3 symbols), the
connector must feature-gate pyo3 entirely out of the rlib the binary consumes:

```toml
# proteon-connector/Cargo.toml
[features]
default = []
python = ["dep:pyo3", "dep:numpy"]
extension-module = ["python", "pyo3/extension-module"]
```
```rust
#[cfg(feature = "python")] mod py_dssp;          // gate every shim
#[cfg_attr(feature = "python", pyfunction)]      // gate every #[pyfunction]
```
The maturin build enables `extension-module`; `proteon-bin` depends with
`default-features = false` (no `python`, no pyo3 in the link graph at all).

- **Pros:** smallest diff; no new crate.
- **Cons:** to be safe it requires `#[cfg(feature="python")]` on *every* shim
  and `#[cfg_attr(...)]` on *every* `#[pyfunction]` across the connector — at
  which point you have effectively built "a proto-core crate inside the
  connector," but a worse one: compute fns are committed to a public semver
  surface *and* still co-located with the binding code. Conflates "the Python
  binding crate" with "the reusable compute crate." (codex-review: the safe
  version of A is so much gating that B is cleaner for the same effort.)

### Option B — extract a pyo3-free `proteon-core` (or `proteon-mm`) crate
Move `sasa`, `dssp`, `hbond`, `add_hydrogens`, `reconstruct`, `forcefield`,
`bond_order`, `fragment_templates`, `altloc` into a new dependency-light crate.
`proteon-connector` keeps only the `py_*` shims and depends on `proteon-core`;
`proteon-bin` depends on `proteon-core` directly (no pyo3 anywhere near the binary).

- **Pros:** clean separation; CLI binary never links pyo3; the compute crate
  gets a deliberate public API; mirrors the existing split (`proteon-align`,
  `proteon-search` are already standalone compute crates — the connector is the
  odd one out by holding compute *and* bindings).
- **Cons:** larger up-front move (~several modules, their tests, shared helpers
  like `parallel.rs`); a churned import graph; one big mechanical PR before any
  CLI value lands.

### Option C — hybrid: thin facade module in the connector
Add `proteon-connector/src/api.rs` (`pub mod api`) re-exporting the compute
entry points the CLI needs, leaving everything else `pub(crate)`. proteon-bin
depends on the connector but only touches `api`.

- **Pros:** smaller than B, narrower public surface than A.
- **Cons:** does **not** solve the pyo3 link risk (constraint 2) — proteon-bin
  still links the connector. Only B fully removes pyo3 from the binary.

**Recommendation: Option B — but the prototype changes the urgency, not the
direction.** The empirical finding above shows Option A is *viable today* on
Linux (the prototype ships Phase 1 right now, with exact parity). So B is no
longer a *prerequisite* to shipping the CLI — it's the cleaner long-term shape,
decoupled from "get SASA/DSSP on the command line." The recommendation to
eventually land B stands, not because A "won't link" (it does), but because the
only fully-portable (macOS-safe) version of A requires gating pyo3 out
of the entire connector, which is strictly more work than B for a worse result:
B removes pyo3 from the CLI link graph by construction, matches the existing
workspace shape (`proteon-align` / `proteon-search` are already standalone
compute crates — the connector is the odd one out by holding compute *and*
bindings), and gives the compute a deliberate, reviewable public API instead of
a semver surface accreted by un-gating shims. Accept the one-time move cost.

`proteon-core` is a **workspace-internal API to start** (path-only, not
published; see open questions) — but it is a *real* public Rust API within the
workspace, so its surface is designed deliberately, not leaked.

## Proposed CLI surface (subcommands on a single `proteon` binary)

```
proteon sasa    IN.pdb [--per-residue] [--probe 1.4] [--points 960] [--format tsv|json|parquet]
proteon dssp    IN.pdb [--classes 3|8] [--format string|tsv|json]
proteon hbond   IN.pdb [--energy-cutoff -0.5 | --geometric --dist-cutoff 3.5]
proteon protonate IN.pdb -o OUT.pdb            # place polar H
proteon prepare IN.pdb -o OUT.pdb [--ff charmm19_eef1|amber96] [--minimize-steps 50]
proteon energy  IN.pdb [--ff amber96] [--gb-obc] [--components]   # total + per-term
proteon minimize IN.pdb -o OUT.pdb [--ff ...] [--max-steps 1000] [--method sd]
```

`search` is **deliberately out of this design** (codex-review): it's a different
subsystem (sequence search / MSA, not molecular mechanics), and folding it in
muddies the scope and phase risk. The search-run CLI gap is real and worth a
*separate* design note; this document is scoped to the MM + structure-analysis
surface only.

- Output: default human/TSV to stdout; JSON via `--format json`. **Parquet is
  deferred past v1** (codex-review): leaning on `proteon-arrow` only makes sense
  once those schemas are stable, and TSV/JSON is sufficient for parity and
  scripting. Revisit parquet once the arrow schemas settle.
- The single-binary multi-subcommand pattern (clap `derive` + subcommands)
  keeps `tmalign`/`usalign` as their own binaries (release-tested, C++-parity
  CLIs we don't want to disturb) while the new compute surface lands under one
  `proteon` entry point.

### I/O contract (must be explicit before write commands ship)

The read commands (`sasa`/`dssp`/`hbond`) emit tables; the **write** commands
(`prepare`/`minimize`/`energy`/`protonate`) touch chemistry and must state,
up front, what they do and don't handle — otherwise users assume more
correctness than the CLI guarantees (codex-review):

- **Formats:** PDB and mmCIF in; output format follows input unless `-o` extension
  says otherwise. Round-trip must preserve atom/residue identity.
- **Models:** first model only by default (`--model N` to override); document it.
- **Altloc:** policy stated (e.g. highest-occupancy kept), flag to override.
- **Missing atoms vs missing residues:** reconstruct missing *atoms*; missing
  *residues* (chain breaks) are reported, never fabricated.
- **Non-protein:** ligands / waters / ions / nonstandard residues — explicitly
  pass-through-untouched or dropped, never silently mishandled.
- **Termini & insertion codes:** charged-terminus handling and icode preservation
  stated per command.
- **Metadata:** what header/CRYST1/SEQRES survives a write is documented.

### Batch mode (must be specified, not "accepts a directory")

Every subcommand accepts a directory or multiple files and fans out with rayon
(`-j/--threads`, mirroring `ingest`). Specify, don't hand-wave (codex-review):
directory traversal depth (non-recursive by default), glob support, output
naming for per-file results, **failure isolation** (one bad file ≠ aborted run),
deterministic output ordering regardless of thread count, per-file errors to
stderr, and a **nonzero exit code on any partial failure**.

A canonical **structure-identifier model** (how a residue/atom is named in
output: model · chain · resnum · icode · altloc · atom name) is shared across
CLI, the Python API, and any future parquet — defined once, not per command.

## Scope / phasing proposal

- **Phase 0:** the crate decision above (B) — mechanical move + green tests.
- **Phase 1:** `sasa`, `dssp`, `hbond` (read-only, table outputs; highest value,
  lowest risk; share the structure-identifier model). TSV/JSON only.
- **Phase 2:** `prepare`, `energy`, `minimize`, `protonate` (write structures;
  reuse the exact code the 50K battle test drives) — gated on the I/O contract
  above being written down first.
- **Search-run CLI:** separate design note, separate subsystem (see above).

## Test plan

The parity goal is right but "CLI == Python output" alone is a trap: **if both
paths share a bug, parity passes and proves nothing** (codex-review; this is the
project's own `internal-consistency-isnt-validation` principle). The plan needs
both a shared-source guarantee *and* independent oracles.

- **Single source of truth.** Both the Python wrapper and the CLI call the *same*
  `proteon-core` entry point — no copied orchestration. Add a test asserting they
  invoke the same function, so the CLI can't quietly fork into a second
  implementation.
- **Rust-level parity** between `proteon-core` and the Python wrappers, so the
  CLI is not the *only* guard on the core API.
- **CLI-vs-Python golden tests** on `test-pdbs/` (1ubq, 1crn) for exact fixtures
  and exact options: DSSP 3- and 8-class, SASA point/probe variants, force-field
  options.
- **Independent oracles, not just parity:** `proteon sasa 1crn.pdb` reproduces
  the Biopython-validated SASA; DSSP against pydssp; energy against the existing
  BALL/OpenMM oracle fixtures — wherever an external oracle already exists in the
  EVIDENT suite, the CLI is checked against *it*, not only against Python.
- **Schema snapshot tests** for TSV/JSON: residue/model/chain IDs, insertion
  codes, altlocs, NaN/None rendering, and sort order all pinned.
- **Batch tests** proving deterministic output order (thread-count-independent)
  and failure isolation (one malformed file → that record errors, run continues,
  exit nonzero).
- **Round-trip tests** for write commands: re-parse the output, assert
  atom/residue identity preserved, no duplicate serials, no pathological
  formatting.
- **Cross-platform CI** (Linux + macOS) that specifically builds the binary with
  pyo3 *absent* from the link graph — the guard for the Option-B boundary.

## Resolved by the review

- **Crate choice → Option B** (above), for the sharpened reason.
- **Publish proteon-core?** → path-only inside the workspace to start; publish
  to crates.io later, after CLI/API churn settles.
- **Binary shape?** → single multi-subcommand `proteon` for the new MM/analysis
  commands; leave `tmalign`/`usalign` as their own untouched binaries.
- **Parquet?** → deferred past v1; TSV/JSON first.
- **Semver burden (old Q5)?** → accepted intentionally by choosing B;
  `proteon-core` starts workspace-internal so the surface is deliberate.

## Open questions still to settle

1. **proteon-core API boundary:** low-level algorithms only (`assign_dssp`,
   `shrake_rupley`) or also high-level operations (`prepare(pdb, ff, steps)`)?
   The CLI wants the high-level ops; the question is whether those live in core
   or are composed in the CLI/Python layers (risking divergence — see test plan).
2. **Canonical structure-identifier model:** the exact residue/atom identity
   schema shared across CLI, Python, and future parquet — settle before Phase 1.
3. **Explicitly-unsupported chemistry scope:** what the write commands refuse or
   pass through untouched (nucleic acids, ligands, glycans, multi-model) — stated,
   not discovered by users.
4. **Output schema versioning:** how TSV/JSON schemas are versioned so a column
   change is a deliberate, detectable event (ties to the schema snapshot tests).
5. **Determinism:** are force-field / minimization results bit-stable across
   thread counts and platforms, or only to a documented tolerance? The batch
   determinism test depends on the answer.
