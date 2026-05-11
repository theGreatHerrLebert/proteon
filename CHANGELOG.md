# Changelog

All notable changes to proteon are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The CHANGELOG is the **human narrative**; the machine-readable claims
manifest at `evident/evident.yaml` and per-release bundle at
`evident/reports/<tag>/manifest.json` are the **audit trail**. Each
release tag has a paired EVIDENT bundle pinned by sha256.

## [Unreleased]

### Changed

- **Lock CHARMM fold-preservation 50K claim with first-run headline**
  (`evident/claims/fold_preservation_charmm_50k.yaml`). Headline from
  the 2026-05-11 monster3 run (proteon-side finished 2026-05-08,
  OpenMM-side 2026-05-11 02:17 local): n_attempted = 47 183.
  proteon-side: n_ok = 44 464 (94.2%), median TM = 1.0000.
  OpenMM-side: n_ok = 8 029 (17.0%), median TM = 0.9990. Paired
  records = 7 791. **median tm_diff (openmm − proteon) = −0.0006**
  (passes the `<0.01` drift band by ~17× and the title's tighter
  0.001 by ~1.7×). |tm_diff| < 0.005 covers 85.0% of paired records.
  All three claim tolerances pass on the proteon-side. The
  OpenMM-side pass_rate floor (n_ok/n_attempted ≥ 0.85) does NOT
  clear — same population-narrowing pattern as the AMBER96-SP 50K
  claim (ff14SB-era typer rejects residues with non-canonical
  atom names after PDBFixer prep). Tolerance values left unchanged
  pending a recalibration decision; the failure_modes section
  documents the actual breakdown.


- **Lock AMBER96 single-point 50K claim with first-run data**
  (`evident/claims/forcefield_amber_openmm_50k.yaml`). Headline from
  the 2026-05-08 run (post-histidine PR #60 + NoCutoff PR #59):
  n_attempted=47 183, n_ok=7 775, **median rel_diff = 3.41e-04**
  (passes the `<0.01` median band by ~30×). Median |diff| =
  20.3 kJ/mol. p95 rel_diff = 0.856 — long tail driven by residues
  OpenMM's ff96 typer rejects after PDBFixer prep. Pass-rate
  (n_ok/n_attempted = 16.5%) sits well below the claim's 0.85
  floor; the gap is dominated by population-coverage limits
  (44% missing-heavy-atom skips + 40% ff96 typer fails), not by
  algorithm quality. Title now reflects both outcomes honestly.
  Tolerance values left unchanged pending an explicit recalibration
  decision; failure_modes section documents the actual breakdown.

- **Lock DSSP 8-class 50K claim with first-run headline (post-T-fix)**
  (`evident/claims/dssp_8class_50k.yaml`). Headline from the
  2026-05-10 run with the T-interior fix (PR #73) and PP-helix
  post-process (PR #71): n_attempted=47 183, n_ok=21 370,
  **median agreement_rate = 0.9527** (passes the ≥0.95 median band).
  ≥0.95 = 54.8% / ≥0.90 = 94.6% / ≥0.85 = 97.9% / ≥0.80 = 98.4%.
  Median moved +9.45 pp from the pre-T-fix 0.8582. Pass-rate floors
  fail at population scale (n_ok/n_attempted = 45.3% vs 0.85;
  ≥0.95/n_ok = 54.8% vs 0.80) — driven by length-mismatch skips
  and residual T/S boundary + π-helix disagreement. Tolerance
  values left unchanged pending recalibration; failure_modes
  section documents the actual breakdown and the per-class
  composition deltas.

### Fixed

- **DSSP: stop classifying turn endpoints as T (turn)**
  (`proteon-connector/src/dssp.rs`). proteon's T-classification rule
  treated *any* non-space turn marker as "in turn", which over-counts
  by including the H-bond donor (`>`/`X`) and acceptor (`<`/`X`)
  positions. Canonical DSSP only classifies the *interior* residues
  of an n-turn (markers `3`, `4`, `5`) as T; donor/acceptor positions
  fall through to S (bend) or `-` (loop) unless separately covered
  by another turn. The 50K mkdssp oracle made the over-emission
  visible: proteon emitted T 88.76% more often than mkdssp, and
  66% of all per-residue diffs (1.8M residues) traced to spurious
  T's at boundary positions adjacent to helices and strands. With
  this fix, T-classification matches the canonical "interior only"
  rule. No existing tests broke; impact on the median agreement_rate
  vs mkdssp at 50K scale will be measured in the next oracle run.

- **DSSP runner: normalise mkdssp's PP-helix class to loop in comparison**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 emits a "P" (PP-helix)
  class via its default `--min-pp-stretch=3`, but proteon's DSSP doesn't
  emit P at all — every PP-helix residue therefore counted as a
  per-residue mismatch even when both implementations agreed on the
  underlying H-bond geometry. Map `P → '-'` on the mkdssp side at
  comparison time; the raw `mkdssp_ss` field preserves the unmodified
  P chars per record for forensic reanalysis. Both arms now compare
  on the same 8-class alphabet `{H,G,I,E,B,T,S,-}`. Surfaced by the
  first 50K run (median agreement_rate 0.8447 / only 1.0% structures
  clearing 0.95) — the systematic gap accounts for the bulk of the
  per-residue diff. Runtime disabling via `--min-pp-stretch=N` is
  not viable — mkdssp v4.6.1 / libcifpp 10.0.3 has a CLI bug where
  any non-default value triggers `terminate called without an
  active exception` (SIGABRT) on every PDB. Runner also now
  persists the full proteon/mkdssp SS strings per record (~10 MB
  at 50K) so future reclassifications don't need the 80-min compute.
- **DSSP runner: pass `--output-format dssp` explicitly to mkdssp**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 picks output format
  based on the output filename's extension; we pipe through
  `/dev/stdout` (no extension) so it defaulted to mmCIF, which the
  runner's parser doesn't handle — resulting in `mkdssp_n=0` for every
  PDB and 100% length-mismatch skips. One-line fix: pass the format
  explicitly. Surfaced when the gemmi-bridge run produced records but
  every record was a skip with empty `mkdssp_composition`.
- **DSSP runner: bridge through gemmi for mkdssp compatibility**
  (`validation/dssp_mkdssp_oracle.py`). mkdssp v4 + libcifpp's strict
  validator rejects many PDB-derived datablocks at parse time
  ("Duplicate Key violation" on the internal `_refine` table when a
  PDB has multiple `REMARK 3` records). The runner now pre-converts
  PDB → mmCIF via `gemmi.read_pdb().make_mmcif_document().write_file()`
  before invoking mkdssp on the mmCIF. Pipeline verified on 5 PDBs
  that previously failed (12e8, 8rqw, 7vsc, 1aaj, 7nz7) — all now
  produce valid DSSP output. Sub-second per-PDB conversion overhead.
- **EVIDENT image: enable CCD download in libcifpp build**
  (`evident/Dockerfile{,.cuda}`). Both Dockerfiles previously set
  `-DCIFPP_DOWNLOAD_CCD=OFF` based on an outdated assumption that PDB
  input didn't need the Chemical Component Dictionary. mkdssp v4 in
  fact requires the CCD at runtime for every per-residue lookup ("ALA",
  "GLY", ...) regardless of input format; without it, every PDB fails
  with `compound information not found in /usr/local/share/libcifpp`.
  Surfaced when the v0.2.0 50K DSSP-vs-mkdssp run failed on every
  attempted record. Build-time download adds ~70 MB to the image
  (components.cif.gz); the runtime cost is one mmap per process.

### Added

- **DSSP 8-class vs mkdssp release-tier 50K claim** — completes the v0.2.0
  trust pyramid's DSSP rung. mkdssp v4.6.1 is the wider biology
  community's canonical Kabsch-Sander 1983 reference; proteon ports the
  same algorithm. Per-residue agreement at 50K corpus scale closes
  the gap left by the existing CI (per-PDB fixture) and the existing
  release-tier (proteon-vs-pydssp 1k) claims.
  - `evident/claims/dssp_8class_50k.{yaml,md}` — claim wired with
    `last_verified: null` pending the first monster3 artifact.
  - `validation/dssp_mkdssp_oracle.py` — new runner. Calls
    `proteon.dssp` on one side, `mkdssp` via subprocess on the other,
    parses mkdssp's per-residue 8-class column, computes per-PDB
    agreement rate. Loop characters canonicalised to '-' on both
    sides so the space-vs-character convention isn't a confounder.
    pebble per-task isolation, resume + PROTEON_PDB_LIST support,
    v0.2.0 bind-mount contract — same shape as the AMBER96 and CHARMM
    oracles.
  - `evident/evident.yaml` includes list extended with the new claim.
- **Per-structure batch primitives for SASA, energy, and H-bond counts.**
  Adds `proteon.batch_atom_sasa`, `batch_residue_sasa`, `batch_relative_sasa`,
  `batch_hbond_count`, and `batch_compute_energy` — rayon-parallel wrappers
  that produce one result per input structure. Each new function is
  parity-tested against a Python loop of the corresponding per-structure
  call (exact float equality; same dict shape for energy components and
  topology counts). Unblocks proteon-graphein's batch entry point — see
  proteon-graphein/CHANGELOG once that ships.
  - `batch_atom_sasa`: existing Rust function, previously unexposed in
    Python (`py_sasa.rs::batch_atom_sasa` was registered but never wrapped);
    now surfaced through `proteon.sasa`.
  - `batch_residue_sasa` and `batch_relative_sasa`: new Rust functions in
    `py_sasa.rs`. Pre-extract a `StructureView` (coords + radii + per-residue
    atom counts + residue names) on the main thread so the rayon body
    holds only owned data.
  - `batch_hbond_count`: new Rust function in `py_hbond.rs`. Mirrors the
    existing `batch_backbone_hbonds` data-flow but reduces to per-residue
    counts (matches the single-call `hbond_count_per_residue` semantics).
  - `batch_compute_energy`: new Rust function in `py_forcefield.rs`. Routes
    the same FF aliases as `compute_energy` (`charmm19_eef1`, `amber96`,
    `amber96_obc`) and forwards `nbl_threshold` / `nonbonded_cutoff` per
    structure. Uses the same chunk-clone pattern as `batch_minimize_hydrogens`
    to bound peak memory.
- **HID/HIE/HIP histidine protonation-state variants in AMBER96 data**
  (#60, PR 1 of 3). proteon's AMBER96 oracle previously showed 7-12%
  rel_diff vs OpenMM AMBER96 on every PDB containing histidines
  because only one HIS template shipped (with HIP-like charges by
  accident). This PR adds the three canonical AMBER96 templates as
  separate residue names — mirroring the CHARMM19 HSD precedent
  rather than extending the position-only variant suffix system:
  - `proteon-connector/data/amber96.ini` gains 9 new charge sections
    (HID/HIE/HIP × mid-chain / -N / -C terminal variants), 165 atom
    entries total. Charges and atom-class names extracted directly
    from OpenMM's vendored `amber96.xml` (the v0.2.0 oracle's parity
    target) via the new `scripts/extract_amber96_histidine_charges.py`
    script — hand-authoring 165 partial charges would invite
    sub-percent drift no reviewer can catch.
  - `proteon-connector/data/fragments/HID.json`, `HIE.json`, `HIP.json`
    — fragment templates mirroring `HIS.json`, with each variant's
    `delete` list extended to drop the H atom that doesn't belong in
    that tautomer (HID drops Hε2; HIE drops Hδ1; HIP keeps both).
  - `proteon-connector/src/fragment_templates.rs` — 3 new entries in
    the static `TEMPLATES` array.
  Existing structures with residue name "HIS" continue to load and
  energy-compute identically — no behaviour change on the existing
  HIS path. Structures providing residue name HID/HIE/HIP get the
  correct per-tautomer AMBER96 charges. The detection logic that
  decides which name to apply (based on which Hδ1 / Hε2 atoms are
  present in the PDB) lands in PR 2.

### Changed

- `validation/amber96_oracle.py` now calls
  `proteon.normalize_histidine_tautomers` between PDBFixer-write and
  `proteon.load`, so the runner's proteon arm sees HID/HIE/HIP residue
  names (matching what OpenMM AMBER96 detects internally on the same
  topology). Resolves the 7-12% rel_diff on every histidine-containing
  structure that surfaced in v0.2.0 contract smoke (#60).
  Local end-to-end on 1ubq (1 HIS, HD1-only): rel_diff 0.026% (down from
  ~10% pre-fix). The runner's per-record JSONL now also carries a
  `histidine_tautomer_counts` field for observability.
- `proteon` package exports `normalize_histidine_tautomers` at the top
  level — `proteon.normalize_histidine_tautomers(in, out)` is the public
  call site.

### Added

- **`proteon.prepare.normalize_histidine_tautomers(in_path, out_path)`**
  (#60, PR 2 of 3). Reads a PDB, walks each HIS residue, inspects which
  of Hδ1 / Hε2 are present, and writes a copy with the residue name
  updated to HID / HIE / HIP. Idempotent on already-renamed inputs.
  Returns a per-residue tally so the caller can log / observe the
  classification. The Rust loader is unchanged — proteon's existing
  residue-name-driven dispatch picks up the correct AMBER96 charges
  via the data added in PR #62.
- **`UserWarning` once per process when AMBER96's `compute_energy` sees
  any residue still named "HIS"** — flags the systematic ~7-12% drift
  this issue caused, and points the user at
  `proteon.prepare.normalize_histidine_tautomers`.
- **`tests/test_amber_invariants.py::TestHistidineTautomers`** — 6 tests
  pinning the renaming behaviour: HD1-only→HID, HE2-only→HIE,
  both→HIP, neither→stays HIS, idempotent on second call, and an
  end-to-end check that compute_energy with renamed residues hits the
  HID template's electrostatic charges (i.e. PR #62's data is wired
  into the typer correctly).

- **HID/HIE/HIP histidine protonation-state variants in AMBER96 data**
  (#60, PR 1 of 3). proteon's AMBER96 oracle previously showed 7-12%
  rel_diff vs OpenMM AMBER96 on every PDB containing histidines
  because only one HIS template shipped (with HIP-like charges by
  accident). This PR adds the three canonical AMBER96 templates as
  separate residue names — mirroring the CHARMM19 HSD precedent
  rather than extending the position-only variant suffix system:
  - `proteon-connector/data/amber96.ini` gains 9 new charge sections
    (HID/HIE/HIP × mid-chain / -N / -C terminal variants), 165 atom
    entries total. Charges and atom-class names extracted directly
    from OpenMM's vendored `amber96.xml` (the v0.2.0 oracle's parity
    target) via the new `scripts/extract_amber96_histidine_charges.py`
    script — hand-authoring 165 partial charges would invite
    sub-percent drift no reviewer can catch.
  - `proteon-connector/data/fragments/HID.json`, `HIE.json`, `HIP.json`
    — fragment templates mirroring `HIS.json`, with each variant's
    `delete` list extended to drop the H atom that doesn't belong in
    that tautomer (HID drops Hε2; HIE drops Hδ1; HIP keeps both).
  - `proteon-connector/src/fragment_templates.rs` — 3 new entries in
    the static `TEMPLATES` array.
  Existing structures with residue name "HIS" continue to load and
  energy-compute identically — no behaviour change on the existing
  HIS path. Structures providing residue name HID/HIE/HIP get the
  correct per-tautomer AMBER96 charges. The detection logic that
  decides which name to apply (based on which Hδ1 / Hε2 atoms are
  present in the PDB) lands in PR 2.

### Changed

- **Fold-preservation runners now support resume + `PROTEON_PDB_LIST`**.
  All four runners (proteon CHARMM, proteon AMBER, OpenMM CHARMM,
  OpenMM AMBER) previously opened OUT in `"w"` mode and re-walked the
  full directory glob on every invocation. At 50K scale this means a
  single SIGTERM / OOM / NFS hiccup during the multi-day run dropped
  every record, so the entire run had to restart from zero.
  Mirrors the pattern already shipped in `charmm19_eef1_ball_oracle.py`
  (#44):
  - On startup: read existing OUT, build `done_names` set of PDBs
    already in the JSONL.
  - Compute `pending = sample − done_names`, skip what's already done.
  - Open OUT in `"a"` mode so the resume seed is preserved.
  - `PROTEON_PDB_LIST=path/to/list.txt` overrides the directory glob,
    so 50K runs can use the same `validation/protein_only_50k.txt`
    pre-filter the CHARMM 50K oracle does (drops ~6K non-protein
    structures the FF can't parametrise).
  Surfaced when the first 50K fold-pres CHARMM run on monster3 hit
  proteon-side rate 0.32/s → ~40 hr per side, so the full chain is
  ~3 days. At that scale, no resume is unacceptable.

### Added

- **Three new 50K release-tier claims** (covering the v0.2.0 #42
  vision — full-corpus capture for every release-tier capability where
  monster3 compute permits):
  - `proteon-amber96-vs-openmm-corpus-50k-pdbs`
    (`evident/claims/forcefield_amber_openmm_50k.{yaml,md}`) —
    single-point AMBER96 energy at full corpus scale. Cross-implementation
    oracle counterpart to the fold-preservation AMBER claim, sensitive
    to a different bug class (per-term coefficient mismatches that don't
    surface in TM-score after minimization).
  - `proteon-charmm19-fold-preservation-vs-openmm-release-50k-pdbs`
    (`evident/claims/fold_preservation_charmm_50k.{yaml,md}`) — proteon
    CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2 minimization at 50K. Confirms
    the 1k headline (median tm_diff +0.0040) holds at population scale.
  - `proteon-amber96-fold-preservation-vs-openmm-release-50k-pdbs`
    (`evident/claims/fold_preservation_amber_50k.{yaml,md}`) — proteon
    AMBER96 vs OpenMM AMBER96 at 50K. The cleanest cross-implementation
    oracle proteon ships, since both arms claim the same parameter set.
  Each claim's `last_verified` block is null pending the first monster3
  run; tolerances and assumptions are wired in so the manifest validates
  immediately and the run only has to fill in the headline numbers.
- `evident/evident.yaml` includes list extended with the three new
  claim files.
- **SASA release-tier claim now triangulates against TWO oracles** —
  Biopython AND FreeSASA. `validation/run_validation.py::test_sasa`
  computes a FreeSASA Shrake-Rupley result alongside Biopython's, records
  `freesasa_total` / `freesasa_time_ms` / `freesasa_relative_diff` per
  structure, and downgrades to `warn` if EITHER oracle disagrees by >5%.
  Two independent C lineages agreeing on Shrake-Rupley is much stronger
  evidence than against either alone, per the
  `feedback_triangulate_with_two_oracles` principle (a shared bug in
  proteon vs Biopython couldn't also be present vs FreeSASA's
  independently-implemented core).
- `evident/claims/sasa.yaml` release-tier claim title + claim text +
  tolerances + oracle list updated for the two-oracle gate. Adds:
  - `median_relative_error < 0.005` vs Biopython (existing gate, unchanged)
  - `median_relative_error < 0.02` vs FreeSASA (new — looser because
    Biopython and FreeSASA themselves disagree by ~0.5–1% on most
    structures from atom-radius and probe-discretisation conventions)
  - `pass_rate >= 0.95` (existing gate, now keyed on either oracle
    triggering the warn downgrade)
  - Capability `sasa-cross-tool-parity` added alongside
    `sasa-accuracy-distributional`.
- `validation/report/render_sasa_release.py` extended: new
  `plot_freesasa_distribution` histogram, two new headline-table rows
  (median / p95 / p99 / band-pass count for the FreeSASA side), new
  HTML section explaining the two-oracle framing.

### Fixed

- `validation/amber96_oracle.py` now passes `nonbonded_cutoff=1e6` to
  `proteon.compute_energy(ff="amber96", ...)` to match OpenMM's NoCutoff
  convention. Without this, proteon truncates long-range Coulomb at 15 Å
  while the OpenMM arm goes full-range, producing a systematic ~5%
  median rel_diff. The 0.2% agreement the AMBER96 oracle ought to ship
  was being silently masked. Surfaced by the v0.2.0 contract end-to-end
  smoke on monster3 (50-PDB sample under apptainer bind-mount) — the
  bind-mount itself worked, but the runner output flagged the cutoff
  mismatch via proteon's own UserWarning. PR #56 shipped this gap; this
  PR closes it before any 50K AMBER96 run lands a real artifact.

### Changed

- All four fold-preservation runners now honour `N_PDBS` env override
  (was hardcoded `N = 1000`). Mirrors the pattern in the AMBER96 and
  CHARMM oracles. Required for the new 50K fold-preservation claims to
  scale beyond 1000 structures without code edits.
- `validation/amber96_oracle.py` migrated from
  `concurrent.futures.ProcessPoolExecutor` to `pebble.ProcessPool` for
  per-task subprocess isolation, mirroring the v0.1.4 CHARMM oracle
  pattern (#44). Pre-empts the `BrokenProcessPool` cascade at 50K
  scale that limited the v0.1.3 CHARMM run's `n_ok` to 807 of 44,210.
  Adds `TASK_TIMEOUT_S` env knob (default 60s) to terminate hung
  workers individually instead of stalling the whole run, hoists
  heavy imports (proteon, openmm, pdbfixer, pebble) to module
  top-level so cold-start cost is amortised across tasks, and adds
  `PROTEON_PDB_LIST` support so 50K runs can use a pre-filtered
  protein-only list. Also adds the v0.2.0 universal env var synonyms
  (`PROTEON_CORPUS_DIR`, `PROTEON_OUTPUT_DIR`,
  `PROTEON_AMBER_ORACLE_OUT`) and resume-from-existing-OUT logic to
  match the CHARMM oracle. Unblocks the AMBER96 vs OpenMM 50K
  extension on monster3 (#42).
- `tests/test_evident_runner_contract.py` extended with v0.2.0
  contract checks for `amber96_oracle.py` (env synonym + legacy-wins
  precedence) — same shape as the existing CHARMM tests.

### Added

- **v0.2.0 data-mount contract** — every release-tier oracle runner now
  reads its corpus directory and output directory from
  `PROTEON_CORPUS_DIR` and `PROTEON_OUTPUT_DIR` environment variables.
  The EVIDENT image entrypoint auto-exports them when the well-known
  `/data/pdbs` and `/data/out` bind mounts exist, so the golden replay
  becomes:
  ```bash
  docker run --rm \
    -v $(pwd)/pdbs:/data/pdbs \
    -v $(pwd)/out:/data/out \
    ghcr.io/thegreatherrlebert/proteon-evident:<tag> \
    replay <claim-id>
  ```
  No `-e` flags required. Closes the v0.1.x unreplayability gap (#38)
  where runners hardcoded monster3-only paths and couldn't be replayed
  from the image alone.
- `evident/CAPTURE_SCHEMA.md` gains a "Running in containers" section
  formalising the contract: which env vars exist, which paths take
  precedence, and what the runner authoring rule is. Includes an
  Apptainer / SLURM example for HPC replayability.
- `tests/test_evident_runner_contract.py` guards every release-tier
  runner against contract regression — pure path-resolution check,
  no oracle calls, runs in milliseconds under the existing Python
  test job.
- `USALIGN_BIN` env var on the SASA runner (`validation/run_validation.py`)
  for analogous reasons — image vendors USAlign at
  `/usr/local/bin/USalign`, source-tree dev keeps the
  `/scratch/TMAlign/USAlign/` default.

### Changed

- The four fold-preservation runners
  (`validation/tm_fold_preservation{,_amber,_openmm,_openmm_amber}.py`)
  no longer hardcode
  `/globalscratch/dateschn/proteon-benchmark/pdbs_50k`. Existing monster3
  invocations stay green (the legacy paths are now the unset-env
  fallback); the same scripts run cleanly inside the EVIDENT image
  against any bind-mounted corpus.
- `validation/fold_preservation/join_fold_preservation.py` reads its
  per-side JSONLs from `PROTEON_OUTPUT_DIR` (matching where the runners
  wrote them) when set, falls back to the historical
  `validation/fold_preservation/` location otherwise.
- `validation/charmm19_eef1_ball_oracle.py` accepts both the legacy
  `PROTEON_PDB_DIR` / `PROTEON_CHARMM_ORACLE_OUT` env vars and the new
  universal `PROTEON_CORPUS_DIR` / `PROTEON_OUTPUT_DIR` synonyms,
  legacy first.
- **Skip-missing-atoms fix extended to all PDBFixer-using oracle runners**
  (#48, follow-up to PR #47). The `addMissingAtoms()` deadlock that
  bottlenecked the v0.1.3 50K corpus oracle isn't unique to CHARMM — every
  runner that preprocesses wwPDB inputs through PDBFixer hits it. This
  applies the PR #47 skip pattern to `validation/amber96_oracle.py`,
  `validation/amber96_oracle_triangulate.py`,
  `validation/amber96_obc_oracle.py`,
  `validation/tm_fold_preservation_openmm.py`,
  `validation/tm_fold_preservation_openmm_amber.py`, and
  `validation/diag_obc_params.py`. Each runner detects missing heavy
  atoms via `fixer.findMissingAtoms()` and skips rather than invoking
  the deadlocking `fixer.addMissingAtoms()`. Comparison surface narrows
  to "well-resolved wwPDB" — the same population the v0.1.4 CHARMM
  corpus oracle adopted, and the more defensible scientific scope
  (modeled-back atoms have ad-hoc geometry). Pre-empts v0.2.0 50K
  extensions (#42) from re-discovering the same 79%-timeout regression
  class.

### Compatibility

All existing monster3 batch invocations of the release-tier runners
keep working without any change. The contract is additive: if you set
`PROTEON_CORPUS_DIR` / `PROTEON_OUTPUT_DIR`, runners use them; if you
don't, runners use the same paths they always did.

## [0.1.4] — 2026-05-04

Second EVIDENT release. Trust pyramid completed: the dropped
fold-preservation claims (proteon CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2,
proteon AMBER96 vs OpenMM AMBER96) are back, fully wired with
per-claim renderers. The 50K corpus oracle re-runs cleanly under
`pebble.ProcessPool` with **n_ok = 5,309 / 44,210** vs the v0.1.3
bundle's 807 — a 6.5× improvement on the same population. Manifest
trimmed from 20 → 18 force-of-evidence claims, then re-introduced the
two fold-preservation claims, landing at 20 release-tier claims with
real evidence.

### Added

- **Fold-preservation claims** (#50) — both proteon force fields are
  gated against OpenMM minimization on the same 1000-PDB random
  sample at TM-score level:
  - `proteon-charmm19-fold-preservation-vs-openmm-release-1k-pdbs`:
    proteon CHARMM19+EEF1 median TM=0.9944, OpenMM CHARMM36+OBC2
    median TM=0.9991, median tm_diff +0.0040, n_ok=886/1000.
  - `proteon-amber96-fold-preservation-vs-openmm-release-1k-pdbs`:
    proteon AMBER96 median TM=0.9959, OpenMM AMBER96 median
    TM=0.9992, median tm_diff +0.0028, n_ok=864/1000. Cleanest
    cross-implementation oracle proteon ships — both arms claim
    AMBER96, so the diff is purely implementation, not parameter
    set.
  - `validation/fold_preservation/join_fold_preservation.py` joins
    proteon-side and OpenMM-side per-PDB JSONLs into the canonical
    artifact `{pdb, n_ca, proteon: {...}, openmm: {...}, tm_diff,
    rmsd_diff_A}`.
  - `validation/report/render_fold_preservation.py` — per-claim HTML
    with TM distributions per side, tm_diff histogram against the
    claim's tolerance band, scatter, and top-20 outliers.
- **Three-tier capture schema** (#46) — `evident/CAPTURE_SCHEMA.md`
  defines `minimal` / `extended` / `full` capture levels for oracle
  runners, the per-PDB filesystem layout, the `hardware.json`
  schema, and the renderer contract. `PROTEON_CAPTURE_LEVEL` env
  controls the level.
- `validation/charmm19_eef1_ball_oracle.py` migrates from
  `concurrent.futures.ProcessPoolExecutor` to `pebble.ProcessPool`
  for true per-task subprocess isolation. Closes the
  `BrokenProcessPool` cascade that limited the v0.1.3 50K corpus
  run's `n_ok` to 807 of 44,210 attempted. (#44)
- New env knob `TASK_TIMEOUT_S` (default 60 s) on the corpus oracle
  runner — terminates hung BALL setup tasks individually instead of
  stalling the whole run.
- `pebble` added to both EVIDENT image variants' pip oracle install
  layer.

### Changed

- **Population definition for the 50K oracle is now "well-resolved
  wwPDB"** (#47). The runner skips PDBs with missing heavy atoms
  instead of running `PDBFixer.addMissingAtoms()`, which hangs
  deterministically on certain inputs. The defensible population
  shifts from "everything PDBFixer can repair" → "structures
  resolved well enough to score directly". 18,912 / 44,210 records
  on the v0.1.4 run skip for this reason — they are explicitly
  out-of-population, not failures.
- mkdssp build chain in both `evident/Dockerfile` and
  `evident/Dockerfile.cuda` is now version-pinned: `libmcfp v1.4.2`,
  `libcifpp v10.0.3`, `dssp v4.6.1`. Earlier `--depth 1 main` clones
  caused three breaking-upstream incidents during v1 prep. ARGs are
  exposed (`LIBMCFP_VERSION`, `LIBCIFPP_VERSION`, `DSSP_VERSION`)
  for downstream override. Closes #13.
- `evident/Dockerfile.cuda` switched from `jammy + 2 PPAs` to
  `noble + 0 PPAs` (#45) — eliminates the Launchpad/PPA flake class
  that broke 4 of the v0.1.3 cuda image builds.
- `libglib2.0-0` added to both runtime stages (#45). Apptainer smoke
  on monster3 caught `import ball` failing on
  `libglib-2.0.so.0: cannot open shared object file` in the cuda
  image; same fix applied to slim defensively.

### Removed

- 2 release-tier claims trimmed from `evident/evident.yaml` (#49)
  that asserted evidence we don't currently have:
  - `forcefield_amber_openmm` — removed; AMBER96 vs OpenMM AMBER96
    is now carried by the fold-preservation pair (#50).
  - `msa.yaml` — removed; MSA feature parity is held in unit-test
    scope, not framework-tier.

### Fixed

- 50K corpus run pass rate climbs from 807 / 44,210 (v0.1.3) to
  5,309 / 44,210 (v0.1.4) on the same input population. The lift
  is the sum of: pebble per-task isolation killing the
  `BrokenProcessPool` cascade, the `addMissingAtoms` hang fix,
  and the 60 s task timeout terminating the long tail of stuck
  BALL setups.

## [0.1.3] — 2026-05-03

First EVIDENT release: end-to-end claim / replay / report
architecture with sha256-pinned audit trail.

### Added

- **EVIDENT framework end-to-end** (Phases 1–4 across PRs
  #28 / #30 / #11 / #31):
  - `evident/scripts/lock_release_replays.py` walks every claim
    YAML under `evident/claims/`, sha256-pins each artifact, and
    emits an immutable bundle at `evident/reports/<tag>/`.
    Tier-aware: `pytest console output` style claims correctly
    surface as `ci-only` in the manifest, not `missing`.
  - Lock-time environment snapshot embedded in every manifest:
    Python version, platform, full `pip freeze`, sha256 of
    `cargo metadata`. Closes #36.
  - `evident/scripts/build_index.py` — multi-release aggregator
    served at `<pages-url>/evident/reports/`.
  - `evident/scripts/cut_release.sh` — one-command flow: scp
    artifacts from monster3 → lock bundle → print git tag steps.
  - Per-claim HTML renderers for the corpus oracle, SASA-vs-Biopython
    1k, and 50K battle test. Reports embed plots as base64,
    self-contained, no CDN dependency.
- **Reproducibility images** on GHCR
  (`ghcr.io/thegreatherrlebert/proteon-evident:v0.1.3`):
  - `:slim` — Debian trixie + Python 3.13. Bundles proteon,
    `ball-py`, `openmm`, `pdbfixer`, `biopython`, `gemmi`,
    `freesasa`, `pydssp`, `gromacs`, `mmseqs2`, `mkdssp`,
    `USalign`, `reduce`.
  - `:cuda` — nvidia/cuda 12.8.2 + Ubuntu noble + Python 3.12.
    Same payload, GPU-enabled `proteon-connector` + `cudarc`.
  - Entrypoint dispatches `replay <claim-id>`,
    `render <release-tag>`, plus pass-through to the vendored
    `evident` CLI.
- **CHARMM19+EEF1 oracle** (PRs #16–#26):
  - All seven force-field components (bond_stretch, angle_bend,
    proper_torsion, improper_torsion, vdw, electrostatic,
    EEF1 solvation) gated against BALL on crambin within
    per-component bands.
  - 50k-ready corpus runner with chunked-pool segfault isolation
    (later replaced by pebble in v0.1.4) and a protein-only
    filter that drops nucleic-acid PDBs the force field can't
    parametrise.
  - Distance-dependent dielectric (ε ∝ r) — the canonical
    CHARMM19 convention — replaces the prior constant-ε
    implementation.
  - PHE/TYR aromatic-ring para-diagonal LJ exclusion port from
    BALL's `charmmNonBonded.C:547-565`.
- **CHARMM19+EEF1 corpus oracle**:
  - 1k-PDB BALL oracle on the curated `validation/pdbs/`. Released
    alongside the unit oracle as `proteon-charmm19-vs-ball-corpus-1k-pdbs`.
  - 50K-PDB BALL oracle on a random wwPDB sample, run on monster3.
    Headline `n_ok=807 of 44 210 attempted` reflects the
    `BrokenProcessPool` cascade; per-component bands on the 807
    successful records match the unit oracle (median 0.05%
    bond_stretch, 0.001% torsion, 0.02% electrostatic, 1.62%
    EEF1 solvation, 0.41% improper). Cascade noise framed
    honestly in the claim's `failure_modes`; pebble migration
    is the v0.1.4 follow-up.
- **EVIDENT manifest** (`evident/evident.yaml`) wires 20 claims
  across all proteon subsystems (force fields, alignment, SASA,
  DSSP, hydrogens, GB OBC, supervision, MSA, I/O, pipeline batch).
  Each claim names its oracle, tolerance, replay command, and
  artifact path.

### Changed

- 11 of 13 CI-tier oracle claims now pass against external
  references (BALL, Biopython, USAlign, pydssp, reduce). The two
  pending — `proteon-amber96-vs-openmm-release-1k-pdbs` and
  `proteon-msa-vs-mmseqs2-research` — have monster3-side artifacts
  not yet mirrored.

### Removed

- 6 low-signal claims trimmed in PR #43 (`acceleration_paths`,
  `gpu_cpu_parity`, `forcefield_charmm19_internal`,
  `sasa_freesasa`, both fold-preservation entries). The two
  fold-preservation claims drop until #37 (PDBFixer pre-pass on
  the GROMACS runner) lands real artifacts. Manifest goes from
  26 → 20 claims; every "missing" row is now a real coverage gap,
  not a documentation artefact.

### Known gaps tracked for follow-up

- **#37** — fold-preservation GROMACS artifact regen needs
  PDBFixer pre-pass + monster3 GROMACS install.
- **#42** — v0.2.0 vision: every release-tier claim at 50K scale
  across all oracles.
- **#44** — pebble runner migration (resolved in [Unreleased]).

## [0.1.2] — 2026-04-24

First clean-room MIT release after the 2026-04-24 third-party
lineage audit.

### Changed

- Phase 3 hydrogen placer in `proteon-connector/src/add_hydrogens.rs`
  rewritten clean-room from standard crystallographic geometry; the
  prior dispatcher was structurally derived from BALL's
  `AddHydrogenProcessor` (LGPL-2.1).
- `reconstruct.rs` reattributed to its actual upstream: MIT
  `BiochemicalAlgorithms.jl`, not LGPL BALL C++.
- `forcefield/md.rs` and `forcefield/gb_obc.rs` cite primary
  literature / OpenMM's MIT Reference Platform.

### Added

- `THIRD_PARTY_NOTICES.md` consolidates verbatim upstream license
  texts for TM-align/US-align, MMseqs2, BiochemicalAlgorithms.jl,
  OpenMM, and pdbtbx.

### Performance

- SASA GPU auto-dispatch threshold raised from 500 → 10 000 atoms.
  Small-protein batch throughput recovered ~10× against the
  pre-2026-04-12 regression (monster3 5K: 18.6/s → 192.5/s; numerical
  output bit-identical).

### Benchmark

- `benchmark/run_benchmark.py` now tracks `n_negative_energy`
  alongside `n_converged`. After the 2026-04-11 CHARMM19 heavy-atom
  relaxation default, negative final energy is the documented
  correctness invariant; `converged` is preserved for back-compat.

### No public API changes.

## [0.1.1] — 2026-04-18

First clean PyPI release with populated package metadata.

### Added

- `pip install proteon` installs the Pythonic wrapper; the
  PyO3 `proteon-connector` is pulled in transitively.
- Wheels published for Linux x86_64 / aarch64, macOS x86_64 /
  aarch64, and Windows x86_64.

### Validation highlights at the time of release

- AMBER96 matches OpenMM to within 0.2% on every energy component
  at NoCutoff.
- CHARMM19+EEF1 pipeline: 50K random PDBs at 99.1% end-to-end
  success in 3.5h on RTX 5090.
- TM-align port: 0.003 median TM-score drift from the reference
  C++ USalign across 4 656 pairs.
- SASA: 0.17% median deviation vs Biopython on 1 000 structures.

### No runtime code changes since 0.1.0.

## [0.1.0]

Initial PyPI publication (one-shot to claim the project name; pages
rendered with empty descriptions, superseded by 0.1.1).

[Unreleased]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/theGreatHerrLebert/proteon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/theGreatHerrLebert/proteon/releases/tag/v0.1.0
