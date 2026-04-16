# Oracle Testing in Ferritin

**How we verify correctness, and why this is the thing that lets us move fast.**

---

## The principle

Ferritin implements well-trodden algorithms (force fields, MD, alignment,
structure search, SASA, DSSP, supervision-tensor export). For each of these,
at least one independent, battle-tested implementation already exists
somewhere else. We treat those implementations as **oracles**: ground-truth
references that our own code must agree with to a specified tolerance before
we're allowed to believe it.

The reason this works as a *velocity* strategy, not just a correctness one:

1. **We never ship a component without an oracle behind it.** Every
   scientific claim in the codebase — "ferritin CHARMM19+EEF1 preserves
   folds with median TM=0.9945", "AMBER96 matches OpenMM to 0.2% at
   NoCutoff", "GPU matches CPU to 1e-11 on crambin for OBC GB",
   "50K-PDB battle test passed at 99.1% correct in 3.5h on RTX 5090" —
   points at a reproducible oracle script that anyone can rerun.
2. **When an oracle disagrees, the oracle is right until proven otherwise.**
   The default hypothesis is that the bug is in ferritin. Only after a
   careful read of the oracle's implementation do we consider that *both*
   tools could be correct under different conventions (then we document
   the convention gap explicitly — see *When the oracle is also wrong*
   below).
3. **Oracle diffs are diagnostic, not just pass/fail.** The
   `validation/report.html` artifact is built to make disagreements
   actionable: per-structure deltas, component breakdowns, timing,
   failure classifications. That's why a structure prep bug in 2026-04
   got diagnosed from a histogram of EEF1 energy residuals rather than by
   reading code.
4. **Oracle-driven development turns debates into experiments.** Instead
   of "does the nonbonded cutoff matter?" we write a triangulation
   oracle that runs ferritin + OpenMM at NoCutoff, 10 Å, and 15 Å on the
   same inputs, and compare. Three data points beat a Slack thread
   every time.

---

## Oracle vs. fixture

These are different and the distinction matters.

- An **oracle** is a live, externally-implemented tool we call at test
  time. BALL (Julia), OpenMM, Foldseek, MMseqs2, original TMalign C++,
  Biopython, Gemmi, FreeSASA, DSSP. Recomputation is part of the test.
- A **golden fixture** is a captured value frozen into the repo. Useful
  for CI stability (oracles are slow, sometimes flaky to install), but
  it can go stale silently. Every fixture should carry a header
  documenting which oracle produced it, which version, which inputs,
  and what date.

Rule: a new component lands with at least one live oracle test. Fixtures
are only acceptable as a second, faster layer on top of that.

---

## Domain-by-domain oracle inventory

### I/O and hierarchy

**Oracles:** Biopython, Gemmi.
**Scope:** atom counts, chain counts, residue counts, coordinates, chain
ordering, altloc/insertion-code handling, model count (NMR), HETATM flag.

- `tests/oracle/conftest.py` extracts a common `StructureSummary` from
  ferritin + Biopython + Gemmi using the same struct.
- `tests/oracle/test_io_oracle.py` cross-checks all three on standard
  and edge-case PDBs (`1ubq`, `4hhb` multi-chain, `1crn`,
  `models.pdb` multi-model, `insertion_codes.pdb` altloc + insertion).

**Tolerance:** atom counts match exactly on standard files; edge cases
are tested separately and tools may legitimately disagree (e.g.
hetatm classification).

### Alignment (TM-align, US-align)

**Oracle:** original C++ TMalign (`TMAlign/TMalign.cpp`) and US-align
(`USAlign/`), treated as reference implementations.
**Scope:** TM-score, RMSD, aligned-length, rotation matrix.

- `tests/oracle/test_tmscore_oracle.py` — per-file comparison.
- `ferritin-align/` Rust core is tested against the C++ binary on the
  same inputs.

**Tolerance:** TM-scores match to 4–5 decimal places, not exactly. The
C++ is compiled with `-ffast-math` (reorders FP ops); Rust can't.
Key constants are preserved exactly across the port (Kabsch `epsilon=1e-8`,
`tol=0.01`, `sqrt3=1.73205080756888`), and the `d0` formula uses
`powf(1.0/3.0)` instead of `cbrt()` to match C++ `-ffast-math` behaviour.

### Force fields (AMBER96, CHARMM19+EEF1)

**Oracles:**
- **BALL Julia** (`BiochemicalAlgorithms.jl`) — per-component energies
  on crambin without reconstruction. Same atom set fed to both engines.
- **OpenMM** — AMBER96 and AMBER96+OBC single-point energies on 1000
  random PDBs, using PDBFixer-prepared inputs so both tools see
  identical hydrogens.

**Scope:** bond stretch, angle bend, torsion, improper torsion, vdW,
electrostatic, solvation (EEF1 for CHARMM, OBC for AMBER).

- `tests/oracle/test_ball_energy.py` — fast crambin oracle, runs in CI.
- `validation/amber96_oracle.py` — OpenMM component oracle, 1000 PDBs.
- `validation/amber96_obc_oracle.py` — OpenMM AMBER96+OBC oracle.
- `validation/amber96_oracle_triangulate.py` — three-way: ferritin,
  BALL, OpenMM on the same inputs to localize which tool is the outlier.

**Tolerance matrix:**

| Component | Target | Why this bound |
|-----------|--------|----------------|
| Bond stretch | < 0.1% | Deterministic, Hooke's law — any gap means a bond-type error |
| Angle bend | < 0.1% | Same as bonds |
| Torsion | < 0.1% | Same |
| VdW | < 0.1% at NoCutoff | Cutoff adds ~1.4% at 15 Å — that's policy, not bug |
| Electrostatic | < 0.1% at NoCutoff | Same cutoff caveat |
| Improper | ferritin ≥ BALL, OpenMM ≤ 0.5% | BALL uses single-wildcard improper matching; AMBER spec requires double-wildcard (e.g. `* * N H` for amide-plane). BALL therefore misses ~100 amide-plane impropers on crambin. Ferritin: 125 / 8.1 kJ/mol; BALL: 10 / 2.08 kJ/mol; OpenMM amber96 matches ferritin. Test asserts ferritin ≥ BALL as regression guard. |
| Total energy | < 1% at NoCutoff | Tightest contract — everything else rolls up here |
| GB solvation | < 5% | OBC is an analytical approximation; ferritin matches OpenMM to ≤5% GB / ≤1% total on crambin (Phase B) |

**Critical rule: NoCutoff for oracle-grade comparison.** Production
ferritin uses a 15 Å nonbonded cutoff with switching (a perf-vs-accuracy
policy choice). That produces a 1.4% total-energy gap against
OpenMM NoCutoff on crambin. This is **not a bug**, it's the cost of
the cutoff. Oracle tests set `nonbonded_cutoff=1e6` to disable the
cutoff and isolate the force-field math from the cutoff approximation.
This separation is enforced by a runtime warning: compute_energy
with default cutoff prints a one-line notice explaining what to pass
for oracle-grade comparison.

### Cross-path parity (accelerated implementations)

**Oracle:** ferritin's own slow, canonical path.
**Scope:** any code path with an accelerated variant — NBL
(neighbor-list), SIMD, GPU (CUDA), rayon parallel iterators.

Rule: every accelerated path must have a parity test against the
brute-force path on every energy component, parametrized over every
supported force field. Silent divergence on an accelerated path is
worse than a visible failure on the slow path, because production
throughput runs through the accelerated path.

Historical catches:
- The CHARMM EEF1 sign bug (2026-04-11) was a case where the brute-force
  path was correct and the NBL path skipped EEF1 entirely. Fixed in
  commits `c6db9f9` + `ea297b4` with cross-path regression guards
  added at the same time.
- GPU CHARMM19+EEF1 matches CPU to 1e-11 on crambin (Phase B, `project_obc_gb_phase_b_done`); the parity test is the gate for shipping the `cuda` feature.

### Structure preparation

**Oracles:** PDBFixer (OpenMM ecosystem) for hydrogen placement;
GROMACS `gmx pdb2gmx` for CHARMM polar-H conventions; BALL for
united-atom handling.
**Scope:** which hydrogens to add (all vs. polar-only), which atom
types to absorb into united carbons, post-minimization energy.

- `validation/gmx_fold_preservation/` — GROMACS-minimized structures
  as an independent reference for fold preservation.
- CHARMM19 uses polar-only H placement; non-polar C–H must NOT be
  placed because CH1E/CH2E/CH3E absorb them into united carbon types.
  BALL and GROMACS agree on this — ferritin's prepare pipeline is
  FF-aware (`batch_prepare` defaults now vary by force field).

### Retrieval and search

**Oracle:** Foldseek (structure) and MMseqs2 (sequence), plus
TM-align ground-truth labels for retrieval benchmarks.
**Scope:** top-k recall, nDCG, per-query misses, alignment time.

- `validation/bench_foldseek.py` — ferritin search vs. Foldseek on
  the same queries.
- `validation/bench_foldseek_retrieval.py` — 3Di-alphabet retrieval
  scored against TM-align truth labels.
- `validation/foldseek_ferritin_5k_50q_union50.report.md` —
  per-query delta table, "where is ferritin worse and why".
- `ferritin-search/tests/oracle_search.rs` — Rust-side oracle
  fixture (MMseqs2-style results).

**Contract:** ferritin doesn't need to beat Foldseek on recall at
this stage; we need to be *explicit* about where we're worse and
why. The 5K/50Q retrieval report has a "Worst ferritin Deltas"
table that's explicitly read each release.

### Supervision tensors (AF2-contract)

**Oracles:**
- **Rust-Python parity** — ferritin's Rust supervision builder vs.
  the Python reference implementation. Both must produce
  byte-identical NPZ for the same input.
- **OpenFold data module** — field-by-field shape/type/content match
  for `aatype`, `all_atom_positions`, `rigidgroups_gt_frames`, etc.

- `tests/test_supervision_rust_parity.py` — pins the parity at commit
  time.
- `packages/ferritin/src/ferritin/supervision_export.py` carries the
  tensor-field list as a tuple; both builders consume it.

### Fold preservation (end-to-end)

**Oracle:** OpenMM CHARMM36+OBC2 minimization as an independent
reference.
**Scope:** TM-score between raw PDB and minimized PDB — does the
force field + minimizer actually preserve the fold?

- 1000 random PDBs, ferritin CHARMM19+EEF1 vs OpenMM CHARMM36+OBC2.
- 2026-04-13 result: ferritin median TM=0.9945, OpenMM median
  TM=0.9991. Ferritin is 30× faster on wall clock and 1.5× more
  tolerant of raw PDBs (OpenMM rejects more inputs). Close enough on
  TM, meaningfully better on throughput and intake robustness.

### SASA, DSSP, hydrogen counts

**Oracles:** FreeSASA, DSSP binary, Biopython.

- `tests/test_sasa.py` — Biopython comparison.
- DSSP is shelled out as a subprocess in some flows;
  `packages/ferritin/src/ferritin/dssp.py` has a native port that's
  cross-checked against the binary.

---

## The report.html

`validation/generate_report.py` (legacy) and `validation/report/build_report.py`
(current) turn per-structure oracle results (JSONL) into a standalone HTML
dashboard.

What goes in:

- Pass/warn/fail counts per test, with colored bars.
- Per-structure rows for anything warn or fail, with the oracle's
  numeric delta and a link back to the source file.
- SASA speedup histogram and relative-diff distribution (to catch
  outliers, not just means).
- Component energy tables with ferritin / oracle / delta / %.
- Top-worst retrieval deltas for search benchmarks.

How we read it:

1. Look at overall pass rate first. A 99.1% pass rate on 50K with 458
   residual failures is actionable; a 70% pass rate is a systemic bug.
2. Bucket the failures by error class. If one class dominates
   (e.g. "ValueError: pass chain_id for multi-chain structures"),
   that's a one-fix win — fix it and run again.
3. For components with mean-level agreement but a long tail, sort by
   absolute delta and look at the worst 10 structures. Usually one
   edge case (terminal residue, altloc, missing sidechain) explains
   most of the tail.
4. For retrieval, read the "Worst ferritin Deltas" table verbatim.
   Those queries are the ones to regression-lock.

Generate locally:

```bash
# Full benchmark + report
python validation/amber96_oracle.py           # writes jsonl to /globalscratch/…
python validation/report/build_report.py      # renders HTML
open validation/report.html
```

---

## Running cadence

| Cadence | What runs | Where |
|---------|-----------|-------|
| Every commit (local) | `cargo test --workspace`, `pytest tests/ -k 'not oracle'` | `test.yml` |
| Every push (CI) | Same plus `pytest tests/oracle/ -v` if enabled | `test.yml` |
| Mon/Wed/Fri 3am UTC | Full oracle suite (BALL, OpenMM, TMalign) | `oracle.yml` |
| Pre-release | 1K and 5K scale runs with report.html | Manual on monster3 |
| Major version | 50K battle test (3.5h on RTX 5090) | Manual, documented |

Scale discipline: the 50K battle test (2026-04-12, `project_gpu_cuda_poc`)
caught prep-stability, cutoff, and OOM issues that never surfaced on
100-PDB runs. Regression surface includes archive-scale runs, not just
unit tests.

---

## When the oracle is also wrong

Oracles are not infallible. Three cases we've hit:

1. **Oracle spec divergence — BALL single-wildcard impropers.** BALL's
   improper-torsion matcher supports only single-wildcard atom-type
   patterns; the AMBER spec requires double-wildcard (e.g. `* * N H`
   for amide-plane impropers). On crambin BALL reports 10 impropers
   / 2.08 kJ/mol; ferritin reports 125 / 8.1 kJ/mol, matching OpenMM
   amber96 to 0.5%. This is a BALL gap against the AMBER spec, not a
   ferritin/BALL convention difference. Documented in
   `test_ball_energy.py`; the test asserts ferritin ≥ BALL as a
   regression guard and stays live until BALL gains double-wildcard
   support. Worth reporting upstream to the BALL maintainers — the
   impact scales linearly with chain length (one missed improper
   per peptide N-H).
2. **Oracle bug** — BALL's Python (SIP) bindings don't build cleanly
   on current Python; for CHARMM oracle work we use `libBALL.so` via
   standalone C++ (`reference_ball_python_dead.md`). The oracle
   pipeline had to be rewritten, not the code under test.
3. **Different defaults** — AMBER96 oracle work (2026-04-13) surfaced
   that the default 15 Å cutoff creates a 1.4% total-energy gap vs
   OpenMM NoCutoff. That's the cutoff policy, not a bug — but the
   oracle script has to *force* NoCutoff to separate the concerns.
   A runtime warning now tells the user to pass `nonbonded_cutoff=1e6`
   for oracle-grade comparison.

Decision tree when oracle disagrees:

1. Read the oracle's source for the component in question.
2. Can both implementations produce the same numeric result under some
   shared convention? If yes → fix ferritin to match.
3. If no (genuine convention difference), write down which convention
   ferritin uses, why, and the numerical gap it creates. Pin it as a
   test that asserts the *expected* gap (not equality).
4. If the oracle itself is buggy (rare), file an issue upstream,
   switch to a different oracle (triangulate), and leave ferritin
   correct.

---

## Landmark oracle catches

Listed so new contributors see what "oracle testing paid off" looks like:

- **CHARMM EEF1 sign bug (2026-04-11).** NBL code path skipped EEF1
  entirely; missing 1-2/1-3 exclusions on the brute-force path. Both
  caught by oracle diff against BALL — would have been silent
  otherwise because total energy looked reasonable. Fixed `c6db9f9` +
  `ea297b4`.
- **AMBER96 three-step fix (2026-04-13).** OpenMM oracle showed 3%
  gap on crambin. Root cause: H-name aliases, double-wildcard
  impropers, cutoff policy mismatch. All three fixes driven by the
  oracle script. 218/218 invariants + 80/80 Rust tests after.
- **OBC GB integration (2026-04-14 → 2026-04-15).** Phase A landed
  a deliberately-failing oracle to pin the contract: 15% AMBER96
  total-energy gap == exactly the missing GB term (−1273 kJ/mol on
  crambin). Phase B filled in the math; energy within 5% GB / 1%
  total, forces FD-verified, GPU matches CPU to 1e-11. The
  contract-first pattern made the 4–6h math session tractable.
- **AMBER96 n_threads trap.** `n_threads=0` silently ran serial
  (not all-cores). Caught by a throughput oracle that expected
  multi-core timings.
- **CHARMM19 polar-H radii**. Positive vdW on raw pre-H PDBs looked
  like a bug but wasn't — CH1E/CH2E/CH3E carry inflated radii to
  absorb implicit H. Oracle against GROMACS `pdb2gmx` clarified
  that positive vdW pre-minimization is the expected state.

---

## Anti-patterns

- **"This oracle is flaky, let's skip it."** No. Either fix the oracle
  setup or triangulate with a second oracle; don't ship without one.
- **Golden fixtures with no header.** A fixture without a
  producer/version/date comment rots silently. If you capture one,
  annotate it.
- **Mean-only reporting.** Mean agreement hides long tails.
  Every oracle result in `report.html` carries per-structure
  distributions, not just means.
- **Oracle-only passing.** A component that passes its oracle but
  has no fast in-process test doesn't land — oracles are slow and
  CI can't run them on every commit. Oracle + fast test, both
  required.
- **"Bobby-car" benchmark claims.** No "Nx faster than Biopython"
  comparisons against tools that aren't peers. Report peer-tool
  comparisons (Foldseek, OpenMM, GROMACS) and absolute throughput;
  leave Biopython out unless it's the actual default in the
  downstream pipeline.

---

## Pointers

- Oracle scripts (scale runs): `validation/*_oracle*.py`, `validation/bench_*.py`
- Oracle pytest suite: `tests/oracle/`
- Report generator: `validation/report/build_report.py`
- Rendered report: `validation/report.html`
- CI workflow: `.github/workflows/oracle.yml` (Mon/Wed/Fri 3am UTC)
- Reliability philosophy (broader): `RELIABILITY_ROADMAP.md`
- Cross-path parity testing: the principle is enforced per-PR on any
  accelerated code (NBL, SIMD, GPU, rayon).
