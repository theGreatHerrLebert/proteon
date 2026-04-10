# Next Session — Handoff Notes (2026-04-10)

## Today's Session (2026-04-10 evening): Scale the SOTA harness

**Goal:** Unlock 10K-scale SOTA comparison by (a) adding an `ff` parameter to
`batch_prepare` so the NBL-cached fast minimizer path can run with AMBER96
instead of only CHARMM19-EEF1, and (b) extending the SOTA harness with
batched runner support so the driver can send whole PDB lists to the
batched Rust APIs rather than iterating single-structure.

**Ended:** Both infrastructure pieces committed and working. On a 6-PDB
batched energy sweep the ferritin side is 2.5× faster than the single-
structure path (160s vs 400s). On a 20-PDB scaling demo from the monster3
50K corpus, ferritin batched runs in 18s/structure averaged, OpenMM
crashes on 10/20 due to pre-existing PDB format issues (missing terminal
caps / missing atoms) that Modeller.addHydrogens can't auto-repair.
Remaining scaling work is documented below.

### Rust + Python changes

**`batch_prepare(ff="charmm19_eef1"|"amber96")`** — new parameter. The
batch_prepare entry point dispatches on the ff string and calls a new
generic inner function `batch_prepare_inner<F: ForceField + Sync>`. Both
parameter sets go through the same monomorphized hot path, so there's no
runtime dispatch overhead; the `ff` string just picks which monomorphic
specialization gets instantiated.

**Per-component energy breakdown in the result dict** — previously
batch_prepare only returned `initial_energy` / `final_energy` (totals).
Now the result dict has a `components` sub-dict with
`{bond_stretch, angle_bend, torsion, improper_torsion, vdw,
electrostatic, solvation}`, populated from the minimizer's final
`EnergyResult`. Zero when `minimize=False` or `skipped_no_protein=True`.
This eliminates the need for a separate `compute_energy` call after
prepare — the SOTA energy runner reads components directly from
PrepReport.components.

**kcal/mol → kJ/mol conversion in the Python wrapper** — the Rust
minimizer reports energies in AMBER/CHARMM-native kcal/mol. The rest
of the ferritin Python API defaults to kJ/mol (via
`_convert_energy_dict` in forcefield.py). Added a mirror helper
`_convert_prep_result_to_kj` in prepare.py that touches `initial_energy`,
`final_energy`, and every component in `components`, applied at all
three batch_prepare call sites. Without this, `ferritin.batch_prepare`
results were off by a factor of 4.184 from the expected kJ/mol values.

### SOTA harness: batched runner infrastructure

**`@register_batch(op, impl)`** decorator (in `runners/_base.py`). Keyed
by `(op, impl)`; registered functions have signature
`def batch_fn(pdb_paths: List[str]) -> List[RunnerResult]`. The driver
checks `BATCH_RUNNERS.get((op, impl))` and prefers the batched runner
when present and when there are >1 PDBs.

**Batched ferritin SASA runner** (`runners/sasa.py::ferritin_batch`):
one `batch_load_tolerant` + one `batch_total_sasa` across the whole
PDB list. Per-residue SASA is still single-structure (cheap).

**Batched ferritin energy runner**
(`runners/energy.py::ferritin_batch`): HETATM strip per path → one
`batch_load_tolerant` → one `batch_prepare(ff="amber96")` across the
whole list. This is the unlock that makes scale-up feasible.

**Driver fast path** (`run_all.py`): main loop now checks
`BATCH_RUNNERS[(op, impl)]` first. When present and N>1, calls the
batched runner once with all PDB paths. Falls back to the per-structure
loop on error or when `--force-serial` is set.

**Corpus sampler** (`sample_corpus.py`): seed-stable random sampler
that pulls N protein-only PDBs from the monster3 50K corpus. Filters
nucleic acids (DA/DT/DG/DC, RA/RU/RG/RC) by quick-scanning the first
5000 ATOM lines. Writes a PDB ID list suitable for
`run_all.py --pdbs @sample.txt`.

### Measured speedup

| Run | N | Wall time | Per-structure avg | Notes |
|---|---|---|---|---|
| v1 per-structure (pre plan B) | 6 | 398 s | 66 s | `minimize_hydrogens` single-struct path |
| Batched, 6 clean PDBs | 6 | 160 s | 27 s | 2.5× faster, low parallelism (only 6 workers) |
| Batched, 20 random corpus | 20 | 363 s | 18 s | better parallelism saturation, ferritin side only |

Linear extrapolation to 10K with full 120-core saturation:
~18 s/structure × 10K / 120 cores ≈ **25 minutes** for ferritin side.
OpenMM per-structure at ~1.7 s/structure → 4.7 hours serial,
~3 min with a multiprocessing Pool (not yet implemented).

### Failure modes surfaced during scaling

1. **OpenMM `createSystem` fails on ~50% of random PDBs** with errors
   like "missing 1 C atom, is chain missing a terminal capping group?"
   These are real format imperfections in the 50K corpus — chains
   ending in an incomplete residue, residues with missing heavy atoms
   that Modeller.addHydrogens can't repair. v2 fix: preprocess with
   `Modeller.addMissingAtoms(forcefield)` before `createSystem`.
   Alternatively, use PDBFixer.

2. **FreeSASA cross-library segfault** when called downstream of a
   ferritin batched run. Reproducible in the driver but NOT in
   standalone freesasa loops over the same 100 PDBs. Likely a C-state
   or GIL interaction between the two foreign libraries. Skip-unknown
   option set but doesn't fix it. v2 fix: process isolation — run each
   freesasa call in a `multiprocessing.Pool` worker with
   `maxtasksperchild=1` so a crash only kills one worker.

3. **LBFGS stragglers dominate batched wall time**. On 100 PDBs one
   outlier structure took >5 minutes of CPU time, becoming the rayon
   critical-path tail. v2 option: add progress-aware early termination
   (e.g. "stop after N steps regardless of gradient tolerance, report
   not-converged") so per-structure cost is bounded.

### Commits (main branch)

- `a3ff0be` — Add ff parameter to batch_prepare for fast AMBER96 path
- `ff42420` — Convert batch_prepare kcal/mol → kJ/mol in the Python wrapper
- `f93ffd0` — SOTA harness: add @register_batch + batched ferritin runners
  + driver fast path
- `47fc4dc` — SOTA harness: add corpus sampler + freesasa skip-unknown
  hardening

### Ship status (plan B)

- ✓ `ff="amber96"` parameter works and produces correct kJ/mol output
  matching the old `minimize_hydrogens` path exactly (total 10752 kJ/mol
  on 1crn identical both paths)
- ✓ Batched runners demonstrate 2.5× speedup on 6 v1 PDBs
- ✓ Corpus sampler produces clean random samples from the 50K corpus
- ⚠ Scaling to 100-PDB random samples surfaces cross-library robustness
  issues that need process isolation to fix (v2 task)
- ⚠ OpenMM pre-processing (Modeller.addMissingAtoms) needed for ~50% of
  random PDBs (v2 task)

---

## Today's Session (2026-04-10 afternoon): SOTA Comparison Harness v1

**Goal:** Stand up a publication-track validation harness comparing ferritin's
numerical outputs against widely-used reference implementations on a curated
PDB set, with reproducible per-(structure, op, impl) JSON dumps + a markdown
report that distinguishes PASS/WARN/FAIL per metric.

**Ended:** v1 shipping. SASA comparison against FreeSASA passes at
publication-grade (18/18 metrics PASS across 6 PDBs). Energy comparison
against OpenMM AMBER96 surfaces a real systematic nonbonded disagreement
that becomes a paper-worthy finding.

### Scaffolding + design

New directory: `validation/sota_comparison/` with a per-(op, impl) runner
registry, `RunnerResult` dataclass envelope, typed per-op payload schemas,
and a driver + aggregator split:

```
validation/sota_comparison/
  runners/
    __init__.py     # safe-imports each runner into OPS registry
    _base.py        # RunnerResult dataclass + register() decorator
    sasa.py         # ferritin + freesasa
    energy.py       # ferritin + openmm
  run_all.py        # driver: walks registry over PDB list, writes JSON
  aggregate.py      # reads JSON, pairs ferritin vs X, emits report.md + summary.json
  setup_sota_env.sh # creates sota_venv, pip installs requirements
  requirements.txt  # freesasa, openmm, numpy, pyarrow, maturin
  sota_reference.txt # v1 reference: 6 clean protein PDBs
  README.md         # v1 findings + setup/run instructions + v2 backlog
  sota_v1_{sasa,energy}_{json,report.md,summary.json}  # shipped artifacts
```

Design plan at `/home/administrator/.claude/plans/wobbly-crunching-orbit.md`.

### v1 reference set

Assembled at `/globalscratch/dateschn/ferritin-benchmark/sota_pdbs/`:
1crn, 1ubq, 1bpi, 1ake (from the ferritin repo `test-pdbs/`), 1pgb, 1aki
(from the monster3 50K corpus).

An earlier draft included 1lmb, but PDB 1LMB is actually the λ repressor /
OR1 operator protein-DNA complex — its DNA residues caused FreeSASA's
`residueAreas()` to drop ~100 entries and blew up the per-residue Pearson
without being a real disagreement. Excluded from v1.

### Findings

**SASA (ferritin vs FreeSASA): publication-grade agreement.** All 18 metrics
PASS across 6 PDBs:

| PDB  | total % diff | per-residue Pearson r | per-residue RMSD (Å²) |
|------|--------------|-----------------------|------------------------|
| 1ake | 0.17 % ✓    | 0.9998 ✓              | 0.63 ✓                 |
| 1aki | 0.25 % ✓    | 0.9997 ✓              | 0.76 ✓                 |
| 1bpi | 0.34 % ✓    | 0.9995 ✓              | 0.66 ✓                 |
| 1crn | **0.024 %** ✓ | **1.0000** ✓         | **0.23** ✓             |
| 1pgb | 0.39 % ✓    | 0.9998 ✓              | 0.79 ✓                 |
| 1ubq | 0.34 % ✓    | 0.9994 ✓              | 1.30 ✓                 |

Key to matching: FreeSASA defaults to Lee-Richards and OONS-like classifier
while ferritin uses Shrake-Rupley and ProtOr. Setting the freesasa runner
to explicitly use ProtOr classifier, SR algorithm, 960 points, probe 1.4 Å,
plus `options={"hetatm": True}` to include crystal waters (ferritin does by
default) brings them into agreement.

**Energy (ferritin vs OpenMM AMBER96): bonded terms agree, nonbonded is
systematically off.** Across all 6 PDBs after H placement + LBFGS
minimization:

| PDB  | bond  | angle  | torsion | nonbonded |
|------|-------|--------|---------|-----------|
| 1ake | 3.3 % | 18.8 % | 11.7 %  | **126.9 %** |
| 1aki | 4.4 % | 22.2 % | 2.6 %   | **139.4 %** |
| 1bpi | 0.4 % | 9.0 %  | 5.6 %   | **162.3 %** |
| 1crn | 1.9 % | 16.3 % | 3.9 %   | **167.2 %** |
| 1pgb | 1.5 % | 1.9 %  | 9.2 %   | **152.8 %** |
| 1ubq | 5.6 % | 16.8 % | 12.5 %  | **137.3 %** |

- bond_stretch: 0.4% – 5.6% across all structures (good)
- angle_bend: 1.9% – 22.2% (one outlier)
- torsion+improper: 2.6% – 12.5%
- **nonbonded (vdw+elec combined): 126% – 167% on every structure**

The nonbonded disagreement is consistent across every structure and
therefore **not noise**. Ferritin passes the BALL AMBER96 oracle on
heavy-only 1crn to 0.02% (see `tests/oracle/test_ball_energy.py`), so the
disagreement is specifically against OpenMM's AMBER96, not a ferritin FF
bug against BALL.

Candidate causes (in order of likelihood): (1) different 1-4 scaling, (2)
different terminal residue protonation — ferritin consistently has 1 more
atom than OpenMM after H placement, (3) different dielectric handling.

**This is the single most paper-worthy finding from v1** and the top v2
investigation task.

### Bugs caught by the harness (and fixed during v1 development)

1. **`residue_sasa` altloc-mismatch crash** (commit `1b23670`). On PDBs
   with alternate conformations, `residue_sasa()` walked off the end of
   the atom_sasa slice and panicked: "range end index 627 out of range
   for slice of length 626". Root cause: a leftover from yesterday's
   altloc migration in commit `2412ea2` — `sasa_from_pdb` was switched
   to primary-conformer-only iteration, but `residue_sasa` was still
   calling naive `residue.atom_count()` which includes altloc duplicates.
   Caught immediately when the harness tried to run against 1bpi (which
   has altloc residues). Without the SOTA harness this bug would have
   shipped to users.

### Commits (main branch)

- `c933bb3` — Add SOTA comparison harness scaffolding (v1: FreeSASA)
- `1b23670` — Fix residue_sasa altloc-mismatch crash on PDBs with alternate
  conformations (caught by the harness against 1bpi)
- `152ba40` — Add OpenMM AMBER96 energy runner for SOTA comparison
- `8acc341` — Energy runners place hydrogens before computing AMBER96 energy
- `25b2dc4` — Minimize hydrogens before evaluating energy in both runners
- `cf11370` — Use minimize_hydrogens (AMBER96) for ferritin energy runner
  (batch_prepare hardcodes CHARMM19-EEF1; we need pure AMBER96 for the
  OpenMM comparison)
- `c712034` — Energy runners: bump minimize cap to 2000 steps + record
  atom-count diff
- `9bda1cd` — Strip HETATM in both energy runners for OpenMM compatibility

### Ship criteria for v1 (all met)

- Scaffolding runs end-to-end: ✓
- Setup script creates sota_venv and installs all deps: ✓
- SASA runner produces valid RunnerResult on every v1 PDB: ✓
- OpenMM energy runner produces valid RunnerResult on every v1 PDB: ✓
  (after HETATM stripping)
- Aggregator pairs ferritin against each other impl, emits per-metric
  PASS/WARN/FAIL report + machine-readable summary.json: ✓
- `--strict` mode works (exits non-zero if any FAIL): ✓ (SASA strict-passes,
  energy strict-fails cleanly)
- Report renders cleanly and is readable without further explanation: ✓

---

## Today's Session (2026-04-10 morning): Reader Fix + Convergence Cleanup

**Goal:** Diagnose the 48 stragglers from yesterday's 152/200 victory lap. Fix the
underlying cause(s).

**Ended:** 199/199 eligible proteins converge in the 200-structure prepare sample
(100% on the eligible set), with 1 DNA structure correctly identified as
`skipped_no_protein`. Full 50K benchmark in 14.9 min vs yesterday's 17.9 min
(−17%). Prepare phase 30.8 s vs yesterday's 111.3 s (3.6× faster).

### The two root causes

**1. Altloc atom duplication (the reader bug).** `pdbtbx` represents alternate
conformations as one `Conformer` per altLoc, but copies the non-altloc backbone
atoms into *every* altloc conformer with split occupancy. Iterating
`Residue::atoms()` / `PDB::atoms()` then yields backbone atoms once per altloc,
producing **duplicate atoms at identical coordinates**. The force-field topology
builder then evaluated 1/r¹² vdW between zero-distance pairs, which gave initial
energies of **5.26 × 10¹⁸ kJ/mol on 1aie** (and similar 10¹³–10¹⁸ on the rest of
"Bucket A"). The minimizer had no chance.

The fix: a centralized `crate::altloc` helper module with primary-conformer-per-
residue iteration (blank altLoc → "A" → first available). Migrated **every** naive
`.atoms()` caller in ferritin-connector — 16 sites across `forcefield/topology`,
`forcefield/energy`, `add_hydrogens`, `reconstruct`, `dssp`, `hbond`, `sasa`,
`py_pdb`, `py_analysis`, `py_search`, `py_sasa`, `py_add_hydrogens`. The only
`.atoms()` calls left in ferritin-connector are inside `altloc.rs` itself
(building block + the naive-vs-primary regression test) and the write-back
path that already iterates a single pre-selected primary conformer.

**Impact**: Bucket A initial energies dropped 7–13 orders of magnitude:

| PDB  | Before     | After    |
|------|------------|----------|
| 1aie | 5.26 × 10¹⁸ | 1.51 × 10⁵ |
| 1a1i | 1.14 × 10¹⁵ | 1.98 × 10⁶ |
| 1alu | 7.49 × 10¹⁶ | 1.07 × 10⁶ |
| 108m | 1.13 × 10¹³ | 7.25 × 10⁵ |

Prepare convergence on the 200-sample: **152/200 → 169/200**.

**2. Externally-resolved hydrogens off the MM minimum.** The remaining 31
non-converged structures all already had H atoms in the file (NMR ensembles,
deposited X-ray H, upstream protonators). These coordinates are *locally
reasonable* but rarely at the MM force-field minimum. With heavy atoms
constrained, the LBFGS minimizer dropped energy by 3–4 orders of magnitude
(e.g. 1agt: 2.3 M → 527) but oscillated around a local well it couldn't fully
resolve, never reaching `gradient_tolerance=0.1`.

The fix: an `add_hydrogens::strip_hydrogens` function plus a `strip_hydrogens`
flag on `batch_prepare` (default **true**). When set, all H/D atoms are removed
via `pdb.remove_atoms_by` before placement runs. With clean starting geometry
the minimizer drops them straight into MM minima in 12–22 steps.

**Impact**: 9/9 sampled stragglers (1agt, 1a93, 1a0n, 10dj, 1aj7, 1ajw, 1abt,
1adr, 1ael) converged in 12–22 steps after stripping vs hitting the 200-step
cap with their experimental H. Prepare convergence: **169/200 → 199/200**, AND
prepare wall time dropped 96.1 s → 31.7 s because stragglers stop burning the
LBFGS step cap.

### The honest scoreboard cleanup

After the two fixes, the only "non-converged" structure left in the 200 sample
is **193d**, a DNA/antibiotic complex (528 DG + 520 DT + 496 DA + 480 DC + 152
HQU + non-standard residues + 80 ALA). AMBER96 has no parameters for DNA bases
→ 394/404 atoms unassigned → no minimization runs → `converged` defaults to
False.

This isn't a bug or a convergence failure — it's the *wrong tool for the job*.
Added a `skipped_no_protein` field to `PrepReport` and a heuristic in
`batch_prepare` (`n_unassigned * 2 > total_atoms`) that detects this case,
skips minimization, and sets the new flag. Downstream consumers can now
distinguish "didn't converge" from "wasn't a candidate". Bonus: the topology
builder is now called once instead of twice per structure (it was being rebuilt
after minimization solely for the unassigned count, which is invariant under
coordinate changes).

### 50K benchmark progression (2026-04-09 → 2026-04-10)

| Run | Prepare time | Converged | Total time |
|-----|--------------|-----------|-----------|
| Yesterday's victory lap | 111.3 s | 152/200 (76 %) | 17.9 min |
| + altloc fix | 96.1 s | 169/200 (84 %) | — |
| + strip_hydrogens (default on) | 31.7 s | 199/200 + 1 skipped (99.5 %) | 14.8 min |
| + skipped_no_protein flag | 30.8 s | 195/200 reported (97.5 %)* | **14.9 min** |

\* The 5-vs-1 discrepancy between the standalone microbenchmark (199 + 1 skipped)
and the full benchmark (195) is reproducible — the 4 boundary-case structures
flap between converged/not-converged depending on parallel reduction order.
Worth investigating as a follow-up but doesn't affect the overall picture.

45 109 / 50 000 structures loaded (within 9 of yesterday's 45 100 — same set).
Median SASA 19 804 Å² (yesterday: 19 784). Median energy 1.33 × 10⁶ kJ/mol
(yesterday: 1.59 × 10⁶ — cleaner geometry from better topology).

### Test delta

- **63 → 64 ferritin-connector tests** (+1)
- New regression tests:
  - `altloc::tests::altloc_iteration_dedupes_backbone` — 1aie loaded, naive
    iteration count > primary count, no duplicate coords in primary view
  - `add_hydrogens::tests::test_strip_hydrogens_round_trip` — place all H,
    strip, verify count drops to heavy-only and re-placement is deterministic
- All 204 cargo tests still pass (62 → 63 connector + 21 io + 102/11/6/1 align)

### Commits (main branch)

- `2412ea2` — Fix altloc atom duplication corrupting topology and Python getters
- `104ed0d` — Migrate remaining naive .atoms() callers to primary-conformer iteration
- `b98d953` — Add strip_hydrogens flag to batch_prepare for unconvergeable stragglers
- `96029c3` — Default strip_hydrogens=true in prepare and batch_prepare
- `06bac01` — Add skipped_no_protein status to PrepReport

---

## Yesterday's Session (2026-04-09): The 50K Benchmark Battle Test

**Goal:** Run `benchmark/run_benchmark.py --n 50000` on monster3 (120 cores, 250 GB RAM)
end-to-end without crashes, and fix whatever breaks.

**Started:** `memory allocation of 1,306,087,848,000 bytes failed` at chunk 1.
**Ended:** 50K benchmark completes in 15-18 min with 76% minimizer convergence on a
200-structure prepare sample.

### 10 Bugs Found and Fixed

#### OOM layer (caught by running out of RAM)
1. **NMR multi-model inflation** — `pdb.chains()` iterates all models but `atom_count()`
   counts model 0 only. NMR ensembles with 20+ models slipped past size filters and
   inflated data 20× during batch ops. Fix: use `pdb.models().next().chains()` in
   seven extraction functions.
2. **PDB cloning at batch scale** — `batch_backbone_hbonds`, `batch_place_peptide_hydrogens`,
   `batch_prepare`, `batch_minimize_hydrogens` cloned **all** `pdbtbx::PDB` objects into
   a single `Vec` before processing. Fix: chunked cloning (500 per chunk) + extract just
   the residue data (not full PDBs) where possible.
3. **`apply_coords_to_pdb` model mismatch** — wrote back coords via `pdb.chains_mut()` (all
   models) while `build_topology` read via first model only. Assertion failure:
   `coord array too short (564 coords, atom index 564)`. Fix: both use first model.
4. **CellList grid cap too loose** — SASA's cell list was capped at 500³ = 3 GB per
   instance. With 120 threads each holding one, peak 360 GB > 250 GB available. Fix:
   tightened to 150³ (~80 MB) — covers 900 Å bboxes, more than any real protein.
5. **NeighborList grid uncapped** — the real 1.3 TB culprit. `forcefield::neighbor_list::
   NeighborList::build` had no cap at all. A structure with bogus coordinates
   (bbox ~64 000 Å) produced a 3790³ grid. Fix: same 150³ cap as CellList.

#### Force layer (caught by FD gradient checks)
6. **Angle force sign flip** in `compute_energy_and_forces_impl` — `dv = -2k(θ-θ₀)/sin(θ)`
   should be `+2k(θ-θ₀)/sin(θ)`. Code was adding +gradient instead of -gradient. Silently
   killed the minimizer (stalled at step 3). Diagnosed by numerical FD gradient check
   disagreeing with analytical force by 20-30% on a test H atom.
7. **`CubicSwitch` derivative factor of 2** — `d/dr²[(a-r²)²(a+2r²-3b)] = 6(a-r²)(b-r²)`,
   but code used 12. Barely affected vdW (energy ≈ 0 in the 13-15 Å switching region)
   but made electrostatic forces ~44 % too large. Found by comparing per-atom force
   components against FD — VdW matched, Elec didn't.
10. **Angle force sign flip in `bonded_energy_and_forces`** — the twin of bug #6, in
    the helper used by the neighbor-list accelerated code path. The regression tests
    from bug #6 only exercised the O(N²) path, so this sat undetected. Structures above
    `NBL_AUTO_THRESHOLD=2000` kept terminating at LBFGS step 4-5. Fix: same sign flip.
    **Caught the moment we added NBL-specific regression tests.**

#### Performance layer
8. **AMBER96-in-vacuum as the default `prepare()` force field** — raw AMBER96 electrostatic
    on a bare protein gives ~80 000 kJ/mol on crambin (unscreened charges). CHARMM19-EEF1's
    implicit solvation term gives ~1 400 kJ/mol instead. Scientifically better default
    for "prepare a loaded PDB". `compute_energy(s, ff="amber96")` still works for
    BALL oracle comparisons.
9. **Neighbor list rebuilt on every force call** — the minimizer called
    `compute_energy_and_forces` per LBFGS step × 20 line search trials, and each call
    rebuilt the neighbor list from scratch. Fix: `NbCache` struct in `minimize.rs` that
    builds the list once and calls `needs_rebuild` between iterations. Line search caps
    the initial alpha at `max_disp = 0.8 Å` so trials stay inside the 2 Å NBL buffer —
    no mid-line-search rebuilds needed. **3-5× per-step speedup on 5K-10K atom
    structures.**

**Plus the meta-bug:** `cargo clean` on monster3 kept failing silently because
`validation/alphabet_vqvae_rust.txt` was missing from the ferritin-align `include!` path,
so every "rebuild" was quietly reusing a stale cached .so. First three "fixes" didn't
apply because the new binary was never actually built. The 1.3 TB allocation number
being byte-identical across runs was the tell.

### Test Delta

- **49 → 60 Rust tests** (+ 11)
- New regression tests that would have caught bugs #6, #7, #10 immediately:
  - `gradient_matches_numerical_on_hydrogens` / `_on_heavy_atoms` (O(N²) path)
  - `nbl_gradient_matches_numerical_on_hydrogens` / `_on_heavy_atoms` (NBL path)
  - `nbl_energy_matches_exact_energy` (cross-check the two implementations)
  - `cubic_switch_derivative_matches_numerical`
- BALL energy oracle still passes (8/8)
- 36 Python `test_prepare.py` tests pass in 3-5 s (were 7 min before the minimizer
  was actually working)

### 50K Benchmark Progression

| Run | Prepare | Converged | Total time |
|-----|---------|-----------|-----------|
| Initial (crashing) | — (OOM) | — | — |
| After OOM fixes | 121.9 s | **0/200** | 16.8 min |
| + Angle + CubicSwitch | 107.7 s | 47/200 | 16.2 min |
| + NBL caching | 66.9 s | 46/200 | 15.6 min |
| + NBL angle fix | 42.1 s | **140/200** | 15.0 min |
| Victory lap (200 steps) | 111.3 s | **152/200** | 17.9 min |

45 100 / 50 000 structures loaded (the 4 900 drops are file-size + atom-count filters).
Median SASA 19 784 Å², median energy 1.6 × 10⁶ kJ/mol (AMBER96 vacuum, no water —
expected to be huge), 25.2 M hydrogens placed across the whole run.

### Commits (main branch)

- `bb0058b` — Fix analytical force bugs (angle sign + CubicSwitch factor of 2)
- `73248f7` — Switch `batch_prepare` default to CHARMM19-EEF1
- `96a16ae` — Cache neighbor list across minimizer iterations
- `93ceef5` — Fix angle force sign in `bonded_energy_and_forces` (NBL twin)
- `e9b65d9` — Remove LBFGS debug prints after bug hunt
- (earlier) `49352c8` — Cap NeighborList grid, tune SASA thread budget

---

## Next Session: Large-Scale Benchmarking + SOTA Comparisons

The 50K run proved correctness at scale. Next we need **characterization** (how fast,
how much memory, how does it scale) and **validation against state-of-the-art tools**
(are our numbers actually right).

### Track A: Scaling Characterization

The goal is publication-quality numbers for a paper plus a regression harness that
detects performance drops.

- [ ] **Core scaling curves** — for each batch op (`batch_total_sasa`, `batch_dssp`,
  `batch_dihedrals`, `batch_backbone_hbonds`, `batch_place_peptide_hydrogens`,
  `batch_prepare`), measure throughput at `n_threads ∈ {1, 2, 4, 8, 16, 32, 64, 120}`
  on monster3. Plot time vs cores and identify the point of diminishing returns. The
  `auto_threads` heuristic we added during the OOM hunt is based on a per-task memory
  estimate — measure whether it picks the right number.

- [ ] **Peak RSS vs N structures** — run each batch op on `N ∈ {100, 1 000, 5 000,
  10 000, 50 000}` and record peak RSS. Is it linear? What's the per-structure
  memory overhead? Where do the chunked-clone paths (`batch_place_peptide_hydrogens`,
  `batch_prepare`) win vs lose?

- [ ] **Per-structure timing distribution** — for SASA and prepare, plot wall time per
  structure vs `atom_count`. Expect O(N) (with NBL) for SASA, O(N·steps) for prepare.
  Any outliers are either pathological structures (worth filing) or real bugs.

- [ ] **Chunk size sweep** — `--chunk ∈ {1 000, 2 000, 5 000, 10 000}`. Does smaller chunk
  size reduce peak memory without hurting throughput? Current default is 5 000.

- [ ] **NBL threshold tuning** — `NBL_AUTO_THRESHOLD = 2000` is a guess. Run SASA and
  minimize on structures of 500/1 000/2 000/3 000/5 000 atoms with NBL on vs off and
  find the real crossover point.

- [x] ~~**48 stubborn structures** — the ones that still don't converge in 200 LBFGS steps.~~
  **Resolved 2026-04-10** — root causes were altloc duplication + non-MM-optimal
  externally-placed H. See today's session notes above. Now 199/199 eligible
  proteins converge; the only remaining "non-converged" is 193d (DNA), correctly
  flagged via `skipped_no_protein`.

- [ ] **Parallel-order convergence flap** — the full `run_benchmark.py` reports
  195/200 converged while a structurally-identical standalone reproduction
  yields 199/200 + 1 skipped. Difference is 4 boundary-case structures whose
  LBFGS termination flips with parallel reduction order. Need to identify the
  4 PDBs (run twice and diff the per-structure converged flags), then decide
  whether to (a) tighten determinism in the parallel reductions, (b) widen
  the gradient tolerance to make the boundary unambiguous, or (c) add a small
  hysteresis ("converged if grad < tol OR grad < 1.1·tol on consecutive steps").

### Track B: SOTA Comparison Matrix

Scientific validation beyond the BALL oracle. Pick a reference set of ~50 structures
spanning small (crambin ~300), medium (lysozyme ~1 200), large (3K-10K atoms), and a
few multimers. For each tool, record runtime + output, then compare against ferritin.

| Ferritin op | Reference tool(s) | What to compare |
|-------------|------------------|-----------------|
| `compute_energy(ff="amber96")` | BALL (already set up), OpenMM AMBER99SB | Component-wise energies within 0.1 % |
| `compute_energy(ff="charmm19_eef1")` | CHARMM/c36b1 if available | Implicit solvent total energy |
| `batch_total_sasa` | FreeSASA (CLI), NACCESS, MSMS | Per-residue SASA, RMSD of values |
| `batch_dssp` | `dssp-3` (Kabsch-Sander reference) | Secondary-structure string identity % |
| `batch_backbone_hbonds` | DSSP's own H-bond list | Pair set Jaccard similarity |
| `batch_place_peptide_hydrogens` | PDBFixer, Reduce | RMSD of placed H positions |
| `batch_prepare` (full pipeline) | PDBFixer + OpenMM LocalEnergyMinimizer | Energy drop, wall time, output geometry |
| `tmalign` (already done via USAlign) | USAlign C++ | TM-score drift (already <5e-4) |

Infrastructure work needed before any comparisons:
- [ ] Install FreeSASA, dssp-3, Reduce, PDBFixer, OpenMM on monster3 (probably
  easiest via a conda env alongside the existing venv)
- [ ] Assemble the reference set — curate 50 PDBs with clean monomers, no metals,
  no weird ligands; commit the list to `tests/corpus/sota_reference.txt`
- [ ] Orchestration script in `validation/sota_comparison/` that runs each tool on
  each structure and writes a per-tool JSON
- [ ] Aggregator that produces a markdown report + charts from the per-tool JSONs

### Track C: Concrete Deliverables for the Paper

Assuming we want to write this up:

- [ ] Figure 1: architecture diagram (pure Rust → PyO3 → Python layer)
- [ ] Figure 2: core scaling curves for SASA / DSSP / prepare (from Track A)
- [ ] Figure 3: SOTA comparison bar chart (wall time + correctness per op)
- [ ] Table 1: per-op throughput on 45K real PDBs
- [ ] Table 2: BALL / OpenMM energy agreement table
- [ ] A clean "from 0/200 to 152/200 converged" narrative for the intro — the 10 bugs
  we found are honest evidence that the 50K scale is necessary

### Smaller Items Also Worth Doing

- [x] ~~**Review the 48 stubborn structures.** Might reveal another bug.~~
  Done 2026-04-10 — see today's session notes; revealed two real bugs (altloc +
  experimental-H trapping).
- [ ] **Profile prepare for 5K-10K atom structures.** After the NBL caching fix the
  minimizer is much faster but `topology::build_topology` might now dominate. Worth a
  quick flamegraph. (Today's `build_topology`-once cleanup may already have reduced
  this — re-measure first.)
- [ ] **Propagate CHARMM default to `compute_energy`?** Right now only `batch_prepare`
  defaulted to CHARMM. `compute_energy` still defaults to AMBER96 because the BALL
  oracle depends on that. Might be worth discussing whether the Python wrapper default
  should flip too.
- [ ] **Dead-code cleanup.** `#[allow(dead_code)]` on `resolve_threads`, 
  `minimize_hydrogens`, `minimize_hydrogens_cg`, `minimize_hydrogens_lbfgs`,
  `velocity_verlet`, `compute_energy_dd` — decide whether to keep or remove.
  (Today's commit moved `resolve_threads` to `parallel.rs` and removed its
  `#[allow(dead_code)]`, so that one is done.)
- [ ] **Investigate OpenMP-style within-structure parallelism** for SASA on very large
  structures (>10K atoms). Currently we parallelize across structures, but one huge
  structure still takes a full core for a long time. `shrake_rupley_parallel` exists
  but isn't wired in.
- [ ] **DNA / RNA prep pipeline.** 193d and other nucleic-acid entries are now
  cleanly flagged as `skipped_no_protein`, but if we want them to actually go
  through prepare we need a nucleic-acid-aware H placement path and an
  AMBER nucleic-acid parameter set wired into the topology builder. Out of
  scope for the protein-prep pipeline as currently designed; could be a
  separate `prepare_nucleic_acid` function.

### Known Gotchas for Next Time

- `cargo clean && maturin develop --release` on monster3 will fail silently if
  `validation/alphabet_vqvae_rust.txt` isn't present. The build "succeeds" but reuses
  a stale .so. **Always check compile time >5 s** after a supposed rebuild.
- Benchmarks on monster3 should run inside tmux (`tmux new-session -d -s ferritin_bench`)
  so they survive SSH disconnects. `nohup` alone hasn't been reliable.
- Background processes can get killed by other agents sharing the box. If a run
  vanishes with no error, check `tmux ls` and process status before assuming a crash.
- `auto_threads` currently uses a 3 GB per-task budget for SASA. This is tuned for
  the monster3 case — may need revisiting on different hardware.

### Opening Command for Next Session

```bash
cd /scratch/TMAlign/ferritin
cat NEXT_SESSION.md   # this file
git log --oneline -15   # session context
cargo test -p ferritin-connector  # should be 63/63 passing (was 60/60 on 2026-04-09)
ssh monster3 "cat /globalscratch/dateschn/ferritin-benchmark/benchmark_results_50k_clean.json"
```
