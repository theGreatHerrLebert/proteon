# TODO — Next Session

Focused action list captured at end-of-day **2026-04-10** after three
back-to-back sessions (altloc + strip_hydrogens convergence cleanup, SOTA
harness v1, SOTA scaling plan B). See `NEXT_SESSION.md` for the full
arc and context; this file is the "start here tomorrow" punch list.

## Context in one paragraph

Ferritin's compute pipeline is now at 199/199 eligible-protein convergence
on the 50K-benchmark prepare phase, down from yesterday's 152/200. A SOTA
validation harness (`validation/sota_comparison/`) ships v1 with a clean
FreeSASA SASA comparison (18/18 PASS) and an OpenMM AMBER96 energy
comparison that surfaces a **systematic ~140% nonbonded disagreement**
on every test structure — the single most paper-worthy finding from v1.
Evening work added `batch_prepare(ff="amber96")` and a batched SOTA
runner path to unlock scaling beyond the 6 v1 PDBs, but three robustness
issues block running at 10K.

## Priority 1 — unblock 10K-scale SOTA runs

These three fixes together let the harness process the full 50K corpus
without crashes. Without them, scaling stalls at ~20 PDBs.

### 1a. Preprocess inputs for OpenMM with `Modeller.addMissingAtoms`

**Symptom**: on the 20-PDB scaling demo, OpenMM's `ForceField.createSystem`
failed on **10/20 structures** with errors like:

> "No template found for residue 248 (PHE). The atoms and bonds in the
> residue match PHE, but the set of externally bonded atoms is missing
> 1 C atom. Is the chain missing a terminal capping group?"

These are real PDB format imperfections in the 50K corpus — chains
ending in an incomplete residue, sidechains with missing heavy atoms
that `Modeller.addHydrogens` can't repair. Not a ferritin bug.

**Fix**: in `validation/sota_comparison/runners/energy.py::openmm`,
between `Modeller.addHydrogens(forcefield)` and
`forcefield.createSystem(...)`, call
`modeller.addMissingAtoms(forcefield)` first. That adds missing heavy
atoms using OpenMM's template library before H placement. If it still
fails, fall back to gracefully recording `status="skip"` with the
error so the aggregator can surface the skip rate instead of crashing.

**File**: `validation/sota_comparison/runners/energy.py` — the
`@register("energy", "openmm")` function plus its `ferritin_batch`
twin if any handling there is needed.

**Acceptance**: running `run_all.py --pdbs @sample_100.txt --ops energy`
produces valid records for ≥ 90/100 sampled PDBs (allow some nucleic
acid / weird ligand cases as legitimate skips). The batched ferritin
path already handles these via `batch_load_tolerant` + `skipped_no_protein`.

### 1b. Process-isolate FreeSASA calls

**Symptom**: on the 100-PDB SASA run, FreeSASA segfaults in C at some
point during the per-structure loop (EXIT=139). Reproducible in the
driver but **not** in a standalone FreeSASA loop over the same 100 PDBs.
Likely a C-state / GIL interaction between ferritin's rayon pool and
FreeSASA's C internals. `skip-unknown=True` does not fix it.

**Fix**: in `validation/sota_comparison/runners/sasa.py`, wrap each
`freesasa` call in a `multiprocessing.Pool` worker with
`maxtasksperchild=1` so a crash only kills one worker. Add a
`@register_batch("sasa", "freesasa")` variant that uses the pool. The
driver's batched fast path will pick it up automatically.

**Files**:
- `validation/sota_comparison/runners/sasa.py` — add `freesasa_batch`
  using `multiprocessing.Pool`.

**Acceptance**: `run_all.py --pdbs @sample_100.txt --ops sasa` runs to
completion (exit 0) with results JSON written. If 1-2 structures still
crash in their isolated workers, the pool records them as errors
without killing the driver.

### 1c. Bound LBFGS straggler cost

**Symptom**: on a 100-PDB batched ferritin energy run, the job was
still processing after >10 minutes of wall time (vs ~160s for 6 PDBs).
Some single structure was dominating the rayon critical path with a
>5-minute LBFGS descent.

**Fix options** (pick one):
- **Option A (simple)**: lower the default `minimize_steps` in the SOTA
  energy runner from 2000 to 500 for the batched path. Most structures
  converge in <500 steps thanks to the plateau fallback; the few that
  don't will be flagged `converged=False` but the batch wall time stays
  bounded. This is the cheapest change.
- **Option B (precise)**: add a per-structure wall-clock deadline to
  the Rust minimizer and return partial results. More invasive — needs
  a Rust API change.

Start with Option A.

**Files**:
- `validation/sota_comparison/runners/energy.py` — change
  `minimize_steps=2000` → `minimize_steps=500` in both the
  per-structure and batched ferritin runners.

**Acceptance**: a 100-PDB batched energy run completes in < 5 minutes
wall time. On the v1 6-PDB reference set, the numerical results
(bond/angle/torsion per-component diffs) stay within noise of the
2000-step version.

## Priority 2 — the real scientific finding

### Investigate ferritin vs OpenMM AMBER96 nonbonded disagreement

**The finding** (from the v1 6-PDB energy sweep): bond stretch, angle
bend, and torsion components agree between ferritin and OpenMM to
within 1-22% on every structure. **Nonbonded (vdw + electrostatic
combined) is systematically 126-167% off across every structure.**

Ferritin matches the BALL AMBER96 Julia oracle on heavy-only 1crn to
**0.02%** (see `tests/oracle/test_ball_energy.py`). So the disagreement
is specifically against OpenMM's AMBER96, not a ferritin FF bug per se.

**Candidate causes (order of likelihood)**:
1. **1-4 interaction scaling**: AMBER typically uses `scee=1.2` (1/1.2 ≈ 0.833)
   for electrostatic and `scnb=2.0` (1/2 = 0.5) for vdW on 1-4 pairs.
   Ferritin's `ForceField::scee()` and `scnb()` return these. OpenMM's
   AMBER96 XML declares them as bond-level attributes. If either side
   applies them to the wrong pair set (1-2 vs 1-3 vs 1-4) or with the
   wrong factor, the nonbonded sum diverges.
2. **Terminal residue protonation**: the atom-count diff field showed
   ferritin consistently has **1 more atom than OpenMM** after H
   placement (643 vs 642 on 1crn). A single +1 or -1 charge difference
   on a terminal residue is enough to shift the vacuum electrostatic
   total by tens of kJ/mol.
3. **Nonbonded exclusion set**: AMBER typically excludes 1-2 and 1-3
   pairs from the nonbonded loop, scales 1-4, uses full for 1-5+. If
   ferritin and OpenMM disagree on which pairs are 1-2 vs 1-3 for some
   bond topology (e.g. branched sidechains), the exclusions don't
   match.

**How to investigate** (in rough order):
1. Pick 1crn. Dump ferritin's per-atom partial charges via
   `build_topology` and OpenMM's via `system.getForces()` for the
   NonbondedForce. Diff the charge arrays. If they differ on a single
   terminal atom, that's (2). If they differ on many atoms, the
   atom-typing disagrees.
2. If charges match: inspect which (i, j) pairs each stack puts in the
   NonbondedForce evaluation (OpenMM) vs the vdw+elec inner loop
   (ferritin). Compare the 1-4 pair sets specifically. Compare the
   `scee`/`scnb` factors.
3. If the pair sets match: compare 1-4 scale factors literally. Look
   up `ffr::forcefield::params::amber96.scee()` vs the AMBER96 XML
   value in OpenMM (`<HarmonicBondForce>` + `<NonbondedForce>` sections
   of `amber96.xml`).

**Files to read first**:
- `ferritin-connector/src/forcefield/params.rs` — `pub fn amber96()` at
  line 447 and `scee() / scnb()` in the `ForceField` trait.
- `ferritin-connector/src/forcefield/energy.rs` — the 1-4 scaling code
  in the nonbonded loop.
- OpenMM's `amber96.xml` in the sota_venv:
  `/globalscratch/dateschn/ferritin-benchmark/sota_venv/lib64/python3.12/site-packages/openmm/app/data/amber96.xml`

**Acceptance**: a short writeup in `validation/sota_comparison/NONBONDED_INVESTIGATION.md`
explaining the root cause. If a ferritin bug is found, fix it and
re-run the 6-PDB sweep; expect the nonbonded band to collapse from
FAIL to PASS. If the cause is a legitimate difference (e.g. different
terminal residue conventions), document the convention choice and
propose a v2 mitigation (shared prepared input).

## Priority 3 — ship a clean 10K SOTA run

Only after P1 is done. This is the publication deliverable.

1. Use `sample_corpus.py --n 10000 --seed 42 --output sota_10k.txt` to
   build a stratified 10K sample.
2. Run `run_all.py --pdbs @sota_10k.txt --ops sasa,energy
   --output sota_10k.json` with the P1 fixes in place.
3. Extend `aggregate.py` with **distribution statistics** (mean, median,
   p50, p90, p99, max per metric) because per-structure tables are
   unreadable at 10K. Keep an outlier list (worst 20 per metric).
4. Target total wall time: ~30 minutes.
5. Ship artifacts to `validation/sota_comparison/sota_10k_*`. Update
   the SOTA README headline from "v1 = 6 PDBs" to "v1 = 10K PDBs".

## Priority 4 — deferred items

### 4a. Decide CHARMM default for `compute_energy` (task #2 from today)

`batch_prepare` defaults to CHARMM19-EEF1 since commit `73248f7` because
it gives physically meaningful energies on isolated proteins. But
`compute_energy` still defaults to AMBER96. Decide whether the Python
wrapper `compute_energy` should also flip to CHARMM19-EEF1. The BALL
oracle test (`tests/oracle/test_ball_energy.py`) depends on AMBER96 and
would need an explicit `ff="amber96"` if the default flips.

### 4b. Write up "energy plateau + stragglers" story for the paper

Today's altloc fix + strip_hydrogens + plateau fallback story is
publication-worthy: "from 152/200 to 199/199 eligible-protein
convergence via three orthogonal fixes discovered by a 50K-PDB stress
test." The data is in `NEXT_SESSION.md`; turn it into a methods section.

### 4c. Track A scaling characterization (from 2026-04-09 handoff)

Still open from the earlier NEXT_SESSION plan. Core scaling curves for
batch ops at `n_threads ∈ {1, 2, 4, 8, 16, 32, 64, 120}` on monster3,
peak RSS vs N, per-structure timing distributions. Publication figure
material. Lower priority than the SOTA nonbonded investigation.

### 4d. Native-binary SOTA tools (from the v1 SOTA README)

Build mkdssp ≥ 4.0 from `https://github.com/PDB-REDO/dssp` and Reduce
from `https://github.com/rlabduke/reduce` into
`/globalscratch/dateschn/ferritin-benchmark/sota_bin/`. Wire
`runners/dssp.py` and `runners/h_place.py`. Expands v1's 2-op
comparison (sasa + energy) to 4-op. Lower priority than the P1/P2
work above.

## Opening command for next session

```bash
cd /scratch/TMAlign/ferritin
cat TODO_NEXT_SESSION.md   # this file
cat NEXT_SESSION.md | head -60   # today's arc for context
git log --oneline -25   # all 21 commits from today

# verify the state
cargo test -p ferritin-connector   # should be 69/69 passing

# run a smoke test on monster3
ssh monster3 "source /globalscratch/dateschn/ferritin-benchmark/sota_venv/bin/activate && \
  cd /globalscratch/dateschn/ferritin-benchmark/ferritin && \
  python validation/sota_comparison/run_all.py \
    --pdb-dir /globalscratch/dateschn/ferritin-benchmark/sota_pdbs \
    --ops sasa,energy --output /tmp/smoke.json && \
  python validation/sota_comparison/aggregate.py /tmp/smoke.json \
    --output /tmp/smoke_report.md && cat /tmp/smoke_report.md"
```
