"""SOTA comparison: OpenMM AMBER96 + OBC implicit-solvent fold preservation benchmark.

This is the AMBER arm. (The CHARMM arm — OpenMM CHARMM36+OBC2 — lives in
tm_fold_preservation.py's OpenMM counterpart.) The proteon side of this pair
minimizes the same AMBER96 parameters WITHOUT the OBC implicit-solvent term,
so the TM-score diff reflects implementation + solvent-environment
differences, not pure implementation drift (issue 88).

Same 1000 PDBs (seed=42) as the proteon benchmark. For each:
  1. Load PDB via PDBFixer.
  2. Add missing atoms + hydrogens at pH 7.
  3. Extract CA coords (pre-min).
  4. Build system with amber96_obc.xml (AMBER96 + OBC GB) at a 10 Å
     nonbonded cutoff (nonbondedCutoff=1.0 nm).
  5. LocalEnergyMinimizer (tolerance = 10 kJ/mol/nm, matches proteon's 0.1 kcal/mol/A).
  6. Extract CA coords (post-min).
  7. TM-score pre vs post (via proteon.tm_score — pure geometry op).

Results as JSONL, compatible shape with tm_fold_preservation.jsonl.
"""
from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

# OpenMM imports
import openmm
import openmm.app as app
from openmm import unit
from pdbfixer import PDBFixer

# Proteon only for its TM-score (pure geometry).
import proteon

# Worker pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# v0.2.0 data-mount contract: PROTEON_CORPUS_DIR / PROTEON_OUTPUT_DIR override
# the monster3 defaults. Set by the EVIDENT image entrypoint when the user
# bind-mounts /data/pdbs and /data/out.
PDB_DIR = Path(os.environ.get("PROTEON_CORPUS_DIR") or "/globalscratch/dateschn/proteon-benchmark/pdbs_50k")
OUTPUT_DIR = Path(os.environ.get("PROTEON_OUTPUT_DIR") or "/globalscratch/dateschn/proteon-benchmark")
OUT = OUTPUT_DIR / "tm_fold_preservation_openmm_amber.jsonl"
N = int(os.environ.get("N_PDBS", "1000"))
SEED = 42

MIN_TOL = 10.0 * unit.kilojoule_per_mole / unit.nanometer  # ~0.24 kcal/mol/A
MAX_ITER = 100  # match proteon's minimize_steps=100


def extract_ca(topology, positions) -> np.ndarray:
    """Extract CA atom coordinates as (N, 3) numpy array in Angstroms."""
    ca_idx = [a.index for a in topology.atoms() if a.name == "CA"]
    pos_nm = np.array(positions.value_in_unit(unit.nanometer))
    return pos_nm[ca_idx] * 10.0  # nm -> A


def tm_pair(ca_ref: np.ndarray, ca_mov: np.ndarray) -> dict:
    n = len(ca_ref)
    invmap = np.arange(n, dtype=np.int32)
    tm, n_aln, rmsd_val, _R, _t = proteon.tm_score(ca_mov, ca_ref, invmap)
    return {"tm_score": float(tm), "rmsd": float(rmsd_val),
            "n_ca": int(n), "n_aligned": int(n_aln)}


def prepare_fixer(pdb_path: str) -> tuple[PDBFixer | None, int]:
    """Load + fix + add H. Returns (fixer, 0) on success.

    On missing heavy atoms, returns (None, n_missing) so the caller can skip:
    PDBFixer's addMissingAtoms() hangs deterministically on a non-trivial
    fraction of wwPDB inputs (PR #47). The fold-preservation comparison
    surface narrows to "well-resolved wwPDB" — the more defensible
    scientific population anyway.
    """
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    # Only add missing atoms (skip missing terminal residues — too aggressive).
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        return None, len(fixer.missingAtoms)
    fixer.addMissingHydrogens(7.0)
    return fixer, 0


def run_one(pdb_path: str, ff: app.ForceField, platform: openmm.Platform,
            platform_props: dict = None) -> dict:
    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    try:
        fixer, n_missing = prepare_fixer(pdb_path)
        if fixer is None:
            rec["skipped"] = "missing_heavy_atoms"
            rec["missing_count"] = int(n_missing)
            rec["wall_s"] = float(time.perf_counter() - t0)
            return rec
        top = fixer.topology
        pos_pre = fixer.positions
        ca_pre = extract_ca(top, pos_pre)
        rec["n_ca_pre"] = int(len(ca_pre))
        if len(ca_pre) == 0:
            rec["skipped"] = "no_ca"
            return rec

        system = ff.createSystem(
            top,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=None,
            rigidWater=False,
        )
        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        if platform_props:
            simulation = app.Simulation(top, system, integrator, platform, platform_props)
        else:
            simulation = app.Simulation(top, system, integrator, platform)
        simulation.context.setPositions(pos_pre)

        e_pre = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        rec["initial_energy_kj"] = float(e_pre.value_in_unit(unit.kilojoule_per_mole))

        openmm.LocalEnergyMinimizer.minimize(
            simulation.context, MIN_TOL, MAX_ITER
        )

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        e_post = state.getPotentialEnergy()
        rec["final_energy_kj"] = float(e_post.value_in_unit(unit.kilojoule_per_mole))

        ca_post = extract_ca(top, state.getPositions())
        rec["n_ca_post"] = int(len(ca_post))

        if ca_post.shape != ca_pre.shape:
            rec["error"] = f"CA shape mismatch {ca_pre.shape} vs {ca_post.shape}"
        else:
            rec.update(tm_pair(ca_pre, ca_post))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def _worker(pdb_path: str) -> dict:
    """Worker function: single-threaded CPU platform + full pipeline for one PDB."""
    # Re-import per process to ensure fresh state
    import openmm, openmm.app as app  # noqa: F811
    os.environ["OPENMM_CPU_THREADS"] = "1"
    ff = app.ForceField("amber96_obc.xml")
    platform = openmm.Platform.getPlatformByName("CPU")
    return run_one(pdb_path, ff, platform, {"Threads": "1"})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdb_list = os.environ.get("PROTEON_PDB_LIST")
    if pdb_list:
        list_path = Path(pdb_list)
        if not list_path.is_file():
            raise SystemExit(f"PDB list file not found: {list_path}")
        with open(list_path) as f:
            paths = [line.strip() for line in f if line.strip()]
        paths = [p for p in paths if Path(p).is_file()][:N]
        sample = [Path(p) for p in paths]
        print(f"Loaded {len(sample)} PDB paths from {list_path}", flush=True)
    else:
        pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
        rng = random.Random(SEED)
        rng.shuffle(pdbs)
        sample = [PDB_DIR / name for name in pdbs[:N]]
        print(f"Sampled {len(sample)} PDBs from {PDB_DIR} (seed={SEED})", flush=True)

    done_names: set[str] = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    done_names.add(json.loads(line)["pdb"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_names)} PDBs already in {OUT}", flush=True)
    pending = [p for p in sample if p.name not in done_names]
    print(f"Total sample: {len(sample)} PDBs; {len(pending)} pending after resume", flush=True)
    print(f"Writing to {OUT}", flush=True)

    n_workers = int(os.environ.get("N_WORKERS", "32"))
    print(f"Using {n_workers} parallel workers (CPU platform, 1 thread each)", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    with open(OUT, "a") as f, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, str(p)): (i, p) for i, p in enumerate(pending)}
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            if "tm_score" in rec:
                n_ok += 1
            elif rec.get("skipped"):
                n_skip += 1
            else:
                n_fail += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0 or done == len(pending):
                progress = len(done_names) + done
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(
                    f"[{progress}/{len(sample)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone. ok={n_ok} fail={n_fail} skip={n_skip} in {elapsed/60:.1f} min "
          f"({n_ok/elapsed:.2f} struct/s)", flush=True)

    # Summary
    tms, rmsds = [], []
    with open(OUT) as f:
        recs = [json.loads(l) for l in f]
    for r in recs:
        if "tm_score" in r:
            tms.append(r["tm_score"])
            rmsds.append(r["rmsd"])
    if tms:
        tms_arr = np.array(tms)
        rmsds_arr = np.array(rmsds)
        print(f"\nOpenMM AMBER96+OBC TM-score (n={len(tms_arr)}):")
        print(f"  mean={tms_arr.mean():.4f}  median={np.median(tms_arr):.4f}")
        print(f"  min={tms_arr.min():.4f}  p01={np.percentile(tms_arr,1):.4f}  p05={np.percentile(tms_arr,5):.4f}")
        print(f"  p95={np.percentile(tms_arr,95):.4f}  max={tms_arr.max():.4f}")
        print(f"RMSD: mean={rmsds_arr.mean():.3f}  median={np.median(rmsds_arr):.3f}  max={rmsds_arr.max():.3f}")


if __name__ == "__main__":
    main()
