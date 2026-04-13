"""SOTA comparison: OpenMM CHARMM36+OBC2 fold preservation benchmark.

Same 1000 PDBs (seed=42) as the ferritin benchmark. For each:
  1. Load PDB via PDBFixer.
  2. Add missing atoms + hydrogens at pH 7.
  3. Extract CA coords (pre-min).
  4. Build system with charmm36.xml + implicit/obc2.xml.
  5. LocalEnergyMinimizer (tolerance = 10 kJ/mol/nm, matches ferritin's 0.1 kcal/mol/A).
  6. Extract CA coords (post-min).
  7. TM-score pre vs post (via ferritin.tm_score — pure geometry op).

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

# Ferritin only for its TM-score (pure geometry).
import ferritin

# Worker pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

PDB_DIR = Path("/globalscratch/dateschn/ferritin-benchmark/pdbs_50k")
OUT = Path("/globalscratch/dateschn/ferritin-benchmark/tm_fold_preservation_openmm.jsonl")
N = 1000
SEED = 42

MIN_TOL = 10.0 * unit.kilojoule_per_mole / unit.nanometer  # ~0.24 kcal/mol/A
MAX_ITER = 100  # match ferritin's minimize_steps=100


def extract_ca(topology, positions) -> np.ndarray:
    """Extract CA atom coordinates as (N, 3) numpy array in Angstroms."""
    ca_idx = [a.index for a in topology.atoms() if a.name == "CA"]
    pos_nm = np.array(positions.value_in_unit(unit.nanometer))
    return pos_nm[ca_idx] * 10.0  # nm -> A


def tm_pair(ca_ref: np.ndarray, ca_mov: np.ndarray) -> dict:
    n = len(ca_ref)
    invmap = np.arange(n, dtype=np.int32)
    tm, n_aln, rmsd_val, _R, _t = ferritin.tm_score(ca_mov, ca_ref, invmap)
    return {"tm_score": float(tm), "rmsd": float(rmsd_val),
            "n_ca": int(n), "n_aligned": int(n_aln)}


def prepare_fixer(pdb_path: str) -> PDBFixer:
    """Load + fix + add H. Returns PDBFixer with complete topology."""
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    # Only add missing atoms (skip missing terminal residues — too aggressive).
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def run_one(pdb_path: str, ff: app.ForceField, platform: openmm.Platform,
            platform_props: dict = None) -> dict:
    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    try:
        fixer = prepare_fixer(pdb_path)
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
    ff = app.ForceField("charmm36.xml", "implicit/obc2.xml")
    platform = openmm.Platform.getPlatformByName("CPU")
    return run_one(pdb_path, ff, platform, {"Threads": "1"})


def main():
    # Pick sample deterministically (same seed as ferritin bench).
    pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
    rng = random.Random(SEED)
    rng.shuffle(pdbs)
    sample = [PDB_DIR / name for name in pdbs[:N]]
    print(f"Sampled {len(sample)} PDBs (seed={SEED})", flush=True)
    print(f"Writing to {OUT}", flush=True)

    n_workers = int(os.environ.get("N_WORKERS", "32"))
    print(f"Using {n_workers} parallel workers (CPU platform, 1 thread each)", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    with open(OUT, "w") as f, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, str(p)): (i, p) for i, p in enumerate(sample)}
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
            if done % 25 == 0 or done == len(sample):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(sample) - done) / rate if rate > 0 else 0
                print(
                    f"[{done}/{len(sample)}] ok={n_ok} fail={n_fail} skip={n_skip} "
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
        print(f"\nOpenMM CHARMM36+OBC2 TM-score (n={len(tms_arr)}):")
        print(f"  mean={tms_arr.mean():.4f}  median={np.median(tms_arr):.4f}")
        print(f"  min={tms_arr.min():.4f}  p01={np.percentile(tms_arr,1):.4f}  p05={np.percentile(tms_arr,5):.4f}")
        print(f"  p95={np.percentile(tms_arr,95):.4f}  max={tms_arr.max():.4f}")
        print(f"RMSD: mean={rmsds_arr.mean():.3f}  median={np.median(rmsds_arr):.3f}  max={rmsds_arr.max():.3f}")


if __name__ == "__main__":
    main()
