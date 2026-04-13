"""AMBER96 single-point energy oracle: ferritin vs OpenMM.

Same 1000 PDBs as the fold-preservation benchmark. For each:
  1. Load via PDBFixer + add H (same prep as fold bench).
  2. Save prepared PDB to tmp.
  3. OpenMM: amber96_obc.xml but remove GB force → vacuum AMBER96 single-point.
  4. Ferritin: load same PDB, compute_energy(ff="amber96") in kJ/mol.
  5. Compare totals + components.

Success criterion: per-structure |E_ferritin - E_openmm| / |E_openmm| < 1e-3 on total.
Looser on components because of partitioning ambiguity (impropers live in
PeriodicTorsionForce in OpenMM but are counted separately in ferritin).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PDB_DIR = Path("/globalscratch/dateschn/ferritin-benchmark/pdbs_50k")
OUT = Path("/globalscratch/dateschn/ferritin-benchmark/amber96_oracle.jsonl")
N = 1000
SEED = 42


def compare_one(pdb_path: str) -> dict:
    """Single-point AMBER96 energy: ferritin vs OpenMM (both vacuum)."""
    import openmm
    import openmm.app as app
    from openmm import unit
    from pdbfixer import PDBFixer
    import ferritin

    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    tmp_path = None
    try:
        # 1. PDBFixer prep — both tools see the same atoms + positions.
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        rec["n_atoms"] = len(list(fixer.topology.atoms()))

        # 2. Write to temp PDB for ferritin to ingest.
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        tmp_path = tmp.name
        app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp, keepIds=True)
        tmp.close()

        # 3. OpenMM single-point AMBER96 vacuum (no implicit solvent XML loaded).
        ff = app.ForceField("amber96.xml")
        system = ff.createSystem(
            fixer.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )
        rec["openmm_forces"] = [type(system.getForce(i)).__name__
                                 for i in range(system.getNumForces())]

        # Assign force groups per kept force for component breakdown.
        for i in range(system.getNumForces()):
            system.getForce(i).setForceGroup(i)

        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        plat = openmm.Platform.getPlatformByName("CPU")
        sim = app.Simulation(fixer.topology, system, integrator, plat,
                              {"Threads": "1"})
        sim.context.setPositions(fixer.positions)

        e_total_omm = sim.context.getState(getEnergy=True).getPotentialEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)
        rec["e_total_openmm"] = float(e_total_omm)

        # Per-force-group breakdown
        comps_omm = {}
        for i in range(system.getNumForces()):
            e_i = sim.context.getState(
                getEnergy=True, groups={i}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            comps_omm[type(system.getForce(i)).__name__] = float(e_i)
        rec["components_openmm"] = comps_omm

        # 4. Ferritin single-point on same atoms + positions.
        s = ferritin.load(tmp_path)
        rec["ferritin_n_atoms"] = int(s.atom_count)
        # nbl_threshold large forces exact O(N²) path to match OpenMM NoCutoff.
        result = ferritin.compute_energy(
            s, ff="amber96", units="kJ/mol", nbl_threshold=10**9
        )
        rec["e_total_ferritin"] = float(result["total"])
        rec["components_ferritin"] = {
            k: float(v) for k, v in result.items()
            if k != "total" and isinstance(v, (int, float))
        }

        # 5. Compare.
        diff = rec["e_total_ferritin"] - rec["e_total_openmm"]
        rec["abs_diff_kj"] = float(diff)
        rec["rel_diff"] = float(abs(diff) / (abs(e_total_omm) + 1.0))

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    rec["wall_s"] = time.perf_counter() - t0
    return rec


def main():
    pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
    rng = random.Random(SEED)
    rng.shuffle(pdbs)
    sample = [str(PDB_DIR / name) for name in pdbs[:N]]
    print(f"Sampled {len(sample)} PDBs (seed={SEED})", flush=True)
    print(f"Writing to {OUT}", flush=True)

    n_workers = int(os.environ.get("N_WORKERS", "48"))
    print(f"Using {n_workers} parallel workers", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    big = 0
    with open(OUT, "w") as f, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(compare_one, p): p for p in sample}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if "abs_diff_kj" in rec:
                n_ok += 1
                if abs(rec["abs_diff_kj"]) > 1.0:  # kJ/mol
                    big += 1
            else:
                n_fail += 1
            done += 1
            if done % 25 == 0 or done == len(sample):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (len(sample) - done) / rate if rate > 0 else 0
                print(
                    f"[{done}/{len(sample)}] ok={n_ok} fail={n_fail} "
                    f"|diff|>1kJ: {big}/{n_ok}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone. ok={n_ok} fail={n_fail} in {elapsed/60:.1f} min", flush=True)

    # Summary
    diffs, rels = [], []
    with open(OUT) as f:
        for l in f:
            r = json.loads(l)
            if "abs_diff_kj" in r:
                diffs.append(r["abs_diff_kj"])
                rels.append(r["rel_diff"])
    if diffs:
        d = np.array(diffs); r = np.array(rels)
        print(f"\nEnergy match (n={len(d)}):")
        print(f"  |diff| mean  = {np.mean(np.abs(d)):.3f} kJ/mol")
        print(f"  |diff| median= {np.median(np.abs(d)):.3f} kJ/mol")
        print(f"  |diff| p95   = {np.percentile(np.abs(d), 95):.3f} kJ/mol")
        print(f"  |diff| max   = {np.max(np.abs(d)):.3f} kJ/mol")
        print(f"  rel    mean  = {r.mean():.2e}")
        print(f"  rel    median= {np.median(r):.2e}")
        print(f"  rel    p95   = {np.percentile(r, 95):.2e}")
        print(f"  rel    max   = {r.max():.2e}")


if __name__ == "__main__":
    main()
