"""AMBER96 single-point energy oracle: proteon vs OpenMM.

Same PDB sample as the fold-preservation benchmark. For each:
  1. Load via PDBFixer + add H (skip if missing heavy atoms — PR #47).
  2. Save prepared PDB to tmp.
  3. OpenMM: amber96.xml NoCutoff vacuum single-point.
  4. Proteon: load same PDB, compute_energy(ff="amber96") in kJ/mol.
  5. Compare totals + components.

Runner contract (v0.2.0 onward):

  PROTEON_CORPUS_DIR   directory of input .pdb files (v0.2.0 universal)
  PROTEON_PDB_DIR      legacy alias for the input directory
  PROTEON_PDB_LIST     explicit pre-filtered list of paths (one per line);
                       overrides directory glob when set
  PROTEON_OUTPUT_DIR   directory for the JSONL artifact (v0.2.0)
  PROTEON_AMBER_ORACLE_OUT  legacy explicit JSONL path
  N_PDBS               how many PDBs from the sample to actually score
  N_WORKERS            pool width (default 48)
  TASK_TIMEOUT_S       per-task timeout (default 60s)
  SEED                 sample shuffle seed (default 42)

Success criterion: per-structure |E_proteon - E_openmm| / |E_openmm| < 1e-3
on total. Looser on components because of partitioning ambiguity (impropers
live in PeriodicTorsionForce in OpenMM but are counted separately in proteon).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
import traceback
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from pathlib import Path

import numpy as np

# Heavy oracle imports at module top-level so each pebble worker pays the
# proteon / openmm load cost ONCE per worker process at startup, not once
# per task. The Rust .so + the OpenMM ForceField parser amount to several
# seconds of cold-start that would otherwise dominate the per-task budget
# on small PDBs.
import pebble
import openmm
import openmm.app as openmm_app
from openmm import unit as openmm_unit
from pdbfixer import PDBFixer
import proteon

# Path resolution: legacy per-runner env vars win, then the v0.2.0 universal
# data-mount contract (PROTEON_CORPUS_DIR / PROTEON_OUTPUT_DIR) — which the
# EVIDENT image entrypoint exports when /data/pdbs and /data/out are mounted
# — and finally a monster3 fallback for the existing batch workflow.
PDB_DIR = Path(
    os.environ.get("PROTEON_PDB_DIR")
    or os.environ.get("PROTEON_CORPUS_DIR")
    or "/globalscratch/dateschn/proteon-benchmark/pdbs_50k"
)
_v02_out_dir = os.environ.get("PROTEON_OUTPUT_DIR")
OUT = Path(
    os.environ.get("PROTEON_AMBER_ORACLE_OUT")
    or (Path(_v02_out_dir) / "amber96_oracle.jsonl" if _v02_out_dir else None)
    or "/globalscratch/dateschn/proteon-benchmark/amber96_oracle.jsonl"
)
N = int(os.environ.get("N_PDBS", "1000"))
SEED = int(os.environ.get("SEED", "42"))


def compare_one(pdb_path: str) -> dict:
    """Single-point AMBER96 energy: proteon vs OpenMM (both vacuum)."""
    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    tmp_path = None
    tmp_raw_path = None
    try:
        # 1. PDBFixer prep — both tools see the same atoms + positions.
        # We DETECT but do NOT add missing heavy atoms: PDBFixer's
        # addMissingAtoms() hangs deterministically on a non-trivial fraction
        # of wwPDB inputs. Skip rather than try to repair (per PR #47); the
        # comparison surface becomes "well-resolved wwPDB" rather than
        # "everything PDBFixer can repair" — the more defensible scientific
        # population anyway, since modeled-back atoms have ad-hoc geometry.
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        if fixer.missingAtoms:
            rec["skipped"] = "missing_heavy_atoms"
            rec["missing_count"] = len(fixer.missingAtoms)
            rec["wall_s"] = float(time.perf_counter() - t0)
            return rec
        fixer.addMissingHydrogens(7.0)
        rec["n_atoms"] = len(list(fixer.topology.atoms()))

        # 2. Write to temp PDB for proteon to ingest.
        # We write the raw PDBFixer output, then run the proteon
        # histidine-tautomer normalizer over it. proteon's residue-name-
        # driven typer needs HIS residues renamed to HID / HIE / HIP based
        # on which Hδ1 / Hε2 PDBFixer placed (issue #60); OpenMM's
        # amber96.xml does the equivalent detection internally on
        # `fixer.topology`, so the OpenMM arm above is fine reading the
        # raw topology while proteon below loads the renamed copy.
        tmp_raw = tempfile.NamedTemporaryFile(suffix="_raw.pdb", delete=False, mode="w")
        tmp_raw_path = tmp_raw.name
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp_raw, keepIds=True)
        tmp_raw.close()

        tmp_path = tempfile.NamedTemporaryFile(suffix="_proteon.pdb", delete=False, mode="w").name
        his_counts = proteon.normalize_histidine_tautomers(tmp_raw_path, tmp_path)
        rec["histidine_tautomer_counts"] = his_counts

        # 3. OpenMM single-point AMBER96 vacuum (no implicit solvent XML loaded).
        ff = openmm_app.ForceField("amber96.xml")
        system = ff.createSystem(
            fixer.topology,
            nonbondedMethod=openmm_app.NoCutoff,
            constraints=None,
            rigidWater=False,
        )
        rec["openmm_forces"] = [type(system.getForce(i)).__name__
                                 for i in range(system.getNumForces())]

        # Assign force groups per kept force for component breakdown.
        for i in range(system.getNumForces()):
            system.getForce(i).setForceGroup(i)

        integrator = openmm.VerletIntegrator(0.001 * openmm_unit.picosecond)
        plat = openmm.Platform.getPlatformByName("CPU")
        sim = openmm_app.Simulation(fixer.topology, system, integrator, plat,
                                    {"Threads": "1"})
        sim.context.setPositions(fixer.positions)

        e_total_omm = sim.context.getState(getEnergy=True).getPotentialEnergy(
        ).value_in_unit(openmm_unit.kilojoule_per_mole)
        rec["e_total_openmm"] = float(e_total_omm)

        # Per-force-group breakdown
        comps_omm = {}
        for i in range(system.getNumForces()):
            e_i = sim.context.getState(
                getEnergy=True, groups={i}
            ).getPotentialEnergy().value_in_unit(openmm_unit.kilojoule_per_mole)
            comps_omm[type(system.getForce(i)).__name__] = float(e_i)
        rec["components_openmm"] = comps_omm

        # 4. Proteon single-point on same atoms + positions.
        s = proteon.load(tmp_path)
        rec["proteon_n_atoms"] = int(s.atom_count)
        # nbl_threshold large forces exact O(N²) path; nonbonded_cutoff=1e6
        # disables proteon's default 15 Å cutoff with switching, matching
        # OpenMM's NoCutoff convention. Without this, proteon truncates the
        # long-range Coulomb at 15 Å while OpenMM goes full-range, giving a
        # systematic ~5% rel_diff (per the warning emitted by
        # proteon.forcefield: "pass nonbonded_cutoff=1e6 to disable for
        # oracle-grade comparison").
        result = proteon.compute_energy(
            s, ff="amber96", units="kJ/mol",
            nbl_threshold=10**9, nonbonded_cutoff=1e6,
        )
        rec["e_total_proteon"] = float(result["total"])
        rec["components_proteon"] = {
            k: float(v) for k, v in result.items()
            if k != "total" and isinstance(v, (int, float))
        }

        # 5. Compare.
        diff = rec["e_total_proteon"] - rec["e_total_openmm"]
        rec["abs_diff_kj"] = float(diff)
        rec["rel_diff"] = float(abs(diff) / (abs(e_total_omm) + 1.0))

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if tmp_raw_path and os.path.exists(tmp_raw_path):
            os.unlink(tmp_raw_path)

    rec["wall_s"] = time.perf_counter() - t0
    return rec


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Two corpus modes (mirrors the CHARMM oracle):
    # 1. PROTEON_PDB_LIST=path/to/list.txt — explicit pre-filtered list
    #    (one path per line). For 50K runs use validation/protein_only_50k.txt
    #    so non-protein structures are pre-filtered.
    # 2. PROTEON_CORPUS_DIR / PROTEON_PDB_DIR — directory glob (legacy
    #    1k-PDB mode). Skip path activates per-PDB.
    pdb_list = os.environ.get("PROTEON_PDB_LIST")
    if pdb_list:
        list_path = Path(pdb_list)
        if not list_path.is_file():
            raise SystemExit(f"PDB list file not found: {list_path}")
        with open(list_path) as f:
            sample = [line.strip() for line in f if line.strip()]
        sample = [p for p in sample if Path(p).is_file()]
        sample = sample[:N]
        print(f"Loaded {len(sample)} PDB paths from {list_path}", flush=True)
    else:
        if not PDB_DIR.is_dir():
            raise SystemExit(f"PDB corpus not found: {PDB_DIR}")
        pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
        if not pdbs:
            raise SystemExit(f"No .pdb files in {PDB_DIR}")
        rng = random.Random(SEED)
        rng.shuffle(pdbs)
        sample = [str(PDB_DIR / name) for name in pdbs[:N]]

    # Resume: if OUT already exists, load done PDBs and skip them.
    done_names: set[str] = set()
    if OUT.exists():
        with open(OUT) as f:
            for l in f:
                try:
                    done_names.add(json.loads(l)["pdb"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_names)} PDBs already in {OUT}", flush=True)
    pending = [p for p in sample if Path(p).name not in done_names]
    print(f"Total sample: {len(sample)} PDBs; {len(pending)} pending after resume", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        return _summarize()

    n_workers = int(os.environ.get("N_WORKERS", "48"))
    task_timeout = float(os.environ.get("TASK_TIMEOUT_S", "60"))
    print(
        f"Using {n_workers} pebble workers, "
        f"per-task timeout {task_timeout:.0f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    big = 0

    # Crash isolation via pebble.ProcessPool. Each task runs in its own
    # subprocess; an OpenMM segfault, malloc abort, or assertion in C++
    # becomes a `pebble.ProcessExpired` exception localised to that one
    # task — the pool keeps running and the next task gets a fresh
    # subprocess. Critically, this is NOT what
    # `concurrent.futures.ProcessPoolExecutor(max_tasks_per_child=1)`
    # gives you: that interface marks the entire pool "broken" on any
    # abnormal worker exit and propagates `BrokenProcessPool` to all
    # in-flight futures. The CHARMM 50K corpus run on monster3 (May 2026,
    # locked into v0.1.3) hit that cascade behaviour in ~93% of records.
    # AMBER96 wasn't run at 50K previously; this migration ensures it
    # doesn't repeat the failure mode.
    with open(OUT, "a") as f:
        with pebble.ProcessPool(max_workers=n_workers) as pool:
            futs = {
                pool.schedule(compare_one, args=[p], timeout=task_timeout): p
                for p in pending
            }
            for fut, pdb_path in futs.items():
                pdb_name = Path(pdb_path).name
                try:
                    rec = fut.result()
                except pebble.ProcessExpired as ex:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"worker subprocess died: exit={ex.exitcode}"
                            f" (likely OpenMM SIGSEGV / abort)"
                        ),
                    }
                except _FuturesTimeoutError:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"task exceeded {task_timeout:.0f}s timeout "
                            f"(killed)"
                        ),
                    }
                except Exception as ex:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"worker exception: {type(ex).__name__}: "
                            f"{str(ex)[:120]}"
                        ),
                    }
                f.write(json.dumps(rec) + "\n"); f.flush()
                if "abs_diff_kj" in rec:
                    n_ok += 1
                    if abs(rec["abs_diff_kj"]) > 1.0:
                        big += 1
                elif "skipped" in rec:
                    n_skip += 1
                else:
                    n_fail += 1
                progress = len(done_names) + n_ok + n_skip + n_fail
                if progress % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (n_ok + n_skip + n_fail) / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - (n_ok + n_skip + n_fail)) / rate if rate > 0 else 0
                    print(
                        f"[{progress}/{len(sample)}] ok={n_ok} skip={n_skip} fail={n_fail}  "
                        f"|diff|>1kJ: {big}/{n_ok}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone. new ok={n_ok} skip={n_skip} fail={n_fail} in {elapsed/60:.1f} min",
        flush=True,
    )
    _summarize()


def _summarize():
    """Read OUT and print headline statistics."""
    diffs, rels = [], []
    with open(OUT) as f:
        for l in f:
            r = json.loads(l)
            if "abs_diff_kj" in r:
                diffs.append(r["abs_diff_kj"])
                rels.append(r["rel_diff"])
    if diffs:
        d = np.array(diffs); rels_arr = np.array(rels)
        print(f"\nEnergy match (n={len(d)}):")
        print(f"  |diff| mean  = {np.mean(np.abs(d)):.3f} kJ/mol")
        print(f"  |diff| median= {np.median(np.abs(d)):.3f} kJ/mol")
        print(f"  |diff| p95   = {np.percentile(np.abs(d), 95):.3f} kJ/mol")
        print(f"  |diff| max   = {np.max(np.abs(d)):.3f} kJ/mol")
        print(f"  rel    mean  = {rels_arr.mean():.2e}")
        print(f"  rel    median= {np.median(rels_arr):.2e}")
        print(f"  rel    p95   = {np.percentile(rels_arr, 95):.2e}")
        print(f"  rel    max   = {rels_arr.max():.2e}")


if __name__ == "__main__":
    main()
