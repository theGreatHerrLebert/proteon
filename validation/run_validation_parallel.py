#!/usr/bin/env python3
"""Ferritin parallel validation suite.

Batch-first design: loads all structures at once, then runs ferritin batch
operations (zero GIL, rayon parallelism), then validates results.

Oracle comparisons (Biopython, Gemmi) run via multiprocessing.

Usage:
    python validation/run_validation_parallel.py [--n-structures 5000] [--pdb-dir validation/pdbs_10k/]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import ferritin

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False

GEMMI_H_BIN = "/scratch/TMAlign/gemmi/build/gemmi"
GEMMI_MONLIB = "/scratch/TMAlign/gemmi/ccd/mon_lib"
HAS_GEMMI_H = os.path.isfile(GEMMI_H_BIN) and os.path.isdir(GEMMI_MONLIB)

FREESASA_BIN = "/scratch/TMAlign/freesasa/src/freesasa"
HAS_FREESASA_CLI = os.path.isfile(FREESASA_BIN)


# ---------------------------------------------------------------------------
# Oracle functions (run in worker processes)
# ---------------------------------------------------------------------------

def _freesasa_sasa(path):
    """Compute FreeSASA (C library) SASA for a single file. Runs in worker process."""
    try:
        t0 = time.time()
        s = freesasa.Structure(path)
        result = freesasa.calc(s)
        elapsed = time.time() - t0
        return {
            "total": result.totalArea(),
            "time_ms": round(elapsed * 1000, 1),
            "atoms": s.nAtoms(),
        }
    except Exception as e:
        return {"error": str(e)[:100]}


def _freesasa_cli_sasa(path):
    """Compute FreeSASA via native CLI. Runs in worker process."""
    import subprocess
    try:
        t0 = time.time()
        result = subprocess.run(
            [FREESASA_BIN, "--shrake-rupley", "--resolution=960", path],
            capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return {"error": result.stderr[:100]}
        for line in result.stdout.splitlines():
            if line.startswith("Total"):
                total = float(line.split(":")[1].strip())
                return {"total": total, "time_ms": round(elapsed * 1000, 1)}
        return {"error": "no Total line in output"}
    except Exception as e:
        return {"error": str(e)[:100]}


def _gemmi_counts(path):
    """Get Gemmi atom/chain counts. Runs in worker process."""
    try:
        doc = gemmi.read_pdb(path)
        gm_atoms = sum(1 for _ in doc[0].all())
        gm_chains = len(list(doc[0]))
        return {"atoms": gm_atoms, "chains": gm_chains}
    except Exception as e:
        return {"error": str(e)[:100]}


def _gemmi_h_placement(path):
    """Place hydrogens via Gemmi CLI. Runs in worker process."""
    import subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmp:
            t0 = time.time()
            result = subprocess.run(
                [GEMMI_H_BIN, "h", f"--monomers={GEMMI_MONLIB}", path, tmp.name],
                capture_output=True, text=True, timeout=30,
            )
            elapsed = time.time() - t0
            if result.returncode != 0:
                return {"error": result.stderr[:100]}
            # Count atoms in output
            n_h = sum(1 for line in open(tmp.name)
                      if (line.startswith("ATOM") or line.startswith("HETATM"))
                      and line[76:78].strip() == "H")
            n_total = sum(1 for line in open(tmp.name)
                         if line.startswith("ATOM") or line.startswith("HETATM"))
            return {"n_h": n_h, "n_total": n_total, "time_ms": round(elapsed * 1000, 1)}
    except Exception as e:
        return {"error": str(e)[:100]}


# ---------------------------------------------------------------------------
# Batch validation phases
# ---------------------------------------------------------------------------

def phase_load(files, n_threads):
    """Phase 1: Load all structures in parallel."""
    print(f"  Loading {len(files)} structures...", end=" ", flush=True)
    t0 = time.time()
    results = ferritin.batch_load_tolerant(files, n_threads=n_threads)
    elapsed = time.time() - t0
    n_loaded = len(results)
    print(f"{n_loaded}/{len(files)} loaded in {elapsed:.1f}s "
          f"({n_loaded / elapsed:.0f} structs/s)")
    return results


def phase_ferritin_batch(structures, n_threads):
    """Phase 2: Run all ferritin batch operations."""
    results = {}

    # DSSP (before hydrogen placement)
    print("  Batch DSSP...", end=" ", flush=True)
    t0 = time.time()
    results["dssp"] = ferritin.batch_dssp(structures, n_threads=n_threads)
    print(f"{time.time() - t0:.1f}s")

    # SASA
    print("  Batch SASA...", end=" ", flush=True)
    t0 = time.time()
    results["sasa"] = np.asarray(ferritin.batch_total_sasa(structures, n_threads=n_threads, radii="protor"))
    print(f"{time.time() - t0:.1f}s")

    # H-bonds
    print("  Batch H-bonds...", end=" ", flush=True)
    t0 = time.time()
    results["hbonds"] = ferritin.batch_backbone_hbonds(structures, n_threads=n_threads)
    print(f"{time.time() - t0:.1f}s")

    # Dihedrals
    print("  Batch dihedrals...", end=" ", flush=True)
    t0 = time.time()
    results["dihedrals"] = ferritin.batch_dihedrals(structures, n_threads=n_threads)
    print(f"{time.time() - t0:.1f}s")

    # Place hydrogens
    print("  Batch place hydrogens...", end=" ", flush=True)
    t0 = time.time()
    results["hydrogens"] = ferritin.batch_place_peptide_hydrogens(
        structures, n_threads=n_threads
    )
    print(f"{time.time() - t0:.1f}s")

    # DSSP after hydrogen placement
    print("  Batch DSSP (post-H)...", end=" ", flush=True)
    t0 = time.time()
    results["dssp_post_h"] = ferritin.batch_dssp(structures, n_threads=n_threads)
    print(f"{time.time() - t0:.1f}s")

    # Idempotency: place again — should add 0
    print("  Batch idempotency check...", end=" ", flush=True)
    t0 = time.time()
    results["hydrogens_2nd"] = ferritin.batch_place_peptide_hydrogens(
        structures, n_threads=n_threads
    )
    print(f"{time.time() - t0:.1f}s")

    return results


def _run_oracle_pool(name, func, files, indices, n_workers):
    """Run an oracle function across files using a process pool."""
    print(f"  {name} ({n_workers} workers)...", end=" ", flush=True)
    results = {}
    t0 = time.time()
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(func, files[i]): i for i in indices}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"error": str(e)[:100]}
        print(f"{time.time() - t0:.1f}s ({len(results)} done)")
    except Exception as e:
        print(f"FAILED: {e}")
    return results


def phase_oracle(files, indices, n_workers):
    """Phase 3: Oracle comparisons via multiprocessing."""
    results = {}

    if HAS_FREESASA:
        results["freesasa"] = _run_oracle_pool(
            "FreeSASA (C library)", _freesasa_sasa, files, indices, n_workers)

    if HAS_FREESASA_CLI:
        results["freesasa_cli"] = _run_oracle_pool(
            "FreeSASA (native CLI, Shrake-Rupley 960)", _freesasa_cli_sasa,
            files, indices, n_workers)

    if HAS_GEMMI:
        results["gemmi_counts"] = _run_oracle_pool(
            "Gemmi atom counts", _gemmi_counts, files, indices, n_workers)

    if HAS_GEMMI_H:
        results["gemmi_h"] = _run_oracle_pool(
            "Gemmi H placement", _gemmi_h_placement, files, indices, n_workers)

    return results


# ---------------------------------------------------------------------------
# Validate results
# ---------------------------------------------------------------------------

def validate_all(files, indices, structures, batch_results, oracle_results):
    """Phase 4: Validate batch results and build per-structure reports."""
    stats = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0})
    all_results = []
    sasa_diffs = []
    sasa_speedups = []

    for pos, (idx, structure) in enumerate(zip(indices, structures)):
        basename = os.path.basename(files[idx])
        tests = []

        # --- Loading ---
        r = {"test": "loading", "status": "pass", "details": {
            "atom_count": structure.atom_count,
            "residue_count": structure.residue_count,
            "chain_count": structure.chain_count,
        }}
        if structure.atom_count == 0:
            r["status"] = "fail"
            r["details"]["error"] = "zero atoms"
        tests.append(r)
        stats["loading"][r["status"]] += 1

        # --- Loading oracle (Gemmi) ---
        r = {"test": "loading_oracle", "status": "pass", "details": {
            "ferritin": {"atoms": structure.atom_count, "chains": structure.chain_count}
        }}
        if idx in oracle_results.get("gemmi_counts", {}):
            gm = oracle_results["gemmi_counts"][idx]
            if "error" not in gm:
                r["details"]["gemmi"] = gm
                diff = structure.atom_count - gm["atoms"]
                if abs(diff) > max(5, structure.atom_count * 0.01):
                    r["status"] = "warn"
                    r["details"]["warning"] = f"atom diff vs gemmi: {diff}"
            else:
                r["details"]["gemmi_error"] = gm["error"]
        tests.append(r)
        stats["loading_oracle"][r["status"]] += 1

        # --- SASA ---
        fe_sasa = float(batch_results["sasa"][pos])
        r = {"test": "sasa", "status": "pass", "details": {
            "ferritin_total": round(fe_sasa, 1),
            "n_atoms": structure.atom_count,
        }}
        # Compare against all available SASA oracles
        for oracle_key, label in [
            ("freesasa", "freesasa"),
            ("freesasa_cli", "freesasa_cli"),
        ]:
            if idx in oracle_results.get(oracle_key, {}):
                ora = oracle_results[oracle_key][idx]
                if "error" not in ora:
                    r["details"][f"{label}_total"] = round(ora["total"], 1)
                    if "time_ms" in ora:
                        r["details"][f"{label}_time_ms"] = ora["time_ms"]
                    if ora["total"] > 0:
                        rel_diff = abs(fe_sasa - ora["total"]) / ora["total"]
                        r["details"][f"{label}_diff"] = round(rel_diff, 4)
                        if label == "freesasa_cli":
                            sasa_diffs.append(rel_diff)
                        if rel_diff > 0.10:
                            r["status"] = "warn"
                            r["details"]["warning"] = f"SASA diff vs {label}: {rel_diff:.1%}"
                else:
                    r["details"][f"{label}_error"] = ora["error"]
        tests.append(r)
        stats["sasa"][r["status"]] += 1

        # --- DSSP ---
        ss = batch_results["dssp"][pos]
        r = {"test": "dssp", "status": "pass", "details": {
            "length": len(ss),
        }}
        valid_chars = set("HGIEBTSCC")
        invalid = set(ss) - valid_chars
        if invalid:
            r["status"] = "fail"
            r["details"]["error"] = f"invalid SS chars: {invalid}"
        else:
            counts = Counter(ss)
            r["details"]["composition"] = dict(counts)
            if len(ss) > 30:
                structured = counts.get("H", 0) + counts.get("E", 0) + counts.get("G", 0)
                if structured == 0:
                    r["status"] = "warn"
                    r["details"]["warning"] = "no H/E/G for large protein"
        tests.append(r)
        stats["dssp"][r["status"]] += 1

        # --- H-bonds ---
        hb = batch_results["hbonds"][pos]
        r = {"test": "hbonds", "status": "pass", "details": {
            "n_hbonds": len(hb),
        }}
        if len(hb) > 0:
            hb = np.asarray(hb)
            r["details"]["energy_range"] = [
                round(float(hb[:, 2].min()), 2),
                round(float(hb[:, 2].max()), 2),
            ]
            if np.any(hb[:, 2] > 0):
                r["status"] = "fail"
                r["details"]["error"] = "positive H-bond energy"
        tests.append(r)
        stats["hbonds"][r["status"]] += 1

        # --- Hydrogens ---
        n_added, n_skipped = batch_results["hydrogens"][pos]
        n_added_2nd, _ = batch_results["hydrogens_2nd"][pos]
        ss_post = batch_results["dssp_post_h"][pos]
        r = {"test": "hydrogens", "status": "pass", "details": {
            "n_added": n_added,
            "n_skipped": n_skipped,
            "dssp_match": ss == ss_post,
            "idempotent": n_added_2nd == 0,
        }}
        if ss != ss_post:
            diffs = [i for i, (a, b) in enumerate(zip(ss, ss_post)) if a != b]
            if len(diffs) <= 3:
                r["status"] = "warn"
                r["details"]["warning"] = f"DSSP mismatch at {len(diffs)} positions (borderline H-bond energy)"
            else:
                r["status"] = "fail"
                r["details"]["error"] = f"DSSP mismatch at {len(diffs)} positions"
        if n_added_2nd != 0:
            r["status"] = "fail"
            r["details"]["error"] = f"not idempotent: 2nd pass added {n_added_2nd}"
        n_aa = len(ss)
        if n_aa > 10 and n_added == 0:
            r["status"] = "warn"
            r["details"]["warning"] = f"zero H for {n_aa}-residue protein"
        tests.append(r)
        stats["hydrogens"][r["status"]] += 1

        # --- Dihedrals ---
        phi, psi, omega = batch_results["dihedrals"][pos]
        phi, psi, omega = np.asarray(phi), np.asarray(psi), np.asarray(omega)
        r = {"test": "dihedrals", "status": "pass", "details": {
            "n_residues": len(phi),
        }}
        valid_phi = phi[~np.isnan(phi)]
        valid_omega = omega[~np.isnan(omega)]
        if len(valid_phi) > 0 and np.any(np.abs(valid_phi) > 180.01):
            r["status"] = "fail"
            r["details"]["error"] = "phi out of [-180, 180]"
        if len(valid_omega) > 0:
            n_trans = np.sum(np.abs(valid_omega) > 150)
            trans_frac = n_trans / len(valid_omega)
            r["details"]["trans_fraction"] = round(trans_frac, 3)
            if trans_frac < 0.90:
                r["status"] = "warn"
                r["details"]["warning"] = f"only {trans_frac:.1%} trans"
        tests.append(r)
        stats["dihedrals"][r["status"]] += 1

        # --- Select ---
        r = {"test": "select", "status": "pass", "details": {}}
        try:
            ca_mask = ferritin.select(structure, "CA")
            all_mask = ferritin.select(structure, "all")
            r["details"]["n_ca"] = int(ca_mask.sum())
            if all_mask.sum() != structure.atom_count:
                r["status"] = "fail"
                r["details"]["error"] = "'all' doesn't match atom_count"
        except Exception as e:
            r["status"] = "fail"
            r["details"]["error"] = str(e)[:200]
        tests.append(r)
        stats["select"][r["status"]] += 1

        all_results.append({"file": basename, "index": idx, "tests": tests})

    return stats, all_results, sasa_diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(pdb_dir, n_structures, output_file, n_threads, n_workers):
    files = sorted([
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.endswith((".pdb", ".cif"))
    ])[:n_structures]

    print(f"Parallel validation on {len(files)} structures")
    print(f"Oracles: FreeSASA={HAS_FREESASA}, FreeSASA-CLI={HAS_FREESASA_CLI}, "
          f"Gemmi={HAS_GEMMI}, Gemmi-H={HAS_GEMMI_H}")
    print(f"Threads: {n_threads or 'all'}, Workers: {n_workers}")
    print()

    t_total = time.time()

    # Phase 1: Load
    print("Phase 1: Loading")
    loaded = phase_load(files, n_threads)
    indices = [i for i, _ in loaded]
    structures = [s for _, s in loaded]
    n_loaded = len(structures)

    # Phase 2: Batch ferritin operations
    print(f"\nPhase 2: Batch operations ({n_loaded} structures)")
    batch_results = phase_ferritin_batch(structures, n_threads)

    # Phase 3: Oracle comparisons
    print(f"\nPhase 3: Oracle comparisons")
    oracle_results = phase_oracle(files, indices, n_workers)

    # Phase 4: Validate
    print(f"\nPhase 4: Validating results")
    t0 = time.time()
    stats, all_results, sasa_diffs = validate_all(
        files, indices, structures, batch_results, oracle_results
    )
    print(f"  Validated in {time.time() - t0:.1f}s")

    elapsed = time.time() - t_total

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY ({n_loaded} structures, {elapsed:.1f}s)")
    print(f"{'=' * 60}\n")

    for test_name, counts in sorted(stats.items()):
        total = sum(counts.values())
        pass_pct = counts["pass"] / total * 100 if total > 0 else 0
        print(f"  {test_name:20s}  pass={counts['pass']:4d}  warn={counts.get('warn', 0):3d}  "
              f"fail={counts.get('fail', 0):3d}  ({pass_pct:.1f}%)")

    if sasa_diffs:
        diffs = np.array(sasa_diffs)
        print(f"\n  SASA vs FreeSASA (C, Shrake-Rupley 960pts):")
        print(f"    Median relative diff: {np.median(diffs):.4f} ({np.median(diffs)*100:.2f}%)")
        print(f"    Mean relative diff:   {np.mean(diffs):.4f} ({np.mean(diffs)*100:.2f}%)")
        print(f"    Max relative diff:    {np.max(diffs):.4f} ({np.max(diffs)*100:.2f}%)")
        print(f"    Within 1%: {np.sum(diffs < 0.01)}/{len(diffs)}")
        print(f"    Within 5%: {np.sum(diffs < 0.05)}/{len(diffs)}")
        print(f"    Note: both ferritin and FreeSASA use ProtOr radii.")

    # Hydrogen summary
    h_added = [batch_results["hydrogens"][i][0] for i in range(n_loaded)]
    h_dssp_match = sum(
        1 for i in range(n_loaded)
        if batch_results["dssp"][i] == batch_results["dssp_post_h"][i]
    )
    h_idempotent = sum(
        1 for i in range(n_loaded)
        if batch_results["hydrogens_2nd"][i][0] == 0
    )
    print(f"\n  Hydrogen placement:")
    print(f"    Total H placed: {sum(h_added)}")
    print(f"    Mean per structure: {np.mean(h_added):.1f}")
    print(f"    DSSP match (pre/post H): {h_dssp_match}/{n_loaded}")
    print(f"    Idempotent: {h_idempotent}/{n_loaded}")

    # Save
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "n_structures": n_loaded,
                "n_files": len(files),
                "elapsed_s": round(elapsed, 1),
                "summary": dict(stats),
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"\n  Results saved to {output_file}")

    # Failures
    failures = [r for r in all_results if any(t["status"] == "fail" for t in r["tests"])]
    if failures:
        print(f"\n  FAILURES ({len(failures)} structures):")
        for r in failures[:20]:
            for t in r["tests"]:
                if t["status"] == "fail":
                    err = t.get("details", {}).get("error", "?")
                    print(f"    {r['file']:15s} {t['test']:20s} {err[:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ferritin parallel validation suite")
    parser.add_argument("--n-structures", type=int, default=5000)
    parser.add_argument("--pdb-dir", default="validation/pdbs_10k/")
    parser.add_argument("--output", default="validation/results_5k_parallel.json")
    parser.add_argument("--n-threads", type=int, default=None,
                        help="Rayon threads for ferritin (None = all cores)")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Process pool workers for oracle comparisons")
    args = parser.parse_args()

    run_validation(args.pdb_dir, args.n_structures, args.output,
                   args.n_threads, args.n_workers)
