#!/usr/bin/env python3
"""Ferritin comprehensive benchmark on large corpus.

Benchmarks all major ferritin features on N structures:
- Loading (PDB/mmCIF)
- SASA (Shrake-Rupley, ProtOr)
- DSSP (Kabsch-Sander)
- Backbone dihedrals
- H-bonds
- Selection language
- Energy computation (AMBER96)
- Hydrogen placement
- Structure preparation (prepare pipeline)
- Arrow/Parquet export (ingest)

Usage:
    python run_benchmark.py --pdb-dir pdbs_50k/ --n 50000 --threads 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def timed(name, fn, *args, **kwargs):
    """Run function and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_benchmark(pdb_dir, n_structures, n_threads, output_file):
    import ferritin

    # Collect input files
    files = sorted(Path(pdb_dir).glob("*.pdb")) + sorted(Path(pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[:n_structures]]
    n = len(files)
    print(f"Benchmark: {n} structures, {n_threads or 'all'} threads")
    print(f"=" * 60)

    results = {
        "n_structures": n,
        "n_threads": n_threads,
        "benchmarks": {},
    }

    # 1. Batch loading
    print("\n[1/9] Batch loading...")
    t0 = time.perf_counter()
    structures = ferritin.batch_load_tolerant(files, n_threads=n_threads)
    load_time = time.perf_counter() - t0
    n_loaded = len(structures)
    load_rate = n_loaded / load_time if load_time > 0 else 0
    print(f"  Loaded {n_loaded}/{n} in {load_time:.1f}s ({load_rate:.0f} structs/s)")
    results["benchmarks"]["load"] = {
        "n_loaded": n_loaded, "n_failed": n - n_loaded,
        "elapsed_s": round(load_time, 2), "rate": round(load_rate, 1),
    }

    if n_loaded == 0:
        print("No structures loaded, aborting.")
        return results

    # 2. Batch SASA
    print("\n[2/9] Batch SASA (ProtOr)...")
    t0 = time.perf_counter()
    sasa_values = ferritin.batch_total_sasa(structures, n_threads=n_threads, radii="protor")
    sasa_time = time.perf_counter() - t0
    sasa_rate = n_loaded / sasa_time if sasa_time > 0 else 0
    print(f"  {n_loaded} structures in {sasa_time:.1f}s ({sasa_rate:.0f} structs/s)")
    print(f"  Median SASA: {np.median(sasa_values):.0f} A²")
    results["benchmarks"]["sasa"] = {
        "elapsed_s": round(sasa_time, 2), "rate": round(sasa_rate, 1),
        "median": round(float(np.median(sasa_values)), 1),
    }

    # 3. Batch DSSP
    print("\n[3/9] Batch DSSP...")
    t0 = time.perf_counter()
    dssp_results = ferritin.batch_dssp(structures, n_threads=n_threads)
    dssp_time = time.perf_counter() - t0
    dssp_rate = n_loaded / dssp_time if dssp_time > 0 else 0
    print(f"  {n_loaded} structures in {dssp_time:.1f}s ({dssp_rate:.0f} structs/s)")
    results["benchmarks"]["dssp"] = {
        "elapsed_s": round(dssp_time, 2), "rate": round(dssp_rate, 1),
    }

    # 4. Batch dihedrals
    print("\n[4/9] Batch dihedrals...")
    t0 = time.perf_counter()
    dihed_results = ferritin.batch_dihedrals(structures, n_threads=n_threads)
    dihed_time = time.perf_counter() - t0
    dihed_rate = n_loaded / dihed_time if dihed_time > 0 else 0
    print(f"  {n_loaded} structures in {dihed_time:.1f}s ({dihed_rate:.0f} structs/s)")
    results["benchmarks"]["dihedrals"] = {
        "elapsed_s": round(dihed_time, 2), "rate": round(dihed_rate, 1),
    }

    # 5. Batch H-bonds
    print("\n[5/9] Batch H-bonds...")
    t0 = time.perf_counter()
    hbond_results = ferritin.batch_backbone_hbonds(structures, n_threads=n_threads)
    hbond_time = time.perf_counter() - t0
    hbond_rate = n_loaded / hbond_time if hbond_time > 0 else 0
    print(f"  {n_loaded} structures in {hbond_time:.1f}s ({hbond_rate:.0f} structs/s)")
    results["benchmarks"]["hbonds"] = {
        "elapsed_s": round(hbond_time, 2), "rate": round(hbond_rate, 1),
    }

    # 6. Batch hydrogen placement
    print("\n[6/9] Batch H placement...")
    t0 = time.perf_counter()
    h_results = ferritin.batch_place_peptide_hydrogens(structures, n_threads=n_threads)
    h_time = time.perf_counter() - t0
    h_rate = n_loaded / h_time if h_time > 0 else 0
    total_h = sum(r[0] for r in h_results)
    print(f"  {n_loaded} structures in {h_time:.1f}s ({h_rate:.0f} structs/s)")
    print(f"  Total H placed: {total_h}")
    results["benchmarks"]["hydrogens"] = {
        "elapsed_s": round(h_time, 2), "rate": round(h_rate, 1),
        "total_h_placed": total_h,
    }

    # 7. Energy computation (subsample — O(N²) is slow for large structures)
    n_energy = min(1000, n_loaded)
    print(f"\n[7/9] Energy computation ({n_energy} structures)...")
    t0 = time.perf_counter()
    energies = []
    for s in structures[:n_energy]:
        try:
            e = ferritin.compute_energy(s, units="kJ/mol")
            energies.append(e["total"])
        except Exception:
            pass
    energy_time = time.perf_counter() - t0
    energy_rate = len(energies) / energy_time if energy_time > 0 else 0
    print(f"  {len(energies)} structures in {energy_time:.1f}s ({energy_rate:.0f} structs/s)")
    if energies:
        print(f"  Median total energy: {np.median(energies):.0f} kJ/mol")
    results["benchmarks"]["energy"] = {
        "n_computed": len(energies), "elapsed_s": round(energy_time, 2),
        "rate": round(energy_rate, 1),
    }

    # 8. Batch prepare (subsample)
    n_prep = min(500, n_loaded)
    print(f"\n[8/9] Batch prepare ({n_prep} structures)...")
    t0 = time.perf_counter()
    try:
        prep_reports = ferritin.batch_prepare(
            structures[:n_prep],
            reconstruct=False,
            hydrogens="backbone",
            minimize=True,
            minimize_steps=50,
            minimize_method="lbfgs",
            n_threads=n_threads,
        )
        prep_time = time.perf_counter() - t0
        prep_rate = n_prep / prep_time if prep_time > 0 else 0
        n_converged = sum(1 for r in prep_reports if r.converged)
        print(f"  {n_prep} structures in {prep_time:.1f}s ({prep_rate:.0f} structs/s)")
        print(f"  Converged: {n_converged}/{n_prep}")
        results["benchmarks"]["prepare"] = {
            "n_prepared": n_prep, "elapsed_s": round(prep_time, 2),
            "rate": round(prep_rate, 1), "n_converged": n_converged,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        results["benchmarks"]["prepare"] = {"error": str(e)}

    # 9. Timing summary
    total_time = (load_time + sasa_time + dssp_time + dihed_time +
                  hbond_time + h_time + energy_time)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({n_loaded} structures)")
    print(f"{'=' * 60}")
    print(f"  {'Operation':<25s} {'Time(s)':>8s} {'Rate(s/s)':>10s}")
    print(f"  {'-'*45}")
    for name, bm in results["benchmarks"].items():
        if "elapsed_s" in bm:
            rate = bm.get("rate", 0)
            print(f"  {name:<25s} {bm['elapsed_s']:8.1f} {rate:10.0f}")
    print(f"  {'TOTAL':<25s} {total_time:8.1f}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ferritin benchmark")
    parser.add_argument("--pdb-dir", required=True, help="Directory with PDB files")
    parser.add_argument("--n", type=int, default=50000, help="Max structures")
    parser.add_argument("--threads", type=int, default=0, help="Threads (0=all)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON")
    args = parser.parse_args()

    run_benchmark(args.pdb_dir, args.n, args.threads or None, args.output)
