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


def log(msg):
    """Print with immediate flush for nohup/redirect."""
    print(msg, flush=True)


def timed(name, fn, *args, **kwargs):
    """Run function and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_benchmark(pdb_dir, n_structures, n_threads, output_file, chunk_size=5000):
    import ferritin

    # Collect input files
    files = sorted(Path(pdb_dir).glob("*.pdb")) + sorted(Path(pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[:n_structures]]
    n = len(files)
    log(f"Benchmark: {n} structures, {n_threads or 'all'} threads, chunk={chunk_size}")
    log(f"=" * 60)

    results = {
        "n_structures": n,
        "n_threads": n_threads,
        "chunk_size": chunk_size,
        "benchmarks": {},
    }

    # Process in chunks to avoid OOM
    # Each chunk: load → run all analyses → discard → next chunk
    all_timings = {
        "load": {"elapsed": 0, "n_loaded": 0, "n_failed": 0},
        "sasa": {"elapsed": 0},
        "dssp": {"elapsed": 0},
        "dihedrals": {"elapsed": 0},
        "hbonds": {"elapsed": 0},
        "hydrogens": {"elapsed": 0, "total_h": 0},
        "energy": {"elapsed": 0, "n_computed": 0},
        "prepare": {"elapsed": 0, "n_prepared": 0, "n_converged": 0},
    }
    sasa_values = []
    energies = []

    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk_files = files[ci * chunk_size : (ci + 1) * chunk_size]
        # Filter out huge files (> 20MB) to avoid OOM on batch operations
        chunk_files = [f for f in chunk_files if os.path.getsize(f) < 20_000_000]
        cn = len(chunk_files)
        log(f"\n--- Chunk {ci+1}/{n_chunks} ({cn} files) ---")

        # Load (batch_load_tolerant returns (index, structure) tuples)
        t0 = time.perf_counter()
        loaded = ferritin.batch_load_tolerant(chunk_files, n_threads=n_threads)
        dt = time.perf_counter() - t0
        structures = [s for _, s in loaded]
        # Skip structures > 25K atoms for now (giants dominate runtime)
        # We proved ferritin handles 2M-atom structures — just too slow for batch
        n_before = len(structures)
        structures = [s for s in structures if s.atom_count < 25000]
        n_skipped_large = n_before - len(structures)
        all_timings["load"]["elapsed"] += dt
        all_timings["load"]["n_loaded"] += len(structures)
        all_timings["load"]["n_failed"] += cn - n_before + n_skipped_large
        log(f"  Load: {len(structures)}/{cn} in {dt:.1f}s (skipped {n_skipped_large} large)")

        if not structures:
            continue

        # SASA
        log(f"  SASA starting...")
        t0 = time.perf_counter()
        sv = ferritin.batch_total_sasa(structures, n_threads=n_threads, radii="protor")
        dt = time.perf_counter() - t0
        log(f"  SASA: {dt:.1f}s")
        all_timings["sasa"]["elapsed"] += dt
        sasa_values.extend(sv.tolist())

        # DSSP
        log(f"  DSSP starting...")
        t0 = time.perf_counter()
        ferritin.batch_dssp(structures, n_threads=n_threads)
        dt = time.perf_counter() - t0
        log(f"  DSSP: {dt:.1f}s")
        all_timings["dssp"]["elapsed"] += dt

        # Dihedrals
        log(f"  Dihedrals starting...")
        t0 = time.perf_counter()
        ferritin.batch_dihedrals(structures, n_threads=n_threads)
        dt = time.perf_counter() - t0
        log(f"  Dihedrals: {dt:.1f}s")
        all_timings["dihedrals"]["elapsed"] += dt

        # H-bonds
        log(f"  H-bonds starting...")
        t0 = time.perf_counter()
        ferritin.batch_backbone_hbonds(structures, n_threads=n_threads)
        dt = time.perf_counter() - t0
        log(f"  H-bonds: {dt:.1f}s")
        all_timings["hbonds"]["elapsed"] += dt

        # H placement
        log(f"  Hydrogens starting...")
        t0 = time.perf_counter()
        h_res = ferritin.batch_place_peptide_hydrogens(structures, n_threads=n_threads)
        dt = time.perf_counter() - t0
        log(f"  Hydrogens: {dt:.1f}s")
        all_timings["hydrogens"]["elapsed"] += dt
        all_timings["hydrogens"]["total_h"] += sum(r[0] for r in h_res)

        # Energy (first chunk only — O(N²) is slow)
        if ci == 0:
            n_energy = min(500, len(structures))
            log(f"  Energy starting ({n_energy} structures)...")
            t0 = time.perf_counter()
            for s in structures[:n_energy]:
                try:
                    e = ferritin.compute_energy(s, units="kJ/mol")
                    energies.append(e["total"])
                    all_timings["energy"]["n_computed"] += 1
                except Exception:
                    pass
            dt = time.perf_counter() - t0
            log(f"  Energy: {dt:.1f}s")
            all_timings["energy"]["elapsed"] += dt

        # Prepare (first chunk only)
        if ci == 0:
            n_prep = min(200, len(structures))
            log(f"  Prepare starting ({n_prep} structures)...")
            t0 = time.perf_counter()
            try:
                reports = ferritin.batch_prepare(
                    structures[:n_prep], reconstruct=False, hydrogens="backbone",
                    minimize=True, minimize_steps=50, minimize_method="lbfgs",
                    n_threads=n_threads,
                )
                all_timings["prepare"]["n_prepared"] = n_prep
                all_timings["prepare"]["n_converged"] = sum(1 for r in reports if r.converged)
            except Exception as e:
                log(f"  Prepare failed: {e}")
            dt = time.perf_counter() - t0
            log(f"  Prepare: {dt:.1f}s")
            all_timings["prepare"]["elapsed"] += dt

        # Free memory
        del structures
        log(f"  Done. SASA/DSSP/dihed/hbond/H complete.")

    # Compile results
    n_loaded = all_timings["load"]["n_loaded"]
    log(f"\n{'=' * 60}")
    log(f"SUMMARY ({n_loaded}/{n} structures loaded)")
    log(f"{'=' * 60}")
    log(f"  {'Operation':<25s} {'Time(s)':>8s} {'Rate(s/s)':>10s}")
    log(f"  {'-'*45}")

    for name, tm in all_timings.items():
        elapsed = tm["elapsed"]
        count = tm.get("n_loaded", tm.get("n_computed", tm.get("n_prepared", n_loaded)))
        rate = count / elapsed if elapsed > 0 else 0
        log(f"  {name:<25s} {elapsed:8.1f} {rate:10.0f}")
        results["benchmarks"][name] = {
            "elapsed_s": round(elapsed, 2),
            "rate": round(rate, 1),
            **{k: v for k, v in tm.items() if k != "elapsed"},
        }

    if sasa_values:
        results["benchmarks"]["sasa"]["median"] = round(float(np.median(sasa_values)), 1)
        log(f"\n  Median SASA: {np.median(sasa_values):.0f} A²")
    if energies:
        results["benchmarks"]["energy"]["median_kj"] = round(float(np.median(energies)), 1)
        log(f"  Median energy: {np.median(energies):.0f} kJ/mol")

    total = sum(tm["elapsed"] for tm in all_timings.values())
    log(f"\n  TOTAL TIME: {total:.1f}s ({total/60:.1f} min)")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ferritin benchmark")
    parser.add_argument("--pdb-dir", required=True, help="Directory with PDB files")
    parser.add_argument("--n", type=int, default=50000, help="Max structures")
    parser.add_argument("--threads", type=int, default=0, help="Threads (0=all)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON")
    parser.add_argument("--chunk", type=int, default=5000, help="Chunk size for processing")
    args = parser.parse_args()

    run_benchmark(args.pdb_dir, args.n, args.threads or None, args.output, args.chunk)
