#!/usr/bin/env python3
"""Alignment benchmark: proteon (Rust) vs C++ USAlign.

Runs TM-align on all pairs from a set of PDB structures using both
proteon and the C++ USAlign binary, comparing TM-scores, RMSD,
aligned lengths, and timing.

Usage:
    python validation/bench_alignment.py [--n-structures 50] [--pdb-dir test-pdbs/]
    python validation/bench_alignment.py --n-structures 104  # all test PDBs
    python validation/bench_alignment.py --n-structures 20 --output validation/align_bench.json
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from typing import NamedTuple, Optional

import numpy as np

from proteon_connector import py_align_funcs, py_io

USALIGN_BIN = "/scratch/TMAlign/USAlign/USalign"


class CppResult(NamedTuple):
    tm1: float  # normalized by chain1
    tm2: float  # normalized by chain2
    rmsd: float
    n_aligned: int
    len1: int
    len2: int
    time_ms: float


def run_cpp_usalign(path1: str, path2: str) -> Optional[CppResult]:
    """Run C++ USAlign and parse tabular output."""
    try:
        t0 = time.time()
        result = subprocess.run(
            [USALIGN_BIN, path1, path2, "-outfmt", "2"],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = (time.time() - t0) * 1000
        for line in result.stdout.strip().split("\n"):
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 11:
                return CppResult(
                    tm1=float(parts[2]),
                    tm2=float(parts[3]),
                    rmsd=float(parts[4]),
                    n_aligned=int(parts[10]),
                    len1=int(parts[8]),
                    len2=int(parts[9]),
                    time_ms=elapsed,
                )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def run_proteon(pdb1, pdb2) -> dict:
    """Run proteon TM-align and return results + timing."""
    t0 = time.time()
    r = py_align_funcs.tm_align_pair(pdb1, pdb2)
    elapsed = (time.time() - t0) * 1000
    return {
        # proteon convention: chain1 = norm by L2, chain2 = norm by L1
        # C++ convention: tm1 = norm by L1, tm2 = norm by L2
        # So: proteon.chain1 <-> cpp.tm2, proteon.chain2 <-> cpp.tm1
        "tm_normL1": r.tm_score_chain2,
        "tm_normL2": r.tm_score_chain1,
        "rmsd": r.rmsd,
        "n_aligned": r.n_aligned,
        "time_ms": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Alignment benchmark")
    parser.add_argument("--n-structures", type=int, default=50)
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/test-pdbs/")
    parser.add_argument("--output", default="validation/align_bench.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(USALIGN_BIN):
        print(f"ERROR: C++ USAlign not found at {USALIGN_BIN}")
        sys.exit(1)

    # Collect PDB files
    all_pdbs = sorted([
        os.path.join(args.pdb_dir, f)
        for f in os.listdir(args.pdb_dir)
        if f.endswith(".pdb")
    ])

    if args.n_structures < len(all_pdbs):
        random.seed(args.seed)
        pdbs = sorted(random.sample(all_pdbs, args.n_structures))
    else:
        pdbs = all_pdbs

    n_pairs = len(pdbs) * (len(pdbs) - 1) // 2
    print(f"Benchmarking {len(pdbs)} structures → {n_pairs} pairs")
    print(f"PDB dir: {args.pdb_dir}")
    print(f"C++ USAlign: {USALIGN_BIN}")
    print()

    # Preload all structures for proteon
    print("Loading structures into proteon...", end=" ", flush=True)
    t0 = time.time()
    loaded = {}
    load_failures = []
    for p in pdbs:
        try:
            loaded[p] = py_io.load(p)
        except Exception as e:
            load_failures.append((os.path.basename(p), str(e)[:100]))
    load_time = time.time() - t0
    print(f"{len(loaded)} loaded in {load_time:.1f}s ({len(load_failures)} failures)")

    if load_failures:
        for name, err in load_failures[:5]:
            print(f"  SKIP {name}: {err}")

    # Generate pairs from loaded structures only
    loadable = sorted(loaded.keys())
    pairs = list(combinations(loadable, 2))
    n_pairs = len(pairs)
    print(f"\nRunning {n_pairs} pair alignments...\n")

    # Run benchmarks
    results = []
    tm_diffs_L1 = []
    tm_diffs_L2 = []
    rmsd_diffs = []
    nalign_diffs = []
    proteon_times = []
    cpp_times = []
    n_fail = 0

    t0 = time.time()
    for i, (p1, p2) in enumerate(pairs):
        name1 = os.path.splitext(os.path.basename(p1))[0]
        name2 = os.path.splitext(os.path.basename(p2))[0]
        pair_id = f"{name1}_vs_{name2}"

        # C++ USAlign
        cpp = run_cpp_usalign(p1, p2)
        if cpp is None:
            n_fail += 1
            continue

        # Proteon
        try:
            fe = run_proteon(loaded[p1], loaded[p2])
        except Exception as e:
            n_fail += 1
            continue

        # Compute diffs
        d_tm1 = abs(fe["tm_normL1"] - cpp.tm1)
        d_tm2 = abs(fe["tm_normL2"] - cpp.tm2)
        d_rmsd = abs(fe["rmsd"] - cpp.rmsd)
        d_nalign = abs(fe["n_aligned"] - cpp.n_aligned)

        tm_diffs_L1.append(d_tm1)
        tm_diffs_L2.append(d_tm2)
        rmsd_diffs.append(d_rmsd)
        nalign_diffs.append(d_nalign)
        proteon_times.append(fe["time_ms"])
        cpp_times.append(cpp.time_ms)

        results.append({
            "pair": pair_id,
            "cpp_tm1": cpp.tm1, "cpp_tm2": cpp.tm2,
            "fe_tm1": fe["tm_normL1"], "fe_tm2": fe["tm_normL2"],
            "cpp_rmsd": cpp.rmsd, "fe_rmsd": fe["rmsd"],
            "cpp_nalign": cpp.n_aligned, "fe_nalign": fe["n_aligned"],
            "cpp_ms": round(cpp.time_ms, 1), "fe_ms": round(fe["time_ms"], 1),
        })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_pairs - i - 1) / rate
            print(f"  [{i+1}/{n_pairs}] {rate:.1f} pairs/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    # Summary
    tm_diffs_L1 = np.array(tm_diffs_L1)
    tm_diffs_L2 = np.array(tm_diffs_L2)
    rmsd_diffs = np.array(rmsd_diffs)
    nalign_diffs = np.array(nalign_diffs)
    proteon_times = np.array(proteon_times)
    cpp_times = np.array(cpp_times)

    print(f"\n{'='*65}")
    print(f"ALIGNMENT BENCHMARK ({len(results)} pairs, {elapsed:.1f}s, {n_fail} failures)")
    print(f"{'='*65}\n")

    print("  TM-score agreement (proteon vs C++ USAlign):")
    print(f"    TM (norm L1): median |diff| = {np.median(tm_diffs_L1):.5f}, "
          f"max = {np.max(tm_diffs_L1):.5f}, "
          f"within 1e-3: {np.sum(tm_diffs_L1 < 1e-3)}/{len(tm_diffs_L1)}")
    print(f"    TM (norm L2): median |diff| = {np.median(tm_diffs_L2):.5f}, "
          f"max = {np.max(tm_diffs_L2):.5f}, "
          f"within 1e-3: {np.sum(tm_diffs_L2 < 1e-3)}/{len(tm_diffs_L2)}")

    print(f"\n  RMSD agreement:")
    print(f"    Median |diff| = {np.median(rmsd_diffs):.3f} Å, "
          f"max = {np.max(rmsd_diffs):.3f} Å, "
          f"within 0.1 Å: {np.sum(rmsd_diffs < 0.1)}/{len(rmsd_diffs)}")

    print(f"\n  Aligned length agreement:")
    print(f"    Median |diff| = {np.median(nalign_diffs):.0f}, "
          f"max = {np.max(nalign_diffs):.0f}, "
          f"exact match: {np.sum(nalign_diffs == 0)}/{len(nalign_diffs)}")

    print(f"\n  Speed:")
    print(f"    Proteon: median {np.median(proteon_times):.1f} ms, "
          f"mean {np.mean(proteon_times):.1f} ms")
    print(f"    C++ USAlign: median {np.median(cpp_times):.1f} ms, "
          f"mean {np.mean(cpp_times):.1f} ms")
    ratio = np.median(cpp_times) / np.median(proteon_times) if np.median(proteon_times) > 0 else 0
    print(f"    Proteon/C++ median ratio: {ratio:.2f}x")

    # Worst outliers
    worst_idx = np.argsort(tm_diffs_L1)[-5:][::-1]
    if tm_diffs_L1[worst_idx[0]] > 0.001:
        print(f"\n  Worst TM-score disagreements:")
        for idx in worst_idx:
            r = results[idx]
            print(f"    {r['pair']:30s}  cpp={r['cpp_tm1']:.4f}/{r['cpp_tm2']:.4f}  "
                  f"fe={r['fe_tm1']:.4f}/{r['fe_tm2']:.4f}  "
                  f"|d|={tm_diffs_L1[idx]:.5f}")

    # Save
    if args.output:
        summary = {
            "n_structures": len(loadable),
            "n_pairs": len(results),
            "n_failures": n_fail,
            "elapsed_s": round(elapsed, 1),
            "tm_L1_median_diff": round(float(np.median(tm_diffs_L1)), 6),
            "tm_L1_max_diff": round(float(np.max(tm_diffs_L1)), 6),
            "tm_L2_median_diff": round(float(np.median(tm_diffs_L2)), 6),
            "tm_L2_max_diff": round(float(np.max(tm_diffs_L2)), 6),
            "rmsd_median_diff": round(float(np.median(rmsd_diffs)), 4),
            "rmsd_max_diff": round(float(np.max(rmsd_diffs)), 4),
            "proteon_median_ms": round(float(np.median(proteon_times)), 1),
            "cpp_median_ms": round(float(np.median(cpp_times)), 1),
            "speed_ratio": round(ratio, 2),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
