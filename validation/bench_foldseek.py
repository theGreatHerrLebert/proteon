#!/usr/bin/env python3
"""FoldSeek benchmark: compare proteon TM-align vs FoldSeek structural search.

Creates a FoldSeek database from test PDBs, runs all-vs-all structural search,
then compares FoldSeek TM-scores against proteon TM-align scores.

This benchmarks:
1. FoldSeek 3Di+AA search accuracy vs rigorous TM-align
2. Speed: FoldSeek prefilter+align vs proteon TM-align (sequential & parallel)
3. Rank correlation: do both methods agree on which structures are most similar?

Usage:
    python validation/bench_foldseek.py [--n-structures 50] [--pdb-dir test-pdbs/]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

from proteon_connector import py_align_funcs, py_io

FOLDSEEK_BIN = "/scratch/TMAlign/foldseek/bin/foldseek"


def run_foldseek_allvsall(pdb_dir: str, pdb_files: list, tmpdir: str) -> dict:
    """Run FoldSeek easy-search all-vs-all and parse results.

    Returns dict of (query, target) -> {tmscore, rmsd, alnlen, ...}
    """
    # FoldSeek easy-search: query dir vs target dir
    outfile = os.path.join(tmpdir, "results.tsv")

    # Create symlink dirs with only our selected files
    qdir = os.path.join(tmpdir, "query")
    tdir = os.path.join(tmpdir, "target")
    os.makedirs(qdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    for f in pdb_files:
        src = os.path.join(pdb_dir, f)
        os.symlink(os.path.abspath(src), os.path.join(qdir, f))
        os.symlink(os.path.abspath(src), os.path.join(tdir, f))

    t0 = time.time()
    result = subprocess.run(
        [
            FOLDSEEK_BIN, "easy-search",
            qdir, tdir, outfile, os.path.join(tmpdir, "tmp"),
            "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,qtmscore,ttmscore,rmsd,qlen,tlen",
            "--exhaustive-search", "1",  # all-vs-all, no prefilter
            "-e", "inf",  # no e-value filter
            "--max-accept", "10000",  # accept all hits
        ],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FoldSeek error: {result.stderr[:500]}")
        return {}, elapsed

    # Parse results
    pairs = {}
    with open(outfile) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 18:
                continue
            query = os.path.splitext(parts[0])[0]
            target = os.path.splitext(parts[1])[0]
            if query == target:
                continue
            pairs[(query, target)] = {
                "fident": float(parts[2]),
                "alnlen": int(parts[3]),
                "alntmscore": float(parts[12]),
                "qtmscore": float(parts[13]),
                "ttmscore": float(parts[14]),
                "rmsd": float(parts[15]),
                "qlen": int(parts[16]),
                "tlen": int(parts[17]),
                "evalue": float(parts[10]),
            }

    return pairs, elapsed


def main():
    parser = argparse.ArgumentParser(description="FoldSeek vs proteon benchmark")
    parser.add_argument("--n-structures", type=int, default=50)
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/test-pdbs/")
    parser.add_argument("--output", default="validation/foldseek_bench.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(FOLDSEEK_BIN):
        print(f"ERROR: FoldSeek not found at {FOLDSEEK_BIN}")
        sys.exit(1)

    # Collect PDB files
    all_pdbs = sorted([
        f for f in os.listdir(args.pdb_dir)
        if f.endswith(".pdb")
    ])

    if args.n_structures < len(all_pdbs):
        random.seed(args.seed)
        pdbs = sorted(random.sample(all_pdbs, args.n_structures))
    else:
        pdbs = all_pdbs

    print(f"Benchmarking {len(pdbs)} structures")
    print(f"PDB dir: {args.pdb_dir}")
    print()

    # --- Phase 1: FoldSeek all-vs-all ---
    print("Phase 1: FoldSeek exhaustive all-vs-all search...")
    with tempfile.TemporaryDirectory() as tmpdir:
        fs_pairs, fs_time = run_foldseek_allvsall(args.pdb_dir, pdbs, tmpdir)
    print(f"  {len(fs_pairs)} pairs found in {fs_time:.1f}s")

    # --- Phase 2: Proteon TM-align (parallel batch) ---
    print("\nPhase 2: Proteon TM-align (parallel, many-to-many)...")
    names = [os.path.splitext(f)[0] for f in pdbs]
    paths = [os.path.join(args.pdb_dir, f) for f in pdbs]

    # Load structures
    loaded = {}
    for name, path in zip(names, paths):
        try:
            loaded[name] = py_io.load(path)
        except Exception:
            pass
    print(f"  {len(loaded)} structures loaded")

    loadable_names = sorted(loaded.keys())
    loadable_ptrs = [loaded[n] for n in loadable_names]

    t0 = time.time()
    raw_results = py_align_funcs.tm_align_many_to_many(
        loadable_ptrs, loadable_ptrs, None, None, False
    )
    fe_time = time.time() - t0
    print(f"  {len(raw_results)} pairs computed in {fe_time:.1f}s")

    # Build proteon results dict
    fe_pairs = {}
    for qi, ti, r in raw_results:
        qname = loadable_names[qi]
        tname = loadable_names[ti]
        if qname == tname:
            continue
        fe_pairs[(qname, tname)] = {
            # proteon chain1 = normL2, chain2 = normL1
            "qtmscore": r.tm_score_chain2,  # norm by query length
            "ttmscore": r.tm_score_chain1,  # norm by target length
            "rmsd": r.rmsd,
            "n_aligned": r.n_aligned,
        }

    # --- Phase 3: Compare ---
    print(f"\nPhase 3: Comparing {len(fs_pairs)} FoldSeek pairs vs proteon...")

    # Find common pairs
    common_keys = set(fs_pairs.keys()) & set(fe_pairs.keys())
    print(f"  Common pairs: {len(common_keys)}")

    if not common_keys:
        print("  No common pairs to compare!")
        return

    # Compute TM-score differences
    tm_diffs_q = []
    tm_diffs_t = []
    rmsd_diffs = []
    fs_tms = []
    fe_tms = []
    comparison_data = []

    for key in sorted(common_keys):
        fs = fs_pairs[key]
        fe = fe_pairs[key]

        d_tmq = abs(fs["qtmscore"] - fe["qtmscore"])
        d_tmt = abs(fs["ttmscore"] - fe["ttmscore"])
        d_rmsd = abs(fs["rmsd"] - fe["rmsd"])

        tm_diffs_q.append(d_tmq)
        tm_diffs_t.append(d_tmt)
        rmsd_diffs.append(d_rmsd)
        fs_tms.append(fs["qtmscore"])
        fe_tms.append(fe["qtmscore"])

        comparison_data.append({
            "pair": f"{key[0]}_vs_{key[1]}",
            "fs_qtm": round(fs["qtmscore"], 4),
            "fe_qtm": round(fe["qtmscore"], 4),
            "fs_ttm": round(fs["ttmscore"], 4),
            "fe_ttm": round(fe["ttmscore"], 4),
            "fs_rmsd": round(fs["rmsd"], 2),
            "fe_rmsd": round(fe["rmsd"], 2),
        })

    tm_diffs_q = np.array(tm_diffs_q)
    tm_diffs_t = np.array(tm_diffs_t)
    rmsd_diffs = np.array(rmsd_diffs)
    fs_tms = np.array(fs_tms)
    fe_tms = np.array(fe_tms)

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(fs_tms, fe_tms)

    # Summary
    n_total_pairs = len(loadable_names) * (len(loadable_names) - 1)
    print(f"\n{'='*65}")
    print(f"FOLDSEEK vs PROTEON BENCHMARK ({len(common_keys)} pairs)")
    print(f"{'='*65}\n")

    print("  TM-score agreement (FoldSeek vs proteon TM-align):")
    print(f"    Query-norm:  median |diff| = {np.median(tm_diffs_q):.4f}, "
          f"max = {np.max(tm_diffs_q):.4f}, "
          f"within 0.01: {np.sum(tm_diffs_q < 0.01)}/{len(tm_diffs_q)}")
    print(f"    Target-norm: median |diff| = {np.median(tm_diffs_t):.4f}, "
          f"max = {np.max(tm_diffs_t):.4f}, "
          f"within 0.01: {np.sum(tm_diffs_t < 0.01)}/{len(tm_diffs_t)}")

    print(f"\n  RMSD agreement:")
    print(f"    Median |diff| = {np.median(rmsd_diffs):.2f} Å, "
          f"max = {np.max(rmsd_diffs):.2f} Å")

    print(f"\n  Rank correlation (Spearman):")
    print(f"    rho = {rho:.4f}, p = {pval:.2e}")

    print(f"\n  Speed:")
    print(f"    FoldSeek (exhaustive all-vs-all):  {fs_time:.1f}s "
          f"({len(fs_pairs)/fs_time:.0f} pairs/s)")
    print(f"    Proteon (parallel many-to-many):  {fe_time:.1f}s "
          f"({n_total_pairs/fe_time:.0f} pairs/s)")
    speedup = fs_time / fe_time if fe_time > 0 else 0
    print(f"    Proteon speedup vs FoldSeek: {speedup:.1f}x")

    # FoldSeek coverage: what fraction of all possible pairs did FoldSeek report?
    fs_coverage = len(fs_pairs) / n_total_pairs * 100
    print(f"\n  FoldSeek coverage: {len(fs_pairs)}/{n_total_pairs} "
          f"({fs_coverage:.1f}%) pairs reported")
    print(f"  (FoldSeek may skip very dissimilar pairs even with -e inf)")

    # Worst disagreements
    worst_idx = np.argsort(tm_diffs_q)[-5:][::-1]
    if tm_diffs_q[worst_idx[0]] > 0.01:
        print(f"\n  Worst TM-score disagreements:")
        sorted_data = sorted(comparison_data, key=lambda x: abs(x["fs_qtm"] - x["fe_qtm"]), reverse=True)
        for d in sorted_data[:5]:
            print(f"    {d['pair']:30s}  fs={d['fs_qtm']:.4f}  fe={d['fe_qtm']:.4f}  "
                  f"|d|={abs(d['fs_qtm'] - d['fe_qtm']):.4f}")

    # Save
    if args.output:
        summary = {
            "n_structures": len(loadable_names),
            "n_common_pairs": len(common_keys),
            "foldseek_pairs": len(fs_pairs),
            "foldseek_time_s": round(fs_time, 1),
            "proteon_time_s": round(fe_time, 1),
            "proteon_speedup": round(speedup, 1),
            "tm_q_median_diff": round(float(np.median(tm_diffs_q)), 5),
            "tm_q_max_diff": round(float(np.max(tm_diffs_q)), 5),
            "spearman_rho": round(rho, 4),
            "spearman_pval": float(pval),
            "comparisons": comparison_data,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
