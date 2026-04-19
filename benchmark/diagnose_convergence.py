#!/usr/bin/env python3
"""Diagnose why prepare() minimization doesn't converge for some structures.

Runs the first N structures through the same benchmark config (50 steps, tol=0.1),
then re-runs non-converged ones with 500 steps. Reports convergence breakdown.

Run on monster3:
    python benchmark/diagnose_convergence.py --pdb-dir ../pdbs_50k --n 200
"""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="convergence_diagnosis.json")
    args = parser.parse_args()

    import proteon

    # Collect first N files (same order as benchmark)
    files = sorted(Path(args.pdb_dir).glob("*.pdb")) + sorted(Path(args.pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[: args.n * 3]]  # oversample to account for failed loads
    files = [f for f in files if os.path.getsize(f) < 20_000_000]

    print(f"Loading up to {len(files)} files...", flush=True)
    loaded = proteon.batch_load_tolerant(files, n_threads=0)
    structures = [(Path(files[i]).stem, s) for i, s in loaded if s.atom_count < 25000]
    structures = structures[: args.n]
    print(f"Got {len(structures)} structures after filter\n", flush=True)

    # Round 1: same settings as benchmark (50 steps, tol=0.1)
    print("=== Round 1: 50 steps, tol=0.1 (benchmark config) ===", flush=True)
    t0 = time.perf_counter()
    structs_only = [s for _, s in structures]
    reports_50 = proteon.batch_prepare(
        structs_only, reconstruct=False, hydrogens="backbone",
        minimize=True, minimize_steps=50, minimize_method="lbfgs",
        gradient_tolerance=0.1, n_threads=0,
    )
    dt = time.perf_counter() - t0
    n_conv_50 = sum(1 for r in reports_50 if r.converged)
    print(f"  {n_conv_50}/{len(reports_50)} converged in {dt:.1f}s\n", flush=True)

    # Look at step distribution for ones that didn't converge
    steps_dist = {}
    for r in reports_50:
        if not r.converged:
            bucket = (r.minimizer_steps // 10) * 10
            steps_dist[bucket] = steps_dist.get(bucket, 0) + 1
    print(f"  Non-converged step distribution: {sorted(steps_dist.items())}", flush=True)

    # Note: we need to work on copies since prepare modifies in-place
    # Round 2: 500 steps, tol=0.1 (is 50 too few?)
    print("\n=== Round 2: 500 steps, tol=0.1 (more steps) ===", flush=True)
    structs_fresh = [proteon.load(files[i]) for i, _ in enumerate(loaded) if i < len(loaded)]
    structs_fresh = [s for s in structs_fresh if s.atom_count < 25000][: args.n]
    t0 = time.perf_counter()
    reports_500 = proteon.batch_prepare(
        structs_fresh, reconstruct=False, hydrogens="backbone",
        minimize=True, minimize_steps=500, minimize_method="lbfgs",
        gradient_tolerance=0.1, n_threads=0,
    )
    dt = time.perf_counter() - t0
    n_conv_500 = sum(1 for r in reports_500 if r.converged)
    print(f"  {n_conv_500}/{len(reports_500)} converged in {dt:.1f}s", flush=True)
    gained = n_conv_500 - n_conv_50
    print(f"  (+{gained} vs 50-step run)\n", flush=True)

    # Round 3: 50 steps, looser tolerance (is 0.1 too tight?)
    print("=== Round 3: 50 steps, tol=1.0 (looser tolerance) ===", flush=True)
    structs_fresh = [proteon.load(files[i]) for i, _ in enumerate(loaded) if i < len(loaded)]
    structs_fresh = [s for s in structs_fresh if s.atom_count < 25000][: args.n]
    t0 = time.perf_counter()
    reports_tol1 = proteon.batch_prepare(
        structs_fresh, reconstruct=False, hydrogens="backbone",
        minimize=True, minimize_steps=50, minimize_method="lbfgs",
        gradient_tolerance=1.0, n_threads=0,
    )
    dt = time.perf_counter() - t0
    n_conv_tol1 = sum(1 for r in reports_tol1 if r.converged)
    print(f"  {n_conv_tol1}/{len(reports_tol1)} converged in {dt:.1f}s\n", flush=True)

    # Round 4: 500 steps, tol=1.0 (both relaxed)
    print("=== Round 4: 500 steps, tol=1.0 (both relaxed) ===", flush=True)
    structs_fresh = [proteon.load(files[i]) for i, _ in enumerate(loaded) if i < len(loaded)]
    structs_fresh = [s for s in structs_fresh if s.atom_count < 25000][: args.n]
    t0 = time.perf_counter()
    reports_both = proteon.batch_prepare(
        structs_fresh, reconstruct=False, hydrogens="backbone",
        minimize=True, minimize_steps=500, minimize_method="lbfgs",
        gradient_tolerance=1.0, n_threads=0,
    )
    dt = time.perf_counter() - t0
    n_conv_both = sum(1 for r in reports_both if r.converged)
    print(f"  {n_conv_both}/{len(reports_both)} converged in {dt:.1f}s\n", flush=True)

    # Look at which ones never converge even at 500/1.0
    stubborn = []
    for (pdb_id, s), r in zip(structures, reports_both):
        if not r.converged:
            stubborn.append({
                "pdb_id": pdb_id,
                "atoms": s.atom_count,
                "steps": r.minimizer_steps,
                "h_added": r.hydrogens_added,
                "initial_e": r.initial_energy,
                "final_e": r.final_energy,
                "delta_e": r.initial_energy - r.final_energy,
            })

    print(f"=== Stubborn structures (not converged even at 500/1.0): {len(stubborn)} ===")
    if stubborn:
        print(f"  atoms:         min={min(s['atoms'] for s in stubborn)} max={max(s['atoms'] for s in stubborn)} median={sorted(s['atoms'] for s in stubborn)[len(stubborn)//2]}")
        print(f"  H placed:      min={min(s['h_added'] for s in stubborn)} max={max(s['h_added'] for s in stubborn)}")
        print(f"  steps taken:   min={min(s['steps'] for s in stubborn)} max={max(s['steps'] for s in stubborn)}")
        deltas = [s['delta_e'] for s in stubborn]
        print(f"  energy drop:   min={min(deltas):.1f} max={max(deltas):.1f} median={sorted(deltas)[len(deltas)//2]:.1f}")
        print(f"  First 10:")
        for s in stubborn[:10]:
            print(f"    {s['pdb_id']}: {s['atoms']} atoms, {s['h_added']} H, {s['steps']} steps, "
                  f"E: {s['initial_e']:.1f}->{s['final_e']:.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (convergence rate per config):")
    print("=" * 60)
    print(f"  50 steps, tol=0.1 (benchmark):  {n_conv_50:3d}/{args.n} ({100*n_conv_50/args.n:.0f}%)")
    print(f"  500 steps, tol=0.1:             {n_conv_500:3d}/{args.n} ({100*n_conv_500/args.n:.0f}%)")
    print(f"  50 steps, tol=1.0:              {n_conv_tol1:3d}/{args.n} ({100*n_conv_tol1/args.n:.0f}%)")
    print(f"  500 steps, tol=1.0:             {n_conv_both:3d}/{args.n} ({100*n_conv_both/args.n:.0f}%)")

    with open(args.out, "w") as f:
        json.dump({
            "config": {"n": args.n},
            "rounds": {
                "50_tol0.1": n_conv_50,
                "500_tol0.1": n_conv_500,
                "50_tol1.0": n_conv_tol1,
                "500_tol1.0": n_conv_both,
            },
            "stubborn": stubborn,
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
