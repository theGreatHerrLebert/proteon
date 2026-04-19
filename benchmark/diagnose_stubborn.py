#!/usr/bin/env python3
"""Diagnose the stubborn non-converged structures from the 50K victory-lap run.

Reproduces the victory-lap config exactly (first 200 structures of chunk 1,
200 LBFGS steps, default tol=0.1). Then takes whichever didn't converge and
re-runs them with 1000 steps to see if they're "needs more time" or "stuck".

Per-structure data is dumped to JSON for follow-up analysis.

Run on monster3:
    python benchmark/diagnose_stubborn.py \\
        --pdb-dir /globalscratch/dateschn/proteon-benchmark/pdbs_50k \\
        --out /globalscratch/dateschn/proteon-benchmark/stubborn_diagnosis.json
"""

import argparse
import json
import os
import time
from pathlib import Path


def collect_sample(pdb_dir, n_target=200, chunk_size=5000):
    """Reproduce run_benchmark.py's chunk-1 sample exactly."""
    files = sorted(Path(pdb_dir).glob("*.pdb")) + sorted(Path(pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[:chunk_size]]
    # Size filter (< 20 MB) matches run_benchmark.py line 79
    files = [f for f in files if os.path.getsize(f) < 20_000_000]
    return files


def report_to_dict(pdb_id, atoms, r):
    return {
        "pdb_id": pdb_id,
        "atoms": atoms,
        "h_added": r.hydrogens_added,
        "h_skipped": r.hydrogens_skipped,
        "steps": r.minimizer_steps,
        "converged": bool(r.converged),
        "initial_e": r.initial_energy,
        "final_e": r.final_energy,
        "delta_e": r.initial_energy - r.final_energy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="stubborn_diagnosis.json")
    parser.add_argument("--max-steps-extended", type=int, default=1000,
                        help="Step budget for second pass on non-converged")
    args = parser.parse_args()

    import proteon

    print(f"Collecting sample from {args.pdb_dir}...", flush=True)
    files = collect_sample(args.pdb_dir)
    print(f"  {len(files)} candidate files after size filter", flush=True)

    # Load and apply the atom-count filter (< 25K atoms), keep pdb_id -> struct
    # n_threads=-1 means "all cores"; 0 would incorrectly mean single-threaded.
    loaded = proteon.batch_load_tolerant(files, n_threads=-1)
    pairs = []
    for i, s in loaded:
        if s.atom_count < 25000:
            pairs.append((Path(files[i]).stem, s))
    pairs = pairs[: args.n]
    print(f"  {len(pairs)} structures after atom-count filter\n", flush=True)

    # ---- Round 1: victory-lap config (200 steps, default tol=0.1) ----
    print(f"=== Round 1: victory-lap config (200 steps, tol=0.1) ===", flush=True)
    structs = [s for _, s in pairs]
    t0 = time.perf_counter()
    reports = proteon.batch_prepare(
        structs, reconstruct=False, hydrogens="backbone",
        minimize=True, minimize_steps=200, minimize_method="lbfgs",
        gradient_tolerance=0.1, n_threads=-1,
    )
    dt = time.perf_counter() - t0
    n_conv = sum(1 for r in reports if r.converged)
    print(f"  {n_conv}/{len(reports)} converged in {dt:.1f}s", flush=True)

    round1 = [report_to_dict(pid, s.atom_count, r)
              for (pid, s), r in zip(pairs, reports)]
    n_nonconv = len(reports) - n_conv

    # Step distribution for non-converged
    step_buckets = {}
    for rec in round1:
        if not rec["converged"]:
            b = (rec["steps"] // 25) * 25
            step_buckets[b] = step_buckets.get(b, 0) + 1
    print(f"  Non-converged step distribution: {sorted(step_buckets.items())}", flush=True)

    # ---- Round 2: non-converged only, with extended step budget ----
    nonconv_pdb_ids = {rec["pdb_id"] for rec in round1 if not rec["converged"]}
    print(f"\n=== Round 2: {len(nonconv_pdb_ids)} non-converged at {args.max_steps_extended} steps ===", flush=True)

    # Freshly reload (prepare modified them in place)
    fresh_pairs = []
    for pid, _ in pairs:
        if pid in nonconv_pdb_ids:
            path = next(f for f in files if Path(f).stem == pid)
            s = proteon.load(path)
            fresh_pairs.append((pid, s))

    if fresh_pairs:
        fresh_structs = [s for _, s in fresh_pairs]
        t0 = time.perf_counter()
        reports2 = proteon.batch_prepare(
            fresh_structs, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_steps=args.max_steps_extended, minimize_method="lbfgs",
            gradient_tolerance=0.1, n_threads=-1,
        )
        dt = time.perf_counter() - t0
        n_conv2 = sum(1 for r in reports2 if r.converged)
        gained = n_conv2  # they were all non-converged in round 1
        print(f"  {n_conv2}/{len(reports2)} converged with {args.max_steps_extended} steps in {dt:.1f}s", flush=True)
        print(f"  Gained {gained} structures by raising the step cap", flush=True)
        round2 = [report_to_dict(pid, s.atom_count, r)
                  for (pid, s), r in zip(fresh_pairs, reports2)]
    else:
        round2 = []
        n_conv2 = 0

    # ---- Summarize the truly-stubborn set ----
    stubborn = [rec for rec in round2 if not rec["converged"]]
    stubborn.sort(key=lambda r: r["atoms"])
    print(f"\n=== Truly stubborn (not converged even at {args.max_steps_extended} steps): {len(stubborn)} ===")
    if stubborn:
        atoms = sorted(r["atoms"] for r in stubborn)
        steps_l = sorted(r["steps"] for r in stubborn)
        deltas = sorted(r["delta_e"] for r in stubborn)
        p = lambda xs, q: xs[min(len(xs) - 1, int(len(xs) * q))]
        print(f"  atoms:         min={atoms[0]} p50={p(atoms,0.5)} p95={p(atoms,0.95)} max={atoms[-1]}")
        print(f"  steps taken:   min={steps_l[0]} p50={p(steps_l,0.5)} max={steps_l[-1]}")
        print(f"  energy drop:   min={deltas[0]:.1f} p50={p(deltas,0.5):.1f} max={deltas[-1]:.1f}")
        print(f"  First 15 (by atom count):")
        for rec in stubborn[:15]:
            print(f"    {rec['pdb_id']}: {rec['atoms']} atoms, {rec['h_added']} H, "
                  f"{rec['steps']} steps, E: {rec['initial_e']:.1f} -> {rec['final_e']:.1f} "
                  f"(ΔE={rec['delta_e']:.1f})")

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Round 1 (200 steps):  {n_conv:3d}/{len(reports)} ({100*n_conv/len(reports):.0f}%)")
    print(f"  Round 2 (1000 steps): {n_conv + n_conv2:3d}/{len(reports)} "
          f"({100*(n_conv+n_conv2)/len(reports):.0f}%)   "
          f"[+{n_conv2} gained]")
    print(f"  Truly stubborn:       {len(stubborn):3d}/{len(reports)} "
          f"({100*len(stubborn)/len(reports):.0f}%)")

    out = {
        "config": {
            "n": args.n,
            "pdb_dir": args.pdb_dir,
            "round1_steps": 200,
            "round2_steps": args.max_steps_extended,
            "gradient_tolerance": 0.1,
        },
        "counts": {
            "total": len(reports),
            "converged_r1": n_conv,
            "converged_r2_extra": n_conv2,
            "stubborn": len(stubborn),
        },
        "round1": round1,
        "round2": round2,
        "stubborn": stubborn,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
