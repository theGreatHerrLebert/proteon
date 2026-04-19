#!/usr/bin/env python3
"""Find which structures have pathologically slow prepare() minimization.

Runs prepare on each structure individually with a timeout, records time.
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path


class Timeout(Exception):
    pass


def _alarm(sig, frame):
    raise Timeout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=10,
                        help="Per-structure timeout (s)")
    parser.add_argument("--out", default="slow_prepare.json")
    args = parser.parse_args()

    import proteon

    files = sorted(Path(args.pdb_dir).glob("*.pdb")) + sorted(Path(args.pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[: args.n * 3]]
    files = [f for f in files if os.path.getsize(f) < 20_000_000]

    print(f"Loading up to {len(files)} files...", flush=True)
    loaded = proteon.batch_load_tolerant(files, n_threads=0)
    items = [(i, s, files[i]) for i, s in loaded if s.atom_count < 25000]
    items = items[: args.n]
    print(f"Running prepare on {len(items)} structures with {args.timeout}s timeout\n", flush=True)

    signal.signal(signal.SIGALRM, _alarm)
    results = []
    slow = []
    timed_out = []

    for idx, (orig_idx, s, path) in enumerate(items):
        pdb_id = Path(path).stem
        if idx % 20 == 0:
            print(f"  [{idx}/{len(items)}] slow={len(slow)} timeout={len(timed_out)}", flush=True)

        # Reload fresh copy (prepare modifies in-place)
        s_copy = proteon.load(path)

        signal.alarm(args.timeout)
        try:
            t0 = time.perf_counter()
            reports = proteon.batch_prepare(
                [s_copy], reconstruct=False, hydrogens="backbone",
                minimize=True, minimize_steps=50, minimize_method="lbfgs",
                gradient_tolerance=0.1, n_threads=1,
            )
            dt = time.perf_counter() - t0
            signal.alarm(0)
            r = reports[0]
            rec = {
                "pdb_id": pdb_id,
                "atoms": s.atom_count,
                "models": s.model_count,
                "h_added": r.hydrogens_added,
                "steps": r.minimizer_steps,
                "converged": r.converged,
                "initial_e": r.initial_energy,
                "final_e": r.final_energy,
                "time_s": dt,
            }
            results.append(rec)
            if dt > 2.0:
                slow.append(rec)
                print(f"  SLOW {pdb_id}: {s.atom_count} atoms, {s.model_count} models, "
                      f"{r.minimizer_steps} steps, {dt:.1f}s, converged={r.converged}", flush=True)
        except Timeout:
            signal.alarm(0)
            rec = {
                "pdb_id": pdb_id,
                "atoms": s.atom_count,
                "models": s.model_count,
                "timeout": True,
            }
            timed_out.append(rec)
            results.append(rec)
            print(f"  ! TIMEOUT {pdb_id}: {s.atom_count} atoms, {s.model_count} models", flush=True)
        except Exception as e:
            signal.alarm(0)
            print(f"  ! ERROR {pdb_id}: {e}", flush=True)

    # Summary
    n_total = len(results)
    n_ok = sum(1 for r in results if r.get("converged", False))
    n_slow = len(slow)
    n_timeout = len(timed_out)
    times = sorted(r["time_s"] for r in results if "time_s" in r)
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total: {n_total}")
    print(f"  Converged: {n_ok}")
    print(f"  Slow (>2s): {n_slow}")
    print(f"  Timeout (>{args.timeout}s): {n_timeout}")
    if times:
        print(f"  Time: min={times[0]:.3f}s, median={times[len(times)//2]:.3f}s, "
              f"p95={times[int(len(times)*0.95)]:.3f}s, max={times[-1]:.3f}s")

    with open(args.out, "w") as f:
        json.dump({"results": results, "slow": slow, "timeout": timed_out}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
