#!/usr/bin/env python3
"""Stage-1 smoke test: post-fix validation on a random 100-PDB sample.

Runs batch_prepare under CHARMM19+EEF1 (the production default) on a
deterministic random sample of the 50K corpus and records per-structure
metrics. Purpose is to catch obvious regressions introduced by the
2026-04-12 CHARMM19 fix series before running the full 50K battle test.

Success criteria (must all hold):
  * Zero panics / exceptions. Any crash = stop and investigate.
  * % structures with skipped_no_protein <= 5%. Otherwise the
    "is this a protein" heuristic still has a hole we missed.
  * % structures with negative final_energy >= 90% (among non-skipped).
    This is the key correctness invariant post-heavy-atom-relax fix.
  * No structure takes > 30s (sanity — if heavy-atom minimization
    scales badly, Stage 1 is where we notice).

Non-blocking diagnostics collected for stage 2/3 planning:
  * Wall clock per structure vs atom count (scaling curve)
  * Convergence rate (`converged=True` fraction at 100 steps)
  * Sign distribution of vdw, electrostatic, solvation
  * Top-5 outliers by |final_energy| and by wall time
  * n_unassigned distribution (ligand density proxy)

Usage:
    python stage1_smoke.py \\
        --pdb-dir /globalscratch/dateschn/proteon-benchmark/pdbs_50k \\
        --output stage1_smoke_results.json \\
        --n 100 --seed 42

The script is intentionally single-threaded and runs structures
serially. Stage 1 is a SMOKE test — we're looking for "does anything
crash?", not "how fast can we go?". Stage 2 will add parallelism
budget exploration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def run_one(pdb_path: str, minimize_steps: int) -> dict:
    """Run batch_prepare on one PDB, return a result dict.

    Never raises — any exception is caught and recorded as
    {"status": "error", "exception": ..., "traceback": ...}. That way
    one bad structure doesn't abort the whole smoke test.
    """
    import proteon

    result = {"pdb": os.path.basename(pdb_path)}
    try:
        t_load = time.perf_counter()
        s = proteon.load(pdb_path)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        result["atom_count_pre_h"] = s.atom_count
        result["residue_count"] = s.residue_count
        result["load_ms"] = load_ms

        t_prep = time.perf_counter()
        reports = proteon.batch_prepare(
            [s],
            reconstruct=False,
            hydrogens="all",
            minimize=True,
            minimize_method="lbfgs",
            minimize_steps=minimize_steps,
            gradient_tolerance=0.1,
            strip_hydrogens=False,
            ff="charmm19_eef1",
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000.0
        r = reports[0]

        result.update(
            {
                "status": "ok",
                "prep_ms": prep_ms,
                "skipped_no_protein": r.skipped_no_protein,
                "initial_energy": r.initial_energy,
                "final_energy": r.final_energy,
                "hydrogens_added": r.hydrogens_added,
                "n_unassigned_atoms": r.n_unassigned_atoms,
                "minimizer_steps": r.minimizer_steps,
                "converged": r.converged,
                "components": dict(r.components),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    return result


def summarize(records: list[dict]) -> dict:
    """Derive summary stats from the per-structure records."""
    n = len(records)
    errors = [r for r in records if r.get("status") == "error"]
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in ok if r.get("skipped_no_protein")]
    processed = [r for r in ok if not r.get("skipped_no_protein")]

    negative_total = [r for r in processed if r["final_energy"] < 0]
    converged = [r for r in processed if r.get("converged")]

    def _sign_count(key):
        pos = sum(1 for r in processed if r["components"][key] > 0)
        neg = sum(1 for r in processed if r["components"][key] < 0)
        zero = len(processed) - pos - neg
        return {"pos": pos, "neg": neg, "zero": zero}

    def _percentiles(values, ps=(50, 90, 99, 100)):
        if not values:
            return {f"p{p}": None for p in ps}
        vs = sorted(values)
        out = {}
        for p in ps:
            if p == 100:
                out[f"p{p}"] = vs[-1]
            else:
                idx = min(len(vs) - 1, int(len(vs) * p / 100))
                out[f"p{p}"] = vs[idx]
        return out

    prep_times = [r["prep_ms"] for r in processed]
    atom_counts = [r.get("atom_count_pre_h", 0) for r in processed]

    # Outliers: top-5 most-positive final_energy (worst) and top-5 slowest
    worst_final = sorted(
        processed, key=lambda r: -r["final_energy"]
    )[:5]
    slowest = sorted(processed, key=lambda r: -r["prep_ms"])[:5]

    summary = {
        "n_total": n,
        "n_errors": len(errors),
        "n_skipped_no_protein": len(skipped),
        "n_processed": len(processed),
        "n_negative_total": len(negative_total),
        "n_converged": len(converged),
        "pct_errors": 100.0 * len(errors) / max(1, n),
        "pct_skipped_no_protein": 100.0 * len(skipped) / max(1, n),
        "pct_negative_total": 100.0 * len(negative_total) / max(1, len(processed)),
        "pct_converged": 100.0 * len(converged) / max(1, len(processed)),
        "prep_ms_percentiles": _percentiles(prep_times),
        "atom_count_percentiles": _percentiles(atom_counts),
        "sign_distribution": {
            "vdw": _sign_count("vdw"),
            "electrostatic": _sign_count("electrostatic"),
            "solvation": _sign_count("solvation"),
        },
        "total_wall_seconds": sum(prep_times) / 1000.0,
        "worst_final_energies": [
            {"pdb": r["pdb"], "final": r["final_energy"], "atoms": r.get("atom_count_pre_h", 0)}
            for r in worst_final
        ],
        "slowest_structures": [
            {"pdb": r["pdb"], "prep_ms": r["prep_ms"], "atoms": r.get("atom_count_pre_h", 0)}
            for r in slowest
        ],
        "error_examples": [
            {"pdb": r["pdb"], "exception": r["exception"]}
            for r in errors[:5]
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=100,
                        help="Number of structures to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sampling")
    parser.add_argument("--minimize-steps", type=int, default=100,
                        help="LBFGS max steps per structure")
    args = parser.parse_args()

    if not args.pdb_dir.is_dir():
        log(f"ERROR: --pdb-dir {args.pdb_dir} not a directory")
        return 2

    # Deterministic random sample
    all_files = sorted(args.pdb_dir.glob("*.pdb"))
    if len(all_files) == 0:
        log(f"ERROR: no .pdb files in {args.pdb_dir}")
        return 2
    rng = random.Random(args.seed)
    sample = rng.sample(all_files, min(args.n, len(all_files)))
    log(f"Stage 1 smoke test: {len(sample)} / {len(all_files)} PDBs "
        f"(seed={args.seed}, minimize_steps={args.minimize_steps})")
    log(f"Output: {args.output}")
    log("-" * 60)

    # Incremental JSONL output — one line per structure, written
    # immediately. This is critical: the 2026-04-12 heavy-atom relax fix
    # made per-structure minimization O(minutes) instead of O(seconds),
    # so a 100-PDB run can take hours and any crash/kill mid-run loses
    # data if we only write JSON at the end.
    jsonl_path = args.output.with_suffix(".jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    records = []
    with open(jsonl_path, "w") as jsonl_fp:
        for i, path in enumerate(sample, 1):
            rec = run_one(str(path), args.minimize_steps)
            records.append(rec)
            jsonl_fp.write(json.dumps(rec, default=str) + "\n")
            jsonl_fp.flush()
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (len(sample) - i)
            atoms = rec.get("atom_count_pre_h", "?")
            prep_ms = rec.get("prep_ms", "?")
            prep_str = f"{prep_ms:.0f}" if isinstance(prep_ms, (int, float)) else str(prep_ms)
            final = rec.get("final_energy", None)
            final_str = f"{final:+.0f}" if isinstance(final, (int, float)) else "-"
            log(
                f"  [{i:3d}/{len(sample)}] elapsed={elapsed:6.1f}s  "
                f"eta={eta:6.1f}s  {rec['pdb']:<12} atoms={atoms} "
                f"prep_ms={prep_str:>6} final={final_str:>10} "
                f"status={rec.get('status')}"
            )
    t_wall = time.perf_counter() - t_start
    log(f"JSONL checkpoint: {jsonl_path}")

    summary = summarize(records)
    summary["wall_seconds"] = t_wall
    summary["seed"] = args.seed
    summary["minimize_steps"] = args.minimize_steps
    summary["pdb_dir"] = str(args.pdb_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": records}
    args.output.write_text(json.dumps(payload, indent=2, default=str))

    log("-" * 60)
    log(f"Wall time:          {t_wall:.1f}s")
    log(f"Errors:             {summary['n_errors']}/{summary['n_total']}")
    log(f"Skipped no_protein: {summary['n_skipped_no_protein']}/{summary['n_total']} "
        f"({summary['pct_skipped_no_protein']:.1f}%)")
    log(f"Processed:          {summary['n_processed']}")
    log(f"Negative total:     {summary['n_negative_total']}/{summary['n_processed']} "
        f"({summary['pct_negative_total']:.1f}%)")
    log(f"Converged:          {summary['n_converged']}/{summary['n_processed']} "
        f"({summary['pct_converged']:.1f}%)")
    log(f"Prep ms (median):   {summary['prep_ms_percentiles']['p50']:.0f}")
    log(f"Prep ms (p99):      {summary['prep_ms_percentiles']['p99']:.0f}")
    log(f"Prep ms (max):      {summary['prep_ms_percentiles']['p100']:.0f}")
    log(f"Sign vdw:           {summary['sign_distribution']['vdw']}")
    log(f"Sign electrostatic: {summary['sign_distribution']['electrostatic']}")
    log(f"Sign solvation:     {summary['sign_distribution']['solvation']}")

    # Decision gate: print PASS/FAIL for stage 1 success criteria
    log("-" * 60)
    criteria = [
        ("zero crashes", summary["n_errors"] == 0),
        ("skipped_no_protein <= 5%", summary["pct_skipped_no_protein"] <= 5.0),
        ("negative_total >= 90% (of processed)",
         summary["pct_negative_total"] >= 90.0),
        ("max prep time < 30000 ms",
         (summary["prep_ms_percentiles"]["p100"] or 0) < 30000),
    ]
    for name, passed in criteria:
        mark = "PASS" if passed else "FAIL"
        log(f"  [{mark}] {name}")
    all_pass = all(p for _, p in criteria)
    log(f"Stage 1: {'GREEN' if all_pass else 'RED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
