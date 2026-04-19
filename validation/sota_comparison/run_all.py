#!/usr/bin/env python3
"""SOTA comparison harness driver.

Walks the (op, impl) registry over a list of PDB IDs and writes one
RunnerResult envelope per (pdb, op, impl) tuple to a JSON file. The
aggregator (`aggregate.py`) consumes that JSON.

Usage:
    # All ops, all v1 reference PDBs (from sota_reference.txt)
    python run_all.py --pdb-dir <path> --output results.json

    # Subset
    python run_all.py --pdb-dir <path> --pdbs 1crn,1ubq --ops sasa --output /tmp/smoke.json

    # Use a custom reference list
    python run_all.py --pdb-dir <path> --pdbs @my_reference.txt --output results.json

The runner determinism check (--determinism) runs each registered impl twice
on 1crn and aborts if any payload differs (with float tolerance 1e-9).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

# Make the package importable when run as a script.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]  # /scratch/TMAlign/proteon
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from validation.sota_comparison.runners import (  # noqa: E402
    BATCH_RUNNERS,
    IMPORT_FAILURES,
    OPS,
)
from validation.sota_comparison.runners._base import (  # noqa: E402
    RunnerResult,
    make_error_result,
    time_call,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_pdb_list(arg: str) -> List[str]:
    """Parse `--pdbs` argument: comma list, "all", or "@file"."""
    if arg == "all":
        return _load_reference_file(_THIS.parent / "sota_reference.txt")
    if arg.startswith("@"):
        return _load_reference_file(Path(arg[1:]))
    return [s.strip() for s in arg.split(",") if s.strip()]


def _load_reference_file(path: Path) -> List[str]:
    """Load PDB IDs from a one-per-line file (# comments OK, inline OK)."""
    out: List[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def resolve_pdb_paths(pdb_ids: Iterable[str], pdb_dir: Path) -> dict:
    """Map PDB ID -> absolute path under pdb_dir.

    Tries `<id>.pdb` then `<id>.cif`. Returns dict; missing IDs map to None.
    """
    paths = {}
    for pid in pdb_ids:
        for ext in (".pdb", ".cif"):
            cand = pdb_dir / f"{pid}{ext}"
            if cand.exists():
                paths[pid] = str(cand)
                break
        else:
            paths[pid] = None
    return paths


def filter_ops(requested: List[str]) -> List[str]:
    """Validate the requested op list against the registry."""
    if not requested or requested == ["all"]:
        return sorted(OPS.keys())
    bad = [op for op in requested if op not in OPS]
    if bad:
        log(f"ERROR: unknown op(s) {bad}. Registered ops: {sorted(OPS.keys())}")
        sys.exit(2)
    return requested


def call_runner(op: str, impl: str, fn, pdb_id: str, pdb_path: str) -> RunnerResult:
    """Invoke a runner with uniform error handling.

    The runner is responsible for setting `op`, `impl`, `impl_version`, and
    `payload`. We catch any exception and wrap it in a `status="error"`
    RunnerResult so one runner blowing up doesn't take down the harness.
    """
    t0 = time.perf_counter()
    try:
        result = fn(pdb_path)
        # Sanity check: runner returned the right shape
        if not isinstance(result, RunnerResult):
            return make_error_result(
                op, impl, "unknown", pdb_id, pdb_path,
                time.perf_counter() - t0,
                f"runner returned {type(result).__name__}, not RunnerResult",
            )
        # Stamp pdb_id/pdb_path on the result so individual runners don't have to.
        result.pdb_id = pdb_id
        result.pdb_path = pdb_path
        return result
    except Exception as e:
        return make_error_result(
            op, impl, "unknown", pdb_id, pdb_path,
            time.perf_counter() - t0,
            f"{type(e).__name__}: {e}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="SOTA comparison harness")
    parser.add_argument("--pdb-dir", required=True, type=Path,
                        help="Directory containing the PDB files")
    parser.add_argument("--pdbs", default="all",
                        help="Comma-separated PDB IDs, 'all', or '@file'")
    parser.add_argument("--ops", default="all",
                        help="Comma-separated op names or 'all'")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path")
    parser.add_argument("--warmup", action="store_true",
                        help="Warm each runner with a 1crn pass before timing")
    parser.add_argument("--force-serial", action="store_true",
                        help="Skip batched runners, always use per-structure path")
    args = parser.parse_args()

    if not args.pdb_dir.is_dir():
        log(f"ERROR: --pdb-dir {args.pdb_dir} is not a directory")
        return 2

    pdb_ids = parse_pdb_list(args.pdbs)
    op_list = filter_ops([s.strip() for s in args.ops.split(",")] if args.ops != "all" else ["all"])

    if IMPORT_FAILURES:
        log("Runner module import failures (these ops will have fewer impls):")
        for mod, err in IMPORT_FAILURES.items():
            log(f"  {mod}: {err}")
        log("")

    log(f"Driver: {len(pdb_ids)} PDBs × {len(op_list)} ops")
    log(f"  ops registered: {sorted(OPS.keys())}")
    for op in op_list:
        impls = [name for name, _ in OPS.get(op, [])]
        log(f"  {op}: {impls}")
    log("")

    paths = resolve_pdb_paths(pdb_ids, args.pdb_dir)
    missing = [pid for pid, p in paths.items() if p is None]
    if missing:
        log(f"ERROR: {len(missing)} PDB(s) missing from {args.pdb_dir}: {missing}")
        return 2

    # Optional warmup pass on 1crn (if available) to stabilize timings
    if args.warmup and "1crn" in paths:
        log("Warmup pass on 1crn...")
        for op in op_list:
            for impl, fn in OPS[op]:
                _ = call_runner(op, impl, fn, "1crn", paths["1crn"])
        log("")

    # Main loop. For each (op, impl) we prefer the batched runner when one
    # is registered — it processes all PDBs in one Rust call and unlocks
    # the in-Rust rayon parallelism. Falls back to the per-structure loop
    # when no batch runner is registered.
    records: List[dict] = []
    t_start = time.perf_counter()
    for op in op_list:
        log(f"=== op: {op} ===")
        for impl, fn in OPS[op]:
            batch_fn = BATCH_RUNNERS.get((op, impl))
            if batch_fn is not None and len(pdb_ids) > 1 and not args.force_serial:
                t0 = time.perf_counter()
                batch_paths = [paths[pid] for pid in pdb_ids]
                try:
                    batch_results = batch_fn(batch_paths)
                except Exception as e:
                    log(f"  ✗ {impl} (batched) crashed: {type(e).__name__}: {e}")
                    # Fall back to per-structure path below
                    batch_results = None
                if batch_results is not None:
                    dt = time.perf_counter() - t0
                    n_ok = sum(1 for r in batch_results if r.status == "ok")
                    n_err = sum(1 for r in batch_results if r.status == "error")
                    log(f"  batched {impl}: {dt*1000:.0f} ms total "
                        f"({n_ok} ok, {n_err} err, {dt*1000/len(pdb_ids):.1f} ms/structure avg)")
                    for pid, r in zip(pdb_ids, batch_results):
                        r.pdb_id = pid
                        records.append(r.to_dict())
                    continue
            # Per-structure fallback
            for pid in pdb_ids:
                r = call_runner(op, impl, fn, pid, paths[pid])
                rec = r.to_dict()
                records.append(rec)
                marker = {"ok": "✓", "skip": "○", "error": "✗"}.get(r.status, "?")
                log(f"  {marker} {pid}/{impl}: {r.elapsed_s*1000:.1f} ms"
                    + (f"  ({r.error})" if r.error else ""))
    total = time.perf_counter() - t_start
    log("")
    log(f"Done in {total:.1f}s. {len(records)} records written.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "meta": {
                    "pdb_dir": str(args.pdb_dir),
                    "n_pdbs": len(pdb_ids),
                    "n_ops": len(op_list),
                    "wall_s": total,
                    "import_failures": IMPORT_FAILURES,
                },
                "records": records,
            },
            f,
            indent=2,
        )
    log(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
