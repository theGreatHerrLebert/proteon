#!/usr/bin/env python3
"""Stage-3 50K battle test: load + CHARMM19+EEF1 prepare/minimize at scale.

Reconstruction of the runner that produced
``validation/stage3_50k_gpu_results.jsonl`` (issue #53). The artifact was
sha256-pinned in the v0.1.3 / v0.1.4 EVIDENT bundles but its runner was
never committed, so the `proteon-50k-battle-test-release` claim was
replayable in name only. This restores the script so a third party can
re-derive the artifact from the EVIDENT image.

What it does, per PDB:
  1. ``proteon.batch_load_tolerant`` — pdbtbx load; failures become
     ``status: load_error`` records carrying the parser exception.
  2. ``proteon.batch_prepare(ff="charmm19_eef1", minimize_steps=...)`` —
     reconstruct + place polar H + minimize. proteon's own reconstruction
     handles incomplete residues (no PDBFixer dependency); structures with
     no protein are reported ``skipped: true``.
  3. Record ``{pdb, atoms, status, skipped, initial_energy, final_energy,
     steps, converged}`` — matching the existing artifact schema exactly so
     ``validation/report/render_50k_battle_test.py`` keeps working.

GPU: proteon's CHARMM19+EEF1 energy/minimization auto-dispatches to CUDA
when a usable device is present and silently falls back to CPU otherwise,
so ``--gpu`` is advisory (kept for command-line compatibility with the
recorded ``evidence.command``); there is no separate CPU/GPU code path to
select here.

Crash isolation: when ``pebble`` is installed, each chunk runs in its own
subprocess via ``pebble.ProcessPool`` so a single segfault in the native
layer becomes a per-chunk failure rather than killing the run (the #44
pattern). Without pebble, chunks run inline (fine for smoke tests).

v0.2.0 data-mount contract: ``--pdb-dir`` falls back to
``PROTEON_CORPUS_DIR`` and the output/log directory to ``PROTEON_OUTPUT_DIR``
so the runner works unchanged against a bind-mounted corpus + output dir.

Usage (the recorded release invocation)::

    python benchmark/stage3_50k_gpu.py \\
        --pdb-dir /globalscratch/dateschn/proteon-benchmark/pdbs_50k \\
        --minimize-steps 50 --chunk 500 --gpu --threads 128 --seed 42 \\
        --out validation/stage3_50k_gpu_results.jsonl \\
        --log validation/stage3_50k_gpu.log

Smoke test (10 PDBs, CPU fallback)::

    python benchmark/stage3_50k_gpu.py --pdb-dir test-pdbs --limit 10 \\
        --out /tmp/stage3_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import proteon

try:
    import pebble
    _HAVE_PEBBLE = True
except ImportError:  # pragma: no cover - pebble is optional
    _HAVE_PEBBLE = False


def _gather_pdbs(pdb_dir: Path) -> list[Path]:
    """All .pdb / .cif files under pdb_dir, sorted for determinism."""
    files = sorted(
        p for p in pdb_dir.iterdir()
        if p.suffix.lower() in (".pdb", ".cif", ".ent")
    )
    return files


def process_chunk(paths: list[str], minimize_steps: int, threads: int | None) -> list[dict]:
    """Load + prepare a chunk of PDBs; return one record per input path.

    Runs in a worker subprocess under pebble (or inline). Imports are at
    module top so each worker pays the proteon import cost once.
    """
    recs: list[dict] = []
    loaded = proteon.batch_load_tolerant(paths, n_threads=threads)  # [(idx, struct)]
    by_idx = dict(loaded)
    structs = [s for _i, s in loaded]

    reports = []
    if structs:
        reports = proteon.batch_prepare(
            structs, ff="charmm19_eef1", minimize_steps=minimize_steps, n_threads=threads
        )
    report_by_idx = {idx: rep for (idx, _s), rep in zip(loaded, reports)}

    for i, path in enumerate(paths):
        name = Path(path).name
        if i in by_idx:
            s = by_idx[i]
            rep = report_by_idx[i]
            recs.append({
                "pdb": name,
                "atoms": int(s.atom_count),
                "status": "ok",
                "skipped": bool(getattr(rep, "skipped_no_protein", False)),
                "initial_energy": _num(getattr(rep, "initial_energy", None)),
                "final_energy": _num(getattr(rep, "final_energy", None)),
                "steps": int(getattr(rep, "minimizer_steps", 0) or 0),
                "converged": bool(getattr(rep, "converged", False)),
            })
        else:
            # Not in the tolerant-load result => it failed to load. Re-load
            # individually to capture the parser exception for the record.
            try:
                proteon.load(path)
                recs.append({"pdb": name, "status": "load_error",
                             "exception": "load failed without a raised exception"})
            except Exception as e:  # noqa: BLE001 - record any loader failure
                recs.append({"pdb": name, "status": "load_error",
                             "exception": f"{type(e).__name__}: {e}"})
    return recs


def _num(x):
    """JSON-safe float (None passes through; non-finite -> None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdb-dir", default=os.environ.get("PROTEON_CORPUS_DIR"),
                    help="Directory of PDB/mmCIF files (env: PROTEON_CORPUS_DIR).")
    ap.add_argument("--out", "--output", dest="out", default=None,
                    help="Output JSONL path (default: <PROTEON_OUTPUT_DIR>/stage3_50k_gpu_results.jsonl).")
    ap.add_argument("--log", default=None, help="Optional log path (progress also goes to stdout).")
    ap.add_argument("--minimize-steps", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=500, help="PDBs per worker chunk.")
    ap.add_argument("--gpu", action="store_true",
                    help="Advisory: proteon auto-dispatches to CUDA when available, CPU otherwise.")
    ap.add_argument("--threads", type=int, default=None, help="Intra-chunk thread budget (None=all cores).")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic shuffle of the file order.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N files (smoke tests).")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", "1")),
                    help="Pebble worker processes (chunks in flight). Ignored without pebble.")
    ap.add_argument("--task-timeout-s", type=float, default=float(os.environ.get("TASK_TIMEOUT_S", "1800")),
                    help="Per-chunk timeout when using pebble.")
    args = ap.parse_args()

    if not args.pdb_dir:
        ap.error("--pdb-dir is required (or set PROTEON_CORPUS_DIR)")
    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.is_dir():
        ap.error(f"--pdb-dir does not exist: {pdb_dir}")

    out_dir = Path(os.environ.get("PROTEON_OUTPUT_DIR") or ".")
    out_path = Path(args.out) if args.out else (out_dir / "stage3_50k_gpu_results.jsonl")
    log_path = Path(args.log) if args.log else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    files = _gather_pdbs(pdb_dir)
    random.Random(args.seed).shuffle(files)
    if args.limit:
        files = files[:args.limit]

    # Resume: skip PDBs already recorded in an existing output.
    done: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["pdb"])
                except (json.JSONDecodeError, KeyError):
                    continue
    pending = [p for p in files if p.name not in done]

    log(f"Stage-3 50K battle test")
    log(f"  corpus:   {pdb_dir}  ({len(files)} files, {len(done)} already done, {len(pending)} pending)")
    log(f"  steps:    {args.minimize_steps}   chunk: {args.chunk}   threads: {args.threads}")
    log(f"  gpu:      {'requested (auto-dispatched if available)' if args.gpu else 'auto'}")
    log(f"  pebble:   {'yes, %d workers' % args.workers if _HAVE_PEBBLE else 'no (inline)'}")
    log(f"  out:      {out_path}")

    chunks = list(_chunks([str(p) for p in pending], args.chunk))
    t0 = time.perf_counter()
    n_ok = n_skip = n_err = 0

    def tally_and_write(recs: list[dict], f) -> None:
        nonlocal n_ok, n_skip, n_err
        for rec in recs:
            if rec.get("status") == "load_error":
                n_err += 1
            elif rec.get("skipped"):
                n_skip += 1
            else:
                n_ok += 1
            f.write(json.dumps(rec) + "\n")
        f.flush()

    with out_path.open("a", encoding="utf-8") as f:
        if _HAVE_PEBBLE and args.workers > 1:
            with pebble.ProcessPool(max_workers=args.workers) as pool:
                futs = {
                    pool.schedule(process_chunk,
                                  args=[ch, args.minimize_steps, args.threads],
                                  timeout=args.task_timeout_s): ch
                    for ch in chunks
                }
                for fut, ch in futs.items():
                    try:
                        recs = fut.result()
                    except Exception as e:  # noqa: BLE001 - chunk crash/timeout
                        recs = [{"pdb": Path(p).name, "status": "load_error",
                                 "exception": f"chunk failed: {type(e).__name__}: {e}"}
                                for p in ch]
                    tally_and_write(recs, f)
                    _progress(log, n_ok, n_skip, n_err, len(pending), t0)
        else:
            for ch in chunks:
                tally_and_write(process_chunk(ch, args.minimize_steps, args.threads), f)
                _progress(log, n_ok, n_skip, n_err, len(pending), t0)

    elapsed = time.perf_counter() - t0
    n_proc = n_ok + n_skip + n_err
    log(f"\nDone. ok={n_ok} skip={n_skip} load_error={n_err} "
        f"in {elapsed/60:.1f} min ({n_proc/elapsed:.1f}/s)" if elapsed else "")
    if n_proc:
        coverage = (n_ok + n_skip) / n_proc
        log(f"  coverage (loaded / attempted): {coverage:.4f}")
    if log_fh:
        log_fh.close()
    return 0


def _progress(log, n_ok, n_skip, n_err, total, t0) -> None:
    done = n_ok + n_skip + n_err
    el = time.perf_counter() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate / 60 if rate > 0 else 0
    log(f"[{done}/{total}] ok={n_ok} skip={n_skip} load_error={n_err} "
        f"rate={rate:.1f}/s eta={eta:.1f}min")


if __name__ == "__main__":
    raise SystemExit(main())
