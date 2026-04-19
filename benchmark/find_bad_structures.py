#!/usr/bin/env python3
"""Find structures that hang or OOM in SASA computation.

Processes structures one at a time with a timeout. Records bad ones to a file.
Run on monster3:
    python benchmark/find_bad_structures.py --pdb-dir ../pdbs_50k --n 5000
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


class Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise Timeout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=10, help="Per-structure timeout (s)")
    parser.add_argument("--out", default="bad_structures.json")
    args = parser.parse_args()

    import proteon

    files = sorted(Path(args.pdb_dir).glob("*.pdb")) + sorted(Path(args.pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[: args.n]]
    files = [f for f in files if os.path.getsize(f) < 20_000_000]
    print(f"Files: {len(files)}", flush=True)

    print("Loading...", flush=True)
    loaded = proteon.batch_load_tolerant(files, n_threads=0)
    structures_with_files = []
    file_idx = 0
    for orig_idx, s in loaded:
        # orig_idx maps back to files[orig_idx]
        if s.atom_count < 25000:
            structures_with_files.append((Path(files[orig_idx]).stem, s))
    print(f"Structures (after 25K filter): {len(structures_with_files)}", flush=True)

    bad = []
    slow = []
    signal.signal(signal.SIGALRM, _alarm_handler)

    for i, (pdb_id, s) in enumerate(structures_with_files):
        if i % 100 == 0:
            print(f"  [{i}/{len(structures_with_files)}] checked, bad={len(bad)}, slow={len(slow)}", flush=True)

        # Bounding box check (pre-flight)
        try:
            coords = s.coords
            span = float(max(coords.max(axis=0) - coords.min(axis=0)))
            if span > 3000:
                print(f"  ! {pdb_id}: HUGE bbox span={span:.0f}Å (atoms={s.atom_count}, models={s.model_count})", flush=True)
                bad.append({"pdb_id": pdb_id, "reason": "huge_bbox", "bbox_span": span,
                            "atoms": s.atom_count, "models": s.model_count})
                continue
        except Exception as e:
            print(f"  ! {pdb_id}: coords access failed: {e}", flush=True)
            bad.append({"pdb_id": pdb_id, "reason": "coords_failed", "error": str(e)})
            continue

        # SASA with timeout (single thread to isolate)
        signal.alarm(args.timeout)
        try:
            t0 = time.perf_counter()
            proteon.batch_total_sasa([s], n_threads=1, radii="protor")
            dt = time.perf_counter() - t0
            signal.alarm(0)
            if dt > 5:
                print(f"  ~ {pdb_id}: SLOW SASA {dt:.1f}s (atoms={s.atom_count}, models={s.model_count}, bbox={span:.0f})", flush=True)
                slow.append({"pdb_id": pdb_id, "elapsed": dt, "atoms": s.atom_count,
                             "models": s.model_count, "bbox_span": span})
        except Timeout:
            signal.alarm(0)
            print(f"  ! {pdb_id}: TIMEOUT after {args.timeout}s (atoms={s.atom_count}, models={s.model_count}, bbox={span:.0f})", flush=True)
            bad.append({"pdb_id": pdb_id, "reason": "timeout", "atoms": s.atom_count,
                        "models": s.model_count, "bbox_span": span})
        except Exception as e:
            signal.alarm(0)
            print(f"  ! {pdb_id}: SASA failed: {e}", flush=True)
            bad.append({"pdb_id": pdb_id, "reason": "sasa_error", "error": str(e),
                        "atoms": s.atom_count, "models": s.model_count, "bbox_span": span})

    print(f"\nDone. Bad: {len(bad)}, Slow: {len(slow)}")
    with open(args.out, "w") as f:
        json.dump({"bad": bad, "slow": slow}, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
