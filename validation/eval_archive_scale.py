#!/usr/bin/env python
"""Archive-scale battle test of the FULL label-safe path:
prepare -> coverage gate -> supervision export WITH trustworthiness masking.

The point is robustness + throughput at volume on real, diverse PDB — surface the
structures that crash the pipeline, the perf cliffs, and the yield/mask
distribution, before depending on it. Every per-structure failure is caught and
recorded (never aborts the run).

Usage: python validation/eval_archive_scale.py [N] [--floor 0.8] [--out path.json]
"""

import collections
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

import proteon
from proteon.residue_mask import structure_coverage
from proteon.supervision import build_structure_supervision_example

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdbs_10k")


def _first_protein_chain(structure):
    for ch in structure.models[0].chains:
        if any(r.is_amino_acid for r in ch.residues):
            return ch.id
    return None


def run(paths, floor, out_path):
    t0 = time.time()
    n = len(paths)
    counts = collections.Counter()
    cov_hist = collections.Counter()       # coverage bucket (tenths)
    mask_frac_hist = collections.Counter()  # trust-masked residue fraction (tenths)
    failures = []                           # (path, stage, exception)
    slowest = []                            # (seconds, path, n_res)
    total_res = 0
    total_zeroed = 0

    # Prepare in chunks (release memory); minimize=False — masking/coverage are
    # structural, and H-only relax never moves heavy atoms or changes the verdict.
    CHUNK = 200
    done = 0
    for c0 in range(0, n, CHUNK):
        chunk = paths[c0 : c0 + CHUNK]
        try:
            results = proteon.prepare_for_supervision(chunk, minimize=False, n_threads=-1)
        except Exception as e:  # whole-chunk prepare failure (should be rare)
            for p in chunk:
                failures.append((p, "prepare_chunk", repr(e)))
                counts["prepare_failed"] += 1
            done += len(chunk)
            continue

        for res in results:
            done += 1
            if not res.loaded or res.report is None:
                counts["load_failed"] += 1
                continue
            counts["prepared"] += 1
            st = time.time()
            try:
                cid = _first_protein_chain(res.structure)
                if cid is None:
                    counts["no_protein_chain"] += 1
                    continue
                cov = structure_coverage(res.structure, chain_id=cid, report=res.report)
                cov_hist[min(10, int(cov.coverage * 10))] += 1
                if cov.coverage < floor:
                    counts["below_floor"] += 1
                    continue
                counts["gated_in"] += 1
                # Full export WITH masking — the path that produces training tensors.
                ex = build_structure_supervision_example(
                    res.structure, chain_id=cid, prep_report=res.report,
                    mask_untrustworthy_coords=True,
                )
                counts["exported"] += 1
                # Trust-masking stats: residues whose atoms are PRESENT but the
                # trust mask zeroed (i.e. masked for clash/altloc, not missing).
                present = cov.node_valid.shape[0]
                atom_rows = ex.all_atom_mask.sum(axis=1)
                # Fully-zeroed atom rows = residues with NO usable coordinate
                # label (missing OR trust-masked). The export's loss mask.
                masked_rows = int((atom_rows == 0).sum())
                total_res += present
                total_zeroed += masked_rows
                frac = masked_rows / present if present else 0.0
                mask_frac_hist[min(10, int(frac * 10))] += 1
                dt = time.time() - st
                slowest.append((round(dt, 2), os.path.basename(res.path), present))
                slowest.sort(reverse=True)
                del slowest[12:]
            except Exception as e:
                counts["export_failed"] += 1
                failures.append((os.path.basename(res.path), "export", repr(e)))
                if len(failures) <= 30:
                    failures[-1] = (os.path.basename(res.path), "export",
                                    traceback.format_exc().splitlines()[-1])

        if done % 1000 < CHUNK:
            el = time.time() - t0
            print(f"  {done}/{n}  ({done / el:.1f}/s)  exported={counts['exported']}",
                  file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    out = {
        "corpus_size": n,
        "floor": floor,
        "seconds": round(elapsed, 1),
        "throughput_per_s": round(n / elapsed, 2),
        "counts": dict(counts),
        "coverage_histogram_tenths": {str(k): cov_hist[k] for k in sorted(cov_hist)},
        "trust_masked_fraction_histogram_tenths": {
            str(k): mask_frac_hist[k] for k in sorted(mask_frac_hist)
        },
        "total_exported_residues": total_res,
        "total_zeroed_label_rows": total_zeroed,
        "n_failures": len(failures),
        "failures_sample": failures[:30],
        "slowest_exports": slowest,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "failures_sample"}, indent=2))
    if failures:
        print("\n--- first failures ---", file=sys.stderr)
        for p, stage, e in failures[:30]:
            print(f"  {p} [{stage}]: {e}", file=sys.stderr)
    return out


if __name__ == "__main__":
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    floor = 0.8
    out_path = None
    for i, a in enumerate(sys.argv):
        if a == "--floor":
            floor = float(sys.argv[i + 1])
        if a == "--out":
            out_path = sys.argv[i + 1]
    paths = [p for p in sorted(glob.glob(os.path.join(CORPUS, "*.pdb"))) if os.path.exists(p)]
    if n:
        paths = paths[:n]
    print(f"archive-scale: {len(paths)} structures, floor={floor}", file=sys.stderr)
    run(paths, floor, out_path)
