"""TM-align fold preservation benchmark (parallel version).

For each PDB in a 1000-structure sample of the 50K corpus:
  1. Parallel-load reference + working copies via batch_load_tolerant.
  2. Extract CA coords from references (pre).
  3. Parallel-run batch_prepare on working copies (CHARMM19+EEF1).
  4. Extract CA coords from working copies (post).
  5. Compute TM-score + RMSD per pair.

Processed in chunks so progress is visible and memory stays bounded.
Results written incrementally as JSONL.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

import ferritin

PDB_DIR = Path("/globalscratch/dateschn/ferritin-benchmark/pdbs_50k")
OUT = Path("/globalscratch/dateschn/ferritin-benchmark/tm_fold_preservation.jsonl")
N = 1000
SEED = 42
CHUNK = 25
MINIMIZE_STEPS = 100  # match 50K smoke-test settings; caps straggler cost


def tm_pair(ca_ref: np.ndarray, ca_mov: np.ndarray) -> dict:
    n = len(ca_ref)
    invmap = np.arange(n, dtype=np.int32)
    tm, n_aln, rmsd_val, _R, _t = ferritin.tm_score(ca_mov, ca_ref, invmap)
    return {
        "tm_score": float(tm),
        "rmsd": float(rmsd_val),
        "n_ca": int(n),
        "n_aligned": int(n_aln),
    }


def process_chunk(paths_chunk, out_fh):
    """Process one chunk: parallel load x2 + parallel batch_prepare + per-pair TM."""
    results = []
    # Reference load
    ref_loaded = ferritin.batch_load_tolerant([str(p) for p in paths_chunk])
    ref_by_idx = dict(ref_loaded)
    ca_pre_by_idx = {i: ferritin.extract_ca_coords(s) for i, s in ref_loaded}

    # Working-copy load
    work_loaded = ferritin.batch_load_tolerant([str(p) for p in paths_chunk])
    work_by_idx = dict(work_loaded)

    # Only structures that loaded BOTH times can be compared. Intersect.
    both_idx = sorted(set(ref_by_idx) & set(work_by_idx))
    missing_idx = sorted((set(range(len(paths_chunk))) - set(ref_by_idx))
                         | (set(range(len(paths_chunk))) - set(work_by_idx)))
    for mi in missing_idx:
        rec = {"pdb": paths_chunk[mi].name, "error": "load_failed"}
        out_fh.write(json.dumps(rec) + "\n")
        results.append(rec)

    if not both_idx:
        out_fh.flush()
        return results

    # Batch minimize
    work_structs = [work_by_idx[i] for i in both_idx]
    try:
        reports = ferritin.batch_prepare(
            work_structs,
            ff="charmm19_eef1",
            minimize_steps=MINIMIZE_STEPS,
        )
    except Exception as e:
        for i in both_idx:
            rec = {"pdb": paths_chunk[i].name, "error": f"batch_prepare: {type(e).__name__}: {e}"}
            out_fh.write(json.dumps(rec) + "\n")
            results.append(rec)
        out_fh.flush()
        return results

    # Compute TM-score per structure
    for i, report in zip(both_idx, reports):
        rec = {"pdb": paths_chunk[i].name}
        try:
            if report.skipped_no_protein:
                rec["skipped"] = "no_protein"
            else:
                rec["initial_energy"] = float(report.initial_energy)
                rec["final_energy"] = float(report.final_energy)
                rec["minimizer_steps"] = int(report.minimizer_steps)
                rec["converged"] = bool(report.converged)

                ca_pre = ca_pre_by_idx[i]
                ca_post = ferritin.extract_ca_coords(work_by_idx[i])
                rec["n_ca_pre"] = int(len(ca_pre))
                rec["n_ca_post"] = int(len(ca_post))

                if ca_pre.shape != ca_post.shape:
                    rec["error"] = f"CA shape mismatch {ca_pre.shape} vs {ca_post.shape}"
                else:
                    rec.update(tm_pair(ca_pre, ca_post))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out_fh.write(json.dumps(rec) + "\n")
        results.append(rec)

    out_fh.flush()
    return results


def main():
    pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
    rng = random.Random(SEED)
    rng.shuffle(pdbs)
    sample = [PDB_DIR / name for name in pdbs[:N]]
    print(f"Sampled {len(sample)} PDBs (seed={SEED})", flush=True)
    print(f"Writing to {OUT}", flush=True)
    if hasattr(ferritin, "gpu_available") and ferritin.gpu_available():
        print(f"GPU: {ferritin.gpu_info()}", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0

    with open(OUT, "w") as f:
        for start in range(0, len(sample), CHUNK):
            chunk = sample[start:start + CHUNK]
            t_c = time.perf_counter()
            recs = process_chunk(chunk, f)
            dt = time.perf_counter() - t_c
            for r in recs:
                if "tm_score" in r:
                    n_ok += 1
                elif r.get("skipped"):
                    n_skip += 1
                else:
                    n_fail += 1
            done = start + len(chunk)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(sample) - done) / rate if rate > 0 else 0
            print(
                f"[{done}/{len(sample)}] chunk={len(chunk)} in {dt:.1f}s "
                f"({len(chunk)/dt:.2f}/s)  ok={n_ok} fail={n_fail} skip={n_skip}  "
                f"total_rate={rate:.2f}/s  eta={eta/60:.1f}min",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"\nDone. ok={n_ok} fail={n_fail} skip={n_skip} in {elapsed/60:.1f} min "
          f"({n_ok/elapsed:.2f} struct/s)", flush=True)

    # Summary
    tms, rmsds, dE, conv = [], [], [], []
    with open(OUT) as f:
        all_recs = [json.loads(l) for l in f]
    for r in all_recs:
        if "tm_score" in r:
            tms.append(r["tm_score"])
            rmsds.append(r["rmsd"])
            conv.append(bool(r.get("converged", False)))
            if "initial_energy" in r and "final_energy" in r:
                dE.append(r["final_energy"] - r["initial_energy"])

    if tms:
        tms_arr = np.array(tms)
        rmsds_arr = np.array(rmsds)
        print(f"\nTM-score (n={len(tms_arr)}):")
        print(f"  mean   = {tms_arr.mean():.4f}")
        print(f"  median = {np.median(tms_arr):.4f}")
        print(f"  min    = {tms_arr.min():.4f}")
        print(f"  p01    = {np.percentile(tms_arr, 1):.4f}")
        print(f"  p05    = {np.percentile(tms_arr, 5):.4f}")
        print(f"  p95    = {np.percentile(tms_arr, 95):.4f}")
        print(f"  max    = {tms_arr.max():.4f}")
        print(f"RMSD (A):")
        print(f"  mean   = {rmsds_arr.mean():.3f}")
        print(f"  median = {np.median(rmsds_arr):.3f}")
        print(f"  p95    = {np.percentile(rmsds_arr, 95):.3f}")
        print(f"  p99    = {np.percentile(rmsds_arr, 99):.3f}")
        print(f"  max    = {rmsds_arr.max():.3f}")
        print(f"Converged: {sum(conv)}/{len(conv)} "
              f"({100*sum(conv)/len(conv):.1f}%)")

        scored = [r for r in all_recs if "tm_score" in r]
        scored.sort(key=lambda r: r["tm_score"])
        print("\nBottom 10 by TM-score:")
        for r in scored[:10]:
            print(f"  {r['pdb']:<12} TM={r['tm_score']:.4f}  RMSD={r['rmsd']:.2f}  "
                  f"n_ca={r['n_ca']}  steps={r.get('minimizer_steps','?')}  "
                  f"conv={r.get('converged','?')}")


if __name__ == "__main__":
    main()
