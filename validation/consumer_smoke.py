"""AF2-style consumer smoke over a proteon corpus release.

What we're pinning:
  - training.parquet (scalars + structure supervision tensors) and
    sequence.parquet (sequence + MSA + templates) must join cleanly on
    record_id.
  - Per-row tensor shapes are consistent across both sides: `length`
    matches between sequence and structure examples; atom37 positions
    are (L, 37, 3); MSA is (depth, L) when present; rigidgroups are
    (L, 8, 4, 4), etc.
  - `iter_training_examples(batch_size=N)` yields contiguous chunks of
    length ≤ N (the contract fixed in 1dc5a3f).
  - Round-trip scalars — sequence string, code_rev, prep_run_id,
    quality — are NOT empty/None on reload (the contract fixed in
    1dc5a3f; a silent regression here would mean training code gets
    degraded objects).

Usage:
  python validation/consumer_smoke.py <release_dir> [--batch-size 4] [--n-batches 8]

Exits non-zero on any contract violation. Prints a compact per-batch
summary + totals so it doubles as a release-shape inspector.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_sequence_index(release_dir: Path) -> Dict[str, object]:
    """Load all sequence examples into a record_id -> SequenceExample map.

    Uses the streaming iterator so MSA-heavy releases don't peak before
    the consumer loop starts. Small corpora (≤ a few thousand chains)
    comfortably fit; big corpora should instead query a Parquet
    predicate-pushdown reader by record_id as needed.
    """
    from proteon.sequence_export import iter_sequence_examples

    seq_dir = release_dir / "sequence" / "examples"
    if not seq_dir.exists():
        raise SystemExit(f"no sequence release under {release_dir}")
    return {ex.record_id: ex for ex in iter_sequence_examples(seq_dir)}


def assert_contract(record_id: str, ex, seq_ex) -> List[str]:
    """Return a list of contract violations for one training example.

    Empty list ⇒ this record passes. Caller collects all violations
    before deciding to fail, so one bad record doesn't mask others.
    """
    violations: List[str] = []

    # ----- join integrity -----
    if seq_ex is None:
        violations.append(f"{record_id}: no sequence example (join miss)")
        return violations
    if seq_ex.record_id != record_id:
        violations.append(
            f"{record_id}: sequence side returned record_id={seq_ex.record_id!r}"
        )

    # ----- length consistency -----
    L = ex.sequence.length
    if ex.structure.length != L:
        violations.append(
            f"{record_id}: seq.length={L} != struc.length={ex.structure.length}"
        )
    if seq_ex.length != L:
        violations.append(
            f"{record_id}: training seq.length={L} != standalone seq.length={seq_ex.length}"
        )

    # ----- round-trip scalars (regression for #1) -----
    if not ex.sequence.sequence:
        violations.append(
            f"{record_id}: sequence string is empty on reload — #1 regression"
        )
    if ex.sequence.sequence != seq_ex.sequence:
        violations.append(
            f"{record_id}: training.sequence != sequence.sequence "
            f"({ex.sequence.sequence[:10]!r}... vs {seq_ex.sequence[:10]!r}...)"
        )
    if len(ex.sequence.sequence) != L:
        violations.append(
            f"{record_id}: len(sequence)={len(ex.sequence.sequence)} != L={L}"
        )

    # ----- structure tensors -----
    checks = [
        ("all_atom_positions", (L, 37, 3), np.float32),
        ("all_atom_mask", (L, 37), np.float32),
        ("atom14_gt_positions", (L, 14, 3), np.float32),
        ("rigidgroups_gt_frames", (L, 8, 4, 4), np.float32),
        ("chi_angles", (L, 4), np.float32),
        ("pseudo_beta", (L, 3), np.float32),
    ]
    for attr, expected_shape, expected_dtype in checks:
        arr = getattr(ex.structure, attr)
        if arr.shape != expected_shape:
            violations.append(
                f"{record_id}: structure.{attr} shape {arr.shape} != {expected_shape}"
            )
        if arr.dtype != expected_dtype:
            violations.append(
                f"{record_id}: structure.{attr} dtype {arr.dtype} != {expected_dtype}"
            )

    # ----- sequence-side MSA (when present) -----
    if seq_ex.msa is not None:
        depth, L_msa = seq_ex.msa.shape
        if L_msa != L:
            violations.append(
                f"{record_id}: msa shape {seq_ex.msa.shape} has column count != L={L}"
            )
        if seq_ex.deletion_matrix is not None and seq_ex.deletion_matrix.shape != seq_ex.msa.shape:
            violations.append(
                f"{record_id}: deletion_matrix shape {seq_ex.deletion_matrix.shape} "
                f"!= msa shape {seq_ex.msa.shape}"
            )
        if seq_ex.msa_mask is not None and seq_ex.msa_mask.shape != seq_ex.msa.shape:
            violations.append(
                f"{record_id}: msa_mask shape {seq_ex.msa_mask.shape} "
                f"!= msa shape {seq_ex.msa.shape}"
            )

    # ----- aatype parity (sequence side and structure side share the axis) -----
    if not np.array_equal(ex.sequence.aatype, ex.structure.aatype):
        violations.append(
            f"{record_id}: aatype differs between sequence and structure sides"
        )

    return violations


def collate_batch_structure_tensors(batch) -> Dict[str, np.ndarray]:
    """Stand-in for an AF2 DataLoader collate: pad a batch to max-L.

    Returns a dict of (B, L_max, ...) tensors. Not optimized — this is a
    correctness smoke, not a perf benchmark. Proves a real consumer can
    combine examples of varying length without crashing.
    """
    L_max = max(ex.sequence.length for ex in batch)
    B = len(batch)

    out: Dict[str, np.ndarray] = {
        "aatype": np.zeros((B, L_max), dtype=np.int32),
        "seq_mask": np.zeros((B, L_max), dtype=np.float32),
        "all_atom_positions": np.zeros((B, L_max, 37, 3), dtype=np.float32),
        "all_atom_mask": np.zeros((B, L_max, 37), dtype=np.float32),
        "rigidgroups_gt_frames": np.zeros((B, L_max, 8, 4, 4), dtype=np.float32),
    }
    for bi, ex in enumerate(batch):
        L = ex.sequence.length
        out["aatype"][bi, :L] = ex.sequence.aatype
        out["seq_mask"][bi, :L] = ex.sequence.seq_mask
        out["all_atom_positions"][bi, :L] = ex.structure.all_atom_positions
        out["all_atom_mask"][bi, :L] = ex.structure.all_atom_mask
        out["rigidgroups_gt_frames"][bi, :L] = ex.structure.rigidgroups_gt_frames
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release_dir", type=Path, help="corpus release root (has training/, sequence/, …)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=8, help="stop after N batches (0 = all)")
    ap.add_argument("--split", type=str, default=None, help="filter to one split (predicate pushdown)")
    args = ap.parse_args()

    from proteon.training_example import iter_training_examples

    release = args.release_dir.resolve()
    if not (release / "training" / "release_manifest.json").exists():
        raise SystemExit(f"no training release under {release}")

    print(f"release: {release}", flush=True)
    print(f"RSS_start_MB: {rss_mb():.0f}", flush=True)

    t0 = time.time()
    seq_index = load_sequence_index(release)
    print(f"sequence_index: n={len(seq_index)}  wall={time.time()-t0:.1f}s  RSS={rss_mb():.0f}MB", flush=True)

    all_violations: List[str] = []
    n_examples = 0
    n_msa_populated = 0
    lengths: List[int] = []
    split_counts: Dict[str, int] = {}

    t0 = time.time()
    for batch_i, batch in enumerate(
        iter_training_examples(release / "training", batch_size=args.batch_size, split=args.split)
    ):
        # Every record must have a sequence-side peer.
        for ex in batch:
            seq_ex = seq_index.get(ex.record_id)
            all_violations.extend(assert_contract(ex.record_id, ex, seq_ex))
            n_examples += 1
            lengths.append(ex.sequence.length)
            split_counts[ex.split] = split_counts.get(ex.split, 0) + 1
            if seq_ex is not None and seq_ex.msa is not None:
                n_msa_populated += 1

        # Collate smoke — prove a consumer can build a padded batch.
        padded = collate_batch_structure_tensors(batch)
        for name, arr in padded.items():
            if np.isnan(arr).any() or np.isinf(arr).any():
                all_violations.append(
                    f"batch {batch_i}: padded.{name} contains NaN/Inf"
                )

        if batch_i < 3 or batch_i % 10 == 0:
            ids = [ex.record_id for ex in batch]
            Ls = [ex.sequence.length for ex in batch]
            print(
                f"batch {batch_i}: n={len(batch)} Ls={Ls} ids={ids[:3]}"
                f"{'...' if len(ids) > 3 else ''}  L_max={max(Ls)}  RSS={rss_mb():.0f}MB",
                flush=True,
            )
        if args.n_batches and batch_i + 1 >= args.n_batches:
            break
    wall = time.time() - t0

    print("---", flush=True)
    print(f"n_examples_checked: {n_examples}", flush=True)
    print(f"n_with_msa: {n_msa_populated} of {n_examples}", flush=True)
    print(f"split_counts: {split_counts}", flush=True)
    if lengths:
        print(
            f"length_summary: min={min(lengths)} max={max(lengths)} "
            f"mean={sum(lengths)/len(lengths):.1f}",
            flush=True,
        )
    print(f"batch_wall: {wall:.1f}s   RSS_end: {rss_mb():.0f}MB", flush=True)
    print(f"violations: {len(all_violations)}", flush=True)

    if all_violations:
        for v in all_violations[:20]:
            print(f"  ! {v}", flush=True)
        if len(all_violations) > 20:
            print(f"  ... and {len(all_violations) - 20} more", flush=True)
        sys.exit(1)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
