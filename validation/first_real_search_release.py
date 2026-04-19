"""First real end-to-end corpus release with GPU MSA + templates.

Drives proteon's full Layer-5 stack over the four single-chain PDB
fixtures (1crn / 1ubq / 1bpi / 1aaj), building:

  1. Prepared-structure manifest   (Layer 3)
  2. Structure supervision release (Layer 5 — atom37/atom14/frames/…)
  3. Sequence release WITH MSA     (Layer 5 — GPU warp-collab SW inner)
  4. Template features per query   (Layer 5 — sequence-based top-K)
  5. Training release WITH NPZ     (Layer 5 — denormalized)
  6. Corpus release manifest       (Layer 5 — top-level)
  7. Validation report             (Layer 5 — cross-layer sanity)

Every on-disk artifact is SHA-256 checksummed. Every failure
(including the known-bad multi-chain 1ake drop) lands in a
canonical-taxonomy failure record. This is the "first real search"
milestone: the entire pipeline running against actual crystal
structures instead of hand-built SimpleNamespaces.

Run:
    cd /scratch/TMAlign/proteon
    .venv/bin/python validation/first_real_search_release.py \\
        --out /tmp/proteon_first_real_search
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import proteon
from proteon.supervision_constants import residue_to_one_letter


DEFAULT_INPUTS = [
    "test-pdbs/1crn.pdb",
    "test-pdbs/1ubq.pdb",
    "test-pdbs/1bpi.pdb",
    "test-pdbs/1aaj.pdb",
]


def run(inputs, out_dir: Path, release_id: str) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    timings: dict = {}

    # 1. Raw intake.
    t0 = time.time()
    pairs = proteon.batch_load_tolerant([str(p) for p in inputs])
    loaded_structures = [pair[1] for pair in pairs]
    record_ids = [Path(inputs[pair[0]]).stem + ":A" for pair in pairs]
    source_ids = [str(inputs[pair[0]]) for pair in pairs]
    timings["intake_s"] = time.time() - t0
    print(f"  loaded {len(loaded_structures)}/{len(inputs)} structures "
          f"in {timings['intake_s']:.2f}s")

    # 2. Prep.
    t0 = time.time()
    prep_reports = proteon.batch_prepare(loaded_structures)
    timings["prep_s"] = time.time() - t0
    print(f"  prepared {len(prep_reports)} structures in {timings['prep_s']:.2f}s")

    # 3. Structure supervision release (direct API, skipping
    # build_local_corpus_smoke_release so we can layer MSA + templates
    # via the lower-level builders).
    t0 = time.time()
    struc_root = proteon.build_structure_supervision_dataset_from_prepared(
        loaded_structures,
        prep_reports,
        out_dir / "prepared",
        release_id=f"{release_id}-structure",
        record_ids=record_ids,
        source_ids=source_ids,
        code_rev="local",
        config_rev="local",
        provenance={"input_paths": [str(p) for p in inputs]},
        overwrite=True,
    )
    timings["supervision_s"] = time.time() - t0

    # 4. Build a search engine over every chain's sequence. Keyed by
    # its index so templates can cross-reference target supervisions.
    t0 = time.time()
    supervisions = proteon.batch_build_structure_supervision_examples(
        loaded_structures, record_ids=record_ids, source_ids=source_ids
    )
    chain_seqs = [
        (i, sup.sequence) for i, sup in enumerate(supervisions)
    ]
    engine = proteon.MsaSearch.build(
        chain_seqs, k=3, reduce_to=None, min_score=0, max_results=10
    )
    timings["engine_build_s"] = time.time() - t0
    print(f"  search engine built over {engine.target_count} sequences "
          f"in {timings['engine_build_s']:.2f}s")

    # 5. Sequence release WITH GPU-backed MSA assembly.
    t0 = time.time()
    sequence_root = proteon.build_sequence_dataset(
        loaded_structures,
        out_dir / "sequence",
        release_id=f"{release_id}-sequence",
        record_ids=record_ids,
        source_ids=source_ids,
        msa_engine=engine,
        code_rev="local",
        config_rev="local",
        provenance={"input_paths": [str(p) for p in inputs]},
        overwrite=True,
    )
    timings["sequence_with_msa_s"] = time.time() - t0
    print(f"  sequence release w/ MSA in {timings['sequence_with_msa_s']:.2f}s")

    # 6. Template features per query. Save as sidecar NPZs under
    # <release>/templates/<record_id>.npz — each query excludes its
    # own target_id to avoid self-template collapse.
    t0 = time.time()
    template_dir = out_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    target_supervisions = {i: sup for i, sup in enumerate(supervisions)}
    template_stats = []
    for i, sup in enumerate(supervisions):
        feats = proteon.build_template_features(
            query_length=sup.length,
            engine=engine,
            target_supervisions=target_supervisions,
            query_sequence=sup.sequence,
            exclude_target_ids=[i],
            max_templates=4,
        )
        np.savez_compressed(
            template_dir / f"{sup.record_id.replace(':', '_')}.npz",
            template_aatype=feats.template_aatype,
            template_all_atom_positions=feats.template_all_atom_positions,
            template_all_atom_masks=feats.template_all_atom_masks,
            template_sum_probs=feats.template_sum_probs,
        )
        template_stats.append(
            {"record_id": sup.record_id, "n_templates": int(feats.n_templates)}
        )
    (template_dir / "templates_manifest.json").write_text(
        json.dumps({"templates": template_stats}, indent=2),
        encoding="utf-8",
    )
    timings["templates_s"] = time.time() - t0
    print(f"  template features ({sum(s['n_templates'] for s in template_stats)} "
          f"total across {len(template_stats)} queries) in {timings['templates_s']:.2f}s")

    # 7. Training release with denormalized NPZ.
    t0 = time.time()
    split_assignments = {rid: ("val" if i == len(record_ids) - 1 else "train")
                         for i, rid in enumerate(record_ids)}
    training_root = proteon.build_training_release(
        sequence_root,
        struc_root / "supervision_release",
        out_dir / "training",
        release_id=f"{release_id}-training",
        split_assignments=split_assignments,
        code_rev="local",
        config_rev="local",
        provenance={"input_paths": [str(p) for p in inputs]},
        overwrite=True,
    )
    timings["training_s"] = time.time() - t0
    print(f"  training release in {timings['training_s']:.2f}s")

    # 8. Corpus release manifest.
    t0 = time.time()
    corpus_root = proteon.build_corpus_release_manifest(
        out_dir / "corpus",
        release_id=release_id,
        prepared_manifest=struc_root / "prepared_structures.jsonl",
        sequence_release=sequence_root,
        structure_release=struc_root / "supervision_release",
        training_release=training_root,
        code_rev="local",
        config_rev="local",
        prep_policy_version="v0",
        split_policy_version="v0",
        provenance={
            "input_paths": [str(p) for p in inputs],
            "templates_dir": str(template_dir),
            "template_stats": template_stats,
        },
        overwrite=True,
    )
    timings["corpus_s"] = time.time() - t0

    # 9. Validation.
    proteon.validate_corpus_release(
        corpus_root / "corpus_release_manifest.json",
        out_path=corpus_root / "validation_report.json",
    )

    return {
        "out_dir": str(out_dir),
        "inputs": [str(p) for p in inputs],
        "timings_s": timings,
        "template_stats": template_stats,
    }


def summarize(report: dict) -> None:
    out_dir = Path(report["out_dir"])

    print("\n=== SUMMARY ===")
    corpus = json.loads(
        (out_dir / "corpus" / "corpus_release_manifest.json").read_text()
    )
    for key in (
        "count_prepared",
        "count_sequence_examples",
        "count_structure_examples",
        "count_training_examples",
        "count_ingestion_failures",
        "split_counts",
        "failure_breakdown",
    ):
        print(f"  {key}: {corpus[key]}")

    validation = json.loads(
        (out_dir / "corpus" / "validation_report.json").read_text()
    )
    print(f"  validation_issues: {len(validation.get('issues', []))}")

    trn_mf = json.loads((out_dir / "training" / "release_manifest.json").read_text())
    print(f"\n  training parquet_sha256: {trn_mf['parquet_sha256'][:16]}…")
    print(f"  training parquet_fields: {len(trn_mf['parquet_fields'])}")

    seq_mf = json.loads((out_dir / "sequence" / "examples" / "manifest.json").read_text())
    print(f"  sequence tensor_sha256: {seq_mf['tensor_sha256'][:16]}…")

    sup_mf = json.loads(
        (out_dir / "prepared" / "supervision_release" / "examples" / "manifest.json").read_text()
    )
    print(f"  supervision tensor_sha256: {sup_mf['tensor_sha256'][:16]}…")

    # Load training Parquet + verify shape.
    trn_examples = proteon.load_training_examples(out_dir / "training")
    print(f"\n  training examples loaded: {len(trn_examples)}")
    for ex in trn_examples:
        struc = ex.structure
        seq = ex.sequence
        msa_info = f"msa={None if seq.msa is None else seq.msa.shape}"
        print(
            f"    {ex.record_id:10s}  split={ex.split:5s}  L={seq.length:3d}  "
            f"atom14={struc.atom14_gt_positions.shape}  {msa_info}"
        )

    print(f"\n  total wall: {sum(report['timings_s'].values()):.2f}s")
    for stage, t in report["timings_s"].items():
        print(f"    {stage:22s} {t:.2f}s")
    print(f"\n  template stats: {report['template_stats']}")
    print("\nOK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/proteon_first_real_search")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--release-id", default="proteon-smoke-real-v0")
    args = parser.parse_args()

    if proteon.io._io is None:
        print("ERROR: Rust I/O backend unavailable. Run maturin develop -r in "
              "proteon-connector/ first.", file=sys.stderr)
        return 1
    if not proteon.rust_msa_available():
        print("ERROR: Rust MSA backend unavailable. Same fix as above.", file=sys.stderr)
        return 1

    report = run(args.inputs, Path(args.out), args.release_id)
    summarize(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
