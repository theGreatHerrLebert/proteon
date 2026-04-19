"""Tests for the local corpus smoke-release builder."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import proteon
import proteon.corpus_smoke as corpus_smoke


def _fake_structure(name: str):
    residues = [
        SimpleNamespace(name="GLY", serial_number=1, is_amino_acid=True, atoms=[]),
        SimpleNamespace(name="SER", serial_number=2, is_amino_acid=True, atoms=[]),
    ]
    chain = SimpleNamespace(id="A", residues=residues)
    return SimpleNamespace(identifier=name, chain_count=1, chains=[chain], residue_count=2, atom_count=10)


def _write_fake_supervision_parquet(path: Path, n: int = 2, length: int = 2) -> None:
    """Emit a minimal supervision Parquet file matching the real schema.

    Only the columns the corpus validator reads are populated with ones
    (seq_mask, pseudo_beta_mask, rigidgroups_gt_exists, chi_mask); the
    rest are populated with zeros at the right shape so the file still
    type-checks against the schema.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from proteon.supervision_export import (
        TENSOR_FIELDS,
        build_supervision_schema,
        _make_ragged_column,
    )

    schema = build_supervision_schema()
    columns = [
        pa.array([f"r{i}" for i in range(n)], type=pa.string()),
        pa.array([None] * n, type=pa.string()),  # source_id
        pa.array([None] * n, type=pa.string()),  # prep_run_id
        pa.array(["A"] * n, type=pa.string()),   # chain_id
        pa.array(["A" * length] * n, type=pa.string()),  # sequence
        pa.array([length] * n, type=pa.int32()),  # length
        pa.array([None] * n, type=pa.string()),  # code_rev
        pa.array([None] * n, type=pa.string()),  # config_rev
        pa.array([None] * n, type=pa.string()),  # quality_json
    ]
    for name, inner_shape, dtype, _attr in TENSOR_FIELDS:
        fill = np.ones if name in {"seq_mask", "pseudo_beta_mask", "rigidgroups_gt_exists", "chi_mask"} else np.zeros
        per_row = [fill((length,) + inner_shape, dtype=dtype) for _ in range(n)]
        columns.append(_make_ragged_column(per_row, inner_shape, dtype))
    batch = pa.RecordBatch.from_arrays(columns, schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=3) as w:
        w.write_batch(batch)


def _write_fake_supervision_tree(sup: Path, n: int = 2, length: int = 2) -> None:
    """Write a supervision_release/ directory tree the validator will accept."""
    sup.mkdir(parents=True, exist_ok=True)
    (sup / "release_manifest.json").write_text(
        json.dumps({"count_examples": n, "lengths": {"mean": float(length)}}),
        encoding="utf-8",
    )
    (sup / "failures.jsonl").write_text("", encoding="utf-8")
    (sup / "examples").mkdir(exist_ok=True)
    _write_fake_supervision_parquet(sup / "examples" / "tensors.parquet", n=n, length=length)
    (sup / "examples" / "manifest.json").write_text(
        json.dumps({
            "format": "proteon.structure_supervision.parquet.v0",
            "schema_version": 1,
            "count": n,
            "examples_file": "examples.jsonl",
            "row_group_size": 512,
            "tensor_file": "tensors.parquet",
        }),
        encoding="utf-8",
    )
    (sup / "examples" / "examples.jsonl").write_text(
        "\n".join(json.dumps({"record_id": f"r{i}", "chain_id": "A", "length": length, "sequence": "A" * length}) for i in range(n)) + "\n",
        encoding="utf-8",
    )


def test_build_local_corpus_smoke_release_orchestrates_pipeline(tmp_path, monkeypatch):
    loaded = [(0, _fake_structure("one")), (1, _fake_structure("two"))]

    def fake_batch_load_tolerant(paths, n_threads=None):
        return loaded

    def fake_batch_prepare(structures, n_threads=None):
        return [proteon.PrepReport(hydrogens_added=1, converged=True) for _ in structures]

    def fake_build_structure_supervision_dataset_from_prepared(structures, prep_reports, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prepared_structures.jsonl").write_text('{"record_id":"one"}\n{"record_id":"two"}\n', encoding="utf-8")
        _write_fake_supervision_tree(out / "supervision_release", n=2, length=2)
        return out

    def fake_build_sequence_dataset(structures, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(json.dumps({"count_examples": 2, "lengths": {"mean": 2.0}}), encoding="utf-8")
        (out / "failures.jsonl").write_text("", encoding="utf-8")
        (out / "examples").mkdir(exist_ok=True)
        return out

    def fake_build_training_release(sequence_release_dir, structure_release_dir, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(json.dumps({"count_examples": 2, "split_counts": {"train": 1, "val": 1}}), encoding="utf-8")
        (out / "training_examples.jsonl").write_text(
            '{"record_id":"one","split":"train"}\n{"record_id":"two","split":"val"}\n',
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(corpus_smoke, "batch_load_tolerant", fake_batch_load_tolerant)
    monkeypatch.setattr(corpus_smoke, "batch_prepare", fake_batch_prepare)
    monkeypatch.setattr(corpus_smoke, "build_structure_supervision_dataset_from_prepared", fake_build_structure_supervision_dataset_from_prepared)
    monkeypatch.setattr(corpus_smoke, "build_sequence_dataset", fake_build_sequence_dataset)
    monkeypatch.setattr(corpus_smoke, "build_training_release", fake_build_training_release)

    root = corpus_smoke.build_local_corpus_smoke_release(
        [tmp_path / "one.pdb", tmp_path / "two.pdb"],
        tmp_path / "smoke",
        release_id="smoke-v0",
        overwrite=True,
    )

    assert (root / "prepared" / "prepared_structures.jsonl").exists()
    assert (root / "sequence" / "release_manifest.json").exists()
    assert (root / "training" / "release_manifest.json").exists()
    assert (root / "corpus" / "corpus_release_manifest.json").exists()
    assert (root / "corpus" / "validation_report.json").exists()


def test_partial_ingestion_emits_failure_records(tmp_path, monkeypatch):
    """3 inputs requested, 2 loaded → 1 ingestion failure must appear in
    the corpus manifest's failure_breakdown and count_ingestion_failures.
    Regression guard for "silently drops unreadable inputs" bug: a
    partial release must never look clean at the top-level manifest.
    """
    loaded = [(0, _fake_structure("one")), (2, _fake_structure("three"))]

    def fake_batch_load_tolerant(paths, n_threads=None):
        # idx=1 ("two.pdb") intentionally dropped — no pair for it.
        return loaded

    def fake_batch_prepare(structures, n_threads=None):
        return [proteon.PrepReport(hydrogens_added=1, converged=True) for _ in structures]

    def fake_supervision(structures, prep_reports, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prepared_structures.jsonl").write_text(
            '{"record_id":"one"}\n{"record_id":"three"}\n', encoding="utf-8"
        )
        sup = out / "supervision_release"
        sup.mkdir(parents=True, exist_ok=True)
        (sup / "release_manifest.json").write_text(
            json.dumps({"count_examples": 2, "lengths": {"mean": 2.0}}), encoding="utf-8"
        )
        (sup / "failures.jsonl").write_text("", encoding="utf-8")
        (sup / "examples").mkdir(exist_ok=True)
        import numpy as np
        np.savez_compressed(
            sup / "examples" / "tensors.npz",
            seq_mask=np.ones((2, 2), dtype=np.float32),
            rigidgroups_gt_exists=np.ones((2, 2, 8), dtype=np.float32),
            pseudo_beta_mask=np.ones((2, 2), dtype=np.float32),
            chi_mask=np.ones((2, 2, 4), dtype=np.float32),
        )
        return out

    def fake_sequence(structures, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(
            json.dumps({"count_examples": 2, "lengths": {"mean": 2.0}}), encoding="utf-8"
        )
        (out / "failures.jsonl").write_text("", encoding="utf-8")
        (out / "examples").mkdir(exist_ok=True)
        return out

    def fake_training(*args, **kwargs):
        out = Path(kwargs.get("out_dir") or args[2])
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(
            json.dumps({"count_examples": 2, "split_counts": {"train": 1, "val": 1}}),
            encoding="utf-8",
        )
        (out / "training_examples.jsonl").write_text(
            '{"record_id":"one","split":"train"}\n{"record_id":"three","split":"val"}\n',
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(corpus_smoke, "batch_load_tolerant", fake_batch_load_tolerant)
    monkeypatch.setattr(corpus_smoke, "batch_prepare", fake_batch_prepare)
    monkeypatch.setattr(corpus_smoke, "build_structure_supervision_dataset_from_prepared", fake_supervision)
    monkeypatch.setattr(corpus_smoke, "build_sequence_dataset", fake_sequence)
    monkeypatch.setattr(corpus_smoke, "build_training_release", fake_training)

    root = corpus_smoke.build_local_corpus_smoke_release(
        [tmp_path / "one.pdb", tmp_path / "two.pdb", tmp_path / "three.pdb"],
        tmp_path / "partial_smoke",
        release_id="smoke-partial-v0",
        overwrite=True,
    )

    # 1. Per-file failure row exists with the canonical parse_error class.
    failures_path = root / "ingestion_failures.jsonl"
    assert failures_path.exists(), "ingestion_failures.jsonl not written"
    lines = [line for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["failure_class"] == "parse_error"
    assert row["stage"] == "raw_intake"
    assert "two.pdb" in row["source_id"]

    # 2. Top-level manifest counts it and merges into failure_breakdown.
    manifest = json.loads((root / "corpus" / "corpus_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count_ingestion_failures"] == 1
    assert manifest["failure_breakdown"].get("parse_error", 0) == 1


def test_rescue_load_writes_rescued_input_manifest(tmp_path, monkeypatch):
    rescued = proteon.LoadRescueResult(
        structure=_fake_structure("one"),
        path=tmp_path / "one.pdb",
        rescued=True,
        rescue_bucket=proteon.LoaderFailureBucket(
            code="seqadv_invalid_field",
            rescueable=True,
            summary="bad seqadv",
            rescue_strategy="drop seqadv",
        ),
        rescue_steps=("drop_seqadv",),
        original_error="Failed to read one.pdb: Invalid data in field",
    )

    def fake_batch_load_tolerant_with_rescue(paths, n_threads=None, allow=None, force_format=None):
        del n_threads, allow, force_format
        return [(0, rescued)]

    def fake_batch_prepare(structures, n_threads=None):
        return [proteon.PrepReport(hydrogens_added=1, converged=True) for _ in structures]

    def fake_supervision(structures, prep_reports, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prepared_structures.jsonl").write_text('{"record_id":"one"}\n', encoding="utf-8")
        sup = out / "supervision_release"
        sup.mkdir(parents=True, exist_ok=True)
        (sup / "release_manifest.json").write_text(
            json.dumps({"count_examples": 1, "lengths": {"mean": 2.0}}), encoding="utf-8"
        )
        (sup / "failures.jsonl").write_text("", encoding="utf-8")
        (sup / "examples").mkdir(exist_ok=True)
        import numpy as np
        np.savez_compressed(
            sup / "examples" / "tensors.npz",
            seq_mask=np.ones((1, 2), dtype=np.float32),
            rigidgroups_gt_exists=np.ones((1, 2, 8), dtype=np.float32),
            pseudo_beta_mask=np.ones((1, 2), dtype=np.float32),
            chi_mask=np.ones((1, 2, 4), dtype=np.float32),
        )
        return out

    def fake_sequence(structures, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(
            json.dumps({"count_examples": 1, "lengths": {"mean": 2.0}}), encoding="utf-8"
        )
        (out / "failures.jsonl").write_text("", encoding="utf-8")
        (out / "examples").mkdir(exist_ok=True)
        return out

    def fake_training(*args, **kwargs):
        out = Path(kwargs.get("out_dir") or args[2])
        out.mkdir(parents=True, exist_ok=True)
        (out / "release_manifest.json").write_text(
            json.dumps({"count_examples": 1, "split_counts": {"train": 1}}),
            encoding="utf-8",
        )
        (out / "training_examples.jsonl").write_text(
            '{"record_id":"one","split":"train"}\n',
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(corpus_smoke, "batch_load_tolerant_with_rescue", fake_batch_load_tolerant_with_rescue)
    monkeypatch.setattr(corpus_smoke, "batch_prepare", fake_batch_prepare)
    monkeypatch.setattr(corpus_smoke, "build_structure_supervision_dataset_from_prepared", fake_supervision)
    monkeypatch.setattr(corpus_smoke, "build_sequence_dataset", fake_sequence)
    monkeypatch.setattr(corpus_smoke, "build_training_release", fake_training)

    root = corpus_smoke.build_local_corpus_smoke_release(
        [tmp_path / "one.pdb"],
        tmp_path / "rescued_smoke",
        release_id="smoke-rescue-v0",
        rescue_load=True,
        overwrite=True,
    )

    rescued_manifest = root / "rescued_inputs.jsonl"
    assert rescued_manifest.exists()
    rows = [json.loads(line) for line in rescued_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["rescue_bucket"] == "seqadv_invalid_field"
    assert rows[0]["rescue_steps"] == ["drop_seqadv"]

    manifest = json.loads((root / "corpus" / "corpus_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count_rescued_inputs"] == 1
    assert manifest["rescued_inputs_manifest"].endswith("rescued_inputs.jsonl")
    assert manifest["provenance"]["rescue_load"] is True
    assert len(manifest["provenance"]["rescued_paths"]) == 1


def _fake_structure(name: str):
    residues = [
        SimpleNamespace(name="GLY", serial_number=1, is_amino_acid=True, atoms=[]),
        SimpleNamespace(name="SER", serial_number=2, is_amino_acid=True, atoms=[]),
    ]
    chain = SimpleNamespace(id="A", residues=residues)
    return SimpleNamespace(identifier=name, chain_count=1, chains=[chain], residue_count=2, atom_count=10)
