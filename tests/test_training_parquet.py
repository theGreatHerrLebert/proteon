"""Round-trip tests for the Parquet training release."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from proteon.sequence_example import SequenceExample
from proteon.supervision import StructureSupervisionExample
from proteon.training_example import (
    TRAINING_EXPORT_FORMAT,
    TrainingExample,
    _build_training_schema,
    _make_ragged_column,
    _training_examples_to_record_batch,
    build_training_release,
    iter_training_examples,
    load_training_examples,
)


def _fake_seq(record_id: str, chain_id: str, L: int, seed: int) -> SequenceExample:
    rng = np.random.default_rng(seed)
    return SequenceExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        chain_id=chain_id,
        sequence="A" * L,
        length=L,
        code_rev=None,
        config_rev=None,
        aatype=rng.integers(0, 20, size=L, dtype=np.int32),
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        seq_mask=np.ones(L, dtype=np.float32),
    )


def _fake_struc(record_id: str, chain_id: str, L: int, seed: int) -> StructureSupervisionExample:
    rng = np.random.default_rng(seed + 1)
    return StructureSupervisionExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        prep_run_id=None,
        chain_id=chain_id,
        sequence="A" * L,
        length=L,
        code_rev=None,
        config_rev=None,
        aatype=rng.integers(0, 20, size=L, dtype=np.int32),
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        seq_mask=np.ones(L, dtype=np.float32),
        all_atom_positions=rng.standard_normal((L, 37, 3), dtype=np.float32),
        all_atom_mask=rng.random((L, 37), dtype=np.float32),
        atom37_atom_exists=rng.random((L, 37), dtype=np.float32),
        atom14_gt_positions=rng.standard_normal((L, 14, 3), dtype=np.float32),
        atom14_gt_exists=rng.random((L, 14), dtype=np.float32),
        atom14_atom_exists=rng.random((L, 14), dtype=np.float32),
        residx_atom14_to_atom37=rng.integers(0, 37, size=(L, 14), dtype=np.int32),
        residx_atom37_to_atom14=rng.integers(0, 14, size=(L, 37), dtype=np.int32),
        atom14_atom_is_ambiguous=rng.random((L, 14), dtype=np.float32),
        pseudo_beta=rng.standard_normal((L, 3), dtype=np.float32),
        pseudo_beta_mask=rng.random(L, dtype=np.float32),
        phi=rng.standard_normal(L, dtype=np.float32),
        psi=rng.standard_normal(L, dtype=np.float32),
        omega=rng.standard_normal(L, dtype=np.float32),
        phi_mask=rng.random(L, dtype=np.float32),
        psi_mask=rng.random(L, dtype=np.float32),
        omega_mask=rng.random(L, dtype=np.float32),
        chi_angles=rng.standard_normal((L, 4), dtype=np.float32),
        chi_mask=rng.random((L, 4), dtype=np.float32),
        rigidgroups_gt_frames=rng.standard_normal((L, 8, 4, 4), dtype=np.float32),
        rigidgroups_gt_exists=rng.random((L, 8), dtype=np.float32),
        rigidgroups_group_exists=rng.random((L, 8), dtype=np.float32),
        rigidgroups_group_is_ambiguous=rng.random((L, 8), dtype=np.float32),
        quality=None,
    )


def _fake_training_example(record_id: str, L: int, split: str, seed: int) -> TrainingExample:
    return TrainingExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        chain_id="A",
        split=split,
        crop_start=None,
        crop_stop=None,
        weight=1.0,
        sequence=_fake_seq(record_id, "A", L, seed),
        structure=_fake_struc(record_id, "A", L, seed),
    )


def _write_release(tmp_path: Path, training_examples, row_group_size: int = 512) -> Path:
    # Write synthetic sequence + structure releases on disk, then
    # build the training release from them via the real public API.
    # The release builders expect per-exporter output to live under
    # `<release>/examples/` so the training-release loader can find it.
    from proteon.sequence_export import export_sequence_examples
    from proteon.supervision_export import export_structure_supervision_examples

    seq_dir = tmp_path / "sequence"
    struc_dir = tmp_path / "structure"
    export_sequence_examples([ex.sequence for ex in training_examples], seq_dir / "examples", overwrite=True)
    export_structure_supervision_examples([ex.structure for ex in training_examples], struc_dir / "examples", overwrite=True)

    out = tmp_path / "training"
    build_training_release(
        seq_dir,
        struc_dir,
        out,
        release_id="test-v0",
        split_assignments={ex.record_id: ex.split for ex in training_examples},
        weights={ex.record_id: ex.weight for ex in training_examples},
        overwrite=True,
        row_group_size=row_group_size,
    )
    return out


def test_iter_training_examples_batch_size_yields_lists(tmp_path):
    """`batch_size=N` yields list[TrainingExample] of length ≤ N.

    Previously the parameter was silently ignored and the iterator
    always yielded single TrainingExamples, breaking the public
    contract documented in the docstring.
    """
    examples = [
        _fake_training_example(f"r{i}", L=4 + (i % 3), split="train", seed=i)
        for i in range(7)
    ]
    out = _write_release(tmp_path, examples, row_group_size=2)

    chunks = list(iter_training_examples(out, batch_size=3))
    # 7 examples, batch_size=3 → [3, 3, 1].
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert all(isinstance(c, list) for c in chunks)
    assert all(isinstance(ex, TrainingExample) for c in chunks for ex in c)
    flat_ids = [ex.record_id for c in chunks for ex in c]
    assert flat_ids == [ex.record_id for ex in examples]

    # batch_size=None → single-example yields (unchanged behavior).
    single = list(iter_training_examples(out, batch_size=None))
    assert [ex.record_id for ex in single] == [ex.record_id for ex in examples]
    assert all(isinstance(ex, TrainingExample) for ex in single)


def test_iter_training_examples_rejects_nonpositive_batch_size(tmp_path):
    examples = [_fake_training_example("r0", L=3, split="train", seed=0)]
    out = _write_release(tmp_path, examples)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(iter_training_examples(out, batch_size=0))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(iter_training_examples(out, batch_size=-1))


def test_scalar_fields_round_trip(tmp_path):
    """Regression: sequence / code_rev / config_rev / prep_run_id / quality
    must round-trip through training.parquet. Previously the reader zeroed
    them (sequence='', revs=None, prep_run_id=None, quality=None)."""
    from proteon.supervision import StructureQualityMetadata
    rng = np.random.default_rng(99)
    L = 5
    seq = SequenceExample(
        record_id="with_revs", source_id="src.pdb", chain_id="A",
        sequence="MKLVV", length=L,
        code_rev="abc123", config_rev="def456",
        aatype=rng.integers(0, 20, size=L, dtype=np.int32),
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        seq_mask=np.ones(L, dtype=np.float32),
    )
    struc = _fake_struc("with_revs", "A", L, seed=99)
    # Overwrite the fake with scalars we want to assert on.
    struc.sequence = "MKLVV"
    struc.code_rev = "abc123"
    struc.config_rev = "def456"
    struc.prep_run_id = "prep-2026-04-17-T00:00"
    struc.quality = StructureQualityMetadata(
        prep_success=True, force_field="amber96", atoms_reconstructed=3, hydrogens_added=12,
    )
    ex = TrainingExample(
        record_id="with_revs", source_id="src.pdb", chain_id="A",
        split="train", crop_start=None, crop_stop=None, weight=1.0,
        sequence=seq, structure=struc,
    )
    out = _write_release(tmp_path, [ex])
    loaded = list(iter_training_examples(out))
    assert len(loaded) == 1
    got = loaded[0]
    assert got.sequence.sequence == "MKLVV"
    assert got.structure.sequence == "MKLVV"
    assert got.sequence.code_rev == "abc123"
    assert got.sequence.config_rev == "def456"
    assert got.structure.code_rev == "abc123"
    assert got.structure.config_rev == "def456"
    assert got.structure.prep_run_id == "prep-2026-04-17-T00:00"
    assert got.structure.quality is not None
    assert got.structure.quality.prep_success is True
    assert got.structure.quality.force_field == "amber96"
    assert got.structure.quality.atoms_reconstructed == 3
    assert got.structure.quality.hydrogens_added == 12


def test_schema_has_expected_fields():
    fields = {f.name for f in _build_training_schema()}
    expected = {
        "record_id", "source_id", "chain_id", "split",
        "length", "weight", "crop_start", "crop_stop",
        # Round-trip scalars added 2026-04-17 — previously zeroed on load.
        "sequence", "code_rev", "config_rev", "prep_run_id", "quality_json",
        "aatype", "residue_index", "seq_mask",
        "all_atom_positions", "all_atom_mask", "atom37_atom_exists",
        "atom14_gt_positions", "atom14_gt_exists", "atom14_atom_exists",
        "atom14_atom_is_ambiguous", "residx_atom14_to_atom37", "residx_atom37_to_atom14",
        "pseudo_beta", "pseudo_beta_mask",
        "phi", "psi", "omega", "phi_mask", "psi_mask", "omega_mask",
        "chi_angles", "chi_mask",
        "rigidgroups_gt_frames", "rigidgroups_gt_exists",
        "rigidgroups_group_exists", "rigidgroups_group_is_ambiguous",
    }
    assert fields == expected


def test_make_ragged_column_preserves_values():
    import pyarrow as pa
    arrays = [np.arange(5 * 37 * 3, dtype=np.float32).reshape(5, 37, 3),
              np.arange(3 * 37 * 3, dtype=np.float32).reshape(3, 37, 3)]
    col = _make_ragged_column(arrays, (37, 3), np.float32)
    assert len(col) == 2
    # Round-trip through a Table to get per-row nested Python lists.
    tbl = pa.table({"x": col})
    pyrows = tbl.column("x").to_pylist()
    row0 = np.asarray(pyrows[0], dtype=np.float32)
    row1 = np.asarray(pyrows[1], dtype=np.float32)
    assert row0.shape == (5, 37, 3)
    assert row1.shape == (3, 37, 3)
    assert np.array_equal(row0, arrays[0])
    assert np.array_equal(row1, arrays[1])


def test_variable_length_round_trip_bit_identical(tmp_path):
    # Two examples of different L to exercise ragged storage.
    examples = [
        _fake_training_example("rec_A", L=12, split="train", seed=1),
        _fake_training_example("rec_B", L=7, split="val", seed=2),
    ]
    release_dir = _write_release(tmp_path, examples, row_group_size=8)
    loaded = load_training_examples(release_dir)
    assert len(loaded) == 2
    loaded_by_id = {ex.record_id: ex for ex in loaded}
    for orig in examples:
        got = loaded_by_id[orig.record_id]
        assert got.split == orig.split
        assert got.sequence.length == orig.sequence.length
        assert np.array_equal(got.sequence.aatype, orig.sequence.aatype)
        assert np.array_equal(got.structure.all_atom_positions, orig.structure.all_atom_positions)
        assert np.array_equal(got.structure.rigidgroups_gt_frames, orig.structure.rigidgroups_gt_frames)
        assert np.array_equal(got.structure.residx_atom14_to_atom37, orig.structure.residx_atom14_to_atom37)
        assert np.array_equal(got.structure.chi_angles, orig.structure.chi_angles)
        assert np.array_equal(got.structure.pseudo_beta, orig.structure.pseudo_beta)


def test_row_group_chunking_writes_multiple_groups(tmp_path):
    # 5 examples at row_group_size=2 should produce 3 row groups.
    examples = [_fake_training_example(f"r_{i}", L=6, split="train", seed=10 + i) for i in range(5)]
    release_dir = _write_release(tmp_path, examples, row_group_size=2)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(release_dir / "training.parquet")
    assert pf.metadata.num_row_groups == 3
    loaded = load_training_examples(release_dir)
    assert len(loaded) == 5


def test_split_filter_is_predicate_pushed(tmp_path):
    examples = [
        _fake_training_example("t1", 5, "train", 1),
        _fake_training_example("t2", 5, "train", 2),
        _fake_training_example("v1", 5, "val", 3),
        _fake_training_example("e1", 5, "test", 4),
    ]
    release_dir = _write_release(tmp_path, examples, row_group_size=2)
    train_only = list(iter_training_examples(release_dir, split="train"))
    val_only = list(iter_training_examples(release_dir, split="val"))
    test_only = list(iter_training_examples(release_dir, split="test"))
    assert {ex.record_id for ex in train_only} == {"t1", "t2"}
    assert {ex.record_id for ex in val_only} == {"v1"}
    assert {ex.record_id for ex in test_only} == {"e1"}


def test_manifest_records_parquet_metadata(tmp_path):
    examples = [_fake_training_example("only", 8, "train", 42)]
    release_dir = _write_release(tmp_path, examples)
    import json
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert manifest["format"] == TRAINING_EXPORT_FORMAT
    assert manifest["parquet_file"] == "training.parquet"
    assert manifest["parquet_sha256"] is not None
    assert manifest["count_examples"] == 1
    assert manifest["split_counts"] == {"train": 1}
    assert manifest["parquet_schema_version"] == 1
    assert "aatype" in manifest["parquet_fields"]


def test_checksum_verification_detects_tamper(tmp_path):
    examples = [_fake_training_example("t", 5, "train", 1)]
    release_dir = _write_release(tmp_path, examples)
    parquet_path = release_dir / "training.parquet"
    parquet_path.write_bytes(parquet_path.read_bytes()[:-10] + b"corrupted1")
    with pytest.raises(Exception):
        load_training_examples(release_dir, verify_checksum=True)
