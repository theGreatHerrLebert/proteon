"""Streaming and MSA-handling tests for the sequence Parquet path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from proteon.sequence_example import SequenceExample
from proteon.sequence_export import (
    SEQUENCE_EXPORT_FORMAT,
    SequenceParquetWriter,
    build_sequence_schema,
    export_sequence_examples,
    iter_sequence_examples,
    load_sequence_examples,
)


def _fake_sequence(
    record_id: str,
    L: int,
    seed: int,
    *,
    depth: int = 0,
    with_template: bool = False,
) -> SequenceExample:
    rng = np.random.default_rng(seed)
    msa = None
    deletion = None
    msa_mask = None
    if depth > 0:
        msa = rng.integers(0, 22, size=(depth, L), dtype=np.int32)
        deletion = rng.random(size=(depth, L), dtype=np.float32)
        msa_mask = np.ones((depth, L), dtype=np.float32)
    template = None
    if with_template:
        template = rng.random(size=(3,), dtype=np.float32)
    return SequenceExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        chain_id="A",
        sequence="X" * L,
        length=L,
        code_rev=None,
        config_rev=None,
        aatype=rng.integers(0, 20, size=(L,), dtype=np.int32),
        residue_index=np.arange(L, dtype=np.int32),
        seq_mask=np.ones((L,), dtype=np.float32),
        msa=msa,
        deletion_matrix=deletion,
        msa_mask=msa_mask,
        template_mask=template,
    )


def _assert_examples_equal(a: SequenceExample, b: SequenceExample) -> None:
    assert a.record_id == b.record_id
    assert a.chain_id == b.chain_id
    assert a.sequence == b.sequence
    assert a.length == b.length
    for attr in ("aatype", "residue_index", "seq_mask"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    for attr in ("msa", "deletion_matrix", "msa_mask", "template_mask"):
        av, bv = getattr(a, attr), getattr(b, attr)
        if av is None:
            assert bv is None, f"{attr}: expected None, got {bv!r}"
        else:
            np.testing.assert_array_equal(av, bv)


def test_sequence_parquet_roundtrip_with_and_without_msa(tmp_path: Path):
    examples = [
        _fake_sequence("a", L=5, seed=0, depth=4),
        _fake_sequence("b", L=7, seed=1, depth=0),  # no MSA at all
        _fake_sequence("c", L=3, seed=2, depth=2, with_template=True),
    ]
    out = export_sequence_examples(examples, tmp_path / "seq", row_group_size=2)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["format"] == SEQUENCE_EXPORT_FORMAT
    assert manifest["schema_version"] == 1
    assert manifest["count"] == 3
    assert manifest["tensor_file"] == "tensors.parquet"

    loaded = load_sequence_examples(out)
    assert len(loaded) == 3
    for src, dst in zip(examples, loaded):
        _assert_examples_equal(src, dst)


def test_sequence_parquet_iter_streams_rowgroup_at_a_time(tmp_path: Path):
    examples = [_fake_sequence(f"r{i}", L=4 + (i % 3), seed=i, depth=2) for i in range(10)]
    out = export_sequence_examples(examples, tmp_path / "seq", row_group_size=3)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(out / "tensors.parquet")
    # ceil(10/3) = 4 row groups; writer flushes the tail.
    assert pf.metadata.num_row_groups == 4

    streamed = list(iter_sequence_examples(out, verify_checksum=True))
    assert [ex.record_id for ex in streamed] == [ex.record_id for ex in examples]
    for src, dst in zip(examples, streamed):
        _assert_examples_equal(src, dst)


def test_sequence_parquet_mixed_null_and_nonnull_msa_same_rowgroup(tmp_path: Path):
    """MSA is ragged on both axes; null MSAs share row groups with non-null ones."""
    batch = [
        _fake_sequence("has_msa_a", L=3, seed=10, depth=4),
        _fake_sequence("no_msa_a", L=3, seed=11, depth=0),
        _fake_sequence("has_msa_b", L=5, seed=12, depth=2),
        _fake_sequence("no_msa_b", L=2, seed=13, depth=0),
    ]
    out = export_sequence_examples(batch, tmp_path / "seq", row_group_size=16)

    loaded = load_sequence_examples(out)
    assert loaded[0].msa.shape == (4, 3)
    assert loaded[1].msa is None
    assert loaded[2].msa.shape == (2, 5)
    assert loaded[3].msa is None
    for src, dst in zip(batch, loaded):
        _assert_examples_equal(src, dst)


def test_sequence_parquet_empty_release(tmp_path: Path):
    out = export_sequence_examples([], tmp_path / "seq")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["count"] == 0
    assert "tensor_file" not in manifest  # no parquet written for empty release
    # Directory must agree with the manifest: no tensors.parquet on disk
    # when the manifest says there isn't one. Prior behavior left a
    # zero-row file behind.
    assert not (out / "tensors.parquet").exists()
    assert list(iter_sequence_examples(out)) == []
    assert load_sequence_examples(out) == []


def test_sequence_release_outer_manifest_agrees_on_empty(tmp_path: Path):
    """Outer release_manifest.json must not point at a nonexistent
    tensors.parquet on empty releases. Prior dataclass default was
    tensor_file = 'examples/tensors.parquet' regardless of whether
    the file was written."""
    from proteon.sequence_release import build_sequence_release

    root = build_sequence_release([], tmp_path / "release", release_id="empty")
    outer = json.loads((root / "release_manifest.json").read_text())
    assert outer["count_examples"] == 0
    assert outer["tensor_file"] is None
    # Also verify the advertised path doesn't exist (if it did, consumers
    # could reasonably trust the manifest and try to open it).
    assert not (root / "examples" / "tensors.parquet").exists()


def test_sequence_release_outer_manifest_points_at_parquet_when_nonempty(tmp_path: Path):
    from proteon.sequence_release import build_sequence_release

    root = build_sequence_release(
        [_fake_sequence("a", L=3, seed=0, depth=1)],
        tmp_path / "release",
        release_id="one",
    )
    outer = json.loads((root / "release_manifest.json").read_text())
    assert outer["count_examples"] == 1
    assert outer["tensor_file"] == "examples/tensors.parquet"
    assert (root / outer["tensor_file"]).exists()


def test_build_sequence_release_consumes_generator(tmp_path: Path):
    """Regression: build_sequence_release must not materialize the input.

    The old path called `examples = list(examples)` + `lengths = [ex.length
    for ex in examples]`, which held the full corpus (including MSA
    blocks) in Python memory before export. This test verifies the
    release accepts a generator that can only be consumed once and
    produces the correct artifact.
    """
    from proteon.sequence_release import build_sequence_release

    def _one_shot_gen():
        for i in range(5):
            yield _fake_sequence(f"r{i}", L=3 + i, seed=i, depth=4)

    root = build_sequence_release(
        _one_shot_gen(),
        tmp_path / "release",
        release_id="stream-smoke",
    )
    loaded = load_sequence_examples(root / "examples")
    assert [ex.record_id for ex in loaded] == [f"r{i}" for i in range(5)]
    # Every example carries its MSA (depth=4) through the streaming path.
    for ex in loaded:
        assert ex.msa is not None and ex.msa.shape == (4, ex.length)


def test_sequence_parquet_writer_context_manager(tmp_path: Path):
    """Callers can stream examples into the writer without materializing a list."""
    with SequenceParquetWriter(tmp_path / "seq", row_group_size=2) as w:
        for i in range(5):
            w.append(_fake_sequence(f"r{i}", L=3, seed=i, depth=1))
        assert w.count == 5
    loaded = load_sequence_examples(tmp_path / "seq")
    assert len(loaded) == 5
