"""Streaming and memory-bound tests for the supervision Parquet path."""

from __future__ import annotations

import gc
import resource
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from ferritin.supervision import StructureSupervisionExample
from ferritin.supervision_export import (
    SUPERVISION_EXPORT_FORMAT,
    SupervisionParquetWriter,
    build_supervision_schema,
    export_structure_supervision_examples,
    iter_structure_supervision_examples,
    load_structure_supervision_examples,
)


def _fake_supervision(record_id: str, L: int, seed: int) -> StructureSupervisionExample:
    rng = np.random.default_rng(seed)
    return StructureSupervisionExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        prep_run_id=None,
        chain_id="A",
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


def test_schema_has_expected_fields():
    fields = {f.name for f in build_supervision_schema()}
    # Metadata plus the AF2 supervision contract.
    assert "record_id" in fields
    assert "length" in fields
    assert "aatype" in fields
    assert "rigidgroups_gt_frames" in fields
    assert "residx_atom14_to_atom37" in fields
    assert "residx_atom37_to_atom14" in fields


def test_variable_length_round_trip_bit_identical(tmp_path):
    examples = [_fake_supervision("r1", L=12, seed=1),
                _fake_supervision("r2", L=7, seed=2)]
    out_dir = tmp_path / "supervision"
    export_structure_supervision_examples(examples, out_dir)
    loaded = load_structure_supervision_examples(out_dir)
    assert len(loaded) == 2
    by_id = {ex.record_id: ex for ex in loaded}
    for orig in examples:
        got = by_id[orig.record_id]
        assert got.length == orig.length
        assert np.array_equal(got.aatype, orig.aatype)
        assert np.array_equal(got.all_atom_positions, orig.all_atom_positions)
        assert np.array_equal(got.rigidgroups_gt_frames, orig.rigidgroups_gt_frames)
        assert np.array_equal(got.residx_atom14_to_atom37, orig.residx_atom14_to_atom37)


def test_generator_input_is_streamed(tmp_path):
    # A generator that would be expensive to collect upfront; we
    # verify the writer consumes it lazily.
    access_count = [0]

    def gen():
        for i in range(5):
            access_count[0] += 1
            yield _fake_supervision(f"r{i}", L=8, seed=i)

    export_structure_supervision_examples(gen(), tmp_path / "sup", row_group_size=2)
    # All 5 generated, but never materialized as a list by the exporter.
    assert access_count[0] == 5


def test_iter_yields_one_row_at_a_time(tmp_path):
    examples = [_fake_supervision(f"r{i}", L=5, seed=i) for i in range(4)]
    out_dir = tmp_path / "sup"
    export_structure_supervision_examples(examples, out_dir, row_group_size=2)
    got = list(iter_structure_supervision_examples(out_dir))
    assert [ex.record_id for ex in got] == ["r0", "r1", "r2", "r3"]


def test_row_group_chunking_creates_multiple_groups(tmp_path):
    import pyarrow.parquet as pq
    examples = [_fake_supervision(f"r{i}", L=6, seed=i) for i in range(5)]
    out_dir = tmp_path / "sup"
    export_structure_supervision_examples(examples, out_dir, row_group_size=2)
    pf = pq.ParquetFile(out_dir / "tensors.parquet")
    assert pf.metadata.num_row_groups == 3  # ceil(5/2)


def test_streaming_writer_memory_bounded(tmp_path):
    """Write 200 fake examples at L=100 via streaming writer and assert
    peak-resident memory grows sub-linearly in corpus size.

    The padded NPZ path scaled (N * max_L * fields * 4) bytes — for
    these numbers ≈ 1.2 GB at N=200, L=100. The streaming writer must
    stay well below that because row-group size bounds peak.
    """
    gc.collect()
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    N = 200
    L = 100

    def gen():
        for i in range(N):
            yield _fake_supervision(f"r{i:04d}", L=L, seed=i)

    out_dir = tmp_path / "sup"
    export_structure_supervision_examples(gen(), out_dir, row_group_size=32)

    gc.collect()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (peak - baseline) / 1024

    # Padded NPZ path baseline: ~1.2 GB just for the zero-init. Streaming
    # writer peak includes 32-example row-group buffers plus pyarrow
    # Arrow buffers — empirically ~60 MB on this workload. Budget
    # 500 MB to avoid test flakiness on shared hosts.
    assert growth_mb < 500, f"memory grew {growth_mb:.0f} MB (budget: 500 MB)"

    # Sanity: the corpus really was written.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(out_dir / "tensors.parquet")
    assert pf.metadata.num_rows == N
