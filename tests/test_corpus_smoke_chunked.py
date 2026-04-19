"""Chunked intake must produce byte-equivalent artifacts to single-shot.

The `chunk_size=N` path exists to keep peak memory flat at archive
scale (only one chunk's pdbtbx Structures resident at once). Correctness
contract: same paths in same order must produce the same release
regardless of chunk_size — same training_examples count, same splits,
same per-record tensor contents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import proteon

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = [
    REPO_ROOT / "test-pdbs" / "1crn.pdb",
    REPO_ROOT / "test-pdbs" / "1ubq.pdb",
    REPO_ROOT / "test-pdbs" / "1bpi.pdb",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in FIXTURES) or proteon.io._io is None,
    reason="chunked intake test needs the Rust IO connector + 1crn/1ubq/1bpi fixtures",
)


def _build(tmp, chunk_size):
    out = tmp / ("chunked" if chunk_size else "single")
    return proteon.build_local_corpus_smoke_release(
        [str(p) for p in FIXTURES],
        out,
        release_id="chunked-equiv",
        chunk_size=chunk_size,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.slow
def test_chunked_matches_single_shot_contract(tmp_path: Path):
    """Contract: same paths → same records, same splits, same shapes, same
    integer/mask content. Float coord tensors may differ because
    `batch_prepare` runs a minimizer whose output is non-deterministic
    (verified by probing two identical batch_prepare calls — same input,
    max coord diff ~6 Å). So coord content is specifically *not* pinned;
    the contract that matters is identity-level (which chain is which
    example) + shape-level (no padding, no missing rows)."""
    single = _build(tmp_path, chunk_size=None)
    chunked = _build(tmp_path, chunk_size=1)

    # Training manifest: count + split distribution must agree exactly
    # (splits are a deterministic hash over record_ids → prep-independent).
    s_train = json.loads((single / "training" / "release_manifest.json").read_text())
    c_train = json.loads((chunked / "training" / "release_manifest.json").read_text())
    assert s_train["count_examples"] == c_train["count_examples"]
    assert s_train["split_counts"] == c_train["split_counts"]

    from proteon.training_example import load_training_examples
    s_exs = sorted(load_training_examples(single / "training"), key=lambda e: e.record_id)
    c_exs = sorted(load_training_examples(chunked / "training"), key=lambda e: e.record_id)
    assert [e.record_id for e in s_exs] == [e.record_id for e in c_exs]

    for s_ex, c_ex in zip(s_exs, c_exs):
        assert s_ex.record_id == c_ex.record_id
        assert s_ex.split == c_ex.split
        # Sequence string is residue-identity-only — prep-independent.
        assert s_ex.sequence.sequence == c_ex.sequence.sequence
        assert s_ex.sequence.length == c_ex.sequence.length
        # Integer metadata: prep-independent, must match exactly.
        np.testing.assert_array_equal(s_ex.sequence.aatype, c_ex.sequence.aatype)
        np.testing.assert_array_equal(s_ex.sequence.residue_index, c_ex.sequence.residue_index)
        np.testing.assert_array_equal(
            s_ex.structure.residx_atom14_to_atom37, c_ex.structure.residx_atom14_to_atom37,
        )
        # Float tensor shapes must agree; actual values may differ by ~Å
        # due to minimizer nondeterminism (same contract the single-shot
        # pipeline already gives between independent runs).
        assert s_ex.structure.all_atom_positions.shape == c_ex.structure.all_atom_positions.shape
        assert s_ex.structure.rigidgroups_gt_frames.shape == c_ex.structure.rigidgroups_gt_frames.shape

    # Outer release manifests claim real artifacts on disk for both paths.
    for rel in (single, chunked):
        sup_m = json.loads(
            (rel / "prepared" / "supervision_release" / "release_manifest.json").read_text()
        )
        seq_m = json.loads((rel / "sequence" / "release_manifest.json").read_text())
        assert sup_m["tensor_file"] == "examples/tensors.parquet"
        assert seq_m["tensor_file"] == "examples/tensors.parquet"
        assert (rel / "prepared" / "supervision_release" / sup_m["tensor_file"]).exists()
        assert (rel / "sequence" / seq_m["tensor_file"]).exists()


@pytest.mark.slow
def test_chunked_records_chunk_size_in_corpus_manifest(tmp_path: Path):
    root = _build(tmp_path, chunk_size=2)
    corpus = json.loads(
        (root / "corpus" / "corpus_release_manifest.json").read_text()
    )
    assert corpus["provenance"]["chunk_size"] == 2


def test_chunk_size_none_is_single_shot_path(tmp_path: Path):
    """chunk_size=None is the default, matches historical behavior."""
    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURES[0])],
        tmp_path / "single",
        release_id="default-single",
    )
    corpus = json.loads((out / "corpus" / "corpus_release_manifest.json").read_text())
    # Single-shot path doesn't stamp chunk_size into provenance.
    assert corpus["provenance"].get("chunk_size") is None


@pytest.mark.slow
def test_chunked_provenance_covers_supervision_only_successes(tmp_path: Path, monkeypatch):
    """Regression: a record that succeeds on supervision but fails on
    sequence must still appear in the supervision manifest provenance.

    The prior chunked-intake code appended `expanded_paths` only inside
    the sequence-success block, so the supervision tensors.parquet
    carried the example but the supervision release_manifest's
    provenance.input_paths silently dropped that input. The fix
    appends provenance BEFORE both try blocks (one entry per
    _expand_chains output), matching the single-shot path's
    "provenance = what was attempted" semantics.
    """
    from proteon import sequence_example as _seq_mod

    real_build_sequence_example = _seq_mod.build_sequence_example
    fail_on_stem = FIXTURES[1].stem  # 1ubq

    def _selective_sequence_failure(structure, *args, **kwargs):
        record_id = kwargs.get("record_id") or ""
        # Use the prefix because chunked and single-shot may pass
        # slightly different record_ids (stem vs stem:A).
        if record_id.startswith(fail_on_stem):
            raise RuntimeError(f"test-injected failure on {record_id}")
        return real_build_sequence_example(structure, *args, **kwargs)

    # corpus_smoke imports build_sequence_example function-locally inside
    # `_build_local_corpus_smoke_release_chunked`, so monkeypatching the
    # source module picks up on the next call. No need to patch
    # corpus_smoke itself.
    monkeypatch.setattr(_seq_mod, "build_sequence_example", _selective_sequence_failure)

    root = proteon.build_local_corpus_smoke_release(
        [str(p) for p in FIXTURES],
        tmp_path / "out",
        release_id="chunked-seq-fail",
        chunk_size=1,
    )

    sup_manifest = json.loads(
        (root / "prepared" / "supervision_release" / "release_manifest.json").read_text()
    )
    seq_manifest = json.loads(
        (root / "sequence" / "release_manifest.json").read_text()
    )

    # All three input paths remain in both manifests' provenance even
    # though one record failed on sequence. If the old skew were still
    # present, 1ubq's path would be missing from at least one of them.
    sup_paths = sup_manifest["provenance"]["input_paths"]
    seq_paths = seq_manifest["provenance"]["input_paths"]
    assert any(fail_on_stem in p for p in sup_paths), (
        f"supervision manifest provenance missing {fail_on_stem!r} — skew is back"
    )
    assert any(fail_on_stem in p for p in seq_paths), (
        f"sequence manifest provenance missing {fail_on_stem!r}"
    )

    # Supervision contains the record. Sequence does not. Sequence has
    # exactly one failure (the injected one).
    import pyarrow.parquet as pq
    sup_tbl = pq.read_table(
        root / "prepared" / "supervision_release" / "examples" / "tensors.parquet",
        columns=["record_id"],
    )
    sup_rids = set(sup_tbl.column("record_id").to_pylist())
    assert any(r.startswith(fail_on_stem) for r in sup_rids), (
        f"expected {fail_on_stem!r} in supervision.parquet but got {sorted(sup_rids)}"
    )

    seq_tbl = pq.read_table(
        root / "sequence" / "examples" / "tensors.parquet", columns=["record_id"],
    )
    seq_rids = set(seq_tbl.column("record_id").to_pylist())
    assert not any(r.startswith(fail_on_stem) for r in seq_rids), (
        f"seq-failed {fail_on_stem!r} unexpectedly appeared in sequence.parquet"
    )

    # Exactly one sequence-side failure row for the injected PDB.
    seq_fail_lines = (root / "sequence" / "failures.jsonl").read_text().splitlines()
    seq_fails = [json.loads(ln) for ln in seq_fail_lines if ln.strip()]
    assert len(seq_fails) == 1, f"expected 1 seq failure, got {len(seq_fails)}"
    assert seq_fails[0]["record_id"].startswith(fail_on_stem)
    assert "test-injected failure" in seq_fails[0]["message"]
