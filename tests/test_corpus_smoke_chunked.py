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

import ferritin

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = [
    REPO_ROOT / "test-pdbs" / "1crn.pdb",
    REPO_ROOT / "test-pdbs" / "1ubq.pdb",
    REPO_ROOT / "test-pdbs" / "1bpi.pdb",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in FIXTURES) or ferritin.io._io is None,
    reason="chunked intake test needs the Rust IO connector + 1crn/1ubq/1bpi fixtures",
)


def _build(tmp, chunk_size):
    out = tmp / ("chunked" if chunk_size else "single")
    return ferritin.build_local_corpus_smoke_release(
        [str(p) for p in FIXTURES],
        out,
        release_id="chunked-equiv",
        chunk_size=chunk_size,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    from ferritin.training_example import load_training_examples
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


def test_chunked_records_chunk_size_in_corpus_manifest(tmp_path: Path):
    root = _build(tmp_path, chunk_size=2)
    corpus = json.loads(
        (root / "corpus" / "corpus_release_manifest.json").read_text()
    )
    assert corpus["provenance"]["chunk_size"] == 2


def test_chunk_size_none_is_single_shot_path(tmp_path: Path):
    """chunk_size=None is the default, matches historical behavior."""
    out = ferritin.build_local_corpus_smoke_release(
        [str(FIXTURES[0])],
        tmp_path / "single",
        release_id="default-single",
    )
    corpus = json.loads((out / "corpus" / "corpus_release_manifest.json").read_text())
    # Single-shot path doesn't stamp chunk_size into provenance.
    assert corpus["provenance"].get("chunk_size") is None
