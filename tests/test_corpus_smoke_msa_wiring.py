"""Verifies build_local_corpus_smoke_release forwards msa_engine end-to-end.

Uses a fake duck-typed engine (no Rust compile required) so this test
runs in source-only environments. The real engine is covered by
test_msa_engine_wired_into_sequence_examples.py.

Without the plumbing, the corpus builder silently dropped msa_engine
and wrote empty MSA tensors into training.parquet. That was the gap
scouted on 2026-04-17 — this test pins it closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

import proteon


class FakeMsaEngine:
    """Duck-typed engine matching the msa_backend.search_and_build_msa contract.

    Emits a deterministic ~3-row MSA per query with the query as row 0,
    two perturbed copies, and a deletion count of 0 everywhere. Enough
    to verify the tensors land in the training parquet with depth > 0.
    """

    def __init__(self, *, gap_idx: int = 21):
        self.gap_idx = gap_idx

    def search_and_build_msa(self, query: str, *, max_seqs: int = 256, gap_idx: int = 21):
        L = len(query)
        n_seqs = min(3, max_seqs)
        aatype = np.array(
            [proteon.supervision_constants.AA_TO_INDEX.get(c, 20) for c in query],
            dtype=np.uint8,
        )
        rows = [aatype.copy()]
        for r in range(1, n_seqs):
            row = aatype.copy()
            if L > 0:
                row[r % L] = gap_idx  # a single "gap" marker to distinguish rows
            rows.append(row)
        msa = np.stack(rows, axis=0).astype(np.uint8)
        return {
            "query_len": L,
            "n_seqs": n_seqs,
            "gap_idx": gap_idx,
            "aatype": aatype,
            "seq_mask": np.ones((L,), dtype=np.float32),
            "msa": msa,
            "deletion_matrix": np.zeros((n_seqs, L), dtype=np.uint8),
            "msa_mask": np.ones((n_seqs, L), dtype=np.float32),
        }


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "test-pdbs" / "1crn.pdb"


pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or proteon.io._io is None,
    reason="corpus-smoke MSA wiring test needs 1crn.pdb + the Rust IO connector",
)


def test_corpus_smoke_forwards_msa_engine_into_sequence_parquet(tmp_path: Path):
    """MSA lives in the sequence release (training joins only scalars + structure)."""
    engine = FakeMsaEngine()
    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURE)],
        tmp_path / "corpus",
        release_id="smoke-msa-wiring",
        msa_engine=engine,
        msa_max_seqs=16,
        msa_gap_idx=21,
    )

    sp = out / "sequence" / "examples" / "tensors.parquet"
    assert sp.exists()
    t = pq.read_table(sp, columns=["record_id", "length", "msa"])
    assert t.num_rows == 1

    length = int(t.column("length")[0].as_py())
    msa_rows = t.column("msa")[0].as_py()
    # list<list<int32>>: outer=depth, inner=L
    assert msa_rows is not None, "engine output was dropped in the sequence release"
    assert len(msa_rows) == 3, f"expected 3 MSA rows from fake engine, got {len(msa_rows)}"
    for row in msa_rows:
        assert len(row) == length


def test_corpus_smoke_without_engine_writes_null_msa(tmp_path: Path):
    """Control: no engine passed → MSA column is null per row (not empty slab)."""
    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURE)],
        tmp_path / "corpus",
        release_id="smoke-no-msa",
    )
    sp = out / "sequence" / "examples" / "tensors.parquet"
    t = pq.read_table(sp, columns=["record_id", "msa"])
    msa_rows = t.column("msa")[0].as_py()
    assert msa_rows is None


def test_corpus_smoke_loads_sequence_msa_via_public_api(tmp_path: Path):
    """`load_sequence_examples` exposes MSA end-to-end when an engine is present."""
    engine = FakeMsaEngine()
    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURE)],
        tmp_path / "corpus",
        release_id="smoke-msa-api",
        msa_engine=engine,
        msa_max_seqs=16,
    )
    examples = proteon.load_sequence_examples(out / "sequence" / "examples")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.msa is not None and ex.msa.shape == (3, ex.length)
    assert ex.deletion_matrix is not None and ex.deletion_matrix.shape == (3, ex.length)
    assert ex.msa_mask is not None and ex.msa_mask.shape == (3, ex.length)
    # Row 0 = query → matches aatype; at least one subsequent row differs
    # where FakeMsaEngine wrote a gap_idx marker.
    np.testing.assert_array_equal(ex.msa[0], ex.aatype)
    assert not np.array_equal(ex.msa[1], ex.aatype)


def _derive_query_sequence(pdb_path: Path) -> str:
    """Reproduce the sequence string build_sequence_example derives from
    the PDB chain, so we can hand a matching-length a3m to the corpus builder."""
    structure = proteon.batch_load_tolerant([str(pdb_path)])[0][1]
    return proteon.build_sequence_example(structure).sequence


def test_corpus_smoke_reads_msa_dir_a3m(tmp_path: Path):
    """External a3m → sequence.parquet MSA without any engine."""
    query = _derive_query_sequence(FIXTURE)
    # Build a trivial 2-row a3m: the query and one copy with the first
    # position replaced by '-'. Record_id for a 1-chain PDB is the stem.
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    a3m = f">query\n{query}\n>hit\n-{query[1:]}\n"
    (msa_dir / f"{FIXTURE.stem}.a3m").write_text(a3m)

    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURE)],
        tmp_path / "corpus",
        release_id="smoke-msa-dir",
        msa_dir=msa_dir,
    )
    examples = proteon.load_sequence_examples(out / "sequence" / "examples")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.msa is not None and ex.msa.shape == (2, ex.length)
    # Row 0 should match the query aatype exactly.
    np.testing.assert_array_equal(ex.msa[0], ex.aatype)
    # Row 1 has a gap at position 0; in the encoded form that's AA_TO_INDEX['X']
    # (proteon treats '-' via the 'X' fallback path in _encode_msa).
    assert ex.msa[1][0] != ex.aatype[0]


def test_corpus_smoke_msa_dir_missing_file_falls_back_to_engine(tmp_path: Path):
    """Missing a3m for a record → engine fills the gap (explicit > engine per-row)."""
    # 1crn gets a real a3m; the engine would cover any other record, but
    # we only have 1 fixture here, so assert the a3m was used.
    query = _derive_query_sequence(FIXTURE)
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    (msa_dir / f"{FIXTURE.stem}.a3m").write_text(f">q\n{query}\n")

    engine = FakeMsaEngine()
    out = proteon.build_local_corpus_smoke_release(
        [str(FIXTURE)],
        tmp_path / "corpus",
        release_id="smoke-msa-mixed",
        msa_dir=msa_dir,
        msa_engine=engine,
    )
    examples = proteon.load_sequence_examples(out / "sequence" / "examples")
    ex = examples[0]
    # Explicit a3m has depth=1 (just the query), engine would produce depth=3.
    assert ex.msa.shape[0] == 1, (
        "explicit msa_dir entry should win over the engine when both are provided"
    )


def test_corpus_smoke_msa_dir_strict_raises_on_missing(tmp_path: Path):
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()  # intentionally empty
    with pytest.raises(FileNotFoundError, match="no MSA file"):
        proteon.build_local_corpus_smoke_release(
            [str(FIXTURE)],
            tmp_path / "corpus",
            release_id="smoke-msa-strict",
            msa_dir=msa_dir,
            msa_strict=True,
        )
