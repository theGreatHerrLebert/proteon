"""End-to-end: GPU MSA search engine → SequenceExample MSA fields.

After the 4.5a/b warp-collab SW landed, the search engine can
produce AF2-style MSA tensor bundles per query at GPU speed. This
test wires it through the Layer-5 sequence-example builder:

  MsaSearch over protein corpus
    → batch_build_sequence_examples_with_msa(structures, engine)
    → SequenceExample with populated msa / deletion_matrix / msa_mask

Plus the release-builder integration: `build_sequence_dataset(...,
msa_engine=...)` routes through the engine-aware path when the
engine is provided, still falling back to the explicit-MSA path
when callers pass pre-computed `msas` instead.

Uses the repo's real PDB fixtures (crambin, ubiquitin, BPTI). Skips
when the Rust connector isn't available — this path requires both
`proteon.io._io` (for batch_load_tolerant) and the
`proteon_connector.py_msa.SearchEngine` (for the search itself).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import proteon
from proteon import io as _proteon_io

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = {
    "1crn": REPO_ROOT / "test-pdbs" / "1crn.pdb",
    "1ubq": REPO_ROOT / "test-pdbs" / "1ubq.pdb",
    "1bpi": REPO_ROOT / "test-pdbs" / "1bpi.pdb",
}


def _msa_backend_available() -> bool:
    try:
        from proteon import rust_msa_available
    except ImportError:
        return False
    return rust_msa_available()


pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in FIXTURES.values())
    or _proteon_io._io is None
    or not _msa_backend_available(),
    reason=(
        "MSA-engine wiring test requires the Rust connector (py_io + py_msa) "
        "and the crambin / ubiquitin / BPTI fixtures. Skipping in source-only envs."
    ),
)


def _load_all():
    paths = [str(FIXTURES[k]) for k in ("1crn", "1ubq", "1bpi")]
    pairs = proteon.batch_load_tolerant(paths)
    assert len(pairs) == 3, "failed to load all three fixtures"
    return [pair[1] for pair in pairs]


def _build_engine(structures):
    """Build an MsaSearch over the three protein chain sequences."""
    from proteon.supervision_constants import residue_to_one_letter

    targets = []
    for i, structure in enumerate(structures):
        chain = structure.chains[0]
        seq = "".join(
            residue_to_one_letter(r.name) for r in chain.residues if r.is_amino_acid
        )
        targets.append((i, seq))

    # reduce_to=None + small k keeps the hashmap-based index happy
    # for tiny corpora. In production, reduce_to=13 + k=6 is the
    # MMseqs2-compatible default.
    return proteon.MsaSearch.build(
        targets, k=3, reduce_to=None, min_score=0, max_results=10
    )


class TestBatchEngineIntegration:
    def test_batch_build_sequence_examples_with_msa_populates_msa_tensors(self):
        structures = _load_all()
        engine = _build_engine(structures)
        examples = proteon.batch_build_sequence_examples_with_msa(
            structures, engine
        )
        assert len(examples) == 3

        for i, (ex, identifier) in enumerate(zip(examples, ["1crn", "1ubq", "1bpi"])):
            assert ex.msa is not None, f"{identifier}: msa not populated"
            assert ex.deletion_matrix is not None, f"{identifier}: deletion_matrix not populated"
            assert ex.msa_mask is not None, f"{identifier}: msa_mask not populated"

            # MSA rows must have the query length on the column axis.
            assert ex.msa.shape[1] == ex.length, (
                f"{identifier}: msa col axis {ex.msa.shape[1]} != query length {ex.length}"
            )
            # Matching-shape deletion matrix + msa mask — AF2 contract.
            assert ex.deletion_matrix.shape == ex.msa.shape
            assert ex.msa_mask.shape == ex.msa.shape

            # The query row (row 0) encodes the query itself. Must match
            # the structure-derived aatype in valid positions — this is
            # the strongest sanity check that the engine and the
            # sequence-example builder agreed on the query identity.
            first_row = ex.msa[0]
            # Valid positions where the MSA mask is 1 must match aatype.
            valid = ex.msa_mask[0].astype(bool)
            np.testing.assert_array_equal(
                first_row[valid], ex.aatype[valid],
                err_msg=f"{identifier}: MSA row 0 (query) disagrees with aatype",
            )


class TestReleaseBuilderWithEngine:
    def test_build_sequence_dataset_uses_engine_when_provided(self, tmp_path):
        """`build_sequence_dataset(..., msa_engine=engine)` must route
        through the engine path. Check that the exported release's
        tensors carry non-empty MSA (shape[0] > 0) even though the
        caller didn't pass any explicit `msas`."""
        structures = _load_all()
        engine = _build_engine(structures)

        out_dir = proteon.build_sequence_dataset(
            structures,
            tmp_path / "seq_engine",
            release_id="msa-engine-v0",
            record_ids=["1crn:A", "1ubq:A", "1bpi:A"],
            source_ids=["1crn", "1ubq", "1bpi"],
            msa_engine=engine,
            overwrite=True,
        )
        assert (out_dir / "release_manifest.json").exists()

        loaded = proteon.load_sequence_examples(out_dir / "examples")
        assert len(loaded) == 3
        for ex in loaded:
            assert ex.msa is not None and ex.msa.shape[0] >= 1, (
                f"release-level MSA empty for {ex.record_id}"
            )
            assert ex.deletion_matrix is not None
            assert ex.msa_mask is not None

    def test_explicit_msas_still_win_when_engine_absent(self, tmp_path):
        """Backwards-compat: without `msa_engine`, explicit `msas`
        still route through the plain builder. Pins the old path
        against accidental regression."""
        structures = _load_all()
        explicit_msa = [
            ["TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"],
            ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"],
            ["RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"],
        ]
        out_dir = proteon.build_sequence_dataset(
            structures,
            tmp_path / "seq_explicit",
            release_id="explicit-v0",
            record_ids=["1crn:A", "1ubq:A", "1bpi:A"],
            source_ids=["1crn", "1ubq", "1bpi"],
            msas=explicit_msa,
            overwrite=True,
        )
        loaded = proteon.load_sequence_examples(out_dir / "examples")
        assert all(ex.msa is not None and ex.msa.shape[0] == 1 for ex in loaded)
