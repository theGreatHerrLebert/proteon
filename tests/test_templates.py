"""Tests for AF2-style template features built from search hits.

Covers the v0 sequence-based template pipeline:
  `build_template_features(query_len, engine, target_supervisions,
                            query_sequence=..., max_templates=K, ...)`

Focus areas:
- shape correctness against the roadmap's template feature spec
- CIGAR walking: M/I/D ops map correctly into template slots
- self-hit exclusion via `exclude_target_ids`
- empty corpus / no-hits / target-id-missing-from-lookup degrade
  gracefully to zero-row output
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import proteon
from proteon import io as _proteon_io
from proteon.templates import (
    TEMPLATE_GAP_INDEX,
    TemplateFeatures,
    _fill_template_from_alignment,
    _parse_cigar,
)


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


class TestCigarParser:
    def test_parse_cigar_simple_match(self):
        assert _parse_cigar("10M") == [(10, "M")]

    def test_parse_cigar_mixed(self):
        assert _parse_cigar("42M2I8M") == [(42, "M"), (2, "I"), (8, "M")]

    def test_parse_cigar_with_deletions(self):
        assert _parse_cigar("5M3D10M") == [(5, "M"), (3, "D"), (10, "M")]


class TestFillTemplateFromAlignment:
    """Unit tests on the CIGAR-driven template filler without a real engine.

    Uses a synthetic target supervision-style namespace. The filler only
    reads `all_atom_positions`, `all_atom_mask`, `aatype`, `length` — so
    a minimal SimpleNamespace suffices.
    """

    @staticmethod
    def _fake_target(length=10):
        # Target aatype: 0..length-1 (distinct so we can tell them apart).
        # Coords: all ones at every atom — filler must copy them.
        return SimpleNamespace(
            length=length,
            aatype=np.arange(length, dtype=np.int32),
            all_atom_positions=np.ones((length, 37, 3), dtype=np.float32),
            all_atom_mask=np.ones((length, 37), dtype=np.float32),
        )

    def test_pure_match_copies_target_positions(self):
        """A pure-M alignment means every query position has a template
        partner. All template tensors should be populated."""
        query_len = 10
        aatype = np.full((query_len,), TEMPLATE_GAP_INDEX, dtype=np.int32)
        positions = np.zeros((query_len, 37, 3), dtype=np.float32)
        masks = np.zeros((query_len, 37), dtype=np.float32)

        _fill_template_from_alignment(
            aatype, positions, masks,
            target=self._fake_target(),
            query_start=0, target_start=0,
            cigar="10M",
            query_length=query_len,
        )
        np.testing.assert_array_equal(aatype, np.arange(10))
        np.testing.assert_array_equal(masks, np.ones((10, 37), dtype=np.float32))
        np.testing.assert_array_equal(positions, np.ones((10, 37, 3), dtype=np.float32))

    def test_query_insertion_leaves_positions_masked(self):
        """`I` op: query advances, target doesn't. Those query slots
        must stay at mask=0, aatype=GAP."""
        query_len = 8
        aatype = np.full((query_len,), TEMPLATE_GAP_INDEX, dtype=np.int32)
        positions = np.zeros((query_len, 37, 3), dtype=np.float32)
        masks = np.zeros((query_len, 37), dtype=np.float32)

        # 3M then 2I (query_pos advances by 2 with no template content)
        # then 3M (query_pos 5..7 matched to target 3..5).
        _fill_template_from_alignment(
            aatype, positions, masks,
            target=self._fake_target(),
            query_start=0, target_start=0,
            cigar="3M2I3M",
            query_length=query_len,
        )
        # Query 0..2 matched target 0..2
        np.testing.assert_array_equal(aatype[:3], np.arange(3))
        # Query 3..4 are insertions — stayed at gap.
        assert aatype[3] == TEMPLATE_GAP_INDEX
        assert aatype[4] == TEMPLATE_GAP_INDEX
        assert masks[3].sum() == 0 and masks[4].sum() == 0
        # Query 5..7 matched target 3..5.
        np.testing.assert_array_equal(aatype[5:8], np.arange(3, 6))

    def test_target_deletion_skips_target_residues(self):
        """`D` op: target advances, query doesn't. Those target
        residues shouldn't appear anywhere — the template is indexed
        by query position."""
        query_len = 6
        aatype = np.full((query_len,), TEMPLATE_GAP_INDEX, dtype=np.int32)
        positions = np.zeros((query_len, 37, 3), dtype=np.float32)
        masks = np.zeros((query_len, 37), dtype=np.float32)

        # 3M, 2D (target advances by 2, skipped), 3M.
        _fill_template_from_alignment(
            aatype, positions, masks,
            target=self._fake_target(),
            query_start=0, target_start=0,
            cigar="3M2D3M",
            query_length=query_len,
        )
        # Query 0..2 matched target 0..2.
        np.testing.assert_array_equal(aatype[:3], np.arange(3))
        # Target positions 3, 4 were skipped (2D). Query 3..5 matched
        # target 5..7 — the deleted target residues don't leak.
        np.testing.assert_array_equal(aatype[3:6], np.arange(5, 8))


class TestBuildTemplateFeaturesSynthetic:
    """End-to-end test of build_template_features with a mocked engine
    + hand-built target supervisions. No connector / fixtures needed."""

    @staticmethod
    def _fake_structure_supervision(target_id: int, length: int = 10):
        """Build a StructureSupervisionExample-like namespace sufficient
        for the template filler (reads only atom37 + aatype + length)."""
        return SimpleNamespace(
            length=length,
            aatype=np.full((length,), target_id, dtype=np.int32),
            all_atom_positions=np.full(
                (length, 37, 3), float(target_id), dtype=np.float32
            ),
            all_atom_mask=np.ones((length, 37), dtype=np.float32),
        )

    @staticmethod
    def _mock_engine(hits):
        class _FakeEngine:
            def search(self, query):
                return list(hits)
        return _FakeEngine()

    def test_shapes_match_roadmap_spec(self):
        hits = [
            {"target_id": 1, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 100},
            {"target_id": 2, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 50},
        ]
        targets = {
            1: self._fake_structure_supervision(1),
            2: self._fake_structure_supervision(2),
        }
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine(hits),
            target_supervisions=targets,
            query_sequence="A" * 10,
            max_templates=4,
        )
        assert isinstance(feats, TemplateFeatures)
        assert feats.n_templates == 2
        assert feats.query_len == 10
        assert feats.template_aatype.shape == (2, 10)
        assert feats.template_all_atom_positions.shape == (2, 10, 37, 3)
        assert feats.template_all_atom_masks.shape == (2, 10, 37)
        assert feats.template_sum_probs.shape == (2,)

    def test_max_templates_caps_the_output(self):
        hits = [
            {"target_id": i, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 100 - i}
            for i in range(10)
        ]
        targets = {i: self._fake_structure_supervision(i) for i in range(10)}
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine(hits),
            target_supervisions=targets,
            query_sequence="A" * 10,
            max_templates=4,
        )
        assert feats.n_templates == 4
        # Top-4 hits retained in order — template 0 gets hit[0] with
        # target_id=0, which has aatype=0 everywhere.
        np.testing.assert_array_equal(
            feats.template_aatype[0], np.zeros(10, dtype=np.int32)
        )

    def test_exclude_target_ids_drops_self_hits(self):
        """Self-templating collapses the prediction to an identity copy
        of the query. Exclusion is mandatory for well-formed templates."""
        hits = [
            {"target_id": 42, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 100},  # self-hit
            {"target_id": 1, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 50},
        ]
        targets = {
            42: self._fake_structure_supervision(42),
            1: self._fake_structure_supervision(1),
        }
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine(hits),
            target_supervisions=targets,
            query_sequence="A" * 10,
            max_templates=4,
            exclude_target_ids=[42],
        )
        assert feats.n_templates == 1
        # Only target_id=1 remains → aatype filled with 1s.
        np.testing.assert_array_equal(
            feats.template_aatype[0], np.ones(10, dtype=np.int32)
        )

    def test_empty_hits_returns_zero_row_bundle(self):
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine([]),
            target_supervisions={},
            query_sequence="A" * 10,
        )
        assert feats.n_templates == 0
        assert feats.template_aatype.shape == (0, 10)
        assert feats.template_all_atom_positions.shape == (0, 10, 37, 3)
        assert feats.template_sum_probs.shape == (0,)

    def test_missing_target_supervision_keeps_slot_masked(self):
        """If the engine returns a hit for a target_id that's not in
        target_supervisions, that slot must stay zero/gap — not crash."""
        hits = [
            {"target_id": 1, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 100},
            {"target_id": 99, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 50},  # 99 not in target_supervisions
        ]
        targets = {1: self._fake_structure_supervision(1)}
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine(hits),
            target_supervisions=targets,
            query_sequence="A" * 10,
        )
        assert feats.n_templates == 2
        # Slot 0 populated from target 1.
        np.testing.assert_array_equal(
            feats.template_aatype[0], np.ones(10, dtype=np.int32)
        )
        # Slot 1 stayed at gap sentinel.
        np.testing.assert_array_equal(
            feats.template_aatype[1], np.full(10, TEMPLATE_GAP_INDEX, dtype=np.int32)
        )
        assert feats.template_all_atom_masks[1].sum() == 0

    def test_sum_probs_min_max_normalized(self):
        hits = [
            {"target_id": 1, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 100},
            {"target_id": 2, "query_start": 0, "target_start": 0,
             "cigar": "10M", "score": 50},
        ]
        targets = {i: self._fake_structure_supervision(i) for i in (1, 2)}
        feats = proteon.build_template_features(
            query_length=10,
            engine=self._mock_engine(hits),
            target_supervisions=targets,
            query_sequence="A" * 10,
        )
        # Max score → 1.0 after normalization.
        assert feats.template_sum_probs[0] == pytest.approx(1.0)
        assert feats.template_sum_probs[1] == pytest.approx(0.5)


@pytest.mark.skipif(
    not all(p.exists() for p in FIXTURES.values())
    or _proteon_io._io is None
    or not _msa_backend_available(),
    reason="real-engine template test needs Rust connector + real PDB fixtures",
)
class TestBuildTemplateFeaturesReal:
    """Integration test: real engine + real structure supervisions."""

    def test_end_to_end_with_real_corpus(self):
        # Load + build supervision for crambin, ubiquitin, BPTI.
        paths = [str(FIXTURES[k]) for k in ("1crn", "1ubq", "1bpi")]
        pairs = proteon.batch_load_tolerant(paths)
        structures = [p[1] for p in pairs]
        supervisions = proteon.batch_build_structure_supervision_examples(
            structures, record_ids=["1crn:A", "1ubq:A", "1bpi:A"]
        )

        # Search engine over the same three sequences, keyed by index.
        targets_for_search = [
            (i, sup.sequence) for i, sup in enumerate(supervisions)
        ]
        engine = proteon.MsaSearch.build(
            targets_for_search, k=3, reduce_to=None, min_score=0, max_results=10
        )
        target_supervisions = {i: sup for i, sup in enumerate(supervisions)}

        # Query crambin; exclude the self-hit (target_id=0).
        crambin_sup = supervisions[0]
        feats = proteon.build_template_features(
            query_length=crambin_sup.length,
            engine=engine,
            target_supervisions=target_supervisions,
            query_sequence=crambin_sup.sequence,
            exclude_target_ids=[0],
            max_templates=4,
        )

        # Shapes consistent with the query.
        assert feats.query_len == crambin_sup.length
        assert feats.template_all_atom_positions.shape[1] == crambin_sup.length

        # Self-hit was excluded — none of the retained templates can
        # be target_id=0. Crambin is distant from ubiquitin and BPTI,
        # but the search may still return them with low scores. All
        # we strictly guarantee is: no self-template collapse.
        # When a template IS retained, its aatype at any mask-1
        # position must NOT match crambin's own aatype at EVERY
        # position (because target_id=0 was dropped).
        for t in range(feats.n_templates):
            non_gap = feats.template_aatype[t] != TEMPLATE_GAP_INDEX
            if non_gap.any():
                assert not np.array_equal(
                    feats.template_aatype[t][non_gap],
                    crambin_sup.aatype[non_gap],
                ), "self-hit leaked past exclude_target_ids"
