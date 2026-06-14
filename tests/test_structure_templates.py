"""Structure-based template features (TM-align correspondence).

No external bit-oracle for the structural path, so these gate the correspondence
contract + the featurizer by invariants: self-template identity, explicit-index
safety, raw (non-renormalized) sum_probs, unaligned-row masking, top-k. Pure
proteon (no torch/openfold), so it runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import proteon
    from proteon.supervision_constants import AA_TO_INDEX, residue_to_one_letter
    from proteon.supervision_geometry import extract_atom37
    from proteon.structure_templates import (
        _amino_acid_residues,
        build_structure_template_features,
        structural_correspondence,
    )
    from proteon.templates import TEMPLATE_GAP_INDEX
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"proteon unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(pdb: str):
    path = REPO_ROOT / "test-pdbs" / pdb
    if not path.exists():
        pytest.skip(f"{path} not found")
    pairs = proteon.batch_load_tolerant([str(path)])
    return pairs[0][1] if isinstance(pairs[0], tuple) else pairs[0]


def _query_aatype(residues):
    return np.array(
        [AA_TO_INDEX.get(residue_to_one_letter(r.name), AA_TO_INDEX["X"]) for r in residues],
        dtype=np.int32,
    )


def test_self_template_is_identity():
    q = _load("1crn.pdb")
    tf = build_structure_template_features(q, [q])
    assert tf.n_templates == 1
    qres = _amino_acid_residues(q, None)
    q37 = extract_atom37(qres)
    assert tf.query_len == len(qres)
    assert tf.template_all_atom_positions.shape == (1, len(qres), 37, 3)
    np.testing.assert_allclose(tf.template_all_atom_positions[0], q37["positions"])
    np.testing.assert_array_equal(tf.template_all_atom_masks[0], q37["mask"])
    np.testing.assert_array_equal(tf.template_aatype[0], _query_aatype(qres))
    assert tf.template_sum_probs[0] == pytest.approx(1.0, abs=1e-3)


def test_cross_template_sorts_and_keeps_raw_tm_score():
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    tf = build_structure_template_features(q, [u, q], top_k=2)
    assert tf.n_templates == 2
    # Sorted descending by TM-score; self (≈1) ranks above the weak 1ubq hit.
    assert tf.template_sum_probs[0] >= tf.template_sum_probs[1]
    assert tf.template_sum_probs[0] == pytest.approx(1.0, abs=1e-3)
    # Raw score, NOT per-set max-normalized — the weak hit stays well below 1.
    assert 0.0 < tf.template_sum_probs[1] < 0.9
    # The 1ubq template aligns *some* but not all query rows (partial coverage).
    n_aligned = int((tf.template_all_atom_masks[1].sum(axis=-1) > 0).sum())
    assert 0 < n_aligned < tf.query_len


def test_unaligned_rows_are_zero_masked_gap():
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    tf = build_structure_template_features(q, [u], top_k=1)
    covered = tf.template_all_atom_masks[0].sum(axis=-1) > 0
    unaligned = ~covered
    assert unaligned.any()
    assert np.all(tf.template_all_atom_positions[0][unaligned] == 0.0)
    assert np.all(tf.template_aatype[0][unaligned] == TEMPLATE_GAP_INDEX)


def test_top_k_caps_templates():
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    tf = build_structure_template_features(q, [q, u, q, u], top_k=2)
    assert tf.n_templates == 2


def test_no_candidates_is_empty():
    q = _load("1crn.pdb")
    tf = build_structure_template_features(q, [])
    assert tf.n_templates == 0
    assert tf.template_all_atom_positions.shape[0] == 0
    assert tf.template_sum_probs.shape == (0,)


class _StubAlign:
    """Minimal AlignResult stand-in for the correspondence-contract tests."""

    def __init__(self, ax, ay, tm=0.5, n=0):
        self.aligned_seq_x = ax
        self.aligned_seq_y = ay
        self.tm_score_chain1 = tm
        self.n_aligned = n


def test_correspondence_rejects_unsafe_index_map():
    q = _load("1crn.pdb")
    qres = _amino_acid_residues(q, None)
    # An aligned string whose ungapped residues don't match the CA residues must
    # be rejected (the column→atom37-index map would be unsafe).
    bogus = "Z" * len(qres)
    with pytest.raises(ValueError):
        structural_correspondence(_StubAlign(bogus, bogus), qres, qres)
    # Mismatched aligned-string lengths are rejected too.
    with pytest.raises(ValueError):
        structural_correspondence(_StubAlign("AAAA", "AAA"), qres, qres)


def test_multichain_query_requires_explicit_chain():
    class _FakeChain:
        id = "A"
        residues = []

    class _FakeStruct:
        chains = [_FakeChain(), _FakeChain()]

    with pytest.raises(ValueError):
        _amino_acid_residues(_FakeStruct(), None)
