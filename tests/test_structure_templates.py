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


def test_confidence_is_query_normalized_not_template_normalized():
    """`template_sum_probs` must be the *query*-length-normalized TM-score.
    proteon's `tm_score_chain1` is normalized by chain2 (the template) and
    `tm_score_chain2` by chain1 (the query) — inverted names (core/types.rs).
    For an asymmetric pair (query 1crn=46 vs template 1ubq=76) the two diverge,
    so this pins the right one (codex catch)."""
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    res = proteon.tm_align(q, u)  # query=chain1, template=chain2
    # Sanity: the two normalizations genuinely differ for this pair.
    assert abs(res.tm_score_chain1 - res.tm_score_chain2) > 0.05
    tf = build_structure_template_features(q, [u], top_k=1)
    assert tf.template_sum_probs[0] == pytest.approx(res.tm_score_chain2, abs=1e-4)
    assert tf.template_sum_probs[0] != pytest.approx(res.tm_score_chain1, abs=1e-4)


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


def test_self_template_geometry_matches_supervision():
    from proteon.supervision import build_structure_supervision_example

    q = _load("1crn.pdb")
    tf = build_structure_template_features(q, [q])
    ex = build_structure_supervision_example(q)
    # The self-template's derived geometry must equal the query's own supervision
    # geometry (same atom37, identity correspondence).
    np.testing.assert_allclose(tf.template_pseudo_beta[0], ex.pseudo_beta)
    np.testing.assert_array_equal(tf.template_pseudo_beta_mask[0], ex.pseudo_beta_mask)
    np.testing.assert_allclose(
        tf.template_torsion_angles_sin_cos[0], ex.torsion_angles_sin_cos, atol=1e-6
    )
    np.testing.assert_allclose(
        tf.template_alt_torsion_angles_sin_cos[0],
        ex.alt_torsion_angles_sin_cos,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        tf.template_torsion_angles_mask[0], ex.torsion_angles_mask
    )


def test_template_insertion_masks_backbone_torsions():
    """Query-adjacent rows mapping to nonconsecutive *template* residues (a template
    insertion) must NOT form a peptide bond — pre_omega/phi mask there, even with
    atoms present. The continuity reuse (codex catch)."""
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    qres = _amino_acid_residues(q, None)
    ures = _amino_acid_residues(u, None)
    corr = structural_correspondence(proteon.tm_align(q, u), qres, ures)
    qi, ti = corr.query_idx, corr.template_idx
    insertions = [
        int(qi[k])
        for k in range(1, len(qi))
        if qi[k] == qi[k - 1] + 1 and ti[k] != ti[k - 1] + 1
    ]
    assert insertions, "expected at least one template insertion in 1crn vs 1ubq"

    tf = build_structure_template_features(q, [u], top_k=1)
    tmask = tf.template_torsion_angles_mask[0]
    for row in insertions:
        # pre_omega (0) and phi (1) depend on the previous residue → masked at a break.
        assert tmask[row, 0] == 0.0 and tmask[row, 1] == 0.0, row


def _assert_features_equal(a, b):
    assert a.n_templates == b.n_templates
    np.testing.assert_array_equal(a.template_aatype, b.template_aatype)
    np.testing.assert_array_equal(
        a.template_all_atom_positions, b.template_all_atom_positions
    )
    np.testing.assert_array_equal(a.template_all_atom_masks, b.template_all_atom_masks)
    np.testing.assert_array_equal(a.template_sum_probs, b.template_sum_probs)


def test_n_threads_is_deterministic():
    """The parallel pool (`tm_align_one_to_many`) must be order- and
    thread-count-independent: more threads, identical features."""
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    one = build_structure_template_features(q, [u, q, u], top_k=3, n_threads=1)
    many = build_structure_template_features(q, [u, q, u], top_k=3, n_threads=4)
    _assert_features_equal(one, many)


def test_parallel_and_serial_paths_agree():
    """`_batch_align`'s parallel branch (uniform `None` chains) and its serial
    branch (explicit per-candidate chains) must produce identical features for
    single-chain fixtures (chain 'A')."""
    q = _load("1crn.pdb")
    u = _load("1ubq.pdb")
    parallel = build_structure_template_features(q, [u, q], top_k=2)
    serial = build_structure_template_features(
        q, [u, q], top_k=2, query_chain="A", candidate_chains=["A", "A"]
    )
    _assert_features_equal(parallel, serial)


class _StubAlign:
    """Minimal AlignResult stand-in for the correspondence-contract tests."""

    def __init__(self, ax, ay, tm=0.5, n=0):
        self.aligned_seq_x = ax
        self.aligned_seq_y = ay
        # Confidence reads tm_score_chain2 (query-length-normalized); chain1 is a
        # decoy so a regression back to chain1 fails the contract test.
        self.tm_score_chain2 = tm
        self.tm_score_chain1 = -1.0
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


def test_skipped_candidate_warns_not_silent():
    q = _load("1crn.pdb")
    # A candidate with a non-existent chain can't be extracted → skipped, but the
    # skip must be visible (codex: silent drops are undiagnosable).
    with pytest.warns(UserWarning, match="skipped"):
        tf = build_structure_template_features(q, [q], candidate_chains=["ZZ"])
    assert tf.n_templates == 0


def test_multichain_query_requires_explicit_chain():
    class _FakeChain:
        id = "A"
        residues = []

    class _FakeStruct:
        chains = [_FakeChain(), _FakeChain()]

    with pytest.raises(ValueError):
        _amino_acid_residues(_FakeStruct(), None)
