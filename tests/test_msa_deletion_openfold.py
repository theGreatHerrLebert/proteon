"""Oracle parity: proteon's MSA deletion features vs OpenFold's `make_msa_feat`.

`compute_msa_deletion_features` must reproduce the deterministic per-(seq, residue)
deletion features OpenFold packs into `msa_feat`:
    has_deletion   = clip(deletion_matrix, 0, 1)
    deletion_value = arctan(deletion_matrix / 3) · 2/π

Needs `torch` + a checked-out `openfold` (neither a proteon/CI dependency), so it
is skipped unless both import.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_of = pytest.importorskip("openfold.data.data_transforms")

try:
    from proteon.sequence_example import compute_msa_deletion_features
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(
        f"proteon.sequence_example unavailable: {exc}", allow_module_level=True
    )


def test_msa_deletion_features_match_openfold():
    rng = np.random.default_rng(0)
    depth, length = 6, 16
    # Raw deletion counts: a realistic mix of zeros, small, and a few large.
    d = rng.integers(0, 9, size=(depth, length)).astype(np.float32)
    d[rng.random((depth, length)) < 0.5] = 0.0
    msa = rng.integers(0, 22, size=(depth, length)).astype(np.int64)

    has_del, del_val = compute_msa_deletion_features(d)

    prot = {
        "aatype": torch.zeros(length, dtype=torch.long),
        "between_segment_residues": torch.zeros(length, dtype=torch.long),
        "msa": torch.tensor(msa),
        "deletion_matrix": torch.tensor(d, dtype=torch.float64),
    }
    out = _of.make_msa_feat()(prot)  # curried transform; mutates + returns prot
    # msa_feat = [msa_1hot(23), has_deletion(1), deletion_value(1)] → cols 23, 24.
    msa_feat = out["msa_feat"].numpy()
    assert msa_feat.shape[-1] == 25
    of_has = msa_feat[..., 23]
    of_val = msa_feat[..., 24]

    np.testing.assert_allclose(has_del, of_has, atol=1e-6)
    np.testing.assert_allclose(del_val, of_val, atol=1e-6)


def test_deletion_features_none_without_msa():
    has_del, del_val = compute_msa_deletion_features(None)
    assert has_del is None and del_val is None
