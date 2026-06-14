"""OpenFold parity oracle for the MSA feature transforms.

Feeds *identical* OpenFold-encoded state (same clustered/extra MSA, masks,
deletion, cluster assignment) to proteon's deterministic transforms and
OpenFold's `data_transforms`, and asserts they agree: integer-exact for the
cluster assignment, float-`allclose` for profiles / deletion means / `msa_feat`.
Stochastic steps are not compared here (RNG differs) — the invariants live in
test_msa_features.py.

Runs only where torch + openfold are importable (the oracle venv); skips in CI.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

# OpenFold checkout (sibling read-only repo), if present.
_OPENFOLD = "/scratch/TMAlign/openfold"
if os.path.isdir(_OPENFOLD) and _OPENFOLD not in sys.path:
    sys.path.insert(0, _OPENFOLD)

torch = pytest.importorskip("torch")
dt = pytest.importorskip("openfold.data.data_transforms")


def _load_msa_features():
    """Import proteon.msa_features (pure NumPy) without running proteon/__init__
    (the oracle venv has a stale connector that makes `import proteon` fail)."""
    try:
        import proteon.msa_features as m  # working-connector envs
        return m
    except Exception:
        src = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "packages", "proteon", "src", "proteon")
        )
        if "proteon" not in sys.modules:
            pkg = types.ModuleType("proteon")
            pkg.__path__ = [src]
            sys.modules["proteon"] = pkg
        spec = importlib.util.spec_from_file_location(
            "proteon.msa_features", os.path.join(src, "msa_features.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["proteon.msa_features"] = mod
        spec.loader.exec_module(mod)
        return mod


mf = _load_msa_features()


def _synthetic(n_clust=4, n_extra=9, length=11, seed=0):
    """OpenFold-encoded clustered + extra MSA (tokens 0-22, full masks)."""
    rng = np.random.default_rng(seed)
    msa = rng.integers(0, 22, (n_clust, length)).astype(np.int64)
    extra_msa = rng.integers(0, 22, (n_extra, length)).astype(np.int64)
    deletion = rng.integers(0, 6, (n_clust, length)).astype(np.float32)
    extra_deletion = rng.integers(0, 6, (n_extra, length)).astype(np.float32)
    msa_mask = np.ones((n_clust, length), np.float32)
    extra_mask = np.ones((n_extra, length), np.float32)
    aatype = rng.integers(0, 21, (length,)).astype(np.int64)
    return dict(
        msa=msa, extra_msa=extra_msa, deletion=deletion, extra_deletion=extra_deletion,
        msa_mask=msa_mask, extra_mask=extra_mask, aatype=aatype,
    )


def _of_protein(s):
    return {
        "msa": torch.tensor(s["msa"], dtype=torch.long),
        "extra_msa": torch.tensor(s["extra_msa"], dtype=torch.long),
        "deletion_matrix": torch.tensor(s["deletion"], dtype=torch.float32),
        "extra_deletion_matrix": torch.tensor(s["extra_deletion"], dtype=torch.float32),
        "msa_mask": torch.tensor(s["msa_mask"], dtype=torch.float32),
        "extra_msa_mask": torch.tensor(s["extra_mask"], dtype=torch.float32),
        "aatype": torch.tensor(s["aatype"], dtype=torch.long),
        "between_segment_residues": torch.zeros(s["msa"].shape[1], dtype=torch.long),
    }


def test_nearest_neighbor_clusters_matches_openfold():
    s = _synthetic(seed=1)
    of = dt.nearest_neighbor_clusters()(_of_protein(s))
    of_assign = of["extra_cluster_assignment"].cpu().numpy()
    pt_assign = mf.nearest_neighbor_clusters(
        s["msa"], s["msa_mask"], s["extra_msa"], s["extra_mask"]
    )
    np.testing.assert_array_equal(pt_assign, of_assign)  # integer-exact


def test_summarize_clusters_matches_openfold():
    s = _synthetic(seed=2)
    prot = _of_protein(s)
    prot = dt.nearest_neighbor_clusters()(prot)
    assign = prot["extra_cluster_assignment"].cpu().numpy()
    prot = dt.summarize_clusters()(prot)
    of_profile = prot["cluster_profile"].cpu().numpy()
    of_del_mean = prot["cluster_deletion_mean"].cpu().numpy()

    pt_profile, pt_del_mean = mf.summarize_clusters(
        s["msa"], s["msa_mask"], s["deletion"],
        s["extra_msa"], s["extra_mask"], s["extra_deletion"], assign,
    )
    np.testing.assert_allclose(pt_profile, of_profile, atol=1e-5)
    np.testing.assert_allclose(pt_del_mean, of_del_mean, atol=1e-5)


def test_make_msa_feat_matches_openfold():
    s = _synthetic(seed=3)
    prot = _of_protein(s)
    prot = dt.nearest_neighbor_clusters()(prot)
    prot = dt.summarize_clusters()(prot)
    cp = prot["cluster_profile"].cpu().numpy()
    cdm = prot["cluster_deletion_mean"].cpu().numpy()
    prot = dt.make_msa_feat()(prot)
    of_msa_feat = prot["msa_feat"].cpu().numpy()
    of_target_feat = prot["target_feat"].cpu().numpy()

    pt_msa_feat, pt_target_feat = mf.make_msa_feat(
        s["msa"], s["deletion"], cp, cdm, s["aatype"],
        between_segment_residues=np.zeros(s["msa"].shape[1], np.float32),
    )
    assert pt_msa_feat.shape == of_msa_feat.shape  # both (N, L, 49)
    np.testing.assert_allclose(pt_msa_feat, of_msa_feat, atol=1e-5)
    np.testing.assert_allclose(pt_target_feat, of_target_feat, atol=1e-5)


def test_empty_extra_summarize_matches_openfold():
    s = _synthetic(n_extra=0, seed=4)
    prot = _of_protein(s)
    prot = dt.nearest_neighbor_clusters()(prot)
    assign = prot["extra_cluster_assignment"].cpu().numpy()
    prot = dt.summarize_clusters()(prot)
    pt_profile, pt_del_mean = mf.summarize_clusters(
        s["msa"], s["msa_mask"], s["deletion"],
        s["extra_msa"], s["extra_mask"], s["extra_deletion"], assign,
    )
    np.testing.assert_allclose(pt_profile, prot["cluster_profile"].cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(pt_del_mean, prot["cluster_deletion_mean"].cpu().numpy(), atol=1e-5)
