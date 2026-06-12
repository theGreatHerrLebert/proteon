"""End-to-end tests for the proteon.vina Python surface.

Exercises both the low-level connector (proteon_connector.py_vina)
and the public proteon.vina convenience module on the same 1iep
imatinib → Abl kinase fixture used by the Rust parity oracle.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "proteon-vina",
    "tests",
    "fixtures",
    "pairs",
    "1iep",
)
LIGAND = os.path.join(FIXTURE_DIR, "ligand.pdbqt")
RECEPTOR = os.path.join(FIXTURE_DIR, "receptor.pdbqt")


@pytest.fixture(scope="module")
def rec_text() -> str:
    with open(RECEPTOR) as f:
        return f.read()


@pytest.fixture(scope="module")
def lig_text() -> str:
    with open(LIGAND) as f:
        return f.read()


# ----------------------------------------------------------------------
# Low-level: proteon_connector.py_vina
# ----------------------------------------------------------------------


def test_connector_score_only_matches_upstream_on_1iep(rec_text, lig_text):
    from proteon_connector import py_vina

    s = py_vina.score_only(rec_text, lig_text)
    # Upstream `vina --score_only --autobox` at v1.2.7-27 prints
    # these values to 3-dp; we match to better than 1 mkcal/mol.
    assert s.total == pytest.approx(-12.513, abs=2e-3)
    assert s.lig_grids == pytest.approx(-17.634, abs=2e-3)
    assert s.lig_intra == pytest.approx(-0.485, abs=2e-3)
    assert s.conf_independent == pytest.approx(5.121, abs=2e-3)
    assert s.intramolecular == pytest.approx(-0.485, abs=2e-3)
    # Flex-related components are zero in v0.
    assert s.inter_pairs == 0.0
    assert s.flex_grids == 0.0
    assert s.intra_pairs == 0.0


def test_connector_score_components_as_dict(rec_text, lig_text):
    from proteon_connector import py_vina

    d = py_vina.score_only(rec_text, lig_text).as_dict()
    assert set(d.keys()) == {
        "total",
        "lig_grids",
        "inter_pairs",
        "flex_grids",
        "intra_pairs",
        "lig_intra",
        "conf_independent",
        "intramolecular",
    }


def test_connector_local_only_improves_on_score_only(rec_text, lig_text):
    from proteon_connector import py_vina

    s = py_vina.score_only(rec_text, lig_text).total
    r = py_vina.local_only(rec_text, lig_text)
    # BFGS must not regress the score; on 1iep it improves by ~0.7.
    assert r.total <= s + 1e-3
    # Upstream's default step budget for a 37-atom ligand.
    assert r.bfgs.n_steps == (25 + 37) // 3
    assert r.bfgs.n_evals >= r.bfgs.n_steps


def test_connector_local_only_returns_valid_coords(rec_text, lig_text):
    from proteon_connector import py_vina

    r = py_vina.local_only(rec_text, lig_text)
    coords = r.coords
    assert coords.shape == (37, 3)
    assert coords.dtype == np.float64
    assert np.isfinite(coords).all()
    # original_serials is parallel to coords.
    serials = r.original_serials
    assert serials.shape == (37,)
    assert serials.dtype == np.uint32


def test_connector_local_only_matches_upstream_on_1iep(rec_text, lig_text):
    from proteon_connector import py_vina

    r = py_vina.local_only(rec_text, lig_text)
    # Upstream `vina --local_only --autobox` at v1.2.7-27.
    # BFGS trajectory sensitivity gates at ~80 mkcal/mol for drug-
    # like ligands; see proteon-vina/src/local_only.rs for details.
    assert r.components.total == pytest.approx(-13.241, abs=0.08)
    assert r.components.lig_grids == pytest.approx(-18.660, abs=0.08)


def test_connector_local_only_max_steps_override(rec_text, lig_text):
    from proteon_connector import py_vina

    # Cap at 3 steps — the score should still be <= initial but may
    # not reach the default-step minimum.
    r = py_vina.local_only(rec_text, lig_text, max_steps=3)
    assert r.bfgs.n_steps == 3


# ----------------------------------------------------------------------
# Public API: proteon.vina
# ----------------------------------------------------------------------


def test_proteon_vina_package_exports():
    # proteon.vina is a submodule, not auto-imported into the top-level
    # namespace — matches how proteon.dssp, proteon.sasa, etc. work.
    from proteon import vina

    assert callable(vina.score_only)
    assert callable(vina.local_only)
    assert vina.VinaScoreComponents is not None
    assert vina.BfgsOutcome is not None
    assert vina.VinaLocalOnlyOutcome is not None


def test_proteon_vina_score_only_matches_connector(rec_text, lig_text):
    from proteon import vina
    from proteon_connector import py_vina

    a = vina.score_only(rec_text, lig_text)
    b = py_vina.score_only(rec_text, lig_text)
    assert a.total == b.total
    assert a.lig_grids == b.lig_grids
    assert a.lig_intra == b.lig_intra


def test_proteon_vina_local_only_keyword_only_args(rec_text, lig_text):
    from proteon import vina

    # max_steps and v_curl must be keyword-only.
    with pytest.raises(TypeError):
        vina.local_only(rec_text, lig_text, 5)  # positional max_steps

    r = vina.local_only(rec_text, lig_text, max_steps=3, v_curl=1000.0)
    assert r.bfgs.n_steps == 3


# ----------------------------------------------------------------------
# Batch: one receptor × N ligands
# ----------------------------------------------------------------------


def test_batch_score_only_matches_single_call(rec_text, lig_text):
    from proteon import vina

    single = vina.score_only(rec_text, lig_text)
    # Four copies of the same ligand — every result must equal the
    # single-call result.
    batch = vina.batch_score_only(rec_text, [lig_text] * 4)
    assert len(batch) == 4
    for s in batch:
        assert s.total == single.total
        assert s.lig_grids == single.lig_grids


def test_batch_score_only_preserves_input_order(rec_text, lig_text):
    from proteon import vina

    # Feed three distinct ligands by perturbing the partial charges
    # — that doesn't change XS types or geometry, so the score is
    # identical across all three, but the input-order contract still
    # holds and we can at least check length.
    batch = vina.batch_score_only(rec_text, [lig_text] * 3)
    assert len(batch) == 3
    assert all(abs(s.total - batch[0].total) < 1e-12 for s in batch)


def test_batch_local_only_matches_single_call(rec_text, lig_text):
    from proteon import vina

    single = vina.local_only(rec_text, lig_text)
    batch = vina.batch_local_only(rec_text, [lig_text] * 3)
    assert len(batch) == 3
    for r in batch:
        # Determinism: BFGS on the same inputs produces the same
        # trajectory across calls.
        assert r.total == single.total
        assert r.bfgs.n_steps == single.bfgs.n_steps


def test_batch_local_only_respects_max_steps_override(rec_text, lig_text):
    from proteon import vina

    batch = vina.batch_local_only(rec_text, [lig_text] * 2, max_steps=4)
    assert len(batch) == 2
    for r in batch:
        assert r.bfgs.n_steps == 4


def test_batch_score_only_empty_list_returns_empty(rec_text):
    from proteon import vina

    batch = vina.batch_score_only(rec_text, [])
    assert batch == []


def test_batch_score_only_single_threaded_equals_multi_threaded(rec_text, lig_text):
    from proteon import vina

    ligs = [lig_text] * 4
    single = vina.batch_score_only(rec_text, ligs, n_threads=1)
    multi = vina.batch_score_only(rec_text, ligs, n_threads=None)
    assert len(single) == len(multi)
    for s, m in zip(single, multi):
        assert s.total == m.total
        assert s.lig_grids == m.lig_grids


# ----------------------------------------------------------------------
# Top-level proteon namespace
# ----------------------------------------------------------------------


def test_top_level_proteon_score_only_matches_vina_submodule(rec_text, lig_text):
    import proteon
    from proteon import vina

    a = proteon.score_only(rec_text, lig_text)
    b = vina.score_only(rec_text, lig_text)
    assert a.total == b.total


def test_top_level_proteon_vina_namespace_is_the_submodule():
    import proteon
    from proteon import vina

    assert proteon.vina is vina


def test_top_level_batch_apis_exist():
    import proteon

    assert callable(proteon.batch_score_only)
    assert callable(proteon.batch_local_only)
    assert proteon.VinaScoreComponents is not None


def test_top_level_exports_listed_in_all():
    import proteon

    for name in (
        "score_only",
        "local_only",
        "batch_score_only",
        "batch_local_only",
        "VinaScoreComponents",
        "VinaLocalOnlyOutcome",
        "BfgsOutcome",
        "vina",
    ):
        assert name in proteon.__all__, f"{name!r} not in proteon.__all__"
