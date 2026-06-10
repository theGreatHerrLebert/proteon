"""Tests for the in-process SES mesh binding (proteon_connector.py_surface)."""

import numpy as np
import pytest

py_surface = pytest.importorskip("proteon_connector").py_surface


def _tetra():
    # Four overlapping atoms -> a single watertight SES blob.
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.7, 0.0], [1.0, 0.6, 1.6]]
    )
    radii = np.full(len(coords), 1.7)
    return coords, radii


def test_ses_mesh_returns_numpy_mesh():
    coords, radii = _tetra()
    r = py_surface.ses_mesh_py(coords, radii=radii)
    assert r["vertices"].shape[1] == 3 and r["vertices"].dtype == np.float64
    assert r["triangles"].shape[1] == 3 and r["triangles"].dtype == np.uint32
    assert r["triangles"].max() < len(r["vertices"])  # indices in range
    assert r["area"] > 0 and r["watertight"] is True
    assert r["method"] in ("analytic", "analytic_perturbed", "numerical_grid")
    assert isinstance(r["perturbations"], int)


def test_elements_path_matches_explicit_radii():
    coords, _ = _tetra()
    via_el = py_surface.ses_mesh_py(coords, elements=["C", "C", "C", "C"])
    via_r = py_surface.ses_mesh_py(coords, radii=np.full(4, 1.70))  # carbon vdW
    assert via_el["triangles"].shape == via_r["triangles"].shape
    assert abs(via_el["area"] - via_r["area"]) < 1e-9


def _bad_cases():
    coords, radii = _tetra()
    nan_coords = coords.copy()
    nan_coords[1, 0] = np.nan
    neg_radii = radii.copy()
    neg_radii[1] = -1.0
    return [
        ("nan_coord", nan_coords, radii, {}),
        ("neg_radius", coords, neg_radii, {}),
        ("probe_zero", coords, radii, {"probe": 0.0}),
        ("bad_n_theta", coords, radii, {"n_theta": 2}),
        ("nan_grid", coords, radii, {"grid": float("nan")}),
    ]


@pytest.mark.parametrize("case", _bad_cases(), ids=lambda c: c[0])
def test_bad_inputs_raise_not_panic(case):
    _name, coords, radii, kwargs = case
    with pytest.raises(ValueError):
        py_surface.ses_mesh_py(coords, radii=radii, **kwargs)


def test_radii_xor_elements():
    coords, radii = _tetra()
    with pytest.raises(ValueError):  # both
        py_surface.ses_mesh_py(coords, radii=radii, elements=["C"] * 4)
    with pytest.raises(ValueError):  # neither
        py_surface.ses_mesh_py(coords)


def test_needs_two_atoms():
    with pytest.raises(ValueError):
        py_surface.ses_mesh_py(np.zeros((1, 3)), radii=np.array([1.7]))
