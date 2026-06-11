"""Mesh / charge file-format I/O round-trips through the Python API.

Covers the NESSie `format/` layer exposed via `proteon.read_off` / `read_pqr` /
`read_hmo` / `read_msms` / `write_off`, and the end-to-end load → solve pipeline
on the committed Na+ fixtures. Rust-side parity vs NESSie is gated separately in
`proteon-electrostatics/tests/format_parity.rs`; this guards the Python bridge
(numpy shapes/dtypes, the dict/tuple contracts, and error mapping).
"""

import os

import numpy as np
import pytest

from conftest import REPO_ROOT

import proteon

FORMAT_DIR = os.path.join(
    REPO_ROOT, "proteon-electrostatics", "tests", "fixtures", "format"
)
NA_OFF = os.path.join(FORMAT_DIR, "na.off")
NA_PQR = os.path.join(FORMAT_DIR, "na.pqr")


def test_read_off_shapes_and_first_vertex():
    v, t = proteon.read_off(NA_OFF)
    assert v.shape == (258, 3) and v.dtype == np.float64
    assert t.shape == (512, 3)
    assert np.allclose(v[0], [0.0, 0.0, 1.0049999952])
    # 0-based indices, all in range.
    assert t.min() >= 0 and t.max() < len(v)


def test_read_pqr_single_charge():
    cp, cv = proteon.read_pqr(NA_PQR)
    assert cp.shape == (1, 3)
    assert cv.shape == (1,)
    assert np.allclose(cp[0], [0.0, 0.0, 0.0])
    assert cv[0] == pytest.approx(1.0)


def test_off_round_trip(tmp_path):
    v, t = proteon.read_off(NA_OFF)
    dst = str(tmp_path / "rt.off")
    proteon.write_off(dst, v, t)
    v2, t2 = proteon.read_off(dst)
    assert np.allclose(v, v2)
    assert np.array_equal(t, t2)


def test_hmo_round_trip(tmp_path):
    src = tmp_path / "tri.hmo"
    src.write_text(
        "BEG_NODL_DATA\n3\n1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\nEND_NODL_DATA\n"
        "BEG_ELEM_DATA\n1\n1 0 0 1 2 3\nEND_ELEM_DATA\n"
        "BEG_CHARGE_DATA\n1\n1 0.3 0.3 0.0 -0.5\nEND_CHARGE_DATA\n"
    )
    d = proteon.read_hmo(str(src))
    assert d["vertices"].shape == (3, 3)
    assert d["triangles"].shape == (1, 3)
    assert np.array_equal(d["triangles"][0], [0, 1, 2])  # 1-based ids → 0-based
    assert d["charge_values"][0] == pytest.approx(-0.5)


def test_msms_round_trip(tmp_path):
    vert = tmp_path / "s.vert"
    face = tmp_path / "s.face"
    vert.write_text(
        "# h1\n# h2\n4 0 1.5 1.0\n"
        "0.0 0.0 0.0 0 0 1 1 1 2\n1.0 0.0 0.0 0 0 1 1 1 2\n"
        "0.0 1.0 0.0 0 0 1 1 1 2\n1.0 1.0 0.0 0 0 1 1 1 2\n"
    )
    face.write_text("# h1\n# h2\n2 0 1.5 1.0\n1 2 3 1 0\n2 4 3 1 0\n")
    v, t = proteon.read_msms(str(vert), str(face))
    assert v.shape == (4, 3)
    assert t.shape == (2, 3)
    assert np.array_equal(t[0], [0, 1, 2])


def test_load_then_solve_end_to_end():
    v, t = proteon.read_off(NA_OFF)
    cp, cv = proteon.read_pqr(NA_PQR)
    out = proteon.surface_potential(v, t, cp, cv)
    assert out["converged"] is True
    assert out["watertight"] is True
    # Born-like solvation of a +1 ion at r≈1.005 Å in water: strongly negative.
    assert out["rfenergy"] < 0.0
    assert out["surface_potential"].shape == (len(v),)


def test_missing_file_raises_oserror():
    with pytest.raises(OSError):
        proteon.read_off("/nonexistent/path/x.off")


def test_malformed_off_raises_valueerror(tmp_path):
    bad = tmp_path / "bad.off"
    bad.write_text("OFF\n2 0 0\n0.0 0.0\n")  # node line missing a coordinate
    with pytest.raises(ValueError):
        proteon.read_off(str(bad))
