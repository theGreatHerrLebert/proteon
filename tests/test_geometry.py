"""Tests for geometry building blocks: Kabsch, RMSD, secondary structure, TM-score."""

import os

import numpy as np
import pytest

from proteon_connector import py_geometry, py_io

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
UBIQ = os.path.join(EXAMPLE_DIR, "1ubq.pdb")


@pytest.fixture
def coords():
    pdb = py_io.load(UBIQ)
    return pdb.coords


# ---------------------------------------------------------------------------
# Kabsch superposition
# ---------------------------------------------------------------------------


class TestKabschSuperpose:
    def test_self_superposition(self, coords):
        rmsd, rot, trans = py_geometry.kabsch_superpose(coords, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-8)

    def test_translated(self, coords):
        shifted = coords + np.array([10.0, 20.0, 30.0])
        rmsd, rot, trans = py_geometry.kabsch_superpose(shifted, coords)
        assert rmsd == pytest.approx(0.0, abs=1e-4)

    def test_returns_valid_rotation(self, coords):
        perturbed = coords + np.random.randn(*coords.shape) * 0.5
        _, rot, _ = py_geometry.kabsch_superpose(perturbed, coords)
        assert rot.shape == (3, 3)
        # Should be approximately orthogonal
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        # Determinant should be +1 (proper rotation)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_lengths(self):
        x = np.random.randn(10, 3)
        y = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            py_geometry.kabsch_superpose(x, y)

    def test_wrong_shape(self):
        x = np.random.randn(10, 2)
        y = np.random.randn(10, 2)
        with pytest.raises(ValueError):
            py_geometry.kabsch_superpose(x, y)


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------


class TestRMSD:
    def test_self_rmsd(self, coords):
        assert py_geometry.rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_after_translation(self, coords):
        shifted = coords + np.array([100.0, 0.0, 0.0])
        # After Kabsch superposition, should be ~0
        assert py_geometry.rmsd(shifted, coords) == pytest.approx(0.0, abs=1e-6)

    def test_rmsd_with_noise(self, coords):
        noisy = coords + np.random.randn(*coords.shape) * 0.1
        r = py_geometry.rmsd(coords, noisy)
        assert 0.0 < r < 1.0  # small noise → small RMSD

    def test_rmsd_no_super_self(self, coords):
        assert py_geometry.rmsd_no_super(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_rmsd_no_super_translated(self, coords):
        shifted = coords + np.array([10.0, 0.0, 0.0])
        # Without superposition, RMSD should be 10.0
        r = py_geometry.rmsd_no_super(coords, shifted)
        assert r == pytest.approx(10.0, abs=0.01)

    def test_rmsd_no_super_vs_kabsch(self, coords):
        noisy = coords + np.random.randn(*coords.shape) * 0.5
        r_super = py_geometry.rmsd(coords, noisy)
        r_no = py_geometry.rmsd_no_super(coords, noisy)
        assert r_super <= r_no  # superposition should always improve or equal


# ---------------------------------------------------------------------------
# Apply transform
# ---------------------------------------------------------------------------


class TestApplyTransform:
    def test_identity(self, coords):
        rot = np.eye(3)
        trans = np.zeros(3)
        result = py_geometry.apply_transform(coords, rot, trans)
        np.testing.assert_allclose(result, coords, atol=1e-10)

    def test_translation_only(self, coords):
        rot = np.eye(3)
        trans = np.array([1.0, 2.0, 3.0])
        result = py_geometry.apply_transform(coords, rot, trans)
        np.testing.assert_allclose(result, coords + trans, atol=1e-10)

    def test_roundtrip_with_kabsch(self, coords):
        # Apply a known transform, then recover it with Kabsch
        angle = np.pi / 6  # 30 degrees around Z
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ])
        trans = np.array([5.0, -3.0, 7.0])
        transformed = py_geometry.apply_transform(coords, rot, trans)
        # Kabsch should recover ~0 RMSD
        r = py_geometry.rmsd(transformed, coords)
        assert r == pytest.approx(0.0, abs=1e-4)

    def test_shape(self, coords):
        result = py_geometry.apply_transform(coords, np.eye(3), np.zeros(3))
        assert result.shape == coords.shape


# ---------------------------------------------------------------------------
# Secondary structure
# ---------------------------------------------------------------------------


class TestSecondaryStructure:
    def test_returns_string(self, coords):
        ss = py_geometry.assign_secondary_structure(coords)
        assert isinstance(ss, str)
        assert len(ss) == coords.shape[0]

    def test_valid_characters(self, coords):
        ss = py_geometry.assign_secondary_structure(coords)
        for c in ss:
            assert c in "HETC", f"unexpected character: {c}"

    def test_small_coords(self):
        # 5 points in a line — should all be coil
        coords = np.array([[i, 0.0, 0.0] for i in range(5)])
        ss = py_geometry.assign_secondary_structure(coords)
        assert len(ss) == 5

    def test_empty(self):
        coords = np.zeros((0, 3))
        ss = py_geometry.assign_secondary_structure(coords)
        assert ss == ""


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------


class TestTMScore:
    def test_self_score(self, coords):
        n = coords.shape[0]
        invmap = np.arange(n, dtype=np.int32)
        tm, nali, rmsd_val, rot, trans = py_geometry.tm_score(coords, coords, invmap)
        assert tm == pytest.approx(1.0, abs=0.001)
        assert nali == n
        assert rmsd_val == pytest.approx(0.0, abs=0.01)

    def test_returns_valid_shapes(self, coords):
        n = coords.shape[0]
        invmap = np.arange(n, dtype=np.int32)
        tm, nali, rmsd_val, rot, trans = py_geometry.tm_score(coords, coords, invmap)
        assert rot.shape == (3, 3)
        assert trans.shape == (3,)

    def test_partial_alignment(self, coords):
        n = coords.shape[0]
        # Only align first half
        invmap = np.full(n, -1, dtype=np.int32)
        invmap[:n // 2] = np.arange(n // 2, dtype=np.int32)
        tm, nali, _, _, _ = py_geometry.tm_score(coords, coords, invmap)
        assert 0.0 < tm <= 1.0
        assert nali == n // 2

    def test_no_alignment(self, coords):
        n = coords.shape[0]
        invmap = np.full(n, -1, dtype=np.int32)  # all gaps
        tm, nali, _, _, _ = py_geometry.tm_score(coords, coords, invmap)
        assert tm == 0.0
        assert nali == 0

    def test_mismatched_invmap_length(self, coords):
        invmap = np.zeros(5, dtype=np.int32)  # wrong length
        with pytest.raises(ValueError):
            py_geometry.tm_score(coords, coords, invmap)
