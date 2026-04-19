"""Rust-vs-Python parity regression guards for supervision extraction.

The supervision hot path dispatches to the Rust batch extractor
(`proteon_connector.py_supervision.batch_extract_structure_supervision`)
whenever it's available, falling back to the Python implementation
in `supervision.py` / `supervision_geometry.py` otherwise. Both paths
are required to produce bit-for-bit identical tensors — any drift
silently poisons downstream training since half the corpus might be
built under Rust and half under Python.

These tests pin the invariant. They run only when the Rust backend
imports cleanly (so CPU-only Python envs without the connector still
have a green test suite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proteon
from proteon.supervision_backend import (
    batch_extract_structure_supervision,
    rust_supervision_available,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAMBIN = REPO_ROOT / "test-pdbs" / "1crn.pdb"
UBIQUITIN = REPO_ROOT / "test-pdbs" / "1ubq.pdb"


pytestmark = pytest.mark.skipif(
    not rust_supervision_available(),
    reason="proteon_connector.py_supervision not importable — Rust parity test skipped",
)


# Tensors we verify Rust and Python produce identically. Picked to
# cover every computation path that has a chance to drift:
#   - coordinate extraction (all_atom, atom14)
#   - mask computation (atom14_gt_exists, pseudo_beta_mask)
#   - torsion computation (phi/psi/omega/chi — all four dihedral kinds)
#   - rigidgroup frame construction (all 8 groups)
_COORD_FIELDS = ("all_atom_positions", "atom14_gt_positions", "pseudo_beta")
_MASK_FIELDS = (
    "all_atom_mask",
    "atom14_gt_exists",
    "atom14_atom_exists",
    "atom37_atom_exists",
    "pseudo_beta_mask",
    "rigidgroups_gt_exists",
    "phi_mask",
    "psi_mask",
    "omega_mask",
    "chi_mask",
)
_ANGLE_FIELDS = ("phi", "psi", "omega", "chi_angles")
_FRAME_FIELDS = ("rigidgroups_gt_frames",)


def _load(path: Path):
    pairs = proteon.batch_load_tolerant([str(path)])
    assert len(pairs) == 1, f"failed to load {path}"
    return pairs[0][1]


def _rust_extract_single(structure):
    """Rust batch extractor, sliced back down to a single example."""
    batch = batch_extract_structure_supervision([structure])
    # Batch returns `(1, N, ...)` padded to longest; this is a 1-element
    # batch so there's no padding to strip (except trailing zeros on the
    # per-example length axis, which we handle by comparing per-example).
    return batch


def _compare_single(rust_batch, py_example, tag: str):
    n = py_example.length

    for name in _COORD_FIELDS:
        rust = np.asarray(rust_batch[name])[0, :n]
        py = np.asarray(getattr(py_example, name))
        np.testing.assert_array_equal(
            rust, py, err_msg=f"{tag}: coord field `{name}` drifted Rust vs Python"
        )

    for name in _MASK_FIELDS:
        rust = np.asarray(rust_batch[name])[0, :n]
        py = np.asarray(getattr(py_example, name))
        np.testing.assert_array_equal(
            rust, py, err_msg=f"{tag}: mask field `{name}` drifted Rust vs Python"
        )

    for name in _ANGLE_FIELDS:
        rust = np.asarray(rust_batch[name])[0, :n]
        py = np.asarray(getattr(py_example, name))
        # Angles produced via dihedral math involve sqrts / atan2 — allow
        # a tiny float tolerance in case either side reorders a sum.
        # In practice (2026-04-14 crambin + ubiquitin) diff == 0.
        np.testing.assert_allclose(
            rust, py, atol=1e-5, rtol=1e-5,
            err_msg=f"{tag}: angle field `{name}` drifted Rust vs Python",
        )

    for name in _FRAME_FIELDS:
        rust = np.asarray(rust_batch[name])[0, :n]
        py = np.asarray(getattr(py_example, name))
        np.testing.assert_allclose(
            rust, py, atol=1e-5, rtol=1e-5,
            err_msg=f"{tag}: frame field `{name}` drifted Rust vs Python",
        )


class TestRustPythonParity:
    @pytest.mark.skipif(not CRAMBIN.exists(), reason="1crn fixture missing")
    def test_crambin_rust_matches_python(self):
        """46-residue structure with real sidechains (THR/CYS/PRO/VAL/LEU).
        Exercises chi-angle + rigidgroup-frame paths."""
        structure = _load(CRAMBIN)
        rust = _rust_extract_single(structure)
        py = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1crn:A"]
        )[0]
        _compare_single(rust, py, tag="crambin")

    @pytest.mark.skipif(not UBIQUITIN.exists(), reason="1ubq fixture missing")
    def test_ubiquitin_rust_matches_python(self):
        """76-residue structure, bigger sidechain variety than crambin.
        Mostly a coverage check: a different residue composition can
        flag chi-angle atom-lookup bugs that crambin doesn't hit."""
        structure = _load(UBIQUITIN)
        rust = _rust_extract_single(structure)
        py = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1ubq:A"]
        )[0]
        _compare_single(rust, py, tag="ubiquitin")

    @pytest.mark.skipif(
        not (CRAMBIN.exists() and UBIQUITIN.exists()),
        reason="need both 1crn and 1ubq for batch-padding parity test",
    )
    def test_mixed_length_batch_rust_matches_python(self):
        """Two structures of different length through the batch path.
        Confirms Rust's per-example pad-then-slice matches Python's
        unpadded per-example output. Padding bugs typically manifest
        as off-by-one shape mismatches or zeros bleeding into valid
        positions — this pins both shape AND valid-region equality."""
        crambin = _load(CRAMBIN)
        ubq = _load(UBIQUITIN)
        rust = batch_extract_structure_supervision([crambin, ubq])
        py = proteon.batch_build_structure_supervision_examples(
            [crambin, ubq], record_ids=["1crn:A", "1ubq:A"]
        )

        assert len(py) == 2
        for i, (py_ex, tag) in enumerate(zip(py, ["crambin", "ubiquitin"])):
            n = py_ex.length
            # Slice each field's leading batch axis to this example,
            # then its valid length. Rust pads to max length in the
            # batch; Python returns each at its real length.
            for name in _COORD_FIELDS + _MASK_FIELDS + _ANGLE_FIELDS + _FRAME_FIELDS:
                rust_arr = np.asarray(rust[name])[i, :n]
                py_arr = np.asarray(getattr(py_ex, name))
                np.testing.assert_allclose(
                    rust_arr, py_arr, atol=1e-5, rtol=1e-5,
                    err_msg=f"batch idx {i} ({tag}): `{name}` drifted Rust vs Python",
                )
