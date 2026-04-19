"""Real-PDB end-to-end smoke test for the corpus release pipeline.

The existing `test_corpus_smoke.py` monkey-patches every pipeline
step, which guards orchestration plumbing but not correctness of any
single stage. This test complements it: load a real crystal
structure (crambin, PDB 1CRN, 46 residues), run the full Layer-5
chain against it, and assert the resulting artifacts have
plausible AF2-shaped tensors and consistent cross-layer counts.

Uses the repo's own `test-pdbs/1crn.pdb` fixture (also used by the
CHARMM/AMBER invariant suites), so no external download and no
network. Crambin is small enough to run in well under a second but
has real sidechains — exercises chi-angle + rigidgroup-frame paths
that GLY-only synthetic fixtures do not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import proteon
from proteon import io as _proteon_io
from proteon import supervision_export as sup_export

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAMBIN_PDB = REPO_ROOT / "test-pdbs" / "1crn.pdb"


# Skip when either the PDB fixture is missing OR the Rust I/O backend
# isn't importable. In source-only environments without
# proteon_connector, `proteon.io._io` is None and
# `proteon.batch_load_tolerant()` raises AttributeError on first call.
# The structure-supervision tests can run on hand-built namespaces,
# but this file is explicitly a "real PDB through real I/O" smoke —
# no fake-structure shortcut applies.
pytestmark = pytest.mark.skipif(
    not CRAMBIN_PDB.exists() or _proteon_io._io is None,
    reason=(
        "real-PDB smoke requires both test-pdbs/1crn.pdb and the Rust "
        "I/O backend (proteon_connector.py_io). Skipping; run the "
        "source-only Layer 5 tests via tests/test_supervision.py etc."
    ),
)


def _load_crambin():
    """Load the crambin fixture via the public batch API."""
    pairs = proteon.batch_load_tolerant([str(CRAMBIN_PDB)])
    assert len(pairs) == 1, "expected exactly one parsed structure"
    _, structure = pairs[0]
    return structure


class TestStructureSupervisionOnRealCrambin:
    """End-to-end supervision extraction on the crambin crystal structure."""

    def test_supervision_tensors_have_af2_canonical_shapes(self):
        structure = _load_crambin()
        examples = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1crn:A"]
        )
        assert len(examples) == 1
        ex = examples[0]

        # Crambin is 46 residues — chain length pinned to the literature
        # value so a regression in residue filtering (HETATM handling,
        # altloc pruning, etc.) shows up as a shape mismatch, not a
        # silently-corrupted example.
        n = 46
        assert ex.length == n, f"expected {n} residues, got {ex.length}"
        assert len(ex.sequence) == n

        # Core AF2-shaped supervision tensors per roadmap Section 8.
        assert ex.all_atom_positions.shape == (n, 37, 3)
        assert ex.all_atom_mask.shape == (n, 37)
        assert ex.atom14_gt_positions.shape == (n, 14, 3)
        assert ex.atom14_gt_exists.shape == (n, 14)
        assert ex.atom14_atom_exists.shape == (n, 14)
        assert ex.atom14_atom_is_ambiguous.shape == (n, 14)
        assert ex.atom37_atom_exists.shape == (n, 37)
        assert ex.pseudo_beta.shape == (n, 3)
        assert ex.pseudo_beta_mask.shape == (n,)
        assert ex.rigidgroups_gt_frames.shape == (n, 8, 4, 4)
        assert ex.rigidgroups_gt_exists.shape == (n, 8)
        assert ex.rigidgroups_group_exists.shape == (n, 8)
        assert ex.rigidgroups_group_is_ambiguous.shape == (n, 8)
        assert ex.chi_angles.shape == (n, 4)
        assert ex.chi_mask.shape == (n, 4)
        assert ex.phi.shape == (n,) and ex.psi.shape == (n,) and ex.omega.shape == (n,)

    def test_atom14_mask_consistent_with_coords(self):
        """`atom14_gt_exists` must be 1 exactly where coords are finite."""
        structure = _load_crambin()
        ex = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1crn:A"]
        )[0]

        coords_finite = np.isfinite(ex.atom14_gt_positions).all(axis=-1)
        exists = ex.atom14_gt_exists.astype(bool)
        # Every present atom must have finite coords. The reverse
        # direction (finite coords imply present) is allowed to be
        # looser — some exporters zero-fill missing atoms.
        assert np.all(~exists | coords_finite), (
            "atom14_gt_exists claims atoms at NaN/Inf coordinates"
        )

        # Crambin has no disulfide-cleaved residues; at least backbone
        # CA must always exist. That's a weak but nonzero sanity.
        assert exists.sum() > 46, (
            f"far too few present atoms ({int(exists.sum())}) for crambin"
        )

    def test_rigidgroups_backbone_frame_is_proper_rotation(self):
        """The first rigidgroup frame (backbone) must be a proper
        rigid transform: rotation has det ~ +1, last row is [0,0,0,1].
        A bug in frame construction would show up here as a reflection
        (det ~ -1) or a non-homogeneous last row."""
        structure = _load_crambin()
        ex = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1crn:A"]
        )[0]

        frames = ex.rigidgroups_gt_frames  # (N, 8, 4, 4)
        exists = ex.rigidgroups_gt_exists.astype(bool)
        # Walk present backbone frames.
        bb_exists = exists[:, 0]
        assert bb_exists.any(), "no backbone frames exist — parsing likely failed"
        present = frames[bb_exists, 0]
        rot = present[:, :3, :3]
        det = np.linalg.det(rot)
        np.testing.assert_allclose(
            det, np.ones_like(det), atol=1e-3,
            err_msg="backbone rotation determinants drifted from +1",
        )
        np.testing.assert_allclose(
            present[:, 3, :], np.tile(np.array([0, 0, 0, 1]), (present.shape[0], 1)),
            atol=1e-6,
            err_msg="backbone frame last row is not homogeneous [0,0,0,1]",
        )


class TestExportRoundtripOnRealCrambin:
    """Supervision export + reload against real tensors and checksum path."""

    def test_export_roundtrip_preserves_structure_tensors(self, tmp_path):
        structure = _load_crambin()
        examples = proteon.batch_build_structure_supervision_examples(
            [structure], record_ids=["1crn:A"]
        )

        out_dir = sup_export.export_structure_supervision_examples(
            examples, tmp_path / "supervision_1crn"
        )
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        # Gap 4 integration on real payload: tensor_sha256 present +
        # matches the on-disk hex digest, load verifies without error.
        assert "tensor_sha256" in manifest
        assert len(manifest["tensor_sha256"]) == 64  # SHA-256 hex

        loaded = sup_export.load_structure_supervision_examples(out_dir)
        assert len(loaded) == 1
        assert loaded[0].sequence == examples[0].sequence
        np.testing.assert_array_equal(
            loaded[0].atom14_gt_positions, examples[0].atom14_gt_positions
        )
        np.testing.assert_array_equal(
            loaded[0].rigidgroups_gt_frames, examples[0].rigidgroups_gt_frames
        )


class TestSequenceExampleOnRealCrambin:
    """SequenceExample alone (sequence-only path, no MSA)."""

    def test_sequence_example_basics(self):
        structure = _load_crambin()
        examples = proteon.batch_build_sequence_examples([structure])
        assert len(examples) == 1
        ex = examples[0]
        assert ex.length == 46
        assert ex.aatype.shape == (46,)
        assert ex.residue_index.shape == (46,)
        assert ex.seq_mask.shape == (46,)
        # aatype must be in [0, 20] for standard residues.
        assert int(ex.aatype.min()) >= 0 and int(ex.aatype.max()) <= 20
