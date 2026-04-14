"""Tests for framework-neutral structure supervision artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import ferritin


def _atom(name, xyz):
    return SimpleNamespace(name=name, pos=tuple(float(x) for x in xyz))


def _fake_structure(chain_id="A"):
    # Three residues with enough atoms for backbone torsions, gly pseudo-beta,
    # chi1 on SER, and canonical atom37/atom14 mapping checks.
    residues = [
        SimpleNamespace(
            name="GLY",
            serial_number=1,
            is_amino_acid=True,
            atoms=[
                _atom("N", (0.0, 0.0, 0.0)),
                _atom("CA", (1.0, 0.0, 0.0)),
                _atom("C", (1.8, 1.0, 0.0)),
                _atom("O", (1.8, 2.1, 0.0)),
            ],
        ),
        SimpleNamespace(
            name="SER",
            serial_number=2,
            is_amino_acid=True,
            atoms=[
                _atom("N", (2.6, 0.8, 0.8)),
                _atom("CA", (3.5, 1.6, 1.0)),
                _atom("C", (4.7, 0.9, 1.4)),
                _atom("O", (5.0, -0.2, 1.1)),
                _atom("CB", (3.2, 2.9, 1.8)),
                _atom("OG", (2.1, 3.5, 1.4)),
            ],
        ),
        SimpleNamespace(
            name="PHE",
            serial_number=3,
            is_amino_acid=True,
            atoms=[
                _atom("N", (5.5, 1.6, 2.2)),
                _atom("CA", (6.7, 1.1, 2.7)),
                _atom("C", (7.8, 2.0, 3.0)),
                _atom("O", (7.7, 3.2, 3.0)),
                _atom("CB", (6.8, -0.1, 3.7)),
                _atom("CG", (8.0, -0.9, 4.0)),
                _atom("CD1", (9.2, -0.4, 3.6)),
                _atom("CD2", (7.9, -2.1, 4.7)),
                _atom("CE1", (10.3, -1.1, 3.9)),
                _atom("CE2", (9.0, -2.8, 5.0)),
                _atom("CZ", (10.2, -2.3, 4.6)),
            ],
        ),
    ]
    chain = SimpleNamespace(id=chain_id, residues=residues)
    return SimpleNamespace(
        identifier="fake",
        chain_count=1,
        chains=[chain],
    )


def _bad_structure():
    chain = SimpleNamespace(
        id="Z",
        residues=[SimpleNamespace(name="HOH", serial_number=1, is_amino_acid=False, atoms=[])],
    )
    return SimpleNamespace(
        identifier="bad",
        chain_count=1,
        chains=[chain],
    )


class TestStructureSupervisionExample:
    def test_builds_core_example(self):
        s = _fake_structure()
        prep = ferritin.PrepReport(hydrogens_added=3, hydrogens_skipped=0)
        ex = ferritin.build_structure_supervision_example(
            s,
            prep_report=prep,
            record_id="fake:A",
            source_id="fake",
        )

        assert isinstance(ex, ferritin.StructureSupervisionExample)
        assert ex.record_id == "fake:A"
        assert ex.chain_id == "A"
        assert ex.sequence == "GSF"
        assert ex.length == 3
        assert ex.aatype.shape == (3,)
        assert ex.residue_index.tolist() == [1, 2, 3]
        assert np.all(ex.seq_mask == 1.0)

    def test_atom37_and_atom14_shapes_are_present(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())

        assert ex.all_atom_positions.shape == (3, 37, 3)
        assert ex.all_atom_mask.shape == (3, 37)
        assert ex.atom37_atom_exists.shape == (3, 37)
        assert ex.atom14_gt_positions.shape == (3, 14, 3)
        assert ex.atom14_gt_exists.shape == (3, 14)
        assert ex.atom14_atom_exists.shape == (3, 14)
        assert ex.residx_atom14_to_atom37.shape == (3, 14)
        assert ex.residx_atom37_to_atom14.shape == (3, 37)

    def test_pseudo_beta_uses_ca_for_gly_and_cb_for_non_gly(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())

        # GLY residue 0 should use CA
        np.testing.assert_allclose(ex.pseudo_beta[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert ex.pseudo_beta_mask[0] == 1.0
        # SER residue 1 should use CB
        np.testing.assert_allclose(ex.pseudo_beta[1], np.array([3.2, 2.9, 1.8], dtype=np.float32))
        assert ex.pseudo_beta_mask[1] == 1.0

    def test_backbone_and_chi_masks_are_computed(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())

        assert ex.phi.shape == (3,)
        assert ex.psi.shape == (3,)
        assert ex.omega.shape == (3,)
        assert ex.phi_mask.tolist() == [0.0, 1.0, 1.0]
        assert ex.psi_mask.tolist() == [1.0, 1.0, 0.0]
        assert ex.omega_mask.tolist() == [0.0, 1.0, 1.0]
        assert ex.chi_angles.shape == (3, 4)
        # SER has chi1, GLY has none, PHE has chi1/chi2
        assert ex.chi_mask[0].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert ex.chi_mask[1].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert ex.chi_mask[2].tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_ambiguity_flags_mark_symmetric_sidechains(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())

        # PHE residue is index 2 and should mark CD1/CD2 and CE1/CE2 ambiguous.
        assert ex.atom14_atom_is_ambiguous.shape == (3, 14)
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[0]) == 0
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[1]) == 0
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[2]) == 4

    def test_quality_metadata_is_attached(self):
        prep = ferritin.PrepReport(
            hydrogens_added=3,
            hydrogens_skipped=1,
            atoms_reconstructed=2,
            n_unassigned_atoms=4,
        )
        ex = ferritin.build_structure_supervision_example(_fake_structure(), prep_report=prep)

        assert isinstance(ex.quality, ferritin.StructureQualityMetadata)
        assert ex.quality.hydrogens_added == 3
        assert ex.quality.hydrogens_skipped == 1
        assert ex.quality.atoms_reconstructed == 2
        assert ex.quality.n_unassigned_atoms == 4

    def test_rigidgroups_are_materialized(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())
        assert ex.rigidgroups_gt_frames.shape == (3, 8, 4, 4)
        assert ex.rigidgroups_gt_exists.shape == (3, 8)
        assert ex.rigidgroups_group_exists.shape == (3, 8)
        assert ex.rigidgroups_group_is_ambiguous.shape == (3, 8)
        assert ex.rigidgroups_group_exists[0].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert ex.rigidgroups_group_exists[1].tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert ex.rigidgroups_group_exists[2].tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        assert ex.rigidgroups_group_is_ambiguous[2].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(ex.rigidgroups_gt_frames[0, 0, :3, 3], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(ex.rigidgroups_gt_frames[1, 3, :3, 3], np.array([4.7, 0.9, 1.4], dtype=np.float32))

    def test_is_partial_is_false_once_rigid_groups_exist(self):
        ex = ferritin.build_structure_supervision_example(_fake_structure())
        assert not ex.is_partial

    def test_batch_builder_matches_single_builder_metadata(self):
        batch = ferritin.batch_build_structure_supervision_examples([_fake_structure(), _fake_structure()])
        assert len(batch) == 2
        assert batch[0].sequence == batch[1].sequence == "GSF"

    def test_export_roundtrip_preserves_examples(self, tmp_path):
        examples = ferritin.batch_build_structure_supervision_examples(
            [_fake_structure("A"), _fake_structure("B")]
        )
        out_dir = ferritin.export_structure_supervision_examples(examples, tmp_path / "supervision")

        manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
        assert ferritin.SUPERVISION_EXPORT_FORMAT in manifest

        loaded = ferritin.load_structure_supervision_examples(out_dir)
        assert len(loaded) == 2
        assert loaded[0].sequence == "GSF"
        assert loaded[1].sequence == "GSF"
        np.testing.assert_array_equal(loaded[0].aatype, examples[0].aatype)
        np.testing.assert_array_equal(loaded[1].residue_index, examples[1].residue_index)
        np.testing.assert_allclose(loaded[0].rigidgroups_gt_frames, examples[0].rigidgroups_gt_frames)
        np.testing.assert_allclose(loaded[1].chi_angles, examples[1].chi_angles)

    def test_export_writes_tensor_sha256_and_load_verifies(self, tmp_path):
        """Per roadmap Section 6 — every artifact carries a checksum.

        Export must write the hex SHA-256 of tensors.npz into the
        manifest, and load must reject a tampered payload.
        """
        import hashlib
        import json

        examples = ferritin.batch_build_structure_supervision_examples(
            [_fake_structure("A")]
        )
        out_dir = ferritin.export_structure_supervision_examples(
            examples, tmp_path / "supervision"
        )
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256((out_dir / "tensors.npz").read_bytes()).hexdigest()
        assert manifest["tensor_sha256"] == expected_hash

        # Good path: default load verifies and succeeds.
        ferritin.load_structure_supervision_examples(out_dir)

        # Tamper the tensor file; default load must raise with a
        # checksum-mismatch error (not an obscure np.load error later).
        # With verify_checksum=False the hash check is skipped; what
        # happens after is whatever np.load does — we only guarantee
        # the checksum gate here.
        (out_dir / "tensors.npz").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum mismatch"):
            ferritin.load_structure_supervision_examples(out_dir)

    def test_release_builder_writes_manifest_and_failures(self, tmp_path):
        examples = ferritin.batch_build_structure_supervision_examples([_fake_structure("A")])
        failures = [
            ferritin.FailureRecord(
                record_id="bad:1",
                failure_class="missing_required_atoms",
                message="missing CA atom",
                source_id="bad",
            )
        ]
        release_dir = ferritin.build_structure_supervision_release(
            examples,
            tmp_path / "release",
            release_id="demo-v0",
            failures=failures,
            code_rev="abc123",
            config_rev="cfg1",
            provenance={"source_manifest": "raw-v1"},
        )

        manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["release_id"] == "demo-v0"
        assert manifest["count_examples"] == 1
        assert manifest["count_failures"] == 1
        assert manifest["code_rev"] == "abc123"
        assert manifest["provenance"]["source_manifest"] == "raw-v1"

        loaded_failures = ferritin.load_failure_records(release_dir / "failures.jsonl")
        assert len(loaded_failures) == 1
        assert loaded_failures[0].failure_class == "missing_required_atoms"

    def test_dataset_builder_captures_failures_and_writes_release(self, tmp_path):
        release_dir = ferritin.build_structure_supervision_dataset(
            [_fake_structure("A"), _bad_structure()],
            tmp_path / "dataset_release",
            release_id="dataset-v0",
            record_ids=["good:A", "bad:Z"],
            source_ids=["good", "bad"],
            provenance={"source_manifest": "raw-demo"},
        )

        manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["count_examples"] == 1
        assert manifest["count_failures"] == 1
        assert manifest["release_id"] == "dataset-v0"

        examples = ferritin.load_structure_supervision_examples(release_dir / "examples")
        assert len(examples) == 1
        assert examples[0].record_id == "good:A"

        failures = ferritin.load_failure_records(release_dir / "failures.jsonl")
        assert len(failures) == 1
        assert failures[0].record_id == "bad:Z"
        assert failures[0].failure_class == "missing_required_atoms"

    def test_prepared_bridge_writes_manifest_and_supervision_release(self, tmp_path):
        prep = ferritin.PrepReport(
            atoms_reconstructed=2,
            hydrogens_added=3,
            hydrogens_skipped=1,
            minimizer_steps=7,
            converged=True,
        )
        root = ferritin.build_structure_supervision_dataset_from_prepared(
            [_fake_structure("A")],
            [prep],
            tmp_path / "prepared_bridge",
            release_id="prepared-v0",
            record_ids=["fake:A"],
            source_ids=["fake"],
            prep_run_ids=["prep-1"],
            provenance={"source_manifest": "prepared-demo"},
        )

        prepared_rows = ferritin.load_prepared_structure_manifest(root / "prepared_structures.jsonl")
        assert len(prepared_rows) == 1
        assert prepared_rows[0].record_id == "fake:A"
        assert prepared_rows[0].atoms_reconstructed == 2
        assert prepared_rows[0].hydrogens_added == 3

        examples = ferritin.load_structure_supervision_examples(root / "supervision_release" / "examples")
        assert len(examples) == 1
        assert examples[0].record_id == "fake:A"
