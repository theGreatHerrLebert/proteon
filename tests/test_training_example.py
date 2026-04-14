"""Tests for joined training example artifacts."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ferritin


def _atom(name, xyz):
    return SimpleNamespace(name=name, pos=tuple(float(x) for x in xyz))


def _fake_structure(chain_id="A"):
    residues = [
        SimpleNamespace(
            name="GLY",
            serial_number=1,
            is_amino_acid=True,
            atoms=[_atom("N", (0, 0, 0)), _atom("CA", (1, 0, 0)), _atom("C", (1.8, 1, 0)), _atom("O", (1.8, 2.1, 0))],
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
    ]
    chain = SimpleNamespace(id=chain_id, residues=residues)
    return SimpleNamespace(identifier="fake", chain_count=1, chains=[chain])


class TestTrainingExample:
    def test_join_training_release_from_sequence_and_structure(self, tmp_path):
        structures = [_fake_structure("A")]
        seq_release = ferritin.build_sequence_dataset(
            structures,
            tmp_path / "seq_release",
            release_id="seq-v0",
            record_ids=["fake:A"],
        )
        prep = ferritin.PrepReport(hydrogens_added=2, converged=True)
        struc_release_root = ferritin.build_structure_supervision_dataset_from_prepared(
            structures,
            [prep],
            tmp_path / "struc_release_root",
            release_id="struc-v0",
            record_ids=["fake:A"],
        )
        train_release = ferritin.build_training_release(
            seq_release,
            struc_release_root / "supervision_release",
            tmp_path / "train_release",
            release_id="train-v0",
            split_assignments={"fake:A": "train"},
            crop_metadata={"fake:A": (0, 2)},
            weights={"fake:A": 0.5},
        )

        manifest = json.loads((train_release / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["release_id"] == "train-v0"
        assert manifest["count_examples"] == 1
        assert manifest["split_counts"]["train"] == 1

        row = json.loads((train_release / "training_examples.jsonl").read_text(encoding="utf-8").strip())
        assert row["record_id"] == "fake:A"
        assert row["split"] == "train"
        assert row["crop_start"] == 0
        assert row["crop_stop"] == 2
        assert row["weight"] == 0.5

    def test_training_release_writes_denormalized_tensors_npz(self, tmp_path):
        """build_training_release with export_tensors=True (default) must
        produce a tensors.npz, checksum it into the manifest, and
        roundtrip cleanly via load_training_examples."""
        structures = [_fake_structure("A"), _fake_structure("B")]
        seq_release = ferritin.build_sequence_dataset(
            structures,
            tmp_path / "seq",
            release_id="seq-v0",
            record_ids=["fake:A", "fake:B"],
        )
        preps = [ferritin.PrepReport(hydrogens_added=0, converged=True) for _ in structures]
        struc_root = ferritin.build_structure_supervision_dataset_from_prepared(
            structures,
            preps,
            tmp_path / "struc_root",
            release_id="struc-v0",
            record_ids=["fake:A", "fake:B"],
        )
        train = ferritin.build_training_release(
            seq_release,
            struc_root / "supervision_release",
            tmp_path / "train",
            release_id="train-v0",
            split_assignments={"fake:A": "train", "fake:B": "val"},
            weights={"fake:A": 1.0, "fake:B": 0.5},
        )

        # 1. tensors.npz exists + manifest carries a real SHA-256.
        tensor_path = train / "tensors.npz"
        assert tensor_path.exists(), "training release should emit tensors.npz"
        manifest = json.loads((train / "release_manifest.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        assert manifest["tensor_sha256"] == expected
        assert len(manifest["tensor_fields"]) > 0

        # 2. NPZ carries the concatenated tensors with correct shapes.
        # Synthetic residues: A has 2 (GLY, SER), B has 2. Max length 2.
        payload = np.load(tensor_path, allow_pickle=False)
        assert payload["aatype"].shape == (2, 2)
        assert payload["all_atom_positions"].shape == (2, 2, 37, 3)
        assert payload["rigidgroups_gt_frames"].shape == (2, 2, 8, 4, 4)
        # Per-example bookkeeping lets downstream slice correctly.
        np.testing.assert_array_equal(payload["length"], np.asarray([2, 2], dtype=np.int32))
        np.testing.assert_array_equal(payload["weight"], np.asarray([1.0, 0.5], dtype=np.float32))

        # 3. load_training_examples round-trips via the NPZ path.
        reloaded = ferritin.load_training_examples(train)
        assert len(reloaded) == 2
        by_id = {ex.record_id: ex for ex in reloaded}
        assert set(by_id) == {"fake:A", "fake:B"}
        assert by_id["fake:A"].split == "train"
        assert by_id["fake:B"].split == "val"
        assert by_id["fake:A"].weight == pytest.approx(1.0)
        assert by_id["fake:B"].weight == pytest.approx(0.5)
        # Tensors populated on both sides of the join.
        assert by_id["fake:A"].sequence is not None
        assert by_id["fake:A"].structure is not None
        assert by_id["fake:A"].sequence.aatype.shape == (2,)
        assert by_id["fake:A"].structure.all_atom_positions.shape == (2, 37, 3)

        # 4. Tamper the NPZ and assert load rejects by default.
        tensor_path.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum mismatch"):
            ferritin.load_training_examples(train)

    def test_export_tensors_false_leaves_pointer_only_release(self, tmp_path):
        """Backwards-compat: `export_tensors=False` keeps the legacy
        pointer-only behavior. Manifest has no tensor_file; loading
        falls back to re-joining via the child releases."""
        structures = [_fake_structure("A")]
        seq_release = ferritin.build_sequence_dataset(
            structures, tmp_path / "seq", release_id="seq-v0", record_ids=["fake:A"]
        )
        preps = [ferritin.PrepReport(hydrogens_added=0, converged=True)]
        struc_root = ferritin.build_structure_supervision_dataset_from_prepared(
            structures, preps, tmp_path / "struc_root",
            release_id="struc-v0", record_ids=["fake:A"],
        )
        train = ferritin.build_training_release(
            seq_release,
            struc_root / "supervision_release",
            tmp_path / "train",
            release_id="train-v0",
            split_assignments={"fake:A": "train"},
            export_tensors=False,
        )
        assert not (train / "tensors.npz").exists()
        manifest = json.loads((train / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["tensor_file"] is None

        reloaded = ferritin.load_training_examples(train)
        assert len(reloaded) == 1
        assert reloaded[0].sequence is not None
