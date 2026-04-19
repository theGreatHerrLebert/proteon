"""Tests for joined training example artifacts."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import proteon


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
        seq_release = proteon.build_sequence_dataset(
            structures,
            tmp_path / "seq_release",
            release_id="seq-v0",
            record_ids=["fake:A"],
        )
        prep = proteon.PrepReport(hydrogens_added=2, converged=True)
        struc_release_root = proteon.build_structure_supervision_dataset_from_prepared(
            structures,
            [prep],
            tmp_path / "struc_release_root",
            release_id="struc-v0",
            record_ids=["fake:A"],
        )
        train_release = proteon.build_training_release(
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

    def test_training_release_writes_training_parquet(self, tmp_path):
        """build_training_release with export_tensors=True (default) must
        produce a training.parquet, checksum it into the manifest, and
        roundtrip cleanly via load_training_examples."""
        structures = [_fake_structure("A"), _fake_structure("B")]
        seq_release = proteon.build_sequence_dataset(
            structures,
            tmp_path / "seq",
            release_id="seq-v0",
            record_ids=["fake:A", "fake:B"],
        )
        preps = [proteon.PrepReport(hydrogens_added=0, converged=True) for _ in structures]
        struc_root = proteon.build_structure_supervision_dataset_from_prepared(
            structures,
            preps,
            tmp_path / "struc_root",
            release_id="struc-v0",
            record_ids=["fake:A", "fake:B"],
        )
        train = proteon.build_training_release(
            seq_release,
            struc_root / "supervision_release",
            tmp_path / "train",
            release_id="train-v0",
            split_assignments={"fake:A": "train", "fake:B": "val"},
            weights={"fake:A": 1.0, "fake:B": 0.5},
        )

        # 1. training.parquet exists + manifest carries a real SHA-256.
        parquet_path = train / "training.parquet"
        assert parquet_path.exists(), "training release should emit training.parquet"
        manifest = json.loads((train / "release_manifest.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        assert manifest["parquet_sha256"] == expected
        assert len(manifest["parquet_fields"]) > 0
        assert "aatype" in manifest["parquet_fields"]
        assert "rigidgroups_gt_frames" in manifest["parquet_fields"]

        # 2. Parquet carries per-example ragged rows with correct shapes.
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        tbl = pf.read()
        # Synthetic residues: A has 2 (GLY, SER), B has 2.
        assert tbl.num_rows == 2
        lengths = tbl.column("length").to_pylist()
        assert lengths == [2, 2]
        weights = tbl.column("weight").to_pylist()
        assert weights == pytest.approx([1.0, 0.5])

        # 3. load_training_examples round-trips via the Parquet path.
        reloaded = proteon.load_training_examples(train)
        assert len(reloaded) == 2
        by_id = {ex.record_id: ex for ex in reloaded}
        assert set(by_id) == {"fake:A", "fake:B"}
        assert by_id["fake:A"].split == "train"
        assert by_id["fake:B"].split == "val"
        assert by_id["fake:A"].weight == pytest.approx(1.0)
        assert by_id["fake:B"].weight == pytest.approx(0.5)
        # Tensors populated on both sides of the join, correctly reshaped.
        assert by_id["fake:A"].sequence is not None
        assert by_id["fake:A"].structure is not None
        assert by_id["fake:A"].sequence.aatype.shape == (2,)
        assert by_id["fake:A"].structure.all_atom_positions.shape == (2, 37, 3)

        # 4. Tamper the parquet and assert load rejects by default.
        parquet_path.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum mismatch"):
            proteon.load_training_examples(train)

    def test_export_tensors_false_leaves_pointer_only_release(self, tmp_path):
        """`export_tensors=False` keeps the pointer-only behaviour.
        Manifest has no parquet_file; loading falls back to re-joining
        via the child releases."""
        structures = [_fake_structure("A")]
        seq_release = proteon.build_sequence_dataset(
            structures, tmp_path / "seq", release_id="seq-v0", record_ids=["fake:A"]
        )
        preps = [proteon.PrepReport(hydrogens_added=0, converged=True)]
        struc_root = proteon.build_structure_supervision_dataset_from_prepared(
            structures, preps, tmp_path / "struc_root",
            release_id="struc-v0", record_ids=["fake:A"],
        )
        train = proteon.build_training_release(
            seq_release,
            struc_root / "supervision_release",
            tmp_path / "train",
            release_id="train-v0",
            split_assignments={"fake:A": "train"},
            export_tensors=False,
        )
        assert not (train / "training.parquet").exists()
        assert not (train / "tensors.npz").exists()  # belt-and-suspenders
        manifest = json.loads((train / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["parquet_file"] is None

        reloaded = proteon.load_training_examples(train)
        assert len(reloaded) == 1
        assert reloaded[0].sequence is not None
