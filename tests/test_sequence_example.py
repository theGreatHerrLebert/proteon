"""Tests for framework-neutral sequence example artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import ferritin


def _atom(name, xyz):
    return SimpleNamespace(name=name, pos=tuple(float(x) for x in xyz))


def _fake_structure(chain_id="A"):
    residues = [
        SimpleNamespace(name="GLY", serial_number=1, is_amino_acid=True, atoms=[_atom("CA", (1, 0, 0))]),
        SimpleNamespace(name="SER", serial_number=2, is_amino_acid=True, atoms=[_atom("CA", (2, 0, 0))]),
        SimpleNamespace(name="PHE", serial_number=3, is_amino_acid=True, atoms=[_atom("CA", (3, 0, 0))]),
    ]
    chain = SimpleNamespace(id=chain_id, residues=residues)
    return SimpleNamespace(identifier="fake", chain_count=1, chains=[chain])


def _bad_structure():
    chain = SimpleNamespace(id="Z", residues=[SimpleNamespace(name="HOH", serial_number=1, is_amino_acid=False, atoms=[])])
    return SimpleNamespace(identifier="bad", chain_count=1, chains=[chain])


class TestSequenceExample:
    def test_build_sequence_example_core(self):
        ex = ferritin.build_sequence_example(_fake_structure(), record_id="fake:A", source_id="fake")
        assert ex.record_id == "fake:A"
        assert ex.sequence == "GSF"
        assert ex.length == 3
        assert ex.aatype.shape == (3,)
        assert ex.residue_index.tolist() == [1, 2, 3]
        assert ex.seq_mask.tolist() == [1.0, 1.0, 1.0]

    def test_build_sequence_example_with_msa(self):
        ex = ferritin.build_sequence_example(
            _fake_structure(),
            msa=["GSF", "GXF"],
            deletion_matrix=[[0, 0, 0], [0, 1, 0]],
            template_mask=[1.0, 0.0],
        )
        assert ex.msa.shape == (2, 3)
        assert ex.deletion_matrix.shape == (2, 3)
        assert ex.msa_mask.shape == (2, 3)
        assert ex.template_mask.tolist() == [1.0, 0.0]

    def test_sequence_export_roundtrip(self, tmp_path):
        examples = ferritin.batch_build_sequence_examples(
            [_fake_structure("A"), _fake_structure("B")],
            msas=[["GSF"], None],
            deletion_matrices=[[[0, 0, 0]], None],
        )
        out_dir = ferritin.export_sequence_examples(examples, tmp_path / "sequence")
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format"] == ferritin.SEQUENCE_EXPORT_FORMAT

        loaded = ferritin.load_sequence_examples(out_dir)
        assert len(loaded) == 2
        assert loaded[0].sequence == "GSF"
        assert loaded[0].msa.shape == (1, 3)
        assert loaded[1].msa is None
        np.testing.assert_array_equal(loaded[0].aatype, examples[0].aatype)

    def test_sequence_export_writes_tensor_sha256_and_load_verifies(self, tmp_path):
        """Checksum parity with the structure-supervision exporter."""
        import hashlib
        import pytest

        examples = ferritin.batch_build_sequence_examples([_fake_structure("A")])
        out_dir = ferritin.export_sequence_examples(examples, tmp_path / "sequence")
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        tensor_path = out_dir / manifest["tensor_file"]
        assert tensor_path.name == "tensors.parquet"
        expected = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        assert manifest["tensor_sha256"] == expected

        ferritin.load_sequence_examples(out_dir)  # verifies and succeeds
        tensor_path.write_bytes(b"x")
        with pytest.raises(ValueError, match="checksum mismatch"):
            ferritin.load_sequence_examples(out_dir)

    def test_sequence_release_builder_captures_failures(self, tmp_path):
        root = ferritin.build_sequence_dataset(
            [_fake_structure("A"), _bad_structure()],
            tmp_path / "sequence_release",
            release_id="seq-v0",
            record_ids=["fake:A", "bad:Z"],
            source_ids=["fake", "bad"],
            provenance={"source_manifest": "seq-demo"},
        )
        manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["count_examples"] == 1
        assert manifest["count_failures"] == 1
        loaded = ferritin.load_sequence_examples(root / "examples")
        assert len(loaded) == 1
        failures = ferritin.load_failure_records(root / "failures.jsonl")
        assert len(failures) == 1
        assert failures[0].stage == "sequence_example"

    def test_sequence_release_allows_failure_only_release(self, tmp_path):
        failure = ferritin.FailureRecord(
            record_id="bad:A",
            stage="sequence_example",
            failure_class="missing_required_atoms",
            message="missing CA atom",
            source_id="bad",
        )
        root = ferritin.build_sequence_release(
            [],
            tmp_path / "sequence_fail_only",
            release_id="seq-fail-only-v0",
            failures=[failure],
        )

        manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["count_examples"] == 0
        assert manifest["count_failures"] == 1
        assert ferritin.load_sequence_examples(root / "examples") == []
