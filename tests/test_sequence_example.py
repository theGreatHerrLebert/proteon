"""Tests for framework-neutral sequence example artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import proteon
from proteon import sequence_export as seq_export
from proteon import sequence_release as seq_release
from proteon import supervision_release as sup_release


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
        ex = proteon.build_sequence_example(_fake_structure(), record_id="fake:A", source_id="fake")
        assert ex.record_id == "fake:A"
        assert ex.sequence == "GSF"
        assert ex.length == 3
        assert ex.aatype.shape == (3,)
        # residue_index is the 0-based positional sequence coordinate (was the
        # author serial_number [1,2,3]); see positional_residue_index.
        assert ex.residue_index.tolist() == [0, 1, 2]
        assert ex.seq_mask.tolist() == [1.0, 1.0, 1.0]

    def test_build_sequence_example_with_msa(self):
        ex = proteon.build_sequence_example(
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
        examples = proteon.batch_build_sequence_examples(
            [_fake_structure("A"), _fake_structure("B")],
            msas=[["GSF"], None],
            deletion_matrices=[[[0, 0, 0]], None],
        )
        out_dir = seq_export.export_sequence_examples(examples, tmp_path / "sequence")
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format"] == seq_export.SEQUENCE_EXPORT_FORMAT

        loaded = seq_export.load_sequence_examples(out_dir)
        assert len(loaded) == 2
        assert loaded[0].sequence == "GSF"
        assert loaded[0].msa.shape == (1, 3)
        assert loaded[1].msa is None
        np.testing.assert_array_equal(loaded[0].aatype, examples[0].aatype)
        # Author identity must survive the sequence-release round trip (not reset
        # to None / zeros) — guards the residue_index/author-identity contract.
        np.testing.assert_array_equal(loaded[0].residue_index, examples[0].residue_index)
        np.testing.assert_array_equal(loaded[0].author_seq_id, examples[0].author_seq_id)
        np.testing.assert_array_equal(loaded[0].insertion_code, examples[0].insertion_code)

    def test_sequence_export_writes_tensor_sha256_and_load_verifies(self, tmp_path):
        """Checksum parity with the structure-supervision exporter."""
        import hashlib
        import pytest

        examples = proteon.batch_build_sequence_examples([_fake_structure("A")])
        out_dir = seq_export.export_sequence_examples(examples, tmp_path / "sequence")
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        tensor_path = out_dir / manifest["tensor_file"]
        assert tensor_path.name == "tensors.parquet"
        expected = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        assert manifest["tensor_sha256"] == expected

        seq_export.load_sequence_examples(out_dir)  # verifies and succeeds
        tensor_path.write_bytes(b"x")
        with pytest.raises(ValueError, match="checksum mismatch"):
            seq_export.load_sequence_examples(out_dir)

    def test_sequence_release_builder_captures_failures(self, tmp_path):
        root = seq_release.build_sequence_dataset(
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
        loaded = seq_export.load_sequence_examples(root / "examples")
        assert len(loaded) == 1
        failures = sup_release.load_failure_records(root / "failures.jsonl")
        assert len(failures) == 1
        assert failures[0].stage == "sequence_example"

    def test_sequence_release_allows_failure_only_release(self, tmp_path):
        failure = sup_release.FailureRecord(
            record_id="bad:A",
            stage="sequence_example",
            failure_class="missing_required_atoms",
            message="missing CA atom",
            source_id="bad",
        )
        root = seq_release.build_sequence_release(
            [],
            tmp_path / "sequence_fail_only",
            release_id="seq-fail-only-v0",
            failures=[failure],
        )

        manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["count_examples"] == 0
        assert manifest["count_failures"] == 1
        assert seq_export.load_sequence_examples(root / "examples") == []


def test_crop_sequence_example_slices_author_identity(tmp_path):
    # codex P1: crop must slice the author identity arrays in lockstep with the
    # sequence, or they desync from residue_index after a crop.
    from proteon.supervision_crop import crop_sequence_example
    ex = proteon.build_sequence_example(_fake_structure("A"))
    assert ex.length == 3
    cropped = crop_sequence_example(ex, 1, 3)
    assert cropped.length == 2
    assert cropped.author_seq_id.shape == (2,)
    assert cropped.insertion_code.shape == (2,)
    np.testing.assert_array_equal(cropped.author_seq_id, ex.author_seq_id[1:3])
    # crop slices (no re-base); positional [0,1,2] -> [1,2] (relpos preserved).
    np.testing.assert_array_equal(cropped.residue_index, np.array([1, 2]))


def test_sequence_export_none_identity_zero_fills_to_length(tmp_path):
    # codex P1: a SequenceExample with None identity must serialize as a (L,) zero
    # column, not an empty (0,) one that breaks the residue-axis invariant.
    import dataclasses
    ex = proteon.build_sequence_example(_fake_structure("A"))
    ex = dataclasses.replace(ex, author_seq_id=None, insertion_code=None)
    out = seq_export.export_sequence_examples([ex], tmp_path / "seq")
    (back,) = seq_export.load_sequence_examples(out)
    assert back.author_seq_id.shape == (ex.length,)
    assert back.insertion_code.shape == (ex.length,)
    assert not back.author_seq_id.any() and not back.insertion_code.any()
