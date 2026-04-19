"""Tests for top-level corpus release manifests."""

import json
from types import SimpleNamespace

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
    return SimpleNamespace(identifier="fake", chain_count=1, chains=[chain], residue_count=2, atom_count=10)


class TestCorpusRelease:
    def test_build_corpus_release_manifest(self, tmp_path):
        structures = [_fake_structure("A")]
        prep = proteon.PrepReport(hydrogens_added=2, converged=True)

        prepared_root = proteon.build_structure_supervision_dataset_from_prepared(
            structures,
            [prep],
            tmp_path / "prepared_root",
            release_id="struc-v0",
            record_ids=["fake:A"],
        )
        seq_release = proteon.build_sequence_dataset(
            structures,
            tmp_path / "seq_release",
            release_id="seq-v0",
            record_ids=["fake:A"],
        )
        train_release = proteon.build_training_release(
            seq_release,
            prepared_root / "supervision_release",
            tmp_path / "train_release",
            release_id="train-v0",
            split_assignments={"fake:A": "train"},
        )
        corpus_root = proteon.build_corpus_release_manifest(
            tmp_path / "corpus_release",
            release_id="corpus-v0",
            prepared_manifest=prepared_root / "prepared_structures.jsonl",
            sequence_release=seq_release,
            structure_release=prepared_root / "supervision_release",
            training_release=train_release,
            prep_policy_version="prep-v1",
            split_policy_version="split-v1",
            provenance={"source_manifest": "raw-v1"},
        )

        manifest = proteon.load_corpus_release_manifest(corpus_root / "corpus_release_manifest.json")
        assert manifest.release_id == "corpus-v0"
        assert manifest.count_prepared == 1
        assert manifest.count_sequence_examples == 1
        assert manifest.count_structure_examples == 1
        assert manifest.count_training_examples == 1
        assert manifest.split_counts["train"] == 1
        assert manifest.prep_policy_version == "prep-v1"
        assert manifest.split_policy_version == "split-v1"
        assert manifest.provenance["source_manifest"] == "raw-v1"

        raw = json.loads((corpus_root / "corpus_release_manifest.json").read_text(encoding="utf-8"))
        assert raw["sequence_release"].endswith("seq_release")

        report = proteon.validate_corpus_release(
            corpus_root / "corpus_release_manifest.json",
            out_path=corpus_root / "validation_report.json",
        )
        assert report.ok
        assert report.counts["training_examples"] == 1
        assert report.split_counts["train"] == 1
        assert "pseudo_beta_fraction" in report.completeness
        saved = json.loads((corpus_root / "validation_report.json").read_text(encoding="utf-8"))
        assert saved["ok"] is True

    def test_build_corpus_release_manifest_counts_rescued_inputs(self, tmp_path):
        out = tmp_path / "corpus_release"
        out.mkdir()
        rescued_path = tmp_path / "rescued_inputs.jsonl"
        rescued_path.write_text(
            '{"record_id":"rescued-a","artifact_type":"rescued_input","status":"rescued"}\n'
            '{"record_id":"rescued-b","artifact_type":"rescued_input","status":"rescued"}\n',
            encoding="utf-8",
        )

        corpus_root = proteon.build_corpus_release_manifest(
            out,
            release_id="corpus-v0",
            rescued_inputs_manifest=rescued_path,
            overwrite=True,
        )

        manifest = proteon.load_corpus_release_manifest(corpus_root / "corpus_release_manifest.json")
        assert manifest.count_rescued_inputs == 2
        assert manifest.rescued_inputs_manifest == str(rescued_path)
