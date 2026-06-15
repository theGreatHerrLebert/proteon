"""Tests for framework-neutral structure supervision artifacts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import proteon
from proteon import prepared_manifest as prep_manifest
from proteon import supervision_dataset as sup_dataset
from proteon import supervision_export as sup_export
from proteon import supervision_release as sup_release


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
        prep = proteon.PrepReport(hydrogens_added=3, hydrogens_skipped=0)
        ex = proteon.build_structure_supervision_example(
            s,
            prep_report=prep,
            record_id="fake:A",
            source_id="fake",
        )

        assert isinstance(ex, proteon.StructureSupervisionExample)
        assert ex.record_id == "fake:A"
        assert ex.chain_id == "A"
        assert ex.sequence == "GSF"
        assert ex.length == 3
        assert ex.aatype.shape == (3,)
        # residue_index is the 0-based positional sequence coordinate (was the
        # author serial_number [1,2,3]); author numbering is in author_seq_id.
        assert ex.residue_index.tolist() == [0, 1, 2]
        assert ex.author_seq_id.tolist() == [1, 2, 3]
        assert np.all(ex.seq_mask == 1.0)

    def test_atom37_and_atom14_shapes_are_present(self):
        ex = proteon.build_structure_supervision_example(_fake_structure())

        assert ex.all_atom_positions.shape == (3, 37, 3)
        assert ex.all_atom_mask.shape == (3, 37)
        assert ex.atom37_atom_exists.shape == (3, 37)
        assert ex.atom14_gt_positions.shape == (3, 14, 3)
        assert ex.atom14_gt_exists.shape == (3, 14)
        assert ex.atom14_atom_exists.shape == (3, 14)
        assert ex.residx_atom14_to_atom37.shape == (3, 14)
        assert ex.residx_atom37_to_atom14.shape == (3, 37)

    def test_pseudo_beta_uses_ca_for_gly_and_cb_for_non_gly(self):
        ex = proteon.build_structure_supervision_example(_fake_structure())

        # GLY residue 0 should use CA
        np.testing.assert_allclose(ex.pseudo_beta[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert ex.pseudo_beta_mask[0] == 1.0
        # SER residue 1 should use CB
        np.testing.assert_allclose(ex.pseudo_beta[1], np.array([3.2, 2.9, 1.8], dtype=np.float32))
        assert ex.pseudo_beta_mask[1] == 1.0

    def test_backbone_and_chi_masks_are_computed(self):
        ex = proteon.build_structure_supervision_example(_fake_structure())

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
        ex = proteon.build_structure_supervision_example(_fake_structure())

        # PHE residue is index 2 and should mark CD1/CD2 and CE1/CE2 ambiguous.
        assert ex.atom14_atom_is_ambiguous.shape == (3, 14)
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[0]) == 0
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[1]) == 0
        assert np.count_nonzero(ex.atom14_atom_is_ambiguous[2]) == 4

    def test_quality_metadata_is_attached(self):
        prep = proteon.PrepReport(
            hydrogens_added=3,
            hydrogens_skipped=1,
            atoms_reconstructed=2,
            n_unassigned_atoms=4,
        )
        ex = proteon.build_structure_supervision_example(_fake_structure(), prep_report=prep)

        assert isinstance(ex.quality, proteon.StructureQualityMetadata)
        assert ex.quality.hydrogens_added == 3
        assert ex.quality.hydrogens_skipped == 1
        assert ex.quality.atoms_reconstructed == 2
        assert ex.quality.n_unassigned_atoms == 4

    def test_rigidgroups_are_materialized(self):
        ex = proteon.build_structure_supervision_example(_fake_structure())
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
        ex = proteon.build_structure_supervision_example(_fake_structure())
        assert not ex.is_partial

    def test_batch_builder_matches_single_builder_metadata(self):
        batch = proteon.batch_build_structure_supervision_examples([_fake_structure(), _fake_structure()])
        assert len(batch) == 2
        assert batch[0].sequence == batch[1].sequence == "GSF"

    def test_export_roundtrip_preserves_examples(self, tmp_path):
        examples = proteon.batch_build_structure_supervision_examples(
            [_fake_structure("A"), _fake_structure("B")]
        )
        out_dir = sup_export.export_structure_supervision_examples(examples, tmp_path / "supervision")

        manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
        assert sup_export.SUPERVISION_EXPORT_FORMAT in manifest

        loaded = sup_export.load_structure_supervision_examples(out_dir)
        assert len(loaded) == 2
        assert loaded[0].sequence == "GSF"
        assert loaded[1].sequence == "GSF"
        np.testing.assert_array_equal(loaded[0].aatype, examples[0].aatype)
        np.testing.assert_array_equal(loaded[1].residue_index, examples[1].residue_index)
        np.testing.assert_allclose(loaded[0].rigidgroups_gt_frames, examples[0].rigidgroups_gt_frames)
        np.testing.assert_allclose(loaded[1].chi_angles, examples[1].chi_angles)

    def test_export_writes_tensor_sha256_and_load_verifies(self, tmp_path):
        """Per roadmap Section 6 — every artifact carries a checksum.

        Export must write the hex SHA-256 of the Parquet payload into
        the manifest, and load must reject a tampered payload.
        """
        import hashlib
        import json

        examples = proteon.batch_build_structure_supervision_examples(
            [_fake_structure("A")]
        )
        out_dir = sup_export.export_structure_supervision_examples(
            examples, tmp_path / "supervision"
        )
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        parquet_path = out_dir / "tensors.parquet"
        assert parquet_path.exists()
        expected_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        assert manifest["tensor_sha256"] == expected_hash
        assert manifest["tensor_file"] == "tensors.parquet"

        # Good path: default load verifies and succeeds.
        sup_export.load_structure_supervision_examples(out_dir)

        # Tamper the tensor file; default load must raise with a
        # checksum-mismatch error before any parquet parsing attempts.
        parquet_path.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum mismatch"):
            sup_export.load_structure_supervision_examples(out_dir)

    def test_release_builder_writes_manifest_and_failures(self, tmp_path):
        examples = proteon.batch_build_structure_supervision_examples([_fake_structure("A")])
        failures = [
            sup_release.FailureRecord(
                record_id="bad:1",
                failure_class="missing_required_atoms",
                message="missing CA atom",
                source_id="bad",
            )
        ]
        release_dir = sup_release.build_structure_supervision_release(
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

        loaded_failures = sup_release.load_failure_records(release_dir / "failures.jsonl")
        assert len(loaded_failures) == 1
        assert loaded_failures[0].failure_class == "missing_required_atoms"

    def test_dataset_builder_captures_failures_and_writes_release(self, tmp_path):
        release_dir = sup_dataset.build_structure_supervision_dataset(
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

        examples = sup_export.load_structure_supervision_examples(release_dir / "examples")
        assert len(examples) == 1
        assert examples[0].record_id == "good:A"

        failures = sup_release.load_failure_records(release_dir / "failures.jsonl")
        assert len(failures) == 1
        assert failures[0].record_id == "bad:Z"
        assert failures[0].failure_class == "missing_required_atoms"

    def test_prepared_bridge_writes_manifest_and_supervision_release(self, tmp_path):
        prep = proteon.PrepReport(
            atoms_reconstructed=2,
            hydrogens_added=3,
            hydrogens_skipped=1,
            minimizer_steps=7,
            converged=True,
        )
        root = sup_dataset.build_structure_supervision_dataset_from_prepared(
            [_fake_structure("A")],
            [prep],
            tmp_path / "prepared_bridge",
            release_id="prepared-v0",
            record_ids=["fake:A"],
            source_ids=["fake"],
            prep_run_ids=["prep-1"],
            provenance={"source_manifest": "prepared-demo"},
        )

        prepared_rows = prep_manifest.load_prepared_structure_manifest(root / "prepared_structures.jsonl")
        assert len(prepared_rows) == 1
        assert prepared_rows[0].record_id == "fake:A"
        assert prepared_rows[0].atoms_reconstructed == 2
        assert prepared_rows[0].hydrogens_added == 3

        examples = sup_export.load_structure_supervision_examples(root / "supervision_release" / "examples")
        assert len(examples) == 1
        assert examples[0].record_id == "fake:A"

    def test_release_builder_allows_failure_only_release(self, tmp_path):
        failures = [
            sup_release.FailureRecord(
                record_id="bad:A",
                failure_class="missing_required_atoms",
                message="missing CA atom",
                source_id="bad",
            )
        ]
        release_dir = sup_release.build_structure_supervision_release(
            [],
            tmp_path / "release_fail_only",
            release_id="fail-only-v0",
            failures=failures,
        )

        manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
        assert manifest["count_examples"] == 0
        assert manifest["count_failures"] == 1
        examples = sup_export.load_structure_supervision_examples(release_dir / "examples")
        assert examples == []


# ---------------------------------------------------------------------------
# Quality signal: relax_ok / report_present / no None-quality footgun
# (guards the two Tier-1 traps — silent minimize no-op + quality=None)
# ---------------------------------------------------------------------------

from proteon.prepare import PrepReport
from proteon.supervision import _quality_from_prep_report


def test_quality_none_report_is_explicit_unknown_not_attributeerror():
    # The observed bug: no prep report threaded -> quality was None ->
    # `.prep_success` raised/returned None. Now it is an explicit "unknown".
    q = _quality_from_prep_report(None)
    assert q is not None, "must not return None (AttributeError footgun)"
    assert q.report_present is False
    assert q.relax_ok is None  # unknown, NOT a measured failure
    assert q.protein_eligible is None
    # Reading any field must not raise.
    assert q.prep_success is False


def test_quality_relax_ok_true_when_minimized_and_converged():
    rep = PrepReport(
        skipped_no_protein=False,
        minimized=True,
        converged=True,
        minimizer_status="converged_gradient",
        minimizer_steps=42,
    )
    q = _quality_from_prep_report(rep)
    assert q.report_present is True
    assert q.protein_eligible is True
    assert q.prep_success is True  # legacy alias of protein_eligible
    assert q.relax_ok is True
    assert q.minimizer_status == "converged_gradient"


def test_quality_relax_ok_false_when_minimizer_stalled():
    # A stall (line_search_failed) ran but did not converge -> NOT relaxed.
    rep = PrepReport(
        skipped_no_protein=False,
        minimized=True,
        converged=False,
        minimizer_status="line_search_failed",
    )
    q = _quality_from_prep_report(rep)
    assert q.protein_eligible is True  # it is a protein
    assert q.relax_ok is False         # but it was not successfully relaxed
    assert q.minimizer_status == "line_search_failed"


def test_quality_non_protein_is_not_eligible():
    rep = PrepReport(skipped_no_protein=True, minimized=False, converged=False)
    q = _quality_from_prep_report(rep)
    assert q.protein_eligible is False
    assert q.prep_success is False
    assert q.relax_ok is False


# ---------------------------------------------------------------------------
# residue_index: positional sequence coordinate + author identity
# (guards the insertion-code collapse: 10/10A no longer share an index)
# ---------------------------------------------------------------------------

import os as _os
from proteon.supervision_geometry import positional_residue_index, insertion_code_ord

_ICODE_PDB = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "tests", "corpus", "insertion_codes", "icode_interleave.pdb",
)


def test_positional_residue_index_is_arange():
    np.testing.assert_array_equal(positional_residue_index(4), np.array([0, 1, 2, 3]))
    assert positional_residue_index(0).tolist() == []


def test_insertion_code_ord_encoding():
    # Reversible ASCII-ordinal encoding (0 = blank), so any single-char code
    # round-trips via chr(n) — including non-A-Z codes.
    assert insertion_code_ord(None) == 0
    assert insertion_code_ord("") == 0
    assert insertion_code_ord("A") == ord("A")
    assert insertion_code_ord("B") == ord("B")
    assert insertion_code_ord("1") == ord("1")  # numeric code preserved, not 0
    assert chr(insertion_code_ord("A")) == "A"


def test_residue_index_positional_no_collapse_on_insertion_codes():
    # The fixture interleaves SER 3 and VAL 3A (both serial_number=3). The old
    # serial-number residue_index produced a duplicate [1,2,3,3,4]; positional is
    # strictly increasing and distinct.
    ex = proteon.build_structure_supervision_example(proteon.load(_ICODE_PDB), record_id="icode:A")
    ri = ex.residue_index.tolist()
    assert ri == [0, 1, 2, 3, 4], ri
    assert len(set(ri)) == len(ri), "residue_index must be distinct (no icode collapse)"
    # Author identity preserved separately.
    assert ex.author_seq_id.tolist() == [1, 2, 3, 3, 4]
    assert ex.insertion_code.tolist() == [0, 0, 0, ord("A"), 0]  # VAL 3A -> ord('A')


def test_sequence_and_structure_residue_index_agree():
    # claudex #3: one shared index policy across both builders.
    s = proteon.load(_ICODE_PDB)
    struct = proteon.build_structure_supervision_example(s, record_id="icode:A")
    seq = proteon.build_sequence_example(proteon.load(_ICODE_PDB), record_id="icode:A")
    np.testing.assert_array_equal(struct.residue_index, seq.residue_index)


def test_author_identity_round_trips_through_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    ex = proteon.build_structure_supervision_example(proteon.load(_ICODE_PDB), record_id="icode:A")
    out = tmp_path / "examples"
    sup_export.export_structure_supervision_examples([ex], out)
    (back,) = sup_export.load_structure_supervision_examples(out)
    np.testing.assert_array_equal(back.residue_index, ex.residue_index)
    np.testing.assert_array_equal(back.author_seq_id, ex.author_seq_id)
    np.testing.assert_array_equal(back.insertion_code, ex.insertion_code)


def test_field_or_zeros_rejects_missing_required_field_but_fills_identity():
    # codex P1: the export zero-fill fallback must apply ONLY to the optional
    # identity fields, never fabricate a required supervision label from None.
    from types import SimpleNamespace
    from proteon.supervision_export import _field_or_zeros
    # A real example always HAS the field (dataclass attr), possibly None.
    ex = SimpleNamespace(length=3, author_seq_id=None, all_atom_positions=None)
    # Optional identity field None -> zero-filled, shape (L,).
    z = _field_or_zeros(ex, "author_seq_id", (), np.int32)
    assert z.shape == (3,) and not z.any()
    # Required label field None -> raises (no silent zero label).
    with pytest.raises(ValueError, match="refusing to fabricate"):
        _field_or_zeros(ex, "all_atom_positions", (37, 3), np.float32)


def test_v1_artifact_is_rejected_with_clear_error():
    # codex P1: loading a pre-v2 artifact must fail loudly (the columns / residue_index
    # semantics differ), not KeyError on a missing column.
    from proteon.supervision_export import require_supervision_schema_version
    with pytest.raises(ValueError, match="schema_version 1 is not supported"):
        require_supervision_schema_version({"schema_version": 1})
    # Current version passes.
    require_supervision_schema_version(
        {"schema_version": sup_export.SUPERVISION_PARQUET_SCHEMA_VERSION}
    )
