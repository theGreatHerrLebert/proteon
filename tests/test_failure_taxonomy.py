"""Tests for the canonical failure taxonomy (roadmap Section 7).

The taxonomy is load-bearing for dataset quality tracking — if
`missing_required_atoms` silently drifts into `internal_pipeline_error`
over time, release diffs stop meaning what they used to mean. These
tests pin down (a) the closed class list, (b) the exception→class
classifier's discriminating cases, (c) FailureRecord's rejection of
non-canonical labels.
"""

import pytest

import proteon
from proteon import supervision_release as sup_release
from proteon.failure_taxonomy import (
    ALL_FAILURE_CLASSES,
    INTERNAL_PIPELINE_ERROR,
    MINIMIZATION_NONCONVERGENCE,
    MISSING_REQUIRED_ATOMS,
    NUMERICAL_INSTABILITY,
    PARSE_ERROR,
    RESIDUE_MAPPING_ERROR,
    UNSUPPORTED_CHEMISTRY,
    classify_exception,
)


class TestCanonicalList:
    def test_class_list_is_exactly_the_roadmap_ten(self):
        assert set(ALL_FAILURE_CLASSES) == {
            "parse_error",
            "unsupported_chemistry",
            "missing_required_atoms",
            "residue_mapping_error",
            "hydrogen_placement_error",
            "forcefield_parameterization_error",
            "minimization_nonconvergence",
            "numerical_instability",
            "postprep_quality_failure",
            "internal_pipeline_error",
        }

    def test_constants_surface_at_package_level(self):
        assert proteon.ALL_FAILURE_CLASSES == ALL_FAILURE_CLASSES


class TestClassify:
    def test_requires_a_protein_chain_maps_to_missing_required_atoms(self):
        # Existing production emitter from batch_build_sequence_examples.
        assert (
            classify_exception(ValueError("chain requires a protein chain"))
            == MISSING_REQUIRED_ATOMS
        )

    def test_missing_ca_atom_maps_to_missing_required_atoms(self):
        assert (
            classify_exception(RuntimeError("missing CA atom at residue 42"))
            == MISSING_REQUIRED_ATOMS
        )

    def test_unknown_residue_name_maps_to_residue_mapping_error(self):
        assert (
            classify_exception(ValueError("unknown residue 'XYZ'"))
            == RESIDUE_MAPPING_ERROR
        )

    def test_keyerror_routes_to_residue_mapping_error(self):
        # KeyErrors in AA-lookup tables are one of the most common
        # production cases; they should not land in the internal bucket.
        assert classify_exception(KeyError("UNK")) == RESIDUE_MAPPING_ERROR

    def test_parse_error_signatures(self):
        assert (
            classify_exception(RuntimeError("malformed mmcif block"))
            == PARSE_ERROR
        )
        assert (
            classify_exception(ValueError("failed to parse CIF header"))
            == PARSE_ERROR
        )

    def test_minimization_nonconvergence(self):
        assert (
            classify_exception(RuntimeError("minimization did not converge after 5000 steps"))
            == MINIMIZATION_NONCONVERGENCE
        )

    def test_numerical_instability_detects_floating_point_error(self):
        assert (
            classify_exception(FloatingPointError("overflow in matmul"))
            == NUMERICAL_INSTABILITY
        )

    def test_numerical_instability_detects_nan_message(self):
        assert (
            classify_exception(ValueError("NaN in forces"))
            == NUMERICAL_INSTABILITY
        )

    def test_unsupported_chemistry(self):
        assert (
            classify_exception(ValueError("unsupported modified residue SEP"))
            == UNSUPPORTED_CHEMISTRY
        )

    def test_unknown_shape_falls_back_to_internal(self):
        # Conservative default: if we can't identify the failure mode
        # we bucket it as internal rather than misclassify. Skewing a
        # release report is worse than owning an "unknown" bucket.
        assert (
            classify_exception(RuntimeError("something weird happened"))
            == INTERNAL_PIPELINE_ERROR
        )


class TestFailureRecordValidation:
    def test_canonical_class_accepted(self):
        sup_release.FailureRecord(
            record_id="x",
            failure_class=MISSING_REQUIRED_ATOMS,
            message="",
        )

    def test_noncanonical_class_rejected(self):
        with pytest.raises(ValueError, match="not in the canonical"):
            sup_release.FailureRecord(
                record_id="x",
                failure_class="kinda_sorta_broken",
                message="",
            )
