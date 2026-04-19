"""Tests for rescue-oriented loader failure bucketing."""

from proteon.loader_failure_analysis import (
    ATOM_CHARGE_SUFFIX,
    MODEL_ATOM_MISMATCH,
    SEQADV_INVALID_FIELD,
    SEQRES_MULTIPLE_RESIDUES,
    SOLITARY_DBREF1,
    SSBOND_MISSING_PARTNER,
    bucket_loader_failure,
    summarize_loader_failures,
)


def test_bucket_seqadv_invalid_field():
    exc = (
        "Failed to read x.pdb: InvalidatingError: Invalid data in field\n"
        "432 | SEQADV 1BGX L GB 437099 SER 31 DELETION\n"
        "The text presented is not of the right kind (isize)."
    )
    assert bucket_loader_failure(exc) == SEQADV_INVALID_FIELD


def test_bucket_atom_charge_suffix():
    exc = (
        "Failed to read x.pdb: InvalidatingError: Atom charge is not correct\n"
        "ATOM      1  N   MET W   1      -7.341  14.092 113.200  1.00 56.51           N0"
    )
    assert bucket_loader_failure(exc) == ATOM_CHARGE_SUFFIX


def test_bucket_solitary_dbref1():
    exc = "Failed to read x.pdb: StrictWarning: Solitary DBREF1 definition"
    assert bucket_loader_failure(exc) == SOLITARY_DBREF1


def test_bucket_seqres_multiple_residues():
    exc = "Failed to read x.pdb: StrictWarning: Multiple residues in SEQRES validation"
    assert bucket_loader_failure(exc) == SEQRES_MULTIPLE_RESIDUES


def test_bucket_model_atom_mismatch():
    exc = "Failed to read x.pdb: StrictWarning: Atoms in Models not corresponding"
    assert bucket_loader_failure(exc) == MODEL_ATOM_MISMATCH


def test_bucket_ssbond_missing_partner():
    exc = (
        "Failed to read x.pdb: InvalidatingError: Could not find a bond partner\n"
        "SSBOND   1 MET A   35    MET A   35"
    )
    assert bucket_loader_failure(exc) == SSBOND_MISSING_PARTNER


def test_summarize_loader_failures_aggregates_counts():
    rows = [
        {"pdb": "a.pdb", "status": "load_error", "exception": "Atom charge is not correct"},
        {"pdb": "b.pdb", "status": "load_error", "exception": "Atom charge is not correct"},
        {"pdb": "c.pdb", "status": "load_error", "exception": "Solitary DBREF1 definition"},
    ]
    summaries = summarize_loader_failures(rows)
    assert summaries[0].bucket.code == "atom_charge_suffix"
    assert summaries[0].count == 2
    assert summaries[1].bucket.code == "solitary_dbref1_definition"
    assert summaries[1].count == 1
