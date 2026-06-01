"""Tests for the EVIDENT claim-tolerance scorer (evident/tools/claim_scoring.py).

Covers the metric engine, artifact loaders, predicate/value extraction, and
the schema validation of the ``scoring:`` block. Closes the gap flagged in
issue #85: tolerance bands were never enforced by code, only transcribed by
a human reading a report.

The metric-engine tests run on tiny in-memory fixtures so they are fast and
hermetic. A final end-to-end test scores a real release artifact when it is
present in the working tree (the large artifacts are gitignored / shipped in
the evidence bundle, so it skips in CI).
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evident" / "tools"))

import claim_scoring as cs  # noqa: E402
import validate_manifest as vm  # noqa: E402


# --------------------------------------------------------------------------- #
# load_records
# --------------------------------------------------------------------------- #

def test_load_jsonl_skips_blank_keeps_records(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
    recs = cs.load_records(p, "jsonl", None)
    assert [r["x"] for r in recs] == [1, 2]


def test_load_jsonl_malformed_line_raises(tmp_path):
    # A corrupt (non-blank) line must error, not silently shrink the denominator.
    p = tmp_path / "a.jsonl"
    p.write_text('{"x": 1}\n{bad json}\n{"x": 2}\n', encoding="utf-8")
    with pytest.raises(cs.ScoringError):
        cs.load_records(p, "jsonl", None)


def test_non_finite_value_raises():
    # NaN/inf present in evidence must not silently poison an aggregate.
    with pytest.raises(cs.ScoringError):
        cs._extract_value({"v": float("nan")}, "v")
    with pytest.raises(cs.ScoringError):
        cs._extract_value({"v": float("inf")}, "v")


def test_load_json_records_path(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"results": [{"x": 1}, {"x": 2}]}), encoding="utf-8")
    recs = cs.load_records(p, "json", "results")
    assert len(recs) == 2


def test_load_json_records_path_missing_raises(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"results": [{"x": 1}]}), encoding="utf-8")
    with pytest.raises(cs.ScoringError):
        cs.load_records(p, "json", "nope")


def test_load_unknown_format_raises(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(cs.ScoringError):
        cs.load_records(p, "csv", None)


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(cs.ScoringError):
        cs.load_records(tmp_path / "nope.jsonl", "jsonl", None)


# --------------------------------------------------------------------------- #
# value extraction
# --------------------------------------------------------------------------- #

def test_dig_nested():
    assert cs._dig({"a": {"b": 3}}, "a.b") == 3
    assert cs._dig({"a": {"b": 3}}, "a.c") is None
    assert cs._dig({"a": 1}, "a.b") is None


def test_extract_value_field_path():
    assert cs._extract_value({"rel_diff": {"vdw": 0.5}}, "rel_diff.vdw") == 0.5
    assert cs._extract_value({"x": None}, "x") is None
    assert cs._extract_value({}, "missing") is None


def test_extract_value_rel_with_floor():
    rec = {"p": 11.0, "b": 10.0}
    # |11-10| / max(10, 1) = 0.1
    assert cs._extract_value(rec, {"rel": ["p", "b"], "floor": 1.0}) == pytest.approx(0.1)
    # floor kicks in when |b| < floor: |1.5-0.5|/max(0.5,1.0) = 1.0
    assert cs._extract_value({"p": 1.5, "b": 0.5}, {"rel": ["p", "b"], "floor": 1.0}) == pytest.approx(1.0)


def test_extract_value_abs():
    assert cs._extract_value({"a": 5.0, "b": 2.0}, {"abs": ["a", "b"]}) == pytest.approx(3.0)


def test_extract_value_rel_missing_field_is_none():
    assert cs._extract_value({"p": 1.0}, {"rel": ["p", "b"]}) is None


# --------------------------------------------------------------------------- #
# predicates
# --------------------------------------------------------------------------- #

def test_predicate_present():
    assert cs._eval_predicate({"x": 1}, {"field": "x", "present": True})
    assert not cs._eval_predicate({"x": None}, {"field": "x", "present": True})
    assert not cs._eval_predicate({}, {"field": "x", "present": True})


def test_predicate_eq_and_in():
    assert cs._eval_predicate({"s": "pass"}, {"field": "s", "eq": "pass"})
    assert cs._eval_predicate({"s": "warn"}, {"field": "s", "in": ["pass", "warn"]})
    assert not cs._eval_predicate({"s": "err"}, {"field": "s", "in": ["pass", "warn"]})


def test_predicate_numeric_op_and_default_value():
    assert cs._eval_predicate({"v": 0.97}, {"field": "v", "op": ">=", "value": 0.95})
    assert not cs._eval_predicate({"v": 0.90}, {"field": "v", "op": ">=", "value": 0.95})
    # no field => test the supplied default_value (used for pass_when on value)
    assert cs._eval_predicate({}, {"op": "<", "value": 0.5}, default_value=0.1)


# --------------------------------------------------------------------------- #
# metric engine
# --------------------------------------------------------------------------- #

def _records(values, key="v"):
    return [{key: v} for v in values]


def test_median_relative_error():
    recs = _records([0.001, 0.002, 0.003])
    obs, n_in, n_tot, agg = cs.compute_metric(
        recs, {"value": "v", "aggregate": "median"}, "median_relative_error"
    )
    assert obs == pytest.approx(0.002)
    assert (n_in, n_tot, agg) == (3, 3, "median")


def test_where_filters_in_scope():
    recs = [{"v": 0.1, "ok": True}, {"v": 0.2, "ok": False}, {"v": 0.3, "ok": True}]
    obs, n_in, _, _ = cs.compute_metric(
        recs,
        {"value": "v", "where": {"field": "ok", "eq": True}, "aggregate": "median"},
        "median_relative_error",
    )
    assert obs == pytest.approx(0.2)  # median of [0.1, 0.3]
    assert n_in == 2


def test_pass_rate_fraction_on_value():
    recs = _records([0.96, 0.97, 0.5, 0.99])  # 3 of 4 >= 0.95
    obs, n_in, _, agg = cs.compute_metric(
        recs,
        {"value": "v", "pass_when": {"op": ">=", "value": 0.95}},
        "pass_rate",
    )
    assert obs == pytest.approx(0.75)
    assert agg == "fraction"


def test_pass_rate_fraction_on_presence():
    # n_ok / n_attempted: denominator is all records, numerator = field present
    recs = [{"agreement_rate": 0.9}, {"skipped": "len"}, {"agreement_rate": 0.8}]
    obs, n_in, n_tot, _ = cs.compute_metric(
        recs,
        {"pass_when": {"field": "agreement_rate", "present": True}},
        "pass_rate",
    )
    assert obs == pytest.approx(2 / 3)
    assert n_in == 3  # denominator = all (no where filter)


def test_drift_is_abs_of_signed_median():
    # signed values average out to a small positive bias; drift = |median|
    recs = _records([0.004, -0.002, 0.003, 0.005])
    obs, _, _, _ = cs.compute_metric(
        recs, {"value": "v", "aggregate": "median"}, "drift"
    )
    # median of sorted [-0.002, 0.003, 0.004, 0.005] = (0.003+0.004)/2 = 0.0035
    assert obs == pytest.approx(0.0035)


def test_drift_with_reference():
    recs = _records([1.02, 1.04])
    obs, _, _, _ = cs.compute_metric(
        recs, {"value": "v", "aggregate": "median", "reference": 1.0}, "drift"
    )
    assert obs == pytest.approx(0.03)  # |median(1.02,1.04) - 1.0|


def test_aggregate_value_requires_single_record():
    with pytest.raises(cs.ScoringError):
        cs.compute_metric(_records([0.1, 0.2]), {"value": "v", "aggregate": "value"}, "relative_error")
    obs, _, _, _ = cs.compute_metric(_records([0.1]), {"value": "v", "aggregate": "value"}, "relative_error")
    assert obs == pytest.approx(0.1)


def test_fraction_empty_denominator_raises():
    with pytest.raises(cs.ScoringError):
        cs.compute_metric(
            [{"x": 1}],
            {"value": "v", "where": {"field": "v", "present": True},
             "pass_when": {"op": ">", "value": 0}},
            "pass_rate",
        )


def test_select_flattens_nested_list():
    recs = [
        {"tests": [{"test": "sasa", "details": {"rd": 0.01}}, {"test": "dssp"}]},
        {"tests": [{"test": "sasa", "details": {"rd": 0.03}}]},
        {"tests": [{"test": "dssp"}]},  # no sasa => dropped
    ]
    scoring = {
        "select": {"list": "tests", "match": {"test": "sasa"}, "take": "details"},
        "value": "rd",
        "where": {"field": "rd", "present": True},
        "aggregate": "median",
    }
    obs, n_in, n_tot, _ = cs.compute_metric(recs, scoring, "median_relative_error")
    assert obs == pytest.approx(0.02)
    assert n_tot == 2  # two records had a sasa sub-test


# --------------------------------------------------------------------------- #
# score_tolerance / score_claim
# --------------------------------------------------------------------------- #

def _write_jsonl(tmp_path, rows):
    p = tmp_path / "art.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_score_tolerance_unscored_without_block():
    res = cs.score_tolerance({"metric": "relative_error", "op": "<", "value": 0.02}, REPO_ROOT)
    assert res.scored is False
    assert res.status == "unscored"


def test_score_tolerance_pass_and_fail(tmp_path):
    _write_jsonl(tmp_path, [{"d": 0.001}, {"d": 0.002}, {"d": 0.9}])
    tol_pass = {
        "metric": "median_relative_error", "op": "<", "value": 0.01, "output": "d",
        "scoring": {"artifact": "art.jsonl", "format": "jsonl", "value": "d",
                    "where": {"field": "d", "present": True}, "aggregate": "median"},
    }
    res = cs.score_tolerance(tol_pass, tmp_path)
    assert res.scored and res.passed and res.observed == pytest.approx(0.002)

    tol_fail = dict(tol_pass)
    tol_fail = {**tol_pass, "value": 0.0001}
    res2 = cs.score_tolerance(tol_fail, tmp_path)
    assert res2.scored and res2.passed is False


def test_score_claim_aggregates_status(tmp_path):
    # median of [0.001, 0.002, 0.9] = 0.002 < 0.01 — one outlier doesn't sink it
    _write_jsonl(tmp_path, [{"d": 0.001}, {"d": 0.002}, {"d": 0.9}])
    claim = {
        "id": "demo",
        "tolerances": [
            {"metric": "median_relative_error", "op": "<", "value": 0.01, "output": "d",
             "scoring": {"artifact": "art.jsonl", "format": "jsonl", "value": "d",
                         "where": {"field": "d", "present": True}, "aggregate": "median"}},
            {"metric": "relative_error", "op": "<", "value": 0.02},  # unscored
        ],
    }
    score = cs.score_claim(claim, tmp_path)
    assert len(score.scored) == 1
    assert score.status == "pass"
    assert score.all_passed


# --------------------------------------------------------------------------- #
# tolerance_bands helper
# --------------------------------------------------------------------------- #

def test_tolerance_bands_maps_output_to_value():
    claim = {"tolerances": [
        {"metric": "median_relative_error", "output": "vdw", "value": 0.025},
        {"metric": "median_relative_error", "output": "bond_stretch", "value": 0.01},
        {"metric": "pass_rate", "output": "vdw", "value": 0.85},
        {"metric": "median_relative_error", "value": 0.5},  # no output => skipped
    ]}
    all_bands = cs.tolerance_bands(claim)
    assert all_bands["vdw"] == 0.85  # last write wins across metrics
    median_only = cs.tolerance_bands(claim, metric="median_relative_error")
    assert median_only == {"vdw": 0.025, "bond_stretch": 0.01}


def test_first_claim():
    assert cs.first_claim({"claims": [{"id": "a"}, {"id": "b"}]})["id"] == "a"
    assert cs.first_claim({"id": "bare"})["id"] == "bare"


# --------------------------------------------------------------------------- #
# validate_manifest: scoring block schema
# --------------------------------------------------------------------------- #

def _tol(**over):
    base = {"metric": "median_relative_error", "op": "<", "value": 0.01,
            "prose": "x", "scoring": {"artifact": "a.jsonl", "format": "jsonl",
                                      "value": "d", "aggregate": "median"}}
    base.update(over)
    return base


def test_validate_scoring_accepts_well_formed():
    vm.validate_scoring(_tol(), 0, "demo")  # no raise


def test_validate_scoring_rejects_bad_format():
    tol = _tol()
    tol["scoring"] = {**tol["scoring"], "format": "parquet"}
    with pytest.raises(ValueError):
        vm.validate_scoring(tol, 0, "demo")


def test_validate_scoring_requires_pass_when_for_fraction():
    tol = {"metric": "pass_rate", "op": ">=", "value": 0.8, "prose": "x",
           "scoring": {"artifact": "a.jsonl", "format": "jsonl"}}
    with pytest.raises(ValueError):
        vm.validate_scoring(tol, 0, "demo")


def test_validate_scoring_requires_metric_op_value():
    tol = {"prose": "x", "scoring": {"artifact": "a.jsonl", "format": "jsonl"}}
    with pytest.raises(ValueError):
        vm.validate_scoring(tol, 0, "demo")


def test_real_manifest_still_validates():
    """The live manifest (now carrying scoring: blocks) passes schema check."""
    vm.validate_manifest(REPO_ROOT / "evident" / "evident.yaml")


# --------------------------------------------------------------------------- #
# end-to-end against a real artifact (skips when the artifact isn't present)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (REPO_ROOT / "validation" / "results.json").exists(),
    reason="release artifact validation/results.json not in working tree",
)
def test_end_to_end_sasa_release_claim():
    import yaml
    doc = yaml.safe_load((REPO_ROOT / "evident" / "claims" / "sasa.yaml").read_text())
    claim = next(c for c in doc["claims"] if c["id"] == "proteon-sasa-vs-biopython-release-1k-pdbs")
    score = cs.score_claim(claim, REPO_ROOT)
    # All three tolerances carry scoring blocks and are computed.
    assert len(score.scored) == 3
    biopy = next(t for t in score.scored
                 if t.metric == "median_relative_error" and t.observed < 0.01)
    assert biopy.passed  # Biopython median ~0.2% is within its 0.5% band
