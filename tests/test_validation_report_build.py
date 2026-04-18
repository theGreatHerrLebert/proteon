from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_build_report_module():
    path = Path(__file__).resolve().parent.parent / "validation" / "report" / "build_report.py"
    spec = spec_from_file_location("ferritin_validation_build_report", path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rescue_summary_table_reports_rescueable_buckets():
    build_report = _load_build_report_module()
    rows = [
        {
            "pdb": "9l05.pdb",
            "status": "load_error",
            "exception": (
                "Failed to read x.pdb: InvalidatingError: Invalid data in field\n"
                "432 | SEQADV 1BGX L GB 437099 SER 31 DELETION\n"
                "The text presented is not of the right kind (isize)."
            ),
        },
        {
            "pdb": "7zdy.pdb",
            "status": "load_error",
            "exception": (
                "Failed to read x.pdb: InvalidatingError: Atom charge is not correct\n"
                "ATOM      1  N   MET W   1      -7.341  14.092 113.200  1.00 56.51           N0"
            ),
        },
        {
            "pdb": "bad.pdb",
            "status": "load_error",
            "exception": "Failed to read x.pdb: unexpected parse failure",
        },
    ]

    html = build_report.rescue_summary_table(rows)

    assert "2/3 current loader failures" in html
    assert "seqadv_invalid_field" in html
    assert "atom_charge_suffix" in html
    assert "drop_seqadv" not in html
    assert "rewrite malformed element/charge columns and retry parse" in html


def test_rescue_summary_table_handles_missing_rows():
    build_report = _load_build_report_module()

    html = build_report.rescue_summary_table([])

    assert "No loader-failure artifact found for rescue summary" in html


def test_realized_rescue_summary_table_renders_buckets(tmp_path):
    build_report = _load_build_report_module()
    summary_path = tmp_path / "rescue_benchmark_summary.json"
    summary_path.write_text(
        """{
  "candidate_failure_count": 32,
  "available_slice_count": 2,
  "rescued_count": 2,
  "rescued_bucket_counts": {
    "seqres_total_invalid": 1,
    "solitary_dbref1_definition": 1
  },
  "pipeline_ok": false
}""",
        encoding="utf-8",
    )

    html = build_report.realized_rescue_summary_table(summary_path)

    assert "2/2 available failed inputs" in html
    assert "Only 2/32 original failure files are present" in html
    assert "seqres_total_invalid" in html
    assert "solitary_dbref1_definition" in html
