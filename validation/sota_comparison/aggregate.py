#!/usr/bin/env python3
"""Aggregator for the SOTA comparison harness.

Reads the per-(pdb, op, impl) JSON produced by `run_all.py`, pairs `ferritin`
against every other registered impl per op, computes per-metric agreement
values, classifies each metric as PASS / WARN / FAIL per the locked tolerance
table, and emits both a human-readable `report.md` and a machine-readable
`summary.json`.

`--strict` exits non-zero if any metric is in the FAIL band — this is the
gate a future CI job hooks into.

Usage:
    python aggregate.py results.json --output report.md --json summary.json
    python aggregate.py results.json --output report.md --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tolerance tables (locked, see plan file)
# ---------------------------------------------------------------------------

# Each entry: (metric_name, lambda(value) -> "PASS" | "WARN" | "FAIL")
# Higher-is-better metrics use ≥ thresholds; lower-is-better use ≤.

def _band(value: float, pass_at: float, warn_at: float, lower_is_better: bool = True) -> str:
    """Classify `value` against pass/warn thresholds.

    `lower_is_better=True`: PASS if value ≤ pass_at, WARN if ≤ warn_at, else FAIL.
    `lower_is_better=False`: PASS if value ≥ pass_at, WARN if ≥ warn_at, else FAIL.
    """
    if math.isnan(value):
        return "FAIL"
    if lower_is_better:
        if value <= pass_at:
            return "PASS"
        if value <= warn_at:
            return "WARN"
        return "FAIL"
    else:
        if value >= pass_at:
            return "PASS"
        if value >= warn_at:
            return "WARN"
        return "FAIL"


# ---------------------------------------------------------------------------
# Per-op comparison functions
# ---------------------------------------------------------------------------

def compare_sasa(ferritin_payload: dict, other_payload: dict) -> Dict[str, dict]:
    """Compare ferritin SASA against another impl's SASA.

    Returns:
        {
            "total_pct_diff": {"value": float, "band": "PASS"|"WARN"|"FAIL"},
            "per_residue_pearson": {...},
            "per_residue_rmsd": {...},
        }
    """
    out: Dict[str, dict] = {}

    # Total %diff (always defined when both runs succeeded)
    a_total = ferritin_payload.get("total_sasa")
    b_total = other_payload.get("total_sasa")
    if a_total is not None and b_total is not None and a_total > 0:
        pct = abs(a_total - b_total) / a_total * 100.0
    else:
        pct = float("nan")
    out["total_pct_diff"] = {
        "value": pct,
        "band": _band(pct, 1.0, 3.0, lower_is_better=True),
    }

    # Per-residue: join on (chain, resi, icode), compute Pearson r and RMSD on
    # the intersection. Missing-on-either-side residues are dropped from both
    # the numerator and denominator and reported as `n_joined`.
    a_by_key = {
        (r["chain"], int(r["resi"]), r.get("icode", "")): float(r["sasa"])
        for r in ferritin_payload.get("per_residue", [])
    }
    b_by_key = {
        (r["chain"], int(r["resi"]), r.get("icode", "")): float(r["sasa"])
        for r in other_payload.get("per_residue", [])
    }
    joint_keys = sorted(a_by_key.keys() & b_by_key.keys())
    a_vals = [a_by_key[k] for k in joint_keys]
    b_vals = [b_by_key[k] for k in joint_keys]

    n = len(joint_keys)
    if n >= 2:
        mean_a = sum(a_vals) / n
        mean_b = sum(b_vals) / n
        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(a_vals, b_vals))
        var_a = sum((a - mean_a) ** 2 for a in a_vals)
        var_b = sum((b - mean_b) ** 2 for b in b_vals)
        denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
        pearson = cov / denom if denom > 0 else float("nan")
        rmsd = math.sqrt(sum((a - b) ** 2 for a, b in zip(a_vals, b_vals)) / n)
    else:
        pearson = float("nan")
        rmsd = float("nan")

    out["per_residue_pearson"] = {
        "value": pearson,
        "band": _band(pearson, 0.999, 0.99, lower_is_better=False),
    }
    out["per_residue_rmsd"] = {
        "value": rmsd,
        "band": _band(rmsd, 2.0, 5.0, lower_is_better=True),
    }
    out["_n_joined_residues"] = {"value": n, "band": "INFO"}
    out["_n_ferritin_residues"] = {"value": len(a_by_key), "band": "INFO"}
    out["_n_other_residues"] = {"value": len(b_by_key), "band": "INFO"}
    return out


def _pct_diff(a: float, b: float) -> float:
    """Return abs(a-b)/|a| in percent, or nan if a is ~zero."""
    if a is None or b is None:
        return float("nan")
    if abs(a) < 1e-9:
        return float("nan")
    return abs(a - b) / abs(a) * 100.0


def compare_energy(ferritin_payload: dict, other_payload: dict) -> Dict[str, dict]:
    """Compare ferritin energy against another impl's energy.

    Returns total %diff (PASS ≤ 1%, WARN ≤ 5%, FAIL > 5%) and per-component
    %diff (PASS ≤ 2%, WARN ≤ 10%, FAIL > 10%).

    Handles cross-impl component-grouping differences:

    - OpenMM reports `torsion` as the sum of proper + improper (PeriodicTorsionForce
      handles both); ferritin splits them. For the comparison, we sum ferritin's
      torsion + improper_torsion and compare against OpenMM's torsion.

    - OpenMM reports `nonbonded_total` (NonbondedForce); ferritin splits into
      vdw + electrostatic. We compare the sum to OpenMM's nonbonded_total.
    """
    out: Dict[str, dict] = {}

    # Atom-count diff: a non-zero value here is the single most likely
    # explanation for energy disagreements, because it usually means the
    # two tools placed different numbers of H atoms on terminal residues
    # (different protonation states -> different charges).
    a_atoms = ferritin_payload.get("n_atoms_after_h")
    b_atoms = other_payload.get("n_atoms_after_h")
    if a_atoms is not None and b_atoms is not None and a_atoms >= 0 and b_atoms >= 0:
        out["_n_atoms_diff"] = {
            "value": a_atoms - b_atoms,
            "band": "INFO",
        }

    # Total
    out["total_pct_diff"] = {
        "value": _pct_diff(ferritin_payload.get("total"), other_payload.get("total")),
        "band": _band(
            _pct_diff(ferritin_payload.get("total"), other_payload.get("total")),
            1.0, 5.0, lower_is_better=True,
        ),
    }

    a_comp = ferritin_payload.get("components", {}) or {}
    b_comp = other_payload.get("components", {}) or {}

    # Direct per-component comparisons: bond_stretch, angle_bend
    for key in ("bond_stretch", "angle_bend"):
        a = a_comp.get(key)
        b = b_comp.get(key)
        if a is None or b is None:
            continue
        pct = _pct_diff(a, b)
        out[f"{key}_pct_diff"] = {
            "value": pct,
            "band": _band(pct, 2.0, 10.0, lower_is_better=True),
        }

    # Torsion: OpenMM PeriodicTorsionForce combines proper + improper.
    # If the "other" runner only has a combined torsion, sum ferritin's
    # torsion + improper for the comparison. Otherwise compare directly.
    a_tor = a_comp.get("torsion")
    a_imp = a_comp.get("improper_torsion")
    b_tor = b_comp.get("torsion")
    b_imp = b_comp.get("improper_torsion")
    if a_tor is not None and b_tor is not None:
        if b_imp is None and a_imp is not None:
            # Other runner has combined torsion; sum ferritin sides.
            a_sum = a_tor + a_imp
            pct = _pct_diff(a_sum, b_tor)
            out["torsion_combined_pct_diff"] = {
                "value": pct,
                "band": _band(pct, 2.0, 10.0, lower_is_better=True),
            }
        else:
            pct = _pct_diff(a_tor, b_tor)
            out["torsion_pct_diff"] = {
                "value": pct,
                "band": _band(pct, 2.0, 10.0, lower_is_better=True),
            }
            if a_imp is not None and b_imp is not None:
                pct_i = _pct_diff(a_imp, b_imp)
                out["improper_torsion_pct_diff"] = {
                    "value": pct_i,
                    "band": _band(pct_i, 2.0, 10.0, lower_is_better=True),
                }

    # Nonbonded: OpenMM reports a single nonbonded_total; ferritin splits
    # into vdw + electrostatic. Compare the sum to the other's nonbonded_total
    # if present, otherwise compare vdw and electrostatic directly.
    a_vdw = a_comp.get("vdw")
    a_elec = a_comp.get("electrostatic")
    b_nonbonded_total = other_payload.get("nonbonded_total")
    if b_nonbonded_total is not None and a_vdw is not None and a_elec is not None:
        a_sum = a_vdw + a_elec
        pct = _pct_diff(a_sum, b_nonbonded_total)
        out["nonbonded_combined_pct_diff"] = {
            "value": pct,
            "band": _band(pct, 2.0, 10.0, lower_is_better=True),
        }
    else:
        for key in ("vdw", "electrostatic"):
            a = a_comp.get(key)
            b = b_comp.get(key)
            if a is None or b is None:
                continue
            pct = _pct_diff(a, b)
            out[f"{key}_pct_diff"] = {
                "value": pct,
                "band": _band(pct, 2.0, 10.0, lower_is_better=True),
            }

    return out


COMPARATORS = {
    "sasa": compare_sasa,
    "energy": compare_energy,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def index_records(records: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Index records by (pdb_id, op, impl)."""
    return {(r["pdb_id"], r["op"], r["impl"]): r for r in records}


def aggregate(records: List[dict]) -> dict:
    """Pair ferritin against every other impl per (pdb, op) and run comparators.

    Returns a nested summary:
        {
            "by_op": {
                "sasa": {
                    "by_impl": {
                        "freesasa": {
                            "by_pdb": {
                                "1crn": {"metrics": {...}, "ferritin_status": "ok", "other_status": "ok"},
                                ...
                            },
                            "n_pdbs": int,
                            "totals": {"PASS": int, "WARN": int, "FAIL": int},
                        }
                    }
                }
            },
            "global_totals": {"PASS": int, "WARN": int, "FAIL": int, "errored_pairs": int},
        }
    """
    idx = index_records(records)
    pdbs = sorted({r["pdb_id"] for r in records if r["pdb_id"]})
    ops = sorted({r["op"] for r in records})

    by_op: Dict[str, dict] = {}
    global_totals: Dict[str, int] = defaultdict(int)

    for op in ops:
        if op not in COMPARATORS:
            continue
        compare_fn = COMPARATORS[op]
        impls = sorted({r["impl"] for r in records if r["op"] == op})
        if "ferritin" not in impls:
            continue

        op_section: Dict[str, dict] = {"by_impl": {}}
        for other in impls:
            if other == "ferritin":
                continue
            impl_section: Dict[str, dict] = {"by_pdb": {}}
            band_counts = defaultdict(int)
            for pid in pdbs:
                fer = idx.get((pid, op, "ferritin"))
                oth = idx.get((pid, op, other))
                if fer is None or oth is None:
                    continue
                if fer["status"] != "ok" or oth["status"] != "ok":
                    impl_section["by_pdb"][pid] = {
                        "ferritin_status": fer["status"],
                        "other_status": oth["status"],
                        "ferritin_error": fer.get("error"),
                        "other_error": oth.get("error"),
                        "metrics": {},
                    }
                    global_totals["errored_pairs"] += 1
                    continue
                metrics = compare_fn(fer["payload"], oth["payload"])
                impl_section["by_pdb"][pid] = {
                    "ferritin_status": "ok",
                    "other_status": "ok",
                    "metrics": metrics,
                }
                for k, v in metrics.items():
                    if k.startswith("_"):
                        continue
                    band = v.get("band", "INFO")
                    if band in ("PASS", "WARN", "FAIL"):
                        band_counts[band] += 1
                        global_totals[band] += 1
            impl_section["n_pdbs"] = len(impl_section["by_pdb"])
            impl_section["totals"] = dict(band_counts)
            op_section["by_impl"][other] = impl_section
        by_op[op] = op_section

    return {
        "by_op": by_op,
        "global_totals": dict(global_totals),
        "n_pdbs": len(pdbs),
        "n_ops": len(ops),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _band_emoji(band: str) -> str:
    return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "INFO": "·"}.get(band, "?")


def _fmt_value(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def render_markdown(summary: dict, n_records: int) -> str:
    lines = ["# SOTA Comparison Report", ""]
    totals = summary["global_totals"]
    lines.append(
        f"**{summary['n_pdbs']} PDBs, {summary['n_ops']} ops, {n_records} records.** "
        f"PASS={totals.get('PASS', 0)} WARN={totals.get('WARN', 0)} "
        f"FAIL={totals.get('FAIL', 0)} errored={totals.get('errored_pairs', 0)}"
    )
    lines.append("")

    for op, op_section in summary["by_op"].items():
        lines.append(f"## {op}")
        lines.append("")
        for other, impl_section in op_section["by_impl"].items():
            lines.append(f"### ferritin vs {other} (n={impl_section['n_pdbs']} PDBs)")
            lines.append("")
            t = impl_section["totals"]
            lines.append(
                f"Totals: PASS={t.get('PASS', 0)} WARN={t.get('WARN', 0)} "
                f"FAIL={t.get('FAIL', 0)}"
            )
            lines.append("")
            # Build a per-pdb table
            # First, collect the metric names from any successful comparison.
            metric_names: List[str] = []
            for pdb_section in impl_section["by_pdb"].values():
                for k in pdb_section.get("metrics", {}).keys():
                    if not k.startswith("_") and k not in metric_names:
                        metric_names.append(k)
            if not metric_names:
                lines.append("_No successful comparisons._")
                lines.append("")
                continue
            header = "| PDB | " + " | ".join(metric_names) + " |"
            sep = "|" + "|".join(["---"] * (len(metric_names) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
            for pid in sorted(impl_section["by_pdb"]):
                row = impl_section["by_pdb"][pid]
                metrics = row.get("metrics", {})
                if not metrics:
                    err = row.get("ferritin_error") or row.get("other_error") or "skipped"
                    cell = f"_{err}_"
                    lines.append(f"| {pid} | " + " | ".join([cell] * len(metric_names)) + " |")
                    continue
                cells = []
                for m in metric_names:
                    if m not in metrics:
                        cells.append("—")
                        continue
                    v = metrics[m]["value"]
                    band = metrics[m]["band"]
                    cells.append(f"{_band_emoji(band)} {_fmt_value(v)}")
                lines.append(f"| {pid} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="SOTA comparison aggregator")
    parser.add_argument("results_json", type=Path, help="run_all.py output JSON")
    parser.add_argument("--output", type=Path, default=Path("report.md"))
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the summary as JSON to this path")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any metric is FAIL")
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)
    records = data.get("records", [])
    summary = aggregate(records)

    md = render_markdown(summary, n_records=len(records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"Wrote {args.output}", file=sys.stderr)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.json}", file=sys.stderr)

    if args.strict and summary["global_totals"].get("FAIL", 0) > 0:
        print("STRICT: at least one metric is FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
