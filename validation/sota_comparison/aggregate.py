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
# Distribution statistics (for runs with N >> 6 PDBs)
# ---------------------------------------------------------------------------
#
# At ≥100 PDBs the per-structure table becomes unreadable, and at 10K it's
# useless. The distribution mode collapses each metric across all PDBs in
# an (op, impl) section into mean/median/p50/p90/p99/max/min/std plus a
# top-K outlier list (worst PDBs per metric, ranked by the metric's
# "wrong direction"). Trigger threshold is configurable but defaults to
# auto-switch when n_pdbs > 20.

# Default outlier list size. Captures the worst 20 PDBs per metric — small
# enough to read in a terminal, large enough to spot patterns in the tail.
_DEFAULT_OUTLIER_TOP_K = 20

# Auto-switch threshold: per-PDB table at or below this many structures,
# distribution summary above. Picked at the boundary where a markdown
# table becomes hard to scan in a terminal.
_DEFAULT_AUTO_DISTRIBUTION_THRESHOLD = 20


def _is_lower_better(metric_name: str) -> bool:
    """Whether 'better' for `metric_name` means smaller values.

    Heuristic: agreement metrics like Pearson r are higher-is-better;
    everything else (percent diffs, RMSDs) is lower-is-better. Currently
    only `*pearson*` and `*_r2*` are higher-is-better. If this list grows,
    consider an explicit registry instead.
    """
    if "pearson" in metric_name or "_r2" in metric_name:
        return False
    return True


def _percentile(sorted_values: list, p: float) -> float:
    """Linear-interpolation percentile of an already-sorted list.

    Matches numpy's default `linear` interpolation. Returns nan on empty.
    """
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(rank)
    hi = lo + 1
    if hi >= len(sorted_values):
        return float(sorted_values[-1])
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _summarize_values(values: list) -> dict:
    """Compute distribution statistics for a list of metric values.

    Drops NaN/None silently. Returns nan-filled stats when fewer than 1
    values remain after filtering.
    """
    clean = [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    n_dropped = len(values) - len(clean)
    if not clean:
        return {
            "n": 0,
            "n_dropped": n_dropped,
            "mean": float("nan"),
            "median": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    s = sorted(clean)
    n = len(s)
    mean = sum(s) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in s) / (n - 1)
        std = math.sqrt(var)
    else:
        std = float("nan")
    return {
        "n": n,
        "n_dropped": n_dropped,
        "mean": mean,
        "median": _percentile(s, 50.0),
        "p50": _percentile(s, 50.0),
        "p90": _percentile(s, 90.0),
        "p99": _percentile(s, 99.0),
        "min": float(s[0]),
        "max": float(s[-1]),
        "std": std,
    }


def _collect_distribution(impl_section: dict) -> dict:
    """Walk impl_section['by_pdb'] and compute per-metric distribution stats.

    Returns:
        {
            "metric_name": {<stats dict from _summarize_values>},
            ...
        }
    Excludes metric keys starting with '_' (those are INFO-only sidecar
    fields like _n_atoms_diff that don't have a band).
    """
    by_metric: Dict[str, list] = defaultdict(list)
    for pdb_section in impl_section["by_pdb"].values():
        metrics = pdb_section.get("metrics", {})
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            by_metric[k].append(v.get("value"))
    return {k: _summarize_values(vs) for k, vs in by_metric.items()}


def _collect_outliers(impl_section: dict, top_k: int = _DEFAULT_OUTLIER_TOP_K) -> dict:
    """For each metric, return the top-k WORST PDBs.

    "Worst" is direction-aware: for lower-is-better metrics (e.g.
    pct_diff, RMSD), the worst are the largest values; for higher-is-
    better (e.g. Pearson r), the worst are the smallest values.

    Returns:
        {
            "metric_name": [
                {"pdb_id": "1abc", "value": 167.12, "band": "FAIL"},
                ...
            ],
            ...
        }
    """
    by_metric: Dict[str, list] = defaultdict(list)
    for pdb_id, pdb_section in impl_section["by_pdb"].items():
        metrics = pdb_section.get("metrics", {})
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            value = v.get("value")
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            by_metric[k].append({
                "pdb_id": pdb_id,
                "value": float(value),
                "band": v.get("band", "INFO"),
            })

    out: Dict[str, list] = {}
    for k, entries in by_metric.items():
        reverse = _is_lower_better(k)  # lower-better → biggest values are worst
        entries.sort(key=lambda e: e["value"], reverse=reverse)
        out[k] = entries[:top_k]
    return out


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
            # Distribution stats and outliers always computed; the
            # renderer decides whether to show them based on n_pdbs.
            # Cheap to compute even at 10K (sorted-list percentiles).
            impl_section["distribution"] = _collect_distribution(impl_section)
            impl_section["outliers"] = _collect_outliers(impl_section)
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


def _render_per_pdb_table(impl_section: dict) -> List[str]:
    """Render the original per-PDB table. Used in 'table' mode (small N)."""
    lines: List[str] = []
    metric_names: List[str] = []
    for pdb_section in impl_section["by_pdb"].values():
        for k in pdb_section.get("metrics", {}).keys():
            if not k.startswith("_") and k not in metric_names:
                metric_names.append(k)
    if not metric_names:
        lines.append("_No successful comparisons._")
        lines.append("")
        return lines
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
    return lines


def _render_distribution(impl_section: dict, top_k: int) -> List[str]:
    """Render the distribution-stats table + outlier list. Large-N mode."""
    lines: List[str] = []
    distribution = impl_section.get("distribution", {})
    outliers = impl_section.get("outliers", {})
    if not distribution:
        lines.append("_No successful comparisons._")
        lines.append("")
        return lines

    # Distribution table: rows = metrics, cols = stats. Same metric order
    # the per-PDB table would have used (insertion order from by_pdb).
    metric_names: List[str] = []
    for pdb_section in impl_section["by_pdb"].values():
        for k in pdb_section.get("metrics", {}).keys():
            if not k.startswith("_") and k not in metric_names:
                metric_names.append(k)

    lines.append("**Distribution per metric** (n = number of valid pairs after filtering NaN):")
    lines.append("")
    lines.append("| metric | n | mean | median | p90 | p99 | max | std | direction |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in metric_names:
        stats = distribution.get(m)
        if stats is None:
            continue
        direction = "↓" if _is_lower_better(m) else "↑"
        lines.append(
            f"| {m} | {stats['n']} | {_fmt_value(stats['mean'])} | "
            f"{_fmt_value(stats['median'])} | {_fmt_value(stats['p90'])} | "
            f"{_fmt_value(stats['p99'])} | {_fmt_value(stats['max'])} | "
            f"{_fmt_value(stats['std'])} | {direction} |"
        )
    lines.append("")

    # Outlier lists per metric. Compact format: a small section per metric
    # with the worst K PDBs ranked by direction.
    any_outliers = any(outliers.get(m) for m in metric_names)
    if any_outliers:
        lines.append(f"**Top-{top_k} outliers per metric** (worst values first):")
        lines.append("")
        for m in metric_names:
            entries = outliers.get(m, [])
            if not entries:
                continue
            direction = "lowest" if not _is_lower_better(m) else "highest"
            lines.append(f"_{m}_ (worst = {direction})")
            lines.append("")
            lines.append("| rank | pdb | value | band |")
            lines.append("|---|---|---|---|")
            for i, e in enumerate(entries, start=1):
                lines.append(
                    f"| {i} | {e['pdb_id']} | {_fmt_value(e['value'])} | "
                    f"{_band_emoji(e['band'])} {e['band']} |"
                )
            lines.append("")
    return lines


def render_markdown(
    summary: dict,
    n_records: int,
    summary_mode: str = "auto",
    auto_threshold: int = _DEFAULT_AUTO_DISTRIBUTION_THRESHOLD,
    outlier_top_k: int = _DEFAULT_OUTLIER_TOP_K,
) -> str:
    """Render the report.

    summary_mode:
        "table"        — always use the per-PDB table (small N or v1 reports)
        "distribution" — always use distribution stats + outliers (10K runs)
        "auto"         — switch based on n_pdbs vs auto_threshold (default)
    """
    lines = ["# SOTA Comparison Report", ""]
    totals = summary["global_totals"]
    lines.append(
        f"**{summary['n_pdbs']} PDBs, {summary['n_ops']} ops, {n_records} records.** "
        f"PASS={totals.get('PASS', 0)} WARN={totals.get('WARN', 0)} "
        f"FAIL={totals.get('FAIL', 0)} errored={totals.get('errored_pairs', 0)}"
    )
    lines.append("")

    use_distribution = (
        summary_mode == "distribution"
        or (summary_mode == "auto" and summary["n_pdbs"] > auto_threshold)
    )
    if use_distribution:
        lines.append(
            f"_Mode: distribution (n_pdbs={summary['n_pdbs']} > "
            f"auto_threshold={auto_threshold} or explicit). Use "
            f"`--summary-mode table` for the per-PDB layout._"
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
            if use_distribution:
                lines.extend(_render_distribution(impl_section, outlier_top_k))
            else:
                lines.extend(_render_per_pdb_table(impl_section))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="SOTA comparison aggregator")
    parser.add_argument("results_json", type=Path, help="run_all.py output JSON")
    parser.add_argument("--output", type=Path, default=Path("report.md"))
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the summary as JSON to this path")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any metric is FAIL")
    parser.add_argument(
        "--summary-mode",
        choices=("auto", "table", "distribution"),
        default="auto",
        help=(
            "How to render per-(op, impl) sections. 'table' is the original "
            "per-PDB markdown table (good for ≤20 PDBs). 'distribution' "
            "shows mean/median/p90/p99/max/std plus a top-K outlier list "
            "(good for ≥100 PDBs). 'auto' (default) picks based on n_pdbs."
        ),
    )
    parser.add_argument(
        "--auto-threshold",
        type=int,
        default=_DEFAULT_AUTO_DISTRIBUTION_THRESHOLD,
        help=(
            "When --summary-mode=auto, switch to distribution mode if "
            f"n_pdbs > this value. Default: {_DEFAULT_AUTO_DISTRIBUTION_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--outlier-top-k",
        type=int,
        default=_DEFAULT_OUTLIER_TOP_K,
        help=(
            "How many worst-PDB outliers to list per metric in distribution "
            f"mode. Default: {_DEFAULT_OUTLIER_TOP_K}."
        ),
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)
    records = data.get("records", [])
    summary = aggregate(records)

    md = render_markdown(
        summary,
        n_records=len(records),
        summary_mode=args.summary_mode,
        auto_threshold=args.auto_threshold,
        outlier_top_k=args.outlier_top_k,
    )
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
