#!/usr/bin/env python3
"""Generate an HTML validation report from JSON results.

Usage:
    python validation/generate_report.py [--input validation/results.json] [--output validation/report.html]
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np


def generate_html(results_file: str, output_file: str):
    with open(results_file) as f:
        data = json.load(f)

    n = data["n_structures"]
    elapsed = data["elapsed_s"]
    results = data["results"]

    # Aggregate stats per test
    test_stats = defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0, "error": 0})
    sasa_diffs = []
    sasa_speedups = []
    sasa_fe_times = []
    sasa_bp_times = []
    sasa_n_atoms = []
    failures = []
    warnings = []

    for r in results:
        for t in r["tests"]:
            test_stats[t["test"]][t["status"]] += 1
            if t["status"] == "fail":
                failures.append((r["file"], t["test"], t.get("details", {}).get("error", "?")))
            if t["status"] == "warn":
                warnings.append((r["file"], t["test"], t.get("details", {}).get("warning", "?")))
            if t["test"] == "sasa":
                d = t.get("details", {})
                if "relative_diff" in d:
                    sasa_diffs.append(d["relative_diff"])
                if "speedup" in d:
                    sasa_speedups.append(d["speedup"])
                if "proteon_time_ms" in d:
                    sasa_fe_times.append(d["proteon_time_ms"])
                if "biopython_time_ms" in d:
                    sasa_bp_times.append(d["biopython_time_ms"])
                if "n_atoms" in d:
                    sasa_n_atoms.append(d["n_atoms"])

    sasa_arr = np.array(sasa_diffs) if sasa_diffs else np.array([])
    speedup_arr = np.array(sasa_speedups) if sasa_speedups else np.array([])

    # Compute overall pass rate
    total_tests = sum(sum(v.values()) for v in test_stats.values())
    total_pass = sum(v["pass"] for v in test_stats.values())
    overall_pct = total_pass / total_tests * 100 if total_tests > 0 else 0

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Proteon Validation Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
           max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ border-bottom: 2px solid #e1e4e8; padding-bottom: 10px; }}
    h2 {{ color: #24292e; margin-top: 30px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
    .summary-card {{ background: #f6f8fa; border-radius: 8px; padding: 20px; text-align: center; }}
    .summary-card .number {{ font-size: 36px; font-weight: bold; }}
    .summary-card .label {{ color: #586069; margin-top: 5px; }}
    .pass {{ color: #22863a; }}
    .warn {{ color: #b08800; }}
    .fail {{ color: #cb2431; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #e1e4e8; padding: 8px 12px; text-align: left; }}
    th {{ background: #f6f8fa; font-weight: 600; }}
    tr:nth-child(even) {{ background: #fafbfc; }}
    .bar {{ height: 20px; border-radius: 3px; display: inline-block; }}
    .bar-pass {{ background: #28a745; }}
    .bar-warn {{ background: #ffd33d; }}
    .bar-fail {{ background: #cb2431; }}
    .pct {{ font-weight: bold; margin-left: 8px; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 90%; }}
    .timestamp {{ color: #586069; font-size: 14px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px;
              font-weight: bold; color: white; }}
    .badge-pass {{ background: #28a745; }}
    .badge-warn {{ background: #ffd33d; color: #333; }}
    .badge-fail {{ background: #cb2431; }}
</style>
</head>
<body>

<h1>Proteon Validation Report</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
   Structures: {n} | Time: {elapsed:.1f}s</p>

<div class="summary-grid">
    <div class="summary-card">
        <div class="number">{n}</div>
        <div class="label">Structures Tested</div>
    </div>
    <div class="summary-card">
        <div class="number class="pass">{overall_pct:.1f}%</div>
        <div class="label">Overall Pass Rate</div>
    </div>
    <div class="summary-card">
        <div class="number">{len(failures)}</div>
        <div class="label">Failures</div>
    </div>
    <div class="summary-card">
        <div class="number">{len(warnings)}</div>
        <div class="label">Warnings</div>
    </div>
</div>

<h2>Test Results by Category</h2>
<table>
<tr>
    <th>Test</th>
    <th>Pass</th>
    <th>Warn</th>
    <th>Fail</th>
    <th>Error</th>
    <th>Pass Rate</th>
    <th>Visual</th>
</tr>
"""

    for test_name in sorted(test_stats.keys()):
        s = test_stats[test_name]
        total = sum(s.values())
        pct = s["pass"] / total * 100 if total > 0 else 0
        w_pct = s["warn"] / total * 100 if total > 0 else 0
        f_pct = s["fail"] / total * 100 if total > 0 else 0

        color = "pass" if pct > 95 else ("warn" if pct > 80 else "fail")
        html += f"""<tr>
    <td><code>{test_name}</code></td>
    <td>{s['pass']}</td>
    <td>{s['warn']}</td>
    <td>{s['fail']}</td>
    <td>{s['error']}</td>
    <td><span class="{color} pct">{pct:.1f}%</span></td>
    <td>
        <span class="bar bar-pass" style="width:{pct*2}px"></span>
        <span class="bar bar-warn" style="width:{w_pct*2}px"></span>
        <span class="bar bar-fail" style="width:{f_pct*2}px"></span>
    </td>
</tr>
"""

    html += "</table>\n"

    # SASA oracle section
    if len(sasa_arr) > 0:
        html += f"""
<h2>SASA Oracle Validation (vs Biopython)</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Structures compared</td><td>{len(sasa_arr)}</td></tr>
<tr><td>Median relative difference</td><td>{np.median(sasa_arr)*100:.2f}%</td></tr>
<tr><td>Mean relative difference</td><td>{np.mean(sasa_arr)*100:.2f}%</td></tr>
<tr><td>Max relative difference</td><td>{np.max(sasa_arr)*100:.2f}%</td></tr>
<tr><td>Within 1%</td><td>{np.sum(sasa_arr < 0.01)}/{len(sasa_arr)} ({np.sum(sasa_arr < 0.01)/len(sasa_arr)*100:.1f}%)</td></tr>
<tr><td>Within 5%</td><td>{np.sum(sasa_arr < 0.05)}/{len(sasa_arr)} ({np.sum(sasa_arr < 0.05)/len(sasa_arr)*100:.1f}%)</td></tr>
<tr><td>Within 10%</td><td>{np.sum(sasa_arr < 0.10)}/{len(sasa_arr)} ({np.sum(sasa_arr < 0.10)/len(sasa_arr)*100:.1f}%)</td></tr>
</table>
"""

    if len(speedup_arr) > 0:
        html += f"""
<h2>SASA Speed: Proteon vs Biopython</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Structures benchmarked</td><td>{len(speedup_arr)}</td></tr>
<tr><td>Median speedup</td><td><strong>{np.median(speedup_arr):.1f}x faster</strong></td></tr>
<tr><td>Mean speedup</td><td>{np.mean(speedup_arr):.1f}x faster</td></tr>
<tr><td>Min speedup</td><td>{np.min(speedup_arr):.1f}x</td></tr>
<tr><td>Max speedup</td><td>{np.max(speedup_arr):.1f}x</td></tr>
<tr><td>Total proteon time</td><td>{sum(sasa_fe_times)/1000:.1f}s</td></tr>
<tr><td>Total biopython time</td><td>{sum(sasa_bp_times)/1000:.1f}s</td></tr>
</table>
"""

    # Failures section
    if failures:
        html += f"""
<h2>Failures ({len(failures)})</h2>
<table>
<tr><th>File</th><th>Test</th><th>Error</th></tr>
"""
        for f_file, f_test, f_err in failures[:100]:
            html += f'<tr><td><code>{f_file}</code></td><td>{f_test}</td><td>{f_err[:80]}</td></tr>\n'
        html += "</table>\n"

    # Warnings section
    if warnings:
        html += f"""
<h2>Warnings ({len(warnings)})</h2>
<table>
<tr><th>File</th><th>Test</th><th>Warning</th></tr>
"""
        for w_file, w_test, w_msg in warnings[:100]:
            html += f'<tr><td><code>{w_file}</code></td><td>{w_test}</td><td>{w_msg[:80]}</td></tr>\n'
        html += "</table>\n"

    html += """
<h2>Methodology</h2>
<p>This report validates proteon's core functionality against established tools:</p>
<ul>
    <li><strong>Loading</strong>: Structure loads successfully, atom counts match Biopython and Gemmi</li>
    <li><strong>SASA</strong>: Shrake-Rupley total SASA within 5% of Biopython</li>
    <li><strong>Dihedrals</strong>: phi/psi/omega in valid ranges, >90% trans peptide bonds</li>
    <li><strong>DSSP</strong>: Valid SS characters, structured regions detected for proteins >30 residues</li>
    <li><strong>H-bonds</strong>: Negative energies, reasonable count</li>
    <li><strong>Contact map</strong>: Symmetric, diagonal True, adjacent CA in contact</li>
    <li><strong>Energy</strong>: Positive bond/angle energies, no NaN</li>
    <li><strong>Selection</strong>: 'all' matches atom_count, backbone/CA ratio ~4</li>
</ul>

<p><em>Generated by <a href="https://github.com/theGreatHerrLebert/proteon">Proteon</a> validation suite.</em></p>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html)
    print(f"Report written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="validation/results.json")
    parser.add_argument("--output", default="validation/report.html")
    args = parser.parse_args()
    generate_html(args.input, args.output)
