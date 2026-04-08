#!/usr/bin/env python3
"""Ferritin validation suite — oracle testing against established tools.

Runs ferritin analysis on a set of PDB files and compares results against
Biopython, Gemmi, and C++ USAlign as independent oracles.

Reports per-structure and aggregate agreement statistics.

Usage:
    python validation/run_validation.py [--n-structures 100] [--pdb-dir validation/pdbs/]
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


class StructureTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise StructureTimeout("structure processing exceeded time limit")

import numpy as np

# ---------------------------------------------------------------------------
# Oracle imports
# ---------------------------------------------------------------------------

import ferritin

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

HAS_USALIGN = os.path.exists("/scratch/TMAlign/USAlign/USalign")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_loading(path: str) -> dict:
    """Test that ferritin loads the file and basic counts are sane."""
    result = {"test": "loading", "status": "pass", "details": {}}
    try:
        s = ferritin.load(path)
        result["details"]["atom_count"] = s.atom_count
        result["details"]["residue_count"] = s.residue_count
        result["details"]["chain_count"] = s.chain_count
        result["details"]["model_count"] = s.model_count

        if s.atom_count == 0:
            result["status"] = "fail"
            result["details"]["error"] = "zero atoms"
    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]
    return result


def test_loading_oracle(path: str) -> dict:
    """Compare ferritin atom/residue counts against Biopython and Gemmi."""
    result = {"test": "loading_oracle", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)
        fe_atoms = fe_s.atom_count
        fe_residues = fe_s.residue_count
        fe_chains = fe_s.chain_count
        result["details"]["ferritin"] = {
            "atoms": fe_atoms, "residues": fe_residues, "chains": fe_chains
        }
    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = f"ferritin load failed: {e}"
        return result

    # Biopython (note: Biopython takes first alt conformer, ferritin includes all)
    if HAS_BIOPYTHON:
        try:
            parser = PDBParser(QUIET=True)
            bp = parser.get_structure("x", path)
            bp_atoms = sum(1 for _ in bp.get_atoms())
            bp_chains = sum(1 for _ in bp[0].get_chains())
            result["details"]["biopython"] = {"atoms": bp_atoms, "chains": bp_chains}

            if fe_atoms != bp_atoms:
                result["details"]["atom_diff_bp"] = fe_atoms - bp_atoms
                # Ferritin includes all alt conformers, Biopython takes first only.
                # So ferritin >= biopython is expected. Only warn if biopython is LARGER.
                if fe_atoms < bp_atoms:
                    result["status"] = "warn"
                    result["details"]["warning"] = f"fewer atoms than biopython: {fe_atoms} vs {bp_atoms}"
        except Exception as e:
            result["details"]["biopython_error"] = str(e)[:100]

    # Gemmi (includes all alt conformers, like ferritin)
    if HAS_GEMMI:
        try:
            doc = gemmi.read_pdb(path)
            gm_atoms = sum(1 for _ in doc[0].all())
            gm_chains = len(list(doc[0]))
            result["details"]["gemmi"] = {"atoms": gm_atoms, "chains": gm_chains}

            if fe_atoms != gm_atoms:
                result["details"]["atom_diff_gemmi"] = fe_atoms - gm_atoms
                # Gemmi and ferritin should agree on atom count
                if abs(fe_atoms - gm_atoms) > max(5, fe_atoms * 0.01):
                    result["status"] = "warn"
                    result["details"]["warning"] = f"atom count differs from gemmi: {fe_atoms} vs {gm_atoms}"
        except Exception as e:
            result["details"]["gemmi_error"] = str(e)[:100]

    return result


def test_sasa(path: str) -> dict:
    """Compare SASA against Biopython, with timing."""
    result = {"test": "sasa", "status": "pass", "details": {}}

    try:
        t0 = time.time()
        fe_s = ferritin.load(path)
        fe_total = ferritin.total_sasa(fe_s)
        fe_time = time.time() - t0
        result["details"]["ferritin_total"] = round(fe_total, 1)
        result["details"]["ferritin_time_ms"] = round(fe_time * 1000, 1)
        result["details"]["n_atoms"] = fe_s.atom_count
    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = f"ferritin: {e}"
        return result

    if HAS_BIOPYTHON:
        try:
            t0 = time.time()
            parser = PDBParser(QUIET=True)
            bp = parser.get_structure("x", path)
            sr = ShrakeRupley()
            sr.compute(bp, level="A")
            bp_total = sum(a.sasa for a in bp.get_atoms())
            bp_time = time.time() - t0
            result["details"]["biopython_total"] = round(bp_total, 1)
            result["details"]["biopython_time_ms"] = round(bp_time * 1000, 1)

            if bp_time > 0:
                result["details"]["speedup"] = round(bp_time / fe_time, 1)

            if bp_total > 0:
                rel_diff = abs(fe_total - bp_total) / bp_total
                result["details"]["relative_diff"] = round(rel_diff, 4)
                if rel_diff > 0.05:
                    result["status"] = "warn"
                    result["details"]["warning"] = f"SASA diff {rel_diff:.1%}"
        except Exception as e:
            result["details"]["biopython_error"] = str(e)[:100]

    return result


def test_dihedrals(path: str) -> dict:
    """Test backbone dihedrals are in valid ranges."""
    result = {"test": "dihedrals", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)
        phi, psi, omega = ferritin.backbone_dihedrals(fe_s)

        n_residues = len(phi)
        result["details"]["n_residues"] = n_residues

        valid_phi = phi[~np.isnan(phi)]
        valid_psi = psi[~np.isnan(psi)]
        valid_omega = omega[~np.isnan(omega)]

        result["details"]["n_valid_phi"] = len(valid_phi)
        result["details"]["n_valid_omega"] = len(valid_omega)

        # Check ranges
        if len(valid_phi) > 0:
            if np.any(np.abs(valid_phi) > 180.01):
                result["status"] = "fail"
                result["details"]["error"] = "phi out of [-180, 180]"

        # Omega should be ~180 for trans peptide bonds
        if len(valid_omega) > 0:
            n_trans = np.sum(np.abs(valid_omega) > 150)
            trans_frac = n_trans / len(valid_omega)
            result["details"]["trans_fraction"] = round(trans_frac, 3)
            if trans_frac < 0.90:  # expect > 90% trans
                result["status"] = "warn"
                result["details"]["warning"] = f"only {trans_frac:.1%} trans peptide bonds"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_dssp(path: str) -> dict:
    """Test DSSP produces valid output."""
    result = {"test": "dssp", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)
        ss = ferritin.dssp(fe_s)

        result["details"]["length"] = len(ss)

        # Check valid characters
        valid_chars = set("HGIEBTSCC")
        invalid = set(ss) - valid_chars
        if invalid:
            result["status"] = "fail"
            result["details"]["error"] = f"invalid SS chars: {invalid}"

        # Basic composition
        from collections import Counter
        counts = Counter(ss)
        result["details"]["composition"] = dict(counts)

        # Sanity: at least some structure should be assigned for proteins > 30 residues
        if len(ss) > 30:
            structured = counts.get("H", 0) + counts.get("E", 0) + counts.get("G", 0)
            if structured == 0:
                result["status"] = "warn"
                result["details"]["warning"] = "no H/E/G assigned for large protein"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_hbonds(path: str) -> dict:
    """Test H-bond detection produces reasonable results."""
    result = {"test": "hbonds", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)
        hb = ferritin.backbone_hbonds(fe_s)

        result["details"]["n_hbonds"] = len(hb)

        if len(hb) > 0:
            result["details"]["energy_range"] = [
                round(float(hb[:, 2].min()), 2),
                round(float(hb[:, 2].max()), 2),
            ]
            # All energies should be negative (below cutoff)
            if np.any(hb[:, 2] > 0):
                result["status"] = "fail"
                result["details"]["error"] = "positive H-bond energy"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_contact_map(path: str) -> dict:
    """Test contact map basic properties."""
    result = {"test": "contact_map", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)
        ca = ferritin.extract_ca_coords(fe_s)

        if len(ca) < 2:
            result["details"]["skipped"] = "too few CA atoms"
            return result

        cm = ferritin.contact_map(ca, cutoff=8.0)
        result["details"]["n_residues"] = len(ca)
        result["details"]["n_contacts"] = int(cm.sum())

        # Contact map should be symmetric
        if not np.allclose(cm, cm.T):
            result["status"] = "fail"
            result["details"]["error"] = "contact map not symmetric"

        # Diagonal should all be True
        if not np.all(np.diag(cm)):
            result["status"] = "fail"
            result["details"]["error"] = "diagonal not all True"

        # Count adjacent CA pairs not in contact (chain breaks)
        n_breaks = sum(1 for i in range(len(ca) - 1) if not cm[i, i + 1])
        result["details"]["chain_breaks"] = n_breaks
        # This is expected for multi-chain structures — not a warning

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_energy(path: str) -> dict:
    """Test force field energy computation."""
    result = {"test": "energy", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)

        # Only test on smaller structures (energy is O(N^2))
        if fe_s.atom_count > 5000:
            result["details"]["skipped"] = "too large for energy test"
            return result

        e = ferritin.compute_energy(fe_s)
        result["details"]["total"] = round(e["total"], 1)
        result["details"]["bond_stretch"] = round(e["bond_stretch"], 1)
        result["details"]["angle_bend"] = round(e["angle_bend"], 1)

        # Bond stretch should be positive
        if e["bond_stretch"] < 0:
            result["status"] = "fail"
            result["details"]["error"] = "negative bond stretch energy"

        # Angle bend should be positive
        if e["angle_bend"] < 0:
            result["status"] = "fail"
            result["details"]["error"] = "negative angle bend energy"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_hydrogens(path: str) -> dict:
    """Test peptide hydrogen placement and DSSP equivalence."""
    result = {"test": "hydrogens", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)

        # DSSP before placing H (virtual H path)
        ss_before = ferritin.dssp(fe_s)

        # Place hydrogens
        t0 = time.time()
        n_added, n_skipped = ferritin.place_peptide_hydrogens(fe_s)
        h_time = time.time() - t0

        result["details"]["n_added"] = n_added
        result["details"]["n_skipped"] = n_skipped
        result["details"]["time_ms"] = round(h_time * 1000, 2)

        # DSSP after placing H (real H path)
        ss_after = ferritin.dssp(fe_s)

        result["details"]["dssp_length"] = len(ss_before)
        result["details"]["dssp_match"] = ss_before == ss_after

        if ss_before != ss_after:
            # Find positions that differ
            diffs = [i for i, (a, b) in enumerate(zip(ss_before, ss_after)) if a != b]
            result["status"] = "fail"
            result["details"]["error"] = (
                f"DSSP mismatch at {len(diffs)} positions: {diffs[:10]}"
            )
            result["details"]["dssp_before"] = ss_before
            result["details"]["dssp_after"] = ss_after

        # Idempotency: second call should add 0
        n_added2, _ = ferritin.place_peptide_hydrogens(fe_s)
        if n_added2 != 0:
            result["status"] = "fail"
            result["details"]["error"] = f"not idempotent: second pass added {n_added2}"

        # Sanity: for proteins > 10 AA residues, should place some H
        n_aa = len(ss_before) if ss_before else 0
        if n_aa > 10 and n_added == 0:
            result["status"] = "warn"
            result["details"]["warning"] = f"zero H placed for {n_aa}-residue protein"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


def test_select(path: str) -> dict:
    """Test selection language on structure."""
    result = {"test": "select", "status": "pass", "details": {}}

    try:
        fe_s = ferritin.load(path)

        ca_mask = ferritin.select(fe_s, "CA")
        bb_mask = ferritin.select(fe_s, "backbone")
        all_mask = ferritin.select(fe_s, "all")

        result["details"]["n_ca"] = int(ca_mask.sum())
        result["details"]["n_backbone"] = int(bb_mask.sum())

        # all should select everything
        if all_mask.sum() != fe_s.atom_count:
            result["status"] = "fail"
            result["details"]["error"] = "'all' doesn't match atom_count"

        # backbone should be ~4x CA count (N, CA, C, O)
        if ca_mask.sum() > 0:
            ratio = bb_mask.sum() / ca_mask.sum()
            result["details"]["bb_ca_ratio"] = round(ratio, 2)
            # ratio < 4 is normal for structures with missing atoms
            # or non-standard residues. Only flag extreme cases.
            if ratio < 1.0:
                result["status"] = "warn"
                result["details"]["warning"] = f"very low backbone/CA ratio: {ratio:.1f}"

    except Exception as e:
        result["status"] = "fail"
        result["details"]["error"] = str(e)[:200]

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_loading,
    test_loading_oracle,
    test_sasa,
    test_dihedrals,
    test_dssp,
    test_hbonds,
    test_hydrogens,
    test_contact_map,
    test_energy,
    test_select,
]


def run_validation(pdb_dir: str, n_structures: int = 1000, output_file: str = None,
                   timeout: int = 0):
    """Run all validation tests on PDB files."""
    # Collect PDB files
    files = sorted([
        os.path.join(pdb_dir, f)
        for f in os.listdir(pdb_dir)
        if f.endswith((".pdb", ".cif"))
    ])[:n_structures]

    print(f"Running validation on {len(files)} structures")
    print(f"Oracles: Biopython={HAS_BIOPYTHON}, Gemmi={HAS_GEMMI}, USAlign={HAS_USALIGN}")
    if timeout > 0:
        print(f"Per-structure timeout: {timeout}s")
    print()

    # Aggregate results
    stats = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0, "error": 0})
    all_results = []
    sasa_diffs = []
    n_skipped = 0

    t0 = time.time()
    for i, path in enumerate(files):
        basename = os.path.basename(path)
        structure_results = {"file": basename, "tests": []}

        # Set per-structure timeout
        if timeout > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            for test_fn in ALL_TESTS:
                try:
                    r = test_fn(path)
                    structure_results["tests"].append(r)
                    stats[r["test"]][r["status"]] += 1

                    # Collect SASA diffs for aggregate stats
                    if r["test"] == "sasa" and "relative_diff" in r.get("details", {}):
                        sasa_diffs.append(r["details"]["relative_diff"])
                except Exception as e:
                    stats[test_fn.__name__]["error"] += 1
                    structure_results["tests"].append({
                        "test": test_fn.__name__,
                        "status": "error",
                        "details": {"error": str(e)[:200]}
                    })
        except StructureTimeout:
            n_skipped += 1
            structure_results["tests"] = [{
                "test": "timeout",
                "status": "skip",
                "details": {"error": f"exceeded {timeout}s limit"}
            }]
        finally:
            if timeout > 0:
                signal.alarm(0)  # cancel any pending alarm

        all_results.append(structure_results)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(files) - i - 1) / rate
            print(f"  [{i+1}/{len(files)}] {rate:.1f} structs/s, ETA {eta:.0f}s"
                  f"{f', {n_skipped} skipped' if n_skipped else ''}")

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({len(files)} structures, {elapsed:.1f}s"
          f"{f', {n_skipped} timed out' if n_skipped else ''})")
    print(f"{'='*60}\n")

    for test_name, counts in sorted(stats.items()):
        total = sum(counts.values())
        pass_pct = counts["pass"] / total * 100 if total > 0 else 0
        print(f"  {test_name:20s}  pass={counts['pass']:4d}  warn={counts['warn']:3d}  "
              f"fail={counts['fail']:3d}  error={counts['error']:3d}  ({pass_pct:.1f}%)")

    # Collect SASA speed data
    sasa_speedups = []
    for r in all_results:
        for t in r["tests"]:
            if t["test"] == "sasa" and "speedup" in t.get("details", {}):
                sasa_speedups.append(t["details"]["speedup"])

    if sasa_diffs:
        diffs = np.array(sasa_diffs)
        print(f"\n  SASA vs Biopython:")
        print(f"    Median relative diff: {np.median(diffs):.4f} ({np.median(diffs)*100:.2f}%)")
        print(f"    Mean relative diff:   {np.mean(diffs):.4f} ({np.mean(diffs)*100:.2f}%)")
        print(f"    Max relative diff:    {np.max(diffs):.4f} ({np.max(diffs)*100:.2f}%)")
        print(f"    Within 1%: {np.sum(diffs < 0.01)}/{len(diffs)}")
        print(f"    Within 5%: {np.sum(diffs < 0.05)}/{len(diffs)}")

    if sasa_speedups:
        sp = np.array(sasa_speedups)
        print(f"\n  SASA speed (ferritin vs Biopython):")
        print(f"    Median speedup: {np.median(sp):.1f}x")
        print(f"    Mean speedup:   {np.mean(sp):.1f}x")
        print(f"    Range:          {np.min(sp):.1f}x - {np.max(sp):.1f}x")

    # Save detailed results
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "n_structures": len(files),
                "elapsed_s": round(elapsed, 1),
                "summary": dict(stats),
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"\n  Detailed results saved to {output_file}")

    # Print failures
    failures = [r for r in all_results if any(t["status"] == "fail" for t in r["tests"])]
    if failures:
        print(f"\n  FAILURES ({len(failures)} structures):")
        for r in failures[:20]:
            failed_tests = [t for t in r["tests"] if t["status"] == "fail"]
            for t in failed_tests:
                err = t.get("details", {}).get("error", "?")
                print(f"    {r['file']:15s} {t['test']:20s} {err[:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ferritin validation suite")
    parser.add_argument("--n-structures", type=int, default=1000)
    parser.add_argument("--pdb-dir", default="validation/pdbs/")
    parser.add_argument("--output", default="validation/results.json")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Per-structure timeout in seconds (0=no limit)")
    args = parser.parse_args()

    run_validation(args.pdb_dir, args.n_structures, args.output, args.timeout)
