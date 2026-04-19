#!/usr/bin/env python3
"""Diagnose the proteon CHARMM19+EEF1 wrong-sign-total finding.

Tier-2 weak oracle (commit 8f979f4) flagged that proteon's
charmm19_eef1 totals are POSITIVE on every v1 PDB (1crn +1849, 1ubq
+5591, ...) while OpenMM CHARMM36+OBC2 totals are NEGATIVE. The bug
appears concentrated in the EEF1 solvation term — fer_solv is positive
on 5/6 v1 PDBs and exactly zero on 1ake.

This script does a stepwise teardown on 1crn to localize the problem:

  Step 1: Load 1crn raw, compute CHARMM19+EEF1 energy WITHOUT
          minimization. Rules out "minimizer is driving us into the
          wrong basin" artifacts.
  Step 2: Print every component, including solvation, with sign.
  Step 3: Print the canonical expectation from the EEF1 parameter
          table — for 1crn (46 residues, all-protein) the
          contribution from peptide N (NH1) atoms ALONE should
          dominate at ≥-1100 kJ/mol from solvation.
  Step 4: Repeat with explicit H placement (no minimization).
  Step 5: Repeat with H + minimization for completeness.
  Step 6: Run the SAME structure through OpenMM CHARMM36+OBC2 for
          direct comparison.

If solvation is positive in step 1 (raw, no H, no minimize), the bug
is purely in the energy kernel — atom typing, parameter lookup, or
sign in the eef1_energy() accumulation. If solvation is negative
in step 1 but positive after H placement, the H placement is
corrupting the atom types. If solvation is negative in step 4 and
positive in step 5, the minimizer's gradient has wrong sign on the
EEF1 contribution.

Output: a single markdown table to stdout, no fancy formatting.

Usage:
    python diagnose_charmm_eef1.py [pdb_path]
"""

from __future__ import annotations

import argparse
import sys
import os

import proteon

# Default test structure: 1crn from the v1 SOTA reference set.
DEFAULT_PDB = "/globalscratch/dateschn/proteon-benchmark/sota_pdbs/1crn.pdb"

# Component keys for CHARMM19+EEF1
COMPONENTS = (
    "bond_stretch",
    "angle_bend",
    "torsion",
    "improper_torsion",
    "vdw",
    "electrostatic",
    "solvation",
)


def _find_charmm_ini():
    """Locate proteon-connector/data/charmm19_eef1.ini in the repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    # validation/sota_comparison/ → repo root → proteon-connector/data/...
    candidates = [
        os.path.join(here, "..", "..", "proteon-connector", "data", "charmm19_eef1.ini"),
        os.path.join(here, "..", "..", "..", "proteon-connector", "data", "charmm19_eef1.ini"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _parse_charmm_ini(path):
    """Read [ChargesAndTypeNames] and [EEF1Solvation] sections.

    Returns (type_for, dg_ref_for) where:
        type_for["ALA:N"]   == "NH1"
        dg_ref_for["NH1"]   == -5.95   (kcal/mol)
    """
    type_for = {}
    dg_ref_for = {}
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('@'):
                continue
            if line.startswith('ver:') or line.startswith('key:') or line.startswith('value:'):
                continue
            if line.startswith('['):
                section = line.strip('[]')
                continue
            fields = line.split()
            if section == "ChargesAndTypeNames" and len(fields) >= 4:
                # ver name q type
                name = fields[1]
                ctype = fields[3]
                type_for[name] = ctype
            elif section == "EEF1Solvation" and len(fields) >= 9:
                # ver type V dG_ref dG_free dH_ref Cp_ref sig_w R_min
                ctype = fields[1]
                try:
                    dg_ref_for[ctype] = float(fields[3])
                except ValueError:
                    pass
    return type_for, dg_ref_for


def canonical_eef1_self_solvation(structure):
    """Walk every residue:atom in `structure`, look up its CHARMM type and
    dg_ref independently of proteon's energy kernel, and return the
    canonical Σ dg_ref over heavy atoms.

    Returns a dict with keys: hits, misses, miss_examples, sum_kcal,
    sum_kj, type_breakdown (sorted by absolute total contribution).
    """
    ini_path = _find_charmm_ini()
    if ini_path is None:
        return {
            "hits": 0, "misses": 0, "miss_examples": ["<charmm19_eef1.ini not found>"],
            "sum_kcal": 0.0, "sum_kj": 0.0, "type_breakdown": [],
        }
    type_for, dg_ref_for = _parse_charmm_ini(ini_path)

    sum_kcal = 0.0
    hits = 0
    misses = 0
    miss_examples = []
    type_counts = {}  # ctype -> (count, total_kcal)

    for residue in structure.residues:
        rname = (residue.name or "").strip()
        for atom in residue.atoms:
            aname = (atom.name or "").strip()
            element = (atom.element or "").strip().upper()
            # eef1_energy() in proteon skips hydrogens
            if element in ("H", "D"):
                continue
            key = f"{rname}:{aname}"
            ctype = type_for.get(key)
            if ctype is None:
                misses += 1
                if len(miss_examples) < 30:
                    miss_examples.append(f"{key} → <no type in [ChargesAndTypeNames]>")
                continue
            dg = dg_ref_for.get(ctype)
            if dg is None:
                misses += 1
                if len(miss_examples) < 30:
                    miss_examples.append(f"{key} → {ctype} → <no dg_ref in [EEF1Solvation]>")
                continue
            hits += 1
            sum_kcal += dg
            n, t = type_counts.get(ctype, (0, 0.0))
            type_counts[ctype] = (n + 1, t + dg)

    type_breakdown = sorted(
        ((ct, n, t) for ct, (n, t) in type_counts.items()),
        key=lambda r: -abs(r[2]),
    )

    return {
        "hits": hits,
        "misses": misses,
        "miss_examples": miss_examples,
        "sum_kcal": sum_kcal,
        "sum_kj": sum_kcal * 4.184,
        "type_breakdown": type_breakdown,
    }


def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:>+12.3f}"
    return f"{v:>12}"


def print_energy(label: str, e: dict):
    print(f"\n  {label}")
    print(f"    {'component':<20} {'value (kJ/mol)':>14}")
    print(f"    {'-' * 20} {'-' * 14}")
    for k in COMPONENTS:
        v = e.get(k)
        marker = ""
        if k == "solvation" and v is not None:
            marker = "  ← wrong sign?" if v > 0 else "  (negative ✓)"
        print(f"    {k:<20} {fmt(v)}{marker}")
    print(f"    {'total':<20} {fmt(e.get('total'))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb", nargs="?", default=DEFAULT_PDB)
    args = parser.parse_args()

    if not os.path.isfile(args.pdb):
        print(f"ERROR: pdb not found: {args.pdb}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"CHARMM19+EEF1 sign diagnostic on {os.path.basename(args.pdb)}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: raw structure, no H, no minimize, compute_energy
    # ------------------------------------------------------------------
    s_raw = proteon.load(args.pdb)
    print(f"\nLoaded {args.pdb}: {s_raw.atom_count} atoms")
    try:
        n_residues = sum(1 for _ in s_raw.residues)
    except Exception:
        n_residues = "?"
    print(f"  residues: {n_residues}")

    e_raw = proteon.compute_energy(s_raw, ff="charmm19_eef1", units="kJ/mol")
    print_energy("STEP 1: RAW (no H, no minimization)", e_raw)
    raw_solv_sign = "POSITIVE (BUG)" if e_raw["solvation"] > 0 else "negative (canonical)"
    print(f"\n    => solvation sign on raw structure: {raw_solv_sign}")

    # ------------------------------------------------------------------
    # Canonical expectation (back-of-envelope from EEF1 parameter table)
    # ------------------------------------------------------------------
    print("\n  Canonical expectation for a typical small protein:")
    print(f"    n_residues × NH1 dg_ref ≈ {n_residues} × -5.95 kcal/mol "
          f"= {n_residues * -5.95:.1f} kcal/mol "
          f"= {n_residues * -5.95 * 4.184:.1f} kJ/mol")
    print("    (just from peptide nitrogens — actual sum should be more negative)")

    # ------------------------------------------------------------------
    # Step 2: H placed but no minimization
    # ------------------------------------------------------------------
    s_h = proteon.load(args.pdb)
    reports_h = proteon.batch_prepare(
        [s_h],
        reconstruct=False,
        hydrogens="all",
        minimize=False,
        strip_hydrogens=False,
        ff="charmm19_eef1",
    )
    print(f"\n  After H placement: {s_h.atom_count} atoms (was {s_raw.atom_count})")
    e_h = proteon.compute_energy(s_h, ff="charmm19_eef1", units="kJ/mol")
    print_energy("STEP 2: H placed, NO minimization", e_h)

    # ------------------------------------------------------------------
    # Step 3: H + 200 LBFGS steps
    # ------------------------------------------------------------------
    s_min = proteon.load(args.pdb)
    reports_min = proteon.batch_prepare(
        [s_min],
        reconstruct=False,
        hydrogens="all",
        minimize=True,
        minimize_method="lbfgs",
        minimize_steps=200,
        gradient_tolerance=0.1,
        strip_hydrogens=False,
        ff="charmm19_eef1",
    )
    r = reports_min[0]
    print(f"\nSTEP 3: H + LBFGS minimization ({r.minimizer_steps} steps, "
          f"converged={r.converged})")
    print(f"  initial_energy: {r.initial_energy:>+14.3f} kJ/mol")
    print(f"  final_energy:   {r.final_energy:>+14.3f} kJ/mol")
    e_min = proteon.compute_energy(s_min, ff="charmm19_eef1", units="kJ/mol")
    print_energy("  re-evaluated post-minimization", e_min)

    # ------------------------------------------------------------------
    # Step 3.5: canonical EEF1 expectation, computed independently
    # of proteon's energy kernel by reading the .ini file directly.
    # This tells us what the correct value SHOULD be.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 3.5: canonical EEF1 expectation (computed from .ini directly)")
    print("=" * 70)
    canonical = canonical_eef1_self_solvation(s_raw)
    print(f"\n  This walks every residue:atom in 1crn, looks up the CHARMM type")
    print(f"  via [ChargesAndTypeNames], looks up the dg_ref via [EEF1Solvation],")
    print(f"  and sums. Bypasses proteon's energy kernel completely.\n")
    print(f"  hits   (atom found in both tables):  {canonical['hits']}")
    print(f"  misses (atom not found / no dg_ref): {canonical['misses']}")
    print(f"  expected Σ dg_ref: {canonical['sum_kcal']:>+12.3f} kcal/mol")
    print(f"  expected Σ dg_ref: {canonical['sum_kj']:>+12.3f} kJ/mol")
    print(f"  proteon actual:   {e_raw['solvation']:>+12.3f} kJ/mol")
    diff = e_raw['solvation'] - canonical['sum_kj']
    print(f"  difference (proteon - canonical): {diff:>+12.3f} kJ/mol")
    print()
    if canonical['miss_examples']:
        print("  Sample missed atoms (first 10):")
        for ex in canonical['miss_examples'][:10]:
            print(f"    {ex}")
    print()
    if canonical['type_breakdown']:
        print("  Top contributing atom types (kcal/mol total):")
        for t, n, total in canonical['type_breakdown'][:15]:
            print(f"    {t:<8} count={n:>4}  total={total:>+10.3f} kcal/mol")

    # ------------------------------------------------------------------
    # Step 4: comparison with raw AMBER96 on the same input (sanity)
    # ------------------------------------------------------------------
    s_amber = proteon.load(args.pdb)
    e_amber = proteon.compute_energy(s_amber, ff="amber96", units="kJ/mol")
    print_energy("STEP 4: AMBER96 on the same raw structure (control)", e_amber)
    print("\n  (AMBER96 has no solvation term, so 'solvation' should be 0 or None.)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Raw CHARMM19+EEF1 solvation: {e_raw['solvation']:>+12.3f} kJ/mol")
    print(f"  H+placement solvation:       {e_h['solvation']:>+12.3f} kJ/mol")
    print(f"  Post-minimization solvation: {e_min['solvation']:>+12.3f} kJ/mol")
    print()
    print(f"  Raw CHARMM19+EEF1 total:     {e_raw['total']:>+12.3f} kJ/mol")
    print(f"  H+placement total:           {e_h['total']:>+12.3f} kJ/mol")
    print(f"  Post-minimization total:     {e_min['total']:>+12.3f} kJ/mol")
    print()
    print(f"  Raw AMBER96 total (control): {e_amber['total']:>+12.3f} kJ/mol")
    print()
    print("Localization key:")
    print("  - solvation > 0 in STEP 1 (raw):")
    print("      ⇒ bug is in eef1_energy() kernel itself")
    print("        (atom typing, dg_ref accumulation, or sign in pair correction)")
    print("  - solvation < 0 in STEP 1, > 0 in STEP 2 (H placed):")
    print("      ⇒ H placement corrupts atom types (e.g. NH1 reclassified as NH2)")
    print("  - solvation < 0 in STEP 2, > 0 in STEP 3 (minimized):")
    print("      ⇒ minimizer gradient on EEF1 has wrong sign — energy and force")
    print("        diverge, structure converges to local max instead of min")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
