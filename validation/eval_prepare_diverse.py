#!/usr/bin/env python
"""Diverse-structure evaluation of the prepare detect->repair pipeline.

Runs ``prepare_for_supervision`` over a diverse PDB corpus and tallies:
  * load success / failure
  * label_safe rate (strict gate) and per-profile safe rates
  * per-hazard frequency across the full taxonomy
  * how much a sensible RepairPolicy recovers (passes_policy + actions)

Usage:
    python validation/eval_prepare_diverse.py [N] [--repair]
"""

import collections
import glob
import json
import os
import sys
import time

import proteon

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdbs_10k")

HAZARDS = [
    "severe_heavy_clashes", "reconstructed_atoms", "missing_atoms", "untyped_atoms",
    "altlocs", "multiple_models", "insertion_codes", "nonstandard_residues",
    "metals", "chain_gaps", "chirality_outliers", "assembly_mismatch",
]
PROFILES = [
    "label_safe", "label_safe_heavy_coords", "label_safe_all_atom_coords",
    "label_safe_energy", "label_safe_sequence_indexed", "label_safe_interface",
]


def detection_pass(paths, n_threads=-1):
    # Every hazard in the taxonomy is structural (heavy-atom clashes, missing
    # atoms, altlocs, models, insertion codes, non-standard residues, metals,
    # chain gaps, chirality, assembly) — none depend on minimization, and H-only
    # relaxation never moves heavy atoms. So skip minimize for a fast, identical
    # hazard read-out.
    t0 = time.time()
    results = proteon.prepare_for_supervision(paths, n_threads=n_threads, minimize=False)
    dt = time.time() - t0

    n = len(results)
    loaded = [r for r in results if r.loaded and r.report is not None]
    hazard_counts = collections.Counter()
    profile_safe = collections.Counter()
    n_hazards_per = collections.Counter()
    clash_mags, missing_mags, clashscores = [], [], []
    for r in loaded:
        rep = r.report
        hz = rep.label_hazards
        n_hazards_per[len(hz)] += 1
        for h in hz:
            hazard_counts[h] += 1
        for p in PROFILES:
            if getattr(rep, p):
                profile_safe[p] += 1
        if rep.has_heavy_clashes:
            clash_mags.append(rep.n_heavy_clashes)
        if rep.has_missing_atoms:
            missing_mags.append(rep.n_missing_heavy_atoms)
        # Use the REPORTED clashscore (templated-atom denominator, matching the
        # gate) — NOT a structure-wide non-H count, which would dilute it for
        # ligand/metal structures and disagree with has_severe_clashes.
        clashscores.append(rep.clashscore)

    def _q(xs):
        if not xs:
            return {}
        s = sorted(xs)
        return {"min": s[0], "median": s[len(s) // 2], "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}

    return {
        "corpus_size": n,
        "loaded": len(loaded),
        "load_failed": n - len(loaded),
        "seconds": round(dt, 1),
        "per_struct_ms": round(1000 * dt / max(n, 1), 1),
        "label_safe": profile_safe["label_safe"],
        "profile_safe": {p: profile_safe[p] for p in PROFILES},
        "hazard_counts": {h: hazard_counts[h] for h in HAZARDS},
        "n_hazards_histogram": dict(sorted(n_hazards_per.items())),
        "clash_magnitude": _q(clash_mags),
        "missing_magnitude": _q(missing_mags),
        "clashscore_distribution": {
            "min": round(min(clashscores), 1) if clashscores else None,
            "median": round(sorted(clashscores)[len(clashscores) // 2], 1) if clashscores else None,
            "p90": round(sorted(clashscores)[int(0.9 * (len(clashscores) - 1))], 1) if clashscores else None,
            "max": round(max(clashscores), 1) if clashscores else None,
        },
        # Of the structures with a clashscore (all loaded with heavy atoms;
        # clash-free ones score 0), how many sit at or below each candidate gate.
        # t=0 is today's binary "any clash" gate. This is the clash-only yield —
        # an upper bound on what a severity threshold buys before other hazards.
        "clashscore_denominator": len(clashscores),
        "yield_at_clashscore": {
            str(t): sum(1 for cs in clashscores if cs <= t) for t in (0, 2, 5, 10, 20, 40)
        },
    }


def repair_pass(paths, clash_action, n_threads=-1):
    """A DL-training-oriented heavy_coords policy: reconstruct missing atoms,
    pick a single altloc/model, and either ``relax`` deposited clashes away or
    ``drop`` the structures that still clash. ``clash_action`` is "relax" or "drop".
    """
    from proteon.repair import RepairSummary

    rules = dict(
        missing_atoms="reconstruct", reconstructed_atoms="accept",
        severe_heavy_clashes=clash_action,
        altlocs="select_highest_occupancy",
        multiple_models="select_first",
    )
    if clash_action == "relax":
        rules["relaxed_coords"] = "accept"  # relax provenance must be explicit
    pol = proteon.RepairPolicy.for_profile("heavy_coords", **rules)
    # Base prepare pass: minimize=False (H-only relaxation never moves heavy
    # atoms, so it can't change the clash/heavy-coord verdict). The "relax"
    # action runs its own heavy-atom minimization on just the clashy structures.
    t0 = time.time()
    results = proteon.prepare_for_supervision(
        paths, repair=pol, n_threads=n_threads, minimize=False)
    dt = time.time() - t0

    summary = RepairSummary.from_results(results)
    drifts = [r.repair.coords_drift for r in results
              if r.repair is not None and getattr(r.repair, "coords_drift", None) is not None]
    drifts.sort()

    def _q(xs):
        if not xs:
            return {}
        return {"min": round(xs[0], 3), "median": round(xs[len(xs) // 2], 3),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 3), "max": round(xs[-1], 3)}

    return {
        "clash_action": clash_action,
        "n": summary.total,
        "passes_policy": summary.passed,
        "dropped": summary.dropped,
        "seconds": round(dt, 1),
        "by_action": summary.by_action,
        "dropped_by_hazard": summary.dropped_by_hazard,
        "accepted_by_hazard": summary.accepted_by_hazard,
        "relax_ca_drift_A": _q(drifts),
    }


if __name__ == "__main__":
    n = None
    do_repair = "--repair" in sys.argv
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
    # Keep only files that actually resolve (some corpora are symlink farms).
    paths = [p for p in sorted(glob.glob(os.path.join(CORPUS, "*.pdb"))) if os.path.exists(p)]
    if n:
        # Even stride for a representative slice across the (id-sorted) corpus.
        stride = max(1, len(paths) // n)
        paths = paths[::stride][:n]
    print(f"corpus: {len(paths)} structures from {CORPUS}", file=sys.stderr)

    out = {}
    if "--repair-drop" in sys.argv:
        out["repair_drop"] = repair_pass(paths, "drop")
    elif "--repair-relax" in sys.argv:
        out["repair_relax"] = repair_pass(paths, "relax")
    else:
        out["detection"] = detection_pass(paths)
    print(json.dumps(out, indent=2))
