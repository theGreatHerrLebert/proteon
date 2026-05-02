"""CHARMM19+EEF1 single-point energy oracle: proteon vs BALL on N PDBs.

Mirrors the AMBER96 OpenMM oracle pattern (see amber96_oracle.py) but
swaps the force field to CHARMM19+EEF1 and the reference to BALL via
ball-py. The closure of the crambin oracle gates (PR #24) is what
makes this corpus run meaningful — before that, the per-component
gaps were too large to be informative across structures.

For each PDB:
  1. Load via proteon.
  2. Place polar hydrogens (CHARMM19 is polar-H + N-terminal NH3+):
     `proteon.place_all_hydrogens(s, polar_only=True)` — same prep as
     the unit oracle test.
  3. proteon.compute_energy(s, ff="charmm19_eef1", nonbonded_cutoff=1e6)
     — NoCutoff to match BALL's reference (no switching, no Ewald).
  4. Save to temp PDB. ball.charmm_energy(pdb, use_eef1=True,
     nonbonded_cutoff=1e6, add_hydrogens=False).
  5. Compare per-component (bond_stretch / angle_bend / vdw /
     electrostatic / solvation; proper_torsion + improper still xfail
     per the unit test, recorded but not asserted-on).

Adapter note:
  BALL's `nonbonded` includes vdw + es + solvation; the true
  electrostatic is `nonbonded - vdw - solvation`. (See the bug-fix
  history in evident/claims/forcefield_charmm19_ball.yaml — the older
  "nonbonded - vdw" adapter silently double-counted EEF1.)

Output: JSONL with per-PDB records + a final summary block.

Success criterion (this is a measurement, not a gate): record per-
component median, p95, p99 rel diff. The unit oracle already gates
crambin. The corpus run shows whether crambin's parity generalizes.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PDB_DIR = Path(os.environ.get(
    "PROTEON_PDB_DIR",
    "/scratch/TMAlign/proteon/validation/pdbs_1k_sample",
))
OUT = Path(os.environ.get(
    "PROTEON_CHARMM_ORACLE_OUT",
    "/scratch/TMAlign/proteon/validation/charmm19_eef1_ball_oracle.jsonl",
))
N = int(os.environ.get("N_PDBS", "1000"))
SEED = int(os.environ.get("SEED", "42"))


# Components compared and their tolerance bands (matching the unit oracle).
# proper_torsion + improper_torsion are recorded but NOT asserted on
# (they're known xfails per evident/claims/forcefield_charmm19_ball.yaml).
GATED_COMPONENTS = ("bond_stretch", "angle_bend", "vdw", "electrostatic", "solvation")
BANDS = {
    "bond_stretch":  0.01,   # 1%
    "angle_bend":    0.01,   # 1%
    "vdw":           0.025,  # 2.5%
    "electrostatic": 0.01,   # 1% — was 25% (xfail) before the dist-dep dielectric fix
    "solvation":     0.05,   # 5% (EEF1 is BALL's authoritative term)
}


def _to_proteon_keys(charmm_dict: dict) -> dict:
    """Translate ball.charmm_energy() keys to proteon's schema.

    BALL's `nonbonded` returns vdw + es + solvation (CharmmFF::
    getNonbondedEnergy at charmm.C:434). Subtract both to recover
    pure Coulomb.
    """
    return {
        "bond_stretch":     charmm_dict["bond_stretch"],
        "angle_bend":       charmm_dict["angle_bend"],
        "torsion":          charmm_dict["proper_torsion"],
        "improper_torsion": charmm_dict["improper_torsion"],
        "vdw":              charmm_dict["vdw"],
        "electrostatic":    charmm_dict["nonbonded"] - charmm_dict["vdw"] - charmm_dict["solvation"],
        "solvation":        charmm_dict["solvation"],
        "total":            charmm_dict["total"],
    }


def compare_one(pdb_path: str) -> dict:
    """Compute proteon CHARMM19+EEF1 vs BALL on one PDB.

    Returns a dict with both energy decompositions + per-component
    relative differences. Errors caught and reported under "error".
    """
    import proteon
    import ball  # ball-py — pip install ball-py
    from pdbfixer import PDBFixer
    import openmm.app as app

    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    tmp_path = clean_pdb_path = None
    try:
        # Step 0: clean the PDB. PDBFixer drops heterogens (water, ligands,
        # ions), replaces non-standard residues with their canonical 20-AA
        # equivalents, and adds back missing backbone/sidechain heavy atoms.
        # We do NOT call addMissingHydrogens — proteon's polar-H placement
        # is the canonical CHARMM19 prep and goes downstream.
        # Skip the PDB if it has nucleic-acid residues (CHARMM19 is a
        # protein-only force field).
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        # Detect non-AA residues (nucleic acids etc). PDB CHARMM19 is
        # protein-only — skip and report.
        nuc = {"DA", "DC", "DG", "DT", "DI", "A", "C", "G", "U", "I",
               "RA", "RC", "RG", "RU"}
        non_aa = [r.name for r in fixer.topology.residues() if r.name in nuc]
        if non_aa:
            rec["skipped"] = f"non-AA residues present: {sorted(set(non_aa))[:5]}"
            return rec

        clean_pdb_path = tempfile.NamedTemporaryFile(
            suffix="_clean.pdb", delete=False, mode="w"
        ).name
        with open(clean_pdb_path, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

        # Steps 1-2: load + polar-H prep (same as the unit oracle).
        s = proteon.load(clean_pdb_path)
        n_loaded = sum(1 for _ in s.atoms)
        proteon.place_all_hydrogens(s, polar_only=True)
        n_after = sum(1 for _ in s.atoms)
        rec["n_atoms_loaded"] = int(n_loaded)
        rec["n_atoms_polh"] = int(n_after)

        # Step 3: proteon single-point at NoCutoff (matches BALL ref).
        p = proteon.compute_energy(
            s, ff="charmm19_eef1", units="kJ/mol", nonbonded_cutoff=1e6,
            nbl_threshold=10**9,  # force exact O(N²) path
        )
        rec["proteon"] = {k: float(v) for k, v in p.items()
                          if isinstance(v, (int, float))}

        # Step 4: BALL on the same polar-H PDB (no re-protonation).
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        tmp_path = tmp.name
        tmp.close()
        proteon.save_pdb(s, tmp_path)

        b_raw = ball.charmm_energy(
            tmp_path, use_eef1=True, nonbonded_cutoff=1e6, add_hydrogens=False,
        )
        b = _to_proteon_keys(b_raw)
        rec["ball"] = {k: float(v) for k, v in b.items()}

        # If BALL fails to assign LJ params for any pair, its nonbonded
        # accumulates a NaN that propagates everywhere. Don't treat that
        # as a parity failure — flag the PDB as oracle-unable and skip.
        if any(np.isnan(v) or np.isinf(v) for v in b.values()):
            rec["skipped"] = "BALL produced NaN/inf — typing failure on at least one atom"
            return rec

        # Step 5: per-component relative diffs.
        rels = {}
        for k in ("bond_stretch", "angle_bend", "torsion",
                  "improper_torsion", "vdw", "electrostatic", "solvation"):
            pe, be = p.get(k, float("nan")), b.get(k, float("nan"))
            denom = abs(be) if abs(be) > 1.0 else 1.0  # avoid /0 for tiny terms
            rels[k] = float(abs(pe - be) / denom)
        rec["rel_diff"] = rels

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    finally:
        for p in (tmp_path, clean_pdb_path):
            if p and os.path.exists(p):
                os.unlink(p)

    rec["wall_s"] = float(time.perf_counter() - t0)
    return rec


def main():
    if not PDB_DIR.is_dir():
        raise SystemExit(f"PDB corpus not found: {PDB_DIR}")

    pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No .pdb files in {PDB_DIR}")

    rng = random.Random(SEED)
    rng.shuffle(pdbs)
    sample = [str(PDB_DIR / name) for name in pdbs[:N]]

    # Resume: if OUT already exists, load done PDBs and skip them.
    done_names: set[str] = set()
    if OUT.exists():
        with open(OUT) as f:
            for l in f:
                try:
                    done_names.add(json.loads(l)["pdb"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_names)} PDBs already in {OUT}", flush=True)
    pending = [p for p in sample if Path(p).name not in done_names]
    print(f"Sampled {len(sample)} PDBs (seed={SEED}); {len(pending)} pending", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        return _summarize()

    n_workers = int(os.environ.get("N_WORKERS", "48"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "200"))
    print(f"Using {n_workers} workers, chunk_size={chunk_size}", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Process in chunks so a worker crash (BrokenProcessPool) only loses
    # in-flight work in that chunk, not the entire run. Each chunk gets a
    # fresh pool. If a chunk's pool dies, mark its in-flight PDBs as
    # 'fail' (segfault) and move on.
    with open(OUT, "a") as f:
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            chunk_done_names: set[str] = set()
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = {pool.submit(compare_one, p): p for p in chunk}
                    for fut in as_completed(futs):
                        try:
                            rec = fut.result()
                        except Exception as ex:
                            pdb_path = futs[fut]
                            rec = {
                                "pdb": Path(pdb_path).name,
                                "error": f"worker crash: {type(ex).__name__}: {str(ex)[:120]}",
                            }
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        chunk_done_names.add(rec["pdb"])
                        if "rel_diff" in rec:
                            n_ok += 1
                        elif "skipped" in rec:
                            n_skip += 1
                        else:
                            n_fail += 1
                        progress = len(done_names) + n_ok + n_skip + n_fail
                        if progress % 25 == 0:
                            elapsed = time.perf_counter() - t0
                            rate = (n_ok + n_skip + n_fail) / elapsed if elapsed > 0 else 0
                            eta = (len(pending) - (n_ok + n_skip + n_fail)) / rate if rate > 0 else 0
                            print(
                                f"[{progress}/{len(sample)}] ok={n_ok} skip={n_skip} fail={n_fail}  "
                                f"rate={rate:.2f}/s  eta={eta/60:.1f}min",
                                flush=True,
                            )
            except Exception as ex:
                # Pool died (e.g. segfault propagated). Mark unfinished as fail.
                missing = [Path(p).name for p in chunk
                           if Path(p).name not in chunk_done_names]
                for name in missing:
                    rec = {"pdb": name, "error": f"chunk pool crashed: {type(ex).__name__}"}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    n_fail += 1
                print(
                    f"chunk pool died ({type(ex).__name__}); marked {len(missing)} as failed",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone. new ok={n_ok} skip={n_skip} fail={n_fail} in {elapsed/60:.1f} min",
        flush=True,
    )
    _summarize()


def _summarize():
    """Read OUT and print headline statistics."""

    # Summary stats per component
    by_component: dict[str, list[float]] = {k: [] for k in
        ("bond_stretch", "angle_bend", "torsion", "improper_torsion",
         "vdw", "electrostatic", "solvation")}
    n_passing = {k: 0 for k in by_component}
    n_total = 0
    with open(OUT) as f:
        for l in f:
            r = json.loads(l)
            if "rel_diff" not in r:
                continue
            n_total += 1
            for k, v in r["rel_diff"].items():
                if not (np.isnan(v) or np.isinf(v)):
                    by_component[k].append(v)
                    if k in BANDS and v < BANDS[k]:
                        n_passing[k] += 1

    print(f"\nPer-component relative diff (proteon vs BALL, n={n_total}):")
    print(f"{'component':<18} {'median':>10} {'p95':>10} {'p99':>10} {'max':>10}  {'pass':>6}/{n_total}  band")
    for k in ("bond_stretch", "angle_bend", "vdw", "electrostatic", "solvation",
              "torsion", "improper_torsion"):
        vs = np.array(by_component[k])
        if vs.size == 0:
            print(f"{k:<18} no data")
            continue
        gate = (
            f"  {n_passing[k]:>6}/{n_total}  <{BANDS[k]*100:.1f}%"
            if k in BANDS else "  (xfail; not gated)"
        )
        print(
            f"{k:<18} {np.median(vs):>10.2%} "
            f"{np.percentile(vs, 95):>10.2%} {np.percentile(vs, 99):>10.2%} "
            f"{np.max(vs):>10.2%}{gate}"
        )


if __name__ == "__main__":
    main()
