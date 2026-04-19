"""GROMACS fold preservation benchmark — third tool on the same 1000 PDBs.

Per structure:
  1. gmx pdb2gmx -ff amber96 -water none -ignh      (build topology)
  2. gmx editconf -box 30 30 30 -c                   (large vacuum box)
  3. gmx grompp (steep EM, rcoulomb=rvdw=14 nm)
  4. gmx mdrun -nt 1                                 (single-thread EM)
  5. gmx editconf -> out.pdb
  6. Extract CA_pre from the input PDB, CA_post from the EM output.
  7. proteon.tm_score(CA_pre, CA_post) with identity invmap.

Reports TM-score, RMSD, wall time, failure mode. Runs with a process
pool — GROMACS handles one structure per worker at a time, each mdrun
is single-threaded, so 16 workers saturate 16 cores. Progress written
incrementally to JSONL.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import proteon

PDB_DIR = Path("/scratch/TMAlign/proteon/validation/gmx_fold_preservation/pdbs_1k")
OUT = Path("/scratch/TMAlign/proteon/validation/gmx_fold_preservation/tm_fold_gromacs.jsonl")
SAMPLE_FILE = Path("/tmp/gmx_sample.txt")

GMX = os.environ.get(
    "GMX",
    "/scratch/TMAlign/gromacs-2026.1/build/bin/gmx",
)

# Match OpenMM's LocalEnergyMinimizer budget:
#   * emtol 10 kJ/mol/nm (OpenMM default 10, proteon 0.1 kcal/mol/A ≈ 0.4)
#   * max 100 steps
#   * L-BFGS (same flavor OpenMM uses internally, not steepest descent)
MDP_TEMPLATE = """\
integrator     = l-bfgs
nsteps         = 100
emtol          = 10.0
emstep         = 0.01
nstenergy      = 1
nstlog         = 1
pbc            = xyz
cutoff-scheme  = Verlet
coulombtype    = Cut-off
rcoulomb       = 14.0
rvdw           = 14.0
vdw-type       = Cut-off
nstlist        = 10
verlet-buffer-tolerance = -1
rlist          = 14.0
constraints    = none
"""


def extract_ca_from_pdb(path: str) -> np.ndarray:
    """Parse a PDB file, return Nx3 CA coordinates (Å)."""
    cas = []
    # Use proteon's loader — pdbtbx is stricter than some PDB files, so
    # fall back to a manual parse for GROMACS output (which has REMARK
    # and CRYST1 lines pdbtbx sometimes rejects).
    try:
        s = proteon.load(path)
        return proteon.extract_ca_coords(s)
    except Exception:
        pass
    for line in open(path):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        if name != "CA":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            continue
        cas.append([x, y, z])
    return np.array(cas, dtype=np.float64)


def tm_pair(ca_ref: np.ndarray, ca_mov: np.ndarray) -> dict:
    n = len(ca_ref)
    invmap = np.arange(n, dtype=np.int32)
    tm, n_aln, rmsd_val, _R, _t = proteon.tm_score(ca_mov, ca_ref, invmap)
    return {
        "tm_score": float(tm),
        "rmsd": float(rmsd_val),
        "n_ca": int(n),
        "n_aligned": int(n_aln),
    }


def _gmx_error(stderr_bytes: bytes) -> str:
    """Extract the 'Fatal error:' block's first real sentence from gmx stderr."""
    text = stderr_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Look for a "Fatal error:" marker and grab lines after it until we
    # hit a "For more information" / dashes line.
    for i, ln in enumerate(lines):
        if "Fatal error" in ln or "Inconsistency in" in ln:
            out = []
            for ln2 in lines[i + 1:]:
                if not ln2.strip() or set(ln2.strip()) <= {"-"}:
                    if out:
                        break
                    continue
                if ln2.startswith(("Program:", "Source file:", "Function:",
                                   "For more information", "website at",
                                   "-----")):
                    if out:
                        break
                    continue
                out.append(ln2.strip())
                if len(" ".join(out)) > 140:
                    break
            return " ".join(out) if out else "unknown"
    # Fallback: last meaningful line of stderr.
    for ln in reversed(lines):
        s = ln.strip()
        if s and not set(s) <= {"-"} and not s.startswith(("Program:", "For more")):
            return s
    return f"rc!=0 (empty stderr)"


def run_one(pdb_path: str) -> dict:
    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    work = tempfile.mkdtemp(prefix="gmx_")
    try:
        env = os.environ.copy()
        env["GMX_MAXBACKUP"] = "-1"

        # Strip water/hetero lines — pdb2gmx -water none rejects inputs
        # that contain water, and we want vacuum anyway. Keep protein
        # ATOM records only; drop HETATM (waters, ligands, ions, etc.)
        # and TER/END records (pdb2gmx re-inserts as needed).
        clean_pdb = f"{work}/clean.pdb"
        with open(pdb_path) as inp, open(clean_pdb, "w") as out:
            for line in inp:
                if line.startswith("ATOM"):
                    out.write(line)
                elif line.startswith("TER"):
                    out.write(line)

        # Step 1: pdb2gmx.
        r = subprocess.run(
            [GMX, "pdb2gmx", "-f", clean_pdb,
             "-o", f"{work}/start.gro",
             "-p", f"{work}/topol.top",
             "-ff", "amber96", "-water", "none", "-ignh"],
            cwd=work, capture_output=True, env=env, timeout=60,
        )
        if r.returncode != 0:
            rec["error"] = f"pdb2gmx: {_gmx_error(r.stderr):.200}"
            return rec

        # Pre-minimization CA from the original input PDB.
        ca_pre = extract_ca_from_pdb(pdb_path)
        rec["n_ca_pre"] = int(len(ca_pre))
        if len(ca_pre) == 0:
            rec["error"] = "no CA atoms in input"
            return rec

        # Step 2: big vacuum box.
        r = subprocess.run(
            [GMX, "editconf", "-f", f"{work}/start.gro",
             "-o", f"{work}/big.gro", "-box", "30", "30", "30", "-c"],
            cwd=work, capture_output=True, env=env, timeout=30,
        )
        if r.returncode != 0:
            rec["error"] = "editconf-box failed"
            return rec

        # Step 3: grompp.
        Path(work, "em.mdp").write_text(MDP_TEMPLATE)
        r = subprocess.run(
            [GMX, "grompp", "-f", "em.mdp", "-c", "big.gro",
             "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "5"],
            cwd=work, capture_output=True, env=env, timeout=30,
        )
        if r.returncode != 0:
            rec["error"] = f"grompp: {_gmx_error(r.stderr):.200}"
            return rec

        # Step 4: mdrun EM, single-thread.
        r = subprocess.run(
            [GMX, "mdrun", "-s", "em.tpr", "-c", "em.gro",
             "-e", "em.edr", "-o", "em.trr", "-g", "em.log", "-nt", "1"],
            cwd=work, capture_output=True, env=env, timeout=300,
        )
        if r.returncode != 0:
            rec["error"] = "mdrun failed"
            return rec
        # Parse energy from log for bookkeeping.
        log = Path(work, "em.log").read_text(errors="replace")
        for line in log.splitlines():
            if line.strip().startswith("Potential Energy"):
                try:
                    rec["final_energy_kj"] = float(line.split("=")[1].strip())
                except Exception:
                    pass

        # Step 5: export as PDB.
        r = subprocess.run(
            [GMX, "editconf", "-f", "em.gro", "-o", "em.pdb"],
            cwd=work, capture_output=True, env=env, timeout=30,
        )
        if r.returncode != 0:
            rec["error"] = "editconf-pdb failed"
            return rec

        # Step 6: post-min CA from the EM output PDB.
        ca_post = extract_ca_from_pdb(f"{work}/em.pdb")
        rec["n_ca_post"] = int(len(ca_post))

        if ca_pre.shape != ca_post.shape:
            rec["error"] = f"CA shape mismatch {ca_pre.shape} vs {ca_post.shape}"
            return rec

        # Step 7: TM-score.
        rec.update(tm_pair(ca_pre, ca_post))
    except subprocess.TimeoutExpired as e:
        rec["error"] = f"timeout: {e.cmd[1] if e.cmd and len(e.cmd) > 1 else 'unknown'}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def main():
    # Sample order from the tm_fold_plots/proteon.jsonl — matches the
    # proteon and OpenMM runs exactly.
    with open("/scratch/TMAlign/proteon/validation/tm_fold_plots/proteon.jsonl") as f:
        sample_names = [json.loads(l)["pdb"] for l in f]
    print(f"Sample: {len(sample_names)} PDBs", flush=True)
    sample = [str(PDB_DIR / n) for n in sample_names if (PDB_DIR / n).exists()]
    print(f"Available locally: {len(sample)}", flush=True)

    n_workers = int(os.environ.get("N_WORKERS", "16"))
    print(f"Workers: {n_workers} (1 gmx thread each)", flush=True)
    print(f"Output: {OUT}", flush=True)

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    with open(OUT, "w") as f, ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(run_one, p) for p in sample]
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            if "tm_score" in rec:
                n_ok += 1
            else:
                n_fail += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0 or done == len(sample):
                el = time.perf_counter() - t0
                rate = done / el
                eta = (len(sample) - done) / rate if rate > 0 else 0
                print(
                    f"[{done}/{len(sample)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"\nDone. ok={n_ok} fail={n_fail} in {elapsed/60:.1f} min "
          f"({n_ok/elapsed:.2f} struct/s)", flush=True)

    tms = []
    rmsds = []
    with open(OUT) as fh:
        for l in fh:
            r = json.loads(l)
            if "tm_score" in r:
                tms.append(r["tm_score"])
                rmsds.append(r["rmsd"])
    if tms:
        tms = np.array(tms)
        rmsds = np.array(rmsds)
        print(f"\nGROMACS AMBER96 TM-score (n={len(tms)}):")
        print(f"  mean={tms.mean():.4f}  median={np.median(tms):.4f}")
        print(f"  min={tms.min():.4f}  p05={np.percentile(tms,5):.4f}")
        print(f"RMSD: mean={rmsds.mean():.3f}  median={np.median(rmsds):.3f}  max={rmsds.max():.3f}")


if __name__ == "__main__":
    main()
