"""Per-residue DSSP secondary-structure agreement: proteon vs mkdssp.

The third-party canonical DSSP implementation is `mkdssp` (the libcifpp /
dssp v4.x C++ rewrite of Kabsch-Sander 1983). proteon ships its own
port of the same algorithm. This oracle pins per-residue agreement
between the two implementations across a 50K-PDB random sample.

Different from the existing CI test (`tests/oracle/test_dssp_oracle.py`)
which compares against pydssp (a NumPy/PyTorch reimplementation by a
separate research group): pydssp is the cleanest cross-implementation
agreement target at unit-test scope, but the wider biology community
treats `mkdssp` as the canonical reference. Cross-tool parity at 50K
scale against the canonical reference is the v0.2.0 trust pyramid's
DSSP rung.

Runner contract (v0.2.0 onward):

  PROTEON_CORPUS_DIR   directory of input .pdb files (v0.2.0 universal)
  PROTEON_PDB_DIR      legacy alias for the input directory
  PROTEON_PDB_LIST     explicit pre-filtered list of paths (one per line);
                       overrides directory glob when set
  PROTEON_OUTPUT_DIR   directory for the JSONL artifact (v0.2.0)
  PROTEON_DSSP_ORACLE_OUT  legacy explicit JSONL path
  N_PDBS               how many PDBs from the sample to actually score
  N_WORKERS            pool width (default 32)
  TASK_TIMEOUT_S       per-task timeout (default 60s)
  SEED                 sample shuffle seed (default 42)
  MKDSSP_BIN           path to the mkdssp binary (default auto-detect)

8-class DSSP alphabet:
    H = α-helix
    G = 3_10 helix
    I = π-helix
    E = β-strand (extended)
    B = β-bridge (isolated)
    T = turn
    S = bend
    C = loop / coil (often output as space ' ' by mkdssp; we normalise to '-')

Per-residue agreement rate is the headline metric. Empirically the two
implementations agree 97-100% on canonical structures; gaps usually
sit on helix/strand boundary residues where the H-bond energy crosses
the -0.5 kcal/mol threshold and tiny coordinate differences flip the
call.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from pathlib import Path

import numpy as np

import pebble
import gemmi
import proteon

# ---------------------------------------------------------------------------
# Path resolution: legacy per-runner env vars first, then v0.2.0 universal
# data-mount synonyms, then a monster3 fallback for the existing batch
# workflow.
# ---------------------------------------------------------------------------
PDB_DIR = Path(
    os.environ.get("PROTEON_PDB_DIR")
    or os.environ.get("PROTEON_CORPUS_DIR")
    or "/globalscratch/dateschn/proteon-benchmark/pdbs_50k"
)
_v02_out_dir = os.environ.get("PROTEON_OUTPUT_DIR")
OUT = Path(
    os.environ.get("PROTEON_DSSP_ORACLE_OUT")
    or (Path(_v02_out_dir) / "dssp_mkdssp_oracle.jsonl" if _v02_out_dir else None)
    or "/globalscratch/dateschn/proteon-benchmark/dssp_mkdssp_oracle.jsonl"
)
N = int(os.environ.get("N_PDBS", "1000"))
SEED = int(os.environ.get("SEED", "42"))

# mkdssp binary: image vendors at /usr/local/bin/mkdssp; allow override
# for source-tree dev where the path may differ.
MKDSSP_BIN = os.environ.get("MKDSSP_BIN") or shutil.which("mkdssp") or "/usr/local/bin/mkdssp"


def _parse_dssp_output(text: str) -> str:
    """Extract the per-residue 8-class secondary-structure string from a
    DSSP-format text output (the format mkdssp emits by default).

    The DSSP file has a header followed by a residue table introduced
    by the line ``  #  RESIDUE AA STRUCTURE ...``. Each subsequent
    non-empty line carries one residue:

        column   1-5  : record number (right-justified)
        column  17    : 8-class secondary structure character
                        (' ' for loop)
        column   6-10 : PDB residue number (or '!' for chain breaks)

    Returns a string of 8-class characters, with ' ' (loop) normalised
    to '-' so callers can compare without space-vs-character ambiguity.
    Chain-break records (residue number '!') are skipped.
    """
    lines = text.splitlines()
    in_data = False
    ss_chars: list[str] = []
    for line in lines:
        if not in_data:
            if line.startswith("  #  RESIDUE"):
                in_data = True
            continue
        if len(line) < 17:
            continue
        res_num_field = line[5:10].strip()
        if not res_num_field or res_num_field == "!":
            continue
        ss_char = line[16]
        ss_chars.append(ss_char if ss_char.strip() else "-")
    return "".join(ss_chars)


def _normalise_proteon_ss(ss: str) -> str:
    """Normalise proteon's DSSP output to mkdssp's space convention.

    proteon may emit 'C' for coil/loop where mkdssp emits ' '; we use
    '-' as the canonical loop character on both sides to make
    per-residue comparison straightforward.
    """
    return "".join(c if c not in (" ", "C") else "-" for c in ss)


def compare_one(pdb_path: str) -> dict:
    """Per-residue 8-class DSSP agreement: proteon vs mkdssp on one PDB.

    Returns a record with both per-tool SS strings, per-residue
    agreement rate, and per-class composition counts. Errors caught
    and reported under 'error'; per-PDB skips under 'skipped'.
    """
    rec = {"pdb": Path(pdb_path).name}
    t0 = time.perf_counter()
    try:
        # --- proteon side ---
        s = proteon.load(pdb_path)
        proteon_ss_raw = proteon.dssp(s)  # 8-class output
        proteon_ss = _normalise_proteon_ss(proteon_ss_raw)
        rec["proteon_n"] = len(proteon_ss)

        if rec["proteon_n"] == 0:
            rec["skipped"] = "no_residues"
            rec["wall_s"] = float(time.perf_counter() - t0)
            return rec

        # --- mkdssp side ---
        # mkdssp v4 + libcifpp's strict validator rejects many PDB-derived
        # datablocks (multiple REMARK 3 records → "Duplicate Key violation"
        # on the internal _refine table). Pre-convert PDB → mmCIF via
        # gemmi (more permissive in its mmCIF emission) and feed mmCIF to
        # mkdssp. The runtime cost is one gemmi read + write per PDB
        # (sub-second). The libcifpp issue is upstream; this bridge avoids
        # the workaround needing a libcifpp patch.
        cif_tmp = tempfile.NamedTemporaryFile(
            suffix=".cif", delete=False, mode="w"
        )
        cif_path = cif_tmp.name
        cif_tmp.close()
        try:
            s_gemmi = gemmi.read_pdb(pdb_path)
            s_gemmi.make_mmcif_document().write_file(cif_path)
            result = subprocess.run(
                [MKDSSP_BIN, cif_path, "/dev/stdout"],
                capture_output=True, text=True, timeout=30,
            )
        finally:
            if os.path.exists(cif_path):
                os.unlink(cif_path)
        if result.returncode != 0:
            rec["error"] = f"mkdssp returncode={result.returncode}: {result.stderr[:200]}"
            rec["wall_s"] = float(time.perf_counter() - t0)
            return rec
        mkdssp_ss = _parse_dssp_output(result.stdout)
        rec["mkdssp_n"] = len(mkdssp_ss)

        # --- compare ---
        # Length must match for per-residue alignment. proteon includes
        # all residues; mkdssp may skip non-standard residues entirely
        # depending on flags. Below the size mismatches are recorded but
        # NOT classified as errors — they show up as "skipped" with a
        # length-mismatch reason so the JSONL distinguishes them from
        # genuine SS disagreements.
        if rec["proteon_n"] != rec["mkdssp_n"]:
            rec["skipped"] = (
                f"length_mismatch: proteon={rec['proteon_n']} "
                f"mkdssp={rec['mkdssp_n']}"
            )
            # Even without per-residue comparison, record both
            # compositions for downstream forensic work.
            rec["proteon_composition"] = dict(Counter(proteon_ss))
            rec["mkdssp_composition"] = dict(Counter(mkdssp_ss))
            rec["wall_s"] = float(time.perf_counter() - t0)
            return rec

        n_match = sum(p == m for p, m in zip(proteon_ss, mkdssp_ss))
        rec["n_match"] = int(n_match)
        rec["agreement_rate"] = float(n_match) / rec["proteon_n"]
        rec["proteon_composition"] = dict(Counter(proteon_ss))
        rec["mkdssp_composition"] = dict(Counter(mkdssp_ss))

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-3:]

    rec["wall_s"] = float(time.perf_counter() - t0)
    return rec


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Two corpus modes (mirrors the CHARMM oracle):
    pdb_list = os.environ.get("PROTEON_PDB_LIST")
    if pdb_list:
        list_path = Path(pdb_list)
        if not list_path.is_file():
            raise SystemExit(f"PDB list file not found: {list_path}")
        with open(list_path) as f:
            sample = [line.strip() for line in f if line.strip()]
        sample = [p for p in sample if Path(p).is_file()]
        sample = sample[:N]
        print(f"Loaded {len(sample)} PDB paths from {list_path}", flush=True)
    else:
        if not PDB_DIR.is_dir():
            raise SystemExit(f"PDB corpus not found: {PDB_DIR}")
        pdbs = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
        if not pdbs:
            raise SystemExit(f"No .pdb files in {PDB_DIR}")
        rng = random.Random(SEED)
        rng.shuffle(pdbs)
        sample = [str(PDB_DIR / name) for name in pdbs[:N]]

    # Resume support.
    done_names: set[str] = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    done_names.add(json.loads(line)["pdb"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_names)} PDBs already in {OUT}", flush=True)
    pending = [p for p in sample if Path(p).name not in done_names]
    print(f"Total sample: {len(sample)} PDBs; {len(pending)} pending after resume", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        return _summarize()

    n_workers = int(os.environ.get("N_WORKERS", "32"))
    task_timeout = float(os.environ.get("TASK_TIMEOUT_S", "60"))
    print(
        f"Using {n_workers} pebble workers, "
        f"per-task timeout {task_timeout:.0f}s, "
        f"mkdssp at {MKDSSP_BIN}",
        flush=True,
    )

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0

    # pebble for crash isolation: a single pathological PDB that segfaults
    # mkdssp or hangs proteon DSSP gets cleaned up as a per-task error
    # rather than cascading the pool. Same pattern as the CHARMM oracle.
    with open(OUT, "a") as f:
        with pebble.ProcessPool(max_workers=n_workers) as pool:
            futs = {
                pool.schedule(compare_one, args=[p], timeout=task_timeout): p
                for p in pending
            }
            for fut, pdb_path in futs.items():
                pdb_name = Path(pdb_path).name
                try:
                    rec = fut.result()
                except pebble.ProcessExpired as ex:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"worker subprocess died: exit={ex.exitcode}"
                            f" (likely mkdssp SIGSEGV)"
                        ),
                    }
                except _FuturesTimeoutError:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"task exceeded {task_timeout:.0f}s timeout (killed)"
                        ),
                    }
                except Exception as ex:
                    rec = {
                        "pdb": pdb_name,
                        "error": (
                            f"worker exception: {type(ex).__name__}: "
                            f"{str(ex)[:120]}"
                        ),
                    }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if "agreement_rate" in rec:
                    n_ok += 1
                elif "skipped" in rec:
                    n_skip += 1
                else:
                    n_fail += 1
                progress = len(done_names) + n_ok + n_skip + n_fail
                if progress % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (n_ok + n_skip + n_fail) / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - (n_ok + n_skip + n_fail)) / rate if rate > 0 else 0
                    print(
                        f"[{progress}/{len(sample)}] ok={n_ok} skip={n_skip} fail={n_fail}  "
                        f"rate={rate:.2f}/s  eta={eta/60:.1f}min",
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
    rates: list[float] = []
    n_total = 0
    proteon_comp_total: Counter = Counter()
    mkdssp_comp_total: Counter = Counter()
    with open(OUT) as f:
        for line in f:
            r = json.loads(line)
            if "agreement_rate" not in r:
                continue
            n_total += 1
            rates.append(r["agreement_rate"])
            proteon_comp_total.update(r.get("proteon_composition") or {})
            mkdssp_comp_total.update(r.get("mkdssp_composition") or {})

    if not rates:
        print("\nNo successful records yet.", flush=True)
        return

    arr = np.array(rates)
    print(f"\nPer-residue 8-class agreement (n_ok={n_total}):")
    print(f"  median = {np.median(arr):.4f}")
    print(f"  mean   = {arr.mean():.4f}")
    print(f"  p05    = {np.percentile(arr, 5):.4f}")
    print(f"  p01    = {np.percentile(arr, 1):.4f}")
    print(f"  >=0.95 = {(arr >= 0.95).sum()}/{n_total} ({100*(arr >= 0.95).mean():.1f}%)")
    print(f"  >=0.99 = {(arr >= 0.99).sum()}/{n_total} ({100*(arr >= 0.99).mean():.1f}%)")
    print()
    print(f"Per-class composition (proteon vs mkdssp totals across n_ok records):")
    classes = sorted(set(proteon_comp_total) | set(mkdssp_comp_total))
    for c in classes:
        p = proteon_comp_total.get(c, 0)
        m = mkdssp_comp_total.get(c, 0)
        diff = (p - m) / max(m, 1) if m else 0
        print(f"  {c}: proteon={p:>9d}  mkdssp={m:>9d}  diff={100*diff:+.2f}%")


if __name__ == "__main__":
    main()
