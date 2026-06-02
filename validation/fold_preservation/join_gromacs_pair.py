"""Join the proteon AMBER96 and GROMACS AMBER96 fold-preservation runs.

Third-oracle triangulation for AMBER96 fold preservation: proteon's AMBER96
minimizer vs GROMACS's AMBER96 minimizer on the same 1k seeded sample
(issue #37). Where proteon-vs-OpenMM (amber_pair_1k.jsonl) gives one
independent comparison, this adds GROMACS as a third, independent C lineage.

Inputs (per-side sweeps, keyed by pdb):
  * validation/fold_preservation/tm_fold_preservation_amber.jsonl  (proteon)
  * validation/gmx_fold_preservation/tm_fold_gromacs.jsonl         (GROMACS)

Output:
  * validation/fold_preservation/gromacs_amber_pair_1k.jsonl

Per ok record: {pdb, proteon:{tm_score,...}, gromacs:{tm_score,...},
tm_diff = gromacs.tm_score - proteon.tm_score}. A PDB is ok only when BOTH
arms produced a tm_score; otherwise it is skipped/errored with the per-side
classification preserved. GROMACS skips (pdb2gmx-rejected, out-of-population)
are distinct from GROMACS errors (post-topology pipeline failures).
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

PROTEON = HERE / "tm_fold_preservation_amber.jsonl"
GROMACS = REPO / "validation" / "gmx_fold_preservation" / "tm_fold_gromacs.jsonl"
OUT = HERE / "gromacs_amber_pair_1k.jsonl"


def _load_by_pdb(path: pathlib.Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("pdb"):
                out[d["pdb"]] = d
    return out


def _classify(rec: dict) -> str:
    if not rec:
        return "missing"
    if rec.get("error"):
        return "error"
    if rec.get("skipped"):
        return "skipped"
    if "tm_score" in rec:
        return "ok"
    return "error"


def main() -> int:
    p = _load_by_pdb(PROTEON)
    g = _load_by_pdb(GROMACS)
    pdbs = sorted(set(p) | set(g))

    n_ok = n_skip = n_err = 0
    with OUT.open("w", encoding="utf-8") as f:
        for pdb in pdbs:
            pr, gr = p.get(pdb) or {}, g.get(pdb) or {}
            cp, cg = _classify(pr), _classify(gr)
            if cp != "ok" or cg != "ok":
                rec = {"pdb": pdb, "skipped": f"proteon={cp}; gromacs={cg}"}
                if cg == "skipped":
                    rec["gromacs_skip"] = gr.get("skipped")
                if cp == "error":
                    rec["proteon_error"] = pr.get("error")
                if cg == "error":
                    rec["gromacs_error"] = gr.get("error")
                # A pair is "skipped" (out of population) unless a genuine
                # pipeline error occurred on either side.
                if cp == "error" or cg == "error":
                    n_err += 1
                else:
                    n_skip += 1
                f.write(json.dumps(rec) + "\n")
                continue

            n_ok += 1
            tm_p, tm_g = float(pr["tm_score"]), float(gr["tm_score"])
            rec = {
                "pdb": pdb,
                "n_ca": int(pr.get("n_ca") or gr.get("n_ca") or 0),
                "proteon": {
                    "tm_score": tm_p,
                    "rmsd": float(pr.get("rmsd", 0.0)),
                    "final_energy": pr.get("final_energy"),
                    "label": "proteon AMBER96 (vacuum)",
                },
                "gromacs": {
                    "tm_score": tm_g,
                    "rmsd": float(gr.get("rmsd", 0.0)),
                    "final_energy_kj": gr.get("final_energy_kj"),
                    "label": "GROMACS AMBER96 (vacuum)",
                },
                "tm_diff": tm_g - tm_p,        # gromacs - proteon
                "rmsd_diff_A": float(gr.get("rmsd", 0.0)) - float(pr.get("rmsd", 0.0)),
            }
            f.write(json.dumps(rec) + "\n")

    print(f"{OUT.name}: ok={n_ok} skip={n_skip} err={n_err} of {len(pdbs)} PDBs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
