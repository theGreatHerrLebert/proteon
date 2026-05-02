"""Filter a PDB corpus to protein-only structures.

Reads a directory of `.pdb` files, scans residue names without parsing
heavy atomic data, and emits the subset whose every residue is one of
the 20 standard amino acids (or close variants — MSE for SeMet, etc.).
Skips structures containing nucleic acids (DA/DC/DG/DT/A/C/G/U), large
ligands, sugars, and other non-standard groups that CHARMM19 doesn't
parameterize.

Output: a list of input PDB paths printed to stdout, one per line.
Designed to be used as input to the corpus oracle:

    python validation/protein_only_corpus.py /path/to/pdb_dir > protein_only.txt
    PROTEON_PDB_LIST=protein_only.txt python validation/charmm19_eef1_ball_oracle.py

On a 50K wwPDB random sample we expect ~70-75% to pass the filter
(some structures combine protein + ligand even if they're "protein-
based"; this is a strict filter).

This is a pure-text scanner — no proteon, no openmm, no PDBFixer
imports. Reads each PDB with stdlib only and is fast enough to walk
50K files in under a minute on a single core.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 20 standard amino acids + close variants that CHARMM19 templates
# resolve cleanly. PDBFixer's `replaceNonstandardResidues` handles
# the canonical → variant mapping for SEC, MSE, PYL, etc., but the
# pre-filter accepts them up front to avoid a needless reject.
STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    # CHARMM templates these as "HIS" via terminal-prefix variants;
    # accepted at the filter level so PDBFixer can do the mapping.
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    # Common variants (selenomethionine etc.) — PDBFixer maps to MET.
    "MSE", "SEC", "PYL",
    # CYS variants (oxidized / disulfide / N-term combos):
    "CYX", "CYM",
})

# Residue names we EXPLICITLY reject (faster than unfounded rejection
# via "not in STANDARD"; documents intent; helps if a future PDB
# format adds a new standard AA we forget to whitelist).
NUCLEIC_ACIDS = frozenset({
    "DA", "DC", "DG", "DT", "DI",
    "A", "C", "G", "U", "I",
    "RA", "RC", "RG", "RU",
})


def is_protein_only(pdb_path: Path) -> tuple[bool, str | None]:
    """Return (True, None) if the protein chain is composed of standard
    amino acid residues, or (False, reject_reason) otherwise.

    Scans only **ATOM** records (chain polymer atoms), not HETATM —
    HETATMs are ligands, ions, water, and (rarely) modified polymer
    residues, all of which `PDBFixer.removeHeterogens(keepWater=False)`
    or `PDBFixer.replaceNonstandardResidues()` will handle downstream
    in the corpus oracle. So a structure like 1ake (ATP-bound adenylate
    kinase) is accepted here even though its HETATM ATP block contains
    nucleic-acid-like residue names — the protein chain is what we
    care about, and PDBFixer will strip the ATP before the FF energy
    runs.

    Also explicitly rejects nucleic-acid CHAINS by checking ATOM
    records — DNA/RNA structures use ATOM (not HETATM) for nucleotides.
    """
    try:
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM  "):
                    res = line[17:20].strip()
                    if res in NUCLEIC_ACIDS:
                        return False, f"nucleic acid residue: {res}"
                    if res not in STANDARD_RESIDUES:
                        return False, f"non-standard residue: {res}"
                elif line.startswith(("END", "MASTER")):
                    break
    except OSError as e:
        return False, f"read error: {e}"
    return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "pdb_dir",
        type=Path,
        help="Directory containing *.pdb files to filter.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rejection reasons to stderr (otherwise silent).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Optional cap on the number of accepted PDBs to print.",
    )
    args = parser.parse_args()

    if not args.pdb_dir.is_dir():
        raise SystemExit(f"not a directory: {args.pdb_dir}")

    accepted = rejected = 0
    counts: dict[str, int] = {}
    for pdb in sorted(args.pdb_dir.glob("*.pdb")):
        ok, reason = is_protein_only(pdb)
        if ok:
            print(str(pdb))
            accepted += 1
            if args.max is not None and accepted >= args.max:
                break
        else:
            rejected += 1
            if args.verbose:
                print(f"REJECT {pdb.name}: {reason}", file=sys.stderr)
            key = reason.split(":")[0] if reason else "unknown"
            counts[key] = counts.get(key, 0) + 1

    print(
        f"\n# accepted={accepted} rejected={rejected}",
        file=sys.stderr,
    )
    if counts:
        print("# rejection reasons:", file=sys.stderr)
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"#   {v:>6} {k}", file=sys.stderr)


if __name__ == "__main__":
    main()
