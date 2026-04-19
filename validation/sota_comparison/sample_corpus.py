#!/usr/bin/env python3
"""Sample N PDBs from the monster3 50K corpus for scaling tests.

Filters out nucleic acid structures (AMBER96 has no templates, would be
flagged as skipped_no_protein by proteon anyway) and huge files. The
output is a list of PDB IDs (stem of each filename) one per line, suitable
for feeding to `run_all.py --pdbs @sample.txt`.

Usage:
    python validation/sota_comparison/sample_corpus.py \\
        --pdb-dir /globalscratch/dateschn/proteon-benchmark/pdbs_50k \\
        --n 100 \\
        --seed 42 \\
        --output validation/sota_comparison/sample_100.txt
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Residue name patterns that mark a file as nucleic acid or mixed complex.
# If any of these show up in an ATOM record, we skip the file.
_NA_RESIDUES = frozenset({
    "DA", "DT", "DG", "DC", "DU", "DI",   # DNA
    "A", "T", "G", "C", "U",               # RNA (bare names)
    "RA", "RU", "RG", "RC",                # RNA (alt names)
})


def is_protein_only(pdb_path: Path, max_check_lines: int = 5000) -> bool:
    """Quick-check a PDB file for nucleic acid content.

    Only scans the first `max_check_lines` ATOM records to keep this fast.
    Returns False if any nucleic acid residue is found. The 5000-line cap
    is enough to catch nucleic acid content in the canonical chain order;
    protein-only structures will typically see >= 5000 protein residues
    without hitting any NA residues in the first section.
    """
    try:
        with open(pdb_path, "rb") as f:
            for i, line in enumerate(f):
                if i > max_check_lines:
                    break
                if not line.startswith(b"ATOM"):
                    continue
                # resname is columns 18-20 (1-indexed) in PDB format
                resname = line[17:20].strip().decode("ascii", "ignore")
                if resname in _NA_RESIDUES:
                    return False
    except (IOError, OSError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-size", type=int, default=20_000_000,
                        help="Skip files larger than this many bytes (default 20 MB)")
    parser.add_argument("--min-file-size", type=int, default=5_000,
                        help="Skip files smaller than this many bytes (default 5 KB)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    all_files = sorted(args.pdb_dir.glob("*.pdb"))
    print(f"Found {len(all_files)} PDB files in {args.pdb_dir}", file=sys.stderr)

    # Size filter
    size_ok = [
        f for f in all_files
        if args.min_file_size <= os.path.getsize(f) <= args.max_file_size
    ]
    print(f"  after size filter: {len(size_ok)}", file=sys.stderr)

    # Shuffle then walk until we have N protein-only files
    rng = random.Random(args.seed)
    rng.shuffle(size_ok)

    accepted = []
    rejected_na = 0
    for f in size_ok:
        if len(accepted) >= args.n:
            break
        if is_protein_only(f):
            accepted.append(f)
        else:
            rejected_na += 1

    if len(accepted) < args.n:
        print(
            f"WARNING: only found {len(accepted)} protein-only files (wanted {args.n})",
            file=sys.stderr,
        )
    print(
        f"  accepted: {len(accepted)}, rejected (nucleic acid): {rejected_na}",
        file=sys.stderr,
    )

    # Write PDB IDs (stem) one per line
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# Sampled {len(accepted)} protein-only PDBs from {args.pdb_dir}\n")
        f.write(f"# seed={args.seed} min_size={args.min_file_size} max_size={args.max_file_size}\n")
        for p in sorted(accepted, key=lambda x: x.stem):
            f.write(f"{p.stem}\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
