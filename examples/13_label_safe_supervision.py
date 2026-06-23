"""Test-drive the label-safe supervision path: raw PDB -> masked training tensors.

The whole subsystem in one runnable flow — load + prepare a diverse set, gate on
per-residue coverage, and export masked supervision tensors for the survivors,
both single chains and verified complexes. Prints what was kept, what was
dropped and why, and the mask statistics that make the labels trustworthy.

    PYTHONPATH=packages/proteon/src python examples/13_label_safe_supervision.py \
        test-pdbs/1crn.pdb test-pdbs/1ubq.pdb test-pdbs/4hhb.pdb test-pdbs/1ake.pdb
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import proteon
from proteon.supervision import (
    ComplexSupervisionExamples,
    build_complex_supervision_examples,
)


def _mask_stats(ex):
    """(residues, residues with a usable heavy-atom label, fraction masked)."""
    rows = ex.all_atom_mask.sum(axis=1)
    usable = int((rows > 0).sum())
    return ex.length, usable, 1.0 - usable / ex.length if ex.length else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Input PDB/mmCIF files")
    ap.add_argument("--floor", type=float, default=0.8, help="Min per-residue coverage")
    args = ap.parse_args()

    print(f"\n=== Label-safe supervision test drive ({len(args.paths)} structures, "
          f"coverage floor {args.floor}) ===\n")

    # One call does load + prepare + assembly annotation, label-safe by default
    # (reconstruct=False). minimize=False — every gate signal is structural.
    results = proteon.prepare_for_supervision(args.paths, minimize=False, n_threads=-1)

    monomer_examples = []
    complex_bundles = []

    for res in results:
        name = os.path.basename(res.path)
        if not res.loaded or res.report is None:
            print(f"  {name:14s} LOAD FAILED: {res.error}")
            continue
        rep = res.report
        n_chains = sum(
            1 for ch in res.structure.models[0].chains if any(r.is_amino_acid for r in ch.residues)
        )

        if n_chains >= 2:
            # Interface path: verified-assembly gate + per-chain coverage + masking.
            out = build_complex_supervision_examples(
                res.structure, prep_report=rep, min_coverage=args.floor)
            if isinstance(out, ComplexSupervisionExamples):
                chains = ", ".join(
                    f"{cid}:{out.coverage.chains[cid].coverage:.0%}" for cid in out.chain_order)
                print(f"  {name:14s} COMPLEX KEPT  chains [{chains}]  "
                      f"clashscore={rep.clashscore:.0f}")
                complex_bundles.append(out)
            else:
                print(f"  {name:14s} complex dropped: {out}")
            continue

        # Monomer path: single-chain coverage gate + masked export.
        cid = next(ch.id for ch in res.structure.models[0].chains
                   if any(r.is_amino_acid for r in ch.residues))
        cov = proteon.structure_coverage(res.structure, chain_id=cid, report=rep)
        from proteon.prepare import unmasked_heavy_coord_hazards
        blocked = unmasked_heavy_coord_hazards(rep)
        if cov.coverage < args.floor:
            print(f"  {name:14s} monomer dropped: coverage {cov.coverage:.0%} < floor")
        elif blocked:
            print(f"  {name:14s} monomer dropped: unmasked {sorted(blocked)}")
        else:
            ex = proteon.build_structure_supervision_example(
                res.structure, chain_id=cid, prep_report=rep, mask_untrustworthy_coords=True)
            n, usable, frac = _mask_stats(ex)
            print(f"  {name:14s} MONOMER KEPT  {n} res, {usable} usable "
                  f"({frac:.0%} masked)  clashscore={rep.clashscore:.0f}")
            monomer_examples.append((name, ex))

    print(f"\n=== Result: {len(monomer_examples)} monomer + {len(complex_bundles)} complex "
          f"training examples ===\n")

    # Show that the kept tensors are real, masked, and consistent.
    if monomer_examples:
        name, ex = monomer_examples[0]
        print(f"Sample monomer tensor ({name}):")
        print(f"  atom37 positions : {ex.all_atom_positions.shape}  "
              f"({np.isfinite(ex.all_atom_positions).all()} all finite)")
        print(f"  atom37 loss mask : {ex.all_atom_mask.shape}  "
              f"{int(ex.all_atom_mask.sum())} atom labels")
        print(f"  torsion mask     : {ex.torsion_angles_mask.shape}  "
              f"{int(ex.torsion_angles_mask.sum())} torsion labels")
        print(f"  seq_mask intact  : {int(ex.seq_mask.sum())}/{ex.length} "
              f"(identity never masked)")

    if complex_bundles:
        b = complex_bundles[0]
        total = sum(e.length for e in b.chain_examples.values())
        usable = sum(_mask_stats(e)[1] for e in b.chain_examples.values())
        print(f"\nSample complex ({b.record_id}): {len(b.chain_order)} chains, "
              f"{total} residues, {usable} usable ({1 - usable / total:.0%} masked) "
              f"— cross-chain-correct, ready for interface pair labels.")


if __name__ == "__main__":
    main()
