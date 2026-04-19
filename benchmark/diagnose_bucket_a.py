#!/usr/bin/env python3
"""A/B test to pin down the Bucket A numerical-catastrophe bug.

For each sample PDB, measure:
  1. Min inter-atom distance as loaded
  2. Initial energy as loaded (no H placement)
  3. Energy after place_peptide_hydrogens()
  4. Min inter-atom distance after H placement
  5. The single closest atom pair before and after H placement

If energy is already 10^10+ as loaded → input / parser bug
If energy jumps 10^10x after H placement → H placement bug
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


SAMPLES = [
    # A few small ones first so we can iterate quickly
    "1aie",  # 385 atoms, 30 H placed, E=5.26e18
    "1a1i",  # 1499 atoms, 80 H placed, E=1.14e15
    "1alu",  # 1577 atoms, 152 H placed, E=7.49e16
    "1a6u",  # medium-sized
    "108m",  # 1545 atoms, 149 H placed, E=1.13e13
    # 0-H cases from Bucket A — to test "input is broken" hypothesis
    # (pulled from the full bucket A list)
]


def min_pair_distance(coords: np.ndarray) -> tuple[float, int, int]:
    """Brute force closest pair. coords is (N, 3). Returns (dist, i, j)."""
    n = coords.shape[0]
    # Use KD-tree-ish approach via grid; but for N<20k brute force is fine
    best = (1e30, -1, -1)
    # vectorized: for each i, distance to all j>i
    for i in range(n):
        diff = coords[i + 1 :] - coords[i]
        d2 = (diff * diff).sum(axis=1)
        if len(d2) == 0:
            continue
        j_local = int(d2.argmin())
        d = float(np.sqrt(d2[j_local]))
        if d < best[0]:
            best = (d, i, i + 1 + j_local)
    return best


def atom_info(s, idx: int) -> str:
    """Get a readable atom label for index idx."""
    try:
        return (
            f"idx={idx} name={s.atom_names[idx]} "
            f"res={s.residue_names[idx]}{s.residue_numbers[idx]} "
            f"chain={s.chain_ids[idx]} elem={s.elements[idx]}"
        )
    except Exception as e:
        return f"idx={idx} (label lookup failed: {e})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--out", default="bucket_a_diagnosis.json")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Override sample list")
    args = parser.parse_args()

    import proteon

    ids = args.ids if args.ids else SAMPLES
    results = []

    for pdb_id in ids:
        # Try both .pdb and .cif
        path = None
        for ext in (".pdb", ".cif"):
            cand = Path(args.pdb_dir) / f"{pdb_id}{ext}"
            if cand.exists():
                path = cand
                break
        if path is None:
            print(f"[SKIP] {pdb_id}: file not found")
            continue

        print(f"\n=== {pdb_id} ({path.name}, {os.path.getsize(path)} bytes) ===", flush=True)
        rec = {"pdb_id": pdb_id, "path": str(path)}

        try:
            # Step 1: load as-is
            s = proteon.load(str(path))
            n_atoms = s.atom_count
            print(f"  loaded: {n_atoms} atoms", flush=True)
            rec["atoms_loaded"] = n_atoms
            coords = np.asarray(s.coords)

            # Check for NaN/Inf
            has_nan = bool(np.isnan(coords).any())
            has_inf = bool(np.isinf(coords).any())
            rec["coord_has_nan"] = has_nan
            rec["coord_has_inf"] = has_inf
            if has_nan or has_inf:
                print(f"  !! NaN/Inf in coords (NaN={has_nan}, Inf={has_inf})")

            # Bounding box
            bb_min = coords.min(axis=0).tolist()
            bb_max = coords.max(axis=0).tolist()
            rec["bbox_min"] = bb_min
            rec["bbox_max"] = bb_max
            extent = np.array(bb_max) - np.array(bb_min)
            print(f"  bbox extent: {extent.tolist()}")

            # Min pair dist as loaded
            if n_atoms <= 20000:
                d_before, i_b, j_b = min_pair_distance(coords)
                print(f"  min pair dist (loaded): {d_before:.4f} Å")
                print(f"    {atom_info(s, i_b)}")
                print(f"    {atom_info(s, j_b)}")
                rec["min_dist_loaded"] = d_before
                rec["closest_pair_loaded"] = [atom_info(s, i_b), atom_info(s, j_b)]
            else:
                rec["min_dist_loaded"] = None

            # Step 2: energy as-loaded
            try:
                e_loaded = proteon.compute_energy(s, ff="amber96", units="kJ/mol")
                rec["energy_loaded"] = dict(e_loaded)
                print(f"  energy (loaded): total={e_loaded['total']:.2e} kJ/mol")
                print(f"    bond={e_loaded.get('bond_stretch', 0):.2e}, "
                      f"angle={e_loaded.get('angle_bend', 0):.2e}, "
                      f"vdw={e_loaded.get('vdw', 0):.2e}, "
                      f"elec={e_loaded.get('electrostatic', 0):.2e}")
            except Exception as e:
                rec["energy_loaded_error"] = str(e)
                print(f"  !! energy compute error (loaded): {e}")

            # Step 3: place peptide hydrogens
            s2 = proteon.load(str(path))  # fresh copy
            added, skipped = proteon.place_peptide_hydrogens(s2)
            print(f"  placed {added} H atoms ({skipped} skipped)")
            rec["h_added"] = added
            rec["h_skipped"] = skipped

            # Step 4: min dist after H placement
            coords2 = np.asarray(s2.coords)
            n2 = s2.atom_count
            if n2 <= 20000:
                d_after, i_a, j_a = min_pair_distance(coords2)
                print(f"  min pair dist (after H): {d_after:.4f} Å")
                if d_after < 0.5:
                    print(f"    !! CLASH:")
                    print(f"    {atom_info(s2, i_a)}")
                    print(f"    {atom_info(s2, j_a)}")
                rec["min_dist_after_h"] = d_after
                rec["closest_pair_after_h"] = [atom_info(s2, i_a), atom_info(s2, j_a)]

            # Step 5: energy after H placement
            try:
                e_h = proteon.compute_energy(s2, ff="amber96", units="kJ/mol")
                rec["energy_after_h"] = dict(e_h)
                print(f"  energy (after H): total={e_h['total']:.2e} kJ/mol")
                print(f"    bond={e_h.get('bond_stretch', 0):.2e}, "
                      f"angle={e_h.get('angle_bend', 0):.2e}, "
                      f"vdw={e_h.get('vdw', 0):.2e}, "
                      f"elec={e_h.get('electrostatic', 0):.2e}")
                # Delta
                loaded_total = rec.get("energy_loaded", {}).get("total", 0)
                delta = e_h["total"] - loaded_total
                print(f"  ΔE from H placement: {delta:.2e}")
                rec["delta_e_from_h_placement"] = delta
            except Exception as e:
                rec["energy_after_h_error"] = str(e)
                print(f"  !! energy compute error (after H): {e}")

        except Exception as e:
            rec["error"] = str(e)
            print(f"  !! ERROR: {e}")

        results.append(rec)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
