#!/usr/bin/env python3
"""Diagnose which structures / operations cause OOM in the 50K benchmark.

Run on monster3:
    cd /globalscratch/dateschn/ferritin-benchmark/ferritin
    source ../venv/bin/activate
    python benchmark/diagnose_oom.py --pdb-dir ../pdbs_50k --n 5000
"""

import argparse
import os
import sys
import time
from pathlib import Path


def log(msg):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--n", type=int, default=5000, help="Number of files (first chunk)")
    args = parser.parse_args()

    import ferritin

    # 1. Collect files (same logic as run_benchmark.py chunk 1)
    files = sorted(Path(args.pdb_dir).glob("*.pdb")) + sorted(Path(args.pdb_dir).glob("*.cif"))
    files = [str(f) for f in files[: args.n]]
    files = [f for f in files if os.path.getsize(f) < 20_000_000]
    log(f"Files after 20MB filter: {len(files)}")

    # 2. Load
    log("Loading...")
    loaded = ferritin.batch_load_tolerant(files, n_threads=0)
    structures = [s for _, s in loaded]
    log(f"Loaded: {len(structures)}")

    # 3. Audit every structure: atom_count vs actual atoms via chains()
    log("\n=== STRUCTURE AUDIT ===")
    log(f"{'idx':>5s} {'pdb_id':>8s} {'atom_count':>10s} {'n_models':>8s} {'total_atoms':>12s} {'ratio':>6s} {'bbox_span':>10s}")

    suspects = []
    for i, s in enumerate(structures):
        ac = s.atom_count
        nm = s.model_count
        # Total atoms across ALL models (what chains()/extract_radii sees)
        ta = s.total_atom_count if hasattr(s, "total_atom_count") else ac * nm  # estimate
        ratio = ta / ac if ac > 0 else 0

        # Bounding box span (rough estimate from first model)
        try:
            coords = s.coords  # Nx3 numpy array if available
            if coords is not None and len(coords) > 0:
                span = coords.max(axis=0) - coords.min(axis=0)
                bbox = f"{max(span):.0f}"
            else:
                bbox = "?"
        except Exception:
            bbox = "?"

        pdb_id = Path(files[i]).stem if i < len(files) else "?"

        if nm > 1 or ac > 20000:
            suspects.append((i, pdb_id, ac, nm, ta, bbox))
            log(f"{i:5d} {pdb_id:>8s} {ac:10d} {nm:8d} {ta:12d} {ratio:6.1f}x {bbox:>10s}")

    log(f"\nSuspects (multi-model or large): {len(suspects)}")
    log(f"Structures with >1 model: {sum(1 for s in structures if s.model_count > 1)}")
    log(f"Structures with >25K atoms (model 0): {sum(1 for s in structures if s.atom_count > 25000)}")

    # 4. Apply the same filter as the benchmark
    structures_filtered = [s for s in structures if s.atom_count < 25000]
    log(f"\nAfter 25K atom_count filter: {len(structures_filtered)}")
    multi_model = [s for s in structures_filtered if s.model_count > 1]
    log(f"Multi-model structures that PASSED the filter: {len(multi_model)}")
    if multi_model:
        total_effective_atoms = sum(s.atom_count * s.model_count for s in multi_model)
        log(f"Effective atoms from multi-model (atom_count × n_models): {total_effective_atoms:,}")

    # 5. Test batch_total_sasa on single structures to find the one that OOMs
    log("\n=== SASA SINGLE-STRUCTURE TEST ===")
    log("Testing each structure individually (will catch the OOM trigger)...")

    for i, s in enumerate(structures_filtered):
        nm = s.model_count
        ac = s.atom_count
        effective = ac * nm
        if effective > 50000 or nm > 5:
            log(f"  [{i}] models={nm} atom_count={ac} effective={effective} — testing SASA...")
            try:
                t0 = time.perf_counter()
                ferritin.batch_total_sasa([s], n_threads=1, radii="protor")
                dt = time.perf_counter() - t0
                log(f"       OK in {dt:.2f}s")
            except Exception as e:
                log(f"       FAILED: {e}")

    # 6. Test batch_total_sasa on increasing batches
    log("\n=== SASA BATCH SCALING TEST ===")
    for batch_size in [100, 500, 1000, 2000, len(structures_filtered)]:
        batch = structures_filtered[:batch_size]
        effective = sum(s.atom_count * s.model_count for s in batch)
        log(f"  batch={len(batch)} effective_atoms={effective:,} ... ", end="")
        try:
            t0 = time.perf_counter()
            ferritin.batch_total_sasa(batch, n_threads=0, radii="protor")
            dt = time.perf_counter() - t0
            log(f"OK in {dt:.1f}s")
        except Exception as e:
            log(f"FAILED: {e}")
            break

    # 7. Quick test of other operations on full filtered set
    log("\n=== OTHER OPERATIONS TEST ===")
    ops = [
        ("batch_dssp", lambda: ferritin.batch_dssp(structures_filtered, n_threads=0)),
        ("batch_dihedrals", lambda: ferritin.batch_dihedrals(structures_filtered, n_threads=0)),
        ("batch_backbone_hbonds", lambda: ferritin.batch_backbone_hbonds(structures_filtered, n_threads=0)),
        ("batch_place_peptide_hydrogens", lambda: ferritin.batch_place_peptide_hydrogens(structures_filtered[:500], n_threads=0)),
    ]
    for name, fn in ops:
        log(f"  {name}... ", end="")
        try:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            log(f"OK in {dt:.1f}s")
        except Exception as e:
            log(f"FAILED: {e}")


if __name__ == "__main__":
    main()
