#!/usr/bin/env python
"""How much of the `requires_assembly_expansion` bucket the assembly builder
recovers as label-safe interface supervision (PR3 of the assembly builder).

For each structure whose deposited ASU is NOT the biological assembly
(`assembly_is_asu is False`), build the assembly from BIOMT, re-prepare it (so the
new inter-copy interfaces are validated), and run the interface gate. Reports the
fraction recovered as `ComplexSupervisionExamples` and the drop-reason breakdown.

Usage: python validation/eval_assembly_recovery.py [N] [--floor 0.8]
"""

import collections
import glob
import os
import sys

import proteon
from proteon.assembly_builder import build_assembly_supervision_examples
from proteon.supervision import ComplexSupervisionExamples

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdbs_10k")


def main():
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 500)
    floor = 0.8
    for i, a in enumerate(sys.argv):
        if a == "--floor":
            floor = float(sys.argv[i + 1])

    paths = [p for p in sorted(glob.glob(os.path.join(CORPUS, "*.pdb"))) if os.path.exists(p)]
    stride = max(1, len(paths) // n)
    paths = paths[::stride][:n]

    res = proteon.prepare_for_supervision(paths, minimize=False, n_threads=-1)
    expansion = [r.path for r in res
                 if r.loaded and r.report is not None and r.report.assembly_is_asu is False]
    print(f"requires_assembly_expansion: {len(expansion)} of {len(paths)} sampled", file=sys.stderr)

    out = collections.Counter()
    chain_counts = []
    for p in expansion:
        r = build_assembly_supervision_examples(p, min_coverage=floor)
        if isinstance(r, ComplexSupervisionExamples):
            out["KEPT"] += 1
            chain_counts.append(len(r.chain_order))
        else:
            out[r] += 1

    kept = out.get("KEPT", 0)
    print(f"\nfloor={floor}  recovered {kept}/{len(expansion)} "
          f"= {100 * kept / max(len(expansion), 1):.0f}% as label-safe complexes")
    for k, v in out.most_common():
        print(f"  {k:32s} {v}")
    if chain_counts:
        print(f"recovered complex chain counts: {dict(collections.Counter(chain_counts))}")


if __name__ == "__main__":
    main()
