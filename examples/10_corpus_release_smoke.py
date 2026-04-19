"""Build a small end-to-end corpus release from local structure files.

Requires the native proteon connector to be installed.

Example:
    PYTHONPATH=packages/proteon/src python examples/10_corpus_release_smoke.py \
        --out smoke_release \
        test-pdbs/1crn.pdb test-pdbs/1ubq.pdb test-pdbs/1bpi.pdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import proteon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Input PDB/mmCIF files")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--release-id", default="local-smoke", help="Corpus release id")
    ap.add_argument("--n-threads", type=int, default=None, help="Thread count")
    args = ap.parse_args()

    out = proteon.build_local_corpus_smoke_release(
        [Path(p) for p in args.paths],
        Path(args.out),
        release_id=args.release_id,
        n_threads=args.n_threads,
        overwrite=True,
    )
    print(out)


if __name__ == "__main__":
    main()
