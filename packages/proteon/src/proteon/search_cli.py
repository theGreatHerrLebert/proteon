"""Command-line interface for proteon's structural-alphabet search.

A thin Foldseek-style door onto the existing Python search pipeline
(:func:`proteon.build_search_db` / :func:`proteon.search` /
:func:`proteon.load_search_db`) — it adds *no* search logic of its own, so it
cannot drift from the library API.

    proteon-search build <pdbs-or-dir> -o my_db [-k 6] [-j THREADS]
    proteon-search query my_db query.pdb [--top-k 20] [--no-rerank] [--format json]
    proteon-search inspect my_db

`build` encodes the structures into the 20-state structural alphabet, builds the
k-mer prefilter index, and writes a versioned DB. `query` runs the full pipeline
(prefilter -> diagonal rescore -> optional TM-align rerank) and prints ranked hits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_STRUCTURE_EXTS = {".pdb", ".cif", ".ent", ".mmcif"}


def _gather_inputs(inputs: Sequence[str]) -> List[str]:
    """Expand inputs into a sorted file list: a single directory argument is
    expanded to its structure-file children (non-recursive); otherwise inputs are
    taken verbatim. Sorted for deterministic corpus order."""
    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        files = sorted(
            p for p in Path(inputs[0]).iterdir() if p.suffix.lower() in _STRUCTURE_EXTS
        )
        return [str(p) for p in files]
    return [str(p) for p in inputs]


def _fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def _cmd_build(args: argparse.Namespace) -> int:
    import proteon

    paths = _gather_inputs(args.inputs)
    if not paths:
        print("error: no input structures found", file=sys.stderr)
        return 2
    db = proteon.build_search_db(paths, out=args.out, k=args.k, n_threads=args.threads)
    print(
        f"built {len(db)} entries (k={db.k_values}) from {len(paths)} input(s) -> {args.out}",
        file=sys.stderr,
    )
    return 0


def _hit_row(rank: int, h) -> dict:
    return {
        "rank": rank,
        "id": h.id,
        "score": h.score,
        "tm_score": h.tm_score,
        "rmsd": h.rmsd,
        "n_aligned": h.n_aligned,
        "seq_identity": h.seq_identity,
        "prefilter_score": h.prefilter_score,
        "source_path": h.source_path,
    }


def _cmd_query(args: argparse.Namespace) -> int:
    import proteon

    query = proteon.load(args.query)
    hits = proteon.search(
        query,
        args.db,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
    )
    rows = [_hit_row(i + 1, h) for i, h in enumerate(hits)]
    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        cols = ["rank", "id", "score", "tm_score", "rmsd", "n_aligned", "seq_identity", "prefilter_score"]
        print("\t".join(cols))
        for r in rows:
            print(
                "\t".join(
                    [
                        str(r["rank"]),
                        str(r["id"]),
                        _fmt(r["score"], ".4f"),
                        _fmt(r["tm_score"], ".4f"),
                        _fmt(r["rmsd"], ".3f"),
                        _fmt(r["n_aligned"], "d"),
                        _fmt(r["seq_identity"], ".3f"),
                        _fmt(r["prefilter_score"], ".4f"),
                    ]
                )
            )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    import proteon

    db = proteon.load_search_db(args.db)
    info = {
        "path": str(args.db),
        "version": db.version,
        "n_entries": len(db),
        "k": db.k,
        "k_values": db.k_values,
    }
    if args.format == "json":
        json.dump(info, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for key, val in info.items():
            print(f"{key}\t{val}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteon-search",
        description="Structural-alphabet protein structure search (build / query / inspect).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Encode structures and build a search database.")
    p_build.add_argument("inputs", nargs="+", help="Structure files (PDB/mmCIF) or one directory of them.")
    p_build.add_argument("-o", "--out", required=True, help="Output database directory.")
    p_build.add_argument("-k", type=int, default=6, help="k-mer length for the prefilter index (default 6).")
    p_build.add_argument("-j", "--threads", type=int, default=None, help="Worker threads (default: all cores).")
    p_build.set_defaults(func=_cmd_build)

    p_query = sub.add_parser("query", help="Search a database with a query structure.")
    p_query.add_argument("db", help="Search database directory (from `build`).")
    p_query.add_argument("query", help="Query structure file (PDB/mmCIF).")
    p_query.add_argument("--top-k", type=int, default=10, help="Number of hits to return (default 10).")
    p_query.add_argument("--no-rerank", action="store_true", help="Skip TM-align reranking (prefilter/diagonal only).")
    p_query.add_argument("--rerank-top-k", type=int, default=5, help="How many top hits to TM-align rerank (default 5).")
    p_query.add_argument("--format", choices=["tsv", "json"], default="tsv", help="Output format.")
    p_query.set_defaults(func=_cmd_query)

    p_inspect = sub.add_parser("inspect", help="Print database metadata.")
    p_inspect.add_argument("db", help="Search database directory.")
    p_inspect.add_argument("--format", choices=["tsv", "json"], default="tsv", help="Output format.")
    p_inspect.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
