#!/usr/bin/env python3
"""Retrieval-quality benchmark for proteon structural search.

This benchmark measures recall against exact TM-align ground truth on a sampled
subset of the corpus. It is intended to answer a concrete question:

"When proteon search retrieves top-k candidates, how often does it recover the
true structural neighbors that exhaustive TM-align would find?"

The benchmark builds or loads a search DB over a sampled target set, computes
exact TM-align scores query-vs-target, and compares:
- prefilter top-k hits
- reranked top-k hits

Usage:
    python validation/bench_retrieval.py
    python validation/bench_retrieval.py --pdb-dir validation/pdbs_10k --n-targets 500 --n-queries 25
    python validation/bench_retrieval.py --db-path /tmp/proteon_retrieval_eval --compile-db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from statistics import mean

import proteon


def collect_structure_paths(pdb_dir: Path) -> list[Path]:
    exts = {".pdb", ".cif", ".mmcif"}
    return sorted(p for p in pdb_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def sample_targets(paths: list[Path], n_targets: int, seed: int) -> list[Path]:
    if n_targets >= len(paths):
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n_targets))


def sample_queries(targets: list[Path], n_queries: int, seed: int) -> list[Path]:
    if n_queries >= len(targets):
        return targets
    rng = random.Random(seed + 1)
    return sorted(rng.sample(targets, n_queries))


def build_truth_for_query(
    query_path: Path,
    query_structure,
    target_paths: list[Path],
    target_structures: list,
) -> list[dict]:
    results = proteon.tm_align_one_to_many(
        query_structure,
        target_structures,
        n_threads=-1,
        fast=True,
    )
    rows = []
    for path, result in zip(target_paths, results):
        score = max(result.tm_score_chain1, result.tm_score_chain2)
        rows.append(
            {
                "source_path": str(path),
                "tm_score": float(score),
                "rmsd": float(result.rmsd),
                "n_aligned": int(result.n_aligned),
            }
        )
    rows.sort(key=lambda row: (-row["tm_score"], Path(row["source_path"]).name))
    return rows


def recall_at_k(
    hits,
    truth_rows: list[dict],
    *,
    k: int,
    tm_threshold: float,
    exclude_self_path: Path,
) -> float:
    relevant = {
        row["source_path"]
        for row in truth_rows
        if row["tm_score"] >= tm_threshold and Path(row["source_path"]) != exclude_self_path
    }
    if not relevant:
        return 1.0
    retrieved = {hit.source_path for hit in hits[:k]}
    return len(relevant & retrieved) / len(relevant)


def best_nonself_hit(rows: list[dict], *, query_path: Path) -> dict | None:
    for row in rows:
        if Path(row["source_path"]) != query_path:
            return row
    return None


def truth_cache_key(
    *,
    pdb_dir: Path,
    target_paths: list[Path],
    query_paths: list[Path],
    seed: int,
) -> str:
    payload = {
        "pdb_dir": str(pdb_dir),
        "targets": [str(path) for path in target_paths],
        "queries": [str(path) for path in query_paths],
        "seed": seed,
        "aligner": "proteon.tm_align_one_to_many.fast",
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_truth_cache(path: Path, expected_key: str) -> dict[str, list[dict]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_key") != expected_key:
        return None
    return payload.get("truth", {})


def save_truth_cache(path: Path, *, cache_key: str, truth: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "truth": truth,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark proteon retrieval quality against sampled brute-force TM-align")
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/proteon/validation/pdbs_10k")
    parser.add_argument("--db-path", default="/tmp/proteon_retrieval_eval")
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--k-values", nargs="*", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--diagonal-top-k", type=int, default=200)
    parser.add_argument("--alphabet-weight", type=float, default=0.7)
    parser.add_argument("--aa-weight", type=float, default=0.3)
    parser.add_argument("--no-diagonal-rescore", action="store_true")
    parser.add_argument("--no-diagonal-prefilter", action="store_true")
    parser.add_argument("--diagonal-min-support", type=float, default=1.0)
    parser.add_argument("--diagonal-prefilter-top-k", type=int, default=1000)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-db", action="store_true")
    parser.add_argument("--warm-db", action="store_true")
    parser.add_argument("--reuse-db", action="store_true")
    parser.add_argument("--truth-cache", default=None)
    parser.add_argument("--reuse-truth", action="store_true")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--output", default="validation/retrieval_bench.json")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    all_paths = collect_structure_paths(pdb_dir)
    if not all_paths:
        raise SystemExit(f"No structures found in {pdb_dir}")

    target_paths = sample_targets(all_paths, args.n_targets, args.seed)
    query_paths = sample_queries(target_paths, args.n_queries, args.seed)
    db_path = Path(args.db_path)
    build_k = args.k_values if args.k_values else args.k

    print(f"Corpus sample: {len(target_paths)} targets from {pdb_dir}")
    print(f"Queries: {len(query_paths)}")
    print(f"DB path: {db_path}")

    t0 = time.time()
    if args.reuse_db and db_path.exists():
        db = proteon.load_search_db(db_path)
    else:
        db = proteon.build_search_db(target_paths, out=db_path, k=build_k, n_threads=-1)
    build_s = time.time() - t0

    if args.warm_db:
        t0 = time.time()
        proteon.warm_search_db(db, posting_cache_max_size=128)
        warm_s = time.time() - t0
    else:
        warm_s = 0.0

    if args.compile_db:
        t0 = time.time()
        db = proteon.compile_search_db(db)
        compile_s = time.time() - t0
    else:
        compile_s = 0.0

    loaded_targets = proteon.batch_load_tolerant(target_paths, n_threads=-1)
    if not loaded_targets:
        raise SystemExit("No target structures could be loaded for brute-force evaluation")
    loaded_target_map = {
        str(target_paths[idx]): structure
        for idx, structure in loaded_targets
    }
    target_paths = [Path(entry.source_path) for entry in (db.entries or [])]
    target_structures = [loaded_target_map[str(path)] for path in target_paths if str(path) in loaded_target_map]
    target_paths = [path for path in target_paths if str(path) in loaded_target_map]
    query_paths = [path for path in query_paths if str(path) in loaded_target_map]
    if not query_paths:
        raise SystemExit("No query structures remain after filtering to loadable indexed targets")

    truth_cache_path = Path(args.truth_cache) if args.truth_cache else Path(str(args.output) + ".truth.json")
    cache_key = truth_cache_key(
        pdb_dir=pdb_dir,
        target_paths=target_paths,
        query_paths=query_paths,
        seed=args.seed,
    )
    truth_cache = load_truth_cache(truth_cache_path, cache_key) if args.reuse_truth else None
    truth_cache_hit = truth_cache is not None
    if truth_cache is None:
        truth_cache = {}

    threshold_metrics = {
        str(threshold): {"prefilter": [], "rerank": []}
        for threshold in args.thresholds
    }
    top1_exact_prefilter = []
    top1_exact_rerank = []
    per_query = []
    brute_force_s = 0.0
    prefilter_s = 0.0
    rerank_s = 0.0

    for query_path in query_paths:
        query = proteon.load(query_path)

        query_key = str(query_path)
        if query_key in truth_cache:
            truth_rows = truth_cache[query_key]
        else:
            t0 = time.time()
            truth_rows = build_truth_for_query(query_path, query, target_paths, target_structures)
            brute_force_s += time.time() - t0
            truth_cache[query_key] = truth_rows

        t0 = time.time()
        pre_hits = proteon.search(
            query,
            db,
            top_k=args.top_k,
            rerank=False,
            alphabet_weight=args.alphabet_weight,
            aa_weight=args.aa_weight,
            diagonal_rescore=not args.no_diagonal_rescore,
            diagonal_top_k=args.diagonal_top_k,
            diagonal_prefilter=not args.no_diagonal_prefilter,
            diagonal_min_support=args.diagonal_min_support,
            diagonal_prefilter_top_k=args.diagonal_prefilter_top_k,
        )
        prefilter_s += time.time() - t0

        t0 = time.time()
        rr_hits = proteon.search(
            query,
            db,
            top_k=args.top_k,
            rerank=not args.no_rerank,
            rerank_top_k=args.rerank_top_k,
            alphabet_weight=args.alphabet_weight,
            aa_weight=args.aa_weight,
            diagonal_rescore=not args.no_diagonal_rescore,
            diagonal_top_k=args.diagonal_top_k,
            diagonal_prefilter=not args.no_diagonal_prefilter,
            diagonal_min_support=args.diagonal_min_support,
            diagonal_prefilter_top_k=args.diagonal_prefilter_top_k,
        )
        rerank_s += time.time() - t0

        best_truth = best_nonself_hit(truth_rows, query_path=query_path)
        pre_top_nonself = next((hit for hit in pre_hits if Path(hit.source_path) != query_path), None)
        rr_top_nonself = next((hit for hit in rr_hits if Path(hit.source_path) != query_path), None)
        top1_exact_prefilter.append(
            int(best_truth is not None and pre_top_nonself is not None and pre_top_nonself.source_path == best_truth["source_path"])
        )
        top1_exact_rerank.append(
            int(best_truth is not None and rr_top_nonself is not None and rr_top_nonself.source_path == best_truth["source_path"])
        )

        per_query_row = {
            "query": query_path.name,
            "best_truth_nonself": None if best_truth is None else {
                "source_path": Path(best_truth["source_path"]).name,
                "tm_score": round(best_truth["tm_score"], 4),
            },
            "prefilter_top_nonself": None if pre_top_nonself is None else {
                "source_path": Path(pre_top_nonself.source_path).name,
                "prefilter_score": round(pre_top_nonself.prefilter_score, 4),
            },
            "rerank_top_nonself": None if rr_top_nonself is None else {
                "source_path": Path(rr_top_nonself.source_path).name,
                "tm_score": None if rr_top_nonself.tm_score is None else round(rr_top_nonself.tm_score, 4),
            },
            "thresholds": {},
        }

        for threshold in args.thresholds:
            pre_recall = recall_at_k(
                pre_hits, truth_rows, k=args.top_k, tm_threshold=threshold, exclude_self_path=query_path
            )
            rr_recall = recall_at_k(
                rr_hits, truth_rows, k=args.top_k, tm_threshold=threshold, exclude_self_path=query_path
            )
            threshold_metrics[str(threshold)]["prefilter"].append(pre_recall)
            threshold_metrics[str(threshold)]["rerank"].append(rr_recall)
            per_query_row["thresholds"][str(threshold)] = {
                "prefilter_recall_at_k": round(pre_recall, 4),
                "rerank_recall_at_k": round(rr_recall, 4),
            }

        per_query.append(per_query_row)
        print(
            f"{query_path.name}: truth {per_query_row['best_truth_nonself']} "
            f"prefilter {per_query_row['prefilter_top_nonself']} "
            f"rerank {per_query_row['rerank_top_nonself']}",
            flush=True,
        )

    if not truth_cache_hit:
        save_truth_cache(truth_cache_path, cache_key=cache_key, truth=truth_cache)

    summary = {
        "corpus": {
            "pdb_dir": str(pdb_dir),
            "db_path": str(db_path),
            "n_targets": len(target_paths),
            "n_queries": len(query_paths),
            "k": args.k,
            "k_values": list(getattr(db, "k_values", [args.k])),
            "top_k": args.top_k,
            "rerank_top_k": args.rerank_top_k,
            "diagonal_top_k": args.diagonal_top_k,
            "alphabet_weight": args.alphabet_weight,
            "aa_weight": args.aa_weight,
            "diagonal_rescore": not args.no_diagonal_rescore,
            "diagonal_prefilter": not args.no_diagonal_prefilter,
            "diagonal_min_support": args.diagonal_min_support,
            "diagonal_prefilter_top_k": args.diagonal_prefilter_top_k,
            "rerank": not args.no_rerank,
            "thresholds": args.thresholds,
            "truth_cache": str(truth_cache_path),
            "truth_cache_hit": truth_cache_hit,
        },
        "timing": {
            "build_s": round(build_s, 3),
            "warm_s": round(warm_s, 3),
            "compile_s": round(compile_s, 3),
            "brute_force_s": round(brute_force_s, 3),
            "prefilter_s": round(prefilter_s, 3),
            "rerank_s": round(rerank_s, 3),
        },
        "metrics": {
            "top1_exact_nonself_prefilter": round(mean(top1_exact_prefilter), 4) if top1_exact_prefilter else 0.0,
            "top1_exact_nonself_rerank": round(mean(top1_exact_rerank), 4) if top1_exact_rerank else 0.0,
            "recall_at_k": {
                threshold: {
                    "prefilter_mean": round(mean(values["prefilter"]), 4) if values["prefilter"] else 0.0,
                    "rerank_mean": round(mean(values["rerank"]), 4) if values["rerank"] else 0.0,
                }
                for threshold, values in threshold_metrics.items()
            },
        },
        "per_query": per_query,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
