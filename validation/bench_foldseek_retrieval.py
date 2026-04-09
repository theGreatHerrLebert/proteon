#!/usr/bin/env python3
"""Retrieval-quality benchmark for Foldseek on ferritin's sampled corpus.

This mirrors validation/bench_retrieval.py so Foldseek and ferritin can be
compared against the same sampled brute-force TM-align truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import mean

import ferritin


FOLDSEEK = Path("/scratch/TMAlign/foldseek/bin/foldseek")


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


def assert_unique_stems(paths: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if stem in seen:
            raise SystemExit(f"Duplicate structure stem {stem!r}: {seen[stem]} and {path}")
        seen[stem] = path


def stage_paths(paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dest = out_dir / path.name
        try:
            os.symlink(path, dest)
        except OSError:
            shutil.copy2(path, dest)


def build_truth_for_query(
    query_structure,
    target_paths: list[Path],
    target_structures: list,
) -> list[dict]:
    results = ferritin.tm_align_one_to_many(
        query_structure,
        target_structures,
        n_threads=-1,
        fast=True,
    )
    rows = []
    for path, result in zip(target_paths, results):
        rows.append(
            {
                "source_path": str(path),
                "tm_score": float(max(result.tm_score_chain1, result.tm_score_chain2)),
                "rmsd": float(result.rmsd),
                "n_aligned": int(result.n_aligned),
            }
        )
    rows.sort(key=lambda row: (-row["tm_score"], Path(row["source_path"]).name))
    return rows


def load_structures_for_paths(paths: list[Path]) -> tuple[list[Path], list]:
    loaded = ferritin.batch_load_tolerant(paths, n_threads=-1)
    loaded_paths = [paths[idx] for idx, _structure in loaded]
    structures = [structure for _idx, structure in loaded]
    return loaded_paths, structures


def build_candidate_truth_for_query(
    query_path: Path,
    candidate_paths: list[Path],
) -> list[dict]:
    try:
        query_structure = ferritin.load(query_path)
    except Exception:
        return []
    loaded_paths, structures = load_structures_for_paths(candidate_paths)
    if not structures:
        return []
    return build_truth_for_query(query_structure, loaded_paths, structures)


def truth_cache_key(
    *,
    pdb_dir: Path,
    target_paths: list[Path],
    query_paths: list[Path],
    seed: int,
    truth_mode: str,
    truth_candidate_top_k: int,
    foldseek_args: list[str],
    sensitivity: float,
    max_seqs: int,
) -> str:
    payload = {
        "pdb_dir": str(pdb_dir),
        "targets": [str(path) for path in target_paths],
        "queries": [str(path) for path in query_paths],
        "seed": seed,
        "aligner": "ferritin.tm_align_one_to_many.fast",
        "truth_mode": truth_mode,
        "truth_candidate_top_k": truth_candidate_top_k,
        "foldseek_args": foldseek_args,
        "foldseek_sensitivity": sensitivity,
        "foldseek_max_seqs": max_seqs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_truth_cache(path: Path, expected_key: str) -> dict[str, list[dict]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_key") != expected_key:
        return None
    return payload.get("truth", {})


def save_truth_cache(path: Path, *, cache_key: str, truth: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cache_key": cache_key, "truth": truth}, indent=2), encoding="utf-8")


def run_foldseek(
    *,
    foldseek: Path,
    query_dir: Path,
    target_dir: Path,
    output_path: Path,
    tmp_dir: Path,
    threads: int,
    sensitivity: float,
    max_seqs: int,
    extra_args: list[str],
) -> float:
    cmd = [
        str(foldseek),
        "easy-search",
        str(query_dir),
        str(target_dir),
        str(output_path),
        str(tmp_dir),
        "--format-output",
        "query,target,alntmscore,qtmscore,ttmscore,rmsd,alnlen,bits,evalue",
        "--threads",
        str(threads),
        "-s",
        str(sensitivity),
        "--max-seqs",
        str(max_seqs),
        "-v",
        "1",
        *extra_args,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def resolve_foldseek_id(identifier: str, stem_to_path: dict[str, Path]) -> Path | None:
    direct = stem_to_path.get(identifier)
    if direct is not None:
        return direct
    if "_" in identifier:
        return stem_to_path.get(identifier.split("_", 1)[0])
    return None


def parse_foldseek_hits(output_path: Path, stem_to_path: dict[str, Path]) -> dict[str, list[dict]]:
    hits_by_query: dict[str, list[dict]] = {}
    if not output_path.exists():
        return hits_by_query
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            raise SystemExit(f"Unexpected Foldseek output row with {len(parts)} columns: {line}")
        query, target, alntmscore, qtmscore, ttmscore, rmsd, alnlen, bits, evalue = parts
        query_path = resolve_foldseek_id(query, stem_to_path)
        target_path = resolve_foldseek_id(target, stem_to_path)
        if query_path is None or target_path is None:
            continue
        hits_by_query.setdefault(str(query_path), []).append(
            {
                "source_path": str(target_path),
                "alntmscore": float(alntmscore),
                "qtmscore": float(qtmscore),
                "ttmscore": float(ttmscore),
                "tm_score": max(float(qtmscore), float(ttmscore)),
                "rmsd": float(rmsd),
                "n_aligned": int(alnlen),
                "bits": float(bits),
                "evalue": float(evalue),
            }
        )
    for hits in hits_by_query.values():
        hits.sort(key=lambda hit: (-hit["bits"], hit["evalue"], Path(hit["source_path"]).name))
        deduped: list[dict] = []
        seen: set[str] = set()
        for hit in hits:
            if hit["source_path"] in seen:
                continue
            seen.add(hit["source_path"])
            deduped.append(hit)
        hits[:] = deduped
    return hits_by_query


def recall_at_k(
    hits: list[dict],
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
    retrieved = {hit["source_path"] for hit in hits[:k]}
    return len(relevant & retrieved) / len(relevant)


def best_nonself_row(rows: list[dict], *, query_path: Path) -> dict | None:
    for row in rows:
        if Path(row["source_path"]) != query_path:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Foldseek retrieval quality against ferritin TM-align truth")
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/ferritin/validation/pdbs_10k")
    parser.add_argument("--foldseek", default=str(FOLDSEEK))
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sensitivity", type=float, default=9.5)
    parser.add_argument("--max-seqs", type=int, default=1000)
    parser.add_argument("--truth-mode", choices=["candidates", "exhaustive"], default="candidates")
    parser.add_argument("--truth-candidate-top-k", type=int, default=500)
    parser.add_argument("--truth-cache", default=None)
    parser.add_argument("--reuse-truth", action="store_true")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--output", default="validation/foldseek_retrieval_bench.json")
    parser.add_argument("foldseek_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.foldseek_args and args.foldseek_args[0] == "--":
        args.foldseek_args = args.foldseek_args[1:]

    foldseek = Path(args.foldseek)
    if not foldseek.exists():
        raise SystemExit(f"Foldseek executable not found: {foldseek}")

    pdb_dir = Path(args.pdb_dir)
    all_paths = collect_structure_paths(pdb_dir)
    if not all_paths:
        raise SystemExit(f"No structures found in {pdb_dir}")

    target_paths = sample_targets(all_paths, args.n_targets, args.seed)
    query_paths = sample_queries(target_paths, args.n_queries, args.seed)
    assert_unique_stems(target_paths)
    if args.truth_candidate_top_k < args.top_k:
        raise SystemExit("--truth-candidate-top-k must be >= --top-k")
    print(
        f"Sampled {len(target_paths)} targets and {len(query_paths)} queries from {pdb_dir}",
        flush=True,
    )

    work_context = None if args.keep_work_dir else (
        tempfile.TemporaryDirectory(prefix="ferritin_foldseek_")
        if args.work_dir is None
        else None
    )
    if args.work_dir is not None:
        work_root = Path(args.work_dir)
    elif work_context is not None:
        work_root = Path(work_context.name)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="ferritin_foldseek_"))
    query_dir = work_root / "queries"
    target_dir = work_root / "targets"
    tmp_dir = work_root / "tmp"
    foldseek_out = work_root / "foldseek.m8"
    if args.work_dir is not None:
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    stage_paths(query_paths, query_dir)
    stage_paths(target_paths, target_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Foldseek: {foldseek}")
    print(f"Corpus sample: {len(target_paths)} targets from {pdb_dir}")
    print(f"Queries: {len(query_paths)}")
    print(f"Work dir: {work_root}")
    print("Running Foldseek easy-search...", flush=True)

    foldseek_s = run_foldseek(
        foldseek=foldseek,
        query_dir=query_dir,
        target_dir=target_dir,
        output_path=foldseek_out,
        tmp_dir=tmp_dir,
        threads=args.threads,
        sensitivity=args.sensitivity,
        max_seqs=args.max_seqs,
        extra_args=args.foldseek_args,
    )

    stem_to_path = {path.stem: path for path in target_paths}
    hits_by_query = parse_foldseek_hits(foldseek_out, stem_to_path)

    if args.truth_mode == "exhaustive":
        t0 = time.time()
        print("Loading sampled target structures for exhaustive truth...", flush=True)
        target_paths, target_structures = load_structures_for_paths(target_paths)
        load_s = time.time() - t0
        loadable_target_set = {str(path) for path in target_paths}
        query_paths = [path for path in query_paths if str(path) in loadable_target_set]
        if not query_paths:
            raise SystemExit("No query structures remain after filtering to loadable targets")
        print(
            f"Loaded {len(target_paths)} targets and retained {len(query_paths)} queries in {load_s:.3f}s",
            flush=True,
        )
    else:
        target_structures = []
        load_s = 0.0

    cache_key = truth_cache_key(
        pdb_dir=pdb_dir,
        target_paths=target_paths,
        query_paths=query_paths,
        seed=args.seed,
        truth_mode=args.truth_mode,
        truth_candidate_top_k=args.truth_candidate_top_k,
        foldseek_args=args.foldseek_args,
        sensitivity=args.sensitivity,
        max_seqs=args.max_seqs,
    )
    truth_cache_path = Path(args.truth_cache) if args.truth_cache else Path(str(args.output) + ".truth.json")
    truth_cache = load_truth_cache(truth_cache_path, cache_key) if args.reuse_truth else None
    truth_cache_hit = truth_cache is not None
    if truth_cache is None:
        truth_cache = {}

    brute_force_s = 0.0
    for query_idx, query_path in enumerate(query_paths, start=1):
        query_key = str(query_path)
        if query_key in truth_cache:
            print(f"Truth {query_idx}/{len(query_paths)} cache hit: {query_path.name}", flush=True)
            continue
        if args.truth_mode == "exhaustive":
            print(
                f"Truth {query_idx}/{len(query_paths)} exhaustive {query_path.name} vs {len(target_paths)} targets...",
                flush=True,
            )
            t0 = time.time()
            truth_cache[query_key] = build_truth_for_query(
                ferritin.load(query_path),
                target_paths,
                target_structures,
            )
        else:
            candidate_paths = []
            seen_candidates = set()
            for path in [query_path] + [
                Path(hit["source_path"])
                for hit in hits_by_query.get(str(query_path), [])[:args.truth_candidate_top_k]
            ]:
                key = str(path)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                candidate_paths.append(path)
            print(
                f"Truth {query_idx}/{len(query_paths)} candidates {query_path.name} vs {len(candidate_paths)} structures...",
                flush=True,
            )
            t0 = time.time()
            truth_cache[query_key] = build_candidate_truth_for_query(query_path, candidate_paths)
        query_truth_s = time.time() - t0
        brute_force_s += query_truth_s
        print(
            f"Truth {query_idx}/{len(query_paths)} done in {query_truth_s:.3f}s",
            flush=True,
        )
    if not truth_cache_hit:
        save_truth_cache(truth_cache_path, cache_key=cache_key, truth=truth_cache)

    threshold_metrics = {
        str(threshold): []
        for threshold in args.thresholds
    }
    top1_exact = []
    per_query = []
    for query_path in query_paths:
        truth_rows = truth_cache[str(query_path)]
        hits = hits_by_query.get(str(query_path), [])
        best_truth = best_nonself_row(truth_rows, query_path=query_path)
        top_nonself = best_nonself_row(hits, query_path=query_path)
        top1_exact.append(
            int(best_truth is not None and top_nonself is not None and top_nonself["source_path"] == best_truth["source_path"])
        )
        row = {
            "query": query_path.name,
            "n_hits": len(hits),
            "best_truth_nonself": None if best_truth is None else {
                "source_path": Path(best_truth["source_path"]).name,
                "tm_score": round(best_truth["tm_score"], 4),
            },
            "foldseek_top_nonself": None if top_nonself is None else {
                "source_path": Path(top_nonself["source_path"]).name,
                "tm_score": round(top_nonself["tm_score"], 4),
                "bits": round(top_nonself["bits"], 4),
                "evalue": top_nonself["evalue"],
            },
            "thresholds": {},
        }
        for threshold in args.thresholds:
            recall = recall_at_k(
                hits,
                truth_rows,
                k=args.top_k,
                tm_threshold=threshold,
                exclude_self_path=query_path,
            )
            threshold_metrics[str(threshold)].append(recall)
            row["thresholds"][str(threshold)] = {
                "recall_at_k": round(recall, 4),
            }
        per_query.append(row)
        print(
            f"{query_path.name}: truth {row['best_truth_nonself']} "
            f"foldseek {row['foldseek_top_nonself']} hits {len(hits)}",
            flush=True,
        )

    summary = {
        "corpus": {
            "pdb_dir": str(pdb_dir),
            "n_targets": len(target_paths),
            "n_queries": len(query_paths),
            "top_k": args.top_k,
            "thresholds": args.thresholds,
            "truth_cache": str(truth_cache_path),
            "truth_cache_hit": truth_cache_hit,
            "truth_mode": args.truth_mode,
            "truth_candidate_top_k": args.truth_candidate_top_k,
        },
        "foldseek": {
            "executable": str(foldseek),
            "threads": args.threads,
            "sensitivity": args.sensitivity,
            "max_seqs": args.max_seqs,
            "extra_args": args.foldseek_args,
        },
        "timing": {
            "load_s": round(load_s, 3),
            "brute_force_s": round(brute_force_s, 3),
            "foldseek_s": round(foldseek_s, 3),
        },
        "metrics": {
            "top1_exact_nonself": round(mean(top1_exact), 4) if top1_exact else 0.0,
            "recall_at_k": {
                threshold: round(mean(values), 4) if values else 0.0
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

    if args.keep_work_dir:
        print(f"Kept work dir: {work_root}")


if __name__ == "__main__":
    main()
