#!/usr/bin/env python3
"""Retrieval-quality benchmark for Foldseek on proteon's sampled corpus.

This mirrors validation/bench_retrieval.py so Foldseek and proteon can be
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

import proteon


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
    results = proteon.tm_align_one_to_many(
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
    loaded = proteon.batch_load_tolerant(paths, n_threads=-1)
    loaded_paths = [paths[idx] for idx, _structure in loaded]
    structures = [structure for _idx, structure in loaded]
    return loaded_paths, structures


def build_candidate_truth_for_query(
    query_path: Path,
    candidate_paths: list[Path],
) -> list[dict]:
    try:
        query_structure = proteon.load(query_path)
    except Exception:
        return []
    loaded_paths, structures = load_structures_for_paths(candidate_paths)
    if not structures:
        return []
    return build_truth_for_query(query_structure, loaded_paths, structures)


def filter_truth_candidate_paths(
    paths: list[Path],
    *,
    max_file_size_mb: float | None,
) -> tuple[list[Path], list[dict]]:
    kept = []
    skipped = []
    max_bytes = None if max_file_size_mb is None else int(max_file_size_mb * 1024 * 1024)
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append({"source_path": str(path), "reason": "stat_failed"})
            continue
        if max_bytes is not None and size > max_bytes:
            skipped.append({
                "source_path": str(path),
                "reason": "file_too_large",
                "file_size_mb": round(size / (1024 * 1024), 3),
            })
            continue
        kept.append(path)
    return kept, skipped


def truth_cache_key(
    *,
    pdb_dir: Path,
    target_paths: list[Path],
    query_paths: list[Path],
    seed: int,
    truth_mode: str,
    truth_candidate_top_k: int,
    truth_max_file_size_mb: float | None,
    foldseek_args: list[str],
    sensitivity: float,
    max_seqs: int,
) -> str:
    payload = {
        "pdb_dir": str(pdb_dir),
        "targets": [str(path) for path in target_paths],
        "queries": [str(path) for path in query_paths],
        "seed": seed,
        "aligner": "proteon.tm_align_one_to_many.fast",
        "truth_mode": truth_mode,
        "truth_candidate_top_k": truth_candidate_top_k,
        "truth_max_file_size_mb": truth_max_file_size_mb,
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
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps({"cache_key": cache_key, "truth": truth}, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def retrieval_cache_key(
    *,
    pdb_dir: Path,
    target_paths: list[Path],
    query_paths: list[Path],
    foldseek: Path,
    foldseek_args: list[str],
    sensitivity: float,
    max_seqs: int,
    compare_proteon: bool,
    proteon_db_path: str | None,
    proteon_k: int,
    proteon_candidate_top_k: int,
    proteon_diagonal_top_k: int,
) -> str:
    payload = {
        "pdb_dir": str(pdb_dir),
        "targets": [str(path) for path in target_paths],
        "queries": [str(path) for path in query_paths],
        "foldseek": str(foldseek),
        "foldseek_args": foldseek_args,
        "foldseek_sensitivity": sensitivity,
        "foldseek_max_seqs": max_seqs,
        "compare_proteon": compare_proteon,
        "proteon_db_path": proteon_db_path,
        "proteon_k": proteon_k,
        "proteon_candidate_top_k": proteon_candidate_top_k,
        "proteon_diagonal_top_k": proteon_diagonal_top_k,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_retrieval_cache(path: Path, expected_key: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_key") != expected_key:
        return None
    return payload.get("state")


def save_retrieval_cache(path: Path, *, cache_key: str, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps({"cache_key": cache_key, "state": state}, indent=2), encoding="utf-8")
    tmp_path.replace(path)


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


def proteon_hits_to_rows(hits) -> list[dict]:
    rows = []
    for hit in hits:
        rows.append({
            "source_path": hit.source_path,
            "score": float(hit.score),
            "prefilter_score": float(hit.prefilter_score),
            "diagonal_score": None if hit.diagonal_score is None else float(hit.diagonal_score),
        })
    return rows


def load_query_or_none(query_path: Path):
    try:
        return proteon.load(query_path)
    except Exception as exc:
        print(f"Skipping proteon query {query_path.name}: {exc}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Foldseek retrieval quality against proteon TM-align truth")
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/proteon/validation/pdbs_10k")
    parser.add_argument("--foldseek", default=str(FOLDSEEK))
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sensitivity", type=float, default=9.5)
    parser.add_argument("--max-seqs", type=int, default=1000)
    parser.add_argument("--truth-mode", choices=["candidates", "exhaustive"], default="candidates")
    parser.add_argument("--truth-candidate-top-k", type=int, default=100)
    parser.add_argument("--truth-max-file-size-mb", type=float, default=20.0)
    parser.add_argument("--compare-proteon", action="store_true")
    parser.add_argument("--proteon-db-path", default=None)
    parser.add_argument("--reuse-proteon-db", action="store_true")
    parser.add_argument("--compile-proteon-db", action="store_true")
    parser.add_argument("--warm-proteon-db", action="store_true")
    parser.add_argument("--proteon-k", type=int, default=6)
    parser.add_argument("--proteon-candidate-top-k", type=int, default=100)
    parser.add_argument("--proteon-diagonal-top-k", type=int, default=200)
    parser.add_argument("--diagnostic-top-k", type=int, default=50)
    parser.add_argument("--retrieval-cache", default=None)
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
    if args.compare_proteon and args.proteon_candidate_top_k < args.top_k:
        raise SystemExit("--proteon-candidate-top-k must be >= --top-k")
    print(
        f"Sampled {len(target_paths)} targets and {len(query_paths)} queries from {pdb_dir}",
        flush=True,
    )
    proteon_db_path = (
        str(Path(args.proteon_db_path))
        if args.proteon_db_path
        else str(Path(str(Path(args.output).with_suffix("")) + ".proteon_db"))
    )
    retrieval_cache_path = (
        Path(args.retrieval_cache)
        if args.retrieval_cache
        else Path(str(args.output) + ".retrieval.json")
    )
    retrieval_key = retrieval_cache_key(
        pdb_dir=pdb_dir,
        target_paths=target_paths,
        query_paths=query_paths,
        foldseek=foldseek,
        foldseek_args=args.foldseek_args,
        sensitivity=args.sensitivity,
        max_seqs=args.max_seqs,
        compare_proteon=args.compare_proteon,
        proteon_db_path=proteon_db_path if args.compare_proteon else None,
        proteon_k=args.proteon_k,
        proteon_candidate_top_k=args.proteon_candidate_top_k,
        proteon_diagonal_top_k=args.proteon_diagonal_top_k,
    )
    retrieval_state = load_retrieval_cache(retrieval_cache_path, retrieval_key)

    work_context = None if args.keep_work_dir else (
        tempfile.TemporaryDirectory(prefix="proteon_foldseek_")
        if args.work_dir is None
        else None
    )
    if args.work_dir is not None:
        work_root = Path(args.work_dir)
    elif work_context is not None:
        work_root = Path(work_context.name)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="proteon_foldseek_"))
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

    proteon_db = None
    proteon_build_s = 0.0
    proteon_compile_s = 0.0
    proteon_warm_s = 0.0
    proteon_search_s = 0.0
    hits_by_query: dict[str, list[dict]]
    proteon_hits_by_query: dict[str, list[dict]] = {}
    skipped_proteon_queries: dict[str, str] = {}
    if retrieval_state is not None:
        print(f"Retrieval cache hit: {retrieval_cache_path}", flush=True)
        foldseek_s = float(retrieval_state.get("foldseek_s", 0.0))
        proteon_build_s = float(retrieval_state.get("proteon_build_s", 0.0))
        proteon_compile_s = float(retrieval_state.get("proteon_compile_s", 0.0))
        proteon_warm_s = float(retrieval_state.get("proteon_warm_s", 0.0))
        proteon_search_s = float(retrieval_state.get("proteon_search_s", 0.0))
        hits_by_query = retrieval_state.get("hits_by_query", {})
        proteon_hits_by_query = retrieval_state.get("proteon_hits_by_query", {})
        skipped_proteon_queries = retrieval_state.get("skipped_proteon_queries", {})
    else:
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

        if args.compare_proteon:
            db_path = Path(proteon_db_path)
            print(f"Proteon DB: {db_path}", flush=True)
            t0 = time.time()
            if args.reuse_proteon_db and db_path.exists():
                proteon_db = proteon.load_search_db(db_path)
            else:
                proteon_db = proteon.build_search_db(target_paths, out=db_path, k=args.proteon_k, n_threads=-1)
            proteon_build_s = time.time() - t0
            if args.warm_proteon_db:
                t0 = time.time()
                proteon.warm_search_db(proteon_db, posting_cache_max_size=128)
                proteon_warm_s = time.time() - t0
            if args.compile_proteon_db:
                t0 = time.time()
                proteon_db = proteon.compile_search_db(proteon_db)
                proteon_compile_s = time.time() - t0
            for query_idx, query_path in enumerate(query_paths, start=1):
                print(f"Proteon search {query_idx}/{len(query_paths)} {query_path.name}...", flush=True)
                query_structure = load_query_or_none(query_path)
                if query_structure is None:
                    proteon_hits_by_query[str(query_path)] = []
                    skipped_proteon_queries[str(query_path)] = "load_failed"
                    continue
                t0 = time.time()
                hits = proteon.search(
                    query_structure,
                    proteon_db,
                    top_k=args.proteon_candidate_top_k,
                    rerank=False,
                    diagonal_rescore=True,
                    diagonal_top_k=args.proteon_diagonal_top_k,
                )
                proteon_search_s += time.time() - t0
                proteon_hits_by_query[str(query_path)] = proteon_hits_to_rows(hits)
        save_retrieval_cache(
            retrieval_cache_path,
            cache_key=retrieval_key,
            state={
                "foldseek_s": foldseek_s,
                "proteon_build_s": proteon_build_s,
                "proteon_compile_s": proteon_compile_s,
                "proteon_warm_s": proteon_warm_s,
                "proteon_search_s": proteon_search_s,
                "hits_by_query": hits_by_query,
                "proteon_hits_by_query": proteon_hits_by_query,
                "skipped_proteon_queries": skipped_proteon_queries,
            },
        )
        print(f"Retrieval checkpointed to {retrieval_cache_path}", flush=True)

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
        truth_max_file_size_mb=args.truth_max_file_size_mb,
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
    skipped_truth_candidates_by_query: dict[str, list[dict]] = {}
    for query_idx, query_path in enumerate(query_paths, start=1):
        query_key = str(query_path)
        if query_key in truth_cache:
            print(f"Truth {query_idx}/{len(query_paths)} cache hit: {query_path.name}", flush=True)
            continue
        if args.truth_mode == "exhaustive":
            query_structure = load_query_or_none(query_path)
            if query_structure is None:
                skipped_proteon_queries[str(query_path)] = "load_failed"
                truth_cache[query_key] = []
                continue
            print(
                f"Truth {query_idx}/{len(query_paths)} exhaustive {query_path.name} vs {len(target_paths)} targets...",
                flush=True,
            )
            t0 = time.time()
            truth_cache[query_key] = build_truth_for_query(
                query_structure,
                target_paths,
                target_structures,
            )
        else:
            candidate_paths = []
            seen_candidates = set()
            union_paths = [query_path] + [
                Path(hit["source_path"])
                for hit in hits_by_query.get(str(query_path), [])[:args.truth_candidate_top_k]
            ]
            if args.compare_proteon:
                union_paths.extend(
                    Path(hit["source_path"])
                    for hit in proteon_hits_by_query.get(str(query_path), [])[:args.proteon_candidate_top_k]
                )
            for path in union_paths:
                key = str(path)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                candidate_paths.append(path)
            candidate_paths, skipped_candidates = filter_truth_candidate_paths(
                candidate_paths,
                max_file_size_mb=args.truth_max_file_size_mb,
            )
            skipped_truth_candidates_by_query[query_key] = skipped_candidates
            print(
                f"Truth {query_idx}/{len(query_paths)} candidates {query_path.name} "
                f"vs {len(candidate_paths)} structures ({len(skipped_candidates)} skipped)...",
                flush=True,
            )
            t0 = time.time()
            truth_cache[query_key] = build_candidate_truth_for_query(query_path, candidate_paths)
            if not truth_cache[query_key]:
                skipped_proteon_queries[str(query_path)] = "truth_load_failed"
        query_truth_s = time.time() - t0
        brute_force_s += query_truth_s
        print(
            f"Truth {query_idx}/{len(query_paths)} done in {query_truth_s:.3f}s",
            flush=True,
        )
        save_truth_cache(truth_cache_path, cache_key=cache_key, truth=truth_cache)
        print(
            f"Truth {query_idx}/{len(query_paths)} checkpointed to {truth_cache_path}",
            flush=True,
        )
    save_truth_cache(truth_cache_path, cache_key=cache_key, truth=truth_cache)

    threshold_metrics = {
        str(threshold): []
        for threshold in args.thresholds
    }
    top1_exact = []
    proteon_top1_exact = []
    per_query = []
    for query_path in query_paths:
        truth_rows = truth_cache[str(query_path)]
        hits = hits_by_query.get(str(query_path), [])
        proteon_hits = proteon_hits_by_query.get(str(query_path), [])
        best_truth = best_nonself_row(truth_rows, query_path=query_path)
        top_nonself = best_nonself_row(hits, query_path=query_path)
        proteon_top_nonself = best_nonself_row(proteon_hits, query_path=query_path)
        top1_exact.append(
            int(best_truth is not None and top_nonself is not None and top_nonself["source_path"] == best_truth["source_path"])
        )
        if args.compare_proteon:
            proteon_top1_exact.append(
                int(best_truth is not None and proteon_top_nonself is not None and proteon_top_nonself["source_path"] == best_truth["source_path"])
            )
        row = {
            "query": query_path.name,
            "n_hits": len(hits),
            "n_proteon_hits": len(proteon_hits),
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
            "proteon_top_nonself": None if proteon_top_nonself is None else {
                "source_path": Path(proteon_top_nonself["source_path"]).name,
                "score": round(proteon_top_nonself["score"], 4),
                "prefilter_score": round(proteon_top_nonself["prefilter_score"], 4),
                "diagonal_score": proteon_top_nonself["diagonal_score"],
            },
            "thresholds": {},
        }
        if args.diagnostic_top_k > 0:
            row["truth_top_hits"] = [
                {
                    "source_path": Path(hit["source_path"]).name,
                    "tm_score": round(hit["tm_score"], 4),
                }
                for hit in truth_rows
                if Path(hit["source_path"]) != query_path
            ][:args.diagnostic_top_k]
            row["foldseek_top_hits"] = [
                {
                    "source_path": Path(hit["source_path"]).name,
                    "tm_score": round(hit["tm_score"], 4),
                    "bits": round(hit["bits"], 4),
                    "evalue": hit["evalue"],
                }
                for hit in hits
                if Path(hit["source_path"]) != query_path
            ][:args.diagnostic_top_k]
            row["proteon_top_hits"] = [
                {
                    "source_path": Path(hit["source_path"]).name,
                    "score": round(hit["score"], 4),
                    "prefilter_score": round(hit["prefilter_score"], 4),
                    "diagonal_score": hit["diagonal_score"],
                }
                for hit in proteon_hits
                if Path(hit["source_path"]) != query_path
            ][:args.diagnostic_top_k]
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
            if args.compare_proteon:
                proteon_recall = recall_at_k(
                    proteon_hits,
                    truth_rows,
                    k=args.top_k,
                    tm_threshold=threshold,
                    exclude_self_path=query_path,
                )
                row["thresholds"][str(threshold)]["proteon_recall_at_k"] = round(proteon_recall, 4)
        per_query.append(row)
        print(
            f"{query_path.name}: truth {row['best_truth_nonself']} "
            f"foldseek {row['foldseek_top_nonself']} hits {len(hits)} "
            f"proteon {row['proteon_top_nonself']}",
            flush=True,
        )

    foldseek_recall_at_k = {
        threshold: round(mean(values), 4) if values else 0.0
        for threshold, values in threshold_metrics.items()
    }
    proteon_recall_at_k = {}
    if args.compare_proteon:
        for threshold in args.thresholds:
            values = [
                row["thresholds"][str(threshold)]["proteon_recall_at_k"]
                for row in per_query
            ]
            proteon_recall_at_k[str(threshold)] = round(mean(values), 4) if values else 0.0

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
            "truth_max_file_size_mb": args.truth_max_file_size_mb,
            "compare_proteon": args.compare_proteon,
            "proteon_candidate_top_k": args.proteon_candidate_top_k,
            "diagnostic_top_k": args.diagnostic_top_k,
        },
        "foldseek": {
            "executable": str(foldseek),
            "threads": args.threads,
            "sensitivity": args.sensitivity,
            "max_seqs": args.max_seqs,
            "extra_args": args.foldseek_args,
        },
        "proteon": None if not args.compare_proteon else {
            "db_path": str(Path(args.proteon_db_path) if args.proteon_db_path else Path(str(Path(args.output).with_suffix("")) + ".proteon_db")),
            "k": args.proteon_k,
            "candidate_top_k": args.proteon_candidate_top_k,
            "diagonal_top_k": args.proteon_diagonal_top_k,
            "reuse_db": args.reuse_proteon_db,
            "compile_db": args.compile_proteon_db,
            "warm_db": args.warm_proteon_db,
        },
        "timing": {
            "load_s": round(load_s, 3),
            "brute_force_s": round(brute_force_s, 3),
            "foldseek_s": round(foldseek_s, 3),
            "proteon_build_s": round(proteon_build_s, 3),
            "proteon_compile_s": round(proteon_compile_s, 3),
            "proteon_warm_s": round(proteon_warm_s, 3),
            "proteon_search_s": round(proteon_search_s, 3),
        },
        "metrics": {
            "foldseek_top1_exact_nonself": round(mean(top1_exact), 4) if top1_exact else 0.0,
            "foldseek_recall_at_k": foldseek_recall_at_k,
            "proteon_top1_exact_nonself": round(mean(proteon_top1_exact), 4) if proteon_top1_exact else None,
            "proteon_recall_at_k": proteon_recall_at_k if args.compare_proteon else None,
        },
        "per_query": per_query,
        "skipped_truth_candidates": skipped_truth_candidates_by_query,
        "skipped_proteon_queries": skipped_proteon_queries,
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
