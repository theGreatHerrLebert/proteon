#!/usr/bin/env python3
"""Benchmark proteon structural search on a local corpus.

Measures:
- DB build time
- skip rate during tolerant loading
- on-disk DB size
- prefilter latency
- rerank latency
- self-hit recall@1 and recall@k

Usage:
    python validation/bench_search.py
    python validation/bench_search.py --pdb-dir validation/pdbs_10k --n-queries 100
    python validation/bench_search.py --query-paths validation/pdb_ids_1k.txt --rerank-top-k 20
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import time
from pathlib import Path
from statistics import mean, median

import proteon
from proteon import search as proteon_search_fn
proteon_search_mod = importlib.import_module("proteon.search")
import pyarrow as pa
import pyarrow.parquet as pq


def collect_structure_paths(pdb_dir: Path) -> list[Path]:
    exts = {".pdb", ".cif", ".mmcif"}
    return sorted(p for p in pdb_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_query_panel(path_file: Path | None, corpus_paths: list[Path], n_queries: int, seed: int) -> list[Path]:
    if path_file is not None:
        requested = [line.strip() for line in path_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        corpus_map = {p.stem.upper(): p for p in corpus_paths}
        queries = [corpus_map[pdb_id.upper()] for pdb_id in requested if pdb_id.upper() in corpus_map]
        return queries[:n_queries]

    if n_queries >= len(corpus_paths):
        return corpus_paths

    rng = random.Random(seed)
    return sorted(rng.sample(corpus_paths, n_queries))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "median_ms": round(median(values), 3),
        "mean_ms": round(mean(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def hit_metrics(query_path: Path, hits, k: int) -> tuple[bool, bool]:
    if not hits:
        return False, False
    self_top1 = Path(hits[0].source_path) == query_path
    self_topk = any(Path(hit.source_path) == query_path for hit in hits[:k])
    return self_top1, self_topk


def filtered_query_panel(query_panel: list[Path], db: proteon.SearchDB) -> list[Path]:
    if db.entries is not None:
        indexed_paths = {Path(entry.source_path) for entry in db.entries}
    else:
        entries_path = Path(db.root_path) / "entries.parquet"
        indexed_paths = {Path(path) for path in pq.read_table(entries_path, columns=["source_path"]).column("source_path").to_pylist()}
    return [path for path in query_panel if path in indexed_paths]


def write_manifest(root: Path, *, k: int, n_entries: int) -> None:
    payload = {
        "version": proteon_search_mod.SEARCH_DB_VERSION,
        "k": k,
        "n_entries": n_entries,
        "entries_file": "entries.parquet",
        "postings_dir": "postings",
        "postings_bucket_count": proteon_search_mod.POSTINGS_BUCKET_COUNT,
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark proteon search on a local structure corpus")
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/proteon/validation/pdbs_10k")
    parser.add_argument("--db-path", default="/tmp/proteon_search_10k")
    parser.add_argument("--k", type=int, default=6, help="k-mer length")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--query-paths", type=str, default=None, help="Optional file of PDB IDs to use as queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="validation/search_bench.json")
    parser.add_argument("--build-chunk-size", type=int, default=100)
    parser.add_argument("--stop-after-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.exists():
        raise SystemExit(f"PDB directory not found: {pdb_dir}")

    corpus_paths = collect_structure_paths(pdb_dir)
    if not corpus_paths:
        raise SystemExit(f"No structures found in {pdb_dir}")

    query_panel = load_query_panel(
        Path(args.query_paths) if args.query_paths else None,
        corpus_paths,
        args.n_queries,
        args.seed,
    )
    if not query_panel:
        raise SystemExit("No query structures selected")

    print(f"Corpus: {len(corpus_paths)} files from {pdb_dir}")
    print(f"Query panel: {len(query_panel)} structures")
    print(f"DB path: {args.db_path}")
    print()

    db_root = Path(args.db_path)
    load_elapsed = 0.0
    encode_elapsed = 0.0
    index_elapsed = 0.0
    save_elapsed = 0.0
    build_elapsed = 0.0

    if args.skip_build:
        print("Build stages")
        print("  Skipping build; loading existing DB...", flush=True)
        t0 = time.time()
        db = proteon.load_search_db(db_root)
        build_elapsed = time.time() - t0
        skipped = len(corpus_paths) - len(db)
        db_size_mb = sum(p.stat().st_size for p in db_root.rglob("*") if p.is_file()) / (1024 * 1024)
        print(f"Loaded DB: {len(db)} indexed, {skipped} skipped, {db_size_mb:.1f} MB", flush=True)
    else:
        print("Build stages")
        build_start = time.time()
        if db_root.exists():
            if db_root.is_dir():
                for child in db_root.iterdir():
                    if child.is_file():
                        child.unlink()
            else:
                db_root.unlink()
        db_root.mkdir(parents=True, exist_ok=True)
        entries_path = db_root / "entries.parquet"
        entry_writer = None
        posting_writers = {}
        posting_schema = proteon_search_mod._empty_postings_table().schema

        print(f"  1. Loading structures in chunks of {args.build_chunk_size}...", flush=True)
        total_loaded = 0
        total_indexed = 0
        for start, chunk in chunked(corpus_paths, args.build_chunk_size):
            t0 = time.time()
            chunk_loaded = proteon.batch_load_tolerant(chunk, n_threads=-1)
            load_elapsed += time.time() - t0
            total_loaded += len(chunk_loaded)
            print(
                f"     load {min(start + len(chunk), len(corpus_paths))}/{len(corpus_paths)} "
                f"-> loaded {total_loaded}",
                flush=True,
            )

            if chunk_loaded:
                t0 = time.time()
                chunk_indices, chunk_structures = zip(*chunk_loaded)
                chunk_encoded = proteon.batch_encode_alphabet(chunk_structures, n_threads=-1)
                encode_elapsed += time.time() - t0

                chunk_entries = [
                    proteon_search_mod._build_entry(
                        corpus_paths[start + source_index],
                        start + source_index,
                        structure,
                        result,
                        args.k,
                        entry_index=total_indexed + local_index,
                    )
                    for local_index, (source_index, structure, result) in enumerate(
                        zip(chunk_indices, chunk_structures, chunk_encoded)
                    )
                ]
                entry_rows = [proteon_search_mod.asdict(entry) for entry in chunk_entries]
                entry_table = pa.Table.from_pylist(entry_rows)
                if entry_writer is None:
                    entry_writer = pq.ParquetWriter(entries_path, entry_table.schema, compression="zstd")
                entry_writer.write_table(entry_table)

                posting_rows = []
                sa_postings = proteon_search_mod._build_postings(chunk_entries, args.k, attr="valid_alphabet")
                aa_postings = proteon_search_mod._build_postings(chunk_entries, args.k, attr="valid_aa_sequence")
                for kind, postings in (("sa", sa_postings), ("aa", aa_postings)):
                    for kmer, local_indices in postings.items():
                        for local_index in local_indices:
                            posting_rows.append(
                                {
                                    "kind": kind,
                                    "kmer": kmer,
                                    "entry_index": int(local_index),
                                }
                            )
                rows_by_partition = {}
                for row in posting_rows:
                    bucket = proteon_search_mod._posting_bucket(row["kmer"])
                    rows_by_partition.setdefault((row["kind"], bucket), []).append(
                        {"kmer": row["kmer"], "entry_index": row["entry_index"]}
                    )
                for (kind, bucket), bucket_rows in rows_by_partition.items():
                    bucket_path = proteon_search_mod._bucket_file(db_root, kind=kind, bucket=bucket)
                    bucket_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = posting_writers.get((kind, bucket))
                    if writer is None:
                        writer = pq.ParquetWriter(bucket_path, posting_schema, compression="zstd")
                        posting_writers[(kind, bucket)] = writer
                    writer.write_table(pa.Table.from_pylist(bucket_rows, schema=posting_schema))

                total_indexed += len(chunk_entries)
                print(
                    f"     encode {min(start + len(chunk), len(corpus_paths))}/{len(corpus_paths)} "
                    f"-> indexed {total_indexed}",
                    flush=True,
                )

                del chunk_indices
                del chunk_structures
                del chunk_encoded
                del chunk_entries
                del entry_rows
                del entry_table
                del posting_rows
                del rows_by_partition
                del sa_postings
                del aa_postings
                gc.collect()
        print(
            f"     loaded {total_loaded}/{len(corpus_paths)} in {load_elapsed:.2f}s "
            f"(skipped {len(corpus_paths) - total_loaded})",
            flush=True,
        )
        print(f"     encoded {total_indexed} structures in {encode_elapsed:.2f}s", flush=True)

        print("  3. Finalizing Parquet search DB...", flush=True)
        t0 = time.time()
        if entry_writer is not None:
            entry_writer.close()
        else:
            empty_entries = pa.table(
                {
                    "id": pa.array([], type=pa.string()),
                    "source_path": pa.array([], type=pa.string()),
                    "source_index": pa.array([], type=pa.int64()),
                    "residue_count": pa.array([], type=pa.int64()),
                    "valid_residue_count": pa.array([], type=pa.int64()),
                    "aa_sequence": pa.array([], type=pa.string()),
                    "valid_aa_sequence": pa.array([], type=pa.string()),
                    "alphabet": pa.array([], type=pa.string()),
                    "valid_alphabet": pa.array([], type=pa.string()),
                    "aa_kmer_count": pa.array([], type=pa.int64()),
                    "kmer_count": pa.array([], type=pa.int64()),
                }
            )
            pq.write_table(empty_entries, entries_path, compression="zstd")
        for writer in posting_writers.values():
            writer.close()
        write_manifest(db_root, k=args.k, n_entries=total_indexed)
        save_elapsed = time.time() - t0
        print(f"     finalized DB in {save_elapsed:.2f}s", flush=True)

        build_elapsed = time.time() - build_start
        db = proteon.load_search_db(db_root)
        skipped = len(corpus_paths) - len(db)
        db_size_mb = sum(p.stat().st_size for p in db_root.rglob("*") if p.is_file()) / (1024 * 1024)

        print(f"Build: {len(db)} indexed, {skipped} skipped, {build_elapsed:.2f}s, {db_size_mb:.1f} MB")

    query_panel = filtered_query_panel(query_panel, db)
    if not query_panel:
        raise SystemExit("No query structures remain after filtering to indexed paths")
    print(f"Usable query panel after index filtering: {len(query_panel)}", flush=True)

    if args.stop_after_build:
        summary = {
            "corpus": {
                "pdb_dir": str(pdb_dir),
                "n_input_files": len(corpus_paths),
                "n_indexed": len(db),
                "n_skipped": skipped,
                "db_path": str(args.db_path),
                "db_size_mb": round(db_size_mb, 3),
                "k": args.k,
                "version": db.version,
            },
            "build": {
                "elapsed_s": round(build_elapsed, 3),
                "load_s": round(load_elapsed, 3),
                "encode_s": round(encode_elapsed, 3),
                "index_s": round(index_elapsed, 3),
                "save_s": round(save_elapsed, 3),
                "structures_per_s": round(len(db) / build_elapsed, 3) if build_elapsed > 0 else None,
            },
            "queries": None,
            "prefilter": None,
            "rerank": None,
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Stopped after build stage by request.")
        print(f"Output: {out_path}")
        return

    prefilter_times = []
    rerank_times = []
    prefilter_top1 = 0
    prefilter_topk = 0
    rerank_top1 = 0
    rerank_topk = 0
    query_load_failures = 0
    prefilter_examples = []
    rerank_examples = []

    for i, query_path in enumerate(query_panel, start=1):
        try:
            query = proteon.load(query_path)
        except Exception:
            query_load_failures += 1
            continue

        t0 = time.time()
        pre_hits = proteon_search_fn(
            query,
            db,
            top_k=args.top_k,
            rerank=False,
        )
        pre_ms = (time.time() - t0) * 1000
        prefilter_times.append(pre_ms)
        top1, topk = hit_metrics(query_path, pre_hits, args.top_k)
        prefilter_top1 += int(top1)
        prefilter_topk += int(topk)

        t0 = time.time()
        rr_hits = proteon_search_fn(
            query,
            db,
            top_k=args.top_k,
            rerank=True,
            rerank_top_k=args.rerank_top_k,
            rerank_fast=True,
        )
        rr_ms = (time.time() - t0) * 1000
        rerank_times.append(rr_ms)
        top1, topk = hit_metrics(query_path, rr_hits, args.top_k)
        rerank_top1 += int(top1)
        rerank_topk += int(topk)

        if len(prefilter_examples) < 5:
            prefilter_examples.append(
                {
                    "query": str(query_path),
                    "top_hit": pre_hits[0].source_path if pre_hits else None,
                    "top_score": round(pre_hits[0].score, 4) if pre_hits else None,
                    "self_in_top1": bool(pre_hits and Path(pre_hits[0].source_path) == query_path),
                }
            )
        if len(rerank_examples) < 5:
            rerank_examples.append(
                {
                    "query": str(query_path),
                    "top_hit": rr_hits[0].source_path if rr_hits else None,
                    "top_tm": round(rr_hits[0].tm_score, 4) if rr_hits and rr_hits[0].tm_score is not None else None,
                    "self_in_top1": bool(rr_hits and Path(rr_hits[0].source_path) == query_path),
                }
            )

        if i % 10 == 0 or i == len(query_panel):
            print(
                f"[{i}/{len(query_panel)}] "
                f"prefilter median {median(prefilter_times):.1f} ms, "
                f"rerank median {median(rerank_times):.1f} ms"
            )

    n_effective_queries = len(prefilter_times)
    if n_effective_queries == 0:
        raise SystemExit("No queries completed successfully")

    summary = {
        "corpus": {
            "pdb_dir": str(pdb_dir),
            "n_input_files": len(corpus_paths),
            "n_indexed": len(db),
            "n_skipped": skipped,
            "db_path": str(args.db_path),
            "db_size_mb": round(db_size_mb, 3),
            "k": args.k,
            "version": db.version,
        },
        "build": {
            "elapsed_s": round(build_elapsed, 3),
            "load_s": round(load_elapsed, 3),
            "encode_s": round(encode_elapsed, 3),
            "index_s": round(index_elapsed, 3),
            "save_s": round(save_elapsed, 3),
            "structures_per_s": round(len(db) / build_elapsed, 3) if build_elapsed > 0 else None,
        },
        "queries": {
            "n_queries": len(query_panel),
            "n_completed": n_effective_queries,
            "n_query_load_failures": query_load_failures,
            "top_k": args.top_k,
            "rerank_top_k": args.rerank_top_k,
        },
        "prefilter": {
            **summarize(prefilter_times),
            "self_hit_top1": prefilter_top1,
            "self_hit_top1_rate": round(prefilter_top1 / n_effective_queries, 4),
            "self_hit_topk": prefilter_topk,
            "self_hit_topk_rate": round(prefilter_topk / n_effective_queries, 4),
            "examples": prefilter_examples,
        },
        "rerank": {
            **summarize(rerank_times),
            "self_hit_top1": rerank_top1,
            "self_hit_top1_rate": round(rerank_top1 / n_effective_queries, 4),
            "self_hit_topk": rerank_topk,
            "self_hit_topk_rate": round(rerank_topk / n_effective_queries, 4),
            "examples": rerank_examples,
        },
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Summary")
    print(f"  Build: {summary['build']['elapsed_s']:.2f}s")
    print(
        f"  Prefilter: median {summary['prefilter'].get('median_ms', 0):.1f} ms, "
        f"self-hit@1 {summary['prefilter']['self_hit_top1_rate']:.3f}"
    )
    print(
        f"  Rerank:    median {summary['rerank'].get('median_ms', 0):.1f} ms, "
        f"self-hit@1 {summary['rerank']['self_hit_top1_rate']:.3f}"
    )
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
