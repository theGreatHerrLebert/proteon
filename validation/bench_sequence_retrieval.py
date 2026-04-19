#!/usr/bin/env python3
"""Sequence-retrieval baseline for structural-search benchmarks.

This is intentionally dependency-light: it extracts amino-acid sequences from
the same structure corpus, ranks candidates by k-mer overlap, optionally
Smith-Waterman reranks the top prefilter hits, and evaluates against the same
TM-align truth cache used by bench_foldseek_retrieval.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # HMMER is strict enough that noncanonical residue symbols can abort some
    # searches, so ambiguous/rare residues are normalized to X.
    "SEC": "X",
    "PYL": "X",
    "ASX": "X",
    "GLX": "X",
}


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


def extract_sequence(structure) -> str:
    residues = []
    for residue in structure.residues:
        if not residue.is_amino_acid:
            continue
        name = (residue.name or "").upper()
        residues.append(AA3_TO_1.get(name, "X"))
    return "".join(residues)


def sanitize_sequence_for_hmmer(sequence: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWYX")
    return "".join(char if char in allowed else "X" for char in sequence.upper())


def load_sequences(
    paths: list[Path],
    *,
    min_length: int,
    max_length: int | None,
) -> tuple[dict[str, str], dict[str, str]]:
    import proteon

    sequences: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        try:
            sequence = sanitize_sequence_for_hmmer(extract_sequence(proteon.load(path)))
        except Exception as exc:
            skipped[str(path)] = f"load_failed: {exc}"
            continue
        if len(sequence) < min_length:
            skipped[str(path)] = f"sequence_too_short:{len(sequence)}"
            continue
        if max_length is not None and len(sequence) > max_length:
            skipped[str(path)] = f"sequence_too_long:{len(sequence)}"
            continue
        sequences[str(path)] = sequence
    return sequences, skipped


def load_sequence_cache(path: Path, expected_paths: list[Path]) -> tuple[dict[str, str], dict[str, str]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = [str(path) for path in expected_paths]
    if payload.get("paths") != expected:
        return None
    return payload.get("sequences", {}), payload.get("skipped_sequences", {})


def save_sequence_cache(
    path: Path,
    *,
    target_paths: list[Path],
    sequences: dict[str, str],
    skipped_sequences: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "paths": [str(path) for path in target_paths],
                "sequences": sequences,
                "skipped_sequences": skipped_sequences,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def kmer_counts(sequence: str, k: int) -> Counter[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(sequence) < k:
        return Counter()
    return Counter(sequence[idx : idx + k] for idx in range(len(sequence) - k + 1))


def weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    keys = left.keys() | right.keys()
    numerator = sum(min(left[key], right[key]) for key in keys)
    denominator = sum(max(left[key], right[key]) for key in keys)
    return 0.0 if denominator == 0 else numerator / denominator


def smith_waterman_score(
    query: str,
    target: str,
    *,
    match: int,
    mismatch: int,
    gap: int,
) -> float:
    if not query or not target:
        return 0.0
    prev = [0] * (len(target) + 1)
    best = 0
    for q_char in query:
        curr = [0]
        for col, t_char in enumerate(target, start=1):
            diag = prev[col - 1] + (match if q_char == t_char else mismatch)
            delete = prev[col] + gap
            insert = curr[col - 1] + gap
            score = max(0, diag, delete, insert)
            curr.append(score)
            if score > best:
                best = score
        prev = curr
    normalizer = match * math.sqrt(len(query) * len(target))
    return 0.0 if normalizer <= 0 else best / normalizer


def best_nonself_row(rows: list[dict[str, Any]], *, query_path: Path) -> dict[str, Any] | None:
    for row in rows:
        if Path(row["source_path"]) != query_path:
            return row
    return None


def recall_at_k(
    hits: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
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


def rank_sequence_hits(
    query_path: Path,
    target_paths: list[Path],
    sequences: dict[str, str],
    kmer_index: dict[str, Counter[str]],
    *,
    kmer_prefilter_top_k: int,
    sw_rerank_top_k: int,
    sw_match: int,
    sw_mismatch: int,
    sw_gap: int,
) -> list[dict[str, Any]]:
    query_key = str(query_path)
    query_sequence = sequences.get(query_key)
    if not query_sequence:
        return []
    query_kmers = kmer_index[query_key]
    prefilter_hits = []
    for target_path in target_paths:
        target_key = str(target_path)
        target_sequence = sequences.get(target_key)
        if not target_sequence:
            continue
        score = weighted_jaccard(query_kmers, kmer_index[target_key])
        prefilter_hits.append({
            "source_path": target_key,
            "sequence_score": score,
            "kmer_score": score,
            "sequence_identity_proxy": score,
            "query_length": len(query_sequence),
            "target_length": len(target_sequence),
        })
    prefilter_hits.sort(key=lambda hit: (-hit["sequence_score"], Path(hit["source_path"]).name))
    kept = prefilter_hits[:kmer_prefilter_top_k]
    if sw_rerank_top_k <= 0:
        return kept

    rerank_count = min(sw_rerank_top_k, len(kept))
    reranked = []
    for hit in kept[:rerank_count]:
        target_sequence = sequences[hit["source_path"]]
        sw_score = smith_waterman_score(
            query_sequence,
            target_sequence,
            match=sw_match,
            mismatch=sw_mismatch,
            gap=sw_gap,
        )
        updated = dict(hit)
        updated["smith_waterman_score"] = sw_score
        updated["sequence_score"] = sw_score
        reranked.append(updated)
    reranked.sort(
        key=lambda hit: (
            -hit["sequence_score"],
            -hit["kmer_score"],
            Path(hit["source_path"]).name,
        )
    )
    return reranked + kept[rerank_count:]


def _fasta_id(index: int) -> str:
    return f"seq{index}"


def write_fasta(
    path: Path,
    entries: list[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in entries:
            handle.write(f">{name}\n")
            for idx in range(0, len(sequence), 80):
                handle.write(sequence[idx : idx + 80] + "\n")


def parse_hmmer_tblout(
    tblout_path: Path,
    id_to_path: dict[str, Path],
    sequences: dict[str, str],
) -> list[dict[str, Any]]:
    hits = []
    if not tblout_path.exists():
        return hits
    for line in tblout_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(maxsplit=18)
        if len(parts) < 6:
            continue
        target_id = parts[0]
        target_path = id_to_path.get(target_id)
        if target_path is None:
            continue
        evalue = float(parts[4])
        bits = float(parts[5])
        target_key = str(target_path)
        hits.append({
            "source_path": target_key,
            "sequence_score": bits,
            "hmmer_bits": bits,
            "hmmer_evalue": evalue,
            "query_length": None,
            "target_length": len(sequences.get(target_key, "")),
        })
    hits.sort(key=lambda hit: (-hit["hmmer_bits"], hit["hmmer_evalue"], Path(hit["source_path"]).name))
    return hits


def run_phmmer_hits(
    *,
    phmmer: Path,
    query_paths: list[Path],
    target_paths: list[Path],
    sequences: dict[str, str],
    top_k: int,
    threads: int,
    evalue: float,
    work_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    target_entries = []
    id_to_path = {}
    for idx, target_path in enumerate(target_paths):
        target_key = str(target_path)
        sequence = sequences.get(target_key)
        if not sequence:
            continue
        fasta_id = _fasta_id(idx)
        id_to_path[fasta_id] = target_path
        target_entries.append((fasta_id, sequence))
    target_fasta = work_dir / "targets.fasta"
    write_fasta(target_fasta, target_entries)

    hits_by_query = {}
    for idx, query_path in enumerate(query_paths, start=1):
        query_key = str(query_path)
        query_sequence = sequences.get(query_key)
        if not query_sequence:
            hits_by_query[query_key] = []
            continue
        query_fasta = work_dir / f"query_{idx}.fasta"
        tblout = work_dir / f"query_{idx}.tblout"
        write_fasta(query_fasta, [(query_path.stem, query_sequence)])
        cmd = [
            str(phmmer),
            "--tblout",
            str(tblout),
            "--noali",
            "--cpu",
            str(threads),
            "-E",
            str(evalue),
            str(query_fasta),
            str(target_fasta),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"phmmer failed for {query_path.name} with exit code {result.returncode}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        hits = parse_hmmer_tblout(tblout, id_to_path, sequences)
        for hit in hits:
            hit["query_length"] = len(query_sequence)
        hits_by_query[query_key] = hits[:top_k]
    return hits_by_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", default="/scratch/TMAlign/proteon/validation/pdbs_10k")
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truth-cache", required=True)
    parser.add_argument("--method", choices=["kmer-sw", "phmmer"], default="kmer-sw")
    parser.add_argument("--phmmer", default="/globalscratch/dateschn/proteon-benchmark/bin/phmmer")
    parser.add_argument("--hmmer-evalue", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--kmer-size", type=int, default=3)
    parser.add_argument("--kmer-prefilter-top-k", type=int, default=1000)
    parser.add_argument("--sw-rerank-top-k", type=int, default=200)
    parser.add_argument("--sw-match", type=int, default=2)
    parser.add_argument("--sw-mismatch", type=int, default=-1)
    parser.add_argument("--sw-gap", type=int, default=-2)
    parser.add_argument("--min-sequence-length", type=int, default=20)
    parser.add_argument("--max-sequence-length", type=int, default=100000)
    parser.add_argument("--sequence-cache", default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--diagnostic-top-k", type=int, default=100)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--output", default="validation/sequence_retrieval_bench.json")
    args = parser.parse_args()
    phmmer = Path(args.phmmer)
    if args.method == "phmmer" and not phmmer.exists():
        raise SystemExit(f"phmmer executable not found: {phmmer}")

    pdb_dir = Path(args.pdb_dir)
    all_paths = collect_structure_paths(pdb_dir)
    if not all_paths:
        raise SystemExit(f"No structures found in {pdb_dir}")

    target_paths = sample_targets(all_paths, args.n_targets, args.seed)
    query_paths = sample_queries(target_paths, args.n_queries, args.seed)
    print(f"Sampled {len(target_paths)} targets and {len(query_paths)} queries from {pdb_dir}", flush=True)

    t0 = time.time()
    sequence_cache_path = Path(args.sequence_cache) if args.sequence_cache else None
    cached_sequences = (
        load_sequence_cache(sequence_cache_path, target_paths)
        if sequence_cache_path is not None
        else None
    )
    if cached_sequences is None:
        sequences, skipped_sequences = load_sequences(
            target_paths,
            min_length=args.min_sequence_length,
            max_length=args.max_sequence_length,
        )
        if sequence_cache_path is not None:
            save_sequence_cache(
                sequence_cache_path,
                target_paths=target_paths,
                sequences=sequences,
                skipped_sequences=skipped_sequences,
            )
        sequence_cache_hit = False
    else:
        sequences, skipped_sequences = cached_sequences
        sequence_cache_hit = True
    sequence_load_s = time.time() - t0
    print(f"Loaded {len(sequences)} sequences in {sequence_load_s:.3f}s ({len(skipped_sequences)} skipped)", flush=True)

    query_paths = [path for path in query_paths if str(path) in sequences]
    if not query_paths:
        raise SystemExit("No query sequences remain after sequence extraction")

    truth_payload = json.loads(Path(args.truth_cache).read_text(encoding="utf-8"))
    truth_cache = truth_payload.get("truth", truth_payload)

    hits_by_query: dict[str, list[dict[str, Any]]] = {}
    t0 = time.time()
    if args.method == "phmmer":
        if args.work_dir is None:
            work_context = tempfile.TemporaryDirectory(prefix="proteon_phmmer_")
            work_root = Path(work_context.name)
        else:
            work_context = None
            work_root = Path(args.work_dir)
            shutil.rmtree(work_root, ignore_errors=True)
            work_root.mkdir(parents=True, exist_ok=True)
        print(f"Running phmmer in {work_root}...", flush=True)
        try:
            hits_by_query = run_phmmer_hits(
                phmmer=phmmer,
                query_paths=query_paths,
                target_paths=target_paths,
                sequences=sequences,
                top_k=max(args.kmer_prefilter_top_k, args.diagnostic_top_k, args.top_k),
                threads=args.threads,
                evalue=args.hmmer_evalue,
                work_dir=work_root,
            )
        finally:
            if args.keep_work_dir:
                print(f"Kept work dir: {work_root}", flush=True)
            elif work_context is not None:
                work_context.cleanup()
    else:
        kmer_index = {
            key: kmer_counts(sequence, args.kmer_size)
            for key, sequence in sequences.items()
        }
        for idx, query_path in enumerate(query_paths, start=1):
            print(f"Sequence search {idx}/{len(query_paths)} {query_path.name}...", flush=True)
            hits_by_query[str(query_path)] = rank_sequence_hits(
                query_path,
                target_paths,
                sequences,
                kmer_index,
                kmer_prefilter_top_k=args.kmer_prefilter_top_k,
                sw_rerank_top_k=args.sw_rerank_top_k,
                sw_match=args.sw_match,
                sw_mismatch=args.sw_mismatch,
                sw_gap=args.sw_gap,
            )
    sequence_search_s = time.time() - t0

    threshold_metrics = {str(threshold): [] for threshold in args.thresholds}
    top1_exact = []
    per_query = []
    for query_path in query_paths:
        query_key = str(query_path)
        truth_rows = truth_cache.get(query_key, [])
        hits = hits_by_query.get(query_key, [])
        best_truth = best_nonself_row(truth_rows, query_path=query_path)
        top_nonself = best_nonself_row(hits, query_path=query_path)
        top1_exact.append(
            int(best_truth is not None and top_nonself is not None and top_nonself["source_path"] == best_truth["source_path"])
        )
        row = {
            "query": query_path.name,
            "query_path": query_key,
            "query_length": len(sequences[query_key]),
            "n_sequence_hits": len(hits),
            "best_truth_nonself": None if best_truth is None else {
                "source_path": Path(best_truth["source_path"]).name,
                "tm_score": round(best_truth["tm_score"], 4),
            },
            "sequence_top_nonself": None if top_nonself is None else {
                "source_path": Path(top_nonself["source_path"]).name,
                "sequence_score": round(top_nonself["sequence_score"], 4),
                "kmer_score": None if "kmer_score" not in top_nonself else round(top_nonself["kmer_score"], 4),
                "smith_waterman_score": None if "smith_waterman_score" not in top_nonself else round(top_nonself["smith_waterman_score"], 4),
                "hmmer_bits": None if "hmmer_bits" not in top_nonself else round(top_nonself["hmmer_bits"], 4),
                "hmmer_evalue": top_nonself.get("hmmer_evalue"),
                "query_length": top_nonself["query_length"],
                "target_length": top_nonself["target_length"],
            },
            "sequence_top_hits": [
                {
                    "source_path": Path(hit["source_path"]).name,
                    "sequence_score": round(hit["sequence_score"], 4),
                    "kmer_score": None if "kmer_score" not in hit else round(hit["kmer_score"], 4),
                    "smith_waterman_score": None if "smith_waterman_score" not in hit else round(hit["smith_waterman_score"], 4),
                    "hmmer_bits": None if "hmmer_bits" not in hit else round(hit["hmmer_bits"], 4),
                    "hmmer_evalue": hit.get("hmmer_evalue"),
                    "target_length": hit["target_length"],
                }
                for hit in hits
                if Path(hit["source_path"]) != query_path
            ][:args.diagnostic_top_k],
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
            row["thresholds"][str(threshold)] = {"sequence_recall_at_k": round(recall, 4)}
        per_query.append(row)
        print(
            f"{query_path.name}: truth {row['best_truth_nonself']} sequence {row['sequence_top_nonself']} hits {len(hits)}",
            flush=True,
        )

    metrics = {
        "sequence_top1_exact_nonself": round(mean(top1_exact), 4) if top1_exact else 0.0,
        "sequence_recall_at_k": {
            threshold: round(mean(values), 4) if values else 0.0
            for threshold, values in threshold_metrics.items()
        },
    }
    summary = {
        "corpus": {
            "pdb_dir": str(pdb_dir),
            "n_targets": len(target_paths),
            "n_queries": len(query_paths),
            "top_k": args.top_k,
            "thresholds": args.thresholds,
            "truth_cache": str(args.truth_cache),
            "method": args.method,
            "phmmer": str(phmmer) if args.method == "phmmer" else None,
            "hmmer_evalue": args.hmmer_evalue if args.method == "phmmer" else None,
            "kmer_size": args.kmer_size,
            "sequence_cache": str(sequence_cache_path) if sequence_cache_path else None,
            "sequence_cache_hit": sequence_cache_hit,
            "min_sequence_length": args.min_sequence_length,
            "max_sequence_length": args.max_sequence_length,
            "kmer_prefilter_top_k": args.kmer_prefilter_top_k,
            "sw_rerank_top_k": args.sw_rerank_top_k,
            "diagnostic_top_k": args.diagnostic_top_k,
        },
        "timing": {
            "sequence_load_s": round(sequence_load_s, 3),
            "sequence_search_s": round(sequence_search_s, 3),
        },
        "metrics": metrics,
        "per_query": per_query,
        "skipped_sequences": skipped_sequences,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(json.dumps(metrics, indent=2))
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
