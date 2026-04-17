"""Smoke pipeline for building a small real-file corpus release."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import blake2b
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from .corpus_release import build_corpus_release_manifest
from .corpus_validation import validate_corpus_release
from .failure_taxonomy import PARSE_ERROR
from .io import LoadRescueResult, batch_load_tolerant, batch_load_tolerant_with_rescue
from .prepare import batch_prepare
from .sequence_release import build_sequence_dataset
from .supervision_dataset import build_structure_supervision_dataset_from_prepared
from .supervision_release import FailureRecord
from .training_example import build_training_release


def build_local_corpus_smoke_release(
    paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    release_id: str,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    prep_policy_version: Optional[str] = None,
    split_policy_version: Optional[str] = None,
    n_threads: Optional[int] = None,
    rescue_load: bool = False,
    rescue_allow: Optional[Sequence[str]] = None,
    split_assignments: Optional[Mapping[str, str]] = None,
    split_ratios: Optional[Mapping[str, float]] = None,
    overwrite: bool = False,
) -> Path:
    """Build a small end-to-end corpus release from local structure files.

    This is the intended smoke path for validating the full data-release stack
    on real PDB/mmCIF inputs already available on disk.
    """
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    path_list = [Path(p) for p in paths]
    if rescue_load:
        loaded_pairs = batch_load_tolerant_with_rescue(
            path_list,
            n_threads=n_threads,
            allow=rescue_allow,
        )
        loaded_indices = [idx for idx, _ in loaded_pairs]
        loaded_structures = [result.structure for _, result in loaded_pairs]
        rescued_results = [
            (idx, result) for idx, result in loaded_pairs if result.rescued
        ]
    else:
        raw_pairs = batch_load_tolerant(path_list, n_threads=n_threads)
        loaded_indices = [idx for idx, _ in raw_pairs]
        loaded_structures = [structure for _, structure in raw_pairs]
        rescued_results = []
    loaded_paths = [path_list[idx] for idx in loaded_indices]
    record_ids = [path.stem for path in loaded_paths]
    source_ids = [str(path) for path in loaded_paths]

    # Capture ingestion failures. `batch_load_tolerant` silently drops
    # bad paths; if we only forward what it returned we lose visibility
    # into partial ingestion. Roadmap Section 7 requires every failure
    # mode to be machine-readable, so we write one FailureRecord per
    # dropped path and merge the breakdown into the top-level manifest.
    loaded_set = set(loaded_indices)
    dropped_indices = [i for i in range(len(path_list)) if i not in loaded_set]
    ingestion_failures_path = _write_ingestion_failures(
        root,
        [path_list[i] for i in dropped_indices],
        code_rev=code_rev,
        config_rev=config_rev,
    )
    rescued_inputs_path = _write_rescued_inputs(
        root,
        path_list,
        rescued_results,
        code_rev=code_rev,
        config_rev=config_rev,
    )

    prep_reports = batch_prepare(loaded_structures, n_threads=n_threads)

    # Structure/sequence supervision v0 is chain-level. Expand each loaded
    # structure into one record per chain so multi-chain PDBs don't get
    # silently dropped during the supervision build. Single-chain inputs
    # keep their original record_id and pass chain_id=None for backward
    # compatibility with pre-expansion provenance.
    (
        expanded_structures,
        expanded_prep_reports,
        expanded_record_ids,
        expanded_source_ids,
        expanded_chain_ids,
        expanded_paths,
    ) = _expand_chains(loaded_structures, prep_reports, record_ids, source_ids, loaded_paths)

    prepared_root = build_structure_supervision_dataset_from_prepared(
        expanded_structures,
        expanded_prep_reports,
        root / "prepared",
        release_id=f"{release_id}-structure",
        record_ids=expanded_record_ids,
        source_ids=expanded_source_ids,
        chain_ids=expanded_chain_ids,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance={"input_paths": [str(p) for p in expanded_paths]},
        overwrite=True,
    )
    sequence_root = build_sequence_dataset(
        expanded_structures,
        root / "sequence",
        release_id=f"{release_id}-sequence",
        record_ids=expanded_record_ids,
        source_ids=expanded_source_ids,
        chain_ids=expanded_chain_ids,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance={"input_paths": [str(p) for p in expanded_paths]},
        overwrite=True,
    )
    # Splits are assigned at the expanded (chain-level) record granularity
    # so multi-chain structures can have per-chain split decisions.
    if split_assignments is not None:
        missing = [rid for rid in expanded_record_ids if rid not in split_assignments]
        if missing:
            raise ValueError(
                f"split_assignments missing {len(missing)} record_ids "
                f"(first few: {missing[:3]})"
            )
        split_assignments = {rid: split_assignments[rid] for rid in expanded_record_ids}
        split_strategy = "explicit"
    elif split_ratios is not None:
        split_assignments = _hash_split_assignments(expanded_record_ids, split_ratios)
        split_strategy = f"hash_split:{_format_ratios(split_ratios)}"
    else:
        split_assignments = _default_split_assignments(expanded_record_ids)
        split_strategy = f"default_hash_split:{_format_ratios(DEFAULT_SPLIT_RATIOS)}"
    training_root = build_training_release(
        sequence_root,
        prepared_root / "supervision_release",
        root / "training",
        release_id=f"{release_id}-training",
        split_assignments=split_assignments,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance={"input_paths": [str(p) for p in loaded_paths]},
        overwrite=True,
    )
    corpus_root = build_corpus_release_manifest(
        root / "corpus",
        release_id=release_id,
        prepared_manifest=prepared_root / "prepared_structures.jsonl",
        sequence_release=sequence_root,
        structure_release=prepared_root / "supervision_release",
        training_release=training_root,
        ingestion_failures=ingestion_failures_path,
        rescued_inputs_manifest=rescued_inputs_path,
        code_rev=code_rev,
        config_rev=config_rev,
        prep_policy_version=prep_policy_version,
        split_policy_version=split_policy_version,
        provenance={
            "input_paths": [str(p) for p in path_list],
            "loaded_paths": [str(p) for p in loaded_paths],
            "dropped_paths": [str(path_list[i]) for i in dropped_indices],
            "rescue_load": rescue_load,
            "rescued_paths": [str(path_list[i]) for i, _ in rescued_results],
            "rescue_allow": list(rescue_allow) if rescue_allow is not None else None,
            "split_strategy": split_strategy,
        },
        overwrite=True,
    )
    validate_corpus_release(
        corpus_root / "corpus_release_manifest.json",
        out_path=corpus_root / "validation_report.json",
    )
    return root


def _write_ingestion_failures(
    root: Path,
    dropped_paths: List[Path],
    *,
    code_rev: Optional[str],
    config_rev: Optional[str],
) -> Optional[Path]:
    """Write `FailureRecord` JSONL for inputs that didn't load.

    Returns `None` when nothing was dropped — callers / the corpus
    manifest builder treat `None` as "no ingestion failures to merge",
    so we don't produce empty files.
    """
    if not dropped_paths:
        return None
    ingestion_path = root / "ingestion_failures.jsonl"
    with ingestion_path.open("w", encoding="utf-8") as handle:
        for path in dropped_paths:
            record = FailureRecord(
                record_id=path.stem,
                stage="raw_intake",
                failure_class=PARSE_ERROR,
                message=(
                    f"batch_load_tolerant could not parse {path}; "
                    "the upstream loader silently dropped this input"
                ),
                source_id=str(path),
                code_rev=code_rev,
                config_rev=config_rev,
                provenance={"stage_entry_point": "corpus_smoke.batch_load_tolerant"},
            )
            handle.write(json.dumps(asdict(record), separators=(",", ":")))
            handle.write("\n")
    return ingestion_path


def _write_rescued_inputs(
    root: Path,
    path_list: Sequence[Path],
    rescued_results: Sequence[tuple[int, LoadRescueResult]],
    *,
    code_rev: Optional[str],
    config_rev: Optional[str],
) -> Optional[Path]:
    """Write JSONL provenance rows for inputs that only loaded after rescue."""
    if not rescued_results:
        return None
    rescued_path = root / "rescued_inputs.jsonl"
    with rescued_path.open("w", encoding="utf-8") as handle:
        for idx, result in rescued_results:
            row = {
                "record_id": path_list[idx].stem,
                "artifact_type": "rescued_input",
                "status": "rescued",
                "source_id": str(path_list[idx]),
                "code_rev": code_rev,
                "config_rev": config_rev,
                "rescue_bucket": None if result.rescue_bucket is None else result.rescue_bucket.code,
                "rescue_steps": list(result.rescue_steps),
                "original_error": result.original_error,
                "provenance": {
                    "stage_entry_point": "corpus_smoke.batch_load_tolerant_with_rescue",
                },
            }
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
    return rescued_path


DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def _default_split_assignments(record_ids: Iterable[str]) -> dict[str, str]:
    """Hash-split at DEFAULT_SPLIT_RATIOS (80/10/10).

    The prior behavior ("everything train except the last record, which
    goes to val") produced artifacts with no test split and train≈100%,
    which silently broke downstream evaluation. The default is now an
    80/10/10 deterministic hash-split — same record_id gives the same
    bucket regardless of input order or corpus size.
    """
    return _hash_split_assignments(record_ids, DEFAULT_SPLIT_RATIOS)


def _expand_chains(
    structures: Sequence,
    prep_reports: Sequence,
    record_ids: Sequence[str],
    source_ids: Sequence[str],
    paths: Sequence[Path],
) -> tuple[list, list, list[str], list[str], list[Optional[str]], list[Path]]:
    """Expand per-structure entries into per-chain entries.

    Structure/sequence supervision v0 is chain-scoped. Single-chain
    structures pass through unchanged with chain_id=None so existing
    record_ids are preserved; multi-chain structures produce one
    entry per chain with record_id=f"{stem}_{chain_id}".
    """
    ex_structs: list = []
    ex_prep: list = []
    ex_rids: list[str] = []
    ex_srcs: list[str] = []
    ex_chains: list[Optional[str]] = []
    ex_paths: list[Path] = []

    for structure, prep_report, rid, src, path in zip(
        structures, prep_reports, record_ids, source_ids, paths
    ):
        raw_chains = list(getattr(structure, "chains", []) or [])
        # pdbtbx / ferritin's Structure.chains flattens chains across all
        # models, so NMR structures (and any multi-model input) repeat
        # every chain_id once per model. Dedupe by chain.id while
        # preserving first-seen order so we produce one record per
        # *logical* chain, not one per model × chain.
        seen: set = set()
        chains = []
        for chain in raw_chains:
            cid = getattr(chain, "id", None)
            if cid in seen:
                continue
            seen.add(cid)
            chains.append(chain)

        if len(chains) <= 1:
            ex_structs.append(structure)
            ex_prep.append(prep_report)
            ex_rids.append(rid)
            ex_srcs.append(src)
            ex_chains.append(None)
            ex_paths.append(path)
            continue
        for chain in chains:
            chain_id = getattr(chain, "id", None)
            ex_structs.append(structure)
            ex_prep.append(prep_report)
            ex_rids.append(f"{rid}_{chain_id}" if chain_id else rid)
            ex_srcs.append(src)
            ex_chains.append(chain_id)
            ex_paths.append(path)
    return ex_structs, ex_prep, ex_rids, ex_srcs, ex_chains, ex_paths


def _hash_split_assignments(
    record_ids: Iterable[str],
    ratios: Mapping[str, float],
) -> dict[str, str]:
    """Deterministic hash-split on record_id.

    Each record_id hashes into [0, 1) via blake2b; ratios define contiguous
    buckets in that interval. Same record_id + same ratios ⇒ same split,
    regardless of input order, corpus size, or which other records are present.
    """
    if not ratios:
        raise ValueError("split_ratios must be non-empty")
    total = sum(ratios.values())
    if total <= 0.0 or not all(v >= 0.0 for v in ratios.values()):
        raise ValueError(f"split_ratios must be non-negative and sum to > 0; got {dict(ratios)}")
    ordered_splits = sorted(ratios.items())
    cumulative: List[tuple[str, float]] = []
    running = 0.0
    for name, weight in ordered_splits:
        running += weight / total
        cumulative.append((name, running))
    cumulative[-1] = (cumulative[-1][0], 1.0)

    assignments: dict[str, str] = {}
    for rid in record_ids:
        digest = blake2b(rid.encode("utf-8"), digest_size=8).digest()
        u = int.from_bytes(digest, "big") / 2**64
        for name, cutoff in cumulative:
            if u < cutoff:
                assignments[rid] = name
                break
    return assignments


def _format_ratios(ratios: Mapping[str, float]) -> str:
    total = sum(ratios.values()) or 1.0
    return ",".join(f"{k}={v/total:.3f}" for k, v in sorted(ratios.items()))
