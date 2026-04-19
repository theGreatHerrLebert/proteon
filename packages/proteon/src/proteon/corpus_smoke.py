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
from .msa_io import load_msas_from_dir
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
    msa_engine: Optional[object] = None,
    msa_max_seqs: int = 256,
    msa_gap_idx: int = 21,
    msa_dir: Optional[str | Path] = None,
    msa_suffix: str = ".a3m",
    msa_strict: bool = False,
    chunk_size: Optional[int] = None,
    supervision_row_group_size: int = 512,
    sequence_row_group_size: int = 64,
    overwrite: bool = False,
) -> Path:
    """Build a small end-to-end corpus release from local structure files.

    This is the intended smoke path for validating the full data-release stack
    on real PDB/mmCIF inputs already available on disk.

    MSA inputs (any combination):
      - `msa_dir` — directory of pre-computed a3m files, keyed by
        record_id (`{msa_dir}/{record_id}{msa_suffix}`). Missing files
        silently fall through to the engine path unless `msa_strict`.
      - `msa_engine` — optional `proteon_connector.py_msa.SearchEngine`
        (or duck-compatible wrapper) run per record that `msa_dir`
        didn't cover. `msa_max_seqs` / `msa_gap_idx` flow through to
        `search_and_build_msa`.

    When both are None (the default), sequence examples carry null MSA
    fields and downstream AF2-style pipelines either fall back to
    single-sequence input or supply MSAs some other way.

    `chunk_size` — when set (and >0), runs load + prep + expand + emit
    in chunks of that many **input paths**, keeping only one chunk's
    pdbtbx Structure objects resident at a time. Peak RSS stays ~flat
    regardless of corpus size (bounded by `chunk_size × avg_chains`
    rather than `total_paths × avg_chains`). Use for archive-scale
    runs where the single-shot path would OOM.
    """
    if chunk_size is not None and chunk_size > 0:
        return _build_local_corpus_smoke_release_chunked(
            paths,
            out_dir,
            release_id=release_id,
            code_rev=code_rev,
            config_rev=config_rev,
            prep_policy_version=prep_policy_version,
            split_policy_version=split_policy_version,
            n_threads=n_threads,
            rescue_load=rescue_load,
            rescue_allow=rescue_allow,
            split_assignments=split_assignments,
            split_ratios=split_ratios,
            msa_engine=msa_engine,
            msa_max_seqs=msa_max_seqs,
            msa_gap_idx=msa_gap_idx,
            msa_dir=msa_dir,
            msa_suffix=msa_suffix,
            msa_strict=msa_strict,
            chunk_size=chunk_size,
            supervision_row_group_size=supervision_row_group_size,
            sequence_row_group_size=sequence_row_group_size,
            overwrite=overwrite,
        )

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
        expanded_parent_record_ids,
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
    explicit_msas: Optional[List[Optional[List[str]]]] = None
    explicit_deletions: Optional[List[Optional[List[List[int]]]]] = None
    if msa_dir is not None:
        explicit_msas, explicit_deletions = load_msas_from_dir(
            msa_dir,
            expanded_record_ids,
            suffix=msa_suffix,
            strict=msa_strict,
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
        msas=explicit_msas,
        deletion_matrices=explicit_deletions,
        msa_engine=msa_engine,
        msa_max_seqs=msa_max_seqs,
        msa_gap_idx=msa_gap_idx,
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
        split_assignments = _hash_split_assignments(
            expanded_record_ids,
            split_ratios,
            grouping_keys=expanded_parent_record_ids,
        )
        split_strategy = f"hash_split:{_format_ratios(split_ratios)}"
    else:
        split_assignments = _default_split_assignments(
            expanded_record_ids, grouping_keys=expanded_parent_record_ids
        )
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


def _build_local_corpus_smoke_release_chunked(
    paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    release_id: str,
    code_rev: Optional[str],
    config_rev: Optional[str],
    prep_policy_version: Optional[str],
    split_policy_version: Optional[str],
    n_threads: Optional[int],
    rescue_load: bool,
    rescue_allow: Optional[Sequence[str]],
    split_assignments: Optional[Mapping[str, str]],
    split_ratios: Optional[Mapping[str, float]],
    msa_engine: Optional[object],
    msa_max_seqs: int,
    msa_gap_idx: int,
    msa_dir: Optional[str | Path],
    msa_suffix: str,
    msa_strict: bool,
    chunk_size: int,
    supervision_row_group_size: int,
    sequence_row_group_size: int,
    overwrite: bool,
) -> Path:
    """Chunked intake path: load+prep+expand+emit per chunk, drop, repeat.

    The single-shot path holds every pdbtbx Structure object in RAM
    between `batch_load_tolerant` and the supervision+sequence write
    — that's the ~20 GB floor we measured on the 1K rerun. The chunked
    path keeps ≤ chunk_size structures resident at once.

    Both paths use the same `SupervisionParquetWriter` and
    `SequenceParquetWriter` kept open across chunks, so the on-disk
    artifact is byte-identical regardless of chunk boundaries.
    """
    from .prepared_manifest import (
        build_prepared_structure_records,
        write_prepared_structure_manifest,
    )
    from .sequence_example import build_sequence_example
    from .sequence_export import SequenceParquetWriter
    from .sequence_release import SequenceReleaseManifest, _length_summary as _seq_len_summary
    from .supervision import build_structure_supervision_example
    from .supervision_export import SupervisionParquetWriter
    from .supervision_release import StructureSupervisionReleaseManifest

    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    path_list = [Path(p) for p in paths]
    n_total = len(path_list)

    # Outer accumulators — these stay small (strings/ints, one per chain).
    ordered_loaded_indices: List[int] = []
    rescued_results: List[tuple[int, "LoadRescueResult"]] = []

    expanded_record_ids: List[str] = []
    expanded_parent_record_ids: List[str] = []
    expanded_source_ids: List[str] = []
    expanded_chain_ids: List[Optional[str]] = []
    expanded_paths: List[Path] = []
    prepared_rows: List[dict] = []

    sup_failures: List[FailureRecord] = []
    seq_failures: List[FailureRecord] = []
    sup_lengths: List[int] = []
    seq_lengths: List[int] = []

    prepared_root = root / "prepared"
    sup_release_root = prepared_root / "supervision_release"
    seq_release_root = root / "sequence"
    sup_release_root.mkdir(parents=True, exist_ok=True)
    seq_release_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)

    sup_example_dir = sup_release_root / "examples"
    seq_example_dir = seq_release_root / "examples"

    with SupervisionParquetWriter(sup_example_dir, row_group_size=supervision_row_group_size) as sup_writer, \
         SequenceParquetWriter(seq_example_dir, row_group_size=sequence_row_group_size) as seq_writer:
        for chunk_start in range(0, n_total, chunk_size):
            chunk_paths = path_list[chunk_start : chunk_start + chunk_size]

            # --- load ---
            if rescue_load:
                loaded_pairs = batch_load_tolerant_with_rescue(
                    chunk_paths, n_threads=n_threads, allow=rescue_allow,
                )
                local_indices = [idx for idx, _ in loaded_pairs]
                chunk_structures = [r.structure for _, r in loaded_pairs]
                for idx, r in loaded_pairs:
                    if r.rescued:
                        rescued_results.append((chunk_start + idx, r))
            else:
                raw_pairs = batch_load_tolerant(chunk_paths, n_threads=n_threads)
                local_indices = [idx for idx, _ in raw_pairs]
                chunk_structures = [s for _, s in raw_pairs]

            ordered_loaded_indices.extend(chunk_start + i for i in local_indices)

            chunk_loaded_paths = [chunk_paths[i] for i in local_indices]
            chunk_record_ids = [p.stem for p in chunk_loaded_paths]
            chunk_source_ids = [str(p) for p in chunk_loaded_paths]

            # --- prep ---
            chunk_prep_reports = batch_prepare(chunk_structures, n_threads=n_threads)

            # --- expand chains (with multi-model dedup) ---
            (
                e_structs,
                e_preps,
                e_rids,
                e_srcs,
                e_cids,
                e_paths,
                e_parents,
            ) = _expand_chains(
                chunk_structures,
                chunk_prep_reports,
                chunk_record_ids,
                chunk_source_ids,
                chunk_loaded_paths,
            )

            # Prepared-structure rows are one per loaded STRUCTURE (not per
            # chain); matches the single-shot pipeline's manifest contract.
            chunk_prepared_rows = build_prepared_structure_records(
                chunk_structures,
                chunk_prep_reports,
                record_ids=chunk_record_ids,
                source_ids=chunk_source_ids,
                prep_run_ids=None,
                code_rev=code_rev,
                config_rev=config_rev,
                provenance={"input_paths": [str(p) for p in chunk_loaded_paths]},
            )
            prepared_rows.extend(chunk_prepared_rows)

            # --- per-record emission ---
            if msa_dir is not None:
                chunk_msas, chunk_deletions = load_msas_from_dir(
                    msa_dir, e_rids, suffix=msa_suffix, strict=msa_strict,
                )
            else:
                chunk_msas = [None] * len(e_rids)
                chunk_deletions = [None] * len(e_rids)

            for j in range(len(e_rids)):
                struct = e_structs[j]
                prep = e_preps[j]
                rid = e_rids[j]
                src = e_srcs[j]
                cid = e_cids[j]
                path = e_paths[j]
                parent = e_parents[j]

                # Provenance tracking: one entry per *attempted* record,
                # regardless of per-stage success. Matches the single-shot
                # path — both supervision and sequence release manifests
                # list every path the release tried to process, with
                # per-stage failure detail in failures.jsonl. Prior
                # behavior filtered this list to "sequence-succeeded"
                # records only, which silently omitted supervision-only
                # successes from the supervision manifest provenance.
                expanded_record_ids.append(rid)
                expanded_parent_record_ids.append(parent)
                expanded_source_ids.append(src)
                expanded_chain_ids.append(cid)
                expanded_paths.append(path)

                # Supervision and sequence are independent per-record
                # attempts. Each has its own try/except + failures.jsonl
                # entry; a failure on one doesn't skip the other. This
                # mirrors `build_structure_supervision_dataset` and
                # `build_sequence_dataset` in the single-shot path,
                # which iterate the same structure list independently.
                from .failure_taxonomy import classify_exception
                try:
                    sup_ex = build_structure_supervision_example(
                        struct,
                        prep_report=prep,
                        record_id=rid,
                        source_id=src,
                        chain_id=cid,
                        code_rev=code_rev,
                        config_rev=config_rev,
                    )
                    sup_writer.append(sup_ex)
                    sup_lengths.append(int(sup_ex.length))
                except Exception as exc:
                    sup_failures.append(
                        FailureRecord(
                            record_id=rid,
                            stage="structure_supervision_example",
                            failure_class=classify_exception(exc),
                            message=str(exc),
                            source_id=src,
                            code_rev=code_rev,
                            config_rev=config_rev,
                            provenance={"exception_type": type(exc).__name__},
                        )
                    )

                try:
                    if msa_engine is not None and chunk_msas[j] is None and chunk_deletions[j] is None:
                        from .msa_backend import build_sequence_example_with_msa
                        seq_ex = build_sequence_example_with_msa(
                            struct,
                            msa_engine,
                            record_id=rid,
                            source_id=src,
                            chain_id=cid,
                            code_rev=code_rev,
                            config_rev=config_rev,
                            max_seqs=msa_max_seqs,
                            gap_idx=msa_gap_idx,
                        )
                    else:
                        seq_ex = build_sequence_example(
                            struct,
                            record_id=rid,
                            source_id=src,
                            chain_id=cid,
                            code_rev=code_rev,
                            config_rev=config_rev,
                            msa=chunk_msas[j],
                            deletion_matrix=chunk_deletions[j],
                        )
                    seq_writer.append(seq_ex)
                    seq_lengths.append(int(seq_ex.length))
                except Exception as exc:
                    seq_failures.append(
                        FailureRecord(
                            record_id=rid,
                            stage="sequence_example",
                            failure_class=classify_exception(exc),
                            message=str(exc),
                            source_id=src,
                            code_rev=code_rev,
                            config_rev=config_rev,
                            provenance={"exception_type": type(exc).__name__},
                        )
                    )

            # --- drop chunk refs — the critical memory step ---
            del chunk_structures
            del chunk_prep_reports
            del e_structs, e_preps
            if rescue_load:
                del loaded_pairs
            else:
                del raw_pairs

    # ---- writers closed; finalize release directories ----
    loaded_paths = [path_list[i] for i in ordered_loaded_indices]
    loaded_set = set(ordered_loaded_indices)
    dropped_indices = [i for i in range(n_total) if i not in loaded_set]

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

    # Prepared-structure manifest (one row per loaded structure).
    write_prepared_structure_manifest(prepared_rows, prepared_root / "prepared_structures.jsonl")

    # Supervision release manifest.
    sup_failure_file = sup_release_root / "failures.jsonl"
    with sup_failure_file.open("w", encoding="utf-8") as handle:
        for failure in sup_failures:
            handle.write(json.dumps(asdict(failure), separators=(",", ":")))
            handle.write("\n")
    sup_count = len(sup_lengths)
    sup_manifest = StructureSupervisionReleaseManifest(
        release_id=f"{release_id}-structure",
        code_rev=code_rev,
        config_rev=config_rev,
        count_examples=sup_count,
        count_failures=len(sup_failures),
        tensor_file="examples/tensors.parquet" if sup_count > 0 else None,
        lengths=_seq_len_summary(sup_lengths),
        sequence_lengths=sup_lengths,
        provenance={"input_paths": [str(p) for p in expanded_paths]},
    )
    (sup_release_root / "release_manifest.json").write_text(
        json.dumps(asdict(sup_manifest), indent=2), encoding="utf-8",
    )

    # Sequence release manifest.
    seq_failure_file = seq_release_root / "failures.jsonl"
    with seq_failure_file.open("w", encoding="utf-8") as handle:
        for failure in seq_failures:
            handle.write(json.dumps(asdict(failure), separators=(",", ":")))
            handle.write("\n")
    seq_count = len(seq_lengths)
    seq_manifest = SequenceReleaseManifest(
        release_id=f"{release_id}-sequence",
        code_rev=code_rev,
        config_rev=config_rev,
        count_examples=seq_count,
        count_failures=len(seq_failures),
        tensor_file="examples/tensors.parquet" if seq_count > 0 else None,
        lengths=_seq_len_summary(seq_lengths),
        provenance={"input_paths": [str(p) for p in expanded_paths]},
    )
    (seq_release_root / "release_manifest.json").write_text(
        json.dumps(asdict(seq_manifest), indent=2), encoding="utf-8",
    )

    # Split assignment (same logic as the single-shot path).
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
        split_assignments = _hash_split_assignments(
            expanded_record_ids, split_ratios, grouping_keys=expanded_parent_record_ids,
        )
        split_strategy = f"hash_split:{_format_ratios(split_ratios)}"
    else:
        split_assignments = _default_split_assignments(
            expanded_record_ids, grouping_keys=expanded_parent_record_ids,
        )
        split_strategy = f"default_hash_split:{_format_ratios(DEFAULT_SPLIT_RATIOS)}"

    training_root = build_training_release(
        seq_release_root,
        sup_release_root,
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
        sequence_release=seq_release_root,
        structure_release=sup_release_root,
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
            "chunk_size": chunk_size,
        },
        overwrite=True,
    )
    validate_corpus_release(
        corpus_root / "corpus_release_manifest.json",
        out_path=corpus_root / "validation_report.json",
    )
    return root


DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def _default_split_assignments(
    record_ids: Iterable[str],
    *,
    grouping_keys: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    """Hash-split at DEFAULT_SPLIT_RATIOS (80/10/10).

    The prior behavior ("everything train except the last record, which
    goes to val") produced artifacts with no test split and train≈100%,
    which silently broke downstream evaluation. The default is now an
    80/10/10 deterministic hash-split — same record_id gives the same
    bucket regardless of input order or corpus size.

    `grouping_keys` — optional parallel sequence. When provided, records
    sharing a grouping key share a split (all chains of the same source
    structure go to the same bucket). Without this, multi-chain complexes
    leak across train/val/test.
    """
    return _hash_split_assignments(
        record_ids, DEFAULT_SPLIT_RATIOS, grouping_keys=grouping_keys
    )


def _expand_chains(
    structures: Sequence,
    prep_reports: Sequence,
    record_ids: Sequence[str],
    source_ids: Sequence[str],
    paths: Sequence[Path],
) -> tuple[list, list, list[str], list[str], list[Optional[str]], list[Path], list[str]]:
    """Expand per-structure entries into per-chain entries.

    Structure/sequence supervision v0 is chain-scoped. Single-chain
    structures pass through unchanged with chain_id=None so existing
    record_ids are preserved; multi-chain structures produce one
    entry per chain with record_id=f"{stem}_{chain_id}".

    Returns a parallel `parent_record_ids` list so downstream split
    assignment can bucket sibling chains (e.g. `p0002_A` and `p0002_B`)
    into the same split. Without this, a hash-split over the expanded
    IDs leaks chains of the same biological complex across
    train/val/test.
    """
    ex_structs: list = []
    ex_prep: list = []
    ex_rids: list[str] = []
    ex_srcs: list[str] = []
    ex_chains: list[Optional[str]] = []
    ex_paths: list[Path] = []
    ex_parent_rids: list[str] = []

    for structure, prep_report, rid, src, path in zip(
        structures, prep_reports, record_ids, source_ids, paths
    ):
        raw_chains = list(getattr(structure, "chains", []) or [])
        # pdbtbx / proteon's Structure.chains flattens chains across all
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
            ex_parent_rids.append(rid)
            continue
        for chain in chains:
            chain_id = getattr(chain, "id", None)
            ex_structs.append(structure)
            ex_prep.append(prep_report)
            ex_rids.append(f"{rid}_{chain_id}" if chain_id else rid)
            ex_srcs.append(src)
            ex_chains.append(chain_id)
            ex_paths.append(path)
            ex_parent_rids.append(rid)
    return ex_structs, ex_prep, ex_rids, ex_srcs, ex_chains, ex_paths, ex_parent_rids


def _hash_split_assignments(
    record_ids: Iterable[str],
    ratios: Mapping[str, float],
    *,
    grouping_keys: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    """Deterministic hash-split.

    Each *grouping key* hashes into [0, 1) via blake2b; ratios define
    contiguous buckets. Same key + same ratios ⇒ same split, regardless
    of input order or corpus size.

    `grouping_keys` — optional parallel sequence. When provided, records
    sharing a grouping key are assigned the same split. When omitted,
    each record_id is its own grouping key (per-record hashing).

    The grouping-key path is how we keep sibling chains of the same
    source structure together in the default split: pass the pre-chain-
    expansion parent record_id as the grouping key.
    """
    record_list = list(record_ids)
    if grouping_keys is None:
        grouping_list = record_list
    else:
        grouping_list = list(grouping_keys)
        if len(grouping_list) != len(record_list):
            raise ValueError(
                f"grouping_keys length {len(grouping_list)} != record_ids length "
                f"{len(record_list)}; they must align one-to-one"
            )
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

    # Hash each grouping_key once, cache the split, then fan out to records.
    group_to_split: dict[str, str] = {}
    for gk in grouping_list:
        if gk in group_to_split:
            continue
        digest = blake2b(gk.encode("utf-8"), digest_size=8).digest()
        u = int.from_bytes(digest, "big") / 2**64
        for name, cutoff in cumulative:
            if u < cutoff:
                group_to_split[gk] = name
                break

    return {rid: group_to_split[gk] for rid, gk in zip(record_list, grouping_list)}


def _format_ratios(ratios: Mapping[str, float]) -> str:
    total = sum(ratios.values()) or 1.0
    return ",".join(f"{k}={v/total:.3f}" for k, v in sorted(ratios.items()))
