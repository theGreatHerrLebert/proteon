"""High-level dataset builder for structure supervision releases."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .prepare import PrepReport
from .prepared_manifest import (
    build_prepared_structure_records,
    write_prepared_structure_manifest,
)
from .supervision import (
    StructureSupervisionExample,
    build_structure_supervision_example,
)
from .supervision_release import (
    FailureRecord,
    build_structure_supervision_release,
)


def build_structure_supervision_dataset(
    structures: Sequence,
    out_dir: str | Path,
    *,
    release_id: str,
    prep_reports: Optional[Sequence[Optional[PrepReport]]] = None,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    prep_run_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
) -> Path:
    """Build a supervision release from structures, capturing failures."""
    n = len(structures)
    prep_reports = _expand_optional(prep_reports, n)
    record_ids = _expand_optional(record_ids, n)
    source_ids = _expand_optional(source_ids, n)
    prep_run_ids = _expand_optional(prep_run_ids, n)
    chain_ids = _expand_optional(chain_ids, n)

    # Streaming: a generator that yields one example per structure and
    # appends to `failures` out-of-band when build_*_example fails. The
    # caller (`build_structure_supervision_release`) consumes the
    # iterator and writes each example to the Parquet row-group buffer
    # without ever materializing the full example list.
    failures: list[FailureRecord] = []

    def _iter_examples():
        for i, structure in enumerate(structures):
            record_id = record_ids[i] or _default_record_id(structure, chain_ids[i])
            try:
                yield build_structure_supervision_example(
                    structure,
                    prep_report=prep_reports[i],
                    record_id=record_id,
                    source_id=source_ids[i],
                    prep_run_id=prep_run_ids[i],
                    chain_id=chain_ids[i],
                    code_rev=code_rev,
                    config_rev=config_rev,
                )
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        record_id=record_id,
                        failure_class=_classify_failure(exc),
                        message=str(exc),
                        source_id=source_ids[i],
                        prep_run_id=prep_run_ids[i],
                        code_rev=code_rev,
                        config_rev=config_rev,
                        provenance={"exception_type": type(exc).__name__},
                    )
                )

    return build_structure_supervision_release(
        _iter_examples(),
        out_dir,
        release_id=release_id,
        failures=failures,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance=provenance,
        overwrite=overwrite,
    )


def build_structure_supervision_dataset_from_prepared(
    structures: Sequence,
    prep_reports: Sequence[PrepReport],
    out_dir: str | Path,
    *,
    release_id: str,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    prep_run_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    overwrite: bool = False,
) -> Path:
    """Build prepared-structure manifest rows and a supervision release together."""
    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    prepared_rows = build_prepared_structure_records(
        structures,
        prep_reports,
        record_ids=record_ids,
        source_ids=source_ids,
        prep_run_ids=prep_run_ids,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance=provenance,
    )
    write_prepared_structure_manifest(prepared_rows, root / "prepared_structures.jsonl")

    build_structure_supervision_dataset(
        structures,
        root / "supervision_release",
        release_id=release_id,
        prep_reports=prep_reports,
        record_ids=record_ids,
        source_ids=source_ids,
        prep_run_ids=prep_run_ids,
        chain_ids=chain_ids,
        code_rev=code_rev,
        config_rev=config_rev,
        provenance=provenance,
        overwrite=True,
    )
    return root


def _classify_failure(exc: Exception) -> str:
    # Delegates to the shared roadmap-Section-7 taxonomy. Kept as a
    # thin wrapper so existing call sites don't need to change.
    from .failure_taxonomy import classify_exception
    return classify_exception(exc)


def _default_record_id(structure, chain_id: Optional[str]) -> str:
    ident = getattr(structure, "identifier", None) or "structure"
    if chain_id:
        return f"{ident}:{chain_id}"
    return str(ident)


def _expand_optional(values: Optional[Iterable], n: int) -> list:
    if values is None:
        return [None] * n
    values = list(values)
    if len(values) != n:
        raise ValueError(f"expected {n} items, got {len(values)}")
    return values
