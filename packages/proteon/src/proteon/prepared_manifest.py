"""Prepared-structure manifest rows for corpus building."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .prepare import PrepReport


@dataclass
class PreparedStructureRecord:
    """Machine-readable manifest row for one prepared structure."""

    record_id: str
    artifact_type: str = "prepared_structure"
    status: str = "ok"
    source_id: Optional[str] = None
    prep_run_id: Optional[str] = None
    chain_count: Optional[int] = None
    residue_count: Optional[int] = None
    atom_count: Optional[int] = None
    sequence_length: Optional[int] = None
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prep_success: bool = True
    minimizer_steps: int = 0
    converged: bool = False
    atoms_reconstructed: int = 0
    hydrogens_added: int = 0
    hydrogens_skipped: int = 0
    n_unassigned_atoms: int = 0
    skipped_no_protein: bool = False
    initial_energy: float = 0.0
    final_energy: float = 0.0
    energy_components: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)


def build_prepared_structure_records(
    structures: Sequence,
    prep_reports: Sequence[PrepReport],
    *,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    prep_run_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
) -> List[PreparedStructureRecord]:
    """Build manifest rows from prepared structures and their prep reports."""
    n = len(structures)
    if len(prep_reports) != n:
        raise ValueError(f"expected {n} prep reports, got {len(prep_reports)}")
    record_ids = _expand_optional(record_ids, n)
    source_ids = _expand_optional(source_ids, n)
    prep_run_ids = _expand_optional(prep_run_ids, n)

    rows: List[PreparedStructureRecord] = []
    for i, (structure, prep) in enumerate(zip(structures, prep_reports)):
        rows.append(
            PreparedStructureRecord(
                record_id=record_ids[i] or _default_record_id(structure),
                source_id=source_ids[i],
                prep_run_id=prep_run_ids[i],
                chain_count=getattr(structure, "chain_count", None),
                residue_count=getattr(structure, "residue_count", None),
                atom_count=getattr(structure, "atom_count", None),
                sequence_length=_sequence_length(structure),
                code_rev=code_rev,
                config_rev=config_rev,
                prep_success=not prep.skipped_no_protein,
                minimizer_steps=prep.minimizer_steps,
                converged=prep.converged,
                atoms_reconstructed=prep.atoms_reconstructed,
                hydrogens_added=prep.hydrogens_added,
                hydrogens_skipped=prep.hydrogens_skipped,
                n_unassigned_atoms=prep.n_unassigned_atoms,
                skipped_no_protein=prep.skipped_no_protein,
                initial_energy=prep.initial_energy,
                final_energy=prep.final_energy,
                energy_components=dict(prep.components),
                warnings=list(prep.warnings),
                provenance=dict(provenance or {}),
            )
        )
    return rows


def write_prepared_structure_manifest(
    records: Iterable[PreparedStructureRecord],
    path: str | Path,
) -> Path:
    """Write prepared-structure manifest rows as JSONL."""
    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(asdict(row), separators=(",", ":")))
            handle.write("\n")
    return out


def load_prepared_structure_manifest(path: str | Path) -> List[PreparedStructureRecord]:
    """Load prepared-structure manifest rows from JSONL."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [PreparedStructureRecord(**row) for row in rows]


def _sequence_length(structure) -> Optional[int]:
    """Amino-acid residue count across all chains.

    Deliberately does NOT use `structure.residue_count` — that field is
    documented (io.py docstring) to include waters and ligands, so
    using it would overreport sequence length for any structure with
    HETATM content and skew release-summary statistics. Walks chains
    explicitly and filters on `residue.is_amino_acid`, which matches
    what the structure-supervision and sequence-example builders
    consume downstream.
    """
    chains = getattr(structure, "chains", None)
    if chains is None:
        return None
    return sum(
        1
        for chain in chains
        for residue in getattr(chain, "residues", [])
        if getattr(residue, "is_amino_acid", False)
    )


def _default_record_id(structure) -> str:
    return str(getattr(structure, "identifier", None) or "structure")


def _expand_optional(values: Optional[Sequence[Optional[str]]], n: int) -> List[Optional[str]]:
    if values is None:
        return [None] * n
    if len(values) != n:
        raise ValueError(f"expected {n} items, got {len(values)}")
    return list(values)
