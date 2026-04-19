"""Framework-neutral sequence/alignment-side example artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .supervision_constants import AA_TO_INDEX, residue_to_one_letter


@dataclass
class SequenceExample:
    """Framework-neutral sequence-side example.

    This stays primitive by design:
    - NumPy tensors only
    - no DL framework objects
    - optional alignment/template payloads
    """

    record_id: str
    source_id: Optional[str]
    chain_id: str
    sequence: str
    length: int
    code_rev: Optional[str]
    config_rev: Optional[str]

    aatype: NDArray[np.int32]
    residue_index: NDArray[np.int32]
    seq_mask: NDArray[np.float32]

    msa: Optional[NDArray[np.int32]] = None
    deletion_matrix: Optional[NDArray[np.float32]] = None
    msa_mask: Optional[NDArray[np.float32]] = None
    template_mask: Optional[NDArray[np.float32]] = None


def build_sequence_example(
    structure,
    *,
    record_id: Optional[str] = None,
    source_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    msa: Optional[Sequence[str]] = None,
    deletion_matrix: Optional[Sequence[Sequence[float]]] = None,
    template_mask: Optional[Sequence[float]] = None,
) -> SequenceExample:
    """Build a primitive sequence-side artifact from a protein chain."""
    chain = _select_chain(structure, chain_id)
    residues = [r for r in chain.residues if r.is_amino_acid]
    if not residues:
        raise ValueError("sequence_example currently requires a protein chain")

    sequence = "".join(residue_to_one_letter(r.name) for r in residues)
    aatype = np.asarray([AA_TO_INDEX.get(aa, AA_TO_INDEX["X"]) for aa in sequence], dtype=np.int32)
    residue_index = np.asarray([int(r.serial_number) for r in residues], dtype=np.int32)
    seq_mask = np.ones((len(residues),), dtype=np.float32)

    msa_arr = _encode_msa(msa, len(residues))
    deletion_arr = _normalize_deletion_matrix(deletion_matrix, msa_arr.shape[0] if msa_arr is not None else 0, len(residues))
    msa_mask = None if msa_arr is None else np.ones(msa_arr.shape, dtype=np.float32)
    template_mask_arr = None
    if template_mask is not None:
        template_mask_arr = np.asarray(template_mask, dtype=np.float32)

    return SequenceExample(
        record_id=record_id or _default_record_id(structure, chain.id),
        source_id=source_id,
        chain_id=chain.id,
        sequence=sequence,
        length=len(residues),
        code_rev=code_rev,
        config_rev=config_rev,
        aatype=aatype,
        residue_index=residue_index,
        seq_mask=seq_mask,
        msa=msa_arr,
        deletion_matrix=deletion_arr,
        msa_mask=msa_mask,
        template_mask=template_mask_arr,
    )


def batch_build_sequence_examples(
    structures: Sequence,
    *,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    msas: Optional[Sequence[Optional[Sequence[str]]]] = None,
    deletion_matrices: Optional[Sequence[Optional[Sequence[Sequence[float]]]]] = None,
    template_masks: Optional[Sequence[Optional[Sequence[float]]]] = None,
) -> list[SequenceExample]:
    n = len(structures)
    record_ids = _expand_optional(record_ids, n)
    source_ids = _expand_optional(source_ids, n)
    chain_ids = _expand_optional(chain_ids, n)
    msas = _expand_optional(msas, n)
    deletion_matrices = _expand_optional(deletion_matrices, n)
    template_masks = _expand_optional(template_masks, n)

    return [
        build_sequence_example(
            structure,
            record_id=record_ids[i],
            source_id=source_ids[i],
            chain_id=chain_ids[i],
            code_rev=code_rev,
            config_rev=config_rev,
            msa=msas[i],
            deletion_matrix=deletion_matrices[i],
            template_mask=template_masks[i],
        )
        for i, structure in enumerate(structures)
    ]


def _encode_msa(msa: Optional[Sequence[str]], length: int) -> Optional[NDArray[np.int32]]:
    if msa is None:
        return None
    rows = list(msa)
    if not rows:
        return np.zeros((0, length), dtype=np.int32)
    bad = [row for row in rows if len(row) != length]
    if bad:
        raise ValueError("all MSA rows must match the sequence length")
    return np.asarray(
        [[AA_TO_INDEX.get(ch.upper(), AA_TO_INDEX["X"]) for ch in row] for row in rows],
        dtype=np.int32,
    )


def _normalize_deletion_matrix(
    deletion_matrix: Optional[Sequence[Sequence[float]]],
    n_rows: int,
    length: int,
) -> Optional[NDArray[np.float32]]:
    if deletion_matrix is None:
        return None
    arr = np.asarray(deletion_matrix, dtype=np.float32)
    if arr.shape != (n_rows, length):
        raise ValueError(f"expected deletion_matrix shape {(n_rows, length)}, got {arr.shape}")
    return arr


def _select_chain(structure, chain_id: Optional[str]):
    if chain_id is None:
        if structure.chain_count != 1:
            raise ValueError("sequence_example v0 is chain-level; pass chain_id for multi-chain structures")
        return structure.chains[0]
    for chain in structure.chains:
        if chain.id == chain_id:
            return chain
    raise ValueError(f"chain_id {chain_id!r} not found in structure")


def _default_record_id(structure, chain_id: str) -> str:
    ident = getattr(structure, "identifier", None) or "structure"
    return f"{ident}:{chain_id}"


def _expand_optional(values: Optional[Sequence], n: int) -> list:
    if values is None:
        return [None] * n
    if len(values) != n:
        raise ValueError(f"expected {n} items, got {len(values)}")
    return list(values)
