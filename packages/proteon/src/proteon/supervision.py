"""Framework-neutral structural supervision artifacts.

This module defines Proteon's first public implementation of
`structure_supervision_example`.

Design rules:

- supervision is derived from the post-prep structure
- the public boundary is NumPy + plain Python metadata
- no concrete DL framework objects live here
- the implementation must remain compatible with future Rust-side batching

This module intentionally keeps the Python contract simple so the heavy
extraction path can move into Rust without changing downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .prepare import PrepReport
from .supervision_constants import AA_TO_INDEX, residue_to_one_letter
from .supervision_backend import (
    batch_extract_structure_supervision as _batch_extract_structure_supervision,
    extract_structure_supervision_chain as _extract_structure_supervision_chain,
    rust_supervision_available,
)
from .supervision_geometry import (
    compute_atom14_alt,
    compute_backbone_torsions,
    compute_chi_angles,
    compute_pseudo_beta,
    compute_rigidgroups,
    extract_atom14,
    extract_atom37,
)

# Optional Rust fast path lives behind `supervision_backend.py`. The current
# builder uses it when available and falls back to the Python reference path.


@dataclass
class StructureQualityMetadata:
    """Prep/QC metadata bundled with a supervision example."""

    prep_success: bool
    force_field: Optional[str] = None
    minimizer: Optional[str] = None
    minimizer_steps: Optional[int] = None
    converged: Optional[bool] = None
    atoms_reconstructed: int = 0
    hydrogens_added: int = 0
    hydrogens_skipped: int = 0
    n_unassigned_atoms: int = 0
    skipped_no_protein: bool = False
    initial_energy: Optional[float] = None
    final_energy: Optional[float] = None
    energy_components: Dict[str, float] = field(default_factory=dict)
    parse_warnings: List[str] = field(default_factory=list)
    prep_warnings: List[str] = field(default_factory=list)
    source_format: Optional[str] = None
    structure_checksum: Optional[str] = None


@dataclass
class StructureSupervisionExample:
    """Framework-neutral structural supervision example.

    Notes:
    - Public boundary is NumPy + plain Python metadata.
    - Heavy extraction should eventually be batchable on the Rust side.
    - Rigid-group tensors remain optional until implemented.
    """

    record_id: str
    source_id: Optional[str]
    prep_run_id: Optional[str]
    chain_id: str
    sequence: str
    length: int
    code_rev: Optional[str]
    config_rev: Optional[str]

    aatype: NDArray[np.int32]
    residue_index: NDArray[np.int32]
    seq_mask: NDArray[np.float32]

    all_atom_positions: Optional[NDArray[np.float32]] = None
    all_atom_mask: Optional[NDArray[np.float32]] = None
    atom37_atom_exists: Optional[NDArray[np.float32]] = None

    atom14_gt_positions: Optional[NDArray[np.float32]] = None
    atom14_gt_exists: Optional[NDArray[np.float32]] = None
    atom14_atom_exists: Optional[NDArray[np.float32]] = None
    residx_atom14_to_atom37: Optional[NDArray[np.int32]] = None
    residx_atom37_to_atom14: Optional[NDArray[np.int32]] = None
    atom14_atom_is_ambiguous: Optional[NDArray[np.float32]] = None
    atom14_alt_gt_positions: Optional[NDArray[np.float32]] = None
    atom14_alt_gt_exists: Optional[NDArray[np.float32]] = None

    pseudo_beta: Optional[NDArray[np.float32]] = None
    pseudo_beta_mask: Optional[NDArray[np.float32]] = None

    phi: Optional[NDArray[np.float32]] = None
    psi: Optional[NDArray[np.float32]] = None
    omega: Optional[NDArray[np.float32]] = None
    phi_mask: Optional[NDArray[np.float32]] = None
    psi_mask: Optional[NDArray[np.float32]] = None
    omega_mask: Optional[NDArray[np.float32]] = None

    chi_angles: Optional[NDArray[np.float32]] = None
    chi_mask: Optional[NDArray[np.float32]] = None

    rigidgroups_gt_frames: Optional[NDArray[np.float32]] = None
    rigidgroups_gt_exists: Optional[NDArray[np.float32]] = None
    rigidgroups_group_exists: Optional[NDArray[np.float32]] = None
    rigidgroups_group_is_ambiguous: Optional[NDArray[np.float32]] = None

    quality: Optional[StructureQualityMetadata] = None

    @property
    def is_partial(self) -> bool:
        """Whether optional higher-order tensor groups are still missing."""
        return any(
            x is None
            for x in (
                self.rigidgroups_gt_frames,
                self.rigidgroups_gt_exists,
                self.rigidgroups_group_exists,
                self.rigidgroups_group_is_ambiguous,
            )
        )


def build_structure_supervision_example(
    structure,
    *,
    prep_report: Optional[PrepReport] = None,
    record_id: Optional[str] = None,
    source_id: Optional[str] = None,
    prep_run_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
) -> StructureSupervisionExample:
    """Build a framework-neutral supervision example from a prepared structure.

    Current behavior:
    - validates the chain-level protein-only example boundary
    - materializes metadata, sequence, residue indices, and masks
    - extracts atom37 and atom14 tensors
    - computes pseudo-beta and torsion supervision
    - attaches prep/QC metadata
    - leaves rigid-group tensors unset for now
    """
    chain = _select_chain(structure, chain_id)
    residues = [r for r in chain.residues if r.is_amino_acid]
    if not residues:
        raise ValueError("structure_supervision_example currently requires a protein chain")

    sequence = "".join(residue_to_one_letter(r.name) for r in residues)
    aatype = np.asarray([AA_TO_INDEX.get(aa, AA_TO_INDEX["X"]) for aa in sequence], dtype=np.int32)
    residue_index = np.asarray([int(r.serial_number) for r in residues], dtype=np.int32)
    seq_mask = np.ones((len(residues),), dtype=np.float32)
    tensors = None
    if rust_supervision_available() and hasattr(structure, "get_py_ptr"):
        tensors = _extract_structure_supervision_chain(structure, chain_id=chain.id)
    if tensors is None:
        atom37 = extract_atom37(residues)
        atom14 = extract_atom14(residues, atom37)
    else:
        atom37 = {
            "positions": np.asarray(tensors["all_atom_positions"]),
            "mask": np.asarray(tensors["all_atom_mask"]),
            "exists": np.asarray(tensors["atom37_atom_exists"]),
        }
        a14_pos = np.asarray(tensors["atom14_gt_positions"])
        a14_mask = np.asarray(tensors["atom14_gt_exists"])
        alt = compute_atom14_alt(residues, a14_pos, a14_mask)
        atom14 = {
            "positions": a14_pos,
            "mask": a14_mask,
            "exists": np.asarray(tensors["atom14_atom_exists"]),
            "to_atom37": np.asarray(tensors["residx_atom14_to_atom37"]),
            "from_atom37": np.asarray(tensors["residx_atom37_to_atom14"]),
            "ambiguous": np.asarray(tensors["atom14_atom_is_ambiguous"]),
            "alt_positions": alt["alt_positions"],
            "alt_mask": alt["alt_mask"],
        }
        pseudo_beta = np.asarray(tensors["pseudo_beta"])
        pseudo_beta_mask = np.asarray(tensors["pseudo_beta_mask"])
    if tensors is None:
        pseudo_beta, pseudo_beta_mask = compute_pseudo_beta(residues, atom37)
        backbone = compute_backbone_torsions(residues)
        chi = compute_chi_angles(residues)
        rigidgroups = compute_rigidgroups(residues)
    else:
        backbone = {
            "phi": np.asarray(tensors["phi"]),
            "psi": np.asarray(tensors["psi"]),
            "omega": np.asarray(tensors["omega"]),
            "phi_mask": np.asarray(tensors["phi_mask"]),
            "psi_mask": np.asarray(tensors["psi_mask"]),
            "omega_mask": np.asarray(tensors["omega_mask"]),
        }
        chi = {
            "angles": np.asarray(tensors["chi_angles"]),
            "mask": np.asarray(tensors["chi_mask"]),
        }
        rigidgroups = {
            "frames": np.asarray(tensors["rigidgroups_gt_frames"]),
            "gt_exists": np.asarray(tensors["rigidgroups_gt_exists"]),
            "group_exists": np.asarray(tensors["rigidgroups_group_exists"]),
            "ambiguous": np.asarray(tensors["rigidgroups_group_is_ambiguous"]),
        }

    return StructureSupervisionExample(
        record_id=record_id or _default_record_id(structure, chain.id),
        source_id=source_id,
        prep_run_id=prep_run_id,
        chain_id=chain.id,
        sequence=sequence,
        length=len(residues),
        code_rev=code_rev,
        config_rev=config_rev,
        aatype=aatype,
        residue_index=residue_index,
        seq_mask=seq_mask,
        all_atom_positions=atom37["positions"],
        all_atom_mask=atom37["mask"],
        atom37_atom_exists=atom37["exists"],
        atom14_gt_positions=atom14["positions"],
        atom14_gt_exists=atom14["mask"],
        atom14_atom_exists=atom14["exists"],
        residx_atom14_to_atom37=atom14["to_atom37"],
        residx_atom37_to_atom14=atom14["from_atom37"],
        atom14_atom_is_ambiguous=atom14["ambiguous"],
        atom14_alt_gt_positions=atom14["alt_positions"],
        atom14_alt_gt_exists=atom14["alt_mask"],
        pseudo_beta=pseudo_beta,
        pseudo_beta_mask=pseudo_beta_mask,
        phi=backbone["phi"],
        psi=backbone["psi"],
        omega=backbone["omega"],
        phi_mask=backbone["phi_mask"],
        psi_mask=backbone["psi_mask"],
        omega_mask=backbone["omega_mask"],
        chi_angles=chi["angles"],
        chi_mask=chi["mask"],
        rigidgroups_gt_frames=rigidgroups["frames"],
        rigidgroups_gt_exists=rigidgroups["gt_exists"],
        rigidgroups_group_exists=rigidgroups["group_exists"],
        rigidgroups_group_is_ambiguous=rigidgroups["ambiguous"],
        quality=_quality_from_prep_report(prep_report),
    )


def _batch_alt_positions(residues, batch_tensors, i, length):
    """Compute atom14 alt positions for one example in a Rust batch."""
    a14_pos = np.asarray(batch_tensors["atom14_gt_positions"])[i, :length].astype(np.float32, copy=False)
    a14_mask = np.asarray(batch_tensors["atom14_gt_exists"])[i, :length].astype(np.float32, copy=False)
    alt = compute_atom14_alt(residues, a14_pos, a14_mask)
    return {
        "atom14_alt_gt_positions": alt["alt_positions"],
        "atom14_alt_gt_exists": alt["alt_mask"],
    }


def batch_build_structure_supervision_examples(
    structures: Sequence,
    *,
    prep_reports: Optional[Sequence[Optional[PrepReport]]] = None,
    record_ids: Optional[Sequence[Optional[str]]] = None,
    source_ids: Optional[Sequence[Optional[str]]] = None,
    prep_run_ids: Optional[Sequence[Optional[str]]] = None,
    chain_ids: Optional[Sequence[Optional[str]]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
) -> List[StructureSupervisionExample]:
    """Batch convenience wrapper over `build_structure_supervision_example()`."""
    n = len(structures)
    prep_reports = _expand_optional(prep_reports, n)
    record_ids = _expand_optional(record_ids, n)
    source_ids = _expand_optional(source_ids, n)
    prep_run_ids = _expand_optional(prep_run_ids, n)
    chain_ids = _expand_optional(chain_ids, n)

    if rust_supervision_available() and all(hasattr(s, "get_py_ptr") for s in structures):
        batch_tensors = _batch_extract_structure_supervision(structures, chain_ids=chain_ids)
        out: List[StructureSupervisionExample] = []
        for i, structure in enumerate(structures):
            chain = _select_chain(structure, chain_ids[i])
            residues = [r for r in chain.residues if r.is_amino_acid]
            if not residues:
                raise ValueError("structure_supervision_example currently requires a protein chain")
            sequence = "".join(residue_to_one_letter(r.name) for r in residues)
            length = len(residues)
            out.append(
                StructureSupervisionExample(
                    record_id=record_ids[i] or _default_record_id(structure, chain.id),
                    source_id=source_ids[i],
                    prep_run_id=prep_run_ids[i],
                    chain_id=chain.id,
                    sequence=sequence,
                    length=length,
                    code_rev=code_rev,
                    config_rev=config_rev,
                    aatype=np.asarray(batch_tensors["aatype"])[i, :length].astype(np.int32, copy=False),
                    residue_index=np.asarray(batch_tensors["residue_index"])[i, :length].astype(np.int32, copy=False),
                    seq_mask=np.asarray(batch_tensors["seq_mask"])[i, :length].astype(np.float32, copy=False),
                    all_atom_positions=np.asarray(batch_tensors["all_atom_positions"])[i, :length].astype(np.float32, copy=False),
                    all_atom_mask=np.asarray(batch_tensors["all_atom_mask"])[i, :length].astype(np.float32, copy=False),
                    atom37_atom_exists=np.asarray(batch_tensors["atom37_atom_exists"])[i, :length].astype(np.float32, copy=False),
                    atom14_gt_positions=np.asarray(batch_tensors["atom14_gt_positions"])[i, :length].astype(np.float32, copy=False),
                    atom14_gt_exists=np.asarray(batch_tensors["atom14_gt_exists"])[i, :length].astype(np.float32, copy=False),
                    atom14_atom_exists=np.asarray(batch_tensors["atom14_atom_exists"])[i, :length].astype(np.float32, copy=False),
                    residx_atom14_to_atom37=np.asarray(batch_tensors["residx_atom14_to_atom37"])[i, :length].astype(np.int32, copy=False),
                    residx_atom37_to_atom14=np.asarray(batch_tensors["residx_atom37_to_atom14"])[i, :length].astype(np.int32, copy=False),
                    atom14_atom_is_ambiguous=np.asarray(batch_tensors["atom14_atom_is_ambiguous"])[i, :length].astype(np.float32, copy=False),
                    **_batch_alt_positions(residues, batch_tensors, i, length),
                    pseudo_beta=np.asarray(batch_tensors["pseudo_beta"])[i, :length].astype(np.float32, copy=False),
                    pseudo_beta_mask=np.asarray(batch_tensors["pseudo_beta_mask"])[i, :length].astype(np.float32, copy=False),
                    phi=np.asarray(batch_tensors["phi"])[i, :length].astype(np.float32, copy=False),
                    psi=np.asarray(batch_tensors["psi"])[i, :length].astype(np.float32, copy=False),
                    omega=np.asarray(batch_tensors["omega"])[i, :length].astype(np.float32, copy=False),
                    phi_mask=np.asarray(batch_tensors["phi_mask"])[i, :length].astype(np.float32, copy=False),
                    psi_mask=np.asarray(batch_tensors["psi_mask"])[i, :length].astype(np.float32, copy=False),
                    omega_mask=np.asarray(batch_tensors["omega_mask"])[i, :length].astype(np.float32, copy=False),
                    chi_angles=np.asarray(batch_tensors["chi_angles"])[i, :length].astype(np.float32, copy=False),
                    chi_mask=np.asarray(batch_tensors["chi_mask"])[i, :length].astype(np.float32, copy=False),
                    rigidgroups_gt_frames=np.asarray(batch_tensors["rigidgroups_gt_frames"])[i, :length].astype(np.float32, copy=False),
                    rigidgroups_gt_exists=np.asarray(batch_tensors["rigidgroups_gt_exists"])[i, :length].astype(np.float32, copy=False),
                    rigidgroups_group_exists=np.asarray(batch_tensors["rigidgroups_group_exists"])[i, :length].astype(np.float32, copy=False),
                    rigidgroups_group_is_ambiguous=np.asarray(batch_tensors["rigidgroups_group_is_ambiguous"])[i, :length].astype(np.float32, copy=False),
                    quality=_quality_from_prep_report(prep_reports[i]),
                )
            )
        return out

    out: List[StructureSupervisionExample] = []
    for i, structure in enumerate(structures):
        out.append(
            build_structure_supervision_example(
                structure,
                prep_report=prep_reports[i],
                record_id=record_ids[i],
                source_id=source_ids[i],
                prep_run_id=prep_run_ids[i],
                chain_id=chain_ids[i],
                code_rev=code_rev,
                config_rev=config_rev,
            )
        )
    return out


def _quality_from_prep_report(prep_report: Optional[PrepReport]) -> Optional[StructureQualityMetadata]:
    if prep_report is None:
        return None
    return StructureQualityMetadata(
        prep_success=not prep_report.skipped_no_protein,
        minimizer_steps=prep_report.minimizer_steps,
        converged=prep_report.converged,
        atoms_reconstructed=prep_report.atoms_reconstructed,
        hydrogens_added=prep_report.hydrogens_added,
        hydrogens_skipped=prep_report.hydrogens_skipped,
        n_unassigned_atoms=prep_report.n_unassigned_atoms,
        skipped_no_protein=prep_report.skipped_no_protein,
        initial_energy=prep_report.initial_energy,
        final_energy=prep_report.final_energy,
        energy_components=dict(prep_report.components),
        prep_warnings=list(prep_report.warnings),
    )


def _select_chain(structure, chain_id: Optional[str]):
    if chain_id is None:
        if structure.chain_count != 1:
            raise ValueError(
                "structure_supervision_example v0 is chain-level; pass chain_id for multi-chain structures"
            )
        return structure.chains[0]
    for chain in structure.chains:
        if chain.id == chain_id:
            return chain
    raise ValueError(f"chain_id {chain_id!r} not found in structure")


def _default_record_id(structure, chain_id: str) -> str:
    ident = getattr(structure, "identifier", None) or "structure"
    return f"{ident}:{chain_id}"


def _expand_optional(values: Optional[Sequence], n: int) -> List[Optional[object]]:
    if values is None:
        return [None] * n
    if len(values) != n:
        raise ValueError(f"expected {n} items, got {len(values)}")
    return list(values)
