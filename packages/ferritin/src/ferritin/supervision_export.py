"""Export and import for structure supervision examples.

Format v0:
- `manifest.json`: dataset-level metadata and tensor inventory
- `examples.jsonl`: per-example metadata and quality records
- `tensors.npz`: padded batch-major NumPy arrays

This keeps the public artifact primitive and framework-neutral while remaining
efficient for downstream NumPy/JAX/PyTorch adapters.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ._artifact_checksum import sha256_file, verify_sha256

from .supervision import StructureQualityMetadata, StructureSupervisionExample

SUPERVISION_EXPORT_FORMAT = "ferritin.structure_supervision.v0"

_TENSOR_FIELDS = (
    "aatype",
    "residue_index",
    "seq_mask",
    "all_atom_positions",
    "all_atom_mask",
    "atom37_atom_exists",
    "atom14_gt_positions",
    "atom14_gt_exists",
    "atom14_atom_exists",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "atom14_atom_is_ambiguous",
    "pseudo_beta",
    "pseudo_beta_mask",
    "phi",
    "psi",
    "omega",
    "phi_mask",
    "psi_mask",
    "omega_mask",
    "chi_angles",
    "chi_mask",
    "rigidgroups_gt_frames",
    "rigidgroups_gt_exists",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
)


def export_structure_supervision_examples(
    examples: Iterable[StructureSupervisionExample],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a batch of supervision examples as JSONL metadata + NPZ tensors."""
    examples = list(examples)
    if not examples:
        raise ValueError("expected at least one structure supervision example")

    out_path = Path(out_dir)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    out_path.mkdir(parents=True, exist_ok=True)

    tensor_payload = _stack_tensor_payload(examples)

    with (out_path / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(_example_metadata(ex), separators=(",", ":")))
            handle.write("\n")

    # Write tensors.npz first so we can hash it into the manifest.
    # Keeps integrity check ordered: if the write fails we don't leave
    # a manifest claiming a checksum that no file on disk matches.
    tensor_path = out_path / "tensors.npz"
    np.savez_compressed(tensor_path, **tensor_payload)

    manifest = {
        "format": SUPERVISION_EXPORT_FORMAT,
        "count": len(examples),
        "tensor_file": "tensors.npz",
        "examples_file": "examples.jsonl",
        "tensor_fields": list(tensor_payload.keys()),
        "tensor_sha256": sha256_file(tensor_path),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def load_structure_supervision_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> List[StructureSupervisionExample]:
    """Load supervision examples previously written by `export_structure_supervision_examples()`.

    `verify_checksum` defaults to `True`: if the manifest carries a
    `tensor_sha256`, the tensor file is rehashed and compared. Pass
    `False` to skip — useful when pointing at very large releases
    during iteration. The hash is still written on export so downstream
    release-validation passes can check it on their own cadence.
    """
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SUPERVISION_EXPORT_FORMAT:
        raise ValueError(f"unsupported supervision export format: {manifest.get('format')!r}")
    if verify_checksum:
        expected = manifest.get("tensor_sha256")
        if expected:
            verify_sha256(root / manifest["tensor_file"], expected)

    rows = [
        json.loads(line)
        for line in (root / manifest["examples_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = np.load(root / manifest["tensor_file"], allow_pickle=False)

    out: List[StructureSupervisionExample] = []
    for i, row in enumerate(rows):
        length = int(row["length"])
        quality = row.get("quality")
        out.append(
            StructureSupervisionExample(
                record_id=row["record_id"],
                source_id=row.get("source_id"),
                prep_run_id=row.get("prep_run_id"),
                chain_id=row["chain_id"],
                sequence=row["sequence"],
                length=length,
                code_rev=row.get("code_rev"),
                config_rev=row.get("config_rev"),
                aatype=np.asarray(payload["aatype"])[i, :length].astype(np.int32, copy=False),
                residue_index=np.asarray(payload["residue_index"])[i, :length].astype(np.int32, copy=False),
                seq_mask=np.asarray(payload["seq_mask"])[i, :length].astype(np.float32, copy=False),
                all_atom_positions=np.asarray(payload["all_atom_positions"])[i, :length].astype(np.float32, copy=False),
                all_atom_mask=np.asarray(payload["all_atom_mask"])[i, :length].astype(np.float32, copy=False),
                atom37_atom_exists=np.asarray(payload["atom37_atom_exists"])[i, :length].astype(np.float32, copy=False),
                atom14_gt_positions=np.asarray(payload["atom14_gt_positions"])[i, :length].astype(np.float32, copy=False),
                atom14_gt_exists=np.asarray(payload["atom14_gt_exists"])[i, :length].astype(np.float32, copy=False),
                atom14_atom_exists=np.asarray(payload["atom14_atom_exists"])[i, :length].astype(np.float32, copy=False),
                residx_atom14_to_atom37=np.asarray(payload["residx_atom14_to_atom37"])[i, :length].astype(np.int32, copy=False),
                residx_atom37_to_atom14=np.asarray(payload["residx_atom37_to_atom14"])[i, :length].astype(np.int32, copy=False),
                atom14_atom_is_ambiguous=np.asarray(payload["atom14_atom_is_ambiguous"])[i, :length].astype(np.float32, copy=False),
                pseudo_beta=np.asarray(payload["pseudo_beta"])[i, :length].astype(np.float32, copy=False),
                pseudo_beta_mask=np.asarray(payload["pseudo_beta_mask"])[i, :length].astype(np.float32, copy=False),
                phi=np.asarray(payload["phi"])[i, :length].astype(np.float32, copy=False),
                psi=np.asarray(payload["psi"])[i, :length].astype(np.float32, copy=False),
                omega=np.asarray(payload["omega"])[i, :length].astype(np.float32, copy=False),
                phi_mask=np.asarray(payload["phi_mask"])[i, :length].astype(np.float32, copy=False),
                psi_mask=np.asarray(payload["psi_mask"])[i, :length].astype(np.float32, copy=False),
                omega_mask=np.asarray(payload["omega_mask"])[i, :length].astype(np.float32, copy=False),
                chi_angles=np.asarray(payload["chi_angles"])[i, :length].astype(np.float32, copy=False),
                chi_mask=np.asarray(payload["chi_mask"])[i, :length].astype(np.float32, copy=False),
                rigidgroups_gt_frames=np.asarray(payload["rigidgroups_gt_frames"])[i, :length].astype(np.float32, copy=False),
                rigidgroups_gt_exists=np.asarray(payload["rigidgroups_gt_exists"])[i, :length].astype(np.float32, copy=False),
                rigidgroups_group_exists=np.asarray(payload["rigidgroups_group_exists"])[i, :length].astype(np.float32, copy=False),
                rigidgroups_group_is_ambiguous=np.asarray(payload["rigidgroups_group_is_ambiguous"])[i, :length].astype(np.float32, copy=False),
                quality=StructureQualityMetadata(**quality) if quality is not None else None,
            )
        )
    return out


def _example_metadata(example: StructureSupervisionExample) -> Dict[str, object]:
    return {
        "record_id": example.record_id,
        "source_id": example.source_id,
        "prep_run_id": example.prep_run_id,
        "chain_id": example.chain_id,
        "sequence": example.sequence,
        "length": example.length,
        "code_rev": example.code_rev,
        "config_rev": example.config_rev,
        "quality": asdict(example.quality) if example.quality is not None else None,
    }


def _stack_tensor_payload(examples: List[StructureSupervisionExample]) -> Dict[str, np.ndarray]:
    lengths = [ex.length for ex in examples]
    n_max = max(lengths)
    payload: Dict[str, np.ndarray] = {}
    for field in _TENSOR_FIELDS:
        first = getattr(examples[0], field)
        if first is None:
            raise ValueError(f"tensor field {field} is None; export requires materialized tensors")
        sample = np.asarray(first)
        tail_shape = sample.shape[1:]
        batch_shape = (len(examples), n_max) + tail_shape
        batch = np.zeros(batch_shape, dtype=sample.dtype)
        for i, ex in enumerate(examples):
            arr = np.asarray(getattr(ex, field))
            if arr.shape[1:] != tail_shape:
                raise ValueError(f"inconsistent tensor shape for field {field}: {arr.shape} vs {sample.shape}")
            batch[i, : ex.length] = arr
        payload[field] = batch
    payload["length"] = np.asarray(lengths, dtype=np.int32)
    return payload
