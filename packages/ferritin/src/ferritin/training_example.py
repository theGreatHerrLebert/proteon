"""Framework-neutral joined training examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ._artifact_checksum import sha256_file, verify_sha256
from .sequence_example import SequenceExample
from .sequence_export import load_sequence_examples
from .supervision import StructureSupervisionExample
from .supervision_export import load_structure_supervision_examples

TRAINING_EXPORT_FORMAT = "ferritin.training_example.v0"


@dataclass
class TrainingExample:
    """Thin join artifact over sequence and structure examples."""

    record_id: str
    source_id: Optional[str]
    chain_id: str
    split: str
    crop_start: Optional[int] = None
    crop_stop: Optional[int] = None
    weight: float = 1.0
    sequence: SequenceExample | None = None
    structure: StructureSupervisionExample | None = None


@dataclass
class TrainingReleaseManifest:
    """Shared manifest linking sequence and structure releases.

    `tensor_file` / `tensor_sha256` point to the denormalized per-
    example NPZ (sequence + structure tensors stacked padded to the
    longest example). Older releases written without tensors leave
    these None — consumers can detect that and fall back to loading
    the source sequence / structure releases.
    """

    release_id: str
    artifact_type: str = "release_manifest"
    format: str = TRAINING_EXPORT_FORMAT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_rev: Optional[str] = None
    config_rev: Optional[str] = None
    sequence_release: str = ""
    structure_release: str = ""
    count_examples: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    examples_file: str = "training_examples.jsonl"
    tensor_file: Optional[str] = None
    tensor_sha256: Optional[str] = None
    tensor_fields: List[str] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)


def join_training_examples(
    sequence_examples: Sequence[SequenceExample],
    structure_examples: Sequence[StructureSupervisionExample],
    *,
    split_assignments: Optional[Dict[str, str]] = None,
    crop_metadata: Optional[Dict[str, tuple[int, int]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[TrainingExample]:
    """Join sequence and structure artifacts by `record_id`."""
    seq_by_id = {ex.record_id: ex for ex in sequence_examples}
    struc_by_id = {ex.record_id: ex for ex in structure_examples}
    shared_ids = sorted(set(seq_by_id).intersection(struc_by_id))

    out: List[TrainingExample] = []
    for record_id in shared_ids:
        seq = seq_by_id[record_id]
        struc = struc_by_id[record_id]
        split = (split_assignments or {}).get(record_id, "train")
        crop = (crop_metadata or {}).get(record_id)
        out.append(
            TrainingExample(
                record_id=record_id,
                source_id=seq.source_id or struc.source_id,
                chain_id=seq.chain_id,
                split=split,
                crop_start=None if crop is None else int(crop[0]),
                crop_stop=None if crop is None else int(crop[1]),
                weight=float((weights or {}).get(record_id, 1.0)),
                sequence=seq,
                structure=struc,
            )
        )
    return out


def build_training_release(
    sequence_release_dir: str | Path,
    structure_release_dir: str | Path,
    out_dir: str | Path,
    *,
    release_id: str,
    split_assignments: Optional[Dict[str, str]] = None,
    crop_metadata: Optional[Dict[str, tuple[int, int]]] = None,
    weights: Optional[Dict[str, float]] = None,
    code_rev: Optional[str] = None,
    config_rev: Optional[str] = None,
    provenance: Optional[Dict[str, object]] = None,
    export_tensors: bool = True,
    overwrite: bool = False,
) -> Path:
    """Build a training release by joining sequence and structure releases.

    With `export_tensors=True` (default), writes a denormalized
    `tensors.npz` under the release root so downstream training code
    can load all joined tensors with one mmap instead of walking two
    sub-releases. The NPZ is SHA-256'd into the manifest's
    `tensor_sha256` field for release-validation. Set to False to
    keep the release pointer-only (matches the pre-v0.1 behavior).
    """
    sequence_examples = load_sequence_examples(Path(sequence_release_dir) / "examples")
    structure_examples = load_structure_supervision_examples(Path(structure_release_dir) / "examples")
    training_examples = join_training_examples(
        sequence_examples,
        structure_examples,
        split_assignments=split_assignments,
        crop_metadata=crop_metadata,
        weights=weights,
    )

    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    split_counts: Dict[str, int] = {}
    for ex in training_examples:
        rows.append(
            {
                "record_id": ex.record_id,
                "source_id": ex.source_id,
                "chain_id": ex.chain_id,
                "split": ex.split,
                "crop_start": ex.crop_start,
                "crop_stop": ex.crop_stop,
                "weight": ex.weight,
            }
        )
        split_counts[ex.split] = split_counts.get(ex.split, 0) + 1

    with (root / "training_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")

    tensor_file: Optional[str] = None
    tensor_sha256: Optional[str] = None
    tensor_fields: List[str] = []
    if export_tensors and training_examples:
        tensor_path = root / "tensors.npz"
        payload = _stack_training_payload(training_examples)
        # Write tensors first so the manifest can carry a digest that
        # actually corresponds to the on-disk payload (same ordering
        # as sequence_export / supervision_export).
        np.savez_compressed(tensor_path, **payload)
        tensor_file = "tensors.npz"
        tensor_sha256 = sha256_file(tensor_path)
        tensor_fields = sorted(payload.keys())

    manifest = TrainingReleaseManifest(
        release_id=release_id,
        code_rev=code_rev,
        config_rev=config_rev,
        sequence_release=str(Path(sequence_release_dir)),
        structure_release=str(Path(structure_release_dir)),
        count_examples=len(training_examples),
        split_counts=split_counts,
        tensor_file=tensor_file,
        tensor_sha256=tensor_sha256,
        tensor_fields=tensor_fields,
        provenance=dict(provenance or {}),
    )
    (root / "release_manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return root


def load_training_examples(
    release_dir: str | Path,
    *,
    verify_checksum: bool = True,
) -> List[TrainingExample]:
    """Load a training release back into `TrainingExample` objects.

    Reads the JSONL row ordering as the canonical per-example index,
    reassembles sequence + structure attributes from the denormalized
    `tensors.npz` when present. Falls back to loading via the child
    sequence / structure releases if the training release was written
    with `export_tensors=False`.

    `verify_checksum` rehashes the NPZ against `tensor_sha256` at
    load time — same semantics as the sequence / supervision loaders.
    """
    from .supervision import StructureQualityMetadata

    root = Path(release_dir)
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != TRAINING_EXPORT_FORMAT:
        raise ValueError(f"unsupported training export format: {manifest.get('format')!r}")

    rows = [
        json.loads(line)
        for line in (root / manifest["examples_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tensor_file = manifest.get("tensor_file")
    if tensor_file is None:
        # Pointer-only release — recover by loading the source releases
        # and re-joining.
        seq_examples = load_sequence_examples(Path(manifest["sequence_release"]) / "examples")
        struc_examples = load_structure_supervision_examples(
            Path(manifest["structure_release"]) / "examples"
        )
        return join_training_examples(
            seq_examples,
            struc_examples,
            split_assignments={row["record_id"]: row["split"] for row in rows},
            weights={row["record_id"]: row["weight"] for row in rows},
            crop_metadata={
                row["record_id"]: (row["crop_start"], row["crop_stop"])
                for row in rows
                if row.get("crop_start") is not None
            },
        )

    if verify_checksum:
        expected = manifest.get("tensor_sha256")
        if expected:
            verify_sha256(root / tensor_file, expected)
    payload = np.load(root / tensor_file, allow_pickle=False)
    return _unstack_training_payload(rows, payload)


def _stack_training_payload(
    examples: Sequence[TrainingExample],
) -> Dict[str, np.ndarray]:
    """Denormalize joined sequence + structure tensors into one NPZ payload.

    Padding shape rules (match the child exporters):
      - Residue-length fields: pad to `max(ex.sequence.length)`.
      - MSA axis: pad to `max(ex.sequence.msa.shape[0])` when any
        example has MSA, otherwise the MSA fields are omitted.
      - Template axis: ditto.

    Per-example `length` / `msa_rows` / `template_count` arrays let
    downstream code slice back to valid regions without guessing.
    """
    n = len(examples)
    max_len = max(ex.sequence.length for ex in examples)
    max_msa = max(
        0 if ex.sequence.msa is None else int(ex.sequence.msa.shape[0])
        for ex in examples
    )
    max_templates = max(
        0 if ex.sequence.template_mask is None
        else int(ex.sequence.template_mask.shape[0])
        for ex in examples
    )

    payload: Dict[str, np.ndarray] = {
        # Per-example bookkeeping — callers slice tensors to these.
        "length": np.asarray([ex.sequence.length for ex in examples], dtype=np.int32),
        "msa_rows": np.asarray(
            [
                0 if ex.sequence.msa is None else int(ex.sequence.msa.shape[0])
                for ex in examples
            ],
            dtype=np.int32,
        ),
        "template_count": np.asarray(
            [
                0 if ex.sequence.template_mask is None
                else int(ex.sequence.template_mask.shape[0])
                for ex in examples
            ],
            dtype=np.int32,
        ),
        "weight": np.asarray([ex.weight for ex in examples], dtype=np.float32),
        # Sequence-side tensors — padded residue axis.
        "aatype": np.zeros((n, max_len), dtype=np.int32),
        "residue_index": np.zeros((n, max_len), dtype=np.int32),
        "seq_mask": np.zeros((n, max_len), dtype=np.float32),
        # Structure-side supervision — shared residue axis.
        "all_atom_positions": np.zeros((n, max_len, 37, 3), dtype=np.float32),
        "all_atom_mask": np.zeros((n, max_len, 37), dtype=np.float32),
        "atom37_atom_exists": np.zeros((n, max_len, 37), dtype=np.float32),
        "atom14_gt_positions": np.zeros((n, max_len, 14, 3), dtype=np.float32),
        "atom14_gt_exists": np.zeros((n, max_len, 14), dtype=np.float32),
        "atom14_atom_exists": np.zeros((n, max_len, 14), dtype=np.float32),
        "atom14_atom_is_ambiguous": np.zeros((n, max_len, 14), dtype=np.float32),
        "pseudo_beta": np.zeros((n, max_len, 3), dtype=np.float32),
        "pseudo_beta_mask": np.zeros((n, max_len), dtype=np.float32),
        "phi": np.zeros((n, max_len), dtype=np.float32),
        "psi": np.zeros((n, max_len), dtype=np.float32),
        "omega": np.zeros((n, max_len), dtype=np.float32),
        "phi_mask": np.zeros((n, max_len), dtype=np.float32),
        "psi_mask": np.zeros((n, max_len), dtype=np.float32),
        "omega_mask": np.zeros((n, max_len), dtype=np.float32),
        "chi_angles": np.zeros((n, max_len, 4), dtype=np.float32),
        "chi_mask": np.zeros((n, max_len, 4), dtype=np.float32),
        "rigidgroups_gt_frames": np.zeros((n, max_len, 8, 4, 4), dtype=np.float32),
        "rigidgroups_gt_exists": np.zeros((n, max_len, 8), dtype=np.float32),
        "rigidgroups_group_exists": np.zeros((n, max_len, 8), dtype=np.float32),
        "rigidgroups_group_is_ambiguous": np.zeros((n, max_len, 8), dtype=np.float32),
    }
    if max_msa > 0:
        payload["msa"] = np.zeros((n, max_msa, max_len), dtype=np.int32)
        payload["deletion_matrix"] = np.zeros((n, max_msa, max_len), dtype=np.float32)
        payload["msa_mask"] = np.zeros((n, max_msa, max_len), dtype=np.float32)
    if max_templates > 0:
        payload["template_mask"] = np.zeros((n, max_templates), dtype=np.float32)

    for i, ex in enumerate(examples):
        seq = ex.sequence
        struc = ex.structure
        L = seq.length

        payload["aatype"][i, :L] = seq.aatype
        payload["residue_index"][i, :L] = seq.residue_index
        payload["seq_mask"][i, :L] = seq.seq_mask

        payload["all_atom_positions"][i, :L] = struc.all_atom_positions
        payload["all_atom_mask"][i, :L] = struc.all_atom_mask
        payload["atom37_atom_exists"][i, :L] = struc.atom37_atom_exists
        payload["atom14_gt_positions"][i, :L] = struc.atom14_gt_positions
        payload["atom14_gt_exists"][i, :L] = struc.atom14_gt_exists
        payload["atom14_atom_exists"][i, :L] = struc.atom14_atom_exists
        payload["atom14_atom_is_ambiguous"][i, :L] = struc.atom14_atom_is_ambiguous
        payload["pseudo_beta"][i, :L] = struc.pseudo_beta
        payload["pseudo_beta_mask"][i, :L] = struc.pseudo_beta_mask
        payload["phi"][i, :L] = struc.phi
        payload["psi"][i, :L] = struc.psi
        payload["omega"][i, :L] = struc.omega
        payload["phi_mask"][i, :L] = struc.phi_mask
        payload["psi_mask"][i, :L] = struc.psi_mask
        payload["omega_mask"][i, :L] = struc.omega_mask
        payload["chi_angles"][i, :L] = struc.chi_angles
        payload["chi_mask"][i, :L] = struc.chi_mask
        payload["rigidgroups_gt_frames"][i, :L] = struc.rigidgroups_gt_frames
        payload["rigidgroups_gt_exists"][i, :L] = struc.rigidgroups_gt_exists
        payload["rigidgroups_group_exists"][i, :L] = struc.rigidgroups_group_exists
        payload["rigidgroups_group_is_ambiguous"][i, :L] = struc.rigidgroups_group_is_ambiguous

        if seq.msa is not None and max_msa > 0:
            rows_msa = seq.msa.shape[0]
            payload["msa"][i, :rows_msa, :L] = seq.msa
            if seq.deletion_matrix is not None:
                payload["deletion_matrix"][i, :rows_msa, :L] = seq.deletion_matrix
            if seq.msa_mask is not None:
                payload["msa_mask"][i, :rows_msa, :L] = seq.msa_mask
        if seq.template_mask is not None and max_templates > 0:
            payload["template_mask"][i, : seq.template_mask.shape[0]] = seq.template_mask
    return payload


def _unstack_training_payload(
    rows: Sequence[dict],
    payload,
) -> List[TrainingExample]:
    """Slice the padded NPZ back to per-example `TrainingExample` objects.

    Inverse of `_stack_training_payload` — the `length` / `msa_rows` /
    `template_count` per-example arrays determine valid-region slicing,
    so downstream code never sees padded zeros as real data.
    """
    from .supervision import StructureQualityMetadata

    lengths = np.asarray(payload["length"], dtype=np.int32)
    msa_rows = np.asarray(payload["msa_rows"], dtype=np.int32)
    template_counts = np.asarray(payload["template_count"], dtype=np.int32)
    weights = np.asarray(payload["weight"], dtype=np.float32)

    out: List[TrainingExample] = []
    for i, row in enumerate(rows):
        L = int(lengths[i])
        n_msa = int(msa_rows[i])
        n_templates = int(template_counts[i])

        seq = SequenceExample(
            record_id=row["record_id"],
            source_id=row.get("source_id"),
            chain_id=row["chain_id"],
            sequence="",  # not stored in NPZ — reconstructed only by source-release path
            length=L,
            code_rev=None,
            config_rev=None,
            aatype=np.asarray(payload["aatype"])[i, :L].astype(np.int32, copy=False),
            residue_index=np.asarray(payload["residue_index"])[i, :L].astype(np.int32, copy=False),
            seq_mask=np.asarray(payload["seq_mask"])[i, :L].astype(np.float32, copy=False),
            msa=None if n_msa == 0 else np.asarray(payload["msa"])[i, :n_msa, :L].astype(np.int32, copy=False),
            deletion_matrix=None if n_msa == 0 else np.asarray(payload["deletion_matrix"])[i, :n_msa, :L].astype(np.float32, copy=False),
            msa_mask=None if n_msa == 0 else np.asarray(payload["msa_mask"])[i, :n_msa, :L].astype(np.float32, copy=False),
            template_mask=(
                None if n_templates == 0
                else np.asarray(payload["template_mask"])[i, :n_templates].astype(np.float32, copy=False)
            ),
        )
        struc = StructureSupervisionExample(
            record_id=row["record_id"],
            source_id=row.get("source_id"),
            prep_run_id=None,
            chain_id=row["chain_id"],
            sequence="",
            length=L,
            code_rev=None,
            config_rev=None,
            aatype=np.asarray(payload["aatype"])[i, :L].astype(np.int32, copy=False),
            residue_index=np.asarray(payload["residue_index"])[i, :L].astype(np.int32, copy=False),
            seq_mask=np.asarray(payload["seq_mask"])[i, :L].astype(np.float32, copy=False),
            all_atom_positions=np.asarray(payload["all_atom_positions"])[i, :L].astype(np.float32, copy=False),
            all_atom_mask=np.asarray(payload["all_atom_mask"])[i, :L].astype(np.float32, copy=False),
            atom37_atom_exists=np.asarray(payload["atom37_atom_exists"])[i, :L].astype(np.float32, copy=False),
            atom14_gt_positions=np.asarray(payload["atom14_gt_positions"])[i, :L].astype(np.float32, copy=False),
            atom14_gt_exists=np.asarray(payload["atom14_gt_exists"])[i, :L].astype(np.float32, copy=False),
            atom14_atom_exists=np.asarray(payload["atom14_atom_exists"])[i, :L].astype(np.float32, copy=False),
            residx_atom14_to_atom37=np.zeros((L, 14), dtype=np.int32),
            residx_atom37_to_atom14=np.zeros((L, 37), dtype=np.int32),
            atom14_atom_is_ambiguous=np.asarray(payload["atom14_atom_is_ambiguous"])[i, :L].astype(np.float32, copy=False),
            pseudo_beta=np.asarray(payload["pseudo_beta"])[i, :L].astype(np.float32, copy=False),
            pseudo_beta_mask=np.asarray(payload["pseudo_beta_mask"])[i, :L].astype(np.float32, copy=False),
            phi=np.asarray(payload["phi"])[i, :L].astype(np.float32, copy=False),
            psi=np.asarray(payload["psi"])[i, :L].astype(np.float32, copy=False),
            omega=np.asarray(payload["omega"])[i, :L].astype(np.float32, copy=False),
            phi_mask=np.asarray(payload["phi_mask"])[i, :L].astype(np.float32, copy=False),
            psi_mask=np.asarray(payload["psi_mask"])[i, :L].astype(np.float32, copy=False),
            omega_mask=np.asarray(payload["omega_mask"])[i, :L].astype(np.float32, copy=False),
            chi_angles=np.asarray(payload["chi_angles"])[i, :L].astype(np.float32, copy=False),
            chi_mask=np.asarray(payload["chi_mask"])[i, :L].astype(np.float32, copy=False),
            rigidgroups_gt_frames=np.asarray(payload["rigidgroups_gt_frames"])[i, :L].astype(np.float32, copy=False),
            rigidgroups_gt_exists=np.asarray(payload["rigidgroups_gt_exists"])[i, :L].astype(np.float32, copy=False),
            rigidgroups_group_exists=np.asarray(payload["rigidgroups_group_exists"])[i, :L].astype(np.float32, copy=False),
            rigidgroups_group_is_ambiguous=np.asarray(payload["rigidgroups_group_is_ambiguous"])[i, :L].astype(np.float32, copy=False),
            quality=None,
        )
        out.append(
            TrainingExample(
                record_id=row["record_id"],
                source_id=row.get("source_id"),
                chain_id=row["chain_id"],
                split=row["split"],
                crop_start=row.get("crop_start"),
                crop_stop=row.get("crop_stop"),
                weight=float(weights[i]),
                sequence=seq,
                structure=struc,
            )
        )
    return out
