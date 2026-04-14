"""Export and import for sequence examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ._artifact_checksum import sha256_file, verify_sha256
from .sequence_example import SequenceExample

SEQUENCE_EXPORT_FORMAT = "ferritin.sequence_example.v0"


def export_sequence_examples(
    examples: Iterable[SequenceExample],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    examples = list(examples)
    if not examples:
        raise ValueError("expected at least one sequence example")

    root = Path(out_dir)
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} already exists")
    root.mkdir(parents=True, exist_ok=True)

    with (root / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(_example_metadata(ex), separators=(",", ":")))
            handle.write("\n")

    # Tensors first so the manifest can carry a digest that actually
    # corresponds to the on-disk payload (same ordering as the
    # supervision exporter).
    tensor_path = root / "tensors.npz"
    np.savez_compressed(tensor_path, **_stack_tensor_payload(examples))

    manifest = {
        "format": SEQUENCE_EXPORT_FORMAT,
        "count": len(examples),
        "examples_file": "examples.jsonl",
        "tensor_file": "tensors.npz",
        "tensor_sha256": sha256_file(tensor_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def load_sequence_examples(
    path: str | Path,
    *,
    verify_checksum: bool = True,
) -> List[SequenceExample]:
    """Load sequence examples. See `load_structure_supervision_examples`
    for the `verify_checksum` flag semantics."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SEQUENCE_EXPORT_FORMAT:
        raise ValueError(f"unsupported sequence export format: {manifest.get('format')!r}")
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
    lengths = np.asarray(payload["length"], dtype=np.int32)
    msa_rows = np.asarray(payload["msa_rows"], dtype=np.int32)
    template_counts = np.asarray(payload["template_count"], dtype=np.int32)

    out: List[SequenceExample] = []
    for i, row in enumerate(rows):
        length = int(lengths[i])
        n_msa = int(msa_rows[i])
        n_templates = int(template_counts[i])
        out.append(
            SequenceExample(
                record_id=row["record_id"],
                source_id=row.get("source_id"),
                chain_id=row["chain_id"],
                sequence=row["sequence"],
                length=length,
                code_rev=row.get("code_rev"),
                config_rev=row.get("config_rev"),
                aatype=np.asarray(payload["aatype"])[i, :length].astype(np.int32, copy=False),
                residue_index=np.asarray(payload["residue_index"])[i, :length].astype(np.int32, copy=False),
                seq_mask=np.asarray(payload["seq_mask"])[i, :length].astype(np.float32, copy=False),
                msa=None if n_msa == 0 else np.asarray(payload["msa"])[i, :n_msa, :length].astype(np.int32, copy=False),
                deletion_matrix=None if n_msa == 0 else np.asarray(payload["deletion_matrix"])[i, :n_msa, :length].astype(np.float32, copy=False),
                msa_mask=None if n_msa == 0 else np.asarray(payload["msa_mask"])[i, :n_msa, :length].astype(np.float32, copy=False),
                template_mask=None if n_templates == 0 else np.asarray(payload["template_mask"])[i, :n_templates].astype(np.float32, copy=False),
            )
        )
    return out


def _example_metadata(example: SequenceExample) -> Dict[str, object]:
    return {
        "record_id": example.record_id,
        "source_id": example.source_id,
        "chain_id": example.chain_id,
        "sequence": example.sequence,
        "length": example.length,
        "code_rev": example.code_rev,
        "config_rev": example.config_rev,
    }


def _stack_tensor_payload(examples: List[SequenceExample]) -> Dict[str, np.ndarray]:
    n = len(examples)
    n_max = max(ex.length for ex in examples)
    msa_max = max(0 if ex.msa is None else ex.msa.shape[0] for ex in examples)
    template_max = max(0 if ex.template_mask is None else ex.template_mask.shape[0] for ex in examples)

    payload: Dict[str, np.ndarray] = {
        "aatype": np.zeros((n, n_max), dtype=np.int32),
        "residue_index": np.zeros((n, n_max), dtype=np.int32),
        "seq_mask": np.zeros((n, n_max), dtype=np.float32),
        "length": np.asarray([ex.length for ex in examples], dtype=np.int32),
        "msa_rows": np.asarray([0 if ex.msa is None else ex.msa.shape[0] for ex in examples], dtype=np.int32),
        "template_count": np.asarray([0 if ex.template_mask is None else ex.template_mask.shape[0] for ex in examples], dtype=np.int32),
    }
    if msa_max > 0:
        payload["msa"] = np.zeros((n, msa_max, n_max), dtype=np.int32)
        payload["deletion_matrix"] = np.zeros((n, msa_max, n_max), dtype=np.float32)
        payload["msa_mask"] = np.zeros((n, msa_max, n_max), dtype=np.float32)
    if template_max > 0:
        payload["template_mask"] = np.zeros((n, template_max), dtype=np.float32)

    for i, ex in enumerate(examples):
        payload["aatype"][i, : ex.length] = ex.aatype
        payload["residue_index"][i, : ex.length] = ex.residue_index
        payload["seq_mask"][i, : ex.length] = ex.seq_mask
        if ex.msa is not None:
            rows = ex.msa.shape[0]
            payload["msa"][i, :rows, : ex.length] = ex.msa
            if ex.deletion_matrix is not None:
                payload["deletion_matrix"][i, :rows, : ex.length] = ex.deletion_matrix
            if ex.msa_mask is not None:
                payload["msa_mask"][i, :rows, : ex.length] = ex.msa_mask
        if ex.template_mask is not None:
            payload["template_mask"][i, : ex.template_mask.shape[0]] = ex.template_mask
    return payload
