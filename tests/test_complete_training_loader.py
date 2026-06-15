"""End-to-end training loader: MSA + templates + structure labels in one call.

`iter_complete_training_examples` re-joins the child sequence release (so MSA is
populated — the inline training.parquet drops it) and left-joins a template
artifact, so a single call yields an OpenFold-complete example. Pure NumPy +
proteon (+ pyarrow); runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import proteon
from proteon.sequence_example import SequenceExample, compute_msa_deletion_features
from proteon.sequence_release import build_sequence_release
from proteon.supervision import build_structure_supervision_example
from proteon.supervision_release import build_structure_supervision_release
from proteon.templates import TemplateFeatures
from proteon.template_export import write_template_artifact
from proteon.training_example import (
    build_training_release,
    iter_complete_training_examples,
    join_training_examples,
)
from proteon.supervision_crop import crop_training_example

REPO_ROOT = Path(__file__).resolve().parents[1]


def _structure_example(pdb: str):
    path = REPO_ROOT / "test-pdbs" / pdb
    if not path.exists():
        pytest.skip(f"{path} not found")
    pairs = proteon.batch_load_tolerant([str(path)])
    st = pairs[0][1] if isinstance(pairs[0], tuple) else pairs[0]
    return build_structure_supervision_example(st, record_id=pdb, source_id=pdb)


def _sequence_example_with_msa(struc, depth: int = 4) -> SequenceExample:
    L = struc.length
    rng = np.random.default_rng(L)
    deletion = rng.integers(0, 5, size=(depth, L)).astype(np.float32)
    has_del, del_val = compute_msa_deletion_features(deletion)
    return SequenceExample(
        record_id=struc.record_id,
        source_id=struc.source_id,
        chain_id=struc.chain_id,
        sequence=struc.sequence,
        length=L,
        code_rev=None,
        config_rev=None,
        aatype=struc.aatype,
        residue_index=struc.residue_index,
        seq_mask=struc.seq_mask,
        # Same chain → same author identity as the structure side.
        author_seq_id=struc.author_seq_id,
        insertion_code=struc.insertion_code,
        msa=rng.integers(0, 22, size=(depth, L), dtype=np.int32),
        deletion_matrix=deletion,
        msa_mask=np.ones((depth, L), dtype=np.float32),
        has_deletion=has_del,
        deletion_value=del_val,
    )


def _template_for(struc, n: int = 2) -> TemplateFeatures:
    L = struc.length
    rng = np.random.default_rng(L + 1)
    return TemplateFeatures(
        template_aatype=rng.integers(0, 22, (n, L)).astype(np.int32),
        template_all_atom_positions=rng.standard_normal((n, L, 37, 3)).astype(np.float32),
        template_all_atom_masks=rng.random((n, L, 37)).astype(np.float32),
        template_sum_probs=rng.random(n).astype(np.float32),
        n_templates=n,
        query_len=L,
        template_pseudo_beta=rng.standard_normal((n, L, 3)).astype(np.float32),
        template_pseudo_beta_mask=rng.random((n, L)).astype(np.float32),
        template_torsion_angles_sin_cos=rng.standard_normal((n, L, 7, 2)).astype(np.float32),
        template_alt_torsion_angles_sin_cos=rng.standard_normal((n, L, 7, 2)).astype(np.float32),
        template_torsion_angles_mask=rng.random((n, L, 7)).astype(np.float32),
    )


def _build_release(tmp_path: Path, *, with_templates: bool):
    struc = _structure_example("1crn.pdb")
    seq = _sequence_example_with_msa(struc)

    seq_dir = build_sequence_release([seq], tmp_path / "seq", release_id="seq")
    struc_dir = build_structure_supervision_release(
        [struc], tmp_path / "struc", release_id="struc"
    )

    template_dir = None
    if with_templates:
        template_dir = tmp_path / "tmpl"
        write_template_artifact([(struc.record_id, _template_for(struc))], template_dir)

    train_dir = build_training_release(
        seq_dir,
        struc_dir,
        tmp_path / "train",
        release_id="train",
        template_release=template_dir,
    )
    return train_dir, struc


def test_complete_loader_yields_msa_and_templates(tmp_path: Path):
    train_dir, struc = _build_release(tmp_path, with_templates=True)
    examples = list(iter_complete_training_examples(train_dir))
    assert len(examples) == 1
    ex = examples[0]

    # MSA is populated (the plain training.parquet path drops it to None).
    assert ex.sequence is not None and ex.sequence.msa is not None
    assert ex.sequence.msa.shape == (4, struc.length)
    assert ex.sequence.has_deletion is not None
    # Structure labels present.
    assert ex.structure is not None
    assert ex.structure.all_atom_positions.shape == (struc.length, 37, 3)
    # Templates joined from the artifact (recorded in the manifest).
    assert ex.templates is not None and ex.templates.n_templates == 2
    assert ex.templates.query_len == struc.length
    assert ex.templates.template_all_atom_positions.shape == (2, struc.length, 37, 3)


def test_complete_loader_without_templates_is_none(tmp_path: Path):
    train_dir, _ = _build_release(tmp_path, with_templates=False)
    ex = next(iter_complete_training_examples(train_dir))
    assert ex.templates is None
    # MSA still joined even without templates.
    assert ex.sequence.msa is not None


def test_complete_example_crops_all_axes_together(tmp_path: Path):
    train_dir, struc = _build_release(tmp_path, with_templates=True)
    ex = next(iter_complete_training_examples(train_dir))
    start, stop = 5, 25
    cropped = crop_training_example(ex, start, stop)
    L = stop - start
    # Sequence residue axis + MSA axis-1, structure axis-0, template axis-1 all = L.
    assert cropped.sequence.length == L
    assert cropped.sequence.msa.shape == (4, L)
    assert cropped.structure.length == L
    assert cropped.structure.all_atom_positions.shape == (L, 37, 3)
    assert cropped.templates.query_len == L
    assert cropped.templates.template_all_atom_positions.shape == (2, L, 37, 3)
    # The applied crop is cleared so it can't be double-applied.
    assert cropped.crop_start is None and cropped.crop_stop is None


def test_split_filter_skips_template_validation_in_other_splits(tmp_path: Path):
    """A bad template bundle in a split the caller didn't request must NOT fail
    the requested split's load — split filtering happens before template
    validation (codex). Loading the bad split, or all splits, still raises."""
    crn = _structure_example("1crn.pdb")
    ubq = _structure_example("1ubq.pdb")
    seqs = [_sequence_example_with_msa(crn), _sequence_example_with_msa(ubq)]

    seq_dir = build_sequence_release(seqs, tmp_path / "seq", release_id="seq")
    struc_dir = build_structure_supervision_release([crn, ubq], tmp_path / "struc", release_id="struc")

    # crn's template matches its length; ubq's template has a deliberately wrong
    # query_len (a valid bundle, but a bad join for ubq).
    good = _template_for(crn)
    bad = TemplateFeatures(
        template_aatype=np.zeros((1, ubq.length - 5), np.int32),
        template_all_atom_positions=np.zeros((1, ubq.length - 5, 37, 3), np.float32),
        template_all_atom_masks=np.zeros((1, ubq.length - 5, 37), np.float32),
        template_sum_probs=np.zeros((1,), np.float32),
        n_templates=1,
        query_len=ubq.length - 5,
    )
    tmpl_dir = tmp_path / "tmpl"
    write_template_artifact([(crn.record_id, good), (ubq.record_id, bad)], tmpl_dir)

    train_dir = build_training_release(
        seq_dir, struc_dir, tmp_path / "train", release_id="train",
        split_assignments={crn.record_id: "train", ubq.record_id: "valid"},
        template_release=tmpl_dir,
    )
    # Requesting "train" only never touches ubq's bad template.
    train = list(iter_complete_training_examples(train_dir, split="train"))
    assert [e.record_id for e in train] == [crn.record_id]
    assert train[0].templates is not None
    # Requesting the bad split (or all) raises on the mismatch.
    with pytest.raises(ValueError, match="query_len"):
        list(iter_complete_training_examples(train_dir, split="valid"))
    with pytest.raises(ValueError, match="query_len"):
        list(iter_complete_training_examples(train_dir))


def test_template_query_len_mismatch_raises():
    struc = _structure_example("1crn.pdb")
    seq = _sequence_example_with_msa(struc)
    # A template bundle whose query_len disagrees with the example length is a bad
    # join — it must raise, not silently attach corrupt supervision.
    bad = _template_for(struc)
    bad.n_templates  # touch
    bad = TemplateFeatures(
        template_aatype=np.zeros((1, struc.length + 3), np.int32),
        template_all_atom_positions=np.zeros((1, struc.length + 3, 37, 3), np.float32),
        template_all_atom_masks=np.zeros((1, struc.length + 3, 37), np.float32),
        template_sum_probs=np.zeros((1,), np.float32),
        n_templates=1,
        query_len=struc.length + 3,
    )
    with pytest.raises(ValueError, match="query_len"):
        join_training_examples([seq], [struc], templates={struc.record_id: bad})
