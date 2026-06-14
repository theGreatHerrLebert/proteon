"""Template feature artifact (`templates.parquet`) + template crop.

Doubly-ragged (`N_templates`, `L`) round-trip, `None` vs `N=0`, independently-null
derived geometry, backward-compat missing columns, schema-version, malformed-shape
rejection, checksum, and the template-axis crop with boundary torsion correction.
Pure NumPy + proteon (+ pyarrow); no torch/openfold, so it runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from proteon.templates import TemplateFeatures
from proteon.template_export import (
    TEMPLATE_PARQUET_SCHEMA_VERSION,
    iter_template_artifact,
    load_template_artifact,
    validate_template_features,
    write_template_artifact,
)
from proteon.supervision_crop import crop_template_features


def _features(n: int, L: int, *, derived: bool = True, seed: int = 0) -> TemplateFeatures:
    rng = np.random.default_rng(seed)
    kw = dict(
        template_aatype=rng.integers(0, 22, (n, L)).astype(np.int32),
        template_all_atom_positions=rng.standard_normal((n, L, 37, 3)).astype(np.float32),
        template_all_atom_masks=rng.random((n, L, 37)).astype(np.float32),
        template_sum_probs=rng.random(n).astype(np.float32),
        n_templates=n,
        query_len=L,
    )
    if derived:
        kw.update(
            template_pseudo_beta=rng.standard_normal((n, L, 3)).astype(np.float32),
            template_pseudo_beta_mask=rng.random((n, L)).astype(np.float32),
            template_torsion_angles_sin_cos=rng.standard_normal((n, L, 7, 2)).astype(np.float32),
            template_alt_torsion_angles_sin_cos=rng.standard_normal((n, L, 7, 2)).astype(np.float32),
            template_torsion_angles_mask=rng.random((n, L, 7)).astype(np.float32),
        )
    return TemplateFeatures(**kw)


def _assert_equal(a: TemplateFeatures, b: TemplateFeatures) -> None:
    assert a.n_templates == b.n_templates and a.query_len == b.query_len
    for attr in (
        "template_aatype",
        "template_all_atom_positions",
        "template_all_atom_masks",
        "template_sum_probs",
        "template_pseudo_beta",
        "template_pseudo_beta_mask",
        "template_torsion_angles_sin_cos",
        "template_alt_torsion_angles_sin_cos",
        "template_torsion_angles_mask",
    ):
        av, bv = getattr(a, attr), getattr(b, attr)
        if av is None:
            assert bv is None, f"{attr}: expected None, got array"
        else:
            np.testing.assert_array_equal(np.asarray(av), np.asarray(bv), err_msg=attr)


def test_roundtrip_structure_and_sequence_path(tmp_path: Path):
    items = [
        ("a", _features(2, 5, derived=True, seed=1)),   # structure path
        ("b", _features(3, 8, derived=False, seed=2)),  # sequence path (derived None)
    ]
    write_template_artifact(items, tmp_path / "t", row_group_size=1)
    loaded = load_template_artifact(tmp_path / "t")
    assert set(loaded) == {"a", "b"}
    _assert_equal(loaded["a"], items[0][1])
    _assert_equal(loaded["b"], items[1][1])
    # Sequence path's derived geometry stays None through the round trip.
    assert loaded["b"].template_pseudo_beta is None
    assert loaded["b"].template_torsion_angles_sin_cos is None


def test_none_and_zero_templates_are_distinct(tmp_path: Path):
    items = [
        ("none", None),                              # retrieval not run
        ("zero", _features(0, 6, derived=False)),    # ran, no usable hits
        ("some", _features(1, 4, derived=True, seed=3)),
    ]
    write_template_artifact(items, tmp_path / "t")
    loaded = load_template_artifact(tmp_path / "t")
    assert loaded["none"] is None
    z = loaded["zero"]
    assert z is not None and z.n_templates == 0 and z.query_len == 6
    assert z.template_aatype.shape == (0, 6)
    assert z.template_all_atom_positions.shape == (0, 6, 37, 3)
    assert z.template_sum_probs.shape == (0,)
    _assert_equal(loaded["some"], items[2][1])


def test_manifest_counts_and_checksum(tmp_path: Path):
    import json

    items = [
        ("a", _features(2, 5, seed=1)),
        ("b", None),
        ("c", _features(0, 3)),
        ("d", _features(1, 7, seed=4)),
    ]
    out = write_template_artifact(items, tmp_path / "t")
    m = json.loads((out / "manifest.json").read_text())
    assert m["count"] == 4
    assert m["count_with_templates"] == 2 and m["count_zero_templates"] == 1 and m["count_none"] == 1
    assert m["schema_version"] == TEMPLATE_PARQUET_SCHEMA_VERSION
    assert m["tensor_sha256"]
    # A corrupted tensor fails checksum verification.
    pq = out / "templates.parquet"
    raw = bytearray(pq.read_bytes())
    raw[-64] ^= 0xFF
    pq.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        load_template_artifact(out, verify_checksum=True)


def test_missing_optional_column_reads_as_none(tmp_path: Path):
    """Backward-compat: an artifact missing a derived column (e.g. a future-added
    one dropped) loads it as None, not KeyError — the reader iterates the field
    list and tolerates absent columns."""
    import pyarrow.parquet as pq

    items = [("a", _features(2, 5, derived=True, seed=5))]
    out = write_template_artifact(items, tmp_path / "t")
    path = out / "templates.parquet"
    table = pq.read_table(path)
    table = table.drop(["template_alt_torsion_angles_sin_cos"])
    pq.write_table(table, path)
    # Re-checksum so verification doesn't trip on the deliberate rewrite.
    loaded = load_template_artifact(out, verify_checksum=False)
    assert loaded["a"].template_alt_torsion_angles_sin_cos is None
    # The other tensors still load.
    np.testing.assert_array_equal(loaded["a"].template_aatype, items[0][1].template_aatype)


def test_empty_artifact(tmp_path: Path):
    out = write_template_artifact([], tmp_path / "t")
    import json

    m = json.loads((out / "manifest.json").read_text())
    assert m["count"] == 0 and m["tensor_file"] is None
    assert not (out / "templates.parquet").exists()
    assert load_template_artifact(out) == {}
    assert list(iter_template_artifact(out)) == []


def test_validate_rejects_malformed_bundle():
    tf = _features(2, 5, derived=True, seed=6)
    validate_template_features(tf)  # well-formed: no raise
    # Mandatory tensor shape disagreeing with (n, L).
    bad = TemplateFeatures(
        template_aatype=np.zeros((2, 4), np.int32),  # L=4 != query_len 5
        template_all_atom_positions=np.zeros((2, 5, 37, 3), np.float32),
        template_all_atom_masks=np.zeros((2, 5, 37), np.float32),
        template_sum_probs=np.zeros((2,), np.float32),
        n_templates=2,
        query_len=5,
    )
    with pytest.raises(ValueError):
        validate_template_features(bad)
    # sum_probs of the wrong length.
    bad2 = TemplateFeatures(
        template_aatype=np.zeros((2, 5), np.int32),
        template_all_atom_positions=np.zeros((2, 5, 37, 3), np.float32),
        template_all_atom_masks=np.zeros((2, 5, 37), np.float32),
        template_sum_probs=np.zeros((3,), np.float32),
        n_templates=2,
        query_len=5,
    )
    with pytest.raises(ValueError):
        validate_template_features(bad2)


def test_crop_template_features_slices_query_axis_and_fixes_boundary():
    tf = _features(2, 12, derived=True, seed=7)
    # Make the boundary torsion at the future crop-start clearly "bonded".
    tf.template_torsion_angles_mask[:, 4, 0:2] = 1.0
    start, stop = 4, 9
    c = crop_template_features(tf, start, stop)
    assert c.query_len == stop - start
    # Residue (query) axis sliced on axis 1; N and sum_probs unchanged.
    assert c.template_aatype.shape == (2, stop - start)
    assert c.template_all_atom_positions.shape == (2, stop - start, 37, 3)
    np.testing.assert_array_equal(c.template_sum_probs, tf.template_sum_probs)
    # First kept residue's pre_omega/phi cleared for every template (dropped prev).
    assert np.all(c.template_torsion_angles_mask[:, 0, 0:2] == 0.0)
    # Interior is a faithful slice.
    np.testing.assert_array_equal(
        c.template_aatype[:, 1:], tf.template_aatype[:, start + 1 : stop]
    )
    # Input not mutated.
    assert np.all(tf.template_torsion_angles_mask[:, start, 0:2] == 1.0)


def test_no_manifest_published_on_failed_write(tmp_path: Path):
    """If the input stream raises mid-write, no manifest is published — a partial
    parquet must not look like a complete release (codex catch)."""
    from proteon.template_export import TemplateParquetWriter

    out = tmp_path / "t"
    with pytest.raises(RuntimeError):
        with TemplateParquetWriter(out, row_group_size=1) as w:
            w.append("a", _features(1, 3, seed=1))
            raise RuntimeError("boom mid-stream")
    # The writer was released, but no manifest → the artifact is detectably
    # incomplete rather than silently truncated.
    assert not (out / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load_template_artifact(out)


def test_failed_overwrite_leaves_no_stale_manifest(tmp_path: Path):
    """Re-writing an existing artifact that then fails mid-stream must not leave
    the *previous* run's manifest describing the now-truncated parquet (codex).
    The artifact must be detectably incomplete (no manifest), not falsely
    complete or checksum-mismatched."""
    from proteon.template_export import TemplateParquetWriter

    out = tmp_path / "t"
    # First run: a complete, valid artifact.
    write_template_artifact([("a", _features(1, 4, seed=1))], out)
    assert (out / "manifest.json").exists()

    # Second run over the same dir truncates templates.parquet, then fails.
    with pytest.raises(RuntimeError):
        with TemplateParquetWriter(out, row_group_size=1) as w:
            w.append("a", _features(2, 6, seed=2))
            raise RuntimeError("boom on overwrite")
    # No stale manifest survives → load reports the artifact as incomplete.
    assert not (out / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load_template_artifact(out, verify_checksum=False)


def test_load_rejects_foreign_format_and_future_schema(tmp_path: Path):
    import json

    out = write_template_artifact([("a", _features(1, 3, seed=1))], tmp_path / "t")
    mpath = out / "manifest.json"
    good = json.loads(mpath.read_text())

    bad_fmt = dict(good, format="something.else.v9")
    mpath.write_text(json.dumps(bad_fmt))
    with pytest.raises(ValueError, match="format"):
        load_template_artifact(out, verify_checksum=False)

    bad_ver = dict(good, schema_version=TEMPLATE_PARQUET_SCHEMA_VERSION + 1)
    mpath.write_text(json.dumps(bad_ver))
    with pytest.raises(ValueError, match="schema_version"):
        load_template_artifact(out, verify_checksum=False)


def test_load_rejects_duplicate_record_ids(tmp_path: Path):
    """Two rows with the same record_id is malformed — load must raise, not
    silently keep the last (codex catch)."""
    out = write_template_artifact(
        [("dup", _features(1, 3, seed=1)), ("dup", _features(2, 3, seed=2))],
        tmp_path / "t",
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        load_template_artifact(out, verify_checksum=False)
    # Streaming still exposes both rows (no dedup contract there).
    assert [rid for rid, _ in iter_template_artifact(out, verify_checksum=False)] == ["dup", "dup"]


def test_crop_template_from_start_keeps_boundary():
    tf = _features(1, 10, derived=True, seed=8)
    tf.template_torsion_angles_mask[:, 0, 0:2] = 1.0
    c = crop_template_features(tf, 0, 6)  # start=0 → no dropped predecessor
    np.testing.assert_array_equal(
        c.template_torsion_angles_mask[:, 0, 0:2], tf.template_torsion_angles_mask[:, 0, 0:2]
    )
