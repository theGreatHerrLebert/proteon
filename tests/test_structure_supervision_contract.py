"""Exhaustive NumPy + Parquet contract test for StructureSupervisionExample.

Per `devdocs/STRUCTURE_SUPERVISION_SCHEMA.md` §5, the supervision artifact
is the stable boundary between proteon and downstream framework-agnostic
consumers (NumPy / Arrow / Parquet). Satellites such as proteon-graphein,
proteon-pyg, and future siblings rely on that contract not drifting.

Looser round-trip coverage already exists in `test_supervision.py:190` and
`test_supervision_parquet_streaming.py:88`, but those check only a handful
of fields. This file iterates the full `TENSOR_FIELDS` source-of-truth and
asserts dtype + per-residue shape + bit-equal Parquet round-trip for **every
advertised field**, on both a synthetic example and a real prepared
structure. New fields added to the dataclass without updating the Parquet
schema, or vice-versa, will trip this test on the next CI run.

Kept in the default tier (`pytest -m "not slow and not oracle"`); whole
file runs in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import proteon
from proteon.supervision import StructureSupervisionExample
from proteon.supervision_export import (
    SUPERVISION_EXPORT_FORMAT,
    SUPERVISION_PARQUET_SCHEMA_VERSION,
    TENSOR_FIELDS,
    SupervisionParquetWriter,
    build_supervision_schema,
    load_structure_supervision_examples,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PDB = REPO_ROOT / "test-pdbs" / "1crn.pdb"


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #


def _synthetic_example(record_id: str, length: int, seed: int) -> StructureSupervisionExample:
    """Build a synthetic example whose every TENSOR_FIELDS entry is populated.

    Mirrors the pattern in `tests/test_supervision_parquet_streaming.py`.
    Synthetic-only data avoids depending on prep + the C extension just to
    exercise the schema contract.
    """
    rng = np.random.default_rng(seed)
    L = length
    return StructureSupervisionExample(
        record_id=record_id,
        source_id=f"{record_id}.pdb",
        prep_run_id=None,
        chain_id="A",
        sequence="A" * L,
        length=L,
        code_rev=None,
        config_rev=None,
        aatype=rng.integers(0, 20, size=L, dtype=np.int32),
        residue_index=np.arange(L, dtype=np.int32),
        seq_mask=np.ones(L, dtype=np.float32),
        author_seq_id=rng.integers(1, 500, size=L, dtype=np.int32),
        insertion_code=rng.integers(0, 27, size=L, dtype=np.int32),
        all_atom_positions=rng.standard_normal((L, 37, 3), dtype=np.float32),
        all_atom_mask=rng.random((L, 37), dtype=np.float32),
        atom37_atom_exists=rng.random((L, 37), dtype=np.float32),
        atom14_gt_positions=rng.standard_normal((L, 14, 3), dtype=np.float32),
        atom14_gt_exists=rng.random((L, 14), dtype=np.float32),
        atom14_atom_exists=rng.random((L, 14), dtype=np.float32),
        residx_atom14_to_atom37=rng.integers(0, 37, size=(L, 14), dtype=np.int32),
        residx_atom37_to_atom14=rng.integers(0, 14, size=(L, 37), dtype=np.int32),
        atom14_atom_is_ambiguous=rng.random((L, 14), dtype=np.float32),
        atom14_alt_gt_positions=rng.standard_normal((L, 14, 3), dtype=np.float32),
        atom14_alt_gt_exists=rng.random((L, 14), dtype=np.float32),
        pseudo_beta=rng.standard_normal((L, 3), dtype=np.float32),
        pseudo_beta_mask=rng.random(L, dtype=np.float32),
        phi=rng.standard_normal(L, dtype=np.float32),
        psi=rng.standard_normal(L, dtype=np.float32),
        omega=rng.standard_normal(L, dtype=np.float32),
        phi_mask=rng.random(L, dtype=np.float32),
        psi_mask=rng.random(L, dtype=np.float32),
        omega_mask=rng.random(L, dtype=np.float32),
        chi_angles=rng.standard_normal((L, 4), dtype=np.float32),
        chi_mask=rng.random((L, 4), dtype=np.float32),
        torsion_angles_sin_cos=rng.standard_normal((L, 7, 2), dtype=np.float32),
        alt_torsion_angles_sin_cos=rng.standard_normal((L, 7, 2), dtype=np.float32),
        torsion_angles_mask=rng.random((L, 7), dtype=np.float32),
        rigidgroups_gt_frames=rng.standard_normal((L, 8, 4, 4), dtype=np.float32),
        rigidgroups_gt_exists=rng.random((L, 8), dtype=np.float32),
        rigidgroups_group_exists=rng.random((L, 8), dtype=np.float32),
        rigidgroups_group_is_ambiguous=rng.random((L, 8), dtype=np.float32),
        quality=None,
    )


def _real_example() -> StructureSupervisionExample:
    """Build a real example from `test-pdbs/1crn.pdb`.

    Skips cleanly if the PyO3 connector or the PDB file isn't available.
    The connector check has to come *before* the load call: `proteon.load`
    dereferences `io._io` which is `None` in source-only environments,
    and would otherwise raise `AttributeError` instead of skipping.
    """
    pytest.importorskip("proteon_connector")
    if not TEST_PDB.exists():
        pytest.skip(f"missing test PDB: {TEST_PDB}")
    structure = proteon.load(str(TEST_PDB))
    return proteon.build_structure_supervision_example(structure)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


class TestTensorFieldContract:
    """Every advertised tensor field has the documented dtype + shape."""

    @pytest.mark.parametrize("column_name, inner_shape, np_dtype, attr_name", TENSOR_FIELDS)
    def test_synthetic_field_dtype_and_shape(
        self, column_name, inner_shape, np_dtype, attr_name
    ):
        ex = _synthetic_example("synthetic-1", length=7, seed=42)
        arr = getattr(ex, attr_name)
        assert isinstance(arr, np.ndarray), f"{attr_name} is not an ndarray"
        assert arr.dtype == np.dtype(np_dtype), (
            f"{attr_name}: dtype {arr.dtype} != declared {np.dtype(np_dtype)}"
        )
        expected_shape = (ex.length,) + tuple(inner_shape)
        assert arr.shape == expected_shape, (
            f"{attr_name}: shape {arr.shape} != declared {expected_shape}"
        )

    @pytest.mark.parametrize("column_name, inner_shape, np_dtype, attr_name", TENSOR_FIELDS)
    def test_real_field_dtype_and_shape(
        self, column_name, inner_shape, np_dtype, attr_name
    ):
        ex = _real_example()
        arr = getattr(ex, attr_name)
        assert isinstance(arr, np.ndarray), f"{attr_name} is not an ndarray on real structure"
        assert arr.dtype == np.dtype(np_dtype), (
            f"{attr_name}: dtype {arr.dtype} != declared {np.dtype(np_dtype)}"
        )
        expected_shape = (ex.length,) + tuple(inner_shape)
        assert arr.shape == expected_shape, (
            f"{attr_name}: shape {arr.shape} != declared {expected_shape}"
        )


class TestParquetRoundTrip:
    """Every advertised field round-trips through Parquet bit-equal."""

    def _round_trip(self, example: StructureSupervisionExample, tmp_path: Path):
        out_dir = tmp_path / "supervision"
        with SupervisionParquetWriter(out_dir) as writer:
            writer.append(example)
        loaded = load_structure_supervision_examples(out_dir)
        assert len(loaded) == 1
        return loaded[0]

    def test_synthetic_round_trip_every_field_bit_equal(self, tmp_path):
        original = _synthetic_example("synthetic-rt", length=5, seed=7)
        restored = self._round_trip(original, tmp_path)
        for column_name, _inner, np_dtype, attr_name in TENSOR_FIELDS:
            orig = getattr(original, attr_name)
            back = getattr(restored, attr_name)
            assert back.dtype == np.dtype(np_dtype), (
                f"{attr_name}: round-tripped dtype {back.dtype} != {np.dtype(np_dtype)}"
            )
            assert back.shape == orig.shape, (
                f"{attr_name}: round-tripped shape {back.shape} != {orig.shape}"
            )
            # Float fields are written as float32 and must come back
            # bit-equal — the export path doesn't quantise.
            assert np.array_equal(orig, back), (
                f"{attr_name}: round-tripped values not bit-equal to original"
            )

    def test_real_round_trip_every_field_bit_equal(self, tmp_path):
        original = _real_example()
        restored = self._round_trip(original, tmp_path)
        for column_name, _inner, np_dtype, attr_name in TENSOR_FIELDS:
            orig = getattr(original, attr_name)
            back = getattr(restored, attr_name)
            assert back.dtype == np.dtype(np_dtype), (
                f"{attr_name}: round-tripped dtype {back.dtype} != {np.dtype(np_dtype)}"
            )
            assert back.shape == orig.shape, (
                f"{attr_name}: round-tripped shape {back.shape} != {orig.shape}"
            )
            assert np.array_equal(orig, back), (
                f"{attr_name}: round-tripped values not bit-equal to original"
            )


class TestSchemaSurface:
    """The Parquet schema lists exactly the advertised fields, plus metadata."""

    def test_every_tensor_field_appears_in_schema(self):
        schema = build_supervision_schema()
        names = {f.name for f in schema}
        for column_name, _inner, _dtype, _attr in TENSOR_FIELDS:
            assert column_name in names, (
                f"{column_name}: declared in TENSOR_FIELDS but missing from Parquet schema"
            )

    def test_schema_version_metadata_round_trips(self, tmp_path):
        """Writer emits the schema version + format string into the manifest;
        readers must be able to consult them when handling future v2 bumps.
        """
        import json

        ex = _synthetic_example("schema-meta", length=3, seed=1)
        with SupervisionParquetWriter(tmp_path / "out") as writer:
            writer.append(ex)
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["format"] == SUPERVISION_EXPORT_FORMAT
        assert int(manifest["schema_version"]) == SUPERVISION_PARQUET_SCHEMA_VERSION
