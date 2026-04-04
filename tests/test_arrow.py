"""Tests for Arrow export/import through the full Python stack.

Tests both the low-level connector (ferritin_connector.py_arrow) and
the public package API (ferritin.to_arrow, ferritin.from_arrow, etc.).

Covers: basic round-trip, multi-model, Parquet output, schema rejection,
pyarrow interop, and the public ferritin API surface.
"""

import os
import tempfile

import numpy as np
import pytest

from ferritin_connector import py_arrow, py_io

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
CRAMBIN = os.path.join(TEST_PDBS_DIR, "1crn.pdb")
UBIQUITIN = os.path.join(TEST_PDBS_DIR, "1ubq.pdb")
MULTI_MODEL = os.path.join(TEST_PDBS_DIR, "models.pdb")


# ---------------------------------------------------------------------------
# Low-level connector: py_arrow module
# ---------------------------------------------------------------------------


class TestConnectorArrow:
    """Tests for ferritin_connector.py_arrow directly."""

    def test_to_arrow_ipc_returns_bytes(self):
        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        assert isinstance(ipc, bytes)
        assert len(ipc) > 0

    def test_to_arrow_ipc_default_id(self):
        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_arrow_ipc(pdb)
        assert isinstance(ipc, bytes)
        assert len(ipc) > 0

    def test_roundtrip_atom_count(self):
        pdb = py_io.load(CRAMBIN)
        original_count = pdb.atom_count

        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        results = py_arrow.from_arrow_ipc(ipc)

        assert len(results) == 1
        sid, rebuilt = results[0]
        assert sid == "1crn"
        assert rebuilt.atom_count == original_count

    def test_roundtrip_coordinates(self):
        pdb = py_io.load(CRAMBIN)
        orig_coords = pdb.coords

        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        results = py_arrow.from_arrow_ipc(ipc)
        _, rebuilt = results[0]
        rebuilt_coords = rebuilt.coords

        np.testing.assert_allclose(
            orig_coords, rebuilt_coords, atol=1e-6,
            err_msg="coordinates must survive round-trip"
        )

    def test_roundtrip_b_factors(self):
        pdb = py_io.load(CRAMBIN)
        orig_bf = pdb.b_factors

        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        _, rebuilt = py_arrow.from_arrow_ipc(ipc)[0]
        rebuilt_bf = rebuilt.b_factors

        np.testing.assert_allclose(
            orig_bf, rebuilt_bf, atol=1e-6,
            err_msg="B-factors must survive round-trip"
        )

    def test_roundtrip_atom_names(self):
        pdb = py_io.load(CRAMBIN)
        orig_names = pdb.atom_names

        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        _, rebuilt = py_arrow.from_arrow_ipc(ipc)[0]
        rebuilt_names = rebuilt.atom_names

        assert orig_names == rebuilt_names, "atom names must survive round-trip"

    def test_to_structure_arrow_ipc(self):
        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_structure_arrow_ipc(pdb, "1crn")
        assert isinstance(ipc, bytes)
        assert len(ipc) > 0

    def test_structure_ipc_rejected_by_from_arrow(self):
        """from_arrow_ipc must reject structure-schema bytes with a clear error."""
        pdb = py_io.load(CRAMBIN)
        structure_ipc = py_arrow.to_structure_arrow_ipc(pdb, "1crn")

        with pytest.raises(RuntimeError, match="Missing column"):
            py_arrow.from_arrow_ipc(structure_ipc)

    def test_to_parquet(self):
        pdb = py_io.load(CRAMBIN)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "1crn")
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_multi_model_roundtrip(self):
        if not os.path.exists(MULTI_MODEL):
            pytest.skip("models.pdb not available")
        pdb = py_io.load(MULTI_MODEL)
        original_models = pdb.model_count

        ipc = py_arrow.to_arrow_ipc(pdb, "models")
        results = py_arrow.from_arrow_ipc(ipc)
        _, rebuilt = results[0]

        assert rebuilt.model_count == original_models, (
            f"model count mismatch: {original_models} → {rebuilt.model_count}"
        )

    def test_multiple_structures(self):
        """Load two PDBs, export each, import both, verify."""
        pdb1 = py_io.load(CRAMBIN)
        pdb2 = py_io.load(UBIQUITIN)

        ipc1 = py_arrow.to_arrow_ipc(pdb1, "1crn")
        ipc2 = py_arrow.to_arrow_ipc(pdb2, "1ubq")

        r1 = py_arrow.from_arrow_ipc(ipc1)
        r2 = py_arrow.from_arrow_ipc(ipc2)

        assert r1[0][0] == "1crn"
        assert r2[0][0] == "1ubq"
        assert r1[0][1].atom_count == pdb1.atom_count
        assert r2[0][1].atom_count == pdb2.atom_count


# ---------------------------------------------------------------------------
# pyarrow interop (if available)
# ---------------------------------------------------------------------------


class TestPyArrowInterop:
    """Tests that IPC bytes are valid Arrow, readable by pyarrow."""

    @pytest.fixture(autouse=True)
    def _check_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_read_atom_table(self):
        import pyarrow as pa

        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")

        reader = pa.ipc.open_file(ipc)
        table = reader.read_all()

        assert table.num_rows == pdb.atom_count
        assert "x" in table.column_names
        assert "y" in table.column_names
        assert "z" in table.column_names
        assert "structure_id" in table.column_names
        assert "insertion_code" in table.column_names
        assert "conformer_id" in table.column_names

    def test_to_pandas(self):
        import pyarrow as pa

        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_arrow_ipc(pdb, "1crn")
        table = pa.ipc.open_file(ipc).read_all()
        df = table.to_pandas()

        assert len(df) == pdb.atom_count
        assert df["structure_id"].iloc[0] == "1crn"
        assert df["x"].dtype == np.float64

    def test_read_structure_table(self):
        import pyarrow as pa

        pdb = py_io.load(CRAMBIN)
        ipc = py_arrow.to_structure_arrow_ipc(pdb, "1crn")
        table = pa.ipc.open_file(ipc).read_all()

        assert table.num_rows == 1
        assert "atom_count" in table.column_names


# ---------------------------------------------------------------------------
# Public ferritin API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Tests that the public ferritin package exposes Arrow correctly."""

    def test_import_to_arrow(self):
        import ferritin
        assert hasattr(ferritin, "to_arrow")
        assert callable(ferritin.to_arrow)

    def test_import_from_arrow(self):
        import ferritin
        assert hasattr(ferritin, "from_arrow")
        assert callable(ferritin.from_arrow)

    def test_import_to_parquet(self):
        import ferritin
        assert hasattr(ferritin, "to_parquet")
        assert callable(ferritin.to_parquet)

    def test_import_to_structure_arrow(self):
        import ferritin
        assert hasattr(ferritin, "to_structure_arrow")
        assert callable(ferritin.to_structure_arrow)

    def test_public_roundtrip(self):
        import ferritin

        s = ferritin.load(CRAMBIN)
        ipc = ferritin.to_arrow(s, "1crn")
        results = ferritin.from_arrow(ipc)

        assert len(results) == 1
        sid, rebuilt = results[0]
        assert sid == "1crn"
        assert rebuilt.atom_count == s.atom_count

    def test_public_to_parquet(self):
        import ferritin

        s = ferritin.load(CRAMBIN)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            ferritin.to_parquet(s, path, "1crn")
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_import_from_parquet(self):
        import ferritin
        assert hasattr(ferritin, "from_parquet")
        assert callable(ferritin.from_parquet)


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    """Tests for Parquet write → read round-trip."""

    def test_connector_roundtrip_atom_count(self):
        pdb = py_io.load(CRAMBIN)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "1crn")
            results = py_arrow.from_parquet(path)
            assert len(results) == 1
            sid, rebuilt = results[0]
            assert sid == "1crn"
            assert rebuilt.atom_count == pdb.atom_count
        finally:
            os.unlink(path)

    def test_connector_roundtrip_coordinates(self):
        pdb = py_io.load(CRAMBIN)
        orig_coords = pdb.coords
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "1crn")
            _, rebuilt = py_arrow.from_parquet(path)[0]
            np.testing.assert_allclose(
                orig_coords, rebuilt.coords, atol=1e-6,
                err_msg="coordinates must survive Parquet round-trip"
            )
        finally:
            os.unlink(path)

    def test_connector_roundtrip_b_factors(self):
        pdb = py_io.load(CRAMBIN)
        orig_bf = pdb.b_factors
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "1crn")
            _, rebuilt = py_arrow.from_parquet(path)[0]
            np.testing.assert_allclose(
                orig_bf, rebuilt.b_factors, atol=1e-6,
                err_msg="B-factors must survive Parquet round-trip"
            )
        finally:
            os.unlink(path)

    def test_connector_roundtrip_atom_names(self):
        pdb = py_io.load(CRAMBIN)
        orig_names = pdb.atom_names
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "1crn")
            _, rebuilt = py_arrow.from_parquet(path)[0]
            assert orig_names == rebuilt.atom_names
        finally:
            os.unlink(path)

    def test_public_parquet_roundtrip(self):
        import ferritin

        s = ferritin.load(CRAMBIN)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            ferritin.to_parquet(s, path, "1crn")
            results = ferritin.from_parquet(path)
            assert len(results) == 1
            assert results[0][0] == "1crn"
            assert results[0][1].atom_count == s.atom_count
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            py_arrow.from_parquet("/nonexistent/fake.parquet")

    def test_multi_model_parquet_roundtrip(self):
        if not os.path.exists(MULTI_MODEL):
            pytest.skip("models.pdb not available")
        pdb = py_io.load(MULTI_MODEL)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            py_arrow.to_parquet(pdb, path, "models")
            results = py_arrow.from_parquet(path)
            _, rebuilt = results[0]
            assert rebuilt.model_count == pdb.model_count
        finally:
            os.unlink(path)
