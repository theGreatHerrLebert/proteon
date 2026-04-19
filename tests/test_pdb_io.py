"""Tests for the PyO3 PDB I/O connector.

Tests the full stack: proteon_connector.py_io and py_pdb modules.
Validates: loading, hierarchy navigation, numpy arrays, save/roundtrip.
"""

import os
import tempfile

import numpy as np
import pytest

from proteon_connector import py_io, py_pdb

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
EXAMPLE_PDB = os.path.join(TEST_PDBS_DIR, "1ubq.pdb")
EXAMPLE_CIF = os.path.join(TEST_PDBS_DIR, "1ubq.cif")
MULTI_MODEL = os.path.join(TEST_PDBS_DIR, "models.pdb")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_load_pdb_auto(self):
        pdb = py_io.load(EXAMPLE_PDB)
        assert isinstance(pdb, py_pdb.PyPDB)
        assert pdb.atom_count > 0

    def test_load_pdb_explicit(self):
        pdb = py_io.load_pdb(EXAMPLE_PDB)
        assert pdb.atom_count > 0

    def test_load_mmcif(self):
        pdb = py_io.load_mmcif(EXAMPLE_CIF)
        assert pdb.atom_count > 0

    def test_load_nonexistent_raises(self):
        with pytest.raises(IOError):
            py_io.load("/nonexistent/file.pdb")


# ---------------------------------------------------------------------------
# Hierarchy counts
# ---------------------------------------------------------------------------


class TestHierarchyCounts:
    @pytest.fixture
    def pdb(self):
        return py_io.load(EXAMPLE_PDB)

    def test_model_count(self, pdb):
        assert pdb.model_count >= 1

    def test_chain_count(self, pdb):
        assert pdb.chain_count >= 1

    def test_residue_count(self, pdb):
        assert pdb.residue_count > 50

    def test_atom_count(self, pdb):
        assert pdb.atom_count > 500

    def test_total_atom_count(self, pdb):
        assert pdb.total_atom_count >= pdb.atom_count

    def test_len(self, pdb):
        assert len(pdb) == pdb.atom_count


# ---------------------------------------------------------------------------
# Hierarchy navigation
# ---------------------------------------------------------------------------


class TestHierarchyNavigation:
    @pytest.fixture
    def pdb(self):
        return py_io.load(EXAMPLE_PDB)

    def test_models_list(self, pdb):
        models = pdb.models
        assert len(models) == pdb.model_count
        assert all(isinstance(m, py_pdb.PyModel) for m in models)

    def test_chains_list(self, pdb):
        chains = pdb.chains
        assert len(chains) == pdb.chain_count
        assert all(isinstance(c, py_pdb.PyChain) for c in chains)

    def test_residues_list(self, pdb):
        residues = pdb.residues
        assert len(residues) == pdb.residue_count
        assert all(isinstance(r, py_pdb.PyResidue) for r in residues)

    def test_atoms_list(self, pdb):
        atoms = pdb.atoms
        assert len(atoms) == pdb.atom_count
        assert all(isinstance(a, py_pdb.PyAtom) for a in atoms)

    def test_hierarchy_traversal(self, pdb):
        """Traverse full hierarchy and count atoms manually."""
        count = 0
        for model in pdb.models:
            for chain in model.chains:
                for residue in chain.residues:
                    for atom in residue.atoms:
                        count += 1
        assert count == pdb.total_atom_count

    def test_chain_residues(self, pdb):
        chain = pdb.chains[0]
        assert chain.residue_count == len(chain.residues)

    def test_chain_atoms(self, pdb):
        chain = pdb.chains[0]
        assert chain.atom_count == len(chain.atoms)

    def test_residue_atoms(self, pdb):
        residue = pdb.residues[0]
        assert len(residue) == len(residue.atoms)


# ---------------------------------------------------------------------------
# Atom properties
# ---------------------------------------------------------------------------


class TestAtomProperties:
    @pytest.fixture
    def atom(self):
        pdb = py_io.load(EXAMPLE_PDB)
        return pdb.atoms[0]

    def test_name(self, atom):
        assert isinstance(atom.name, str)
        assert len(atom.name.strip()) > 0

    def test_serial_number(self, atom):
        assert isinstance(atom.serial_number, int)

    def test_coordinates(self, atom):
        assert np.isfinite(atom.x)
        assert np.isfinite(atom.y)
        assert np.isfinite(atom.z)
        x, y, z = atom.pos
        assert x == atom.x
        assert y == atom.y
        assert z == atom.z

    def test_element(self, atom):
        assert atom.element is not None
        assert isinstance(atom.element, str)

    def test_b_factor(self, atom):
        assert np.isfinite(atom.b_factor)
        assert atom.b_factor >= 0.0

    def test_occupancy(self, atom):
        assert 0.0 <= atom.occupancy <= 1.0

    def test_hetero(self, atom):
        assert isinstance(atom.hetero, bool)

    def test_is_backbone(self, atom):
        assert isinstance(atom.is_backbone, bool)

    def test_residue_context(self, atom):
        assert isinstance(atom.residue_name, str)
        assert isinstance(atom.chain_id, str)
        assert isinstance(atom.residue_serial_number, int)


# ---------------------------------------------------------------------------
# Residue properties
# ---------------------------------------------------------------------------


class TestResidueProperties:
    @pytest.fixture
    def residue(self):
        pdb = py_io.load(EXAMPLE_PDB)
        return pdb.residues[0]

    def test_name(self, residue):
        assert residue.name is not None
        assert len(residue.name) > 0

    def test_serial_number(self, residue):
        assert isinstance(residue.serial_number, int)

    def test_chain_id(self, residue):
        assert isinstance(residue.chain_id, str)
        assert len(residue.chain_id) > 0

    def test_is_amino_acid(self, residue):
        assert isinstance(residue.is_amino_acid, bool)

    def test_conformer_names(self, residue):
        names = residue.conformer_names
        assert isinstance(names, list)
        assert len(names) >= 1


# ---------------------------------------------------------------------------
# Chain properties
# ---------------------------------------------------------------------------


class TestChainProperties:
    def test_chain_id(self):
        pdb = py_io.load(EXAMPLE_PDB)
        chain = pdb.chains[0]
        assert isinstance(chain.id, str)
        assert len(chain.id) > 0

    def test_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        chain = pdb.chains[0]
        r = repr(chain)
        assert "Chain" in r


# ---------------------------------------------------------------------------
# Numpy array extraction
# ---------------------------------------------------------------------------


class TestNumpyArrays:
    @pytest.fixture
    def pdb(self):
        return py_io.load(EXAMPLE_PDB)

    def test_coords_shape(self, pdb):
        coords = pdb.coords
        assert isinstance(coords, np.ndarray)
        assert coords.dtype == np.float64
        assert coords.shape == (pdb.atom_count, 3)

    def test_coords_finite(self, pdb):
        assert np.all(np.isfinite(pdb.coords))

    def test_b_factors_shape(self, pdb):
        bf = pdb.b_factors
        assert isinstance(bf, np.ndarray)
        assert bf.dtype == np.float64
        assert bf.shape == (pdb.atom_count,)

    def test_occupancies_shape(self, pdb):
        occ = pdb.occupancies
        assert isinstance(occ, np.ndarray)
        assert occ.shape == (pdb.atom_count,)

    def test_residue_serial_numbers(self, pdb):
        rsn = pdb.residue_serial_numbers
        assert isinstance(rsn, np.ndarray)
        assert rsn.shape == (pdb.atom_count,)

    def test_coords_match_atoms(self, pdb):
        """Coords array should match individual atom positions."""
        coords = pdb.coords
        atoms = pdb.atoms
        for i in range(min(10, len(atoms))):
            x, y, z = atoms[i].pos
            np.testing.assert_allclose(coords[i], [x, y, z])


# ---------------------------------------------------------------------------
# Bulk metadata
# ---------------------------------------------------------------------------


class TestBulkMetadata:
    @pytest.fixture
    def pdb(self):
        return py_io.load(EXAMPLE_PDB)

    def test_atom_names(self, pdb):
        names = pdb.atom_names
        assert isinstance(names, list)
        assert len(names) == pdb.atom_count
        assert all(isinstance(n, str) for n in names)

    def test_elements(self, pdb):
        elems = pdb.elements
        assert len(elems) == pdb.atom_count
        assert "C" in elems
        assert "N" in elems

    def test_residue_names(self, pdb):
        rnames = pdb.residue_names
        assert len(rnames) == pdb.atom_count
        assert all(isinstance(n, str) for n in rnames)

    def test_chain_ids(self, pdb):
        cids = pdb.chain_ids
        assert len(cids) == pdb.atom_count
        assert all(isinstance(c, str) for c in cids)


# ---------------------------------------------------------------------------
# Multi-model
# ---------------------------------------------------------------------------


class TestMultiModel:
    def test_multi_model_count(self):
        pdb = py_io.load(MULTI_MODEL)
        assert pdb.model_count > 1

    def test_models_have_consistent_atoms(self):
        pdb = py_io.load(MULTI_MODEL)
        models = pdb.models
        counts = [m.atom_count for m in models]
        assert all(c == counts[0] for c in counts), "all models should have same atom count"


# ---------------------------------------------------------------------------
# Save / roundtrip
# ---------------------------------------------------------------------------


class TestSaveRoundtrip:
    def test_save_and_reload(self):
        original = py_io.load(EXAMPLE_PDB)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            py_io.save(original, tmp_path)
            reloaded = py_io.load(tmp_path)
            assert reloaded.atom_count == original.atom_count
            # Coordinates should match to PDB precision (3 decimal places)
            np.testing.assert_allclose(
                reloaded.coords, original.coords, atol=0.001
            )
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_pdb_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        r = repr(pdb)
        assert "PDB" in r
        assert "models=" in r

    def test_model_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        r = repr(pdb.models[0])
        assert "Model" in r

    def test_chain_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        r = repr(pdb.chains[0])
        assert "Chain" in r

    def test_residue_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        r = repr(pdb.residues[0])
        assert "Residue" in r

    def test_atom_repr(self):
        pdb = py_io.load(EXAMPLE_PDB)
        r = repr(pdb.atoms[0])
        assert "Atom" in r


# ---------------------------------------------------------------------------
# Multiple test PDB files (smoke test)
# ---------------------------------------------------------------------------


class TestMultiplePDBs:
    """Smoke test: load several different PDB files and check basics."""

    @pytest.fixture(params=["1crn.pdb", "1ubi.pdb", "4hhb.pdb"])
    def pdb_path(self, request):
        path = os.path.join(TEST_PDBS_DIR, request.param)
        if not os.path.exists(path):
            pytest.skip(f"{request.param} not available")
        return path

    def test_load_and_basic_checks(self, pdb_path):
        pdb = py_io.load(pdb_path)
        assert pdb.atom_count > 0
        assert pdb.chain_count >= 1
        assert pdb.coords.shape == (pdb.atom_count, 3)
        assert np.all(np.isfinite(pdb.coords))
        assert len(pdb.atom_names) == pdb.atom_count
        assert len(pdb.elements) == pdb.atom_count
