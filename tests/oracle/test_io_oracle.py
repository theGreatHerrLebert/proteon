"""Oracle tests: I/O validation.

Compare proteon vs Biopython vs Gemmi on atom counts, coordinates,
B-factors, elements, and other metadata for multiple PDB files.

Standard files: strict equality across all three tools.
Edge cases: tested separately with relaxed expectations.
"""

import numpy as np
import pytest

# The conftest's extract_biopython / extract_gemmi helpers import their
# respective libraries lazily inside the function body. Gate here so the
# whole module skips cleanly on machines without them, rather than raising
# ImportError from a fixture — the oracle CI runs with `--tb=short` and a
# hard import error fails the gated job.
pytest.importorskip("Bio.PDB")
pytest.importorskip("gemmi")

from .conftest import (  # noqa: E402  (after importorskip by design)
    available_files,
    extract_biopython,
    extract_proteon,
    extract_gemmi,
)

pytestmark = [
    pytest.mark.oracle("biopython"),
    pytest.mark.oracle("gemmi"),
]


@pytest.fixture(params=available_files(), ids=lambda x: x[0])
def pdb_file(request):
    return request.param


@pytest.fixture
def all_three(pdb_file):
    """Load structure with all three tools."""
    name, path = pdb_file
    f = extract_proteon(path)
    g = extract_gemmi(path)
    b = extract_biopython(path)
    return name, f, g, b


# ---------------------------------------------------------------------------
# Counts — strict agreement
# ---------------------------------------------------------------------------


class TestCounts:
    def test_model_count_gemmi(self, all_three):
        name, f, g, b = all_three
        assert f.model_count == g.model_count, \
            f"{name}: proteon={f.model_count} vs gemmi={g.model_count}"

    def test_model_count_biopython(self, all_three):
        name, f, g, b = all_three
        assert f.model_count == b.model_count, \
            f"{name}: proteon={f.model_count} vs biopython={b.model_count}"

    def test_chain_count(self, all_three):
        name, f, g, b = all_three
        assert f.chain_count == g.chain_count == b.chain_count, \
            f"{name}: proteon={f.chain_count} gemmi={g.chain_count} bio={b.chain_count}"

    def test_chain_ids(self, all_three):
        name, f, g, b = all_three
        assert f.chain_ids == g.chain_ids, \
            f"{name}: proteon={f.chain_ids} vs gemmi={g.chain_ids}"

    def test_atom_count_gemmi(self, all_three):
        name, f, g, b = all_three
        assert f.atom_count == g.atom_count, \
            f"{name}: proteon={f.atom_count} vs gemmi={g.atom_count}"

    def test_atom_count_biopython(self, all_three):
        name, f, g, b = all_three
        assert f.atom_count == b.atom_count, \
            f"{name}: proteon={f.atom_count} vs biopython={b.atom_count}"

    def test_residue_count_gemmi(self, all_three):
        name, f, g, b = all_three
        assert f.residue_count == g.residue_count, \
            f"{name}: proteon={f.residue_count} vs gemmi={g.residue_count}"


# ---------------------------------------------------------------------------
# Coordinates — 3 decimal place precision (PDB format limit)
# ---------------------------------------------------------------------------


class TestCoordinates:
    def test_coords_vs_gemmi(self, all_three):
        name, f, g, _ = all_three
        assert len(f.atoms) == len(g.atoms)
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert abs(fa.x - ga.x) < 0.001, f"{name} atom {i}: x {fa.x} vs {ga.x}"
            assert abs(fa.y - ga.y) < 0.001, f"{name} atom {i}: y {fa.y} vs {ga.y}"
            assert abs(fa.z - ga.z) < 0.001, f"{name} atom {i}: z {fa.z} vs {ga.z}"

    def test_coords_vs_biopython(self, all_three):
        name, f, _, b = all_three
        if len(f.atoms) != len(b.atoms):
            pytest.skip(f"{name}: atom count differs (alt conformer handling)")
        for i, (fa, ba) in enumerate(zip(f.atoms, b.atoms)):
            assert abs(fa.x - ba.x) < 0.001, f"{name} atom {i}: x {fa.x} vs {ba.x}"
            assert abs(fa.y - ba.y) < 0.001, f"{name} atom {i}: y {fa.y} vs {ba.y}"
            assert abs(fa.z - ba.z) < 0.001, f"{name} atom {i}: z {fa.z} vs {ba.z}"


# ---------------------------------------------------------------------------
# Metadata — B-factors, occupancies, elements, names
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_b_factors(self, all_three):
        name, f, g, _ = all_three
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert abs(fa.b_factor - ga.b_factor) < 0.01, \
                f"{name} atom {i}: B {fa.b_factor} vs {ga.b_factor}"

    def test_occupancies(self, all_three):
        name, f, g, _ = all_three
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert abs(fa.occupancy - ga.occupancy) < 0.01, \
                f"{name} atom {i}: occ {fa.occupancy} vs {ga.occupancy}"

    def test_elements(self, all_three):
        name, f, g, _ = all_three
        mismatches = sum(
            1 for fa, ga in zip(f.atoms, g.atoms) if fa.element != ga.element
        )
        assert mismatches <= max(1, len(f.atoms) // 100), \
            f"{name}: {mismatches}/{len(f.atoms)} element mismatches"

    def test_atom_names(self, all_three):
        name, f, g, _ = all_three
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert fa.name == ga.name, \
                f"{name} atom {i}: name '{fa.name}' vs '{ga.name}'"

    def test_residue_names(self, all_three):
        name, f, g, _ = all_three
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert fa.residue_name == ga.residue_name, \
                f"{name} atom {i}: resname '{fa.residue_name}' vs '{ga.residue_name}'"

    def test_chain_ids_per_atom(self, all_three):
        name, f, g, _ = all_three
        for i, (fa, ga) in enumerate(zip(f.atoms, g.atoms)):
            assert fa.chain_id == ga.chain_id, \
                f"{name} atom {i}: chain '{fa.chain_id}' vs '{ga.chain_id}'"


# ---------------------------------------------------------------------------
# Internal consistency: numpy arrays vs per-atom
# ---------------------------------------------------------------------------


class TestInternalConsistency:
    def test_coords_array(self, pdb_file):
        from proteon_connector import py_io
        name, path = pdb_file
        pdb = py_io.load(path)
        coords = pdb.coords
        atoms = pdb.atoms
        assert coords.shape[0] == len(atoms)
        for i in range(min(50, len(atoms))):
            np.testing.assert_allclose(
                coords[i], [atoms[i].x, atoms[i].y, atoms[i].z], atol=1e-10
            )

    def test_bfactors_array(self, pdb_file):
        from proteon_connector import py_io
        name, path = pdb_file
        pdb = py_io.load(path)
        bf = pdb.b_factors
        atoms = pdb.atoms
        for i in range(min(50, len(atoms))):
            assert abs(bf[i] - atoms[i].b_factor) < 1e-10

    def test_atom_names_list(self, pdb_file):
        from proteon_connector import py_io
        name, path = pdb_file
        pdb = py_io.load(path)
        names = pdb.atom_names
        atoms = pdb.atoms
        assert len(names) == len(atoms)
        for i in range(min(50, len(atoms))):
            assert names[i].strip() == atoms[i].name.strip()
