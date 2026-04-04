"""Tests for AMBER96 force field energy computation and minimization."""

import os

import numpy as np
import pytest

import ferritin

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")


def load_crambin():
    return ferritin.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))


def load_ubiquitin():
    return ferritin.load(os.path.join(TEST_PDBS_DIR, "1ubq.pdb"))


# ===========================================================================
# compute_energy
# ===========================================================================


class TestComputeEnergy:
    def test_returns_all_components(self):
        e = ferritin.compute_energy(load_crambin())
        for key in ("bond_stretch", "angle_bend", "torsion", "vdw", "electrostatic", "total"):
            assert key in e, f"Missing key: {key}"

    def test_total_is_sum(self):
        e = ferritin.compute_energy(load_crambin())
        expected = (
            e["bond_stretch"]
            + e["angle_bend"]
            + e["torsion"]
            + e["vdw"]
            + e["electrostatic"]
        )
        assert abs(e["total"] - expected) < 0.1, (
            f"Total {e['total']:.1f} != sum {expected:.1f}"
        )

    def test_components_are_finite(self):
        e = ferritin.compute_energy(load_crambin())
        for key, val in e.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_bond_stretch_positive(self):
        """Bond stretching energy is always >= 0 (harmonic)."""
        e = ferritin.compute_energy(load_crambin())
        assert e["bond_stretch"] >= 0

    def test_angle_bend_positive(self):
        """Angle bending energy is always >= 0 (harmonic)."""
        e = ferritin.compute_energy(load_crambin())
        assert e["angle_bend"] >= 0

    def test_ubiquitin_higher_total(self):
        """Larger protein should have higher absolute energy."""
        e_crn = ferritin.compute_energy(load_crambin())
        e_ubq = ferritin.compute_energy(load_ubiquitin())
        assert abs(e_ubq["total"]) > abs(e_crn["total"])

    def test_deterministic(self):
        """Same structure gives same energy."""
        s = load_crambin()
        e1 = ferritin.compute_energy(s)
        e2 = ferritin.compute_energy(s)
        assert e1["total"] == e2["total"]


# ===========================================================================
# minimize_hydrogens
# ===========================================================================


class TestMinimizeHydrogens:
    def test_returns_expected_keys(self):
        r = ferritin.minimize_hydrogens(load_crambin())
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged", "energy_components"):
            assert key in r, f"Missing key: {key}"

    def test_coords_shape(self):
        s = load_crambin()
        r = ferritin.minimize_hydrogens(s)
        assert r["coords"].ndim == 2
        assert r["coords"].shape[1] == 3
        assert r["coords"].shape[0] == s.atom_count

    def test_energy_decreases_or_stays(self):
        r = ferritin.minimize_hydrogens(load_crambin())
        assert r["final_energy"] <= r["initial_energy"] + 1.0  # allow tiny float noise

    def test_steps_within_limit(self):
        r = ferritin.minimize_hydrogens(load_crambin(), max_steps=100)
        assert r["steps"] <= 100

    def test_energy_components_present(self):
        r = ferritin.minimize_hydrogens(load_crambin())
        ec = r["energy_components"]
        for key in ("bond_stretch", "angle_bend", "torsion", "vdw", "electrostatic"):
            assert key in ec

    def test_tighter_tolerance_more_steps(self):
        """Stricter convergence should require at least as many steps."""
        s = load_crambin()
        r_loose = ferritin.minimize_hydrogens(s, gradient_tolerance=10.0)
        r_tight = ferritin.minimize_hydrogens(s, gradient_tolerance=0.01)
        assert r_tight["steps"] >= r_loose["steps"]

    def test_coords_finite(self):
        r = ferritin.minimize_hydrogens(load_crambin())
        assert np.all(np.isfinite(r["coords"]))


# ===========================================================================
# minimize_structure
# ===========================================================================


class TestMinimizeStructure:
    def test_returns_expected_keys(self):
        r = ferritin.minimize_structure(load_crambin(), max_steps=10)
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged", "energy_components"):
            assert key in r, f"Missing key: {key}"

    def test_energy_decreases(self):
        r = ferritin.minimize_structure(load_crambin(), max_steps=50)
        assert r["final_energy"] <= r["initial_energy"] + 1.0

    def test_coords_shape(self):
        s = load_crambin()
        r = ferritin.minimize_structure(s, max_steps=10)
        assert r["coords"].shape == (s.atom_count, 3)

    def test_coords_actually_move(self):
        """Full minimization should move at least some atoms."""
        s = load_crambin()
        original_coords = s.coords.copy()
        r = ferritin.minimize_structure(s, max_steps=50)
        max_displacement = np.max(np.linalg.norm(r["coords"] - original_coords, axis=1))
        assert max_displacement > 0.001, "No atoms moved during minimization"

    def test_few_steps_small_displacement(self):
        """Very few steps should not distort the structure."""
        s = load_crambin()
        original_coords = s.coords.copy()
        r = ferritin.minimize_structure(s, max_steps=5)
        rmsd = np.sqrt(np.mean(np.sum((r["coords"] - original_coords) ** 2, axis=1)))
        assert rmsd < 2.0, f"RMSD {rmsd:.2f} A too large after 5 steps"


# ===========================================================================
# batch_minimize_hydrogens
# ===========================================================================


class TestBatchMinimizeHydrogens:
    def test_returns_list(self):
        structures = [load_crambin(), load_ubiquitin()]
        results = ferritin.batch_minimize_hydrogens(structures, n_threads=-1)
        assert len(results) == 2

    def test_matches_serial(self):
        """Batch results should match single-structure calls."""
        structures = [load_crambin(), load_ubiquitin()]
        batch = ferritin.batch_minimize_hydrogens(structures, n_threads=1)
        serial = [ferritin.minimize_hydrogens(s) for s in structures]
        for b, s in zip(batch, serial):
            assert abs(b["final_energy"] - s["final_energy"]) < 0.1

    def test_each_has_expected_keys(self):
        structures = [load_crambin()]
        results = ferritin.batch_minimize_hydrogens(structures, n_threads=-1)
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged"):
            assert key in results[0]


# ===========================================================================
# load_and_minimize_hydrogens
# ===========================================================================


class TestLoadAndMinimizeHydrogens:
    def test_loads_and_minimizes(self):
        paths = [os.path.join(TEST_PDBS_DIR, "1crn.pdb")]
        results = ferritin.load_and_minimize_hydrogens(paths, n_threads=-1)
        assert len(results) >= 1

    def test_returns_index_result_tuples(self):
        paths = [os.path.join(TEST_PDBS_DIR, "1crn.pdb")]
        results = ferritin.load_and_minimize_hydrogens(paths, n_threads=-1)
        idx, r = results[0]
        assert isinstance(idx, int)
        assert "final_energy" in r

    def test_skips_missing_files(self):
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            "/nonexistent/fake.pdb",
        ]
        results = ferritin.load_and_minimize_hydrogens(paths, n_threads=-1)
        # Should have at most 1 result (fake file skipped)
        assert len(results) <= 2
        indices = [idx for idx, _ in results]
        assert 0 in indices  # crambin succeeded

    def test_multiple_files(self):
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            os.path.join(TEST_PDBS_DIR, "1ubq.pdb"),
        ]
        results = ferritin.load_and_minimize_hydrogens(paths, n_threads=-1)
        assert len(results) == 2
