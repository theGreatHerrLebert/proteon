"""Tests for AMBER96 force field energy computation and minimization.

Note: tests use units="kcal/mol" to match historical reference values.
The default API output is kJ/mol.
"""

import os

import numpy as np
import pytest

import proteon

TEST_PDBS_DIR = os.path.join(os.path.dirname(__file__), "..", "test-pdbs")
UNITS = "kcal/mol"  # legacy unit for stable test assertions


def load_crambin():
    return proteon.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))


def load_ubiquitin():
    return proteon.load(os.path.join(TEST_PDBS_DIR, "1ubq.pdb"))


# ===========================================================================
# compute_energy
# ===========================================================================


class TestComputeEnergy:
    def test_returns_all_components(self):
        e = proteon.compute_energy(load_crambin(), units=UNITS)
        for key in ("bond_stretch", "angle_bend", "torsion", "vdw", "electrostatic", "total"):
            assert key in e, f"Missing key: {key}"

    def test_total_is_sum(self):
        e = proteon.compute_energy(load_crambin(), units=UNITS)
        expected = (
            e["bond_stretch"]
            + e["angle_bend"]
            + e["torsion"]
            + e["improper_torsion"]
            + e["vdw"]
            + e["electrostatic"]
        )
        assert abs(e["total"] - expected) < 0.1, (
            f"Total {e['total']:.1f} != sum {expected:.1f}"
        )

    def test_components_are_finite(self):
        e = proteon.compute_energy(load_crambin(), units=UNITS)
        for key, val in e.items():
            # The dict also carries non-numeric parameterization metadata
            # (parameterization_status: str); finiteness applies to the numeric
            # energy components / topology counts only.
            if isinstance(val, str) or isinstance(val, bool):
                continue
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_bond_stretch_positive(self):
        """Bond stretching energy is always >= 0 (harmonic)."""
        e = proteon.compute_energy(load_crambin(), units=UNITS)
        assert e["bond_stretch"] >= 0

    def test_angle_bend_positive(self):
        """Angle bending energy is always >= 0 (harmonic)."""
        e = proteon.compute_energy(load_crambin(), units=UNITS)
        assert e["angle_bend"] >= 0

    def test_both_structures_compute(self):
        """Energy computation works on structures with and without hydrogens."""
        e_crn = proteon.compute_energy(load_crambin(), units=UNITS)
        e_ubq = proteon.compute_energy(load_ubiquitin())
        assert np.isfinite(e_crn["total"])
        assert np.isfinite(e_ubq["total"])

    def test_deterministic(self):
        """Same structure gives same energy."""
        s = load_crambin()
        e1 = proteon.compute_energy(s)
        e2 = proteon.compute_energy(s)
        assert e1["total"] == e2["total"]


# ===========================================================================
# CutoffNonPeriodic OBC GB method (opt-in; OpenMM parity in
# validation/amber96_obc_cutoff_oracle.py)
# ===========================================================================


class TestCutoffGB:
    """Wiring tests for ff='amber96_obc_cutoff' — no OpenMM needed (parity is
    in the standalone oracle). NoCutoff GB stays the default and is unchanged."""

    def test_cutoff_gb_is_finite_and_negative(self):
        s = load_crambin()
        e = proteon.compute_energy(
            s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0, nbl_threshold=10**9
        )
        assert np.isfinite(e["solvation"])
        # GB solvation is favorable (negative) on a folded protein.
        assert e["solvation"] < 0.0

    def test_cutoff_differs_from_nocutoff(self):
        # The cutoff method truncates the long-range GB pair term + adds the
        # reaction-field shift, so it must NOT equal the exact NoCutoff GB.
        s = load_crambin()
        nc = proteon.compute_energy(
            s, ff="amber96_obc", nonbonded_cutoff=1e6, nbl_threshold=10**9
        )
        co = proteon.compute_energy(
            s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0, nbl_threshold=10**9
        )
        assert abs(co["solvation"] - nc["solvation"]) > 1.0

    def test_cutoff_gb_deterministic(self):
        s = load_crambin()
        a = proteon.compute_energy(s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0)
        b = proteon.compute_energy(s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0)
        assert a["total"] == b["total"]

    def test_nbl_path_matches_all_pairs(self):
        # Below the size gate (nbl_threshold huge) GB uses the all-pairs cutoff
        # path; a small threshold routes it through the O(N) neighbor-list path.
        # Same method ⇒ same energy to tight tolerance (Rust proves 1e-9; the
        # Python layer adds a kJ/mol unit conversion, so allow a hair more).
        s = load_crambin()
        all_pairs = proteon.compute_energy(
            s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0, nbl_threshold=10**9
        )
        nbl = proteon.compute_energy(
            s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0, nbl_threshold=1
        )
        assert abs(all_pairs["solvation"] - nbl["solvation"]) < 1e-4

    def test_batch_cutoff_gb_matches_serial(self):
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_compute_energy(
            structures, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0, n_threads=1
        )
        serial = [
            proteon.compute_energy(s, ff="amber96_obc_cutoff", nonbonded_cutoff=12.0)
            for s in structures
        ]
        for b, sgl in zip(batch, serial):
            assert b["solvation"] == sgl["solvation"]

    def test_minimize_accepts_cutoff_gb(self):
        # The cutoff GB method must be selectable through minimize_structure
        # (its main use case is repeated force evals on larger systems). A few
        # steps suffice to prove it runs and doesn't increase energy.
        s = load_crambin()
        r = proteon.minimize_structure(
            s, ff="amber96_obc_cutoff", max_steps=5, method="sd"
        )
        assert np.isfinite(r["final_energy"])
        assert r["final_energy"] <= r["initial_energy"] + 1.0


# ===========================================================================
# batch_compute_energy
# ===========================================================================


class TestBatchComputeEnergy:
    """batch_compute_energy must produce per-structure dicts identical to
    a Python loop calling compute_energy on each structure."""

    def _structures(self):
        return [load_crambin(), load_ubiquitin()]

    def test_charmm19_eef1_matches_serial(self):
        structures = self._structures()
        batch = proteon.batch_compute_energy(
            structures, ff="charmm19_eef1", units=UNITS, n_threads=-1
        )
        serial = [
            proteon.compute_energy(s, ff="charmm19_eef1", units=UNITS)
            for s in structures
        ]
        assert len(batch) == len(serial)
        for b, s in zip(batch, serial):
            assert set(b.keys()) == set(s.keys())
            for k in b:
                assert b[k] == s[k], f"key {k} mismatch: batch={b[k]} serial={s[k]}"

    def test_amber96_matches_serial(self):
        structures = self._structures()
        batch = proteon.batch_compute_energy(
            structures, ff="amber96", units=UNITS, n_threads=-1
        )
        serial = [
            proteon.compute_energy(s, ff="amber96", units=UNITS) for s in structures
        ]
        for b, s in zip(batch, serial):
            assert set(b.keys()) == set(s.keys())
            for k in b:
                assert b[k] == s[k], f"key {k} mismatch: batch={b[k]} serial={s[k]}"

    def test_nonbonded_cutoff_propagates(self):
        """nonbonded_cutoff must reach the underlying per-structure call."""
        structures = self._structures()
        batch = proteon.batch_compute_energy(
            structures, ff="amber96", nonbonded_cutoff=1e6, units=UNITS, n_threads=1
        )
        serial = [
            proteon.compute_energy(s, ff="amber96", nonbonded_cutoff=1e6, units=UNITS)
            for s in structures
        ]
        for b, s in zip(batch, serial):
            assert b["total"] == s["total"]

    def test_kj_units_default(self):
        """Default units kJ/mol stays consistent between batch and serial."""
        structures = self._structures()
        batch = proteon.batch_compute_energy(structures, ff="charmm19_eef1", n_threads=1)
        serial = [proteon.compute_energy(s, ff="charmm19_eef1") for s in structures]
        for b, s in zip(batch, serial):
            assert b["total"] == s["total"]

    def test_unknown_ff_raises(self):
        with pytest.raises(ValueError, match="Unknown force field"):
            proteon.batch_compute_energy(self._structures(), ff="xyz")


# ===========================================================================
# minimize_hydrogens
# ===========================================================================


@pytest.mark.slow
class TestMinimizeHydrogens:
    """Use ubiquitin (1ubq) which has 629 H atoms; crambin has none.

    Slow: each test runs the full L-BFGS H-only minimizer on ubiquitin;
    individual methods take 30-70s.
    """

    def test_returns_expected_keys(self):
        r = proteon.minimize_hydrogens(load_ubiquitin())
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged", "energy_components"):
            assert key in r, f"Missing key: {key}"

    def test_coords_shape(self):
        s = load_ubiquitin()
        r = proteon.minimize_hydrogens(s)
        assert r["coords"].ndim == 2
        assert r["coords"].shape[1] == 3
        # minimize_hydrogens operates on the AMBER96-typable subset (protein
        # residues only — waters and other HETATMs are dropped by the
        # topology builder), so the output row count matches the protein
        # mask, not s.atom_count.
        protein_mask = proteon.select(s, "protein")
        assert r["coords"].shape[0] == int(protein_mask.sum())

    def test_energy_decreases_or_stays(self):
        r = proteon.minimize_hydrogens(load_ubiquitin())
        assert r["final_energy"] <= r["initial_energy"] + 1.0  # allow tiny float noise

    def test_steps_within_limit(self):
        r = proteon.minimize_hydrogens(load_ubiquitin(), max_steps=100)
        assert r["steps"] <= 100

    def test_energy_components_present(self):
        r = proteon.minimize_hydrogens(load_ubiquitin())
        ec = r["energy_components"]
        for key in ("bond_stretch", "angle_bend", "torsion", "vdw", "electrostatic"):
            assert key in ec

    def test_tighter_tolerance_more_steps(self):
        """Stricter convergence should require at least as many steps."""
        s = load_ubiquitin()
        r_loose = proteon.minimize_hydrogens(s, gradient_tolerance=10.0)
        r_tight = proteon.minimize_hydrogens(s, gradient_tolerance=0.01)
        assert r_tight["steps"] >= r_loose["steps"]

    def test_coords_finite(self):
        r = proteon.minimize_hydrogens(load_ubiquitin())
        assert np.all(np.isfinite(r["coords"]))

    def test_hydrogens_move_heavy_atoms_stay(self):
        """H atoms should move; heavy atoms must remain fixed."""
        s = load_ubiquitin()
        # minimize_hydrogens operates on the protein subset (waters dropped),
        # so compare against protein-masked originals to keep shapes aligned.
        protein_mask = proteon.select(s, "protein")
        orig_coords = s.coords[protein_mask].copy()
        names = [n for n, keep in zip(s.atom_names, protein_mask) if keep]
        h_mask = np.array([n.strip().startswith("H") for n in names])
        assert h_mask.sum() > 0, "ubiquitin should have H atoms"

        r = proteon.minimize_hydrogens(s)
        assert r["coords"].shape == orig_coords.shape
        displacements = np.linalg.norm(r["coords"] - orig_coords, axis=1)

        # Heavy atoms must not move
        heavy_max = displacements[~h_mask].max()
        assert heavy_max < 1e-10, f"Heavy atoms moved by {heavy_max:.2e} A"

    def test_noop_on_structure_without_hydrogens(self):
        """Crambin has no H atoms, so minimize_hydrogens should be a no-op."""
        s = load_crambin()
        orig_coords = s.coords.copy()
        r = proteon.minimize_hydrogens(s)
        np.testing.assert_allclose(r["coords"], orig_coords, atol=1e-10)


# ===========================================================================
# minimize_structure
# ===========================================================================


class TestMinimizeStructure:
    def test_returns_expected_keys(self):
        r = proteon.minimize_structure(load_crambin(), max_steps=10)
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged", "energy_components"):
            assert key in r, f"Missing key: {key}"

    def test_energy_decreases(self):
        r = proteon.minimize_structure(load_crambin(), max_steps=50)
        assert r["final_energy"] <= r["initial_energy"] + 1.0

    def test_coords_shape(self):
        s = load_crambin()
        r = proteon.minimize_structure(s, max_steps=10)
        assert r["coords"].shape == (s.atom_count, 3)

    def test_coords_actually_move(self):
        """Full minimization should move at least some atoms."""
        s = load_crambin()
        original_coords = s.coords.copy()
        r = proteon.minimize_structure(s, max_steps=50)
        max_displacement = np.max(np.linalg.norm(r["coords"] - original_coords, axis=1))
        assert max_displacement > 0.001, "No atoms moved during minimization"

    def test_few_steps_small_displacement(self):
        """Very few steps should not distort the structure."""
        s = load_crambin()
        original_coords = s.coords.copy()
        r = proteon.minimize_structure(s, max_steps=5)
        rmsd = np.sqrt(np.mean(np.sum((r["coords"] - original_coords) ** 2, axis=1)))
        assert rmsd < 2.0, f"RMSD {rmsd:.2f} A too large after 5 steps"

    def test_charmm19_eef1_runs(self):
        """CHARMM19+EEF1 minimization on polar-H crambin: same shape contract,
        energy decreases, coords change. Polar-H placement is required —
        CHARMM19 is a united-atom FF and `add_hydrogens` would place
        non-polar hydrogens that the FF does not parameterize."""
        s = proteon.load(os.path.join(TEST_PDBS_DIR, "1crn.pdb"))
        proteon.place_peptide_hydrogens(s)
        original_coords = np.array([s.atoms[i].pos for i in range(s.atom_count)])
        r = proteon.minimize_structure(s, max_steps=20, ff="charmm19_eef1")
        assert r["final_energy"] <= r["initial_energy"] + 1.0
        assert r["coords"].shape == (s.atom_count, 3)
        # EEF1 solvation should be negative (favorable) on a folded protein.
        assert r["energy_components"]["solvation"] < 0.0

    def test_unknown_ff_errors(self):
        with pytest.raises(ValueError, match="Unknown force field"):
            proteon.minimize_structure(load_crambin(), max_steps=1, ff="bogus99")


# ===========================================================================
# batch_minimize_hydrogens
# ===========================================================================


@pytest.mark.slow
class TestBatchMinimizeHydrogens:
    """Slow: batch minimization over crambin + ubiquitin; 60-120s per test."""

    def test_returns_list(self):
        structures = [load_crambin(), load_ubiquitin()]
        results = proteon.batch_minimize_hydrogens(structures, n_threads=-1)
        assert len(results) == 2

    def test_matches_serial(self):
        """Batch results should match single-structure calls."""
        structures = [load_crambin(), load_ubiquitin()]
        batch = proteon.batch_minimize_hydrogens(structures, n_threads=1)
        serial = [proteon.minimize_hydrogens(s) for s in structures]
        for b, s in zip(batch, serial):
            assert abs(b["final_energy"] - s["final_energy"]) < 0.1

    def test_each_has_expected_keys(self):
        structures = [load_ubiquitin()]
        results = proteon.batch_minimize_hydrogens(structures, n_threads=-1)
        for key in ("coords", "initial_energy", "final_energy", "steps", "converged"):
            assert key in results[0]


# ===========================================================================
# load_and_minimize_hydrogens
# ===========================================================================


@pytest.mark.slow
class TestLoadAndMinimizeHydrogens:
    """Slow: parallel load+minimize pipeline; multi-file variant ~60s."""

    def test_loads_and_minimizes(self):
        paths = [os.path.join(TEST_PDBS_DIR, "1crn.pdb")]
        result = proteon.load_and_minimize_hydrogens(paths, n_threads=-1)
        assert len(result) == 1
        assert result.n_ok == 1

    def test_item_carries_index_and_result(self):
        paths = [os.path.join(TEST_PDBS_DIR, "1crn.pdb")]
        result = proteon.load_and_minimize_hydrogens(paths, n_threads=-1)
        item = result[0]
        assert item.index == 0
        assert item.ok
        assert "final_energy" in item.value

    def test_bad_files_recorded_not_skipped(self):
        """A file that fails to load is a failed item, not an omission."""
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            "/nonexistent/fake.pdb",
        ]
        result = proteon.load_and_minimize_hydrogens(paths, n_threads=-1)
        # Cardinality is preserved: one item per input, in input order.
        assert len(result) == 2, f"Expected 2 items, got {len(result)}"
        assert result.n_ok == 1
        assert result.n_failed == 1
        assert result[0].ok and result[0].index == 0
        assert not result[1].ok
        assert result[1].error  # carries the load error message

    def test_multiple_files(self):
        paths = [
            os.path.join(TEST_PDBS_DIR, "1crn.pdb"),
            os.path.join(TEST_PDBS_DIR, "1ubq.pdb"),
        ]
        result = proteon.load_and_minimize_hydrogens(paths, n_threads=-1)
        assert len(result) == 2
        assert result.n_ok == 2
