"""End-to-end tests for proteon.prepare() pipeline.

RELIABILITY_ROADMAP P1.6: Structure preparation tests.

Covers:
- prepare() single structure
- batch_prepare() parallel
- load_and_prepare() convenience
- Idempotency (repeated prep doesn't keep mutating)
- Different hydrogen modes
- Minimization actually changes coordinates
- PrepReport correctness
- CHARMM/EEF1 force field selection
- run_md() basic functionality
"""

import os
import numpy as np
import pytest

import proteon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAMBIN = os.path.join(REPO, "test-pdbs", "1crn.pdb")
UBIQ = os.path.join(REPO, "test-pdbs", "1ubq.pdb")


# =========================================================================
# prepare()
# =========================================================================


class TestPrepare:
    def test_returns_prep_report(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, reconstruct=False, minimize=False)
        assert isinstance(report, proteon.PrepReport)

    def test_places_hydrogens(self):
        s = proteon.load(CRAMBIN)
        n_before = s.atom_count
        report = proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=False)
        n_after = s.atom_count
        assert report.hydrogens_added > 0
        assert n_after > n_before

    def test_minimize_changes_coords(self):
        s1 = proteon.load(CRAMBIN)
        proteon.place_peptide_hydrogens(s1)
        coords_before = s1.coords.copy()

        s2 = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s2, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_steps=100,
        )
        coords_after = s2.coords

        assert coords_before.shape == coords_after.shape
        n_moved = (np.abs(coords_before - coords_after) > 1e-6).any(axis=1).sum()
        assert n_moved > 0, "Minimization should move some atoms"
        # Assertion intent: minimization saw a real non-degenerate
        # initial state — not that the starting energy is positive.
        # The "> 0" assertion held only because proteon's CHARMM
        # over-counted torsions (crambin's pre-minimisation energy
        # was always positive on the AMBER-style enumeration). With
        # the [ResidueTorsions] filter the count is correct and
        # pre-minimisation energy on a well-resolved fixture like
        # crambin can come out negative — that's better physics, not
        # a regression.
        assert report.initial_energy != report.final_energy
        assert report.final_energy <= report.initial_energy

    def test_energy_decreases(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_steps=200,
        )
        assert report.final_energy <= report.initial_energy

    def test_minimized_flag_honest_when_optimizer_does_no_work(self):
        # minimize=True but a zero step budget -> the optimizer returns NotRun,
        # so `minimized` must be False (it did not run), not True (codex review).
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_steps=0,
        )
        assert report.minimized is False
        assert report.minimizer_status in ("not_run", "")

    def test_minimized_flag_true_when_optimizer_runs(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_steps=50,
        )
        assert report.minimized is True

    def test_all_hydrogens_mode(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, reconstruct=False, hydrogens="all", minimize=False)
        assert report.hydrogens_added > 10  # sidechain adds many more than backbone

    def test_general_hydrogens_mode(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, reconstruct=False, hydrogens="general", minimize=False)
        assert report.hydrogens_added > 10

    def test_no_hydrogens_mode(self):
        s = proteon.load(CRAMBIN)
        n_before = s.atom_count
        report = proteon.prepare(s, reconstruct=False, hydrogens="none", minimize=False)
        assert report.hydrogens_added == 0
        assert s.atom_count == n_before

    def test_invalid_hydrogens_warns(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, reconstruct=False, hydrogens="INVALID", minimize=False)
        assert report.hydrogens_added == 0
        assert any("Unknown" in w for w in report.warnings)

    def test_reconstruct_adds_atoms(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, reconstruct=True, hydrogens="none", minimize=False)
        assert report.atoms_reconstructed >= 0  # may be 0 if structure is complete

    def test_no_nan_coordinates(self):
        s = proteon.load(CRAMBIN)
        proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=True, minimize_steps=50)
        coords = s.coords
        assert not np.any(np.isnan(coords)), "No NaN coordinates after prepare"
        assert not np.any(np.isinf(coords)), "No Inf coordinates after prepare"

    def test_lbfgs_minimizer(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_method="lbfgs", minimize_steps=50,
        )
        assert report.minimizer_steps > 0

    def test_cg_minimizer(self):
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(
            s, reconstruct=False, hydrogens="backbone",
            minimize=True, minimize_method="cg", minimize_steps=50,
        )
        assert report.minimizer_steps > 0

    def test_default_tolerance_actually_converges(self):
        # The default gradient_tolerance (1.0 kcal/mol/A) is the achievable
        # max-per-atom-force band: a full default prepare must report honest
        # convergence, not burn the step budget at an unreachable 0.1. Regression
        # guard for the minimize-defaults tuning.
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s)  # all defaults: minimize on, tol 1.0
        assert report.minimized is True
        assert report.converged is True, (
            f"crambin should converge at the default tolerance, "
            f"got status={report.minimizer_status}"
        )

    def test_tight_tolerance_can_still_be_requested(self):
        # The tighter 0.1 is still available for callers who want it (and will
        # typically hit the step cap on heavy-atom relaxation — that's expected).
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, gradient_tolerance=0.1, minimize_steps=50)
        assert report.minimizer_steps > 0

    def test_default_is_h_only_heavy_atoms_frozen(self):
        # Unified default constrain_heavy=True (H-only): heavy-atom coordinates
        # are preserved exactly, only hydrogens move.
        ref = proteon.load(CRAMBIN)
        ca_before = proteon.batch_extract_ca([ref])[0].copy()
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s)  # default H-only
        ca_after = proteon.batch_extract_ca([s])[0]
        assert ca_before.shape == ca_after.shape
        max_ca_shift = float(np.max(np.abs(ca_before - ca_after)))
        assert max_ca_shift < 1e-6, f"H-only must not move CA, moved {max_ca_shift} A"
        # The honesty flag: default prep did NOT relax heavy atoms.
        assert report.heavy_relaxed is False

    def test_constrain_heavy_false_relaxes_heavy_atoms(self):
        # Opt-in heavy relaxation moves the backbone (the pre-unification CHARMM
        # behaviour, still available).
        ca_before = proteon.batch_extract_ca([proteon.load(CRAMBIN)])[0].copy()
        s = proteon.load(CRAMBIN)
        report = proteon.prepare(s, constrain_heavy=False)
        ca_after = proteon.batch_extract_ca([s])[0]
        max_ca_shift = float(np.max(np.abs(ca_before - ca_after)))
        assert max_ca_shift > 1e-3, "heavy relax should move CA atoms"
        # The honesty flag: heavy relaxation DID move heavy atoms.
        assert report.heavy_relaxed is True

    def test_reconstruct_under_h_only_warns_unrelaxed(self):
        # reconstruct adds heavy atoms; under the H-only default they stay at
        # template positions, so prepare warns. Heavy relax settles them (no warn).
        import os
        fixture = os.path.join(
            os.path.dirname(__file__), "corpus", "missing_atoms", "missing_cb.pdb"
        )
        r_h = proteon.prepare(proteon.load(fixture))  # default H-only
        assert r_h.atoms_reconstructed > 0
        assert r_h.heavy_relaxed is False
        assert any("left unrelaxed" in w for w in r_h.warnings)

        r_heavy = proteon.prepare(proteon.load(fixture), constrain_heavy=False)
        assert r_heavy.heavy_relaxed is True
        assert not any("left unrelaxed" in w for w in r_heavy.warnings)

        # minimize=False: unrelaxed because minimization was SKIPPED, not H-only
        # (codex) — the message must not blame constrain_heavy.
        r_nomin = proteon.prepare(proteon.load(fixture), minimize=False)
        assert r_nomin.atoms_reconstructed > 0
        assert any("minimization did not run" in w for w in r_nomin.warnings)
        assert not any("H-only" in w for w in r_nomin.warnings)

    def test_batch_prepare_default_matches_prepare(self):
        # The unification: batch_prepare now defaults H-only, same as prepare().
        ca_before = proteon.batch_extract_ca([proteon.load(CRAMBIN)])[0].copy()
        s = proteon.load(CRAMBIN)
        proteon.batch_prepare([s])  # default, was heavy-relax for CHARMM pre-unification
        ca_after = proteon.batch_extract_ca([s])[0]
        max_ca_shift = float(np.max(np.abs(ca_before - ca_after)))
        assert max_ca_shift < 1e-6, "batch_prepare default must now be H-only too"


# =========================================================================
# Idempotency
# =========================================================================


class TestIdempotency:
    def test_double_prepare_stable_atom_count(self):
        s = proteon.load(CRAMBIN)
        proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=False)
        n1 = s.atom_count

        proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=False)
        n2 = s.atom_count

        assert n1 == n2, "Second prepare should not add more atoms"

    def test_double_prepare_stable_coords(self):
        s = proteon.load(CRAMBIN)
        proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=True, minimize_steps=50)
        coords1 = s.coords.copy()

        proteon.prepare(s, reconstruct=False, hydrogens="backbone", minimize=True, minimize_steps=50)
        coords2 = s.coords

        max_diff = np.abs(coords1 - coords2).max()
        assert max_diff < 0.5, f"Repeated prep should converge, got max diff {max_diff:.4f}"


# =========================================================================
# batch_prepare()
# =========================================================================


class TestBatchPrepare:
    def test_returns_list_of_reports(self):
        structures = [proteon.load(CRAMBIN), proteon.load(UBIQ)]
        reports = proteon.batch_prepare(structures, reconstruct=False, minimize=False)
        assert len(reports) == 2
        assert all(isinstance(r, proteon.PrepReport) for r in reports)

    def test_modifies_structures_in_place(self):
        s1 = proteon.load(CRAMBIN)
        s2 = proteon.load(CRAMBIN)  # use same structure twice for consistent test
        n1_before = s1.atom_count
        n2_before = s2.atom_count

        reports = proteon.batch_prepare([s1, s2], reconstruct=False, hydrogens="backbone", minimize=False)

        # At least one structure should have atoms added
        total_added = sum(r.hydrogens_added for r in reports)
        assert total_added > 0
        assert s1.atom_count >= n1_before

    def test_batch_matches_serial(self):
        """Batch and serial should produce same hydrogen counts."""
        s_serial = proteon.load(CRAMBIN)
        r_serial = proteon.prepare(s_serial, reconstruct=False, hydrogens="backbone", minimize=False)

        s_batch = proteon.load(CRAMBIN)
        r_batch = proteon.batch_prepare(
            [s_batch], reconstruct=False, hydrogens="backbone", minimize=False,
        )[0]

        assert r_serial.hydrogens_added == r_batch.hydrogens_added
        assert r_serial.hydrogens_skipped == r_batch.hydrogens_skipped


# =========================================================================
# load_and_prepare()
# =========================================================================


class TestLoadAndPrepare:
    def test_returns_structure_and_report(self):
        s, report = proteon.load_and_prepare(CRAMBIN, reconstruct=False, minimize=False)
        assert s is not None
        assert isinstance(report, proteon.PrepReport)
        assert s.atom_count > 0

    def test_structure_has_hydrogens(self):
        s, report = proteon.load_and_prepare(CRAMBIN, reconstruct=False, minimize=False)
        assert report.hydrogens_added > 0


# =========================================================================
# Force field selection
# =========================================================================


class TestForceField:
    def test_amber96_default(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s)
        assert e["total"] != 0
        assert e["solvation"] == 0.0  # AMBER has no solvation

    def test_amber96_explicit(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="amber96")
        assert e["total"] != 0

    def test_charmm19_eef1(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="charmm19_eef1")
        assert e["total"] != 0
        assert e["solvation"] != 0.0  # EEF1 should contribute

    def test_charmm_solvation_nonzero(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="charmm19_eef1")
        assert abs(e["solvation"]) > 1.0, "EEF1 solvation should be significant"

    def test_amber_vs_charmm_different(self):
        s = proteon.load(CRAMBIN)
        e_amber = proteon.compute_energy(s, ff="amber96")
        e_charmm = proteon.compute_energy(s, ff="charmm19_eef1")
        assert e_amber["total"] != e_charmm["total"]

    def test_unknown_ff_raises(self):
        s = proteon.load(CRAMBIN)
        with pytest.raises(ValueError, match="Unknown force field"):
            proteon.compute_energy(s, ff="invalid")

    def test_typo_ff_raises(self):
        s = proteon.load(CRAMBIN)
        with pytest.raises(ValueError):
            proteon.compute_energy(s, ff="charmm19-eef1")  # dash not underscore

    def test_improper_torsion_nonzero(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="amber96")
        assert e["improper_torsion"] > 0, "Crambin should have improper torsion energy"

    def test_energy_components_sum_to_total(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="amber96")
        expected = (
            e["bond_stretch"] + e["angle_bend"] + e["torsion"]
            + e["improper_torsion"] + e["vdw"] + e["electrostatic"] + e["solvation"]
        )
        assert abs(e["total"] - expected) < 0.01

    def test_charmm_components_sum_to_total(self):
        s = proteon.load(CRAMBIN)
        e = proteon.compute_energy(s, ff="charmm19_eef1")
        expected = (
            e["bond_stretch"] + e["angle_bend"] + e["torsion"]
            + e["improper_torsion"] + e["vdw"] + e["electrostatic"] + e["solvation"]
        )
        assert abs(e["total"] - expected) < 0.01


# =========================================================================
# run_md()
# =========================================================================


class TestRunMD:
    def test_returns_dict(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=10, dt=0.0005, snapshot_freq=5)
        assert isinstance(result, dict)
        assert "coords" in result
        assert "velocities" in result
        assert "trajectory" in result
        assert "energy" in result

    def test_coord_shape(self):
        s = proteon.load(CRAMBIN)
        n_atoms = s.atom_count
        result = proteon.run_md(s, n_steps=5, dt=0.0005, snapshot_freq=5)
        assert result["coords"].shape == (n_atoms, 3)
        assert result["velocities"].shape == (n_atoms, 3)

    def test_trajectory_frames(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=20, dt=0.0005, snapshot_freq=10)
        assert len(result["trajectory"]) >= 2  # at least start + end
        frame = result["trajectory"][0]
        assert "step" in frame
        assert "temperature" in frame
        assert "total_energy" in frame

    def test_nvt_temperature_positive(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=10, dt=0.0005, temperature=300.0, thermostat_tau=0.2)
        last_temp = result["trajectory"][-1]["temperature"]
        assert last_temp > 0
        assert np.isfinite(last_temp)

    def test_nve_mode(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=5, dt=0.0005, thermostat_tau=0.0)
        assert len(result["trajectory"]) >= 2

    def test_shake_mode(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=5, dt=0.001, shake=True)
        assert result["coords"].shape[0] == s.atom_count
        assert np.all(np.isfinite(result["coords"]))

    def test_energy_dict(self):
        s = proteon.load(CRAMBIN)
        result = proteon.run_md(s, n_steps=5, dt=0.0005, snapshot_freq=5)
        energy = result["energy"]
        assert "bond_stretch" in energy
        assert "torsion" in energy
        assert "vdw" in energy
        assert "solvation" in energy
