"""CHARMM energy runners — Tier 2 weak oracle.

WHY THIS FILE EXISTS
--------------------
proteon's production default force field is CHARMM19+EEF1
(`batch_prepare(ff="charmm19_eef1")`), but the only existing tests for
it are smoke checks ("energy is not zero", "solvation magnitude > 1").
The well-validated FF in the repo is AMBER96 (against BALL Julia, 0.02%
match), which is NOT what users actually run.

This module is the Tier-2 piece of the CHARMM validation roadmap:
a deliberately WEAK cross-FF oracle that compares proteon's
CHARMM19+EEF1 against OpenMM's CHARMM36+OBC2. The two parameter sets
are different — CHARMM19 vs CHARMM36 differ in atomtypes, charges,
torsion parameters, and implicit-solvation model (EEF1 is the
Lazaridis-Karplus reference-energy method, OBC2 is the Onufriev-
Bashford-Case generalized-Born method) — so percent diffs would be
*expected* to be 30-100% on every component. Comparing them would
flag PASS/WARN/FAIL meaninglessly under the AMBER tolerance scheme.

Instead, the matching aggregator uses `compare_energy_weak`, a
direction-only comparator that catches:
  - NaN / Inf (numerical blowup)
  - Sign disagreement on the total or any large component (sign error
    in a proteon kernel)
  - log10(|f|/|o|) > 1 (≥ 10× scaling error — kernel computing the
    wrong number of contributions, missing a constant factor, etc.)
This is what "weak oracle" means: it cannot validate parameter
correctness, but it does catch the most common kind of regression
(broken kernel returning obviously wrong magnitudes).

For a STRONG CHARMM19+EEF1 oracle the only available source is BALL
C++ via standalone harness — that's Tier 3, deferred (BALL Python
bindings are dead per the user 2026-04-10).

Per-op payload schema is the same as the AMBER `energy` runner. The op
name is `energy_charmm` so the aggregator can route it to the weak
comparator without entangling with the AMBER96 path.
"""

from __future__ import annotations

import importlib.metadata as _metadata
from typing import List as _List

from ._base import RunnerResult, register, register_batch
from time import perf_counter as _time_perf
import os as _os
import tempfile as _tempfile

# Reuse the helpers + import guards from the AMBER runner module so we
# don't double-up the HETATM stripping logic, the proteon/openmm/pdbfixer
# import detection, or the per-input PDBFixer cache. The functions and
# flags imported here are module-level in `runners/energy.py` and live
# inside its `if _OPENMM_OK:` / `if _PDBFIXER_OK:` blocks (which are
# top-level under any spawn child).
from .energy import (  # noqa: F401  (re-export by side effect of import)
    _COMPONENT_KEYS,
    _normalize_components,
    _strip_hetatm_to_tempfile,
    _prep_input_for_amber96,
    _PREP_CACHE,
    _PROTEON_OK,
    _PROTEON_VERSION,
    _OPENMM_OK,
    _OPENMM_VERSION,
    _PDBFIXER_OK,
    _PDBFixer,
)

# Bring the live OpenMM modules in too. We import them under guard since
# this module is loaded even when openmm isn't installed.
if _OPENMM_OK:
    import openmm as _openmm  # type: ignore
    import openmm.app as _openmm_app  # type: ignore
    import openmm.unit as _openmm_unit  # type: ignore

if _PROTEON_OK:
    import proteon as _proteon  # type: ignore


# ---------------------------------------------------------------------------
# proteon (CHARMM19+EEF1)
# ---------------------------------------------------------------------------

if _PROTEON_OK:

    @register("energy_charmm", "proteon")
    def proteon(pdb_path: str) -> RunnerResult:
        """Compute CHARMM19+EEF1 energy via proteon's batch_prepare.

        Mirror of the AMBER96 proteon runner, with `ff="charmm19_eef1"`.
        Uses the same shared PDBFixer prep helper so this runner sees the
        same atom set as the OpenMM CHARMM36 runner — necessary for the
        weak comparison to be apples-to-apples-shaped (same input atoms,
        different parameter set).

        We keep the prep pipeline identical to the AMBER96 runner: HETATM
        strip + PDBFixer heavy-atom repair + LBFGS minimization with the
        500-step cap (P1c). The minimization gives both runners a
        physically reasonable starting geometry that doesn't blow up
        the nonbonded terms.
        """
        t0 = _time_perf()
        prep_path = _prep_input_for_amber96(pdb_path)
        s = _proteon.load(prep_path)
        reports = _proteon.batch_prepare(
            [s],
            reconstruct=False,
            hydrogens="all",
            minimize=True,
            minimize_method="lbfgs",
            minimize_steps=500,  # P1c straggler cap
            gradient_tolerance=0.1,
            strip_hydrogens=False,
            ff="charmm19_eef1",
        )
        elapsed = _time_perf() - t0
        r = reports[0]
        return RunnerResult(
            op="energy_charmm",
            impl="proteon",
            impl_version=_PROTEON_VERSION,
            pdb_id="",
            pdb_path=pdb_path,
            elapsed_s=elapsed,
            status="ok",
            error=None,
            payload={
                "total": float(r.final_energy),
                "units": "kJ/mol",
                "ff": "charmm19_eef1",
                "components": _normalize_components(r.components),
                "n_atoms_after_h": int(s.atom_count),
                "minimizer_steps": int(r.minimizer_steps),
                "minimizer_converged": bool(r.converged),
                "initial_energy": float(r.initial_energy),
            },
        )

    @register_batch("energy_charmm", "proteon")
    def proteon_batch(pdb_paths: _List[str]) -> _List[RunnerResult]:
        """Batched proteon CHARMM19+EEF1 runner.

        Same shape as the AMBER96 batched runner: shared prep, parallel
        load, single rayon-parallel batch_prepare call across all
        structures, results stamped per-path. The only differences from
        the AMBER96 batched runner are `ff="charmm19_eef1"` and the
        `op` field on the RunnerResult.
        """
        t0 = _time_perf()
        prep_paths = [_prep_input_for_amber96(p) for p in pdb_paths]
        loaded = _proteon.batch_load_tolerant(prep_paths, n_threads=-1)
        load_index = {i: s for i, s in loaded}

        pos_map: _List[int] = []
        structs_to_prep = []
        for i in range(len(pdb_paths)):
            if i in load_index:
                pos_map.append(i)
                structs_to_prep.append(load_index[i])

        if structs_to_prep:
            reports = _proteon.batch_prepare(
                structs_to_prep,
                reconstruct=False,
                hydrogens="all",
                minimize=True,
                minimize_method="lbfgs",
                minimize_steps=500,
                gradient_tolerance=0.1,
                strip_hydrogens=False,
                ff="charmm19_eef1",
                n_threads=-1,
            )
        else:
            reports = []

        out: _List[RunnerResult] = []
        batch_wall = _time_perf() - t0
        for i, pdb_path in enumerate(pdb_paths):
            if i not in load_index:
                out.append(RunnerResult(
                    op="energy_charmm", impl="proteon",
                    impl_version=_PROTEON_VERSION,
                    pdb_id="", pdb_path=pdb_path, elapsed_s=0.0,
                    status="error",
                    error="batch_load_tolerant failed for this file",
                    payload={},
                ))
                continue
            pos = pos_map.index(i)
            r = reports[pos]
            s = structs_to_prep[pos]
            out.append(RunnerResult(
                op="energy_charmm", impl="proteon",
                impl_version=_PROTEON_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=0.0,
                status="ok", error=None,
                payload={
                    "total": float(r.final_energy),
                    "units": "kJ/mol",
                    "ff": "charmm19_eef1",
                    "components": _normalize_components(r.components),
                    "n_atoms_after_h": int(s.atom_count),
                    "minimizer_steps": int(r.minimizer_steps),
                    "minimizer_converged": bool(r.converged),
                    "initial_energy": float(r.initial_energy),
                },
            ))
        if out and out[0].status == "ok":
            out[0].elapsed_s = batch_wall
        return out


# ---------------------------------------------------------------------------
# OpenMM (CHARMM36 + OBC2 implicit solvation)
# ---------------------------------------------------------------------------
#
# OpenMM's CHARMM36 force field combined with OBC2 implicit solvation is the
# closest readily-available analog to proteon's CHARMM19+EEF1. Differences:
#
#   - CHARMM19 (1980s, polar-H model) vs CHARMM36 (2012, all-atom)
#     => atomtypes are different, K_b and K_theta values differ, charges
#        differ, especially on backbone and aromatic sidechains.
#   - EEF1 (Lazaridis-Karplus 1999, reference-energy implicit solvation)
#     vs OBC2 (Onufriev-Bashford-Case 2004, generalized-Born). EEF1 sums
#     per-atom solvation free energies; OBC2 evaluates a GB sum. They have
#     different functional forms; magnitudes don't agree but signs usually
#     do for typical proteins.
#   - CHARMM36 has a CMAP backbone correction term (φ/ψ cross term) that
#     CHARMM19 lacks. This shows up in OpenMM's CMAPTorsionForce and
#     contributes a non-trivial energy that proteon's torsion bucket
#     does not include.
#
# So percent diffs are misleading on this comparison. The aggregator's
# weak comparator uses log10 ratios + sign agreement instead.

if _OPENMM_OK:

    @register("energy_charmm", "openmm_charmm36")
    def openmm_charmm36(pdb_path: str) -> RunnerResult:
        """Compute CHARMM36+OBC2 energy via OpenMM as a weak CHARMM oracle.

        Pipeline:
          1. Shared PDBFixer prep (same atom set as the proteon runner).
          2. Load via PDBFile.
          3. Modeller.addHydrogens(forcefield) — CHARMM36 H placement.
          4. createSystem with charmm36.xml + implicit/obc2.xml.
          5. Freeze heavy atoms, LocalEnergyMinimizer to relax H.
          6. Per-Force-group energy decomposition.

        Force-group → component mapping for CHARMM36 + OBC2:
            HarmonicBondForce      → bond_stretch
            HarmonicAngleForce     → angle_bend
            PeriodicTorsionForce   → torsion (proper + improper combined)
            CMAPTorsionForce       → cmap (CHARMM-specific, lumped into
                                     torsion below for the comparison)
            NonbondedForce         → nonbonded_total (vdw+elec combined)
            CustomGBForce          → solvation (OBC2 implicit)

        Failures during topology prep / System creation become
        status="skip" (not "error") so the aggregator surfaces a skip
        rate rather than a crash count — same convention as the
        AMBER96 runner.
        """
        import time as _time
        t0 = _time.perf_counter()

        prep_path = _prep_input_for_amber96(pdb_path)

        try:
            pdb = _openmm_app.PDBFile(prep_path)
        except Exception as e:
            return RunnerResult(
                op="energy_charmm", impl="openmm_charmm36",
                impl_version=_OPENMM_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=_time.perf_counter() - t0,
                status="error",
                error=f"OpenMM PDBFile load failed: {type(e).__name__}: {e}",
                payload={},
            )

        try:
            forcefield = _openmm_app.ForceField(
                "charmm36.xml", "implicit/obc2.xml"
            )
            modeller = _openmm_app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=_openmm_app.NoCutoff,
                constraints=None,
            )
        except Exception as e:
            return RunnerResult(
                op="energy_charmm", impl="openmm_charmm36",
                impl_version=_OPENMM_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=_time.perf_counter() - t0,
                status="skip",
                error=f"OpenMM topology prep failed: {type(e).__name__}: {e}",
                payload={},
            )

        # Per-Force-group decomposition. Same pattern as the AMBER runner.
        force_groups: dict = {}
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
            force_groups[i] = type(force).__name__

        integrator = _openmm.VerletIntegrator(0.001 * _openmm_unit.picoseconds)
        try:
            platform = _openmm.Platform.getPlatformByName("CPU")
            context = _openmm.Context(system, integrator, platform)
        except Exception:
            context = _openmm.Context(system, integrator)

        context.setPositions(modeller.positions)

        # Freeze heavy atoms, minimize H positions only — same idiom as
        # the AMBER runner. mass=0 → no displacement during minimization.
        for i, atom in enumerate(modeller.topology.atoms()):
            if atom.element is None or atom.element.symbol != "H":
                system.setParticleMass(i, 0.0)
        context.reinitialize(preserveState=True)

        try:
            _openmm.LocalEnergyMinimizer.minimize(
                context, tolerance=0.1, maxIterations=2000
            )
        except Exception as e:
            return RunnerResult(
                op="energy_charmm", impl="openmm_charmm36",
                impl_version=_OPENMM_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=_time.perf_counter() - t0,
                status="error",
                error=f"OpenMM LocalEnergyMinimizer failed: {type(e).__name__}: {e}",
                payload={},
            )

        # Per-group energies
        group_energies = {}
        for g_idx, cls_name in force_groups.items():
            state = context.getState(getEnergy=True, groups=1 << g_idx)
            e_kj = state.getPotentialEnergy().value_in_unit(
                _openmm_unit.kilojoule_per_mole
            )
            group_energies[cls_name] = e_kj

        total_state = context.getState(getEnergy=True)
        total = total_state.getPotentialEnergy().value_in_unit(
            _openmm_unit.kilojoule_per_mole
        )

        elapsed = _time.perf_counter() - t0

        # Force class → component key mapping. Notes:
        # - PeriodicTorsionForce holds proper + improper for CHARMM36.
        # - CMAPTorsionForce is the CHARMM-specific φ/ψ correction; we
        #   add it into the torsion bucket for the weak comparison
        #   because proteon's CHARMM19 lumps everything torsion-shaped
        #   into one slot anyway.
        # - CustomGBForce is OBC2 implicit solvation.
        bond = group_energies.get("HarmonicBondForce")
        angle = group_energies.get("HarmonicAngleForce")
        torsion_periodic = group_energies.get("PeriodicTorsionForce", 0.0)
        torsion_cmap = group_energies.get("CMAPTorsionForce", 0.0)
        nonbonded_total = group_energies.get("NonbondedForce")
        solvation = group_energies.get("CustomGBForce")

        components = {
            "bond_stretch": float(bond) if bond is not None else None,
            "angle_bend": float(angle) if angle is not None else None,
            "torsion": float(torsion_periodic + torsion_cmap),
            "improper_torsion": None,
            "vdw": None,
            "electrostatic": None,
            "solvation": float(solvation) if solvation is not None else None,
        }

        return RunnerResult(
            op="energy_charmm",
            impl="openmm_charmm36",
            impl_version=_OPENMM_VERSION,
            pdb_id="",
            pdb_path=pdb_path,
            elapsed_s=elapsed,
            status="ok",
            error=None,
            payload={
                "total": float(total),
                "units": "kJ/mol",
                "ff": "charmm36+obc2",
                "components": components,
                "nonbonded_total": (
                    float(nonbonded_total) if nonbonded_total is not None else None
                ),
                "torsion_cmap_kj": float(torsion_cmap),  # for diagnostics
                "n_unassigned_atoms": -1,
                "n_atoms_after_h": int(modeller.topology.getNumAtoms()),
            },
        )
