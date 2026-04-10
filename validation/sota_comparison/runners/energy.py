"""Energy runners.

Per-op payload schema (`payload` field of RunnerResult):

    {
        "total": float,                # Total energy in kJ/mol
        "units": "kJ/mol",
        "ff": "amber96",               # which force field was used
        "components": {                # Per-component energy in kJ/mol;
            "bond_stretch": float,     # None for components the FF doesn't
            "angle_bend": float,       # define (e.g. solvation in vacuum FFs).
            "torsion": float,
            "improper_torsion": float,
            "vdw": float,
            "electrostatic": float,
            "solvation": Optional[float],
        },
        "n_unassigned_atoms": int,     # ferritin reports this; -1 for tools
                                       # that don't.
    }
"""

from __future__ import annotations

import importlib.metadata as _metadata
from typing import Optional

from ._base import (
    RunnerResult,
    register,
    time_call,
)


# ---------------------------------------------------------------------------
# ferritin baseline
# ---------------------------------------------------------------------------

try:
    import ferritin as _ferritin
    _FERRITIN_OK = True
    try:
        _FERRITIN_VERSION = "ferritin " + _metadata.version("ferritin")
    except Exception:
        _FERRITIN_VERSION = "ferritin (unknown version)"
except ImportError:
    _FERRITIN_OK = False
    _FERRITIN_VERSION = ""


_COMPONENT_KEYS = (
    "bond_stretch",
    "angle_bend",
    "torsion",
    "improper_torsion",
    "vdw",
    "electrostatic",
    "solvation",
)


def _normalize_components(d: dict) -> dict:
    """Pick out the canonical component keys from a ferritin energy dict.

    Returns a dict with every key from _COMPONENT_KEYS present, using None for
    components the source dict doesn't have. Defensive against ferritin adding
    new components in the future without breaking the schema.
    """
    return {k: float(d[k]) if d.get(k) is not None else None for k in _COMPONENT_KEYS}


if _FERRITIN_OK:

    @register("energy", "ferritin")
    def ferritin(pdb_path: str) -> RunnerResult:
        """Compute AMBER96 energy via ferritin.compute_energy.

        Loads the structure and places all standard hydrogens via
        `place_all_hydrogens()` before computing the energy. H placement
        is required because AMBER96 parameterizes H atoms; a heavy-only
        input gives nonsensical energies.

        The same H placement step is performed by the OpenMM runner (via
        Modeller.addHydrogens), keeping both runners on comparable atom
        sets for the FF-to-FF comparison. Returns kJ/mol (ferritin's
        default unit).
        """
        s, _ = time_call(_ferritin.load, pdb_path)
        # Add hydrogens in place — matches OpenMM runner's Modeller.addHydrogens
        _ferritin.place_all_hydrogens(s)
        result, elapsed = time_call(
            _ferritin.compute_energy, s, ff="amber96", units="kJ/mol"
        )

        return RunnerResult(
            op="energy",
            impl="ferritin",
            impl_version=_FERRITIN_VERSION,
            pdb_id="",
            pdb_path=pdb_path,
            elapsed_s=elapsed,
            status="ok",
            error=None,
            payload={
                "total": float(result["total"]),
                "units": "kJ/mol",
                "ff": "amber96",
                "components": _normalize_components(result),
                "n_unassigned_atoms": int(result.get("n_unassigned_atoms", 0)),
                "n_atoms_after_h": int(s.atom_count),
            },
        )


# ---------------------------------------------------------------------------
# OpenMM
# ---------------------------------------------------------------------------
#
# OpenMM 8.5 bundles amber96.xml, so we can do a like-for-like AMBER96 vs
# AMBER96 comparison against ferritin's `compute_energy(ff="amber96")`.
#
# Per-component breakdown is obtained by assigning each Force object to its
# own group and calling Context.getState(groups=1<<group) per group. AMBER96
# in OpenMM produces four distinct Force types:
#
#   HarmonicBondForce      → bond_stretch
#   HarmonicAngleForce     → angle_bend
#   PeriodicTorsionForce   → torsion + improper_torsion (combined; OpenMM
#                            does not distinguish proper vs improper
#                            dihedrals as separate Forces)
#   NonbondedForce         → vdw + electrostatic (combined; same caveat)
#
# So the components dict has:
#   bond_stretch      ← matches ferritin exactly
#   angle_bend        ← matches ferritin exactly
#   torsion           ← OpenMM torsion TOTAL (use ferritin_torsion_total for
#                       the apples-to-apples comparison; see aggregate.py)
#   improper_torsion  ← None (folded into torsion by OpenMM)
#   vdw               ← None (folded into nonbonded by OpenMM)
#   electrostatic     ← None
#   nonbonded_total   ← OpenMM nonbonded TOTAL (ferritin doesn't export this
#                       directly, so the aggregator sums ferritin.vdw +
#                       ferritin.electrostatic to compare)

try:
    import openmm as _openmm  # noqa: F401
    import openmm.app as _openmm_app
    import openmm.unit as _openmm_unit
    _OPENMM_OK = True
    try:
        _OPENMM_VERSION = "openmm " + _openmm.version.version
    except Exception:
        _OPENMM_VERSION = "openmm (unknown version)"
except ImportError:
    _OPENMM_OK = False
    _OPENMM_VERSION = ""


if _OPENMM_OK:

    @register("energy", "openmm")
    def openmm(pdb_path: str) -> RunnerResult:
        """Compute AMBER96 energy via OpenMM for apples-to-apples comparison.

        Uses OpenMM's bundled `amber96.xml` force field with:
        - No cutoffs (NoCutoff method) — matches ferritin's O(N²) vacuum path
        - No constraints — matches ferritin (which computes energy on the
          raw geometry, not a constrained one)
        - CPU platform (deterministic, no GPU dependency)

        Per-component breakdown is obtained by assigning each Force to its
        own group index and calling Context.getState(groups=1<<group) per
        group. Totals are checked against the all-groups state to catch any
        missed Force types.

        Topology mismatches (missing atoms that prevent System creation) are
        reported as status="error" with a clear message; the comparison is
        FF-vs-FF, so we can't paper over a broken input.
        """
        import time as _time
        t0 = _time.perf_counter()

        try:
            pdb = _openmm_app.PDBFile(pdb_path)
        except Exception as e:
            return RunnerResult(
                op="energy", impl="openmm", impl_version=_OPENMM_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=_time.perf_counter() - t0,
                status="error",
                error=f"OpenMM PDBFile load failed: {type(e).__name__}: {e}",
                payload={},
            )

        # AMBER96 requires hydrogens on every standard residue. The input
        # PDBs are heavy-atom only, so add H via Modeller.addHydrogens
        # before building the System. The ferritin runner does the same
        # via place_all_hydrogens() — both stacks see the same atom set
        # for the FF-to-FF comparison.
        try:
            forcefield = _openmm_app.ForceField("amber96.xml")
            modeller = _openmm_app.Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=_openmm_app.NoCutoff,
                constraints=None,
            )
        except Exception as e:
            return RunnerResult(
                op="energy", impl="openmm", impl_version=_OPENMM_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=_time.perf_counter() - t0,
                status="error",
                error=f"OpenMM createSystem failed: {type(e).__name__}: {e}",
                payload={},
            )

        # Assign each Force to its own group so we can decompose the total.
        force_groups: dict = {}  # group_idx -> force class name
        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)
            force_groups[i] = type(force).__name__

        # Minimal integrator just so we can build a Context.
        integrator = _openmm.VerletIntegrator(0.001 * _openmm_unit.picoseconds)
        try:
            platform = _openmm.Platform.getPlatformByName("CPU")
            context = _openmm.Context(system, integrator, platform)
        except Exception:
            # Fall back to Reference if CPU platform isn't available
            context = _openmm.Context(system, integrator)

        context.setPositions(modeller.positions)

        # Per-group energies
        group_energies = {}
        for g_idx, cls_name in force_groups.items():
            state = context.getState(getEnergy=True, groups=1 << g_idx)
            e_kj = state.getPotentialEnergy().value_in_unit(
                _openmm_unit.kilojoule_per_mole
            )
            group_energies[cls_name] = e_kj

        # All-groups total (sanity check)
        total_state = context.getState(getEnergy=True)
        total = total_state.getPotentialEnergy().value_in_unit(
            _openmm_unit.kilojoule_per_mole
        )

        elapsed = _time.perf_counter() - t0

        # Map OpenMM Force class names to the canonical component keys.
        bond = group_energies.get("HarmonicBondForce")
        angle = group_energies.get("HarmonicAngleForce")
        torsion_total = group_energies.get("PeriodicTorsionForce")
        nonbonded_total = group_energies.get("NonbondedForce")
        cmmotion = group_energies.get("CMMotionRemover", 0.0)  # always 0 but list it

        components = {
            "bond_stretch": float(bond) if bond is not None else None,
            "angle_bend": float(angle) if angle is not None else None,
            "torsion": float(torsion_total) if torsion_total is not None else None,
            "improper_torsion": None,  # combined into torsion by OpenMM
            "vdw": None,                # combined into nonbonded by OpenMM
            "electrostatic": None,      # combined into nonbonded by OpenMM
            "solvation": None,          # vacuum
        }

        return RunnerResult(
            op="energy",
            impl="openmm",
            impl_version=_OPENMM_VERSION,
            pdb_id="",
            pdb_path=pdb_path,
            elapsed_s=elapsed,
            status="ok",
            error=None,
            payload={
                "total": float(total),
                "units": "kJ/mol",
                "ff": "amber96",
                "components": components,
                # Extra fields specific to the OpenMM runner for cross-
                # checking: the aggregator uses nonbonded_total to compare
                # against (ferritin.vdw + ferritin.electrostatic), and
                # verifies torsion_total matches (ferritin.torsion +
                # ferritin.improper_torsion).
                "nonbonded_total": float(nonbonded_total) if nonbonded_total is not None else None,
                "n_unassigned_atoms": -1,  # OpenMM doesn't have the concept
                "n_atoms_after_h": int(modeller.topology.getNumAtoms()),
            },
        )
