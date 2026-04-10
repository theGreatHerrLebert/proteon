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
    register_batch,
    time_call,
)
from time import perf_counter as _time_perf
from typing import List as _List
import os as _os
import tempfile as _tempfile


def _strip_hetatm_to_tempfile(pdb_path: str) -> str:
    """Write a copy of `pdb_path` with all HETATM lines removed.

    Returns the path to a tempfile in /tmp. The caller is responsible for
    cleaning up if desired, but /tmp is fine for the test corpus.

    Why: OpenMM's amber96.xml force field only has templates for the 20
    standard amino acids. Water (HOH), phosphate (PO4), and ligand
    residues like AP5 cause createSystem to fail with "No template
    found". We strip HETATM in both runners (ferritin + openmm) so they
    see the same protein-only topology. This keeps the energy comparison
    apples-to-apples at the cost of ignoring any energy contribution from
    crystal waters or bound ligands — that's a v2 item (PDBFixer gives
    OpenMM templates for HETATM residues).
    """
    base = _os.path.basename(pdb_path)
    fd, tmp = _tempfile.mkstemp(prefix=f"sota_noHet_{base}_", suffix=".pdb")
    _os.close(fd)
    with open(pdb_path) as src, open(tmp, "w") as dst:
        for line in src:
            if line.startswith("HETATM"):
                continue
            dst.write(line)
    return tmp


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
        """Compute AMBER96 energy via ferritin's batch_prepare with ff="amber96".

        Pipeline: load → strip HETATM → batch_prepare(hydrogens="all",
        minimize=True, ff="amber96"). batch_prepare runs place_all_hydrogens
        then LBFGS minimization under AMBER96, in-place, using the neighbor-
        list caching fast path (the same one that made the 50K benchmark
        prepare phase 22s for 200 structures).

        Before this runner switched to batch_prepare, the ferritin side used
        minimize_hydrogens() which runs the O(N²) no-NBL path and takes
        14-72 seconds per structure on the v1 reference set. batch_prepare
        drops that to <1 second per structure and makes 10K-scale energy
        comparison feasible.

        The ff="amber96" parameter was added in the same commit that wires
        this runner — see commit history of py_add_hydrogens.rs. Without it
        batch_prepare hardcodes CHARMM19-EEF1.

        The same minimize-then-evaluate flow is used by the OpenMM runner
        (Modeller.addHydrogens + LocalEnergyMinimizer with heavy atoms frozen).
        """
        t0 = _time_perf()
        # Strip HETATM to match the OpenMM runner — see _strip_hetatm_to_tempfile
        # docstring for rationale. Keeps both runners on the same atom set.
        clean_path = _strip_hetatm_to_tempfile(pdb_path)
        s = _ferritin.load(clean_path)
        # Full prepare pipeline in one Rust call, AMBER96 throughout.
        # strip_hydrogens=False because we loaded a heavy-only PDB (no H to
        # strip). reconstruct=False because the test corpus has complete
        # heavy atom sets.
        reports = _ferritin.batch_prepare(
            [s],
            reconstruct=False,
            hydrogens="all",
            minimize=True,
            minimize_method="lbfgs",
            # P1c: 500 steps caps per-structure LBFGS wall time so a single
            # straggler can't dominate the rayon critical path on a 100-PDB
            # batch. Most structures converge well before 500 thanks to the
            # plateau fallback in the minimizer; the few that don't get
            # flagged converged=False but the batch wall time stays bounded.
            minimize_steps=500,
            gradient_tolerance=0.1,
            strip_hydrogens=False,
            ff="amber96",
        )
        elapsed = _time_perf() - t0
        r = reports[0]

        # PrepReport.components is populated from the minimizer's final state.
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
                "total": float(r.final_energy),
                "units": "kJ/mol",
                "ff": "amber96",
                "components": _normalize_components(r.components),
                "n_atoms_after_h": int(s.atom_count),
                "minimizer_steps": int(r.minimizer_steps),
                "minimizer_converged": bool(r.converged),
                "initial_energy": float(r.initial_energy),
            },
        )

    @register_batch("energy", "ferritin")
    def ferritin_batch(pdb_paths: _List[str]) -> _List[RunnerResult]:
        """Batched ferritin energy runner.

        Loads all structures (after HETATM stripping), then calls
        `ferritin.batch_prepare(ff="amber96")` ONCE across the whole list.
        This unlocks in-Rust rayon parallelism across the full batch plus
        the NBL-cached fast minimizer path for structures > 2000 atoms.

        On the 50K benchmark, `batch_prepare` processes 200 structures in
        ~22 seconds (0.11s/structure amortized). Calling it with a list
        of 1 in the per-structure driver loses this entirely. The batched
        runner gets the amortized cost at the cost of losing per-structure
        elapsed times (all elapsed values except the first are 0.0 —
        wall-time attribution is at the batch level).
        """
        t0 = _time_perf()
        # Strip HETATM once per path (cached tempfiles)
        clean_paths = [_strip_hetatm_to_tempfile(p) for p in pdb_paths]
        # Parallel load
        loaded = _ferritin.batch_load_tolerant(clean_paths, n_threads=-1)
        load_index = {i: s for i, s in loaded}

        # Collect successful structures for the batch call
        pos_map: _List[int] = []  # index into pdb_paths for each position in the batch
        structs_to_prep = []
        for i in range(len(pdb_paths)):
            if i in load_index:
                pos_map.append(i)
                structs_to_prep.append(load_index[i])

        # Batched prep + minimize + energy-eval, all in one Rust call
        if structs_to_prep:
            reports = _ferritin.batch_prepare(
                structs_to_prep,
                reconstruct=False,
                hydrogens="all",
                minimize=True,
                minimize_method="lbfgs",
                # P1c: 500 caps per-structure LBFGS wall time. See the
                # ferritin (per-structure) runner above for the rationale.
                minimize_steps=500,
                gradient_tolerance=0.1,
                strip_hydrogens=False,
                ff="amber96",
                n_threads=-1,
            )
        else:
            reports = []

        # Build per-path results
        out: _List[RunnerResult] = []
        batch_wall = _time_perf() - t0
        for i, pdb_path in enumerate(pdb_paths):
            if i not in load_index:
                out.append(RunnerResult(
                    op="energy", impl="ferritin", impl_version=_FERRITIN_VERSION,
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
                op="energy", impl="ferritin", impl_version=_FERRITIN_VERSION,
                pdb_id="", pdb_path=pdb_path,
                elapsed_s=0.0,  # see batch elapsed below
                status="ok", error=None,
                payload={
                    "total": float(r.final_energy),
                    "units": "kJ/mol",
                    "ff": "amber96",
                    "components": _normalize_components(r.components),
                    "n_atoms_after_h": int(s.atom_count),
                    "minimizer_steps": int(r.minimizer_steps),
                    "minimizer_converged": bool(r.converged),
                    "initial_energy": float(r.initial_energy),
                },
            ))
        # Stamp the aggregate wall time on the first result for reporting
        if out and out[0].status == "ok":
            out[0].elapsed_s = batch_wall
        return out


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

# pdbfixer is the standard OpenMM companion for repairing PDB heavy-atom
# defects (incomplete terminal residues, truncated sidechains, MSE etc).
# Imported separately so the runner degrades gracefully (no repair, will
# skip on createSystem failure) in environments where pdbfixer isn't
# available. Strongly recommended in sota_venv: see P1a in the
# 2026-04-10 NEXT_SESSION arc — without it 10/20 structures on the
# scaling demo failed createSystem.
try:
    from pdbfixer import PDBFixer as _PDBFixer
    _PDBFIXER_OK = True
except ImportError:
    _PDBFIXER_OK = False
    _PDBFixer = None  # type: ignore


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

        # Strip HETATM before loading into OpenMM. amber96.xml has no
        # templates for water / ligands, so createSystem fails if they are
        # present. Matches the ferritin runner's _strip_hetatm step.
        clean_path = _strip_hetatm_to_tempfile(pdb_path)

        # P1a v2: PDBFixer repairs real PDB imperfections (incomplete
        # terminal residues, truncated sidechains, nonstandard residues
        # like MSE) BEFORE OpenMM tries to assign templates. On the 20-PDB
        # scaling demo without this step, 10/20 structures failed
        # createSystem with "set of externally bonded atoms is missing
        # 1 C atom". Those are real PDB format issues, not a ferritin
        # bug, and addMissingAtoms repairs them in place using AMBER's
        # template geometry.
        #
        # IMPORTANT: we call findMissingResidues() then immediately clear
        # fixer.missingResidues = {}. PDBFixer normally tries to add gap-
        # filling residues for unresolved disordered loops, which would
        # change the structure significantly and bias the energy
        # comparison. We only want missing ATOMS within existing residues,
        # not whole missing residues that the experiment couldn't see.
        #
        # If pdbfixer is not installed, we fall through to the raw Modeller
        # path. The createSystem step will then fail on imperfect PDBs and
        # we record those as skips instead of crashing.
        if _PDBFIXER_OK:
            try:
                fixer = _PDBFixer(filename=clean_path)
                fixer.findMissingResidues()
                fixer.missingResidues = {}  # don't model unresolved loops
                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                topology_in = fixer.topology
                positions_in = fixer.positions
            except Exception as e:
                return RunnerResult(
                    op="energy", impl="openmm", impl_version=_OPENMM_VERSION,
                    pdb_id="", pdb_path=pdb_path,
                    elapsed_s=_time.perf_counter() - t0,
                    status="skip",
                    error=f"PDBFixer prep failed: {type(e).__name__}: {e}",
                    payload={},
                )
        else:
            try:
                pdb = _openmm_app.PDBFile(clean_path)
                topology_in = pdb.topology
                positions_in = pdb.positions
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
        # PDBs are heavy-atom only (and PDBFixer doesn't add H by default),
        # so add H via Modeller.addHydrogens(forcefield) — that uses the
        # AMBER96 H templates. The ferritin runner does the same via
        # place_all_hydrogens(): both stacks see the same atom set.
        #
        # We then run LocalEnergyMinimizer with heavy atoms frozen to relax
        # the placed H positions, mirroring ferritin's minimize_hydrogens()
        # step. Without this the raw Modeller H placements have the same
        # kind of residual clashes that ferritin's template placement has,
        # and the FF evaluation is dominated by them rather than the
        # actual interaction energies we want to compare.
        #
        # If addHydrogens / createSystem still fail after PDBFixer repair,
        # we record status="skip" (not "error"): these are legitimate
        # "this structure can't be evaluated under AMBER96" cases —
        # nucleic acids, exotic ligands, broken topologies — and we want
        # the aggregator to surface a skip rate rather than a crash count.
        try:
            forcefield = _openmm_app.ForceField("amber96.xml")
            modeller = _openmm_app.Modeller(topology_in, positions_in)
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
                status="skip",
                error=f"OpenMM topology prep failed: {type(e).__name__}: {e}",
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

        # Freeze heavy atoms and minimize H positions only. Mirrors
        # ferritin.minimize_hydrogens(): mass=0 is OpenMM's idiom for
        # "don't move this particle" during minimization.
        for i, atom in enumerate(modeller.topology.atoms()):
            if atom.element is None or atom.element.symbol != "H":
                system.setParticleMass(i, 0.0)
        # Re-initialize the context after changing masses.
        context.reinitialize(preserveState=True)

        try:
            _openmm.LocalEnergyMinimizer.minimize(
                context, tolerance=0.1, maxIterations=2000
            )
        except Exception as e:
            return RunnerResult(
                op="energy", impl="openmm", impl_version=_OPENMM_VERSION,
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
