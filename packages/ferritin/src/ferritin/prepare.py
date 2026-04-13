"""Structure preparation pipeline.

Composes fragment reconstruction, hydrogen placement, and energy
minimization into a single reliable workflow.

Functions:
    prepare              — full prep on a single structure
    batch_prepare        — parallel prep on many structures
    load_and_prepare     — load + prep in one call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    import ferritin_connector
    _add_h = ferritin_connector.py_add_hydrogens
    _ff = ferritin_connector.py_forcefield
except ImportError:
    _add_h = None
    _ff = None

# Re-use the same once-per-process AMBER96 warning machinery as forcefield.py.
from .forcefield import _maybe_warn_ff  # noqa: E402


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


# Rust batch_prepare returns energies in kcal/mol (the native unit of the
# AMBER96/CHARMM19 parameters). The rest of the ferritin Python API defaults
# to kJ/mol via compute_energy / minimize_hydrogens, so the PrepReport
# dataclass also exposes kJ/mol by default and we convert on the way out.
_KCAL_TO_KJ = 4.184


def _convert_prep_result_to_kj(r: dict) -> dict:
    """Return a copy of the Rust batch_prepare result dict with energy fields
    converted from kcal/mol to kJ/mol.

    Touches: initial_energy, final_energy, and every component in
    ``components`` (bond_stretch, angle_bend, …). Leaves structural counts
    (minimizer_steps, n_unassigned_atoms, …) unchanged.
    """
    out = dict(r)
    for k in ("initial_energy", "final_energy"):
        if k in out and isinstance(out[k], (int, float)):
            out[k] = out[k] * _KCAL_TO_KJ
    comps = out.get("components")
    if isinstance(comps, dict):
        out["components"] = {
            k: (v * _KCAL_TO_KJ if isinstance(v, (int, float)) else v)
            for k, v in comps.items()
        }
    return out


@dataclass
class PrepReport:
    """Report from structure preparation.

    Attributes:
        atoms_reconstructed: Heavy atoms added by fragment reconstruction.
        hydrogens_added: Hydrogen atoms placed.
        hydrogens_skipped: Residues where H placement was skipped (e.g., missing backbone).
        initial_energy: Total energy before minimization (kcal/mol).
        final_energy: Total energy after minimization (kcal/mol).
        components: Per-component energy breakdown at the post-minimization
            geometry, in the same units as ``initial_energy`` /
            ``final_energy``. Keys: ``bond_stretch``, ``angle_bend``,
            ``torsion``, ``improper_torsion``, ``vdw``, ``electrostatic``,
            ``solvation``. All zero if ``minimize=False`` was passed, or if
            ``skipped_no_protein`` is True. Populated from the minimizer's
            final energy state — does NOT require a separate
            ``compute_energy`` call.
        minimizer_steps: Number of minimization steps taken.
        converged: Whether the minimizer converged. Only meaningful when
            ``skipped_no_protein`` is False — see that field's docs.
        n_unassigned_atoms: Atoms without force field type assignment.
        skipped_no_protein: True if the structure was skipped by the
            minimizer because more than half of its atoms have no protein
            force-field type assignment (e.g. nucleic acids, ligand-only
            entries, exotic non-standard residues). When True, ``converged``
            is always False because no minimization ran — distinguish this
            case from real convergence failures by checking
            ``skipped_no_protein`` first.
        warnings: List of warning messages.
    """
    atoms_reconstructed: int = 0
    hydrogens_added: int = 0
    hydrogens_skipped: int = 0
    initial_energy: float = 0.0
    final_energy: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    minimizer_steps: int = 0
    converged: bool = False
    n_unassigned_atoms: int = 0
    skipped_no_protein: bool = False
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"PrepReport(",
            f"  reconstructed={self.atoms_reconstructed} heavy atoms",
            f"  hydrogens={self.hydrogens_added} added, {self.hydrogens_skipped} skipped",
        ]
        if self.skipped_no_protein:
            lines.append(
                f"  skipped_no_protein=True ({self.n_unassigned_atoms} unassigned atoms)"
            )
        else:
            lines.append(
                f"  energy={self.initial_energy:.1f} -> {self.final_energy:.1f} kcal/mol"
            )
            lines.append(
                f"  minimizer={self.minimizer_steps} steps, converged={self.converged}"
            )
            if self.n_unassigned_atoms > 0:
                lines.append(f"  unassigned_atoms={self.n_unassigned_atoms}")
        if self.warnings:
            lines.append(f"  warnings={self.warnings}")
        lines.append(")")
        return "\n".join(lines)


def prepare(
    structure,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    include_water: bool = False,
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
    gradient_tolerance: float = 0.1,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
) -> PrepReport:
    """Prepare a structure for downstream analysis or simulation.

    Pipeline: [strip H] -> reconstruct missing atoms -> place hydrogens
    -> minimize H positions.

    Args:
        structure: Ferritin Structure object (modified in place).
        reconstruct: Add missing heavy atoms from fragment templates (default True).
        hydrogens: Hydrogen placement strategy:
            "backbone" — backbone amide N-H only
            "all"      — backbone + sidechain (standard AA, default)
            "general"  — all atoms including ligands and non-standard residues
            "none"     — skip hydrogen placement
        include_water: Place H on water molecules (only with hydrogens="general").
        minimize: Minimize hydrogen positions after placement (default True).
        minimize_method: Minimizer: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Maximum minimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
        strip_hydrogens: Remove all pre-existing H/D atoms before placement
            (default True). The default rescues structures with externally-
            placed hydrogens (NMR ensembles, deposited X-ray H, upstream
            protonators) whose positions are off the MM minimum and otherwise
            prevent LBFGS convergence within ``gradient_tolerance``. Set to
            False to retain experimental H positions when their provenance
            is trusted. See batch_prepare docstring for the rescue analysis.
        ff: Force field used by the topology builder and the minimizer.
            ``"charmm19_eef1"`` (default) is the **validated production
            path** — used by the 50K battle test and the fold-preservation
            benchmark. ``"amber96"`` is **experimental**; ferritin's AMBER96
            energies do NOT match OpenMM AMBER96 on identical inputs
            (diagnosed 2026-04-13). Emits a UserWarning once per process.
            Do not use for cross-tool oracle comparison; see task #46.

    Returns:
        PrepReport with preparation statistics.

    Examples:
        >>> import ferritin
        >>> s = ferritin.load("structure.pdb")
        >>> report = ferritin.prepare(s)
        >>> print(report)
    """
    ptr = _get_ptr(structure)
    report = PrepReport()
    _maybe_warn_ff(ff)

    # Step 0: Optionally strip existing hydrogens.
    if strip_hydrogens:
        # Use the batch path so strip+reconstruct+place+minimize all run in
        # one Rust call (cleaner and consistent with batch_prepare).
        results = _add_h.batch_prepare(
            [ptr], reconstruct, hydrogens, include_water,
            minimize, minimize_method, minimize_steps, gradient_tolerance, None,
            True,
            ff,
        )
        if results:
            r = _convert_prep_result_to_kj(results[0])
            report.atoms_reconstructed = r["atoms_reconstructed"]
            report.hydrogens_added = r["hydrogens_added"]
            report.hydrogens_skipped = r["hydrogens_skipped"]
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.components = dict(r.get("components", {}))
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]
            report.n_unassigned_atoms = r["n_unassigned_atoms"]
            report.skipped_no_protein = r["skipped_no_protein"]
            if report.skipped_no_protein:
                report.warnings.append(
                    f"skipped: {report.n_unassigned_atoms} atoms have no "
                    "protein force-field type (likely nucleic acid, ligand, "
                    "or non-standard residue)"
                )
            elif report.n_unassigned_atoms > 10:
                report.warnings.append(
                    f"{report.n_unassigned_atoms} atoms without force field type "
                    "(non-standard residues or ligands)"
                )
        return report

    # Step 1: Reconstruct missing heavy atoms
    if reconstruct:
        report.atoms_reconstructed = _add_h.reconstruct_fragments(ptr)

    # Step 2: Place hydrogens
    if hydrogens == "backbone":
        added, skipped = _add_h.place_peptide_hydrogens(ptr)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped
    elif hydrogens == "all":
        added, skipped = _add_h.place_all_hydrogens(ptr)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped
    elif hydrogens == "general":
        added, skipped = _add_h.place_general_hydrogens(ptr, include_water)
        report.hydrogens_added = added
        report.hydrogens_skipped = skipped
    elif hydrogens != "none":
        report.warnings.append(f"Unknown hydrogens option '{hydrogens}', skipping")

    # Step 3: Minimize hydrogen positions
    # Use batch_prepare for a single structure so coords are applied back in Rust.
    if minimize and report.hydrogens_added > 0:
        # Run minimization via the Rust batch path (applies coords in-place)
        results = _add_h.batch_prepare(
            [ptr], False, "none", False,
            True, minimize_method, minimize_steps, gradient_tolerance, None,
            False,
            ff,
        )
        if results:
            r = _convert_prep_result_to_kj(results[0])
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.components = dict(r.get("components", {}))
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]
            report.skipped_no_protein = r["skipped_no_protein"]

    # Step 4: Check force field coverage
    energy_result = _ff.compute_energy(ptr, "amber96")
    report.n_unassigned_atoms = energy_result.get("n_unassigned_atoms", 0)
    if report.n_unassigned_atoms > 10:
        report.warnings.append(
            f"{report.n_unassigned_atoms} atoms without force field type "
            "(non-standard residues or ligands)"
        )

    return report


def batch_prepare(
    structures: Sequence,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    include_water: bool = False,
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
    gradient_tolerance: float = 0.1,
    n_threads: Optional[int] = None,
    strip_hydrogens: bool = True,
    ff: str = "charmm19_eef1",
    constrain_heavy: Optional[bool] = None,
) -> List[PrepReport]:
    """Prepare many structures in parallel (Rust + rayon, zero GIL).

    Each structure is modified in place. Full pipeline runs in Rust:
    [optional strip H] -> reconstruct -> place H -> minimize H,
    parallelized across structures.

    Args:
        structures: List of ferritin Structure objects.
        reconstruct: Add missing heavy atoms (default True).
        hydrogens: "backbone", "all", "general", or "none" (default "all").
        include_water: Place H on water (only with hydrogens="general").
        minimize: Minimize H positions (default True).
        minimize_method: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Max minimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.
        strip_hydrogens: Remove all pre-existing H/D atoms before placement
            (default True). The default rescues structures with externally-
            placed hydrogens (NMR ensembles, deposited X-ray H, upstream
            protonators) whose positions are off the MM force-field minimum
            and would otherwise prevent LBFGS from converging within
            ``gradient_tolerance``. On the 50K benchmark this raised the
            convergence rate from 169/200 to 199/200 and cut wall time ~3x
            (stragglers stop burning the LBFGS step cap). Set to False to
            retain experimental H positions when their provenance is trusted.
        ff: Force field used by the topology builder and the minimizer.
            ``"charmm19_eef1"`` (default) is the **validated production
            path** — used by the 50K battle test and the fold-preservation
            benchmark. ``"amber96"`` is **experimental**; ferritin's AMBER96
            energies do NOT match OpenMM AMBER96 on identical inputs
            (diagnosed 2026-04-13). Emits a UserWarning once per process.
            Do not use for cross-tool oracle comparison; see task #46.
        constrain_heavy: Whether to freeze heavy atoms during minimization.
            ``None`` (default) uses the FF-aware default: True for AMBER96
            (H-only minimization is the intended pattern — all-atom AMBER
            in vacuum has unscreened electrostatic issues, so full minimization
            gives meaningless numbers), False for CHARMM19+EEF1 (polar-H
            united-atom with inflated carbon radii needs heavy-atom relaxation
            for correctly-signed totals). Pass ``True`` or ``False`` to
            override the default explicitly. Primarily useful for testing,
            profiling, or when you specifically want to preserve
            experimentally-determined heavy-atom geometry.

    Returns:
        List of PrepReport, one per structure.
    """
    _maybe_warn_ff(ff)
    ptrs = [_get_ptr(s) for s in structures]
    raw_results = _add_h.batch_prepare(
        ptrs, reconstruct, hydrogens, include_water,
        minimize, minimize_method, minimize_steps, gradient_tolerance, n_threads,
        strip_hydrogens,
        ff,
        constrain_heavy,
    )
    reports = []
    for raw in raw_results:
        r = _convert_prep_result_to_kj(raw)
        report = PrepReport(
            atoms_reconstructed=r["atoms_reconstructed"],
            hydrogens_added=r["hydrogens_added"],
            hydrogens_skipped=r["hydrogens_skipped"],
            initial_energy=r["initial_energy"],
            final_energy=r["final_energy"],
            components=dict(r.get("components", {})),
            minimizer_steps=r["minimizer_steps"],
            converged=r["converged"],
            n_unassigned_atoms=r["n_unassigned_atoms"],
            skipped_no_protein=r["skipped_no_protein"],
        )
        if report.skipped_no_protein:
            report.warnings.append(
                f"skipped: {report.n_unassigned_atoms} atoms have no protein "
                "force-field type (likely nucleic acid, ligand, or non-standard residue)"
            )
        elif report.n_unassigned_atoms > 10:
            report.warnings.append(
                f"{report.n_unassigned_atoms} atoms without force field type"
            )
        reports.append(report)
    return reports


def load_and_prepare(
    path: str,
    *,
    reconstruct: bool = True,
    hydrogens: str = "all",
    minimize: bool = True,
    minimize_method: str = "lbfgs",
    minimize_steps: int = 500,
) -> "tuple[object, PrepReport]":
    """Load a structure file and prepare it in one call.

    Args:
        path: Path to PDB or mmCIF file.
        reconstruct: Add missing heavy atoms (default True).
        hydrogens: "backbone", "all", "general", or "none" (default "all").
        minimize: Minimize H positions (default True).
        minimize_method: "sd", "cg", or "lbfgs" (default "lbfgs").
        minimize_steps: Max minimization steps (default 500).

    Returns:
        (structure, PrepReport) tuple.
    """
    from .io import load
    structure = load(path)
    report = prepare(
        structure,
        reconstruct=reconstruct,
        hydrogens=hydrogens,
        minimize=minimize,
        minimize_method=minimize_method,
        minimize_steps=minimize_steps,
    )
    return structure, report
