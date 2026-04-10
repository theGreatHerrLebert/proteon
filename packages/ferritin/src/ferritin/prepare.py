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


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


@dataclass
class PrepReport:
    """Report from structure preparation.

    Attributes:
        atoms_reconstructed: Heavy atoms added by fragment reconstruction.
        hydrogens_added: Hydrogen atoms placed.
        hydrogens_skipped: Residues where H placement was skipped (e.g., missing backbone).
        initial_energy: Total energy before minimization (kcal/mol).
        final_energy: Total energy after minimization (kcal/mol).
        minimizer_steps: Number of minimization steps taken.
        converged: Whether the minimizer converged.
        n_unassigned_atoms: Atoms without force field type assignment.
        warnings: List of warning messages.
    """
    atoms_reconstructed: int = 0
    hydrogens_added: int = 0
    hydrogens_skipped: int = 0
    initial_energy: float = 0.0
    final_energy: float = 0.0
    minimizer_steps: int = 0
    converged: bool = False
    n_unassigned_atoms: int = 0
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"PrepReport(",
            f"  reconstructed={self.atoms_reconstructed} heavy atoms",
            f"  hydrogens={self.hydrogens_added} added, {self.hydrogens_skipped} skipped",
            f"  energy={self.initial_energy:.1f} -> {self.final_energy:.1f} kcal/mol",
            f"  minimizer={self.minimizer_steps} steps, converged={self.converged}",
        ]
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
) -> PrepReport:
    """Prepare a structure for downstream analysis or simulation.

    Pipeline: reconstruct missing atoms -> place hydrogens -> minimize H positions.

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
        )
        if results:
            r = results[0]
            report.initial_energy = r["initial_energy"]
            report.final_energy = r["final_energy"]
            report.minimizer_steps = r["minimizer_steps"]
            report.converged = r["converged"]

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
) -> List[PrepReport]:
    """Prepare many structures in parallel (Rust + rayon, zero GIL).

    Each structure is modified in place. Full pipeline runs in Rust:
    reconstruct -> place H -> minimize H, parallelized across structures.

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

    Returns:
        List of PrepReport, one per structure.
    """
    ptrs = [_get_ptr(s) for s in structures]
    results = _add_h.batch_prepare(
        ptrs, reconstruct, hydrogens, include_water,
        minimize, minimize_method, minimize_steps, gradient_tolerance, n_threads,
    )
    reports = []
    for r in results:
        report = PrepReport(
            atoms_reconstructed=r["atoms_reconstructed"],
            hydrogens_added=r["hydrogens_added"],
            hydrogens_skipped=r["hydrogens_skipped"],
            initial_energy=r["initial_energy"],
            final_energy=r["final_energy"],
            minimizer_steps=r["minimizer_steps"],
            converged=r["converged"],
            n_unassigned_atoms=r["n_unassigned_atoms"],
        )
        if report.n_unassigned_atoms > 10:
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
