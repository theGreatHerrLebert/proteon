"""AMBER force field energy computation and minimization.

Implements AMBER96 energy terms (bond stretch, angle bend, torsion,
Lennard-Jones, Coulomb) with steepest descent minimization.

Functions:
    compute_energy      — AMBER energy breakdown
    minimize_hydrogens  — optimize H positions (freeze heavy atoms)
    minimize_structure  — full energy minimization
"""

from __future__ import annotations

try:
    import ferritin_connector
    _ff = ferritin_connector.py_forcefield
except ImportError:
    _ff = None


def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


def compute_energy(structure) -> dict:
    """Compute AMBER96 force field energy of a structure.

    Returns dict with energy components (all in kcal/mol):
        bond_stretch, angle_bend, torsion, vdw, electrostatic, total

    Examples:
        >>> e = ferritin.compute_energy(structure)
        >>> print(f"Total: {e['total']:.1f} kcal/mol")
    """
    return _ff.compute_energy(_get_ptr(structure))


def minimize_hydrogens(
    structure,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
) -> dict:
    """Minimize hydrogen positions using AMBER96 force field.

    Freezes all heavy atoms and optimizes only H positions.
    Uses steepest descent with adaptive step size.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.

    Examples:
        >>> result = ferritin.minimize_hydrogens(structure)
        >>> print(f"Energy: {result['initial_energy']:.0f} -> {result['final_energy']:.0f}")
    """
    return _ff.minimize_hydrogens(_get_ptr(structure), max_steps, gradient_tolerance)


def minimize_structure(
    structure,
    max_steps: int = 1000,
    gradient_tolerance: float = 0.1,
) -> dict:
    """Full energy minimization using AMBER96 force field.

    All atoms are free to move.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 1000).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.
    """
    return _ff.minimize_structure(_get_ptr(structure), max_steps, gradient_tolerance)


def batch_minimize_hydrogens(
    structures,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    *,
    n_threads=None,
):
    """Minimize hydrogen positions for many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        max_steps: Maximum optimization steps per structure.
        gradient_tolerance: Convergence criterion in kcal/mol/A.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of result dicts (same format as minimize_hydrogens).

    Examples:
        >>> results = ferritin.batch_minimize_hydrogens(structures, n_threads=-1)
        >>> for r in results:
        ...     print(f"E: {r['initial_energy']:.0f} -> {r['final_energy']:.0f}")
    """
    ptrs = [_get_ptr(s) for s in structures]
    return _ff.batch_minimize_hydrogens(ptrs, max_steps, gradient_tolerance, n_threads)


def load_and_minimize_hydrogens(
    paths,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    *,
    n_threads=None,
):
    """Load files and minimize hydrogens in one parallel call (zero GIL).

    Args:
        paths: List of file paths.
        max_steps: Maximum optimization steps per structure.
        gradient_tolerance: Convergence criterion in kcal/mol/A.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of (index, result_dict) tuples.
    """
    str_paths = [str(p) for p in paths]
    return _ff.load_and_minimize_hydrogens(str_paths, max_steps, gradient_tolerance, n_threads)
