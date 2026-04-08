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

    Agent Notes:
        WATCH: Absolute values not comparable across different structures.
            vdw > 1e6 = steric clashes (normal for raw X-ray, not a bug).
        COST: O(N^2) for nonbonded terms. Slow above 5000 atoms.
    """
    return _ff.compute_energy(_get_ptr(structure))


def minimize_hydrogens(
    structure,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    method: str = "sd",
) -> dict:
    """Minimize hydrogen positions using AMBER96 force field.

    Freezes all heavy atoms and optimizes only H positions.
    Uses steepest descent with adaptive step size.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 500).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).
        method: "sd" (steepest descent, default) or "cg" (conjugate gradient).
            CG converges faster but SD is more robust for severe clashes.

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.

    Examples:
        >>> result = ferritin.minimize_hydrogens(structure)
        >>> result = ferritin.minimize_hydrogens(structure, method="cg")

    Agent Notes:
        WATCH: Check result['converged']. If False, increase max_steps.
        PREFER: batch_minimize_hydrogens() for multiple structures.
        COST: O(N^2) per step. Slow above 5000 atoms.
    """
    return _ff.minimize_hydrogens(_get_ptr(structure), max_steps, gradient_tolerance, method)


def minimize_structure(
    structure,
    max_steps: int = 1000,
    gradient_tolerance: float = 0.1,
    method: str = "sd",
) -> dict:
    """Full energy minimization using AMBER96 force field.

    All atoms are free to move. Uses steepest descent.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 1000).
        gradient_tolerance: Convergence criterion in kcal/mol/A (default 0.1).

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.

    Agent Notes:
        WATCH: Moves ALL atoms — can distort the structure. For most use
            cases, minimize_hydrogens() is safer and sufficient.
        WATCH: In-vacuo force field will collapse exposed sidechains.
            Use only for clash relief (100-500 steps), not refinement.
    """
    return _ff.minimize_structure(_get_ptr(structure), max_steps, gradient_tolerance, method)


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

    Entire pipeline (I/O, parsing, topology, minimization) runs in Rust.

    Args:
        paths: List of file paths.
        max_steps: Maximum optimization steps per structure.
        gradient_tolerance: Convergence criterion in kcal/mol/A.
        n_threads: Thread count. None/-1 = all cores.

    Returns:
        List of (index, result_dict) tuples. Files that fail to load are skipped.

    Agent Notes:
        WATCH: Failed inputs are skipped. Use the returned indices to map results
            back to the original path list.
        PREFER: Use this for many files instead of a Python loop over load() and
            minimize_hydrogens().
    """
    str_paths = [str(p) for p in paths]
    return _ff.load_and_minimize_hydrogens(str_paths, max_steps, gradient_tolerance, n_threads)


def run_md(
    structure,
    n_steps: int = 1000,
    dt: float = 0.001,
    temperature: float = 300.0,
    thermostat_tau: float = 0.2,
    snapshot_freq: int = 10,
) -> dict:
    """Run molecular dynamics simulation using Velocity Verlet integration.

    Args:
        structure: Ferritin Structure object.
        n_steps: Number of MD steps (default 1000).
        dt: Time step in picoseconds (default 0.001 = 1 fs).
        temperature: Initial/target temperature in Kelvin (default 300).
        thermostat_tau: Berendsen coupling time in ps. 0 = NVE (default 0.2 = NVT).
        snapshot_freq: Record trajectory frame every N steps (default 10).

    Returns:
        Dict with keys:
            coords: final coordinates (N, 3) numpy array.
            velocities: final velocities (N, 3) numpy array.
            trajectory: list of frame dicts (step, time_ps, kinetic_energy,
                potential_energy, total_energy, temperature).
            energy: final energy components dict.
            n_steps, dt, temperature_target, thermostat_tau: simulation parameters.
    """
    return _ff.run_md(_get_ptr(structure), n_steps, dt, temperature, thermostat_tau, snapshot_freq)
