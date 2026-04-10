"""Force field energy computation, minimization, and molecular dynamics.

Supports AMBER96 and CHARMM19+EEF1 force fields via the ForceField trait.
Energy output defaults to kJ/mol (set units="kcal/mol" for legacy convention).

Functions:
    compute_energy      — energy breakdown by component
    minimize_hydrogens  — optimize H positions (freeze heavy atoms)
    minimize_structure  — full energy minimization
    batch_minimize_hydrogens — parallel minimization
    load_and_minimize_hydrogens — load + minimize in one call
    run_md              — molecular dynamics with Velocity Verlet
"""

from __future__ import annotations

from typing import Optional

try:
    import ferritin_connector
    _ff = ferritin_connector.py_forcefield
except ImportError:
    _ff = None

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

_KCAL_TO_KJ = 4.184

_VALID_UNITS = {"kj/mol", "kcal/mol"}

_ENERGY_KEYS = frozenset({
    "bond_stretch", "angle_bend", "torsion", "improper_torsion",
    "vdw", "electrostatic", "solvation", "total",
    "initial_energy", "final_energy",
})


def _convert_energy_dict(d: dict, units: str) -> dict:
    """Convert energy values in a dict from internal kcal/mol to requested units."""
    if units == "kcal/mol":
        return d
    factor = _KCAL_TO_KJ
    out = {}
    for k, v in d.items():
        if k in _ENERGY_KEYS and isinstance(v, (int, float)):
            out[k] = v * factor
        elif k == "energy_components" and isinstance(v, dict):
            out[k] = {ek: ev * factor if isinstance(ev, (int, float)) else ev
                      for ek, ev in v.items()}
        else:
            out[k] = v
    return out


def _convert_md_result(d: dict, units: str) -> dict:
    """Convert MD result dict energies from kcal/mol to requested units."""
    if units == "kcal/mol":
        return d
    factor = _KCAL_TO_KJ
    out = dict(d)
    # Convert energy components dict
    if "energy" in out and isinstance(out["energy"], dict):
        out["energy"] = {k: v * factor if isinstance(v, (int, float)) else v
                         for k, v in out["energy"].items()}
    # Convert trajectory frames
    if "trajectory" in out:
        new_frames = []
        for frame in out["trajectory"]:
            nf = dict(frame)
            for ek in ("kinetic_energy", "potential_energy", "total_energy"):
                if ek in nf:
                    nf[ek] = nf[ek] * factor
            new_frames.append(nf)
        out["trajectory"] = new_frames
    return out


def _validate_units(units: str) -> str:
    u = units.lower().replace(" ", "")
    if u not in _VALID_UNITS:
        raise ValueError(
            f"Unknown units '{units}'. Use 'kJ/mol' or 'kcal/mol'."
        )
    return u


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ptr(structure):
    if hasattr(structure, 'get_py_ptr'):
        return structure.get_py_ptr()
    return structure


_VALID_METHODS = {"sd", "steepest_descent", "cg", "conjugate_gradient", "lbfgs", "l-bfgs"}


def _validate_method(method: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown minimizer method '{method}'. Use 'sd', 'cg', or 'lbfgs'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_energy(
    structure,
    ff: str = "amber96",
    units: str = "kJ/mol",
) -> dict:
    """Compute force field energy of a structure.

    Args:
        structure: Ferritin Structure object.
        ff: Force field — "amber96" (default) or "charmm19_eef1".
        units: Energy units — "kJ/mol" (default) or "kcal/mol".

    Returns dict with energy components:
        bond_stretch, angle_bend, torsion, improper_torsion,
        vdw, electrostatic, solvation, total
    """
    u = _validate_units(units)
    result = _ff.compute_energy(_get_ptr(structure), ff)
    return _convert_energy_dict(result, u)


def minimize_hydrogens(
    structure,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    method: str = "sd",
    units: str = "kJ/mol",
) -> dict:
    """Minimize hydrogen positions using AMBER96 force field.

    Freezes all heavy atoms and optimizes only H positions.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 500).
        gradient_tolerance: Convergence criterion (default 0.1).
        method: "sd" (steepest descent), "cg" (conjugate gradient), or "lbfgs".
        units: Energy units — "kJ/mol" (default) or "kcal/mol".

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.
    """
    u = _validate_units(units)
    _validate_method(method)
    result = _ff.minimize_hydrogens(_get_ptr(structure), max_steps, gradient_tolerance, method)
    return _convert_energy_dict(result, u)


def minimize_structure(
    structure,
    max_steps: int = 1000,
    gradient_tolerance: float = 0.1,
    method: str = "sd",
    units: str = "kJ/mol",
) -> dict:
    """Full energy minimization using AMBER96 force field.

    All atoms are free to move.

    Args:
        structure: A ferritin Structure.
        max_steps: Maximum optimization steps (default 1000).
        gradient_tolerance: Convergence criterion (default 0.1).
        method: "sd", "cg", or "lbfgs".
        units: Energy units — "kJ/mol" (default) or "kcal/mol".

    Returns:
        dict with: coords (Nx3), initial_energy, final_energy,
        steps, converged, energy_components.
    """
    u = _validate_units(units)
    _validate_method(method)
    result = _ff.minimize_structure(_get_ptr(structure), max_steps, gradient_tolerance, method)
    return _convert_energy_dict(result, u)


def batch_minimize_hydrogens(
    structures,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    *,
    n_threads=None,
    units: str = "kJ/mol",
):
    """Minimize hydrogen positions for many structures in parallel (Rust + rayon).

    Args:
        structures: List of ferritin Structure objects.
        max_steps: Maximum optimization steps per structure.
        gradient_tolerance: Convergence criterion.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.
        units: Energy units — "kJ/mol" (default) or "kcal/mol".

    Returns:
        List of result dicts (same format as minimize_hydrogens).
    """
    u = _validate_units(units)
    ptrs = [_get_ptr(s) for s in structures]
    results = _ff.batch_minimize_hydrogens(ptrs, max_steps, gradient_tolerance, n_threads)
    return [_convert_energy_dict(r, u) for r in results]


def load_and_minimize_hydrogens(
    paths,
    max_steps: int = 500,
    gradient_tolerance: float = 0.1,
    *,
    n_threads=None,
):
    """Load files and minimize hydrogens in one parallel call (zero GIL).

    Entire pipeline (I/O, parsing, topology, minimization) runs in Rust.
    Returns energies in kcal/mol (internal units, no conversion).

    Args:
        paths: List of file paths.
        max_steps: Maximum optimization steps per structure.
        gradient_tolerance: Convergence criterion.
        n_threads: Thread count. ``None`` / ``-1`` / ``0`` = all cores
            (default); a positive integer = exactly that many threads.

    Returns:
        List of (index, result_dict) tuples. Files that fail to load are skipped.
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
    shake: bool = False,
    units: str = "kJ/mol",
) -> dict:
    """Run molecular dynamics simulation using Velocity Verlet integration.

    Args:
        structure: Ferritin Structure object.
        n_steps: Number of MD steps (default 1000).
        dt: Time step in picoseconds (default 0.001 = 1 fs).
        temperature: Initial/target temperature in Kelvin (default 300).
        thermostat_tau: Berendsen coupling time in ps. 0 = NVE (default 0.2 = NVT).
        snapshot_freq: Record trajectory frame every N steps (default 10).
        shake: Constrain X-H bond lengths via SHAKE/RATTLE (default False).
        units: Energy units — "kJ/mol" (default) or "kcal/mol".

    Returns:
        Dict with keys:
            coords: final coordinates (N, 3) numpy array.
            velocities: final velocities (N, 3) numpy array.
            trajectory: list of frame dicts (step, time_ps, kinetic_energy,
                potential_energy, total_energy, temperature).
            energy: final energy components dict.
            n_steps, dt, temperature_target, thermostat_tau: simulation parameters.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    u = _validate_units(units)
    result = _ff.run_md(_get_ptr(structure), n_steps, dt, temperature, thermostat_tau, snapshot_freq, shake)
    return _convert_md_result(result, u)
