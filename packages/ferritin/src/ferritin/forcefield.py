"""Force field energy computation, minimization, and molecular dynamics.

Supports AMBER96 and CHARMM19+EEF1 force fields via the ForceField trait.
Energy output defaults to kJ/mol (set units="kcal/mol" for legacy convention).

GPU acceleration is automatic when the binary is compiled with the ``cuda``
feature (``maturin develop --features cuda``). Use :func:`gpu_available` to
check at runtime, and :func:`gpu_info` for device details. The GPU is used
transparently for:
  * **Energy + forces** in the minimizer (structures >= 2000 atoms)
  * **SASA** in Shrake-Rupley (structures >= 500 atoms)
No Python code changes are needed — the dispatch is in the Rust layer.

Functions:
    compute_energy      — energy breakdown by component
    minimize_hydrogens  — optimize H positions (freeze heavy atoms)
    minimize_structure  — full energy minimization
    batch_minimize_hydrogens — parallel minimization
    load_and_minimize_hydrogens — load + minimize in one call
    run_md              — molecular dynamics with Velocity Verlet
    gpu_available       — check if GPU acceleration is available
    gpu_info            — get GPU device details
"""

from __future__ import annotations

import functools
import warnings
from typing import Optional

try:
    import ferritin_connector
    _ff = ferritin_connector.py_forcefield
except ImportError:  # pragma: no cover
    _ff = None


@functools.lru_cache(maxsize=1)
def _warn_amber96_experimental() -> None:
    """Warn once per process about the ferritin-vs-OpenMM AMBER96 policy
    difference in the nonbonded cutoff.

    Status after 2026-04-13 fixes (H-name aliases + double-wildcard
    improper lookup + cutoff override): on identical PDBFixer-prepped
    crambin with both tools at NoCutoff, ferritin-AMBER96 agrees with
    OpenMM-AMBER96 to **0.2%** on total energy — bond 0.017%, angle
    0.22%, torsion 0.44%, nonbonded 0.26%. The force-field math is
    reference-quality.

    Ferritin's *default* production setting uses a 15 Å nonbonded cutoff
    with cubic switching from 13 Å (a performance-vs-accuracy choice).
    That contributes the ~1-2% difference against a NoCutoff reference.
    Pass ``nonbonded_cutoff=1e6`` to ``compute_energy`` to disable the
    cutoff and get full-precision agreement for oracle comparisons.
    """
    warnings.warn(
        "ferritin's AMBER96 force-field math matches OpenMM AMBER96 to "
        "0.2% on crambin when both use NoCutoff. The default ferritin "
        "setting applies a 15 Å nonbonded cutoff with switching — that "
        "is a performance-vs-accuracy choice, not a bug. Pass "
        "nonbonded_cutoff=1e6 to disable for oracle-grade comparison.",
        UserWarning,
        stacklevel=3,
    )


def _maybe_warn_ff(ff: Optional[str]) -> None:
    if ff is not None and ff.lower() == "amber96":
        _warn_amber96_experimental()

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
    nbl_threshold: int | None = None,
    nonbonded_cutoff: float | None = None,
) -> dict:
    """Compute force field energy of a structure.

    Force field status:
        * ``"charmm19_eef1"`` — **validated production path.** Used by the
          50K battle test (99.1% negative total) and the fold-preservation
          benchmark (median TM=0.9945 pre/post minimization). This is what
          downstream code should use.
        * ``"amber96"`` — **oracle-validated** against OpenMM AMBER96 on
          crambin: bond/angle/torsion/nonbonded all match within 0.5%,
          total energy within 0.2%, when both tools are run with
          NoCutoff. The default ferritin setting applies a 15 Å
          nonbonded cutoff with cubic switching from 13 Å (the
          performance-vs-accuracy default inherited from BALL); pass
          ``nonbonded_cutoff=1e6`` for oracle-grade full-precision
          agreement. A UserWarning about the policy is emitted once per
          process.

    Args:
        structure: Ferritin Structure object.
        ff: Force field — "amber96" (experimental) or "charmm19_eef1"
            (validated, recommended).
        units: Energy units — "kJ/mol" (default) or "kcal/mol".
        nbl_threshold: Optional override for the neighbor-list atom-count
            threshold. None (default) uses the library default of 2000.
            Set to 0 to force the neighbor-list path for every structure,
            or to a very large value to force the O(N²) exact path. Only
            useful for cross-path parity testing; leave as None in normal
            use.

    Returns dict with energy components:
        bond_stretch, angle_bend, torsion, improper_torsion,
        vdw, electrostatic, solvation, total
    """
    u = _validate_units(units)
    _maybe_warn_ff(ff)
    result = _ff.compute_energy(_get_ptr(structure), ff, nbl_threshold, nonbonded_cutoff)
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

    Note:
        Uses AMBER96, which is experimental in ferritin — energies do not
        match OpenMM AMBER96 on identical inputs. Positions are self-consistent
        but not reference-quality. See ``compute_energy`` docstring.

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
    _maybe_warn_ff("amber96")
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

    Note:
        Uses AMBER96, which is experimental in ferritin — energies do not
        match OpenMM AMBER96 on identical inputs. See ``compute_energy``
        docstring. For validated production use, prepare structures with
        ``batch_prepare(ff="charmm19_eef1")`` (the default).

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
    _maybe_warn_ff("amber96")
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


# ---------------------------------------------------------------------------
# GPU status API
# ---------------------------------------------------------------------------


def gpu_available() -> bool:
    """Check if GPU (CUDA) acceleration is available.

    Returns True if:
      1. The binary was compiled with the ``cuda`` feature
         (``maturin develop --features cuda``), AND
      2. A CUDA-capable GPU was detected at runtime.

    When True, ``batch_prepare()`` and ``atom_sasa()`` automatically
    dispatch large structures to the GPU. No additional configuration
    needed.

    Returns:
        bool: True if GPU is available and ready.

    Examples:
        >>> ferritin.gpu_available()
        True
        >>> if ferritin.gpu_available():
        ...     print("GPU will be used for structures >= 2000 atoms")
    """
    return _ff.gpu_available()


def gpu_info() -> dict:
    """Get GPU device information.

    Returns a dict with:
        - ``cuda_compiled`` (bool): whether the binary has CUDA support
        - ``available`` (bool): whether a GPU was detected
        - ``name`` (str): GPU device name (e.g. "NVIDIA GeForce RTX 5090")
        - ``compute_capability`` (str): e.g. "12.0"
        - ``total_memory_mb`` (int): total GPU memory in MB

    If no GPU is available, only ``cuda_compiled`` and ``available`` are present.

    Examples:
        >>> ferritin.gpu_info()
        {'cuda_compiled': True, 'available': True, 'name': 'NVIDIA GeForce RTX 5090',
         'compute_capability': '12.0', 'total_memory_mb': 32607}
    """
    return dict(_ff.gpu_info())
