"""Continuum electrostatics — the NESSie boundary-element (BEM) port.

Local (Poisson) and nonlocal (Lorentz cavity / Yukawa) reaction-field energies and
electrostatic potentials, plus the closed-form Born model. A high-accuracy
reference/research tier alongside proteon's fast GB/EEF1 solvation.

Typical use — colour a surface mesh by its electrostatic potential::

    import proteon, proteon_connector
    mesh = proteon_connector.py_surface.ses_mesh_coarse_py(
        coords, elements=elems, spacing=1.0,                 # coarse SES for the BEM
    )
    out = proteon.surface_potential(
        mesh["vertices"], mesh["triangles"], charge_xyz, charge_q,
    )
    phi = out["surface_potential"]                           # volts, per vertex

Scaling caveat: the dense BEM is O(N²) in memory and time. `surface_potential` refuses
a job past a ~6 GiB matrix budget (override with `allow_large=True`) and warns past
~15k triangles. Use a *coarse* mesh — the exact analytic SES is easily millions of
triangles. The result dict carries `watertight` / `oriented` diagnostics; a mesh that
is not consistently outward-oriented can give a sign-wrong potential (and warns).
"""

from typing import Dict

import numpy as np
from numpy.typing import NDArray

try:
    import proteon_connector

    _el = proteon_connector.py_electrostatics
except ImportError:  # pragma: no cover
    _el = None


def born_energy(
    charge: float,
    radius: float,
    eps_omega: float = 1.0,
    eps_sigma: float = 78.0,
    eps_inf: float = 1.8,
    lambda_: float = 20.0,
    nonlocal_: bool = False,
) -> float:
    """Closed-form Born solvation (reaction-field) energy of a single ion (kJ/mol).

    Args:
        charge: ion charge (elementary charges).
        radius: ion radius (Å).
        eps_omega/eps_sigma/eps_inf/lambda_: solute / solvent / bulk dielectric and
            correlation length (Å). The Born model assumes a vacuum solute
            (eps_omega = 1); eps_omega is accepted for symmetry but ignored.
        nonlocal_: use the nonlocal (structured-solvent) form instead of local.

    Returns:
        Reaction-field energy in kJ/mol (negative — solvation is favourable).
    """
    return _el.born_energy_py(
        charge, radius, eps_omega, eps_sigma, eps_inf, lambda_, nonlocal_
    )


def surface_potential(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    charge_positions: NDArray[np.float64],
    charge_values: NDArray[np.float64],
    eps_omega: float = 1.0,
    eps_sigma: float = 78.0,
    eps_inf: float = 1.8,
    lambda_: float = 20.0,
    nonlocal_: bool = False,
    tol: float = 1e-7,
    restart: int = 200,
    max_iter: int = 10000,
    allow_large: bool = False,
) -> Dict[str, object]:
    """Solve the BEM on a surface mesh with point charges and read off the potential.

    Args:
        vertices: (V, 3) float64 surface vertices.
        triangles: (F, 3) int triangle indices (CCW wrt. the outward normal).
        charge_positions: (Q, 3) float64 point-charge positions.
        charge_values: (Q,) float64 charges (elementary charges).
        eps_*/lambda_: dielectric parameters (finite, > 0; see `born_energy`).
        nonlocal_: nonlocal (Lorentz/Yukawa) solve instead of local Poisson.
        tol/restart/max_iter: GMRES controls (converges on the true residual).
        allow_large: override the ~6 GiB dense-matrix memory guard.

    Returns:
        Dict with `surface_potential` (V, float64, volts), `rfenergy` (kJ/mol),
        `iterations`, `residual`, `converged` (bool), `n_elements`, and the mesh
        diagnostics `watertight` / `oriented` (bool).

    Raises:
        ValueError: bad shapes/values, a degenerate triangle, an over-budget mesh
            (without `allow_large`), or a non-converged / non-finite solve.
    """
    out = _el.solve_surface_py(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(triangles, dtype=np.int64),
        np.ascontiguousarray(charge_positions, dtype=np.float64),
        np.ascontiguousarray(charge_values, dtype=np.float64),
        eps_omega,
        eps_sigma,
        eps_inf,
        lambda_,
        nonlocal_,
        tol,
        restart,
        max_iter,
        allow_large,
    )
    out["surface_potential"] = np.asarray(out["surface_potential"])
    return out
