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

from typing import Dict, Tuple

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
    quadrature: str = "fixed",
    allow_low_quality: bool = False,
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
        quadrature: regular-Yukawa rule for the *nonlocal* solve — ``"fixed"``
            (default, fast 7-point Radon) or ``"adaptive"`` (the near-singular
            remediation; slower, CPU-only, accurate near clefts). No effect on the
            local solve (the Laplace collocation is exact). When ``"adaptive"`` is
            requested the solve stays on the accurate CPU path (it is never routed to
            the fixed-quadrature GPU path); ``allow_large`` still governs the dense
            memory budget.
        allow_low_quality: override the mesh-acceptance refusal (P6.5). By default the
            solve refuses near-degenerate triangles and charges within a small multiple
            of the local element size of the surface (a near-singular molecular
            potential); set True to solve anyway (the issues are still warned).

    Returns:
        Dict with `surface_potential` (V, float64, volts), `rfenergy` (kJ/mol),
        `iterations`, `residual`, `converged` (bool), `n_elements`, the mesh
        topology diagnostics `watertight` / `oriented` / `is_outward` (bool),
        `signed_volume`, `n_components`, `n_cavities`, `charge_components`
        (per-charge solute-body index, None if in solvent), `n_duplicate_faces`,
        `n_self_intersections`, `flipped_to_outward`
        (an inward mesh is auto-flipped to outward), the `quadrature` rule actually
        used, `capped_panels` (adaptive panels that did not reach tolerance), and the
        geometry/charge metrics `min_angle_deg`, `max_aspect_ratio`,
        `n_near_degenerate`, `min_charge_gap_ratio`, `n_charges_outside`. Quality and
        convergence issues emit warnings. A non-closed / inconsistently-wound / duplicate-
        face mesh, or a charge outside or essentially on the surface, is refused unless
        `allow_low_quality=True`.

    Raises:
        ValueError: bad shapes/values, a degenerate triangle, an over-budget mesh
            (without `allow_large`), unacceptable mesh/charge quality (without
            `allow_low_quality`), an unknown `quadrature`, or a non-converged /
            non-finite solve.
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
        quadrature,
        allow_low_quality,
    )
    out["surface_potential"] = np.asarray(out["surface_potential"])
    return out


def read_off(
    path: str,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Read a Geomview OFF surface mesh.

    Args:
        path: path to the ``.off`` file.

    Returns:
        ``(vertices, triangles)`` — ``(V, 3)`` float64 coordinates and ``(F, 3)``
        int64 0-based vertex indices, ready to pass to `surface_potential`.
    """
    d = _el.read_off_py(path)
    return np.asarray(d["vertices"]), np.asarray(d["triangles"])


def read_pqr(
    path: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a PQR charge set (atomic charges + radii).

    Only ``ATOM`` records are taken (``HETATM`` water excluded) and zero-charge
    atoms are dropped, matching NESSie's reader.

    Args:
        path: path to the ``.pqr`` file.

    Returns:
        ``(charge_positions, charge_values)`` — ``(Q, 3)`` float64 positions and
        ``(Q,)`` float64 charges (elementary charges).
    """
    d = _el.read_pqr_py(path)
    return np.asarray(d["charge_positions"]), np.asarray(d["charge_values"])


def read_hmo(
    path: str,
) -> Dict[str, NDArray]:
    """Read an HMO file (BEM mesh + charges in one document).

    Args:
        path: path to the ``.hmo`` file.

    Returns:
        Dict with ``vertices`` ``(V, 3)``, ``triangles`` ``(F, 3)``,
        ``charge_positions`` ``(Q, 3)``, and ``charge_values`` ``(Q,)`` — the full
        set of arrays `surface_potential` needs.
    """
    d = _el.read_hmo_py(path)
    return {
        "vertices": np.asarray(d["vertices"]),
        "triangles": np.asarray(d["triangles"]),
        "charge_positions": np.asarray(d["charge_positions"]),
        "charge_values": np.asarray(d["charge_values"]),
    }


def read_msms(
    vert_path: str,
    face_path: str,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Read an MSMS surface from its ``.vert`` / ``.face`` pair.

    Args:
        vert_path: path to the ``.vert`` file.
        face_path: path to the ``.face`` file.

    Returns:
        ``(vertices, triangles)`` — ``(V, 3)`` float64 coordinates and ``(F, 3)``
        int64 0-based vertex indices. MSMS carries no charges.
    """
    d = _el.read_msms_py(vert_path, face_path)
    return np.asarray(d["vertices"]), np.asarray(d["triangles"])


def write_off(
    path: str,
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> None:
    """Write a surface mesh to a Geomview OFF file.

    Args:
        path: destination ``.off`` path.
        vertices: ``(V, 3)`` float64 coordinates.
        triangles: ``(F, 3)`` int 0-based vertex indices.
    """
    _el.write_off_py(
        path,
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(triangles, dtype=np.int64),
    )
