//! PyO3 binding for continuum-electrostatics (the NESSie BEM port).
//!
//! Thin shims over `proteon-electrostatics`: the closed-form Born energy, and a
//! mesh-in / surface-potential-out solver so a consumer (proteon-vis, a notebook)
//! can colour an SES by its electrostatic potential without a JSON round-trip.
//!
//! **Scaling caveat (plan §6 / P6.5).** The dense BEM is O(N²) in memory and time;
//! `solve_surface_py` warns past a triangle budget. Pass a watertight, consistently
//! outward-oriented mesh (e.g. `proteon.surface.ses_mesh` output) — the double-layer
//! sign depends on the winding.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    born_rfenergy, espotential, rfenergy, solve_local_elements, solve_nonlocal_elements, Charge,
    Domain, Locality, Params, SolveConfig, Tri,
};

/// Triangle count past which the dense O(N²) solve is warned about.
const N_WARN: usize = 15_000;

/// Closed-form Born reaction-field (solvation) energy of a single ion (kJ/mol).
///
/// The Born model assumes a vacuum solute (`eps_omega = 1`); `eps_omega` is accepted
/// for symmetry but ignored. `nonlocal_=True` uses the structured-solvent form.
#[pyfunction]
#[pyo3(signature = (charge, radius, eps_omega=1.0, eps_sigma=78.0, eps_inf=1.8, lambda_=20.0, nonlocal_=false))]
#[allow(clippy::too_many_arguments)]
fn born_energy_py(
    charge: f64,
    radius: f64,
    eps_omega: f64,
    eps_sigma: f64,
    eps_inf: f64,
    lambda_: f64,
    nonlocal_: bool,
) -> PyResult<f64> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(PyValueError::new_err("radius must be finite and > 0"));
    }
    let params = Params {
        eps_omega,
        eps_sigma,
        eps_inf,
        lambda: lambda_,
    };
    let loc = if nonlocal_ {
        Locality::Nonlocal
    } else {
        Locality::Local
    };
    Ok(born_rfenergy(charge, radius, &params, loc))
}

/// Owned solve result (no Python types — built off the GIL).
struct SolveOut {
    phi: Vec<f64>,
    rfenergy: f64,
    iterations: usize,
    residual: f64,
    converged: bool,
}

/// Solve the local/nonlocal BEM on a surface mesh with point charges and return the
/// per-vertex electrostatic potential plus diagnostics.
///
/// Inputs: `vertices` (V×3 float64), `triangles` (F×3 int), `charge_positions` (Q×3
/// float64), `charge_values` (Q,). Returns a dict: `surface_potential` (V, float64,
/// volts), `rfenergy` (kJ/mol), `iterations`, `residual`, `converged` (bool),
/// `n_elements`.
#[pyfunction]
#[pyo3(signature = (
    vertices, triangles, charge_positions, charge_values,
    eps_omega=1.0, eps_sigma=78.0, eps_inf=1.8, lambda_=20.0,
    nonlocal_=false, tol=1e-7, restart=200, max_iter=10000,
))]
#[allow(clippy::too_many_arguments)]
fn solve_surface_py<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, i64>,
    charge_positions: PyReadonlyArray2<'py, f64>,
    charge_values: PyReadonlyArray1<'py, f64>,
    eps_omega: f64,
    eps_sigma: f64,
    eps_inf: f64,
    lambda_: f64,
    nonlocal_: bool,
    tol: f64,
    restart: usize,
    max_iter: usize,
) -> PyResult<Py<PyDict>> {
    // --- validate shapes + values --------------------------------------------
    if vertices.shape().len() != 2 || vertices.shape()[1] != 3 {
        return Err(PyValueError::new_err("vertices must be V×3"));
    }
    if triangles.shape().len() != 2 || triangles.shape()[1] != 3 {
        return Err(PyValueError::new_err("triangles must be F×3"));
    }
    if charge_positions.shape().len() != 2 || charge_positions.shape()[1] != 3 {
        return Err(PyValueError::new_err("charge_positions must be Q×3"));
    }
    let nv = vertices.shape()[0];
    let nf = triangles.shape()[0];
    let nq = charge_positions.shape()[0];
    if charge_values.shape() != [nq] {
        return Err(PyValueError::new_err(
            "charge_values length must match charge_positions",
        ));
    }
    if nf == 0 || nq == 0 {
        return Err(PyValueError::new_err(
            "need at least one triangle and one charge",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) || max_iter == 0 {
        return Err(PyValueError::new_err("need tol > 0 and max_iter > 0"));
    }

    if nf > N_WARN {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "{nf} triangles: the dense BEM is O(N²) in memory and time — this will \
                 be slow/RAM-heavy. Coarsen the SES mesh, or wait for the matrix-free / \
                 GPU paths (plan §6/P6.5)."
            ),),
        )?;
    }

    let vflat = vertices
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("vertices not C-contiguous: {e}")))?
        .to_vec();
    let tflat = triangles
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("triangles not C-contiguous: {e}")))?
        .to_vec();
    let qpos = charge_positions
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("charge_positions not C-contiguous: {e}")))?
        .to_vec();
    let qval = charge_values
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("charge_values not C-contiguous: {e}")))?
        .to_vec();

    // Bounds-check the triangle indices before the off-GIL block (no panics there).
    for &i in &tflat {
        if i < 0 || i as usize >= nv {
            return Err(PyValueError::new_err("triangle index out of range"));
        }
    }

    let params = Params {
        eps_omega,
        eps_sigma,
        eps_inf,
        lambda: lambda_,
    };
    let cfg = SolveConfig {
        tol,
        restart,
        max_iter,
    };

    // --- heavy compute off the GIL -------------------------------------------
    let result = py.allow_threads(|| -> Result<SolveOut, String> {
        let vert = |k: usize| Vec3::new(vflat[k * 3], vflat[k * 3 + 1], vflat[k * 3 + 2]);
        let elements: Vec<Tri> = (0..nf)
            .map(|f| {
                Tri::new(
                    vert(tflat[f * 3] as usize),
                    vert(tflat[f * 3 + 1] as usize),
                    vert(tflat[f * 3 + 2] as usize),
                )
            })
            .collect();
        let charges: Vec<Charge> = (0..nq)
            .map(|q| Charge {
                pos: Vec3::new(qpos[q * 3], qpos[q * 3 + 1], qpos[q * 3 + 2]),
                val: qval[q],
            })
            .collect();

        // The CauchyData trait is object-safe; box the result so both localities share
        // the post-processing path.
        let (cauchy, engy, stats): (Box<dyn proteon_electrostatics::CauchyData>, f64, _) =
            if nonlocal_ {
                let (r, s) = solve_nonlocal_elements(&elements, &charges, &params, &cfg)
                    .map_err(|e| e.to_string())?;
                let e = rfenergy(&elements, &charges, &r);
                (Box::new(r), e, s)
            } else {
                let (r, s) = solve_local_elements(&elements, &charges, &params, &cfg)
                    .map_err(|e| e.to_string())?;
                let e = rfenergy(&elements, &charges, &r);
                (Box::new(r), e, s)
            };

        // Per-vertex surface potential (Γ trace).
        let phi: Vec<f64> = (0..nv)
            .map(|k| {
                espotential(
                    Domain::Gamma,
                    vert(k),
                    &elements,
                    &charges,
                    &params,
                    &*cauchy,
                )
            })
            .collect();

        Ok(SolveOut {
            phi,
            rfenergy: engy,
            iterations: stats.iterations,
            residual: stats.residual,
            converged: stats.converged,
        })
    });

    let out = result.map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("surface_potential", PyArray1::from_vec(py, out.phi))?;
    dict.set_item("rfenergy", out.rfenergy)?;
    dict.set_item("iterations", out.iterations)?;
    dict.set_item("residual", out.residual)?;
    dict.set_item("converged", out.converged)?;
    dict.set_item("n_elements", nf)?;
    Ok(dict.unbind())
}

#[pymodule]
pub(crate) fn py_electrostatics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(born_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_surface_py, m)?)?;
    Ok(())
}
