//! PyO3 binding for continuum-electrostatics (the NESSie BEM port).
//!
//! Thin shims over `proteon-electrostatics`: the closed-form Born energy, and a
//! mesh-in / surface-potential-out solver so a consumer (proteon-vis, a notebook)
//! can colour an SES by its electrostatic potential without a JSON round-trip.
//!
//! **Scaling caveat (plan §6 / P6.5).** The dense BEM is O(N²) in memory and time;
//! `solve_surface_py` refuses an over-budget mesh (unless `allow_large`) and warns
//! past a triangle budget. Pass a watertight, consistently outward-oriented mesh
//! (e.g. `ses_mesh_coarse_py` output) — the double-layer sign depends on the winding;
//! the result's `oriented` diagnostic flags (and warns on) a bad one.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;
use proteon_electrostatics::{
    born_rfenergy, read_hmo, read_msms, read_off, read_pqr, solve_surface, write_off,
    AdaptiveConfig, Charge, Locality, Params, Quadrature, SolveConfig, SurfaceSolveError,
    SurfaceSolveOptions,
};

/// Closed-form Born reaction-field (solvation) energy of a single ion (kJ/mol).
///
/// The Born model assumes a **vacuum solute**: the formula uses `(1/εΣ − 1)`, so a
/// non-1 `eps_omega` is rejected (`ValueError`) rather than silently ignored.
/// `nonlocal_=True` uses the structured-solvent form.
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
    if !charge.is_finite() {
        return Err(PyValueError::new_err("charge must be finite"));
    }
    if !(radius.is_finite() && radius > 0.0) {
        return Err(PyValueError::new_err("radius must be finite and > 0"));
    }
    if (eps_omega - 1.0).abs() > 1e-9 {
        return Err(PyValueError::new_err(
            "the Born model assumes a vacuum solute: eps_omega must be 1",
        ));
    }
    validate_params(eps_omega, eps_sigma, eps_inf, lambda_)?;
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

/// Finite + positive dielectric / correlation-length parameters.
fn validate_params(eps_omega: f64, eps_sigma: f64, eps_inf: f64, lambda_: f64) -> PyResult<()> {
    for (name, v) in [
        ("eps_omega", eps_omega),
        ("eps_sigma", eps_sigma),
        ("eps_inf", eps_inf),
        ("lambda_", lambda_),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and > 0"
            )));
        }
    }
    Ok(())
}

/// Solve the local/nonlocal BEM on a surface mesh with point charges and return the
/// per-vertex electrostatic potential plus diagnostics.
///
/// A thin shim over `proteon_electrostatics::solve_surface` — the connector validates
/// the numpy inputs, runs the (heavy) solve off the GIL, re-emits the solve's advisories
/// as Python warnings, and packs the outcome into a dict. The science (degeneracy guard,
/// topological acceptance + auto-orient, cavity gate, mesh-acceptance quality gate,
/// charge→body assignment, local/nonlocal dispatch, Γ-trace potential) lives in the
/// pure-Rust entry point, shared with the `proteon electrostatics` CLI so they cannot
/// drift.
///
/// Inputs: `vertices` (V×3 float64), `triangles` (F×3 int), `charge_positions` (Q×3
/// float64), `charge_values` (Q,). `quadrature` selects the regular-Yukawa rule for the
/// **nonlocal** solve: `"fixed"` (default, fast 7-point Radon) or `"adaptive"` (the P6.5
/// near-singular remediation — slower, CPU-only, but accurate near clefts). Returns a
/// dict: `surface_potential` (V, float64, volts), `rfenergy` (kJ/mol), `iterations`,
/// `residual`, `converged` (bool), `n_elements`, `watertight`, `oriented`, `quadrature`
/// (the rule actually used), `capped_panels`, and the P6.5 mesh-acceptance metrics
/// `min_angle_deg` / `max_aspect_ratio` / `n_near_degenerate` / `min_charge_gap_ratio`.
///
/// Refuses (raises `ValueError`) on unacceptable mesh/charge quality — near-degenerate
/// triangles, or a charge within a small multiple of the local element size of the
/// surface (the molecular-potential trace would be near-singular) — unless
/// `allow_low_quality=True`. Sliver elements and near-surface charges otherwise warn.
#[pyfunction]
#[pyo3(signature = (
    vertices, triangles, charge_positions, charge_values,
    eps_omega=1.0, eps_sigma=78.0, eps_inf=1.8, lambda_=20.0,
    nonlocal_=false, tol=1e-7, restart=200, max_iter=10000, allow_large=false,
    quadrature="fixed", allow_low_quality=false,
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
    allow_large: bool,
    quadrature: &str,
    allow_low_quality: bool,
) -> PyResult<Py<PyDict>> {
    // --- validate numpy shapes -----------------------------------------------
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
    let nq = charge_positions.shape()[0];
    if charge_values.shape() != [nq] {
        return Err(PyValueError::new_err(
            "charge_values length must match charge_positions",
        ));
    }
    // Empty check early (matches the pre-extraction ordering: before tol/param checks).
    if triangles.shape()[0] == 0 || nq == 0 {
        return Err(PyValueError::new_err(
            "need at least one triangle and one charge",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) || max_iter == 0 || restart == 0 {
        return Err(PyValueError::new_err(
            "need tol > 0, max_iter > 0, and restart > 0",
        ));
    }
    validate_params(eps_omega, eps_sigma, eps_inf, lambda_)?;

    // Quadrature selector (nonlocal regular-Yukawa). Adaptive is CPU-only and accurate
    // near clefts; fixed is the fast default.
    let quad = match quadrature {
        "fixed" => Quadrature::Fixed,
        "adaptive" => Quadrature::Adaptive(AdaptiveConfig::default()),
        other => {
            return Err(PyValueError::new_err(format!(
                "quadrature must be 'fixed' or 'adaptive', got '{other}'"
            )));
        }
    };
    if !nonlocal_ && matches!(quad, Quadrature::Adaptive(_)) {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "quadrature='adaptive' has no effect on the local solve (the Laplace \
              collocation is analytic/exact); it applies only to nonlocal_=True.",
            ),
        )?;
    }

    // --- extract C-contiguous slices -----------------------------------------
    let vflat = vertices
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("vertices not C-contiguous: {e}")))?;
    let tflat = triangles
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("triangles not C-contiguous: {e}")))?;
    let qpos = charge_positions
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("charge_positions not C-contiguous: {e}")))?;
    let qval = charge_values
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("charge_values not C-contiguous: {e}")))?;

    // Triangle indices must be in range here (negative / overflow would corrupt the u32
    // cast); the pure solver re-checks degeneracy and finiteness.
    for &i in tflat {
        if i < 0 || i as usize >= nv {
            return Err(PyValueError::new_err("triangle index out of range"));
        }
    }

    let mesh = Mesh {
        verts: vflat
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect(),
        normals: Vec::new(),
        tris: tflat
            .chunks_exact(3)
            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
            .collect(),
    };
    let charges: Vec<Charge> = qpos
        .chunks_exact(3)
        .zip(qval)
        .map(|(p, &val)| Charge {
            pos: Vec3::new(p[0], p[1], p[2]),
            val,
        })
        .collect();

    let opts = SurfaceSolveOptions {
        params: Params {
            eps_omega,
            eps_sigma,
            eps_inf,
            lambda: lambda_,
        },
        cfg: SolveConfig {
            tol,
            restart,
            max_iter,
        },
        nonlocal: nonlocal_,
        quadrature: quad,
        allow_large,
        allow_low_quality,
        fast_summation: None, // dense by default; treecode opt-in exposed via the CLI / Rust API
    };

    // --- heavy compute off the GIL -------------------------------------------
    let output = py.allow_threads(|| solve_surface(mesh, &charges, &opts));

    // Re-emit the advisories (size, auto-flip, quality issues) as Python warnings —
    // ALWAYS, before handling the result, so a refused/failed solve still warns exactly
    // as the pre-extraction connector did.
    {
        let warnings = py.import("warnings")?;
        for w in &output.warnings {
            warnings.call_method1("warn", (w.clone(),))?;
        }
    }
    let sol = output.result.map_err(surface_err_to_py)?;

    let topology = &sol.topology;
    let quality = &sol.quality;
    let dict = PyDict::new(py);
    dict.set_item(
        "surface_potential",
        PyArray1::from_vec(py, sol.potential.clone()),
    )?;
    dict.set_item("rfenergy", sol.rfenergy)?;
    dict.set_item("iterations", sol.iterations)?;
    dict.set_item("residual", sol.residual)?;
    dict.set_item("converged", sol.converged)?;
    dict.set_item("n_elements", sol.n_elements)?;
    dict.set_item("watertight", topology.watertight)?;
    dict.set_item("oriented", topology.consistently_oriented)?;
    dict.set_item("is_outward", topology.is_outward)?;
    dict.set_item("signed_volume", topology.signed_volume)?;
    dict.set_item("n_components", topology.num_components)?;
    dict.set_item("n_cavities", topology.num_cavities)?;
    dict.set_item("components_touch", topology.components_touch)?;
    dict.set_item(
        "charge_components",
        sol.charge_components
            .iter()
            .map(|c| c.map(|i| i as i64))
            .collect::<Vec<Option<i64>>>(),
    )?;
    dict.set_item("n_duplicate_faces", topology.num_duplicate_faces)?;
    dict.set_item("n_self_intersections", topology.num_self_intersections)?;
    dict.set_item("flipped_to_outward", sol.flipped_to_outward)?;
    dict.set_item("quadrature", sol.quadrature)?;
    dict.set_item("capped_panels", sol.capped_panels)?;
    dict.set_item("min_angle_deg", quality.min_angle_deg)?;
    dict.set_item("max_aspect_ratio", quality.max_aspect_ratio)?;
    dict.set_item("n_near_degenerate", quality.n_near_degenerate)?;
    dict.set_item("min_charge_gap_ratio", quality.min_charge_gap_ratio)?;
    dict.set_item("n_charges_outside", quality.n_charges_outside)?;

    // A capped adaptive solve means some near-singular panels did not reach tolerance.
    if sol.capped_panels > 0 {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "{} adaptive panels hit the depth cap without converging; the near-singular \
                 result is not certified for those entries.",
                sol.capped_panels
            ),),
        )?;
    }
    Ok(dict.unbind())
}

/// Map a `SurfaceSolveError` to a Python exception (all are caller-fixable ⇒ `ValueError`).
fn surface_err_to_py(e: SurfaceSolveError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// =========================================================================================
// File-format I/O (the NESSie `format/` layer) — load a surface mesh + charge set
// straight off disk into the arrays `solve_surface_py` consumes.

/// Map a format-reader `io::Error` to the right Python exception: a parse error
/// (`InvalidData`) is a `ValueError`; a missing file is a `FileNotFoundError`;
/// anything else (permissions, etc.) is a generic `OSError`.
fn io_err_to_py(e: std::io::Error) -> PyErr {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::InvalidData => PyValueError::new_err(e.to_string()),
        ErrorKind::NotFound => pyo3::exceptions::PyFileNotFoundError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyOSError::new_err(e.to_string()),
    }
}

/// Mesh → (V×3 f64 vertices, F×3 i64 triangles) numpy arrays. Propagates a reshape
/// failure as a real exception rather than panicking across the FFI boundary.
fn mesh_to_py<'py>(
    py: Python<'py>,
    mesh: &Mesh,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let nv = mesh.verts.len();
    let vflat: Vec<f64> = mesh.verts.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let verts = PyArray1::from_vec(py, vflat).reshape([nv, 3])?;
    let nf = mesh.tris.len();
    let tflat: Vec<i64> = mesh
        .tris
        .iter()
        .flat_map(|t| [i64::from(t[0]), i64::from(t[1]), i64::from(t[2])])
        .collect();
    let tris = PyArray1::from_vec(py, tflat).reshape([nf, 3])?;
    Ok((verts, tris))
}

/// Charges → (Q×3 f64 positions, Q f64 values) numpy arrays.
fn charges_to_py<'py>(
    py: Python<'py>,
    charges: &[Charge],
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let nq = charges.len();
    let pflat: Vec<f64> = charges
        .iter()
        .flat_map(|c| [c.pos.x, c.pos.y, c.pos.z])
        .collect();
    let pos = PyArray1::from_vec(py, pflat).reshape([nq, 3])?;
    let vals = PyArray1::from_vec(py, charges.iter().map(|c| c.val).collect::<Vec<_>>());
    Ok((pos, vals))
}

/// Read a Geomview **OFF** surface mesh → dict `{vertices: V×3 f64, triangles: F×3 i64}`.
#[pyfunction]
fn read_off_py(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    // I/O + parsing off the GIL; numpy construction after reacquiring it.
    let mesh = py.allow_threads(|| read_off(path)).map_err(io_err_to_py)?;
    let dict = PyDict::new(py);
    let (v, t) = mesh_to_py(py, &mesh)?;
    dict.set_item("vertices", v)?;
    dict.set_item("triangles", t)?;
    Ok(dict.unbind())
}

/// Read a **PQR** charge set → dict `{charge_positions: Q×3 f64, charge_values: Q f64}`.
/// Only `ATOM` records are taken and zero-charge atoms dropped (as NESSie does).
#[pyfunction]
fn read_pqr_py(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let charges = py.allow_threads(|| read_pqr(path)).map_err(io_err_to_py)?;
    let dict = PyDict::new(py);
    let (pos, vals) = charges_to_py(py, &charges)?;
    dict.set_item("charge_positions", pos)?;
    dict.set_item("charge_values", vals)?;
    Ok(dict.unbind())
}

/// Read an **HMO** file (mesh + charges in one document) → dict with all four
/// arrays: `vertices`, `triangles`, `charge_positions`, `charge_values`.
#[pyfunction]
fn read_hmo_py(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let (mesh, charges) = py.allow_threads(|| read_hmo(path)).map_err(io_err_to_py)?;
    let dict = PyDict::new(py);
    let (v, t) = mesh_to_py(py, &mesh)?;
    let (pos, vals) = charges_to_py(py, &charges)?;
    dict.set_item("vertices", v)?;
    dict.set_item("triangles", t)?;
    dict.set_item("charge_positions", pos)?;
    dict.set_item("charge_values", vals)?;
    Ok(dict.unbind())
}

/// Read an **MSMS** surface from its `.vert` / `.face` pair → dict
/// `{vertices: V×3 f64, triangles: F×3 i64}`. MSMS carries no charges.
#[pyfunction]
fn read_msms_py(py: Python<'_>, vert_path: &str, face_path: &str) -> PyResult<Py<PyDict>> {
    let mesh = py
        .allow_threads(|| read_msms(vert_path, face_path))
        .map_err(io_err_to_py)?;
    let dict = PyDict::new(py);
    let (v, t) = mesh_to_py(py, &mesh)?;
    dict.set_item("vertices", v)?;
    dict.set_item("triangles", t)?;
    Ok(dict.unbind())
}

/// Write a mesh (`vertices` V×3, `triangles` F×3) to a Geomview **OFF** file.
#[pyfunction]
fn write_off_py(
    py: Python<'_>,
    path: &str,
    vertices: PyReadonlyArray2<'_, f64>,
    triangles: PyReadonlyArray2<'_, i64>,
) -> PyResult<()> {
    if vertices.shape().len() != 2 || vertices.shape()[1] != 3 {
        return Err(PyValueError::new_err("vertices must be V×3"));
    }
    if triangles.shape().len() != 2 || triangles.shape()[1] != 3 {
        return Err(PyValueError::new_err("triangles must be F×3"));
    }
    let nv = vertices.shape()[0];
    let vflat = vertices
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("vertices not C-contiguous: {e}")))?;
    let tflat = triangles
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("triangles not C-contiguous: {e}")))?;
    if vflat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("non-finite vertex coordinate"));
    }
    let verts: Vec<Vec3> = vflat
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(tflat.len() / 3);
    for c in tflat.chunks_exact(3) {
        let mut t = [0u32; 3];
        for (k, &id) in c.iter().enumerate() {
            // Validate against nv, then checked-convert — a value past u32::MAX (or
            // a negative id) is an error, never a silent wrap.
            let idx = usize::try_from(id)
                .ok()
                .filter(|&u| u < nv)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("triangle index {id} out of range 0..{nv}"))
                })?;
            t[k] = u32::try_from(idx)
                .map_err(|_| PyValueError::new_err(format!("triangle index {idx} exceeds u32")))?;
        }
        tris.push(t);
    }
    let mesh = Mesh {
        verts,
        normals: Vec::new(),
        tris,
    };
    py.allow_threads(|| write_off(&mesh, path))
        .map_err(io_err_to_py)
}

#[pymodule]
pub(crate) fn py_electrostatics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(born_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_surface_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_off_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_pqr_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_hmo_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_msms_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_off_py, m)?)?;
    Ok(())
}
