//! PyO3 binding for the analytic SES (solvent-excluded surface) mesher.
//!
//! Runs `proteon_core::surface::assemble::ses_mesh` and hands the triangle mesh
//! back to Python as numpy arrays — an **in-process** handoff (no PLY/OBJ file
//! round-trip), so a consumer like `proteon-vis` can build a render mesh directly
//! from atom coordinates.
//!
//! Low-level contract: `coords`/`radii` must be C-contiguous float64 (the Python
//! wrapper coerces). All inputs are validated up front so a bad value raises
//! `ValueError` rather than tripping a `ses_mesh` `assert!` (which would panic the
//! interpreter).

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::{ses_mesh, SesMethod};
use proteon_core::surface::geom::{Sphere, Vec3};

/// Build validated atom spheres from coords (Nx3) and exactly one of `radii` (N,)
/// or element symbols (length N → vdW radii). Errors on a bad shape/length, a
/// non-finite coordinate, or a non-positive radius.
fn spheres_from(
    coords: &PyReadonlyArray2<'_, f64>,
    radii: Option<&PyReadonlyArray1<'_, f64>>,
    elements: Option<&[String]>,
) -> PyResult<Vec<Sphere>> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("coords must be an Nx3 array"));
    }
    let n = shape[0];
    let cs = coords
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("coords not C-contiguous: {e}")))?;

    let rs: Vec<f64> = match (radii, elements) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "pass exactly one of `radii` or `elements`, not both",
            ))
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "provide either `radii` or `elements`",
            ))
        }
        (Some(r), None) => {
            if r.shape() != [n] {
                return Err(PyValueError::new_err(
                    "radii length must match the atom count",
                ));
            }
            r.as_slice()
                .map_err(|e| PyValueError::new_err(format!("radii not C-contiguous: {e}")))?
                .to_vec()
        }
        (None, Some(els)) => {
            if els.len() != n {
                return Err(PyValueError::new_err(
                    "elements length must match the atom count",
                ));
            }
            els.iter()
                .map(|e| vdw_radius(e).unwrap_or(DEFAULT_RADIUS))
                .collect()
        }
    };

    let mut spheres = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, z) = (cs[i * 3], cs[i * 3 + 1], cs[i * 3 + 2]);
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "non-finite coordinate at atom {i}"
            )));
        }
        if !(rs[i].is_finite() && rs[i] > 0.0) {
            return Err(PyValueError::new_err(format!(
                "radius at atom {i} must be finite and positive (got {})",
                rs[i]
            )));
        }
        spheres.push(Sphere::new(Vec3::new(x, y, z), rs[i]));
    }
    Ok(spheres)
}

/// Owned mesh result computed off the GIL (no Python types).
struct MeshOut {
    vflat: Vec<f64>,
    tflat: Vec<u32>,
    nv: usize,
    nt: usize,
    area: f64,
    volume: f64,
    watertight: bool,
    method: &'static str,
    perturbations: usize,
}

/// Mesh the SES of `coords` (Nx3) with per-atom `radii` (N,) **or** `elements`
/// (length N, vdW radii looked up — unknown symbols fall back to a default radius).
/// Returns a dict: `vertices` (Mx3 float64), `triangles` (Kx3 uint32), `area`,
/// `volume`, `watertight` (bool), `method` (str: `"analytic"` /
/// `"analytic_perturbed"` / `"numerical_grid"`), `perturbations` (int — atom
/// jitters used by the analytic retry, 0 otherwise).
///
/// The hybrid mesher always returns a mesh: exact analytic (+ deterministic
/// perturbation retry) where possible, numerical-grid fallback otherwise.
#[pyfunction]
#[pyo3(signature = (
    coords, radii=None, elements=None, probe=1.4,
    n_theta=48, n_phi=10, grid=0.04, weld_eps=1e-5, sdf_spacing=0.3,
))]
#[allow(clippy::too_many_arguments)]
fn ses_mesh_py<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    radii: Option<PyReadonlyArray1<'py, f64>>,
    elements: Option<Vec<String>>,
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    grid: f64,
    weld_eps: f64,
    sdf_spacing: f64,
) -> PyResult<Py<PyDict>> {
    // Validate parameters up front — `ses_mesh`/the SDF fallback `assert!` on
    // these, which would panic the interpreter rather than raise.
    for (name, v) in [
        ("probe", probe),
        ("grid", grid),
        ("weld_eps", weld_eps),
        ("sdf_spacing", sdf_spacing),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and > 0"
            )));
        }
    }
    if n_theta < 3 || n_phi < 1 {
        return Err(PyValueError::new_err("need n_theta >= 3 and n_phi >= 1"));
    }

    let spheres = spheres_from(&coords, radii.as_ref(), elements.as_deref())?;
    if spheres.len() < 2 {
        return Err(PyValueError::new_err(
            "need at least 2 atoms to mesh an SES",
        ));
    }

    // Everything heavy — meshing, metadata (incl. hashmap-based watertightness),
    // and flattening millions of values — runs WITHOUT the GIL.
    let out = py.allow_threads(|| {
        let (mesh, method) = ses_mesh(&spheres, probe, n_theta, n_phi, grid, weld_eps, sdf_spacing);
        let area = mesh.surface_area();
        let volume = mesh.signed_volume();
        let watertight = mesh.is_watertight();
        let (method, perturbations) = match method {
            SesMethod::Analytic => ("analytic", 0),
            SesMethod::AnalyticPerturbed(n) => ("analytic_perturbed", n),
            SesMethod::NumericalGrid(_) => ("numerical_grid", 0),
        };
        let (nv, nt) = (mesh.verts.len(), mesh.tris.len());
        let mut vflat = Vec::with_capacity(nv * 3);
        for v in mesh.verts {
            vflat.extend_from_slice(&[v.x, v.y, v.z]);
        }
        let mut tflat = Vec::with_capacity(nt * 3);
        for t in mesh.tris {
            tflat.extend_from_slice(&t);
        }
        MeshOut {
            vflat,
            tflat,
            nv,
            nt,
            area,
            volume,
            watertight,
            method,
            perturbations,
        }
    });

    // Only the numpy/dict construction touches Python.
    let vertices = PyArray1::from_vec(py, out.vflat)
        .reshape([out.nv, 3])
        .map_err(|e| PyValueError::new_err(format!("vertex reshape: {e}")))?;
    let triangles = PyArray1::from_vec(py, out.tflat)
        .reshape([out.nt, 3])
        .map_err(|e| PyValueError::new_err(format!("triangle reshape: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("vertices", vertices)?;
    dict.set_item("triangles", triangles)?;
    dict.set_item("area", out.area)?;
    dict.set_item("volume", out.volume)?;
    dict.set_item("watertight", out.watertight)?;
    dict.set_item("method", out.method)?;
    dict.set_item("perturbations", out.perturbations)?;
    Ok(dict.unbind())
}

#[pymodule]
pub(crate) fn py_surface(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ses_mesh_py, m)?)?;
    Ok(())
}
