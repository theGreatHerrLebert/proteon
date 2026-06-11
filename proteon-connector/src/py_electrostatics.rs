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

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;
use proteon_electrostatics::{
    born_rfenergy, espotential, rfenergy, solve_local_elements_auto, solve_nonlocal_elements_auto,
    solve_nonlocal_elements_q, AdaptiveConfig, Charge, Domain, Locality, Params, Quadrature,
    QualityReport, Severity, SolveConfig, Tri,
};

/// Triangle count past which the dense O(N²) solve is warned about.
const N_WARN: usize = 15_000;

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

/// Owned solve result (no Python types — built off the GIL).
struct SolveOut {
    phi: Vec<f64>,
    rfenergy: f64,
    iterations: usize,
    residual: f64,
    converged: bool,
    watertight: bool,
    oriented: bool,
    /// Regular-Yukawa quadrature actually used ("fixed" / "adaptive") — surfaced so an
    /// adaptive request that fell back, or a fixed run, is never silently confused.
    quadrature: &'static str,
    /// Adaptive panels that hit the depth cap with the error still above tolerance
    /// (0 for fixed). Non-zero ⇒ the near-singular result is not certified for those.
    capped_panels: usize,
    /// Mesh-acceptance metrics (P6.5): smallest triangle angle, max aspect ratio,
    /// near-degenerate count, and the scale-aware charge-to-surface gap ratio.
    min_angle_deg: f64,
    max_aspect_ratio: f64,
    n_near_degenerate: usize,
    min_charge_gap_ratio: f64,
    /// Quality issues to surface to the caller as Python warnings (`Error`-severity ones
    /// are only present here when the caller overrode the refusal).
    quality_warnings: Vec<String>,
}

/// Solve the local/nonlocal BEM on a surface mesh with point charges and return the
/// per-vertex electrostatic potential plus diagnostics.
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
    // --- validate shapes -----------------------------------------------------
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
    if !(tol.is_finite() && tol > 0.0) || max_iter == 0 || restart == 0 {
        return Err(PyValueError::new_err(
            "need tol > 0, max_iter > 0, and restart > 0",
        ));
    }
    validate_params(eps_omega, eps_sigma, eps_inf, lambda_)?;

    // Quadrature selector (nonlocal regular-Yukawa). Adaptive is CPU-only and accurate
    // near clefts; fixed is the fast default. Parsed here (with the GIL) so the off-GIL
    // closure just consumes the Copy enum.
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
            ("quadrature='adaptive' has no effect on the local solve (the Laplace \
              collocation is analytic/exact); it applies only to nonlocal_=True.",),
        )?;
    }

    // Memory guard: the dense BEM holds 2 (local) or 4 (nonlocal) N×N f64 matrices.
    // Refuse a job that would blow past the budget unless the caller opts in.
    const MEM_BUDGET: u128 = 6 * (1 << 30); // 6 GiB
    let blocks: u128 = if nonlocal_ { 4 } else { 2 };
    let est = (nf as u128).saturating_mul(nf as u128).saturating_mul(8) * blocks;
    if est > MEM_BUDGET && !allow_large {
        return Err(PyValueError::new_err(format!(
            "{nf} triangles would allocate ~{} GiB of dense matrices (the BEM is O(N²)); \
             coarsen the SES mesh or pass allow_large=True to override.",
            est >> 30
        )));
    }
    if nf >= N_WARN {
        let warnings = py.import("warnings")?;
        let tail = "the dense BEM is O(N²) in memory and time — this will be slow/RAM-heavy. \
             Over the dense budget it switches to the O(N)-memory matrix-free GPU solve \
             if a CUDA device is present (slower per solve, but uncapped in mesh size).";
        warnings.call_method1("warn", (format!("{nf} triangles: {tail}"),))?;
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

    // Finite checks (a NaN/inf would otherwise propagate silently into the solve).
    if vflat
        .iter()
        .chain(&qpos)
        .chain(&qval)
        .any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "non-finite value in vertices / charge_positions / charge_values",
        ));
    }
    // Triangle indices in range, and each triangle non-degenerate (so `Tri::new` —
    // which asserts on a zero / non-finite normal — cannot panic off the GIL).
    for f in 0..nf {
        let mut idx = [0usize; 3];
        for (k, slot) in idx.iter_mut().enumerate() {
            let i = tflat[f * 3 + k];
            if i < 0 || i as usize >= nv {
                return Err(PyValueError::new_err("triangle index out of range"));
            }
            *slot = i as usize;
        }
        let p = |k: usize| Vec3::new(vflat[k * 3], vflat[k * 3 + 1], vflat[k * 3 + 2]);
        let cross = (p(idx[1]) - p(idx[0])).cross(p(idx[2]) - p(idx[0]));
        if !(cross.norm() > 0.0 && cross.norm().is_finite()) {
            return Err(PyValueError::new_err(format!(
                "degenerate (zero-area / collinear) triangle at index {f}"
            )));
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

        // P6.5 mesh acceptance: assess BEFORE the expensive solve. Refuse on Error-
        // severity issues (near-degenerate triangles, a charge near Γ in element-size
        // units) unless overridden; collect the rest as warnings. "No silent arbitrary-
        // protein runs."
        let quality = QualityReport::assess(&elements, &charges);
        let issues = quality.issues();
        let has_errors = issues.iter().any(|i| i.severity == Severity::Error);
        if has_errors && !allow_low_quality {
            let msgs: Vec<String> = issues
                .iter()
                .filter(|i| i.severity == Severity::Error)
                .map(|i| i.message.clone())
                .collect();
            return Err(format!(
                "mesh/charge quality unacceptable for a reliable solve: {}. \
                 Fix the mesh/charge placement, or pass allow_low_quality=True to override.",
                msgs.join("; ")
            ));
        }
        // Surface every issue (incl. overridden errors) as a warning.
        let quality_warnings: Vec<String> = issues.iter().map(|i| i.message.clone()).collect();

        // The CauchyData trait is object-safe; box the result so both localities share
        // the post-processing path.
        let (cauchy, engy, stats): (Box<dyn proteon_electrostatics::CauchyData>, f64, _) =
            if nonlocal_ {
                // Adaptive (accuracy requested) → CPU adaptive path DIRECTLY, never the
                // size dispatcher, which could route a large mesh to the fixed-quadrature
                // GPU path and silently swap accuracy. Fixed → the auto (GPU-capable) path.
                let (r, s) = match quad {
                    Quadrature::Adaptive(_) => {
                        solve_nonlocal_elements_q(&elements, &charges, &params, &cfg, quad)
                    }
                    Quadrature::Fixed => {
                        solve_nonlocal_elements_auto(&elements, &charges, &params, &cfg)
                    }
                }
                .map_err(|e| e.to_string())?;
                let e = rfenergy(&elements, &charges, &r);
                (Box::new(r), e, s)
            } else {
                let (r, s) = solve_local_elements_auto(&elements, &charges, &params, &cfg)
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

        if !engy.is_finite() || phi.iter().any(|v| !v.is_finite()) {
            return Err("solve produced a non-finite energy / potential".to_string());
        }

        // Mesh-quality diagnostics: the double-layer sign depends on the winding, so a
        // non-watertight or inconsistently-oriented mesh can give silently-wrong
        // potentials. Surface these so the caller can trust (or distrust) the result.
        let mesh = Mesh {
            verts: (0..nv).map(vert).collect(),
            normals: Vec::new(),
            tris: (0..nf)
                .map(|f| {
                    [
                        tflat[f * 3] as u32,
                        tflat[f * 3 + 1] as u32,
                        tflat[f * 3 + 2] as u32,
                    ]
                })
                .collect(),
        };

        Ok(SolveOut {
            phi,
            rfenergy: engy,
            iterations: stats.iterations,
            residual: stats.residual,
            converged: stats.converged,
            watertight: mesh.is_watertight(),
            oriented: mesh.is_consistently_oriented(),
            quadrature: match stats.quadrature {
                Quadrature::Fixed => "fixed",
                Quadrature::Adaptive(_) => "adaptive",
            },
            capped_panels: stats.capped_panels,
            min_angle_deg: quality.min_angle_deg,
            max_aspect_ratio: quality.max_aspect_ratio,
            n_near_degenerate: quality.n_near_degenerate,
            min_charge_gap_ratio: quality.min_charge_gap_ratio,
            quality_warnings,
        })
    });

    let out = result.map_err(PyValueError::new_err)?;

    // A bad winding silently flips the double-layer sign — warn loudly rather than
    // hand back a wrong-but-plausible potential.
    if !out.oriented {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "mesh is not consistently oriented: the double-layer sign depends on the \
              triangle winding, so the potential may be wrong. Pass a watertight, \
              outward-oriented mesh (e.g. ses_mesh_coarse_py output).",
            ),
        )?;
    }

    let dict = PyDict::new(py);
    dict.set_item("surface_potential", PyArray1::from_vec(py, out.phi))?;
    dict.set_item("rfenergy", out.rfenergy)?;
    dict.set_item("iterations", out.iterations)?;
    dict.set_item("residual", out.residual)?;
    dict.set_item("converged", out.converged)?;
    dict.set_item("n_elements", nf)?;
    dict.set_item("watertight", out.watertight)?;
    dict.set_item("oriented", out.oriented)?;
    dict.set_item("quadrature", out.quadrature)?;
    dict.set_item("capped_panels", out.capped_panels)?;
    dict.set_item("min_angle_deg", out.min_angle_deg)?;
    dict.set_item("max_aspect_ratio", out.max_aspect_ratio)?;
    dict.set_item("n_near_degenerate", out.n_near_degenerate)?;
    dict.set_item("min_charge_gap_ratio", out.min_charge_gap_ratio)?;

    // Surface mesh-acceptance warnings (sliver elements, a charge close to Γ, or an
    // overridden Error) so a low-quality result is never silently trusted.
    if !out.quality_warnings.is_empty() {
        let warnings = py.import("warnings")?;
        for msg in &out.quality_warnings {
            warnings.call_method1("warn", (format!("mesh/charge quality: {msg}"),))?;
        }
    }

    // A capped adaptive solve means some near-singular panels did not reach tolerance;
    // surface it so the caller can raise the depth or distrust those entries.
    if out.capped_panels > 0 {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "{} adaptive panels hit the depth cap without converging; the near-singular \
                 result is not certified for those entries.",
                out.capped_panels
            ),),
        )?;
    }
    Ok(dict.unbind())
}

#[pymodule]
pub(crate) fn py_electrostatics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(born_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_surface_py, m)?)?;
    Ok(())
}
