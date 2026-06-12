//! Mesh-in / surface-potential-out orchestration — the one BEM solve front-end.
//!
//! [`solve_surface`] takes a triangulated molecular surface plus point charges and
//! returns the per-vertex electrostatic potential, the reaction-field energy, and a
//! full bundle of mesh/charge diagnostics. It owns every step between the raw mesh
//! and the linear solve: the degeneracy guard, topological acceptance + per-component
//! auto-orient (so the double-layer sign is right), the cavity science-gate, the
//! combined mesh-acceptance quality gate, the charge→solute-body assignment, the
//! local/nonlocal dispatch, and the Γ-trace potential.
//!
//! This is **the** entry point both front-ends call — the PyO3 connector
//! (`solve_surface_py`) and the `proteon electrostatics` CLI — so they cannot drift:
//! one implementation of the science, two thin presentation layers. Non-fatal
//! advisories are returned as a `warnings` list (the connector re-emits them as
//! Python warnings; the CLI prints them to stderr) rather than being raised here.

use std::fmt;

use proteon_core::surface::mesh::Mesh;

use crate::model::{Charge, Domain, Params, Tri};
use crate::post::{espotential, rfenergy};
use crate::quality::{QualityReport, Severity, TopologyReport};
use crate::fastsum::operator::MAX_FS_ORDER;
use crate::solve::{
    solve_local_elements_auto, solve_local_elements_treecode, solve_nonlocal_elements_auto,
    solve_nonlocal_elements_q, CauchyData, SolveConfig,
};
use crate::system::Quadrature;

/// Dense-matrix memory budget. The BEM holds 2 (local) or 4 (nonlocal) N×N f64 blocks;
/// a job past this is refused unless `allow_large` (the GPU matrix-free path, when
/// present, is uncapped — but the dense CPU path is not).
pub const MEM_BUDGET: u128 = 6 * (1 << 30); // 6 GiB

/// Triangle count past which the dense O(N²) solve earns a "this will be slow" advisory.
pub const N_WARN: usize = 15_000;

/// Opt-in fast-summation (P8 treecode) settings for the **local** solve. The treecode's
/// realized win today is O(N) memory — solving meshes too large for the dense O(N²)
/// matrices — not raw matvec speed (see `devdocs/TO_ELECTROSTATICS_P8.md` §5.1), so it is
/// **never** auto-enabled; a caller asks for it explicitly with calibrated `(p, θ)`.
#[derive(Clone, Copy)]
pub struct FastSummation {
    /// Cartesian expansion order (accuracy knob; higher = tighter, costlier).
    pub p: usize,
    /// MAC admissibility ratio, in `(0, 1)`.
    pub theta: f64,
}

/// Tunables for [`solve_surface`] beyond the mesh + charges themselves.
pub struct SurfaceSolveOptions {
    /// Dielectric / correlation-length parameters.
    pub params: Params,
    /// GMRES controls (tolerance, restart, max iterations).
    pub cfg: SolveConfig,
    /// Nonlocal (Lorentz/Yukawa) solve instead of local Poisson.
    pub nonlocal: bool,
    /// Regular-Yukawa cubature for the nonlocal solve (no effect on local).
    pub quadrature: Quadrature,
    /// Override the dense-matrix memory guard ([`MEM_BUDGET`]).
    pub allow_large: bool,
    /// Override the mesh-acceptance refusal (solve despite Error-severity quality issues).
    pub allow_low_quality: bool,
    /// Opt-in treecode fast summation for the local solve (`None` = exact dense). When
    /// set, the dense-matrix memory guard is bypassed (the treecode is the point of
    /// asking). Has no effect on the nonlocal solve (warned), which stays dense (P8.4).
    pub fast_summation: Option<FastSummation>,
}

impl Default for SurfaceSolveOptions {
    fn default() -> Self {
        Self {
            params: Params {
                eps_omega: 1.0,
                eps_sigma: 78.0,
                eps_inf: 1.8,
                lambda: 20.0,
            },
            cfg: SolveConfig::default(),
            nonlocal: false,
            quadrature: Quadrature::Fixed,
            allow_large: false,
            allow_low_quality: false,
            fast_summation: None,
        }
    }
}

/// The result of a surface solve: potential + energy + every diagnostic the front-ends surface.
pub struct SurfaceSolution {
    /// Per-vertex electrostatic potential (Γ trace), one entry per mesh vertex, in volts.
    pub potential: Vec<f64>,
    /// Reaction-field (solvation) energy, kJ/mol.
    pub rfenergy: f64,
    /// GMRES iterations across both stages.
    pub iterations: usize,
    /// Worst true relative residual.
    pub residual: f64,
    /// Whether the solve met its residual gate.
    pub converged: bool,
    /// Regular-Yukawa rule actually used (`"fixed"` / `"adaptive"`).
    pub quadrature: &'static str,
    /// Adaptive panels that hit the depth cap without reaching tolerance (0 for fixed).
    pub capped_panels: usize,
    /// Triangle count solved.
    pub n_elements: usize,
    /// Topology assessment (after any auto-orient).
    pub topology: TopologyReport,
    /// Geometry + charge-placement quality.
    pub quality: QualityReport,
    /// Per-charge containing solute-body component (`None` ⇒ in solvent / ambiguous).
    pub charge_components: Vec<Option<usize>>,
    /// Whether one or more components were re-oriented outward-from-solute.
    pub flipped_to_outward: bool,
}

/// The outcome of [`solve_surface`]: the non-fatal advisories accumulated up to the
/// point of return (emitted **regardless** of success or failure, matching the original
/// connector's "warn before refuse" behaviour) plus the success/failure result.
pub struct SurfaceSolveOutput {
    /// Advisories (size, auto-flip, quality issues) collected before the function returned.
    pub warnings: Vec<String>,
    /// The solution, or the reason the solve could not complete.
    pub result: Result<SurfaceSolution, SurfaceSolveError>,
}

/// Why a surface solve could not be completed.
#[derive(Debug)]
pub enum SurfaceSolveError {
    /// A solver or dielectric parameter is out of range (non-finite, non-positive, …).
    InvalidParams(String),
    /// No triangles or no charges.
    Empty,
    /// A zero-area / collinear triangle at this index.
    DegenerateTriangle(usize),
    /// A non-finite vertex or charge value.
    NonFiniteInput,
    /// `nonlocal` requested on a mesh with buried cavities — not yet validated science.
    NonlocalCavity(usize),
    /// The dense matrices would exceed [`MEM_BUDGET`] and `allow_large` is off.
    OverBudget {
        /// Triangle count.
        n_elements: usize,
        /// Estimated dense-matrix footprint, GiB.
        gib: u128,
    },
    /// Error-severity mesh/charge quality issues with `allow_low_quality` off.
    QualityUnacceptable(Vec<String>),
    /// The linear solve failed (non-convergence, non-finite result, or a solver error).
    Solve(String),
}

impl fmt::Display for SurfaceSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParams(m) => write!(f, "{m}"),
            Self::Empty => write!(f, "need at least one triangle and one charge"),
            Self::DegenerateTriangle(i) => {
                write!(f, "degenerate (zero-area / collinear) triangle at index {i}")
            }
            Self::NonFiniteInput => {
                write!(f, "non-finite value in vertices / charge positions / charge values")
            }
            Self::NonlocalCavity(n) => write!(
                f,
                "{n} buried cavity / nested component(s) with nonlocal=true: the nonlocal \
                 formulation on cavities is not yet validated (the cavity science gate is \
                 local-only). Use a single-region mesh, or the local solve."
            ),
            Self::OverBudget { n_elements, gib } => write!(
                f,
                "{n_elements} triangles would allocate ~{gib} GiB of dense matrices (the BEM is \
                 O(N²)); coarsen the mesh or set allow_large to override."
            ),
            Self::QualityUnacceptable(msgs) => write!(
                f,
                "mesh/charge quality unacceptable for a reliable solve: {}. Fix the mesh/charge \
                 placement, or set allow_low_quality to override.",
                msgs.join("; ")
            ),
            Self::Solve(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SurfaceSolveError {}

/// Solve the local/nonlocal BEM on `mesh` with point `charges`; see the module docs.
///
/// Returns a [`SurfaceSolveOutput`] whose `warnings` are populated **regardless** of
/// success or failure (so a refused or failed solve still surfaces the advisories the
/// caller would have wanted — exactly as the connector did before this was extracted)
/// and whose `result` carries the solution or the typed error.
///
/// The `mesh` is taken by value because acceptable meshes are auto-oriented in place
/// (each inward component flipped outward-from-solute by nesting parity); callers that
/// need the original keep their own copy.
pub fn solve_surface(
    mesh: Mesh,
    charges: &[Charge],
    opts: &SurfaceSolveOptions,
) -> SurfaceSolveOutput {
    let mut warnings = Vec::new();
    let result = solve_surface_inner(mesh, charges, opts, &mut warnings);
    SurfaceSolveOutput { warnings, result }
}

/// Validate the solver + dielectric parameters. Centralised here so every front-end
/// (connector, CLI) accepts identical inputs — a non-positive `lambda` / `eps_*` would
/// otherwise reach `Params::yukawa()`'s `debug_assert!` (panic in debug; bad numbers in
/// release), and `restart = 0` would be silently clamped by GMRES.
fn validate_options(opts: &SurfaceSolveOptions) -> Result<(), SurfaceSolveError> {
    let cfg = &opts.cfg;
    if !(cfg.tol.is_finite() && cfg.tol > 0.0) || cfg.max_iter == 0 || cfg.restart == 0 {
        return Err(SurfaceSolveError::InvalidParams(
            "need tol > 0, max_iter > 0, and restart > 0".to_string(),
        ));
    }
    let p = &opts.params;
    for (name, v) in [
        ("eps_omega", p.eps_omega),
        ("eps_sigma", p.eps_sigma),
        ("eps_inf", p.eps_inf),
        ("lambda", p.lambda),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(SurfaceSolveError::InvalidParams(format!(
                "{name} must be finite and > 0"
            )));
        }
    }
    // Only validate the treecode params when they will actually be used (the local
    // solve). On a nonlocal solve fast_summation is ignored (warned in the inner solve),
    // so validating its unused fields would reject a combination we otherwise accept.
    if !opts.nonlocal {
        if let Some(fs) = opts.fast_summation {
            if fs.p == 0 || fs.p > MAX_FS_ORDER {
                return Err(SurfaceSolveError::InvalidParams(format!(
                    "fast_summation.p (expansion order) must be in 1..={MAX_FS_ORDER}, got {}",
                    fs.p
                )));
            }
            if !(fs.theta.is_finite() && fs.theta > 0.0 && fs.theta < 1.0) {
                return Err(SurfaceSolveError::InvalidParams(format!(
                    "fast_summation.theta must be in (0, 1), got {}",
                    fs.theta
                )));
            }
        }
    }
    Ok(())
}

fn solve_surface_inner(
    mut mesh: Mesh,
    charges: &[Charge],
    opts: &SurfaceSolveOptions,
    warnings: &mut Vec<String>,
) -> Result<SurfaceSolution, SurfaceSolveError> {
    let nf = mesh.tris.len();
    let nq = charges.len();
    if nf == 0 || nq == 0 {
        return Err(SurfaceSolveError::Empty);
    }

    validate_options(opts)?;

    // A local fast-summation request is treecode-only (O(N) memory), so it bypasses the
    // dense guard; it has no nonlocal back-end yet (P8.4) — warn and fall through to the
    // dense nonlocal solve rather than silently ignoring the request.
    let use_treecode = opts.fast_summation.is_some() && !opts.nonlocal;
    if opts.fast_summation.is_some() && opts.nonlocal {
        warnings.push(
            "fast_summation has no effect on the nonlocal solve yet (treecode is local-only, \
             P8.4); the dense nonlocal path is used."
                .to_string(),
        );
    }

    // Dense memory guard (before the per-element work, matching the original ordering) +
    // the size advisory. The treecode path is O(N) memory, so the guard does not apply.
    let blocks: u128 = if opts.nonlocal { 4 } else { 2 };
    let est = (nf as u128).saturating_mul(nf as u128).saturating_mul(8) * blocks;
    if est > MEM_BUDGET && !opts.allow_large && !use_treecode {
        return Err(SurfaceSolveError::OverBudget {
            n_elements: nf,
            gib: est >> 30,
        });
    }
    if nf >= N_WARN {
        warnings.push(if use_treecode {
            format!(
                "{nf} triangles: the treecode (fast summation) is O(N) memory but its matvec \
                 is traversal-bound — this will be slow per GMRES iteration."
            )
        } else {
            format!(
                "{nf} triangles: the dense BEM is O(N²) in memory and time — this will be \
                 slow/RAM-heavy. Over the dense budget it switches to the O(N)-memory \
                 matrix-free GPU solve if a CUDA device is present (slower per solve, but \
                 uncapped in mesh size)."
            )
        });
    }

    // Finite inputs (a NaN/inf would otherwise propagate silently into the solve).
    if mesh
        .verts
        .iter()
        .flat_map(|p| [p.x, p.y, p.z])
        .chain(charges.iter().flat_map(|c| [c.pos.x, c.pos.y, c.pos.z, c.val]))
        .any(|v| !v.is_finite())
    {
        return Err(SurfaceSolveError::NonFiniteInput);
    }

    // Triangle indices in range + each triangle non-degenerate, so `Tri::new` (which
    // asserts on a zero / non-finite normal) cannot panic downstream.
    let nv = mesh.verts.len();
    for (f, t) in mesh.tris.iter().enumerate() {
        for &idx in t {
            if idx as usize >= nv {
                return Err(SurfaceSolveError::DegenerateTriangle(f));
            }
        }
        let p = |k: u32| mesh.verts[k as usize];
        let cross = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        if !(cross.norm() > 0.0 && cross.norm().is_finite()) {
            return Err(SurfaceSolveError::DegenerateTriangle(f));
        }
    }

    // Topological acceptance + per-component auto-orient. A closed, consistently-oriented
    // mesh whose components are inside-out has the right geometry but a reversed double-layer
    // sign; flip each inward component outward-from-solute by nesting parity (body +, cavity −,
    // island +) and note it. Component orientation is only meaningful when closed + consistent.
    let topo0 = TopologyReport::assess(&mesh);
    let flipped = if topo0.watertight && topo0.consistently_oriented {
        mesh.orient_by_nesting()
    } else {
        false
    };
    if flipped {
        warnings.push(
            "one or more mesh components were re-oriented outward-from-solute (by nesting \
             parity) so the double-layer sign is correct."
                .to_string(),
        );
    }
    let topology = TopologyReport::assess(&mesh);

    // Elements honour any flip (the swap is already reflected in mesh.tris).
    let elements: Vec<Tri> = mesh
        .tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect();

    // Charge → solute-body assignment (diagnostic; the single-region solve uses one dielectric).
    let charge_components: Vec<Option<usize>> = charges
        .iter()
        .map(|c| mesh.containing_component(c.pos))
        .collect();

    // Cavities are validated for the LOCAL solve only — refuse nonlocal-on-cavity
    // UNCONDITIONALLY (a mesh-quality override must not enable unvalidated physics).
    if opts.nonlocal && topology.num_cavities > 0 {
        return Err(SurfaceSolveError::NonlocalCavity(topology.num_cavities));
    }

    // Combined mesh-acceptance: topology + per-element geometry + charge placement.
    let quality = QualityReport::assess(&elements, charges);
    let issues: Vec<_> = topology.issues().into_iter().chain(quality.issues()).collect();
    for issue in &issues {
        warnings.push(format!("mesh/charge quality: {}", issue.message));
    }
    if issues.iter().any(|i| i.severity == Severity::Error) && !opts.allow_low_quality {
        let msgs: Vec<String> = issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .map(|i| i.message.clone())
            .collect();
        return Err(SurfaceSolveError::QualityUnacceptable(msgs));
    }

    // --- linear solve + Γ-trace potential ------------------------------------
    let (cauchy, engy, stats): (Box<dyn CauchyData>, f64, _) = if opts.nonlocal {
        let (r, s) = match opts.quadrature {
            Quadrature::Adaptive(_) => solve_nonlocal_elements_q(
                &elements,
                charges,
                &opts.params,
                &opts.cfg,
                opts.quadrature,
            ),
            Quadrature::Fixed => {
                solve_nonlocal_elements_auto(&elements, charges, &opts.params, &opts.cfg)
            }
        }
        .map_err(|e| SurfaceSolveError::Solve(e.to_string()))?;
        let e = rfenergy(&elements, charges, &r);
        (Box::new(r), e, s)
    } else if let Some(fs) = opts.fast_summation {
        // Opt-in treecode local solve (O(N) memory; see FastSummation docs).
        let (r, s) =
            solve_local_elements_treecode(&elements, charges, &opts.params, &opts.cfg, fs.p, fs.theta)
                .map_err(|e| SurfaceSolveError::Solve(e.to_string()))?;
        let e = rfenergy(&elements, charges, &r);
        (Box::new(r), e, s)
    } else {
        let (r, s) = solve_local_elements_auto(&elements, charges, &opts.params, &opts.cfg)
            .map_err(|e| SurfaceSolveError::Solve(e.to_string()))?;
        let e = rfenergy(&elements, charges, &r);
        (Box::new(r), e, s)
    };

    let potential: Vec<f64> = mesh
        .verts
        .iter()
        .map(|&xi| espotential(Domain::Gamma, xi, &elements, charges, &opts.params, &*cauchy))
        .collect();

    if !engy.is_finite() || potential.iter().any(|v| !v.is_finite()) {
        return Err(SurfaceSolveError::Solve(
            "solve produced a non-finite energy / potential".to_string(),
        ));
    }

    Ok(SurfaceSolution {
        potential,
        rfenergy: engy,
        iterations: stats.iterations,
        residual: stats.residual,
        converged: stats.converged,
        quadrature: match stats.quadrature {
            Quadrature::Fixed => "fixed",
            Quadrature::Adaptive(_) => "adaptive",
        },
        capped_panels: stats.capped_panels,
        n_elements: nf,
        topology,
        quality,
        charge_components,
        flipped_to_outward: flipped,
    })
}
