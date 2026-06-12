//! Solve: rolled matrix-free GMRES + the two-stage local solve, and result types (L3).
//!
//! Decided (plan Q2): **roll our own** matrix-free GMRES for op-order control + zero
//! dep — modified Gram-Schmidt, restart, happy-breakdown, Givens rotations, **right**
//! (Jacobi) preconditioning, and a true-residual gate. Right preconditioning is chosen
//! so the residual the iteration tracks *is* the true `‖b − A·x‖` — converging on it
//! gives the gate directly. NESSie uses `IterativeSolvers.gmres` with a *left* `Pl`
//! `DiagonalPreconditioner`, so the iteration path differs, but the solution is the
//! same. The local system is well conditioned (diagonally dominant `M`, self-dominant
//! `V`) for scalar Jacobi.
//!
//! # Gates (P4)
//! - Cauchy data `u,q` vs NESSie `solve_dump` (the `:blas` fixture is the exact LU
//!   solution, so matching it validates assembly **and** solve).
//! - LU-vs-GMRES on the explicit `M` (dev-dep, test-only).
//! - true relative residual `‖M·u − b‖/‖b‖` below tolerance.

use crate::model::{BemModel, Charge, Params, Tri};
use crate::system::{
    laplace_matrices, mol_potentials, yukawa_matrices_q, JacobiPreconditioner, LinearOperator,
    Preconditioner, Quadrature, TWO_PI,
};

/// Why a solve failed (codex review: non-convergence / non-finite is a hard error,
/// not a `(result, stats{converged:false})` a caller can silently ignore).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// GMRES hit `max_iter` without reaching `tol`.
    NotConverged,
    /// A non-finite value appeared (bad geometry / parameters).
    NonFinite,
    /// The model had no surface elements.
    Empty,
    /// The model's topology is outside the validated formulation scope (e.g. a buried
    /// cavity for the nonlocal solve, which is only gated for the local solve).
    Unsupported,
    /// A solver parameter is out of range (e.g. treecode order / MAC ratio).
    BadParams(String),
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConverged => write!(f, "GMRES did not converge within max_iter"),
            Self::NonFinite => write!(f, "solve produced a non-finite value"),
            Self::Empty => write!(f, "model has no surface elements"),
            Self::Unsupported => {
                write!(
                    f,
                    "model topology is outside the validated formulation scope"
                )
            }
            Self::BadParams(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for SolveError {}

/// Shared accessor over local/nonlocal Cauchy data so [`crate::post`] is generic
/// across localities (avoids the P6 breaking change of a local-only post API).
pub trait CauchyData {
    /// `u = γ₀int(φ*)` per observation point.
    fn u(&self) -> &[f64];
    /// `q = γ₁int(φ*)` per observation point.
    fn q(&self) -> &[f64];
    /// Molecular-potential trace `umol`.
    fn umol(&self) -> &[f64];
    /// Molecular-potential normal-derivative trace `qmol`.
    fn qmol(&self) -> &[f64];
    /// Nonlocal third block `w = γ₀ext(Ψ)`, or `None` for a local solve.
    fn w(&self) -> Option<&[f64]> {
        None
    }
}

/// Result of a local solve. NESSie: `LocalBEMResult` (premultiplied by 4π·ε0).
#[derive(Debug, Clone)]
pub struct LocalResult {
    /// `u = γ₀int(φ*)`.
    pub u: Vec<f64>,
    /// `q = γ₁int(φ*)`.
    pub q: Vec<f64>,
    /// Molecular-potential trace.
    pub umol: Vec<f64>,
    /// Molecular-potential normal-derivative trace.
    pub qmol: Vec<f64>,
}

/// Result of a nonlocal solve. NESSie: `NonlocalBEMResult` (adds `w = γ₀ext Ψ`).
#[derive(Debug, Clone)]
pub struct NonlocalResult {
    /// `u = γ₀int(φ*)`.
    pub u: Vec<f64>,
    /// `q = γ₁int(φ*)`.
    pub q: Vec<f64>,
    /// `w = γ₀ext(Ψ)` — the nonlocal third block.
    pub w: Vec<f64>,
    /// Molecular-potential trace.
    pub umol: Vec<f64>,
    /// Molecular-potential normal-derivative trace.
    pub qmol: Vec<f64>,
}

impl CauchyData for LocalResult {
    fn u(&self) -> &[f64] {
        &self.u
    }
    fn q(&self) -> &[f64] {
        &self.q
    }
    fn umol(&self) -> &[f64] {
        &self.umol
    }
    fn qmol(&self) -> &[f64] {
        &self.qmol
    }
}

impl CauchyData for NonlocalResult {
    fn u(&self) -> &[f64] {
        &self.u
    }
    fn q(&self) -> &[f64] {
        &self.q
    }
    fn umol(&self) -> &[f64] {
        &self.umol
    }
    fn qmol(&self) -> &[f64] {
        &self.qmol
    }
    fn w(&self) -> Option<&[f64]> {
        Some(&self.w)
    }
}

/// GMRES configuration (explicit, not hidden constants — plan §2).
#[derive(Debug, Clone, Copy)]
pub struct SolveConfig {
    /// Relative residual tolerance.
    pub tol: f64,
    /// Restart length (NESSie default `restart = 200`).
    pub restart: usize,
    /// Iteration cap (a non-converging system must fail loudly, not hang).
    pub max_iter: usize,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            restart: 200,
            max_iter: 10_000,
        }
    }
}

/// Convergence/diagnostic info — all **measured** stats (not gates).
#[derive(Debug, Clone)]
pub struct SolveStats {
    /// GMRES iterations across both stages.
    pub iterations: usize,
    /// Worst true (unpreconditioned) relative residual over the two stages.
    pub residual: f64,
    /// True relative residual per stage (`[u-system, q-system]`).
    pub per_block_residual: Vec<f64>,
    /// Whether both stages met the residual gate.
    pub converged: bool,
    /// Regular-Yukawa quadrature used (nonlocal only; `Fixed` for local/GPU). Surfaced
    /// so an adaptive vs fixed solve is never silently confused (review [R6]).
    pub quadrature: Quadrature,
    /// Adaptive panels that hit the depth cap with the error estimate still above
    /// tolerance (`0` for `Fixed`). Non-zero ⇒ the near-singular result is **not**
    /// certified for those entries — a diagnostic the caller should heed.
    pub capped_panels: usize,
}

// ---- small dense-vector helpers ------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Stable Givens rotation `(c, s)` zeroing `b` against `a`.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        (t * s, s)
    } else {
        let t = b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, t * c)
    }
}

/// Solution of one `gmres` run.
#[derive(Debug, PartialEq)]
pub(crate) struct GmresSolution {
    pub(crate) x: Vec<f64>,
    pub(crate) iterations: usize,
}

/// Right-preconditioned restarted GMRES solving `A·x = b`. Converges on the
/// **right**-preconditioned relative residual `‖b − A·x‖ / ‖b‖` — which, for right
/// preconditioning, is exactly the *true* residual the gate cares about (left
/// preconditioning would track `‖M⁻¹(b−Ax)‖`, a different quantity). Returns
/// [`SolveError::NotConverged`] if `max_iter` is reached before `tol`, and
/// [`SolveError::NonFinite`] on a non-finite residual or a zero pivot.
///
/// `pub(crate)` so the GPU matrix-free solver ([`crate::gpu`]) can reuse the exact
/// same iteration over its own [`LinearOperator`]s.
pub(crate) fn gmres(
    op: &dyn LinearOperator,
    b: &[f64],
    precond: &dyn Preconditioner,
    cfg: &SolveConfig,
) -> Result<GmresSolution, SolveError> {
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) || cfg.max_iter == 0 {
        return Err(SolveError::NotConverged);
    }
    let n = op.dim();
    let m = cfg.restart.clamp(1, n.max(1));

    let bnorm = norm(b);
    let mut x = vec![0.0; n];
    if bnorm == 0.0 {
        return Ok(GmresSolution { x, iterations: 0 });
    }

    let mut iterations = 0;
    loop {
        // True residual r = b − A·x (right preconditioning keeps this the tracked one).
        let mut ax = vec![0.0; n];
        op.matvec(&x, &mut ax);
        let r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = norm(&r);
        if !beta.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if beta / bnorm <= cfg.tol {
            return Ok(GmresSolution { x, iterations });
        }
        if iterations >= cfg.max_iter {
            return Err(SolveError::NotConverged);
        }

        // Arnoldi on `A·M⁻¹` with v₁ = r/β; the Hessenberg least-squares residual
        // |g[j+1]| then equals ‖b − A·x‖ directly.
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|&ri| ri / beta).collect());
        let mut zs: Vec<Vec<f64>> = Vec::with_capacity(m); // M⁻¹·v_j, reused to build x
        let mut h = vec![vec![0.0_f64; m]; m + 1];
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut k = m;

        for j in 0..m {
            // w = A·M⁻¹·v_j
            let mut z = vec![0.0; n];
            precond.apply(&v[j], &mut z);
            let mut w = vec![0.0; n];
            op.matvec(&z, &mut w);
            zs.push(z);
            let wnorm0 = norm(&w); // pre-orthogonalization scale, for breakdown

            // Modified Gram–Schmidt.
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                for t in 0..n {
                    w[t] -= h[i][j] * v[i][t];
                }
            }
            let hnext = norm(&w);
            h[j + 1][j] = hnext;
            iterations += 1;

            // Lucky breakdown: the new direction is (numerically) in the span already,
            // scaled to the operator norm so it is dimensionally meaningful.
            let breakdown = hnext <= 1e-14 * wnorm0.max(1e-300);
            if !breakdown {
                v.push(w.iter().map(|&wi| wi / hnext).collect());
            }

            // Apply earlier Givens rotations to column j, then a new one zeroing h[j+1][j].
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }
            let (c, s) = givens(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = s;
            h[j][j] = c * h[j][j] + s * h[j + 1][j];
            h[j + 1][j] = 0.0;
            let temp = c * g[j] + s * g[j + 1];
            g[j + 1] = -s * g[j] + c * g[j + 1];
            g[j] = temp;

            if g[j + 1].abs() / bnorm <= cfg.tol || breakdown || iterations >= cfg.max_iter {
                k = j + 1;
                break;
            }
        }

        // Back-substitute H[0..k,0..k]·yk = g[0..k] (guard a zero pivot), then
        // x += M⁻¹·(Σ yk_i v_i) = Σ yk_i z_i.
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            if h[i][i].abs() <= 1e-300 {
                return Err(SolveError::NonFinite); // singular projected system
            }
            let mut s = g[i];
            for t in (i + 1)..k {
                s -= h[i][t] * y[t];
            }
            y[i] = s / h[i][i];
        }
        for (i, &yi) in y.iter().enumerate() {
            for t in 0..n {
                x[t] += yi * zs[i][t];
            }
        }
    }
}

/// True (unpreconditioned) relative residual `‖A·x − b‖ / ‖b‖`.
pub(crate) fn true_residual(op: &dyn LinearOperator, x: &[f64], b: &[f64]) -> f64 {
    let n = op.dim();
    let mut ax = vec![0.0; n];
    op.matvec(x, &mut ax);
    let r: Vec<f64> = (0..n).map(|i| ax[i] - b[i]).collect();
    let bn = norm(b);
    if bn == 0.0 {
        norm(&r)
    } else {
        norm(&r) / bn
    }
}

/// Core local solve over explicit triangle elements (consuming their normals
/// verbatim, for bit-parity with a fixture). NESSie `_solve_implicit(LocalES, …)`.
///
/// Two stages (`devdocs/ELECTROSTATICS_FORMULATION.md` §5):
/// 1. `M·u = b₁`, `M = 2π(1+εΩ/εΣ)I + (εΩ/εΣ−1)K`, `b₁ = (K−2π)·umol − (εΩ/εΣ)·V·qmol`.
/// 2. `V·q = b₂`, `b₂ = (2π + K)·u`.
///
/// # Errors
/// [`SolveError::Empty`] if there are no elements, [`SolveError::NotConverged`] /
/// [`SolveError::NonFinite`] from GMRES.
pub fn solve_local_elements(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Result<(LocalResult, SolveStats), SolveError> {
    if elements.is_empty() {
        return Err(SolveError::Empty);
    }
    let (v, k) = laplace_matrices(elements);
    solve_local_with_ops(&k, &v, elements, charges, params, cfg)
}

/// Local Poisson solve with the **treecode** `V`/`K` operators (P8 fast summation)
/// instead of the dense matrices — the same two-stage solve, so it cannot drift from
/// [`solve_local_elements`]. `p` is the Cartesian expansion order, `theta ∈ (0,1)` the
/// MAC ratio. The near field stays exact, so the reaction-field energy converges to the
/// dense result (and the analytic Born/Xie energy) as `(p, θ)` tighten.
pub fn solve_local_elements_treecode(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
    p: usize,
    theta: f64,
) -> Result<(LocalResult, SolveStats), SolveError> {
    use crate::fastsum::operator::{CollocationTreecode, MAX_FS_ORDER};
    use crate::model::PotentialKind;
    if elements.is_empty() {
        return Err(SolveError::Empty);
    }
    // Validate up front (return, not panic) so a direct Rust caller gets the same
    // contract the surface front-end enforces.
    if p == 0 || p > MAX_FS_ORDER {
        return Err(SolveError::BadParams(format!(
            "treecode order p must be in 1..={MAX_FS_ORDER}, got {p}"
        )));
    }
    if !(theta.is_finite() && theta > 0.0 && theta < 1.0) {
        return Err(SolveError::BadParams(format!(
            "treecode MAC ratio theta must be in (0, 1), got {theta}"
        )));
    }
    let k = CollocationTreecode::new(elements, PotentialKind::Double, p, theta);
    let v = CollocationTreecode::new(elements, PotentialKind::Single, p, theta);
    solve_local_with_ops(&k, &v, elements, charges, params, cfg)
}

/// `M_u = 2π(1 + frac)·I + (frac − 1)·K` over a borrowed `K` operator — the local
/// system operator, generic over how `K·x` is applied (dense or treecode). Mirrors
/// [`crate::system::LocalOperator`] but holds `&dyn LinearOperator` so the solve core is
/// shared by both back-ends.
struct LocalOpRef<'a> {
    k: &'a dyn LinearOperator,
    frac: f64,
}

impl LinearOperator for LocalOpRef<'_> {
    fn dim(&self) -> usize {
        self.k.dim()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.k.matvec(x, y); // y = K·x
        let diag = TWO_PI * (1.0 + self.frac);
        let off = self.frac - 1.0;
        for i in 0..y.len() {
            y[i] = diag * x[i] + off * y[i];
        }
    }
    fn diagonal(&self) -> Vec<f64> {
        vec![TWO_PI * (1.0 + self.frac); self.k.dim()]
    }
}

/// The shared two-stage local solve over abstract `V`/`K` operators (dense or treecode).
fn solve_local_with_ops(
    k: &dyn LinearOperator,
    v: &dyn LinearOperator,
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Result<(LocalResult, SolveStats), SolveError> {
    let n = elements.len();
    let frac = params.eps_omega / params.eps_sigma;

    let (umol, qmol) = mol_potentials(elements, charges, params.eps_omega);

    // Stage 1: b₁ = K·umol − 2π·umol − frac·(V·qmol).
    let mut k_umol = vec![0.0; n];
    k.matvec(&umol, &mut k_umol);
    let mut v_qmol = vec![0.0; n];
    v.matvec(&qmol, &mut v_qmol);
    let b1: Vec<f64> = (0..n)
        .map(|i| k_umol[i] - TWO_PI * umol[i] - frac * v_qmol[i])
        .collect();

    let m_op = LocalOpRef { k, frac };
    let m_pre = JacobiPreconditioner::from_operator(&m_op);
    let u_sol = gmres(&m_op, &b1, &m_pre, cfg)?;
    let u = u_sol.x;
    let res_u = true_residual(&m_op, &u, &b1);

    // Stage 2: b₂ = 2π·u + K·u; V·q = b₂.
    let mut k_u = vec![0.0; n];
    k.matvec(&u, &mut k_u);
    let b2: Vec<f64> = (0..n).map(|i| TWO_PI * u[i] + k_u[i]).collect();
    let v_pre = JacobiPreconditioner::from_operator(v);
    let q_sol = gmres(v, &b2, &v_pre, cfg)?;
    let q = q_sol.x;
    let res_q = true_residual(v, &q, &b2);

    if !u.iter().chain(&q).all(|x| x.is_finite()) {
        return Err(SolveError::NonFinite);
    }

    // `gmres` returns `Ok` only once the true relative residual is ≤ `cfg.tol`, so a
    // successful return already guarantees convergence — no silent loosening here.
    let stats = SolveStats {
        iterations: u_sol.iterations + q_sol.iterations,
        residual: res_u.max(res_q),
        per_block_residual: vec![res_u, res_q],
        converged: res_u <= cfg.tol && res_q <= cfg.tol,
        quadrature: Quadrature::Fixed, // local uses only the exact analytic Laplace path
        capped_panels: 0,
    };
    Ok((LocalResult { u, q, umol, qmol }, stats))
}

/// Above this many bytes, the dense `V`+`K` pair (`2·N²·8`) is treated as too large to
/// materialize and [`solve_local_elements_auto`] routes to the matrix-free GPU path
/// (feature `cuda`). Matches [`crate::gpu`]'s own build budget so the two decisions
/// agree; below it the cached dense matvec is faster than recomputing each step.
pub const DENSE_MATRIX_BUDGET: u128 = 7 * (1 << 30); // 7 GiB

/// Bytes a dense `V`+`K` pair needs for `n` elements: `2·N²·8`.
#[must_use]
pub fn dense_matrix_bytes(n: usize) -> u128 {
    2 * (n as u128).saturating_mul(n as u128) * 8
}

/// Size-aware local solve: the dense [`solve_local_elements`] while the `V`+`K` pair
/// fits [`DENSE_MATRIX_BUDGET`], else the O(N)-memory matrix-free GPU solve
/// ([`crate::gpu::solve_local_gpu`], feature `cuda`) — which lifts the dense memory
/// ceiling at the cost of recomputing each matvec.
///
/// The matrix-free path is tried **only** when dense would exceed the budget (dense is
/// faster when it fits). If the GPU path declines (no device, CUDA error — it returns
/// `None`), this falls back to dense, which for a very large mesh may itself exhaust
/// host RAM: that is the honest "this size needs a GPU you don't have" outcome.
///
/// # Errors
/// See [`solve_local_elements`].
pub fn solve_local_elements_auto(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Result<(LocalResult, SolveStats), SolveError> {
    #[cfg(feature = "cuda")]
    if !elements.is_empty() && dense_matrix_bytes(elements.len()) > DENSE_MATRIX_BUDGET {
        // Some(..) = the GPU path ran (Ok solution or a genuine numerical failure);
        // None = it declined (no device / CUDA error) → fall through to dense.
        if let Some(res) = crate::gpu::solve_local_gpu(elements, charges, params, cfg) {
            return res;
        }
    }
    solve_local_elements(elements, charges, params, cfg)
}

/// Triangle elements of a [`BemModel`]'s mesh (normals recomputed via `Tri::new`).
fn model_elements(model: &BemModel) -> Vec<Tri> {
    model
        .mesh
        .tris
        .iter()
        .map(|t| {
            Tri::new(
                model.mesh.verts[t[0] as usize],
                model.mesh.verts[t[1] as usize],
                model.mesh.verts[t[2] as usize],
            )
        })
        .collect()
}

/// Solve the local (Poisson) BEM system. NESSie: `solve(LocalES, model)`.
///
/// Routes through [`solve_local_elements_auto`], so a mesh too large for the dense
/// `V`+`K` transparently uses the matrix-free GPU path (feature `cuda`) when a device
/// is present; the dense solve is used otherwise.
///
/// # Errors
/// See [`solve_local_elements`].
pub fn solve_local(
    model: &BemModel,
    cfg: &SolveConfig,
) -> Result<(LocalResult, SolveStats), SolveError> {
    let elements = model_elements(model);
    solve_local_elements_auto(&elements, &model.charges, &model.params, cfg)
}

/// Core nonlocal solve over explicit triangle elements. NESSie
/// `_solve_implicit(NonlocalES, …)` (formulation spec §6): one coupled `3n` system
/// `A·[u;q;w] = [b1;0;0]`, then split.
///
/// ```text
/// b1 = K·umol + (1−εΩ/εΣ)·Ky·umol − 2π·umol − (εΩ/ε∞)·V·qmol + (εΩ/εΣ−εΩ/ε∞)·Vy·qmol
/// ```
///
/// Uses the fixed 7-point Radon quadrature for the regular-Yukawa kernels (the
/// documented near-singular floor). For the P6.5 near-singular remediation use
/// [`solve_nonlocal_elements_q`] with [`Quadrature::Adaptive`].
///
/// # Errors
/// [`SolveError::Empty`] if no elements; [`SolveError::NotConverged`] /
/// [`SolveError::NonFinite`] from GMRES.
pub fn solve_nonlocal_elements(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    solve_nonlocal_elements_q(elements, charges, params, cfg, Quadrature::Fixed)
}

/// As [`solve_nonlocal_elements`], but with a selectable regular-Yukawa quadrature.
/// `Quadrature::Adaptive` applies the near-singular subdivision remediation
/// ([`crate::adaptive`]) and reports any depth-capped panels in
/// [`SolveStats::capped_panels`].
///
/// # Errors
/// [`SolveError::Empty`] if no elements; [`SolveError::NotConverged`] /
/// [`SolveError::NonFinite`] from GMRES.
pub fn solve_nonlocal_elements_q(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
    quad: Quadrature,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    if elements.is_empty() {
        return Err(SolveError::Empty);
    }
    let yuk = params.yukawa();
    let (v, k) = laplace_matrices(elements);
    let (vy, ky, capped_panels) = yukawa_matrices_q(elements, yuk, quad);
    solve_nonlocal_with_ops(
        &v,
        &k,
        &vy,
        &ky,
        elements,
        charges,
        params,
        cfg,
        quad,
        capped_panels,
    )
}

/// Nonlocal solve with the **treecode** V/K (Cartesian) + Vy/Ky (BLTC) operators — the
/// memory-unlocking sibling of [`solve_nonlocal_elements`]. The near field uses the exact
/// fixed-quadrature collocation (so this is the fixed-quadrature path; near-singular
/// adaptive remediation is dense-only). `p`, `theta` as in
/// [`solve_local_elements_treecode`].
pub fn solve_nonlocal_elements_treecode(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
    p: usize,
    theta: f64,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    use crate::fastsum::operator::{CollocationTreecode, YukawaTreecode, MAX_FS_ORDER};
    use crate::model::PotentialKind;
    if elements.is_empty() {
        return Err(SolveError::Empty);
    }
    if p == 0 || p > MAX_FS_ORDER {
        return Err(SolveError::BadParams(format!(
            "treecode order p must be in 1..={MAX_FS_ORDER}, got {p}"
        )));
    }
    if !(theta.is_finite() && theta > 0.0 && theta < 1.0) {
        return Err(SolveError::BadParams(format!(
            "treecode MAC ratio theta must be in (0, 1), got {theta}"
        )));
    }
    let yuk = params.yukawa();
    let v = CollocationTreecode::new(elements, PotentialKind::Single, p, theta);
    let k = CollocationTreecode::new(elements, PotentialKind::Double, p, theta);
    let vy = YukawaTreecode::new(elements, PotentialKind::Single, yuk, p, theta);
    let ky = YukawaTreecode::new(elements, PotentialKind::Double, yuk, p, theta);
    solve_nonlocal_with_ops(
        &v,
        &k,
        &vy,
        &ky,
        elements,
        charges,
        params,
        cfg,
        Quadrature::Fixed,
        0,
    )
}

/// The 3-block nonlocal system operator `(u,q,w)` over borrowed V/K/Vy/Ky — mirrors
/// [`crate::system::NonlocalOperator`] but holds `&dyn LinearOperator`, so the dense and
/// treecode back-ends share the solve core. Matvec and preconditioner diagonal are the
/// same formulas (`bem/nonlocal.jl`), using the operators' `diagonal()` for the
/// preconditioner blocks.
struct NonlocalOpRef<'a> {
    v: &'a dyn LinearOperator,
    k: &'a dyn LinearOperator,
    vy: &'a dyn LinearOperator,
    ky: &'a dyn LinearOperator,
    eps_omega: f64,
    eps_sigma: f64,
    eps_inf: f64,
}

impl LinearOperator for NonlocalOpRef<'_> {
    fn dim(&self) -> usize {
        3 * self.v.dim()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.v.dim();
        let (x1, x2, x3) = (&x[0..n], &x[n..2 * n], &x[2 * n..3 * n]);
        let (eo, es, ei) = (self.eps_omega, self.eps_sigma, self.eps_inf);
        let mut kx1 = vec![0.0; n];
        self.k.matvec(x1, &mut kx1);
        let mut kx3 = vec![0.0; n];
        self.k.matvec(x3, &mut kx3);
        let mut vx2 = vec![0.0; n];
        self.v.matvec(x2, &mut vx2);
        let mut vyx2 = vec![0.0; n];
        self.vy.matvec(x2, &mut vyx2);
        let comb: Vec<f64> = (0..n).map(|i| (ei / es) * x3[i] - x1[i]).collect();
        let mut kycomb = vec![0.0; n];
        self.ky.matvec(&comb, &mut kycomb);
        for i in 0..n {
            y[i] = kycomb[i] - kx1[i]
                + (eo / ei - eo / es) * vyx2[i]
                + (eo / ei) * vx2[i]
                + TWO_PI * x1[i];
            y[n + i] = kx1[i] - vx2[i] + TWO_PI * x1[i];
            y[2 * n + i] = (eo / ei) * vx2[i] - kx3[i] + TWO_PI * x3[i];
        }
    }
    fn diagonal(&self) -> Vec<f64> {
        let n = self.v.dim();
        let kyd = self.ky.diagonal();
        let vd = self.v.diagonal();
        let mut d = Vec::with_capacity(3 * n);
        d.extend((0..n).map(|i| TWO_PI - kyd[i]));
        d.extend((0..n).map(|i| vd[i]));
        d.extend(std::iter::repeat(TWO_PI).take(n));
        d
    }
}

/// Shared 3-block nonlocal solve over abstract V/K/Vy/Ky operators (dense or treecode).
#[allow(clippy::too_many_arguments)]
fn solve_nonlocal_with_ops(
    v: &dyn LinearOperator,
    k: &dyn LinearOperator,
    vy: &dyn LinearOperator,
    ky: &dyn LinearOperator,
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
    quad: Quadrature,
    capped_panels: usize,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    let n = elements.len();
    let (eo, es, ei) = (params.eps_omega, params.eps_sigma, params.eps_inf);

    let (umol, qmol) = mol_potentials(elements, charges, eo);

    let mv = |op: &dyn LinearOperator, x: &[f64]| {
        let mut o = vec![0.0; n];
        op.matvec(x, &mut o);
        o
    };
    let (k_um, ky_um) = (mv(k, &umol), mv(ky, &umol));
    let (v_qm, vy_qm) = (mv(v, &qmol), mv(vy, &qmol));
    let mut b = vec![0.0; 3 * n];
    for i in 0..n {
        b[i] = k_um[i] + (1.0 - eo / es) * ky_um[i] - TWO_PI * umol[i] - (eo / ei) * v_qm[i]
            + (eo / es - eo / ei) * vy_qm[i];
    }

    let op = NonlocalOpRef {
        v,
        k,
        vy,
        ky,
        eps_omega: eo,
        eps_sigma: es,
        eps_inf: ei,
    };
    let pre = JacobiPreconditioner::from_operator(&op);
    let sol = gmres(&op, &b, &pre, cfg)?;
    let res = true_residual(&op, &sol.x, &b);

    let (u, q, w) = (
        sol.x[0..n].to_vec(),
        sol.x[n..2 * n].to_vec(),
        sol.x[2 * n..3 * n].to_vec(),
    );
    if !sol.x.iter().all(|x| x.is_finite()) {
        return Err(SolveError::NonFinite);
    }
    let stats = SolveStats {
        iterations: sol.iterations,
        residual: res,
        per_block_residual: vec![res],
        converged: res <= cfg.tol,
        quadrature: quad,
        capped_panels,
    };
    Ok((
        NonlocalResult {
            u,
            q,
            w,
            umol,
            qmol,
        },
        stats,
    ))
}

/// Size-aware nonlocal solve: the dense [`solve_nonlocal_elements`] while the four
/// `V`/`K`/`Vy`/`Ky` matrices fit [`DENSE_MATRIX_BUDGET`], else the O(N)-memory
/// matrix-free GPU solve ([`crate::gpu::solve_nonlocal_gpu`], feature `cuda`). The
/// nonlocal system holds *four* N×N matrices (`4·N²·8` bytes) vs the local solve's two,
/// so — storage being quadratic in N — it reaches the budget at `1/√2 ≈ 0.71×` the
/// local element count.
///
/// # Errors
/// See [`solve_nonlocal_elements`].
pub fn solve_nonlocal_elements_auto(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    #[cfg(feature = "cuda")]
    if !elements.is_empty() {
        let bytes = 4 * (elements.len() as u128).saturating_mul(elements.len() as u128) * 8;
        if bytes > DENSE_MATRIX_BUDGET {
            if let Some(res) = crate::gpu::solve_nonlocal_gpu(elements, charges, params, cfg) {
                return res;
            }
        }
    }
    solve_nonlocal_elements(elements, charges, params, cfg)
}

/// Solve the nonlocal (Yukawa) 3-block BEM system. NESSie: `solve(NonlocalES, model)`.
///
/// Routes through [`solve_nonlocal_elements_auto`], so a mesh too large for the dense
/// four-matrix system transparently uses the matrix-free GPU path (feature `cuda`).
///
/// **Cavities are refused here** ([`SolveError::Unsupported`]): the nonlocal formulation
/// on buried cavities is not yet validated (only the *local* cavity solve is gated). The
/// model-level API has the topology to check; the element-level
/// [`solve_nonlocal_elements`] cannot, so callers passing a cavity element list directly
/// are responsible for that precondition.
///
/// # Errors
/// [`SolveError::Unsupported`] for a nested (cavity) mesh; otherwise see
/// [`solve_nonlocal_elements`].
pub fn solve_nonlocal(
    model: &BemModel,
    cfg: &SolveConfig,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    if model.mesh.num_nested_components() > 0 {
        return Err(SolveError::Unsupported);
    }
    let elements = model_elements(model);
    solve_nonlocal_elements_auto(&elements, &model.charges, &model.params, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{DenseOperator, JacobiPreconditioner, NonlocalOperator};

    #[test]
    fn nonlocal_opref_equals_nonlocal_operator() {
        // The refactor's NonlocalOpRef (borrowed &dyn) must be operation-identical to the
        // existing NonlocalOperator (owned DenseOperators) — same matvec and same
        // preconditioner diagonal — so the dense nonlocal path cannot drift.
        let n = 4;
        let mk = |seed: usize| {
            let mut a = DenseOperator::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    a.set(i, j, (((i * 7 + j * 3 + seed) % 11) as f64 - 5.0) * 0.1);
                }
            }
            a
        };
        let (v, k, vy, ky) = (mk(0), mk(1), mk(2), mk(3));
        let (eo, es, ei) = (1.0, 78.0, 1.8);
        let owned = NonlocalOperator {
            v: v.clone(),
            k: k.clone(),
            vy: vy.clone(),
            ky: ky.clone(),
            eps_omega: eo,
            eps_sigma: es,
            eps_inf: ei,
        };
        let borrowed = NonlocalOpRef {
            v: &v,
            k: &k,
            vy: &vy,
            ky: &ky,
            eps_omega: eo,
            eps_sigma: es,
            eps_inf: ei,
        };

        let x: Vec<f64> = (0..3 * n).map(|i| ((i % 5) as f64) - 2.0).collect();
        let mut yo = vec![0.0; 3 * n];
        let mut yb = vec![0.0; 3 * n];
        owned.matvec(&x, &mut yo);
        borrowed.matvec(&x, &mut yb);
        assert_eq!(yo, yb, "matvec must be bit-identical");
        assert_eq!(
            owned.diagonal(),
            borrowed.diagonal(),
            "preconditioner diagonal identical"
        );
    }

    fn op_3x3() -> DenseOperator {
        // Diagonally dominant, non-symmetric — a well-posed small system.
        let rows = [[4.0, 1.0, 0.0], [-1.0, 5.0, 1.0], [0.0, 2.0, 6.0]];
        let mut a = DenseOperator::zeros(3);
        for (i, r) in rows.iter().enumerate() {
            for (j, &val) in r.iter().enumerate() {
                a.set(i, j, val);
            }
        }
        a
    }

    #[test]
    fn gmres_converges_to_true_solution() {
        let a = op_3x3();
        let b = [1.0, 2.0, 3.0];
        let pre = JacobiPreconditioner::from_operator(&a);
        let cfg = SolveConfig {
            tol: 1e-13,
            restart: 10,
            max_iter: 100,
        };
        let sol = gmres(&a, &b, &pre, &cfg).expect("converge");
        // The returned solution must actually solve A·x = b (not just stop early).
        assert!(true_residual(&a, &sol.x, &b) <= 1e-13);
        let mut ax = vec![0.0; 3];
        a.matvec(&sol.x, &mut ax);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-11, "row {i}");
        }
    }

    #[test]
    fn gmres_rejects_bad_config() {
        let a = op_3x3();
        let b = [1.0, 2.0, 3.0];
        let pre = JacobiPreconditioner::from_operator(&a);
        // max_iter == 0 and tol <= 0 must fail, not silently return a wrong/zero x.
        for cfg in [
            SolveConfig {
                tol: 1e-10,
                restart: 10,
                max_iter: 0,
            },
            SolveConfig {
                tol: 0.0,
                restart: 10,
                max_iter: 100,
            },
        ] {
            assert_eq!(gmres(&a, &b, &pre, &cfg), Err(SolveError::NotConverged));
        }
    }

    #[test]
    fn dispatcher_equals_dense_below_budget() {
        // The size-aware `solve_local_elements_auto` must equal the explicit dense
        // solve when the mesh fits the budget (always true without the `cuda` feature),
        // so wiring it into `solve_local` cannot regress the common path.
        use crate::analytic::analytic_sphere_mesh;
        use proteon_core::surface::geom::Vec3;

        let mesh = analytic_sphere_mesh(2.0, 1); // 80 triangles
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
        assert!(dense_matrix_bytes(elements.len()) <= DENSE_MATRIX_BUDGET);
        let charges = [Charge {
            pos: Vec3::new(0.2, -0.3, 0.1),
            val: 1.0,
        }];
        let params = Params {
            eps_omega: 1.0,
            eps_sigma: 78.0,
            eps_inf: 1.8,
            lambda: 20.0,
        };
        let cfg = SolveConfig {
            tol: 1e-9,
            ..Default::default()
        };
        let (auto, _) = solve_local_elements_auto(&elements, &charges, &params, &cfg).unwrap();
        let (dense, _) = solve_local_elements(&elements, &charges, &params, &cfg).unwrap();
        assert_eq!(auto.u, dense.u);
        assert_eq!(auto.q, dense.q);
    }

    #[test]
    fn gmres_not_converged_when_capped_below_need() {
        // A system that needs more than `max_iter` iterations must report
        // NotConverged, never a silently-wrong Ok.
        let a = op_3x3();
        let b = [1.0, 2.0, 3.0];
        let pre = JacobiPreconditioner::from_operator(&a);
        let cfg = SolveConfig {
            tol: 1e-15,
            restart: 1, // restart(1) + 1 iter caps progress hard
            max_iter: 1,
        };
        assert_eq!(gmres(&a, &b, &pre, &cfg), Err(SolveError::NotConverged));
    }
}
