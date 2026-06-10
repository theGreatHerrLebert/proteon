//! Solve: rolled matrix-free GMRES + the two-stage local solve, and result types (L3).
//!
//! Decided (plan Q2): **roll our own** matrix-free GMRES for op-order control + zero
//! dep — modified Gram-Schmidt, restart, happy-breakdown, Givens rotations, left
//! (Jacobi) preconditioning, and a **true (unpreconditioned) residual** gate. NESSie
//! uses `IterativeSolvers.gmres` + `DiagonalPreconditioner`; the local system is well
//! enough conditioned (diagonally dominant `M`, self-dominant `V`) for scalar Jacobi.
//!
//! # Gates (P4)
//! - Cauchy data `u,q` vs NESSie `solve_dump` (the `:blas` fixture is the exact LU
//!   solution, so matching it validates assembly **and** solve).
//! - LU-vs-GMRES on the explicit `M` (dev-dep, test-only).
//! - true relative residual `‖M·u − b‖/‖b‖` below tolerance.

use crate::model::{BemModel, Charge, Params, Tri};
use crate::system::{
    laplace_matrices, mol_potentials, JacobiPreconditioner, LinearOperator, LocalOperator,
    Preconditioner, TWO_PI,
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
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConverged => write!(f, "GMRES did not converge within max_iter"),
            Self::NonFinite => write!(f, "solve produced a non-finite value"),
            Self::Empty => write!(f, "model has no surface elements"),
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
struct GmresSolution {
    x: Vec<f64>,
    iterations: usize,
}

/// Left-preconditioned restarted GMRES solving `A·x = b`. Converges on the
/// preconditioned relative residual; the caller checks the true residual for the
/// gate. Returns [`SolveError::NotConverged`] if `max_iter` is hit first.
fn gmres(
    op: &dyn LinearOperator,
    b: &[f64],
    precond: &dyn Preconditioner,
    cfg: &SolveConfig,
) -> Result<GmresSolution, SolveError> {
    let n = op.dim();
    let m = cfg.restart.clamp(1, n.max(1));

    // Preconditioned rhs norm (= initial residual norm for x0 = 0).
    let mut mb = vec![0.0; n];
    precond.apply(b, &mut mb);
    let bnorm = norm(&mb);
    let mut x = vec![0.0; n];
    if bnorm == 0.0 {
        return Ok(GmresSolution { x, iterations: 0 });
    }

    let mut iterations = 0;
    loop {
        // r = M⁻¹(b − A·x)
        let mut ax = vec![0.0; n];
        op.matvec(&x, &mut ax);
        let resid: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let mut mr = vec![0.0; n];
        precond.apply(&resid, &mut mr);
        let beta = norm(&mr);
        if !beta.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if beta / bnorm <= cfg.tol {
            return Ok(GmresSolution { x, iterations });
        }

        // Arnoldi basis + Hessenberg (column-major access via h[i][j]).
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(mr.iter().map(|&r| r / beta).collect());
        let mut h = vec![vec![0.0_f64; m]; m + 1];
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut k = m;

        for j in 0..m {
            // w = M⁻¹·A·v_j
            let mut avj = vec![0.0; n];
            op.matvec(&v[j], &mut avj);
            let mut w = vec![0.0; n];
            precond.apply(&avj, &mut w);

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

            let breakdown = hnext <= 1e-14 * beta.max(1.0);
            if !breakdown {
                v.push(w.iter().map(|&wi| wi / hnext).collect());
            }

            // Apply earlier Givens rotations to column j.
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }
            // New rotation zeroing h[j+1][j].
            let (c, s) = givens(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = s;
            h[j][j] = c * h[j][j] + s * h[j + 1][j];
            h[j + 1][j] = 0.0;
            let temp = c * g[j] + s * g[j + 1];
            g[j + 1] = -s * g[j] + c * g[j + 1];
            g[j] = temp;

            if g[j + 1].abs() / bnorm <= cfg.tol || breakdown {
                k = j + 1;
                break;
            }
            if iterations >= cfg.max_iter {
                k = j + 1;
                break;
            }
        }

        // Back-substitute H[0..k,0..k]·y = g[0..k], then x += Σ y_i v_i.
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for t in (i + 1)..k {
                s -= h[i][t] * y[t];
            }
            y[i] = s / h[i][i];
        }
        for (i, &yi) in y.iter().enumerate() {
            for t in 0..n {
                x[t] += yi * v[i][t];
            }
        }

        if iterations >= cfg.max_iter {
            return Err(SolveError::NotConverged);
        }
    }
}

/// True (unpreconditioned) relative residual `‖A·x − b‖ / ‖b‖`.
fn true_residual(op: &dyn LinearOperator, x: &[f64], b: &[f64]) -> f64 {
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
    let n = elements.len();
    if n == 0 {
        return Err(SolveError::Empty);
    }
    let frac = params.eps_omega / params.eps_sigma;

    let (umol, qmol) = mol_potentials(elements, charges, params.eps_omega);
    let (v, k) = laplace_matrices(elements);

    // Stage 1: b₁ = K·umol − 2π·umol − frac·(V·qmol).
    let mut k_umol = vec![0.0; n];
    k.matvec(&umol, &mut k_umol);
    let mut v_qmol = vec![0.0; n];
    v.matvec(&qmol, &mut v_qmol);
    let b1: Vec<f64> = (0..n)
        .map(|i| k_umol[i] - TWO_PI * umol[i] - frac * v_qmol[i])
        .collect();

    let m_op = LocalOperator { k: k.clone(), frac };
    let m_pre = JacobiPreconditioner::from_operator(&m_op);
    let u_sol = gmres(&m_op, &b1, &m_pre, cfg)?;
    let u = u_sol.x;
    let res_u = true_residual(&m_op, &u, &b1);

    // Stage 2: b₂ = 2π·u + K·u; V·q = b₂.
    let mut k_u = vec![0.0; n];
    k.matvec(&u, &mut k_u);
    let b2: Vec<f64> = (0..n).map(|i| TWO_PI * u[i] + k_u[i]).collect();
    let v_pre = JacobiPreconditioner::from_operator(&v);
    let q_sol = gmres(&v, &b2, &v_pre, cfg)?;
    let q = q_sol.x;
    let res_q = true_residual(&v, &q, &b2);

    if !u.iter().chain(&q).all(|x| x.is_finite()) {
        return Err(SolveError::NonFinite);
    }

    let stats = SolveStats {
        iterations: u_sol.iterations + q_sol.iterations,
        residual: res_u.max(res_q),
        per_block_residual: vec![res_u, res_q],
        converged: res_u <= cfg.tol.max(1e-6) && res_q <= cfg.tol.max(1e-6),
    };
    Ok((LocalResult { u, q, umol, qmol }, stats))
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
/// # Errors
/// See [`solve_local_elements`].
pub fn solve_local(
    model: &BemModel,
    cfg: &SolveConfig,
) -> Result<(LocalResult, SolveStats), SolveError> {
    let elements = model_elements(model);
    solve_local_elements(&elements, &model.charges, &model.params, cfg)
}

/// Solve the nonlocal (Yukawa) 3-block BEM system. NESSie: `solve(NonlocalES, model)`.
///
/// TODO(P6): nonlocal assembly + solve; gate `nonlocal → local` limit invariant.
///
/// # Errors
/// Not yet implemented.
pub fn solve_nonlocal(
    _model: &BemModel,
    _cfg: &SolveConfig,
) -> Result<(NonlocalResult, SolveStats), SolveError> {
    unimplemented!("P6: nonlocal 3-block solve (src/bem/nonlocal.jl)")
}
