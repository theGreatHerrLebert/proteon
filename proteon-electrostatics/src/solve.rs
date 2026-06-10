//! Solve: rolled matrix-free GMRES + preconditioner, and result types (L3).
//!
//! Decided (plan Q2): **roll our own** matrix-free GMRES for op-order control +
//! zero dep — but budgeted as real solver code, **not 150 lines**: modified
//! Gram-Schmidt, restart, happy-breakdown, residual replacement, stagnation
//! handling, diagnostics. Start with scalar Jacobi (mirrors NESSie); the
//! [`crate::system::Preconditioner`] trait leaves room for a block-diagonal one.
//!
//! # Gates (P4)
//! - Cauchy data `u,q,[w]` vs NESSie `solve_dump`.
//! - Gate the **true (unpreconditioned) per-block residual** below tol — not
//!   iteration count (that is a *measured* stat).
//! - Solver correctness via dense LU/QR of the explicit matrix vs GMRES on small
//!   fixtures (dev-dependency, test-only).

use crate::model::{BemModel, Locality};
use crate::system::{assemble, BemSystem};

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
            tol: 1e-8,
            restart: 200,
            max_iter: 10_000,
        }
    }
}

/// Convergence/diagnostic info — all **measured** stats (not gates).
#[derive(Debug, Clone)]
pub struct SolveStats {
    /// GMRES iterations to convergence.
    pub iterations: usize,
    /// Final true (unpreconditioned) relative residual (global).
    pub residual: f64,
    /// True relative residual per block (length = `num_blocks`).
    pub per_block_residual: Vec<f64>,
    /// Whether the residual gate was met (false ⟹ do not trust the result).
    pub converged: bool,
}

/// Solve the local (Poisson) BEM system. NESSie: `solve(LocalES, model)`.
///
/// TODO(P4): assemble via [`assemble`], run the rolled GMRES, package Cauchy data.
pub fn solve_local(model: &BemModel, cfg: &SolveConfig) -> (LocalResult, SolveStats) {
    let _system = assemble(model, Locality::Local);
    unimplemented!("P4: rolled matrix-free GMRES + local Cauchy data (src/bem/local.jl)")
}

/// Solve the nonlocal (Yukawa) 3-block BEM system. NESSie: `solve(NonlocalES, model)`.
///
/// TODO(P6): nonlocal assembly + solve; gate `nonlocal → local` limit invariant.
pub fn solve_nonlocal(model: &BemModel, cfg: &SolveConfig) -> (NonlocalResult, SolveStats) {
    let _system = assemble(model, Locality::Nonlocal);
    unimplemented!("P6: nonlocal 3-block solve (src/bem/nonlocal.jl)")
}

/// Matrix-free restarted GMRES over an assembled [`BemSystem`].
///
/// TODO(P4): real implementation (MGS, restart, breakdown, residual replacement).
/// Returns the solution and the **true** per-block residuals for gating.
pub(crate) fn gmres(system: &BemSystem, cfg: &SolveConfig) -> (Vec<f64>, SolveStats) {
    unimplemented!("P4: roll matrix-free GMRES — not 150 lines (plan §2)")
}
