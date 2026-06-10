//! BEM system assembly — implicit (matrix-free) 2-block / 3-block operators (L3).
//!
//! Port of NESSie's `BEM` assembly (`src/bem/{local,nonlocal,implicit}.jl`). The
//! local system is 2-block `(u, q)`; the nonlocal is 3-block `(u, q, w)` of size
//! `3·numelem`. Production uses the **implicit** (matrix-free) operator (O(N)
//! memory); an **explicit** dense assembly exists for small-fixture parity.
//!
//! The exact block structure, RHS vectors, dielectric factors, and ½-jump terms
//! are pinned by the §1b formulation spec — **do not infer them here**.
//!
//! # Gates (P4), three distinct checks — don't conflate:
//! 1. *operator parity* — `implicit·x == explicit·x` (indexing/matvec only).
//! 2. *assembly correctness* — proteon's blocks + RHS **entrywise** vs NESSie
//!    `system_dump`. (A dense LU of our own matrix can NOT validate assembly.)
//! 3. *solver correctness* — handled in `solve.rs` (LU vs GMRES).

use crate::model::{BemModel, Locality};

/// A matrix-free linear operator: `y ← A·x`. The GMRES in `solve.rs` consumes this.
///
/// **Scaling ceiling:** each `matvec` is O(N²) without a fast-summation method
/// (FMM/treecode/H-matrix). O(N) memory, O(N²) time — see plan §6. Keep this
/// interface open to a fast-summation backend.
pub trait LinearOperator {
    /// Side length of the (square) system.
    fn dim(&self) -> usize;
    /// In-place matvec `y = A·x` (`x`, `y` length [`Self::dim`]).
    fn matvec(&self, x: &[f64], y: &mut [f64]);
}

/// Block structure of the system: `num_blocks · num_elements` unknowns, with the
/// per-block ranges GMRES uses for per-block residuals and a block preconditioner.
#[derive(Debug, Clone, Copy)]
pub struct BlockLayout {
    /// Surface elements (= collocation points).
    pub num_elements: usize,
    /// 2 for local `(u,q)`, 3 for nonlocal `(u,q,w)`.
    pub num_blocks: usize,
}

impl BlockLayout {
    /// Total system dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.num_blocks * self.num_elements
    }
    /// Half-open index range of block `b` within a length-[`Self::dim`] vector.
    #[must_use]
    pub fn block_range(&self, b: usize) -> std::ops::Range<usize> {
        let n = self.num_elements;
        (b * n)..((b + 1) * n)
    }
}

/// Preconditioner `z ← M⁻¹·r`. Scalar Jacobi to start (mirrors NESSie); the trait
/// leaves room for a **block-diagonal** preconditioner without reshaping callers
/// (plan §2 — Jacobi is weak on refined / nonlocal systems).
pub trait Preconditioner {
    /// Apply `z = M⁻¹·r`.
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// Scalar Jacobi: `z_i = r_i / diag_i`.
pub struct JacobiPreconditioner {
    /// System diagonal (NESSie `DiagonalPreconditioner`).
    pub diag: Vec<f64>,
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        unimplemented!("P4: scalar Jacobi apply")
    }
}

/// Assembled BEM system: the matrix-free operator, RHS, preconditioner, and block
/// layout — everything `solve::gmres` needs. The operator is boxed so a
/// fast-summation backend can replace the naive O(N²) matvec later.
pub struct BemSystem {
    /// Matrix-free system operator.
    pub op: Box<dyn LinearOperator>,
    /// Right-hand side (the molecular-potential source terms).
    pub rhs: Vec<f64>,
    /// Preconditioner (boxed so Jacobi → block-diagonal is non-breaking).
    pub precond: Box<dyn Preconditioner>,
    /// Block structure (for per-block residuals + block preconditioning).
    pub layout: BlockLayout,
}

/// Assemble the implicit local (2-block) or nonlocal (3-block) system + RHS.
///
/// TODO(P4): port `BEM._solve_implicit` assembly (the implicit block operators
/// over `Rjasanow`/`Radon` collocation) and the molecular-potential RHS.
#[must_use]
pub fn assemble(model: &BemModel, locality: Locality) -> BemSystem {
    unimplemented!("P4: port BEM implicit system assembly (src/bem/)")
}
