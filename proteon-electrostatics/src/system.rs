//! BEM system assembly — collocation matrices + the implicit local operator (L3).
//!
//! Port of NESSie's `BEM` assembly (`src/bem/{local,implicit}.jl`). The local solve
//! is two sequential `numelem × numelem` systems (`M·u = b₁`, then `V·q = b₂`), not
//! a coupled block system; this module builds the pieces and [`solve`](crate::solve)
//! drives the two stages.
//!
//! The block structure, RHS, dielectric factors, and ½-jump (`2π`) terms are pinned
//! by `devdocs/ELECTROSTATICS_FORMULATION.md` §5.
//!
//! **Scaling ceiling:** the dense `K`/`V` are O(N²) memory; each matvec is O(N²)
//! time without a fast-summation method (FMM/treecode). Fine at fixture scale; the
//! O(N) matrix-free `K·x` and a fast summation backend are the plan §6 follow-ups.

use crate::laplace::{laplace_collocation, ETOL_F64};
use crate::model::{Charge, PotentialKind, Tri};
use crate::yukawa::regular_yukawa_collocation;

/// `2π = 4π·σ` with NESSie's `σ = 1/2` — the ½-solid-angle jump constant.
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// A matrix-free linear operator: `y ← A·x`. The GMRES in [`crate::solve`] consumes
/// this. Each `matvec` is O(N²) without fast summation (plan §6).
pub trait LinearOperator {
    /// Side length of the (square) system.
    fn dim(&self) -> usize;
    /// `y = A·x` (`x`, `y` length [`Self::dim`]).
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    /// The operator's diagonal (for the Jacobi preconditioner).
    fn diagonal(&self) -> Vec<f64>;
}

/// A dense row-major `n × n` matrix as a [`LinearOperator`] — the collocation
/// matrices `V` (single layer) and `K` (double layer).
#[derive(Debug, Clone)]
pub struct DenseOperator {
    /// Side length.
    pub n: usize,
    /// Row-major entries (`data[i*n + j] = A[i][j]`).
    pub data: Vec<f64>,
}

impl DenseOperator {
    /// `n × n` zero matrix.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }
    /// `A[i][j]`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }
    /// Set `A[i][j]`.
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }
}

impl LinearOperator for DenseOperator {
    fn dim(&self) -> usize {
        self.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let row = &self.data[i * self.n..(i + 1) * self.n];
            let mut acc = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                acc += row[j] * xj;
            }
            y[i] = acc;
        }
    }
    fn diagonal(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.get(i, i)).collect()
    }
}

/// The implicit first local system operator `M_u` (NESSie `LocalSystemMatrix`):
/// `M·x = 2π(1 + εΩ/εΣ)·x + (εΩ/εΣ − 1)·(K·x)`, `diag(M) = 2π(1 + εΩ/εΣ)`.
/// Matrix-free over the stored `K` (the only O(N²) object).
#[derive(Debug, Clone)]
pub struct LocalOperator {
    /// Double-layer collocation matrix `K`.
    pub k: DenseOperator,
    /// Dielectric ratio `εΩ/εΣ`.
    pub frac: f64,
}

impl LinearOperator for LocalOperator {
    fn dim(&self) -> usize {
        self.k.n
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
        vec![TWO_PI * (1.0 + self.frac); self.k.n]
    }
}

/// Build the single- and double-layer Laplace collocation matrices `(V, K)` over
/// the element centroids `Ξ` (NESSie `_get_laplace_matrices`). `V[i][j]` / `K[i][j]`
/// are the single/double-layer collocation of element `j` at centroid of element `i`.
#[must_use]
pub fn laplace_matrices(elements: &[Tri]) -> (DenseOperator, DenseOperator) {
    let n = elements.len();
    let centroids: Vec<_> = elements
        .iter()
        .map(|e| (e.v1 + e.v2 + e.v3) * (1.0 / 3.0))
        .collect();
    let mut v = DenseOperator::zeros(n);
    let mut k = DenseOperator::zeros(n);
    for (i, &xi) in centroids.iter().enumerate() {
        for (j, ej) in elements.iter().enumerate() {
            v.set(i, j, laplace_collocation(PotentialKind::Single, xi, ej));
            k.set(i, j, laplace_collocation(PotentialKind::Double, xi, ej));
        }
    }
    (v, k)
}

/// Build the regular single/double-layer Yukawa collocation matrices `(Vy, Ky)` over
/// the element centroids at exponent `yukawa` (NESSie `_get_yukawa_matrices`).
#[must_use]
pub fn yukawa_matrices(elements: &[Tri], yukawa: f64) -> (DenseOperator, DenseOperator) {
    let n = elements.len();
    let centroids: Vec<_> = elements
        .iter()
        .map(|e| (e.v1 + e.v2 + e.v3) * (1.0 / 3.0))
        .collect();
    let mut vy = DenseOperator::zeros(n);
    let mut ky = DenseOperator::zeros(n);
    for (i, &xi) in centroids.iter().enumerate() {
        for (j, ej) in elements.iter().enumerate() {
            vy.set(
                i,
                j,
                regular_yukawa_collocation(PotentialKind::Single, xi, ej, yukawa),
            );
            ky.set(
                i,
                j,
                regular_yukawa_collocation(PotentialKind::Double, xi, ej, yukawa),
            );
        }
    }
    (vy, ky)
}

/// The implicit nonlocal 3-block operator `A·[x1;x2;x3]` (NESSie `NonlocalSystemMatrix`,
/// formulation spec §6). `x1=u`, `x2=q`, `x3=w`; `V,K` Laplace, `Vy,Ky` regular Yukawa.
///
/// ```text
/// row1 = Ky·((ε∞/εΣ)x3 − x1) − K·x1 + (εΩ/ε∞ − εΩ/εΣ)·(Vy·x2) + (εΩ/ε∞)·(V·x2) + 2π·x1
/// row2 = K·x1 − V·x2 + 2π·x1
/// row3 = (εΩ/ε∞)·(V·x2) − K·x3 + 2π·x3
/// ```
#[derive(Debug, Clone)]
pub struct NonlocalOperator {
    /// Single-layer Laplace.
    pub v: DenseOperator,
    /// Double-layer Laplace.
    pub k: DenseOperator,
    /// Regular single-layer Yukawa.
    pub vy: DenseOperator,
    /// Regular double-layer Yukawa.
    pub ky: DenseOperator,
    /// Solute dielectric `εΩ`.
    pub eps_omega: f64,
    /// Solvent dielectric `εΣ`.
    pub eps_sigma: f64,
    /// Bulk solvent response `ε∞`.
    pub eps_inf: f64,
}

impl LinearOperator for NonlocalOperator {
    fn dim(&self) -> usize {
        3 * self.v.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.v.n;
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
        // Ky·((ε∞/εΣ)·x3 − x1).
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
    /// NESSie's **preconditioner** diagonal `[2π − diag(Ky); +diag(V); 2π]`
    /// (`bem/nonlocal.jl:210`). Note the middle block is `+diag(V)`, *not* the
    /// algebraic matrix diagonal `−diag(V)` — see spec §6. Since [`crate::solve`]'s
    /// GMRES converges on the true residual, the preconditioner only affects speed,
    /// so this faithful choice is safe.
    fn diagonal(&self) -> Vec<f64> {
        let n = self.v.n;
        let mut d = Vec::with_capacity(3 * n);
        d.extend((0..n).map(|i| TWO_PI - self.ky.get(i, i)));
        d.extend((0..n).map(|i| self.v.get(i, i)));
        d.extend(std::iter::repeat(TWO_PI).take(n));
        d
    }
}

/// Molecular potential traces at the element centroids (NESSie `_molpotential` /
/// `_molpotential_dn`, divided by `εΩ`):
///
/// ```text
/// umol_i = (1/εΩ) · Σ_c  q_c / max(|ξ_i − r_c|, U_TOL)
/// qmol_i = −(1/εΩ) · Σ_c  q_c · (ξ_i − r_c)·n_i / max(|ξ_i − r_c|³, Q_TOL)
/// ```
///
/// The degenerate-distance guards match NESSie's **implicit** local path: `umol`
/// uses `_etol` (`1.45e-8`), `qmol` the default `1e-10` applied to the *cubed*
/// distance. For non-coincident charges (the usual case) neither triggers.
#[must_use]
pub fn mol_potentials(
    elements: &[Tri],
    charges: &[Charge],
    eps_omega: f64,
) -> (Vec<f64>, Vec<f64>) {
    const Q_TOL: f64 = 1e-10; // applied to |ξ−r|³
    let u_tol = ETOL_F64; // applied to |ξ−r| (NESSie `_molpotential(..., tolerance=_etol)`)
    let inv = 1.0 / eps_omega;
    let mut umol = vec![0.0; elements.len()];
    let mut qmol = vec![0.0; elements.len()];
    for (i, e) in elements.iter().enumerate() {
        let center = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
        let mut u = 0.0;
        let mut q = 0.0;
        for c in charges {
            let d = center - c.pos;
            let r = d.norm();
            u += c.val / r.max(u_tol);
            // ddot(center, pos, n) = (center − pos)·n
            q += c.val * d.dot(e.normal) / (r * r * r).max(Q_TOL);
        }
        umol[i] = inv * u;
        qmol[i] = -inv * q;
    }
    (umol, qmol)
}

/// Block structure of the system: `num_blocks · num_elements` unknowns. Local solves
/// use two single-block stages; the nonlocal 3-block system (P6) is one coupled
/// system where this indexes the `(u, q, w)` blocks.
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
    /// Half-open index range of block `b`.
    #[must_use]
    pub fn block_range(&self, b: usize) -> std::ops::Range<usize> {
        let n = self.num_elements;
        (b * n)..((b + 1) * n)
    }
}

/// Preconditioner `z ← M⁻¹·r`. Scalar Jacobi to start (mirrors NESSie's
/// `DiagonalPreconditioner`); the trait leaves room for a block-diagonal one.
pub trait Preconditioner {
    /// Apply `z = M⁻¹·r`.
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// Scalar Jacobi: `z_i = r_i / diag_i` (NESSie `DiagonalPreconditioner`).
pub struct JacobiPreconditioner {
    /// System diagonal.
    pub diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Build from an operator's diagonal.
    #[must_use]
    pub fn from_operator(op: &dyn LinearOperator) -> Self {
        Self {
            diag: op.diagonal(),
        }
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for i in 0..r.len() {
            z[i] = r[i] / self.diag[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteon_core::surface::geom::Vec3;

    #[test]
    fn local_operator_matches_explicit_matvec() {
        // The implicit LocalOperator must equal an explicitly assembled M = 2π(1+f)I
        // + (f−1)K applied densely (operator-parity, separate from any oracle).
        let n = 4;
        let mut k = DenseOperator::zeros(n);
        for i in 0..n {
            for j in 0..n {
                k.set(i, j, (i as f64 + 1.0) * 0.1 - (j as f64) * 0.07);
            }
        }
        let frac = 0.013;
        let op = LocalOperator { k: k.clone(), frac };

        // explicit M
        let diag = TWO_PI * (1.0 + frac);
        let mut m = k.clone();
        for i in 0..n {
            for j in 0..n {
                let kv = k.get(i, j);
                m.set(i, j, (frac - 1.0) * kv + if i == j { diag } else { 0.0 });
            }
        }
        let x = [0.3, -1.2, 0.7, 2.1];
        let mut y_op = vec![0.0; n];
        let mut y_ex = vec![0.0; n];
        op.matvec(&x, &mut y_op);
        m.matvec(&x, &mut y_ex);
        for i in 0..n {
            assert!(
                (y_op[i] - y_ex[i]).abs() < 1e-13,
                "{i}: {} vs {}",
                y_op[i],
                y_ex[i]
            );
        }
        assert!(op.diagonal().iter().all(|&d| (d - diag).abs() < 1e-13));
    }

    #[test]
    fn mol_potential_single_charge() {
        // One unit charge at the origin; a single element whose centroid is at
        // distance 2 along +z with +z normal. umol = (1/εΩ)·q/r; qmol = −(1/εΩ)·q·(c−p)·n/r³.
        let t = Tri::new(
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(1.0, -1.0, 2.0),
            Vec3::new(1.0, 1.0, 2.0),
        );
        let centroid = (t.v1 + t.v2 + t.v3) * (1.0 / 3.0); // z = 2
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, 0.0),
            val: 1.0,
        }];
        let eps_omega = 2.0;
        let (umol, qmol) = mol_potentials(std::slice::from_ref(&t), &charges, eps_omega);
        let r = centroid.norm();
        assert!((umol[0] - (1.0 / eps_omega) / r).abs() < 1e-12);
        let expected_q = -(1.0 / eps_omega) * centroid.dot(t.normal) / (r * r * r);
        assert!((qmol[0] - expected_q).abs() < 1e-12);
    }
}
