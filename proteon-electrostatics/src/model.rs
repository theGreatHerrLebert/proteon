//! Core types: charges, system parameters, the BEM model, and dispatch tags.
//!
//! Mirrors NESSie's `Charge`, `Option`, and `Model` (`src/base/{model,constants}.jl`).
//! These plain data types are fully implemented; the numerics that consume them
//! are stubbed in sibling modules.

use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;

/// Single/double layer — the BEM kernel/operator selector. NESSie: `PotentialType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotentialKind {
    /// Single-layer (`V`) — the weakly-singular kernel.
    Single,
    /// Double-layer (`K`) — the normal-derivative kernel. The collocation value is
    /// the **principal value only**; the ½ solid-angle jump (`2π`) is added during
    /// system assembly, **not** baked into the kernel (NESSie `Rjasanow`; see §1b).
    Double,
}

/// Local vs nonlocal electrostatics. NESSie: `LocalityType` (`LocalES`/`NonlocalES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    /// Classical local dielectric (Poisson) — 2-block system.
    Local,
    /// Nonlocal (Lorentz cavity / Yukawa) — 3-block system, the differentiator.
    Nonlocal,
}

/// Spatial domain for potential evaluation. NESSie: `:Ω` / `:Σ` / `:Γ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Solute interior (Ω).
    Omega,
    /// Solvent exterior (Σ).
    Sigma,
    /// Molecular surface (Γ).
    Gamma,
}

/// A point charge in the solute. NESSie: `Charge{T}` (`pos`, `val`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Charge {
    /// Position (Å).
    pub pos: Vec3,
    /// Charge value (elementary charges).
    pub val: f64,
}

/// System parameters. NESSie: `Option{T}` (`εΩ`, `εΣ`, `ε∞`, `λ`).
///
/// Note: `eps_sigma == eps_inf` + `lambda → 0` is **not** a clean operator-level
/// collapse of nonlocal → local (the `w` block survives and the regular parts tend
/// to `Vy → −V`, `Ky → −K`, leaving a degenerate 3-field system). Only the limiting
/// physical *potential* may converge, after eliminating `w` — gate that, not an
/// operator identity. See `ELECTROSTATICS_FORMULATION.md` §10. NESSie `defaultopt`:
/// `εΩ=2, εΣ=78, ε∞=1.8, λ=20 Å`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Dielectric constant of the solute (εΩ).
    pub eps_omega: f64,
    /// Dielectric constant of the solvent (εΣ).
    pub eps_sigma: f64,
    /// Large-scale (bulk) solvent response (ε∞).
    pub eps_inf: f64,
    /// Correlation length scale λ (Å).
    pub lambda: f64,
}

impl Params {
    /// Yukawa exponent of the nonlocal fundamental solution: `√(εΣ/ε∞)/λ`.
    /// NESSie: `yukawa(opt) = √(εΣ/ε∞)/λ` (`base/constants.jl`).
    ///
    /// Precondition: `eps_inf > 0`, `eps_sigma >= 0`, `lambda > 0` — otherwise the
    /// result is non-finite. Enforced by `debug_assert` for now; a validated
    /// `Params::new -> Result` is the P4 hardening (plan §6 physical-input domain).
    #[must_use]
    pub fn yukawa(&self) -> f64 {
        debug_assert!(self.eps_inf > 0.0, "eps_inf must be > 0");
        debug_assert!(self.eps_sigma >= 0.0, "eps_sigma must be >= 0");
        debug_assert!(self.lambda > 0.0, "lambda must be > 0");
        (self.eps_sigma / self.eps_inf).sqrt() / self.lambda
    }
}

/// A flat triangle plus the geometric props the collocation kernels need. Mirrors
/// NESSie's `Triangle` (`v1`, `v2`, `v3`, unit `normal`, `distorig = normal·v1`,
/// `area`). Shared by the Laplace ([`crate::laplace`]) and Yukawa ([`crate::yukawa`])
/// kernels (the latter needs `area` and `normal` for the Radon cubature).
///
/// Vertices must be counter-clockwise wrt. the outward `normal` (NESSie convention);
/// the double-layer sign depends on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tri {
    /// First vertex.
    pub v1: Vec3,
    /// Second vertex (CCW after `v1`).
    pub v2: Vec3,
    /// Third vertex (CCW after `v2`).
    pub v3: Vec3,
    /// Unit outward normal.
    pub normal: Vec3,
    /// Signed plane–origin distance, `normal·v1` (NESSie `props`).
    pub distorig: f64,
    /// Triangle area, `|(v2−v1)×(v3−v1)| / 2` (NESSie `props`).
    pub area: f64,
}

impl Tri {
    /// Build from three vertices, computing the props exactly as NESSie's `props`:
    /// `normal = (v2−v1)×(v3−v1) / |·|` (plain division), `distorig = normal·v1`,
    /// `area = |(v2−v1)×(v3−v1)| / 2`.
    ///
    /// NESSie divides by the cross-product norm with no epsilon floor (it guards
    /// degeneracy separately). proteon-core's `Vec3::normalized()` rejects norms
    /// below `1e-6` — stricter than NESSie, and it would drop valid small mesh
    /// triangles — so this divides directly and only rejects a zero / non-finite
    /// normal.
    ///
    /// # Panics
    /// If the three vertices are collinear (zero-area / non-finite normal).
    #[must_use]
    pub fn new(v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        let cross = (v2 - v1).cross(v3 - v1);
        let vnorm = cross.norm();
        assert!(
            vnorm > 0.0 && vnorm.is_finite(),
            "degenerate triangle: zero-area / non-finite normal"
        );
        let normal = cross * (1.0 / vnorm);
        Self {
            v1,
            v2,
            v3,
            normal,
            distorig: normal.dot(v1),
            area: vnorm / 2.0,
        }
    }

    /// Build with an explicit, already-normalized `normal` consumed verbatim (e.g. a
    /// mesh/fixture normal), so the geometry is bit-identical to the source rather
    /// than recomputed. `distorig = normal·v1` and `area = |(v2−v1)×(v3−v1)| / 2`,
    /// as in NESSie `props`.
    #[must_use]
    pub fn with_normal(v1: Vec3, v2: Vec3, v3: Vec3, normal: Vec3) -> Self {
        Self {
            v1,
            v2,
            v3,
            normal,
            distorig: normal.dot(v1),
            area: (v2 - v1).cross(v3 - v1).norm() / 2.0,
        }
    }
}

/// A BEM problem instance: surface mesh + charges + parameters.
///
/// The mesh is proteon's own index `Mesh` (from `proteon-core::surface`), produced
/// by the SES mesher or by [`crate::analytic::analytic_sphere_mesh`]. Observation
/// points for collocation are the triangle centroids.
#[derive(Debug, Clone)]
pub struct BemModel {
    /// Triangulated molecular surface (centroids = collocation points).
    pub mesh: Mesh,
    /// Point charges, pre-assigned to this solute component (plan §1b: charge→region).
    pub charges: Vec<Charge>,
    /// Dielectric / nonlocal parameters.
    pub params: Params,
}
