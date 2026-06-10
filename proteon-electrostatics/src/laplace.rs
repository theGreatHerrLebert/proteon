//! Rjasanow analytic single/double-layer Laplace collocation (L1).
//!
//! Port of NESSie's `Rjasanow` module (`src/Rjasanow.jl`): the **analytic** Laplace
//! potential of a triangle at an observation point ξ, via projection onto the
//! element plane and the InPlane/InSpace closed forms. The InPlane form **is the
//! self/diagonal term** (ξ on its own element); the ½ solid-angle jump is the `σ`
//! constant added at assembly, not here.
//!
//! The arithmetic mirrors NESSie operation-for-operation so the values match the
//! `collocation_dump` fixture to libm precision (asin/log/sqrt may differ by ≤1 ULP
//! between openlibm and Rust's libm; everything else is bit-identical).
//!
//! # Gates (P2)
//! - per-element value vs NESSie `collocation_dump` (`tests/laplace_parity.rs`)
//!   **and** an independent numerical-quadrature oracle at non-singular points
//!   (`tests/laplace_quadrature.rs`) — catches a transcription error the fixture
//!   parity would share. The near-singular numerical corpus is a follow-up.
//! - metamorphic: rigid-motion invariance; cyclic vertex permutation invariant,
//!   **odd** permutation flips the double-layer sign; InPlane double layer = 0.
//!
//! Result is **premultiplied by 4π** (NESSie convention; see §1b unit chain): the
//! single-layer value is `∫_T 1/|ξ−r'| dA`, the double-layer value is
//! `∫_T (ξ−r')·n / |ξ−r'|³ dA`.
//!
//! Tolerances are type-specific in NESSie (`_etol`: 1.45e-8 f64). proteon is f64.

use crate::model::PotentialKind;
use proteon_core::surface::geom::Vec3;

/// Common Laplace tolerance (NESSie `_etol` for f64).
pub const ETOL_F64: f64 = 1.45e-8;

/// A flat triangle plus the geometric props the Rjasanow kernel needs. Mirrors
/// NESSie's `Triangle` (`v1`, `v2`, `v3`, unit `normal`, `distorig = normal·v1`).
///
/// Vertices must be counter-clockwise wrt. the outward `normal` (NESSie convention);
/// the double layer's sign depends on it.
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
}

impl Tri {
    /// Build from three vertices, computing the normal and `distorig` exactly as
    /// NESSie's `props`: `normal = (v2−v1)×(v3−v1)` normalized, `distorig = normal·v1`.
    ///
    /// # Panics
    /// If the three vertices are collinear (zero-area / non-normalizable normal).
    #[must_use]
    pub fn new(v1: Vec3, v2: Vec3, v3: Vec3) -> Self {
        let normal = (v2 - v1)
            .cross(v3 - v1)
            .normalized()
            .expect("degenerate triangle: collinear vertices");
        Self {
            v1,
            v2,
            v3,
            normal,
            distorig: normal.dot(v1),
        }
    }

    /// Build with an explicit, already-normalized `normal` consumed verbatim (e.g. a
    /// mesh/fixture normal), so the geometry is bit-identical to the source rather
    /// than recomputed. `distorig = normal·v1`, as in NESSie `props`.
    #[must_use]
    pub fn with_normal(v1: Vec3, v2: Vec3, v3: Vec3, normal: Vec3) -> Self {
        Self {
            v1,
            v2,
            v3,
            normal,
            distorig: normal.dot(v1),
        }
    }
}

/// Julia `sign`: `-1`/`0`/`+1` (note: differs from `f64::signum`, which never
/// returns `0` and maps `±0.0` to `±1.0`).
#[inline]
fn jsign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `cos∠(u, v)` given pre-computed norms (NESSie `_cos`).
#[inline]
fn cos_between(u: Vec3, v: Vec3, unorm: f64, vnorm: f64) -> f64 {
    u.dot(v) / (unorm * vnorm)
}

/// Cathetus `c₁` from hypotenuse and `cos θ` (NESSie `cathetus`): `√(hyp²(1−cos²θ))`.
#[inline]
fn cathetus(hyp: f64, cos_theta: f64) -> f64 {
    (hyp * hyp * (1.0 - cos_theta * cos_theta)).sqrt()
}

/// `sign((u × v)·n)` (NESSie `_sign`): are the spanned and given normals aligned?
#[inline]
fn orient_sign(u: Vec3, v: Vec3, n: Vec3) -> f64 {
    jsign(u.cross(v).dot(n))
}

/// NESSie `_logterm`: `(√(1−χ²sin²φ) + √(1−χ²)·sinφ) / (√(1−χ²sin²φ) − √(1−χ²)·sinφ)`.
#[inline]
fn logterm(chi2: f64, sinphi: f64) -> f64 {
    let term1 = (1.0 - chi2 * sinphi * sinphi).sqrt();
    let term2 = (1.0 - chi2).sqrt() * sinphi;
    (term1 + term2) / (term1 - term2)
}

/// Innermost closed form: `_laplacepot(ptype, InPlane|InSpace, sinφ1, sinφ2, h, d)`.
/// `in_plane` selects the singular self/in-plane branch (`|dist| < etol`).
#[inline]
fn laplacepot_closed(
    kind: PotentialKind,
    in_plane: bool,
    sinphi1: f64,
    sinphi2: f64,
    h: f64,
    d: f64,
) -> f64 {
    match (kind, in_plane) {
        // SingleLayer / InPlane — h/2 · ln((1+sφ2)(1−sφ1)/((1−sφ2)(1+sφ1)))
        (PotentialKind::Single, true) => {
            h * ((1.0 + sinphi2) * (1.0 - sinphi1) / ((1.0 - sinphi2) * (1.0 + sinphi1))).ln() / 2.0
        }
        // SingleLayer / InSpace — h/2·<1> + d·<2>, χ² = d²/(d²+h²)
        (PotentialKind::Single, false) => {
            let d = d.abs();
            let chi2 = d * d / (d * d + h * h);
            let chi = chi2.sqrt();
            let result = h * (logterm(chi2, sinphi2) / logterm(chi2, sinphi1)).ln() / 2.0;
            result
                + d * ((chi * sinphi2).asin() - sinphi2.asin() - (chi * sinphi1).asin()
                    + sinphi1.asin())
        }
        // DoubleLayer / InPlane — (ξ−r')⊥n ⟹ 0.
        (PotentialKind::Double, true) => 0.0,
        // DoubleLayer / InSpace — sign(d)·(asin(χ sφ1) − asin(sφ1) − asin(χ sφ2) + asin(sφ2))
        (PotentialKind::Double, false) => {
            let chi = d.abs() / (d * d + h * h).sqrt();
            jsign(d)
                * ((chi * sinphi1).asin() - sinphi1.asin() - (chi * sinphi2).asin()
                    + sinphi2.asin())
        }
    }
}

/// One sub-triangle `(ξ, x1, x2)` contribution (NESSie `_laplacepot(ptype, ξ, x1, x2, …)`).
/// `x2` is `x1`'s CCW neighbour. `ξ` is already projected onto the element plane.
#[inline]
fn laplacepot_edge(
    kind: PotentialKind,
    xi: Vec3,
    x1: Vec3,
    x2: Vec3,
    normal: Vec3,
    dist: f64,
) -> f64 {
    let u1 = x1 - xi;
    let u2 = x2 - xi;
    let v = x2 - x1;

    let u1norm = u1.norm();
    let u2norm = u2.norm();
    let vnorm = v.norm();

    let sinphi1 = cos_between(u1, v, u1norm, vnorm).clamp(-1.0, 1.0);
    let sinphi2 = cos_between(u2, v, u2norm, vnorm).clamp(-1.0, 1.0);

    let h = cathetus(u1norm, sinphi1);

    // Degenerate sub-triangles (ξ on the line through x1,x2; or x1,x2 colinear with ξ).
    if h.max(0.0) < ETOL_F64
        || 1.0 - sinphi1.abs() < ETOL_F64
        || 1.0 - sinphi2.abs() < ETOL_F64
        || (sinphi1 - sinphi2).abs() < ETOL_F64
    {
        return 0.0;
    }

    let in_plane = dist.abs() < ETOL_F64;
    orient_sign(u1, u2, normal) * laplacepot_closed(kind, in_plane, sinphi1, sinphi2, h, dist)
}

/// Analytic single/double-layer Laplace collocation of triangle `tri` at ξ.
///
/// Mirrors NESSie `laplacecoll(ptype, ξ, elem)`: signed plane distance, projection
/// onto the element plane, then the sum of the three edge sub-triangles. Result is
/// **premultiplied by 4π** (see module docs / §1b unit chain).
#[must_use]
pub fn laplace_collocation(kind: PotentialKind, xi: Vec3, tri: &Tri) -> f64 {
    // distance(ξ, elem) = ξ·n − distorig.
    let dist = xi.dot(tri.normal) - tri.distorig;

    // Project ξ onto the element plane (only when off-plane).
    let xi_p = if dist.abs() >= ETOL_F64 {
        xi - tri.normal * dist
    } else {
        xi
    };

    laplacepot_edge(kind, xi_p, tri.v1, tri.v2, tri.normal, dist)
        + laplacepot_edge(kind, xi_p, tri.v2, tri.v3, tri.normal, dist)
        + laplacepot_edge(kind, xi_p, tri.v3, tri.v1, tri.normal, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equilateral() -> Tri {
        // Side 2, in the z=0 plane, centroid at origin.
        let a = Vec3::new(-1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
        let b = Vec3::new(1.0, -1.0 / 3.0_f64.sqrt(), 0.0);
        let c = Vec3::new(0.0, 2.0 / 3.0_f64.sqrt(), 0.0);
        Tri::new(a, b, c)
    }

    #[test]
    fn new_computes_props_like_nessie() {
        let t = equilateral();
        // CCW in z=0 ⟹ +z normal; distorig = n·v1 = 0 (plane through origin).
        assert!((t.normal - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-12);
        assert!(t.distorig.abs() < 1e-12);
    }

    #[test]
    fn double_layer_in_plane_is_zero() {
        // ξ in the element plane ⟹ double layer vanishes (every edge InPlane = 0).
        let t = equilateral();
        let v = laplace_collocation(PotentialKind::Double, Vec3::new(0.1, 0.2, 0.0), &t);
        assert!(v.abs() < 1e-14, "in-plane double layer = {v}");
    }

    #[test]
    fn rigid_motion_invariance() {
        // Single-layer value is invariant under a rigid motion of (triangle, ξ).
        let t = equilateral();
        let xi = Vec3::new(0.2, -0.1, 0.7);
        let base = laplace_collocation(PotentialKind::Single, xi, &t);

        // Rotate 90° about z then translate.
        let rot = |p: Vec3| Vec3::new(-p.y, p.x, p.z) + Vec3::new(3.0, -2.0, 1.5);
        let tr = Tri::new(rot(t.v1), rot(t.v2), rot(t.v3));
        let moved = laplace_collocation(PotentialKind::Single, rot(xi), &tr);
        assert!((base - moved).abs() < 1e-12, "{base} vs {moved}");
    }

    #[test]
    fn odd_vertex_permutation_flips_double_layer() {
        // Swapping two vertices reverses orientation ⟹ double layer flips sign;
        // single layer is unchanged.
        let t = equilateral();
        let xi = Vec3::new(0.2, -0.1, 0.7);
        let swapped = Tri::new(t.v2, t.v1, t.v3);

        let d0 = laplace_collocation(PotentialKind::Double, xi, &t);
        let d1 = laplace_collocation(PotentialKind::Double, xi, &swapped);
        assert!(
            (d0 + d1).abs() < 1e-12,
            "double not anti-symmetric: {d0} vs {d1}"
        );

        let s0 = laplace_collocation(PotentialKind::Single, xi, &t);
        let s1 = laplace_collocation(PotentialKind::Single, xi, &swapped);
        assert!(
            (s0 - s1).abs() < 1e-12,
            "single not symmetric: {s0} vs {s1}"
        );
    }

    #[test]
    fn cyclic_permutation_invariant() {
        // Cyclic (even) vertex rotation preserves orientation ⟹ both layers unchanged.
        let t = equilateral();
        let xi = Vec3::new(0.2, -0.1, 0.7);
        let cyc = Tri::new(t.v2, t.v3, t.v1);
        for k in [PotentialKind::Single, PotentialKind::Double] {
            let a = laplace_collocation(k, xi, &t);
            let b = laplace_collocation(k, xi, &cyc);
            assert!((a - b).abs() < 1e-12, "{k:?}: {a} vs {b}");
        }
    }
}
