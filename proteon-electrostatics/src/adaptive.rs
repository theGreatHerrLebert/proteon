//! P6.5 near-singular remediation for the regular-Yukawa collocation.
//!
//! The fixed 7-point Radon cubature ([`crate::quadrature`]) under-resolves the regular
//! Yukawa kernel when the observation point is near a panel relative to its size — the
//! documented "Radon floor" that caps nonlocal Born convergence at a few percent. The
//! kernel is finite (the `1/r` singularity cancels) but **not smooth**: its expansion
//! `(e^{−κr}−1)/r = −κ + κ²r/2 − …` contains `r = |x−ξ|`, which has a cusp at the
//! source, so polynomial cubature loses its high-order accuracy near the panel.
//!
//! The remediation (design: `devdocs/NEAR_SINGULAR_QUADRATURE.md`) is adaptive panel
//! subdivision with a deterministic resolution floor and a centroid fan for the
//! self/near-self panel. This module is built bottom-up and gated per piece; the first
//! piece is the geometric near-field trigger.

use crate::laplace::ETOL_F64;
use crate::model::{PotentialKind, Tri};
use crate::yukawa::regular_yukawa_collocation_parts;
use proteon_core::surface::geom::Vec3;

/// Exact distance from point `p` to the closest point of triangle `(v1, v2, v3)`
/// (Ericson, *Real-Time Collision Detection*, §5.1.5). Projects onto the plane and,
/// when the projection falls outside, clamps to the nearest edge or vertex — so it is
/// correct whether the closest feature is the face interior, an edge, or a vertex.
#[must_use]
pub fn point_to_triangle_distance(p: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> f64 {
    (closest_point_on_triangle(p, v1, v2, v3) - p).norm()
}

/// Closest point on segment `[a, b]` to `p` (with a degenerate-segment guard).
fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let denom = ab.dot(ab);
    if denom <= 0.0 {
        return a; // a == b
    }
    let t = (ab.dot(p - a) / denom).clamp(0.0, 1.0);
    a + ab * t
}

/// Closest point of a **degenerate** (zero-area) triangle to `p`: the nearest point over
/// its three edges as segments. Keeps `point_to_triangle_distance` finite on collinear
/// vertices that `Tri::with_normal` does not reject (the Voronoi divisions would
/// otherwise divide by zero — review [P2]).
fn closest_point_degenerate(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    [
        closest_point_on_segment(p, a, b),
        closest_point_on_segment(p, b, c),
        closest_point_on_segment(p, c, a),
    ]
    .into_iter()
    .min_by(|x, y| {
        (*x - p)
            .norm()
            .partial_cmp(&(*y - p).norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
    .unwrap_or(a)
}

/// Closest point of triangle `(a, b, c)` to `p` (barycentric Voronoi-region method).
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    // Degenerate (collinear / zero-area) triangle ⇒ the Voronoi divisions below can
    // divide by zero; fall back to the three edges as segments.
    if ab.cross(ac).norm() <= 0.0 {
        return closest_point_degenerate(p, a, b, c);
    }
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    // Vertex region B.
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return closest_point_on_segment(p, a, b);
    }
    // Vertex region C.
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return closest_point_on_segment(p, a, c);
    }
    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return closest_point_on_segment(p, b, c);
    }
    // Face interior — barycentric combination (denominator > 0, triangle non-degenerate).
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Characteristic size of a triangle for the near-field test: its **longest edge**
/// (diameter proxy). Deliberately *not* `√(2·area)`, which understates skinny triangles
/// (review [R2]).
#[must_use]
pub fn longest_edge(tri: &Tri) -> f64 {
    let e1 = (tri.v2 - tri.v1).norm();
    let e2 = (tri.v3 - tri.v2).norm();
    let e3 = (tri.v1 - tri.v3).norm();
    e1.max(e2).max(e3)
}

/// Empirically-calibrated near-field trigger: an observation point `xi` is "near" a
/// triangle when its distance is within `NEAR_FACTOR` characteristic sizes. This is a
/// *trigger*, not an accuracy guarantee — far-field acceptance is certified by the
/// micro-corpus gate, not by this predicate (review [R2]).
pub const NEAR_FACTOR: f64 = 4.0;

/// Whether `xi` is in the near field of `tri` (→ adaptive subdivision) or far (→ a
/// single fixed Radon eval suffices).
#[must_use]
pub fn is_near(xi: Vec3, tri: &Tri) -> bool {
    let d = point_to_triangle_distance(xi, tri.v1, tri.v2, tri.v3);
    d < NEAR_FACTOR * longest_edge(tri)
}

// ---- adaptive subdivision -----------------------------------------------------------

/// Outcome of an adaptive panel integration: did the estimator converge, or did it hit
/// the depth cap with the error still above tolerance? A `Capped` result is still the
/// best (most-refined) value, but it is **not** certified — the caller counts these and
/// surfaces them rather than silently trusting a capped matrix (review [R3]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The coarse-vs-refined estimate fell within tolerance.
    Converged,
    /// The depth cap was reached with the estimate still above tolerance.
    Capped,
}

impl Status {
    /// `Capped` if either is `Capped` (a panel is only converged if all its parts are).
    fn combine(self, other: Self) -> Self {
        if self == Self::Capped || other == Self::Capped {
            Self::Capped
        } else {
            Self::Converged
        }
    }
}

/// Adaptive-quadrature tuning. The defaults are the corpus-calibrated starting point;
/// `min_depth`/`near_factor` are validated by the estimator-effectivity gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    /// Relative tolerance on the coarse-vs-refined estimate (scaled to the local
    /// non-cancelling magnitude `Σ|wᵢfᵢ|`).
    pub rtol: f64,
    /// Hard recursion cap (4^depth leaf panels worst case).
    pub max_depth: u32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            max_depth: 6,
        }
    }
}

/// Deterministic resolution floor (review [R1]): refine to at least this depth before
/// trusting the difference estimator, because the cusp at `r → 0` defeats a bare
/// coarse-vs-refined test (false convergence). Deeper when the panel is geometrically
/// close (`d/h` small) or large on the Yukawa scale (`κh` big).
fn min_depth(d_over_h: f64, kh: f64, max_depth: u32) -> u32 {
    let mut depth = 0;
    if d_over_h < 2.0 {
        depth = 1;
    }
    if d_over_h < 1.0 {
        depth = 2;
    }
    if d_over_h < 0.5 {
        depth = 3;
    }
    if kh > 1.0 {
        depth += 1;
    }
    depth.min(max_depth)
}

/// Per-operator absolute tolerance floor for a panel of area `A` at Yukawa scale `κ`,
/// guarding the estimator where the (double-layer) value crosses zero (review [R5]).
/// The single-layer regular kernel is O(κ); the double-layer is O(κ²) — hence the
/// separate scales.
fn atol_panel(kind: PotentialKind, area: f64, kappa: f64) -> f64 {
    // Loose floor: a tiny fraction of (kernel magnitude bound × panel area). Kept small
    // so it only dominates very near a zero crossing, not in the bulk.
    let bound = match kind {
        PotentialKind::Single => kappa,
        PotentialKind::Double => kappa * kappa,
    };
    1e-12 * bound * area.max(f64::MIN_POSITIVE)
}

/// Midpoint (4-way) subdivision: each edge midpoint splits `tri` into four congruent,
/// **coplanar** sub-triangles, so they share the parent normal (passed verbatim, never
/// recomputed) and the double-layer sign is preserved. Vertex order keeps each child CCW.
fn midpoint_split(tri: &Tri) -> [Tri; 4] {
    let (v1, v2, v3, n) = (tri.v1, tri.v2, tri.v3, tri.normal);
    let m12 = (v1 + v2) * 0.5;
    let m23 = (v2 + v3) * 0.5;
    let m31 = (v3 + v1) * 0.5;
    [
        Tri::with_normal(v1, m12, m31, n),
        Tri::with_normal(m12, v2, m23, n),
        Tri::with_normal(m31, m23, v3, n),
        Tri::with_normal(m12, m23, m31, n), // central medial triangle (CCW)
    ]
}

/// Centroid fan about an **interior** point `p` (on the panel plane): three coplanar
/// sub-triangles each with the cusp at a vertex — the bounded special case for the
/// self/near-self panel where the cusp's projection lands inside the face (review [R4]).
fn centroid_fan(tri: &Tri, p: Vec3) -> [Tri; 3] {
    let (v1, v2, v3, n) = (tri.v1, tri.v2, tri.v3, tri.normal);
    [
        Tri::with_normal(p, v1, v2, n),
        Tri::with_normal(p, v2, v3, n),
        Tri::with_normal(p, v3, v1, n),
    ]
}

/// Orthogonal projection of `xi` onto `tri`'s plane, returned only when it lands in the
/// face **interior** (all barycentric coords ≥ 0) — i.e. the cusp sits over the panel,
/// the case the fan handles. Edge/vertex closest points return `None` (midpoint
/// recursion toward that boundary suffices).
fn interior_projection(xi: Vec3, tri: &Tri) -> Option<Vec3> {
    let signed = (xi - tri.v1).dot(tri.normal);
    let proj = xi - tri.normal * signed;
    // Barycentric of proj w.r.t. (v1, v2, v3).
    let (e1, e2, ep) = (tri.v2 - tri.v1, tri.v3 - tri.v1, proj - tri.v1);
    let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
    let (d1p, d2p) = (e1.dot(ep), e2.dot(ep));
    let denom = d11 * d22 - d12 * d12;
    if denom <= 0.0 {
        return None; // degenerate
    }
    let v = (d22 * d1p - d12 * d2p) / denom;
    let w = (d11 * d2p - d12 * d1p) / denom;
    if v >= 0.0 && w >= 0.0 && v + w <= 1.0 {
        Some(proj)
    } else {
        None
    }
}

/// Adaptive regular-Yukawa collocation of `tri` at `xi` (exponent `yukawa`): the
/// near-singular remediation for the fixed [`regular_yukawa_collocation`]. Returns the
/// signed collocation (premultiplied by 4π, same convention) and a [`Status`].
///
/// Strategy (design `devdocs/NEAR_SINGULAR_QUADRATURE.md`):
/// 1. **On-panel** (`d ≤ ETOL`, ξ numerically on the panel) → the existing analytic-limit
///    fixed value. The cusp lies *on* the integration domain there, where pure
///    subdivision cannot converge (it would cap); a polar/graded rule is documented
///    future work. This is a **direct-call safety net**: in the matrix assembly the self
///    term is handled by index identity (`j == i`), not this distance test, so a genuine
///    sub-ETOL cleft can never be misclassified as self (a mesh has no overlapping
///    non-self panels anyway).
/// 2. **Far** panel → one fixed Radon eval (bit-identical to the non-adaptive path).
/// 3. Cusp **interior** but off-panel (`d > ETOL`; the cleft / opposing-surface case) →
///    centroid fan, then resolution-floor recursion on each fan triangle.
/// 4. Otherwise (closest point on an edge/vertex) → resolution-floor recursion directly.
#[must_use]
pub fn adaptive_regular_yukawa_collocation(
    kind: PotentialKind,
    xi: Vec3,
    tri: &Tri,
    yukawa: f64,
    cfg: &AdaptiveConfig,
) -> (f64, Status) {
    let d = point_to_triangle_distance(xi, tri.v1, tri.v2, tri.v3);
    // On-panel self/coincident term — keep the analytic-limit fixed value (see above).
    if d <= ETOL_F64 {
        return (
            regular_yukawa_collocation_parts(kind, xi, tri, yukawa).0,
            Status::Converged,
        );
    }
    if d >= NEAR_FACTOR * longest_edge(tri) {
        return (
            regular_yukawa_collocation_parts(kind, xi, tri, yukawa).0,
            Status::Converged,
        );
    }
    // Cusp over the face interior but off-panel (cleft) → fan, else plain recursion.
    if let Some(p) = interior_projection(xi, tri) {
        let mut value = 0.0;
        let mut status = Status::Converged;
        for sub in centroid_fan(tri, p) {
            // Skip a fan triangle that is degenerate (p on an edge → zero area).
            if sub.area <= f64::MIN_POSITIVE {
                continue;
            }
            let (v, s) = adaptive_recurse(kind, xi, &sub, yukawa, 0, cfg);
            value += v;
            status = status.combine(s);
        }
        return (value, status);
    }
    adaptive_recurse(kind, xi, tri, yukawa, 0, cfg)
}

/// Resolution-floor recursion: force-refine to [`min_depth`], then use the coarse-vs-
/// refined estimator, recursing only on still-near children until converged or capped.
fn adaptive_recurse(
    kind: PotentialKind,
    xi: Vec3,
    tri: &Tri,
    yukawa: f64,
    depth: u32,
    cfg: &AdaptiveConfig,
) -> (f64, Status) {
    let d = point_to_triangle_distance(xi, tri.v1, tri.v2, tri.v3);
    let h = longest_edge(tri);
    let kids = midpoint_split(tri);

    // Forced refinement below the deterministic floor (no estimator yet).
    if depth < min_depth(d / h, yukawa * h, cfg.max_depth) {
        let mut value = 0.0;
        let mut status = Status::Converged;
        for kid in &kids {
            let (v, s) = adaptive_recurse(kind, xi, kid, yukawa, depth + 1, cfg);
            value += v;
            status = status.combine(s);
        }
        return (value, status);
    }

    // Estimator: compare one eval over `tri` against the sum over its four children.
    let coarse = regular_yukawa_collocation_parts(kind, xi, tri, yukawa).0;
    let (mut refined, mut mag) = (0.0, 0.0);
    for kid in &kids {
        let (v, m) = regular_yukawa_collocation_parts(kind, xi, kid, yukawa);
        refined += v;
        mag += m;
    }
    let est = (refined - coarse).abs();
    let tol = cfg.rtol * mag + atol_panel(kind, tri.area, yukawa);
    if est <= tol {
        return (refined, Status::Converged);
    }
    if depth >= cfg.max_depth {
        return (refined, Status::Capped);
    }

    // Recurse on near children; far children take a single eval.
    let mut value = 0.0;
    let mut status = Status::Converged;
    for kid in &kids {
        if is_near(xi, kid) {
            let (v, s) = adaptive_recurse(kind, xi, kid, yukawa, depth + 1, cfg);
            value += v;
            status = status.combine(s);
        } else {
            value += regular_yukawa_collocation_parts(kind, xi, kid, yukawa).0;
        }
    }
    (value, status)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force closest distance by dense barycentric sampling — an independent
    /// reference for `point_to_triangle_distance`.
    fn brute_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3, m: usize) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..=m {
            for j in 0..=(m - i) {
                let u = i as f64 / m as f64;
                let v = j as f64 / m as f64;
                let q = a + (b - a) * u + (c - a) * v; // u+v ≤ 1 ⇒ inside
                best = best.min((q - p).norm());
            }
        }
        best
    }

    fn tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        )
    }

    #[test]
    fn distance_matches_bruteforce_all_feature_regions() {
        let (a, b, c) = tri();
        // Points whose closest feature is, respectively: face interior, above-face,
        // each vertex, each edge, and far outside.
        let pts = [
            Vec3::new(0.3, 0.3, 0.0),   // inside (distance 0)
            Vec3::new(0.4, 0.4, 1.7),   // above the face interior
            Vec3::new(-1.0, -1.0, 0.5), // vertex A region
            Vec3::new(3.5, -0.5, 0.0),  // vertex B region
            Vec3::new(-0.5, 3.5, 0.4),  // vertex C region
            Vec3::new(1.0, -1.0, 0.3),  // edge AB region
            Vec3::new(-1.0, 1.0, 0.2),  // edge AC region
            Vec3::new(2.0, 2.0, 0.6),   // edge BC region
        ];
        for p in pts {
            let exact = point_to_triangle_distance(p, a, b, c);
            let brute = brute_distance(p, a, b, c, 400);
            // Brute sampling slightly overestimates (discrete grid); exact ≤ brute and
            // close. Tolerance scales with the grid spacing.
            assert!(
                exact <= brute + 1e-9 && (brute - exact) < 0.02,
                "p={p:?}: exact={exact} brute={brute}"
            );
        }
    }

    #[test]
    fn distance_finite_on_degenerate_triangle() {
        // Collinear vertices (zero area) — the Voronoi divisions would divide by zero, so
        // the degenerate fallback (closest over the three edges) must keep the result
        // finite and correct (review [P2]).
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(4.0, 0.0, 0.0); // collinear with a, b
        for (p, want) in [
            (Vec3::new(1.0, 1.0, 0.0), 1.0),   // above the segment interior
            (Vec3::new(5.0, 0.0, 0.0), 1.0),   // beyond c → distance to c
            (Vec3::new(-1.0, 0.0, 0.0), 1.0),  // before a → distance to a
            (Vec3::new(3.0, 0.0, 0.0), 0.0),   // on the collinear span
        ] {
            let d = point_to_triangle_distance(p, a, b, c);
            assert!(d.is_finite(), "degenerate distance must be finite, got {d}");
            assert!((d - want).abs() < 1e-12, "p={p:?}: d={d} want={want}");
        }
    }

    #[test]
    fn distance_zero_on_face_and_vertices() {
        let (a, b, c) = tri();
        for q in [a, b, c, (a + b + c) * (1.0 / 3.0), (a + b) * 0.5] {
            assert!(point_to_triangle_distance(q, a, b, c) < 1e-12, "on-tri q={q:?}");
        }
    }

    #[test]
    fn near_field_trigger() {
        let t = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // Longest edge = hypotenuse √2 ≈ 1.414; NEAR_FACTOR·h ≈ 5.66.
        assert!(is_near(Vec3::new(0.3, 0.3, 0.1), &t), "on/over the panel is near");
        assert!(is_near(Vec3::new(0.3, 0.3, 5.0), &t), "5 above is still within 4·h");
        assert!(!is_near(Vec3::new(0.3, 0.3, 20.0), &t), "20 above is far");
    }

    // ---- subdivision sanity --------------------------------------------------------

    fn equilateral() -> Tri {
        Tri::new(
            Vec3::new(-1.0, -1.0 / 3.0_f64.sqrt(), 0.0),
            Vec3::new(1.0, -1.0 / 3.0_f64.sqrt(), 0.0),
            Vec3::new(0.0, 2.0 / 3.0_f64.sqrt(), 0.0),
        )
    }

    #[test]
    fn midpoint_split_conserves_area_and_normal() {
        let t = equilateral();
        let kids = midpoint_split(&t);
        let suma: f64 = kids.iter().map(|k| k.area).sum();
        assert!((suma - t.area).abs() < 1e-13, "area {suma} vs {}", t.area);
        for k in &kids {
            assert!((k.normal - t.normal).norm() < 1e-13, "child normal preserved");
            assert!((k.area - t.area / 4.0).abs() < 1e-13, "congruent quarter-area");
        }
    }

    #[test]
    fn centroid_fan_conserves_area() {
        let t = equilateral();
        let p = (t.v1 + t.v2 + t.v3) * (1.0 / 3.0);
        let fan = centroid_fan(&t, p);
        let suma: f64 = fan.iter().map(|k| k.area).sum();
        assert!((suma - t.area).abs() < 1e-13, "fan area {suma} vs {}", t.area);
        for k in &fan {
            assert!((k.normal - t.normal).norm() < 1e-13);
        }
    }

    #[test]
    fn interior_projection_classifies_correctly() {
        let t = equilateral();
        // Point above the centroid → interior projection.
        let c = (t.v1 + t.v2 + t.v3) * (1.0 / 3.0);
        assert!(interior_projection(c + t.normal * 0.3, &t).is_some());
        // Point off to the side, projecting outside the face → None.
        assert!(interior_projection(Vec3::new(5.0, 0.0, 0.3), &t).is_none());
    }

    // ---- accuracy gates vs high-precision reference --------------------------------

    /// Physical regular Yukawa kernel (the ×4π convention), integrated to get the
    /// collocation. Single: (e^{−yr}−1)/r. Double: (1−(1+yr)e^{−yr})(x−ξ)·n / r³.
    fn regular_phys(kind: PotentialKind, x: Vec3, xi: Vec3, y: f64, n: Vec3) -> f64 {
        let d = (x - xi).norm();
        let c = y * d;
        match kind {
            PotentialKind::Single => ((-c).exp() - 1.0) / d,
            PotentialKind::Double => (1.0 - (1.0 + c) * (-c).exp()) * (x - xi).dot(n) / (d * d * d),
        }
    }

    /// Uniform barycentric midpoint quadrature over a triangle (same helper the yukawa
    /// tests use). Smooth integrand ⇒ O(1/M²)-ish convergence.
    fn integrate_tri(a: Vec3, b: Vec3, c: Vec3, m: usize, f: &dyn Fn(Vec3) -> f64) -> f64 {
        let area = (b - a).cross(c - a).norm() / 2.0;
        let w = area / (m * m) as f64;
        let inv = 1.0 / m as f64;
        let p = |i: usize, j: usize| a + (b - a) * (i as f64 * inv) + (c - a) * (j as f64 * inv);
        let mut acc = 0.0;
        for i in 0..m {
            for j in 0..(m - i) {
                acc += f((p(i, j) + p(i + 1, j) + p(i, j + 1)) * (1.0 / 3.0));
                if i + j + 2 <= m {
                    acc += f((p(i + 1, j) + p(i + 1, j + 1) + p(i, j + 1)) * (1.0 / 3.0));
                }
            }
        }
        acc * w
    }

    /// Convergence-studied high-precision reference (review [R1/R7]): refine M until the
    /// value stabilises, returning the finest. Trustworthy only for an **off-panel** ξ,
    /// where the integrand is smooth on the panel (the cusp at x=ξ is off the domain).
    fn reference_offpanel(kind: PotentialKind, xi: Vec3, tri: &Tri, y: f64) -> f64 {
        let f = |x: Vec3| regular_phys(kind, x, xi, y, tri.normal);
        let mut prev = integrate_tri(tri.v1, tri.v2, tri.v3, 96, &f);
        for &m in &[192usize, 384, 768, 1536] {
            let cur = integrate_tri(tri.v1, tri.v2, tri.v3, m, &f);
            if (cur - prev).abs() <= 1e-8 * cur.abs().max(1e-14) {
                return cur;
            }
            prev = cur;
        }
        prev
    }

    #[test]
    fn far_field_is_bit_identical_to_fixed() {
        // A far observation point must take the single fixed Radon eval verbatim — the
        // adaptive path may not perturb the bulk of the matrix.
        let t = equilateral();
        let xi = Vec3::new(0.3, -0.1, 30.0); // far above
        let y = 0.8;
        assert!(!is_near(xi, &t));
        for kind in [PotentialKind::Single, PotentialKind::Double] {
            let (adapt, st) = adaptive_regular_yukawa_collocation(kind, xi, &t, y, &Default::default());
            let fixed = regular_yukawa_collocation_parts(kind, xi, &t, y).0;
            assert_eq!(adapt, fixed, "far field must be bit-identical ({kind:?})");
            assert_eq!(st, Status::Converged);
        }
    }

    #[test]
    fn near_offpanel_beats_fixed_vs_reference() {
        // The core near-singular gate: at a close off-panel observation point, the fixed
        // 7-point rule misses the peaked integrand while adaptive matches a high-M
        // reference tightly. Tests several gaps d/h and both layers.
        let t = equilateral();
        let h = longest_edge(&t);
        let y = 1.2;
        let c = (t.v1 + t.v2 + t.v3) * (1.0 / 3.0);
        for &frac in &[1.0_f64, 0.5, 0.25] {
            let xi = c + t.normal * (frac * h); // off-panel, distance frac·h
            for kind in [PotentialKind::Single, PotentialKind::Double] {
                let reference = reference_offpanel(kind, xi, &t, y);
                if reference.abs() < 1e-9 {
                    continue; // skip a (double-layer) near-zero reference
                }
                let (adapt, st) =
                    adaptive_regular_yukawa_collocation(kind, xi, &t, y, &Default::default());
                let fixed = regular_yukawa_collocation_parts(kind, xi, &t, y).0;
                let e_adapt = (adapt - reference).abs() / reference.abs();
                let e_fixed = (fixed - reference).abs() / reference.abs();
                assert_eq!(st, Status::Converged, "{kind:?} d/h={frac}: not converged");
                assert!(
                    e_adapt < 1e-4,
                    "{kind:?} d/h={frac}: adaptive rel err {e_adapt:.2e} (ref {reference:.6})"
                );
                // Adaptive must be a clear improvement over the fixed rule near-field.
                assert!(
                    e_adapt < e_fixed,
                    "{kind:?} d/h={frac}: adaptive {e_adapt:.2e} not better than fixed {e_fixed:.2e}"
                );
            }
        }
    }

    #[test]
    fn cleft_opposing_panels_beats_fixed() {
        // The real SES failure mode (review [R7]): two panels facing each other across a
        // narrow gap — the cross collocation `Vy[A←B]` / `Ky[A←B]` (panel B seen from A's
        // centroid) is exactly the near-singular off-panel integral the sphere lacks.
        // Adaptive must resolve it where the fixed 7-point rule misses, for BOTH layers
        // (the double layer is the signed, cancellation-prone one).
        let h = 2.0_f64;
        let y = 1.1;
        // Panel A in the z=0 plane (normal +z), panel B parallel at z=gap (normal −z,
        // facing A). gap = 0.3·h — a tight cleft.
        let gap = 0.3 * h;
        let a = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(0.0, h, 0.0),
        );
        // B facing A: reverse winding so its normal points −z (toward A).
        let b = Tri::new(
            Vec3::new(0.0, 0.0, gap),
            Vec3::new(0.0, h, gap),
            Vec3::new(h, 0.0, gap),
        );
        assert!(b.normal.z < 0.0, "panel B must face panel A");
        let xi_a = (a.v1 + a.v2 + a.v3) * (1.0 / 3.0); // observe B from A's centroid

        for kind in [PotentialKind::Single, PotentialKind::Double] {
            let reference = reference_offpanel(kind, xi_a, &b, y); // cusp off B ⇒ trustworthy
            assert!(reference.abs() > 1e-9, "{kind:?} cleft reference too small");
            let (adapt, st) =
                adaptive_regular_yukawa_collocation(kind, xi_a, &b, y, &Default::default());
            let fixed = regular_yukawa_collocation_parts(kind, xi_a, &b, y).0;
            let e_adapt = (adapt - reference).abs() / reference.abs();
            let e_fixed = (fixed - reference).abs() / reference.abs();
            eprintln!(
                "cleft {kind:?}: adaptive rel {e_adapt:.2e}, fixed rel {e_fixed:.2e} (ref {reference:.6})"
            );
            assert_eq!(st, Status::Converged, "{kind:?} cleft not converged");
            assert!(e_adapt < 1e-4, "{kind:?} cleft adaptive rel err {e_adapt:.2e}");
            assert!(
                e_adapt < 0.25 * e_fixed,
                "{kind:?} cleft: adaptive {e_adapt:.2e} not ≪ fixed {e_fixed:.2e}"
            );
        }
    }
}
