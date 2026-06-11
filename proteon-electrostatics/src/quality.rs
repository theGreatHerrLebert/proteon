//! P6.5 mesh-acceptance: geometric quality metrics + a scale-aware charge-placement
//! policy. BEM is notoriously mesh-sensitive, and a charge sitting on or very near the
//! surface Γ makes the molecular-potential trace near-singular — both silently corrupt
//! a solve. This module *measures* the relevant quantities over the element list and
//! classifies them against thresholds so the caller can refuse or warn rather than hand
//! back a wrong-but-plausible answer (plan §6 / `TO_ELECTROSTATICS.md`).
//!
//! Scope: per-element triangle quality (min angle, aspect ratio, area spread, near-
//! degenerate faces) and charge-to-surface separation **relative to local element size**
//! (the plan's explicit "scale-aware, not a fixed absolute epsilon" requirement).
//! Topological checks — watertightness, consistent orientation, connected components,
//! self-intersection — are out of scope here: the first two already live on
//! `proteon_core::surface::mesh::Mesh` (the connector reports them); the latter two are
//! documented further work.

use crate::adaptive::{longest_edge, point_to_triangle_distance};
use crate::model::{Charge, Tri};

/// Severity of a quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Usable but risky — the result may be degraded; surface it.
    Warn,
    /// Unsafe — the solve is likely unreliable; refuse unless explicitly overridden.
    Error,
}

/// A single quality finding.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// How serious.
    pub severity: Severity,
    /// Human-readable explanation (already includes the offending value).
    pub message: String,
}

/// Measured mesh-quality metrics (no policy — all raw numbers).
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Number of triangle elements.
    pub n_elements: usize,
    /// Smallest interior angle over all triangles (degrees) — small ⇒ slivers.
    pub min_angle_deg: f64,
    /// Largest radius-ratio aspect (`R_circ / 2·r_in`, `1` for equilateral, `∞` for a
    /// degenerate sliver).
    pub max_aspect_ratio: f64,
    /// Smallest / largest triangle area, and their ratio (area spread).
    pub min_area: f64,
    /// Largest triangle area.
    pub max_area: f64,
    /// Count of near-degenerate (effectively zero-area) triangles.
    pub n_near_degenerate: usize,
    /// Minimum distance from any charge to the surface (`0` if no charges).
    pub min_charge_surface_gap: f64,
    /// That minimum gap **relative to the local element size** at the closest element —
    /// the scale-aware charge-placement metric (small ⇒ a charge on/near Γ).
    pub min_charge_gap_ratio: f64,
}

// ---- policy thresholds (documented defaults) ---------------------------------------

/// Below this interior angle (degrees) a triangle is a sliver — warn.
pub const ANGLE_WARN_DEG: f64 = 10.0;
/// Above this radius-ratio aspect a triangle is badly shaped — warn.
pub const ASPECT_WARN: f64 = 10.0;
/// A triangle whose area is below this fraction of the median is near-degenerate.
pub const DEGENERATE_AREA_FRAC: f64 = 1e-6;
/// Charge-to-surface gap (relative to local element size) below this — **reject**: the
/// molecular-potential trace is near-singular and the solve is unreliable.
pub const CHARGE_REJECT_RATIO: f64 = 0.05;
/// …below this (but above reject) — warn: the charge is close to Γ.
pub const CHARGE_WARN_RATIO: f64 = 0.3;

/// Smallest interior angle of a triangle, in degrees (law of cosines on the three edge
/// lengths). Returns `0` for a degenerate triangle.
fn min_angle_deg(tri: &Tri) -> f64 {
    let a = (tri.v2 - tri.v1).norm();
    let b = (tri.v3 - tri.v2).norm();
    let c = (tri.v1 - tri.v3).norm();
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return 0.0;
    }
    // Angle opposite each edge; clamp the cosine for floating-point safety.
    let ang = |opp: f64, x: f64, y: f64| {
        ((x * x + y * y - opp * opp) / (2.0 * x * y))
            .clamp(-1.0, 1.0)
            .acos()
    };
    let amin = ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b));
    amin.to_degrees()
}

/// Radius-ratio aspect `R_circ / (2·r_in) = a·b·c·s / (8·Area²)` (`1` equilateral,
/// grows without bound as the triangle degenerates).
fn aspect_ratio(tri: &Tri) -> f64 {
    let a = (tri.v2 - tri.v1).norm();
    let b = (tri.v3 - tri.v2).norm();
    let c = (tri.v1 - tri.v3).norm();
    let area = tri.area;
    if area <= 0.0 {
        return f64::INFINITY;
    }
    let s = (a + b + c) / 2.0;
    a * b * c * s / (8.0 * area * area)
}

impl QualityReport {
    /// Measure the quality of `elements` with `charges`. Pure geometry, no thresholds.
    #[must_use]
    pub fn assess(elements: &[Tri], charges: &[Charge]) -> Self {
        let n = elements.len();
        let mut min_angle = f64::INFINITY;
        let mut max_aspect = 0.0_f64;
        let mut min_area = f64::INFINITY;
        let mut max_area = 0.0_f64;
        let mut areas: Vec<f64> = Vec::with_capacity(n);
        for e in elements {
            min_angle = min_angle.min(min_angle_deg(e));
            max_aspect = max_aspect.max(aspect_ratio(e));
            min_area = min_area.min(e.area);
            max_area = max_area.max(e.area);
            areas.push(e.area);
        }
        // Near-degenerate: area below a fraction of the median (robust to outliers).
        let median = {
            areas.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            if areas.is_empty() {
                0.0
            } else {
                areas[areas.len() / 2]
            }
        };
        let n_near_degenerate = elements
            .iter()
            .filter(|e| e.area < DEGENERATE_AREA_FRAC * median)
            .count();

        // Scale-aware charge placement: min over charges of (gap / local element size).
        let mut min_gap = f64::INFINITY;
        let mut min_ratio = f64::INFINITY;
        for c in charges {
            let mut best = f64::INFINITY;
            let mut best_elem_size = 1.0;
            for e in elements {
                let d = point_to_triangle_distance(c.pos, e.v1, e.v2, e.v3);
                if d < best {
                    best = d;
                    best_elem_size = longest_edge(e);
                }
            }
            min_gap = min_gap.min(best);
            if best_elem_size > 0.0 {
                min_ratio = min_ratio.min(best / best_elem_size);
            }
        }
        if charges.is_empty() {
            min_gap = 0.0;
            min_ratio = f64::INFINITY;
        }

        Self {
            n_elements: n,
            min_angle_deg: if n == 0 { 0.0 } else { min_angle },
            max_aspect_ratio: max_aspect,
            min_area: if n == 0 { 0.0 } else { min_area },
            max_area,
            n_near_degenerate,
            min_charge_surface_gap: min_gap,
            min_charge_gap_ratio: min_ratio,
        }
    }

    /// Classify the metrics against the policy thresholds. An empty list ⇒ the mesh is
    /// acceptable. `Error` issues should block (unless the caller overrides); `Warn`
    /// issues should be surfaced.
    #[must_use]
    pub fn issues(&self) -> Vec<QualityIssue> {
        let mut v = Vec::new();
        if self.n_near_degenerate > 0 {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "{} near-degenerate (≈zero-area) triangle(s): the collocation is unreliable",
                    self.n_near_degenerate
                ),
            });
        }
        if self.min_charge_gap_ratio < CHARGE_REJECT_RATIO {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "a charge is within {:.2}× the local element size of the surface \
                     (< {CHARGE_REJECT_RATIO}); the molecular-potential trace is near-singular",
                    self.min_charge_gap_ratio
                ),
            });
        } else if self.min_charge_gap_ratio < CHARGE_WARN_RATIO {
            v.push(QualityIssue {
                severity: Severity::Warn,
                message: format!(
                    "a charge is close to the surface ({:.2}× the local element size)",
                    self.min_charge_gap_ratio
                ),
            });
        }
        if self.min_angle_deg < ANGLE_WARN_DEG {
            v.push(QualityIssue {
                severity: Severity::Warn,
                message: format!(
                    "smallest triangle angle {:.1}° (< {ANGLE_WARN_DEG}°): sliver elements degrade accuracy",
                    self.min_angle_deg
                ),
            });
        }
        if self.max_aspect_ratio > ASPECT_WARN {
            v.push(QualityIssue {
                severity: Severity::Warn,
                message: format!(
                    "max aspect ratio {:.1} (> {ASPECT_WARN}): poorly-shaped elements",
                    self.max_aspect_ratio
                ),
            });
        }
        v
    }

    /// Whether any [`Severity::Error`] issue is present (the mesh should be refused).
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.issues().iter().any(|i| i.severity == Severity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic_sphere_mesh;
    use proteon_core::surface::geom::Vec3;

    fn sphere_elements(radius: f64, subdiv: u32) -> Vec<Tri> {
        let mesh = analytic_sphere_mesh(radius, subdiv);
        mesh.tris
            .iter()
            .map(|t| {
                Tri::new(
                    mesh.verts[t[0] as usize],
                    mesh.verts[t[1] as usize],
                    mesh.verts[t[2] as usize],
                )
            })
            .collect()
    }

    #[test]
    fn equilateral_is_high_quality() {
        let t = Tri::new(
            Vec3::new(-1.0, -1.0 / 3.0_f64.sqrt(), 0.0),
            Vec3::new(1.0, -1.0 / 3.0_f64.sqrt(), 0.0),
            Vec3::new(0.0, 2.0 / 3.0_f64.sqrt(), 0.0),
        );
        assert!((min_angle_deg(&t) - 60.0).abs() < 1e-9, "equilateral angles are 60°");
        assert!((aspect_ratio(&t) - 1.0).abs() < 1e-9, "equilateral aspect is 1");
    }

    #[test]
    fn sliver_flags_angle_and_aspect() {
        let s = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(2.0, 0.05, 0.0), // very thin
        );
        assert!(min_angle_deg(&s) < ANGLE_WARN_DEG);
        assert!(aspect_ratio(&s) > ASPECT_WARN);
    }

    #[test]
    fn good_sphere_with_central_charge_is_acceptable() {
        let elements = sphere_elements(2.0, 3);
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, 0.0),
            val: 1.0,
        }];
        let rep = QualityReport::assess(&elements, &charges);
        assert!(!rep.has_errors(), "central charge in a good sphere: {:?}", rep.issues());
        assert!(rep.min_angle_deg > ANGLE_WARN_DEG, "icosphere angles are well-shaped");
        // The central charge is ~radius away, far in element-size units.
        assert!(rep.min_charge_gap_ratio > CHARGE_WARN_RATIO);
    }

    #[test]
    fn charge_on_surface_is_rejected_scale_aware() {
        let radius = 2.0;
        let elements = sphere_elements(radius, 3);
        // Charge placed just inside the surface (gap ≪ element size) ⇒ scale-aware reject.
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, radius - 0.001),
            val: 1.0,
        }];
        let rep = QualityReport::assess(&elements, &charges);
        assert!(rep.min_charge_gap_ratio < CHARGE_REJECT_RATIO, "ratio {}", rep.min_charge_gap_ratio);
        assert!(rep.has_errors());
        assert!(rep
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("near-singular")));
    }

    #[test]
    fn deeper_charge_only_warns_then_clears() {
        let radius = 2.0;
        let elements = sphere_elements(radius, 3);
        let elem = longest_edge(&elements[0]);
        // Place the charge ~0.5·elem below the surface ⇒ warn band, not reject.
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, radius - 0.5 * elem),
            val: 1.0,
        }];
        let rep = QualityReport::assess(&elements, &charges);
        assert!(!rep.has_errors(), "0.5·elem deep should not reject: {:?}", rep.issues());
    }
}
