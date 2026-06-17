//! P6.5 mesh-acceptance: geometric quality metrics + a scale-aware charge-placement
//! policy. BEM is notoriously mesh-sensitive, and a charge sitting on or very near the
//! surface Γ makes the molecular-potential trace near-singular — both silently corrupt
//! a solve. This module *measures* the relevant quantities over the element list and
//! classifies them against thresholds so the caller can refuse or warn rather than hand
//! back a wrong-but-plausible answer (plan §6 / `devdocs/TO_ELECTROSTATICS.md`).
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
use proteon_core::surface::geom::Vec3;
use proteon_core::surface::mesh::Mesh;

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
    /// Minimum over all charges and elements of `dist(charge, element) / longest_edge` —
    /// the scale-aware charge-placement metric (small ⇒ a charge on/near Γ in element-
    /// size units). Taken as `min_t(d/h_t)` over *all* elements (review [High#2]), not the
    /// ratio at the single nearest element, so a grading boundary cannot flip it.
    pub min_charge_gap_ratio: f64,
    /// Number of charges that fall **outside** the (closed) surface by the generalized
    /// winding number — a placement error the unsigned gap cannot see (review [High#1]).
    pub n_charges_outside: usize,
}

// ---- policy thresholds (documented defaults) ---------------------------------------

/// Below this interior angle (degrees) a triangle is a sliver — warn.
pub const ANGLE_WARN_DEG: f64 = 10.0;
/// Above this radius-ratio aspect a triangle is badly shaped — warn.
pub const ASPECT_WARN: f64 = 10.0;
/// An **intrinsic** near-degeneracy threshold on the dimensionless fatness
/// `2·Area / longest_edge²` (`≈0.87` equilateral, `→0` as the triangle collapses).
/// Below this the normal is unreliable — judged per triangle, so it is robust even when
/// most of the mesh is degenerate (review [High#3]).
pub const DEGENERATE_FATNESS: f64 = 1e-4;
/// Charge-to-surface gap (relative to local element size) below this — **reject**: the
/// charge is essentially on/through Γ, so the molecular-potential trace is near-singular
/// regardless of mesh resolution. Deliberately tighter than the warn band: refusal is
/// reserved for the mesh-independent pathological case (review [Med#5]).
pub const CHARGE_REJECT_RATIO: f64 = 0.01;
/// …below this (but above reject) — warn: the charge is close to Γ. This band IS mesh-
/// dependent (a coarse mesh shrinks the ratio for the same physical gap), so it warns
/// rather than refuses. All charge thresholds are uncalibrated starting points pending an
/// SES-corpus study (review [Med#5]).
pub const CHARGE_WARN_RATIO: f64 = 0.3;

/// Dimensionless triangle "fatness" `2·Area / longest_edge²` (`2/√3·… `≈0.87 for an
/// equilateral, `→0` as it degenerates). `0` if non-finite / zero longest edge.
fn fatness(tri: &Tri) -> f64 {
    let m = longest_edge(tri);
    if !m.is_finite() || m <= 0.0 || !tri.area.is_finite() {
        return 0.0;
    }
    2.0 * tri.area / (m * m)
}

/// Smallest interior angle of a triangle, in degrees, via the numerically stable
/// `atan2(|u×v|, u·v)` (no cosine cancellation on slivers). Returns `0` if non-finite.
fn min_angle_deg(tri: &Tri) -> f64 {
    let vs = [tri.v1, tri.v2, tri.v3];
    let mut amin = std::f64::consts::PI;
    for i in 0..3 {
        let u = vs[(i + 1) % 3] - vs[i];
        let v = vs[(i + 2) % 3] - vs[i];
        let ang = u.cross(v).norm().atan2(u.dot(v));
        amin = amin.min(ang);
    }
    let deg = amin.to_degrees();
    if deg.is_finite() {
        deg
    } else {
        0.0
    }
}

/// Radius-ratio aspect `R_circ / (2·r_in)` (`1` equilateral, `→∞` degenerate), computed
/// on edges **rescaled by the longest** so the four-length product cannot over/underflow
/// even for a tiny-but-nonzero area (review [Med#4]). `∞` if non-finite.
fn aspect_ratio(tri: &Tri) -> f64 {
    let m = longest_edge(tri);
    if !m.is_finite() || m <= 0.0 || !tri.area.is_finite() {
        return f64::INFINITY;
    }
    let (a, b, c) = (
        (tri.v2 - tri.v1).norm() / m,
        (tri.v3 - tri.v2).norm() / m,
        (tri.v1 - tri.v3).norm() / m,
    );
    let area = tri.area / (m * m); // rescaled area
    if area <= 0.0 {
        return f64::INFINITY;
    }
    let s = (a + b + c) / 2.0;
    let r = a * b * c * s / (8.0 * area * area);
    if r.is_finite() {
        r
    } else {
        f64::INFINITY
    }
}

/// Generalized winding number of point `p` w.r.t. the (closed) surface: `1/4π · Σ_t
/// Ω(p, t)`, with the Van Oosterom–Strackee signed solid angle per triangle. `≈±1` for an
/// interior point (sign = orientation), `≈0` for an exterior one — so `|w| < 0.5`
/// classifies `p` as **outside**, robustly and orientation-agnostically. Topology-free
/// (works on the raw element geometry), so usable without a watertight `Mesh`.
fn winding_number(p: Vec3, elements: &[Tri]) -> f64 {
    let mut omega = 0.0;
    for t in elements {
        let a = t.v1 - p;
        let b = t.v2 - p;
        let c = t.v3 - p;
        let (la, lb, lc) = (a.norm(), b.norm(), c.norm());
        let num = a.dot(b.cross(c)); // scalar triple product
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        omega += 2.0 * num.atan2(den);
    }
    omega / (4.0 * std::f64::consts::PI)
}

impl QualityReport {
    /// Measure the quality of `elements` with `charges`. Pure geometry, no thresholds.
    ///
    /// Cost is `O(charges × elements)` (nearest-element gap + winding number). That is at
    /// most one BEM matvec's worth of work and the dense solve is `O(N²·iters)`, so it is
    /// not the bottleneck; a spatial index would be the move only if this ever runs
    /// without a following solve (review [#6]).
    #[must_use]
    pub fn assess(elements: &[Tri], charges: &[Charge]) -> Self {
        let n = elements.len();
        let mut min_angle = f64::INFINITY;
        let mut max_aspect = 0.0_f64;
        let mut min_area = f64::INFINITY;
        let mut max_area = 0.0_f64;
        let mut n_near_degenerate = 0usize;
        for e in elements {
            min_angle = min_angle.min(min_angle_deg(e));
            max_aspect = max_aspect.max(aspect_ratio(e));
            min_area = min_area.min(e.area);
            max_area = max_area.max(e.area);
            // Intrinsic per-triangle degeneracy (robust to widespread degeneracy).
            if fatness(e) < DEGENERATE_FATNESS {
                n_near_degenerate += 1;
            }
        }

        // Scale-aware charge placement + containment.
        let mut min_gap = f64::INFINITY;
        let mut min_ratio = f64::INFINITY;
        let mut n_outside = 0usize;
        for c in charges {
            let mut gap = f64::INFINITY;
            for e in elements {
                let d = point_to_triangle_distance(c.pos, e.v1, e.v2, e.v3);
                gap = gap.min(d);
                let h = longest_edge(e);
                if h > 0.0 {
                    // min over ALL elements of d/h — a grading boundary cannot flip it.
                    min_ratio = min_ratio.min(d / h);
                }
            }
            min_gap = min_gap.min(gap);
            if !elements.is_empty() && winding_number(c.pos, elements).abs() < 0.5 {
                n_outside += 1;
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
            n_charges_outside: n_outside,
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
                    "{} near-degenerate (sliver / unreliable-normal) triangle(s): the \
                     collocation is unreliable",
                    self.n_near_degenerate
                ),
            });
        }
        if self.n_charges_outside > 0 {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "{} charge(s) lie outside the closed surface (winding number ≈ 0): the \
                     interior-source model is violated — check charge placement / mesh closure",
                    self.n_charges_outside
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

/// Topological acceptance of the index mesh — the conditions the *interior-source* BEM
/// model and the double-layer sign depend on, beyond per-element geometry.
#[derive(Debug, Clone)]
pub struct TopologyReport {
    /// Every edge shared by exactly two triangles (closed manifold).
    pub watertight: bool,
    /// Every directed edge used once with its reverse once (no flipped windings).
    pub consistently_oriented: bool,
    /// Edges used by other than two triangles (boundary / non-manifold).
    pub num_nonmanifold_edges: usize,
    /// Aggregate signed enclosed volume (divergence theorem) — only meaningful when
    /// closed, and can MASK an inward component, so orientation is judged per component.
    pub signed_volume: f64,
    /// Closed + consistently oriented with **every** component correctly oriented
    /// *outward-from-solute* (signed-volume sign `(−1)^nesting_depth` — body +, cavity −,
    /// island +). The condition the double-layer sign needs; nesting-aware so a cavity's
    /// legitimately-negative volume is not flagged (multi-region work).
    pub is_outward: bool,
    /// Connected surface components (a multi-body solute is > 1).
    pub num_components: usize,
    /// Components whose orientation is *wrong* for their nesting parity (sign ≠
    /// `(−1)^depth`), when closed + consistently oriented. The connector auto-fixes these
    /// via `orient_by_nesting`.
    pub num_misoriented_components: usize,
    /// Any component whose enclosed volume is non-finite or, relative to its area,
    /// effectively zero — an indeterminate orientation (review [Med#2]).
    pub has_degenerate_volume: bool,
    /// Coincident (duplicate) faces.
    pub num_duplicate_faces: usize,
    /// Self-intersecting (transversely penetrating) triangle pairs, or `None` if the
    /// check was inconclusive (mesh too irregular to verify cheaply). `Some(k>0)` is
    /// authoritative; `Some(0)` means "no clear penetration found", not a proof.
    pub num_self_intersections: Option<usize>,
    /// Buried solvent cavities — components at **odd** nesting depth (depth 1 = a cavity
    /// in a body, depth 3 = a cavity in an island, …). Supported via orientation (the
    /// scalar-`f` solve is correct once oriented by nesting; cavity science gates in
    /// `tests/cavity_*.rs`) but less-exercised than the single-body case, so a Warn, not
    /// an Error. Computed only for a closed, consistently-oriented mesh.
    pub num_cavities: usize,
    /// Two different components touch (share a vertex) — the region topology is then
    /// ill-defined, so the nesting/winding classification cannot be trusted (review).
    pub components_touch: bool,
}

impl TopologyReport {
    /// Assess the topology of `mesh` (index-based: shared-vertex/edge structure).
    #[must_use]
    pub fn assess(mesh: &Mesh) -> Self {
        let watertight = mesh.is_watertight();
        let consistently_oriented = mesh.is_consistently_oriented();
        let signed_volume = mesh.signed_volume();
        let va = mesh.component_volumes_areas();
        // Nesting depths are only meaningful (and only worth the O(k·N) cost) for a
        // closed, consistently-oriented mesh; a non-closed mesh refuses regardless.
        let depths = if watertight && consistently_oriented {
            mesh.component_nesting_depths()
        } else {
            Vec::new()
        };
        let mut num_misoriented = 0usize;
        let mut degenerate = false;
        for (c, &(vol, area)) in va.iter().enumerate() {
            // A closed component should enclose ~area^1.5 of volume; far below that (or
            // non-finite) is indeterminate.
            let floor = 1e-9 * area.max(0.0).powf(1.5);
            if !vol.is_finite() || vol.abs() <= floor {
                degenerate = true;
            } else {
                // Correct sign by nesting parity (depth unknown ⇒ assume top-level body).
                let want = if depths.get(c).copied().unwrap_or(0) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                if vol.signum() != want {
                    num_misoriented += 1;
                }
            }
        }
        Self {
            watertight,
            consistently_oriented,
            num_nonmanifold_edges: mesh.num_nonmanifold_edges(),
            signed_volume,
            is_outward: watertight && consistently_oriented && num_misoriented == 0 && !degenerate,
            num_components: mesh.num_connected_components(),
            num_misoriented_components: num_misoriented,
            has_degenerate_volume: degenerate,
            num_duplicate_faces: mesh.num_duplicate_faces(),
            num_self_intersections: mesh.count_self_intersections(),
            // Actual solvent cavities are the ODD-depth components (a body/cavity/island
            // nest has 2 nested components but 1 solvent cavity).
            num_cavities: depths.iter().filter(|&&d| d % 2 == 1).count(),
            components_touch: depths.len() > 1 && mesh.has_touching_components(),
        }
    }

    /// Classify the topology against acceptance policy. `Error` ⇒ the interior model /
    /// double-layer sign is unsound; `Warn` ⇒ usable but worth surfacing.
    #[must_use]
    pub fn issues(&self) -> Vec<QualityIssue> {
        let mut v = Vec::new();
        if !self.watertight {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "{} non-manifold/boundary edge(s): the surface is not closed, so \
                     interior/exterior (and the molecular potential) are undefined",
                    self.num_nonmanifold_edges
                ),
            });
        }
        if !self.consistently_oriented {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: "inconsistent triangle winding: the double-layer sign is unreliable \
                          (orient the mesh consistently)"
                    .to_string(),
            });
        }
        if self.num_duplicate_faces > 0 {
            v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "{} duplicate (coincident) face(s): the surface is double-counted",
                    self.num_duplicate_faces
                ),
            });
        }
        match self.num_self_intersections {
            Some(k) if k > 0 => v.push(QualityIssue {
                severity: Severity::Error,
                message: format!(
                    "{k} self-intersecting triangle pair(s): the surface penetrates itself, so \
                     interior/exterior (and the molecular potential) are undefined"
                ),
            }),
            None => v.push(QualityIssue {
                // Fail CLOSED (review): a validity gate must refuse an unverifiable mesh,
                // not proceed. `allow_low_quality=True` is the explicit override.
                severity: Severity::Error,
                message: "self-intersection check was inconclusive (mesh too irregular / \
                          multi-scale to verify cheaply): a clean closed surface cannot be \
                          assured — pass a cleaner mesh or override"
                    .to_string(),
            }),
            Some(_) => {} // no clear penetration found (not a proof — see the detector docs)
        }
        // Per-component orientation only matters once closed + consistent (else the
        // above fire, and component volumes are not trustworthy anyway).
        if self.watertight && self.consistently_oriented {
            if self.components_touch {
                v.push(QualityIssue {
                    severity: Severity::Error,
                    message: "two surface components touch (share a vertex): the region \
                              topology (and so the nesting/cavity classification) is \
                              ill-defined — separate the components"
                        .to_string(),
                });
            }
            if self.has_degenerate_volume {
                v.push(QualityIssue {
                    severity: Severity::Error,
                    message: "a component has indeterminate enclosed volume (≈0 or \
                              non-finite): orientation cannot be established"
                        .to_string(),
                });
            }
            if self.num_misoriented_components > 0 {
                v.push(QualityIssue {
                    severity: Severity::Error,
                    message: format!(
                        "{} component(s) oriented wrong for their nesting parity (sign ≠ \
                         (−1)^depth): orient outward-from-solute (body +, cavity −, …), or \
                         the double-layer sign is reversed",
                        self.num_misoriented_components
                    ),
                });
            }
            if self.num_cavities > 0 {
                v.push(QualityIssue {
                    severity: Severity::Warn,
                    message: format!(
                        "{} buried cavity / nested component(s): handled via nesting \
                         orientation (cavity science gate passes), but multi-region is \
                         less-exercised than the single-body case — sanity-check the result",
                        self.num_cavities
                    ),
                });
            }
        }
        if self.num_components > 1 {
            v.push(QualityIssue {
                severity: Severity::Warn,
                message: format!(
                    "{} disconnected surface components: ensure each charge is assigned to \
                     (inside) the correct body — a mis-assigned charge is a silent error",
                    self.num_components
                ),
            });
        }
        v
    }

    /// Whether any [`Severity::Error`] issue is present.
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
        assert!(
            (min_angle_deg(&t) - 60.0).abs() < 1e-9,
            "equilateral angles are 60°"
        );
        assert!(
            (aspect_ratio(&t) - 1.0).abs() < 1e-9,
            "equilateral aspect is 1"
        );
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
        assert!(
            !rep.has_errors(),
            "central charge in a good sphere: {:?}",
            rep.issues()
        );
        assert!(
            rep.min_angle_deg > ANGLE_WARN_DEG,
            "icosphere angles are well-shaped"
        );
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
        assert!(
            rep.min_charge_gap_ratio < CHARGE_REJECT_RATIO,
            "ratio {}",
            rep.min_charge_gap_ratio
        );
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
        // Place the charge ~elem below the surface ⇒ ratio ~1, no reject (and above warn).
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, radius - elem),
            val: 1.0,
        }];
        let rep = QualityReport::assess(&elements, &charges);
        assert!(
            !rep.has_errors(),
            "elem deep should not reject: {:?}",
            rep.issues()
        );
        assert_eq!(
            rep.n_charges_outside, 0,
            "an interior charge is not outside"
        );
    }

    #[test]
    fn outside_charge_is_rejected_by_containment() {
        // Review [High#1]: a charge OUTSIDE the closed surface has a comfortable unsigned
        // gap but violates the interior-source model. The winding number must catch it.
        let elements = sphere_elements(2.0, 3);
        let charges = [Charge {
            pos: Vec3::new(0.0, 0.0, 5.0), // well outside the radius-2 sphere
            val: 1.0,
        }];
        let rep = QualityReport::assess(&elements, &charges);
        assert_eq!(rep.n_charges_outside, 1, "exterior charge must be detected");
        assert!(rep.has_errors());
        assert!(rep
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("outside the closed")));
    }

    #[test]
    fn widespread_degeneracy_is_caught() {
        // Review [High#3]: when MOST faces are degenerate, a median-relative test collapses
        // and flags nothing. The intrinsic per-triangle fatness must flag them all.
        let sliver = || {
            Tri::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1e-5, 0.0), // fatness ≈ 1e-5 ≪ DEGENERATE_FATNESS
            )
        };
        let elements: Vec<Tri> = (0..6).map(|_| sliver()).collect();
        let rep = QualityReport::assess(&elements, &[]);
        assert_eq!(rep.n_near_degenerate, 6, "every sliver must be flagged");
        assert!(rep.has_errors());
    }

    #[test]
    fn topology_accepts_outward_sphere_flags_inward_and_open() {
        use proteon_core::surface::mesh::icosphere;
        let m = icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, 2);
        let rep = TopologyReport::assess(&m);
        assert!(
            rep.is_outward && !rep.has_errors(),
            "outward sphere accepted: {:?}",
            rep.issues()
        );
        assert_eq!(rep.num_components, 1);

        // Inward (inside-out) → flagged, with the "outward" message.
        let mut inward = m.clone();
        inward.flip();
        let rin = TopologyReport::assess(&inward);
        assert!(!rin.is_outward && rin.has_errors());
        assert!(rin.signed_volume < 0.0);
        assert!(rin
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("outward")));

        // Open (drop a face) → non-manifold edges → Error.
        let mut open = m.clone();
        open.tris.pop();
        let ropen = TopologyReport::assess(&open);
        assert!(!ropen.watertight && ropen.has_errors());
    }

    #[test]
    fn topology_warns_on_multiple_components() {
        use proteon_core::surface::mesh::icosphere;
        let mut two = icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 1);
        two.append(&icosphere(Vec3::new(5.0, 0.0, 0.0), 1.0, 1));
        let rep = TopologyReport::assess(&two);
        assert_eq!(rep.num_components, 2);
        // Two disjoint outward spheres: a component warning, but no Error.
        assert!(!rep.has_errors());
        assert_eq!(rep.num_cavities, 0);
        assert!(rep.issues().iter().any(|i| i.severity == Severity::Warn));
    }

    #[test]
    fn topology_accepts_oriented_cavity_with_warning() {
        // A small shell inside a big one (a solvent-filled cavity), oriented by nesting
        // (body +, cavity −), is now SUPPORTED — accepted with a Warn, not an Error
        // (multi-region; the cavity science gate passes).
        use proteon_core::surface::mesh::icosphere;
        let mut cavity = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, 2);
        cavity.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 2));
        cavity.orient_by_nesting(); // body +, cavity −
        let rep = TopologyReport::assess(&cavity);
        assert_eq!(rep.num_cavities, 1);
        assert!(rep.is_outward, "correctly oriented per nesting");
        assert!(
            !rep.has_errors(),
            "an oriented cavity is accepted: {:?}",
            rep.issues()
        );
        assert!(rep
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Warn && i.message.contains("cavity")));

        // Without orientation, the inner shell is misoriented for its parity → Error.
        let mut raw = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, 2);
        raw.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 2));
        assert!(
            TopologyReport::assess(&raw).has_errors(),
            "unoriented cavity is misoriented"
        );

        // num_cavities counts actual solvent cavities (odd depth): body/cavity/island has
        // 2 nested components but 1 solvent cavity.
        let mut three = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, 2);
        three.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 2.0, 2));
        three.append(&icosphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 2));
        three.orient_by_nesting();
        assert_eq!(
            TopologyReport::assess(&three).num_cavities,
            1,
            "1 solvent cavity (odd depth)"
        );
    }

    #[test]
    fn topology_flags_self_intersection() {
        use proteon_core::surface::mesh::Mesh;
        // Two non-adjacent crossing triangles.
        let m = Mesh {
            verts: vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.5),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.5, 0.0, 1.0),
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        let rep = TopologyReport::assess(&m);
        assert_eq!(rep.num_self_intersections, Some(1));
        assert!(rep.has_errors());
        assert!(rep
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("self-intersecting")));

        // Inconclusive (None) must fail CLOSED — an Error, not a Warn (review).
        let nan = Mesh {
            verts: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, f64::NAN, 0.0),
                Vec3::new(2.0, 2.0, 2.0),
                Vec3::new(3.0, 2.0, 2.0),
                Vec3::new(2.0, 3.0, 2.0),
            ],
            normals: Vec::new(),
            tris: vec![[0, 1, 2], [3, 4, 5]],
        };
        let rep2 = TopologyReport::assess(&nan);
        assert_eq!(rep2.num_self_intersections, None);
        assert!(rep2
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("inconclusive")));
    }

    #[test]
    fn topology_detects_per_component_inward_under_masking_volume() {
        // Review [High#1]: a big outward shell + a small INWARD shell. The aggregate
        // signed volume stays positive (masking the inward one), but per-component
        // orientation must still flag the inward component as an Error.
        use proteon_core::surface::mesh::icosphere;
        let big = icosphere(Vec3::new(0.0, 0.0, 0.0), 3.0, 1); // outward, large +vol
        let mut small = icosphere(Vec3::new(10.0, 0.0, 0.0), 0.5, 1);
        small.flip(); // inward, small −vol
        let mut mixed = big.clone();
        mixed.append(&small);

        let rep = TopologyReport::assess(&mixed);
        assert!(
            rep.signed_volume > 0.0,
            "aggregate volume masks the inward shell"
        );
        assert_eq!(
            rep.num_misoriented_components, 1,
            "per-component catches the inward shell"
        );
        assert!(!rep.is_outward && rep.has_errors());
        assert!(rep
            .issues()
            .iter()
            .any(|i| i.severity == Severity::Error && i.message.contains("nesting parity")));
    }

    #[test]
    fn metrics_stay_finite_on_a_tiny_triangle() {
        // Review [Med#4]: rescaling guards over/underflow — aspect/angle stay finite even
        // for a tiny (but valid) triangle.
        let t = Tri::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e-6, 0.0, 0.0),
            Vec3::new(0.0, 1e-6, 0.0),
        );
        assert!(min_angle_deg(&t).is_finite() && aspect_ratio(&t).is_finite());
    }
}
