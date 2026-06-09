//! Triangulate a region of a sphere bounded by arbitrary loops, via an
//! azimuthal-equidistant chart + constrained Delaunay (`TO_SES_EXACT_COMPLETION.md`).
//!
//! This is the contact-cap interior mesher: the exposed region of an atom is a
//! sphere minus `k` buried-cap holes (not a disk for `k ≥ 2`, so a fan can't mesh
//! it) — and typically *most of the sphere*, so stereographic projection can't
//! bound it (codex-review). We project the boundary loops with an
//! azimuthal-equidistant chart from a pole *inside* the region (the whole sphere
//! minus the antipode fits in a disk of radius π), constrained-Delaunay-
//! triangulate the planar polygon-with-holes (interior Steiner points for
//! curvature), and lift every vertex back onto the sphere. Boundary vertices keep
//! their exact positions; only interior Steiner points are introduced (and they
//! lie exactly on the analytic sphere).

use super::cdt::constrained_triangulate;
use super::geom::{plane_basis, Vec3};
use super::mesh::Mesh;
use anyhow::{ensure, Result};

/// **Azimuthal-equidistant** projection from a pole *inside* the region (unit),
/// basis `(u, v)`. A direction at angle θ from the pole maps to a planar point at
/// radius θ in its azimuth, so the whole sphere (minus the antipode) maps into the
/// disk of radius π — large regions stay bounded, unlike stereographic
/// (codex-review: a contact cap is most of the sphere minus a few caps, which
/// stereographic from any valid pole sends to infinity).
struct Chart {
    pole: Vec3,
    u: Vec3,
    v: Vec3,
}

impl Chart {
    fn new(pole: Vec3) -> Self {
        let (u, v) = plane_basis(pole);
        Chart { pole, u, v }
    }
    fn forward(&self, dir: Vec3) -> [f64; 2] {
        let theta = dir.dot(self.pole).clamp(-1.0, 1.0).acos();
        let (au, av) = (dir.dot(self.u), dir.dot(self.v));
        let h = (au * au + av * av).sqrt(); // = sin θ
        if h < 1e-12 {
            [theta, 0.0] // at the pole (θ≈0 → origin) or antipode (azimuth arbitrary)
        } else {
            [au / h * theta, av / h * theta]
        }
    }
    fn inverse(&self, p: [f64; 2]) -> Vec3 {
        let theta = (p[0] * p[0] + p[1] * p[1]).sqrt();
        if theta < 1e-12 {
            return self.pole;
        }
        let (cu, cv) = (p[0] / theta, p[1] / theta);
        self.pole * theta.cos() + (self.u * cu + self.v * cv) * theta.sin()
    }
}

/// Center + angular radius of the **minimal enclosing spherical cap** of `dirs`
/// (the spherical 1-center), by the iterative move-toward-farthest method seeded
/// at `seed`. The center minimizes the maximum angle to any boundary direction,
/// so projecting from it keeps every boundary point as close to the chart pole as
/// possible — the least-distorted single azimuthal chart, which is what avoids the
/// projected boundary self-crossing. A poorly-placed pole (e.g. the antipode of
/// one buried cap) can push boundary points past 90° even when the region fits a
/// hemisphere; this finds the pole that doesn't.
fn enclosing_cap(dirs: &[Vec3], seed: Vec3) -> (Vec3, f64) {
    let mut c = seed.normalized().unwrap_or(dirs[0]);
    for step in 0..256 {
        // Farthest boundary direction from the current center (smallest dot).
        let far = dirs
            .iter()
            .copied()
            .fold((c, 2.0_f64), |(best, mind), d| {
                let dot = c.dot(d);
                if dot < mind {
                    (d, dot)
                } else {
                    (best, mind)
                }
            })
            .0;
        let rate = 1.0 / (step as f64 + 2.0);
        c = (c + (far - c) * rate).normalized().unwrap_or(c);
    }
    let r = dirs
        .iter()
        .map(|&d| c.dot(d).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max);
    (c, r)
}

/// Crossing-number (even-odd) point-in-region test over the projected loops.
/// Domain = inside an odd number of loops — consistent with the *sampled* polygon
/// (no analytic-vs-chord mismatch at the boundary; codex-review).
fn in_loops(loops2d: &[Vec<[f64; 2]>], p: [f64; 2]) -> bool {
    let mut inside = false;
    for lp in loops2d {
        let n = lp.len();
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (lp[i], lp[j]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                inside = !inside;
            }
            j = i;
        }
    }
    inside
}

/// Squared distance from `p` to segment `a–b`.
fn pt_seg_sq(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let c = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let d = [p[0] - c[0], p[1] - c[1]];
    d[0] * d[0] + d[1] * d[1]
}

/// Triangulate the sphere region `(center, radius)` bounded by `loops` (each an
/// ordered list of points **on the sphere**; the loops are the full boundary —
/// outer plus holes), filling the interior via an azimuthal-equidistant chart +
/// CDT. `interior_dir` is a unit direction strictly inside the region (the
/// projection pole). `grid` is the interior Steiner spacing in the chart plane.
///
/// The domain is defined by the **sampled** loops (even-odd), so it matches the
/// mesh boundary exactly. Errors if the region reaches the projection antipode
/// (single chart degenerate — needs multi-chart, a future extension). Boundary
/// vertices keep exact positions; interior Steiner points lie on the analytic
/// sphere.
pub fn fill_spherical_region(
    center: Vec3,
    radius: f64,
    loops: &[Vec<Vec3>],
    interior_dir: Vec3,
    grid: f64,
) -> Result<Mesh> {
    let dir_of = |p: Vec3| (p - center).normalized().expect("boundary point at centre");

    // Center the chart pole on the minimal enclosing cap of the boundary (seeded
    // by the caller's interior hint), not on a fixed heuristic direction: an
    // off-center pole can throw boundary points past 90°, where the azimuthal
    // chart distorts enough that the projected polygon self-crosses (the CDT
    // "boundary edge crosses an existing constraint" failure). The 1-center
    // minimizes the worst boundary angle. If even that exceeds a hemisphere, the
    // region genuinely cannot fit one azimuthal chart — bail clearly for
    // multi-chart rather than feed the CDT a self-crossing polygon.
    let all_dirs: Vec<Vec3> = loops.iter().flatten().map(|&w| dir_of(w)).collect();
    ensure!(!all_dirs.is_empty(), "no boundary points to fill");
    let hint = interior_dir.normalized().expect("interior_dir nonzero");
    // Re-center the chart pole on the boundary's minimal enclosing cap *only* when
    // that center lies on the region's side (same hemisphere as the trusted
    // interior hint) AND it strictly reduces the worst boundary angle. For a
    // disk-like region (boundary surrounds it) the enclosing-cap center is the
    // region centre — re-centering pulls boundary points off the high-distortion
    // rim and stops the projected polygon self-crossing. For a *complement* region
    // (sphere minus a small cap: the boundary is a small loop, its enclosing-cap
    // centers on the HOLE, opposite the region) the test fails and we keep the
    // caller's deep-interior pole — the correct one there. The inner per-point
    // antipode check remains the backstop.
    let maxang = |p: Vec3| {
        all_dirs
            .iter()
            .map(|&d| p.dot(d).clamp(-1.0, 1.0).acos())
            .fold(0.0_f64, f64::max)
    };
    let (cap_c, cap_r) = enclosing_cap(&all_dirs, hint);
    let pole0 = if cap_c.dot(hint) > 0.0 && cap_r < maxang(hint) {
        cap_c
    } else {
        hint
    };

    // One azimuthal-chart fill at a given pole: project the boundary, sprinkle an
    // interior Steiner grid, constrained-Delaunay-triangulate, lift back. The pole
    // affects ONLY the 2D triangulation — the boundary vertices (`world`) are the
    // verbatim shared samples regardless of pole — so retrying the pole is local
    // and weld-safe (no re-mesh, no cross-patch crack).
    let attempt = |pole: Vec3| -> Result<Mesh> {
        let chart = Chart::new(pole);
        let mut pts: Vec<[f64; 2]> = Vec::new();
        let mut world: Vec<Vec3> = Vec::new();
        let mut loop_idx: Vec<Vec<usize>> = Vec::new();
        let mut loops2d: Vec<Vec<[f64; 2]>> = Vec::new();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for lp in loops {
            let mut ids = Vec::with_capacity(lp.len());
            let mut l2 = Vec::with_capacity(lp.len());
            for &w in lp {
                let d = dir_of(w);
                ensure!(
                    d.dot(chart.pole) > -1.0 + 1e-6,
                    "region reaches the projection antipode — single chart degenerate"
                );
                let xy = chart.forward(d);
                ensure!(
                    xy[0].is_finite() && xy[1].is_finite(),
                    "non-finite projection"
                );
                lo = [lo[0].min(xy[0]), lo[1].min(xy[1])];
                hi = [hi[0].max(xy[0]), hi[1].max(xy[1])];
                ids.push(pts.len());
                pts.push(xy);
                world.push(w);
                l2.push(xy);
            }
            loop_idx.push(ids);
            loops2d.push(l2);
        }

        // Interior Steiner grid: keep points in the domain and clear of every
        // constraint *edge* (not just vertices — a point on an edge interior would
        // split it; codex-review), so the CDT keeps the boundary intact.
        let clear = (0.5 * grid) * (0.5 * grid);
        let near_edge = |xy: [f64; 2]| {
            loops2d.iter().any(|lp| {
                let n = lp.len();
                (0..n).any(|i| pt_seg_sq(xy, lp[i], lp[(i + 1) % n]) < clear)
            })
        };
        let mut gx = lo[0];
        while gx <= hi[0] {
            let mut gy = lo[1];
            while gy <= hi[1] {
                let xy = [gx, gy];
                if in_loops(&loops2d, xy) && !near_edge(xy) {
                    pts.push(xy);
                    world.push(center + chart.inverse(xy) * radius);
                }
                gy += grid;
            }
            gx += grid;
        }

        let in_domain = |c: [f64; 2]| in_loops(&loops2d, c);
        let tris = constrained_triangulate(&pts, &loop_idx, in_domain)?;
        Ok(Mesh {
            verts: world,
            normals: Vec::new(),
            tris: tris
                .iter()
                .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
                .collect(),
        })
    };

    match attempt(pole0) {
        Ok(m) => Ok(m),
        // A near-pinch boundary (two stretches passing within a sub-sample gap)
        // projects to a self-crossing polygon the CDT rejects — but the smooth
        // curve is simple, so the crossing is a knife-edge of the *projection*.
        // Re-project from slightly rotated poles (local, weld-safe — boundary verts
        // unchanged) before paying for a whole-protein perturbation re-mesh.
        Err(e) if e.to_string().contains("crosses an existing constraint") => {
            for trial in perturbed_poles(pole0) {
                if trial.dot(hint) > 0.0 {
                    if let Ok(m) = attempt(trial) {
                        return Ok(m);
                    }
                }
            }
            Err(e)
        }
        Err(e) => Err(e),
    }
}

/// Candidate chart poles around `pole` — small rotations in its tangent plane,
/// tried when the primary projection self-crosses (a near-pinch). Cheap: each is
/// just a re-projection + CDT, not a re-mesh.
fn perturbed_poles(pole: Vec3) -> Vec<Vec3> {
    let (u, v) = plane_basis(pole);
    let dirs = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7, 0.7),
        (-0.7, 0.7),
        (0.7, -0.7),
        (-0.7, -0.7),
    ];
    let mut out = Vec::new();
    for &mag in &[0.06_f64, 0.13, 0.22, 0.35] {
        for &(du, dv) in &dirs {
            if let Some(p) = (pole + u * (du * mag) + v * (dv * mag)).normalized() {
                out.push(p);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn perturbed_poles_are_unit_and_near_the_seed() {
        let pole = Vec3::new(0.2, -0.3, 1.0).normalized().unwrap();
        let cands = perturbed_poles(pole);
        assert!(cands.len() >= 16);
        for p in &cands {
            assert!((p.norm() - 1.0).abs() < 1e-9, "candidate poles are unit");
            // small rotations stay on the seed's side of the sphere.
            assert!(p.dot(pole) > 0.5, "candidate stays near the seed pole");
        }
    }

    fn circle_loop(center: Vec3, radius: f64, axis: Vec3, alpha: f64, n: usize) -> Vec<Vec3> {
        let (u, v) = plane_basis(axis);
        (0..n)
            .map(|k| {
                let t = 2.0 * PI * k as f64 / n as f64;
                let dir = axis * alpha.cos() + (u * t.cos() + v * t.sin()) * alpha.sin();
                center + dir * radius
            })
            .collect()
    }

    /// A single circular boundary at half-angle α bounds a spherical cap; the
    /// stereographic-chart fill must reproduce its area 2πr²(1−cos α) and leave a
    /// clean disk (boundary = the n loop edges). Validates project→CDT→lift.
    #[test]
    fn fills_a_spherical_cap_to_the_analytic_area() {
        let center = Vec3::new(0.4, -1.0, 2.0);
        let radius = 1.6;
        let axis = Vec3::new(0.1, 0.2, 1.0).normalized().unwrap();
        let alpha = PI / 4.0;
        let n = 64;
        let loops = vec![circle_loop(center, radius, axis, alpha, n)];

        let exact = 2.0 * PI * radius * radius * (1.0 - alpha.cos());
        let coarse = fill_spherical_region(center, radius, &loops, axis, 0.10).unwrap();
        let fine = fill_spherical_region(center, radius, &loops, axis, 0.05).unwrap();

        let (ac, af) = (coarse.surface_area(), fine.surface_area());
        assert!(
            (af - exact).abs() < (ac - exact).abs() + 1e-9,
            "area converges {ac} → {af} vs {exact}"
        );
        assert!(
            (af - exact).abs() / exact < 0.01,
            "fine cap area {af} within 1% of {exact}"
        );
        // Open disk: every boundary edge used once, and that count is the loop.
        assert_eq!(
            fine.num_nonmanifold_edges(),
            n,
            "boundary should be the single n-edge loop"
        );
    }

    /// A region that is *most of the sphere* minus one small cap — the case
    /// stereographic projection could not bound (codex-review). The pole sits in
    /// the large region, away from the hole; area must hit 2πr²(1+cos β).
    #[test]
    fn fills_a_large_region_minus_a_small_cap() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let radius = 1.5;
        let hole_axis = Vec3::new(0.0, 0.0, 1.0); // small cap removed around +z
        let beta = 0.4;
        let n = 64;
        let loops = vec![circle_loop(center, radius, hole_axis, beta, n)];
        // pole deep in the region, antipodal to the hole.
        let interior = Vec3::new(0.0, 0.0, -1.0);

        let exact = 2.0 * PI * radius * radius * (1.0 + beta.cos());
        let m = fill_spherical_region(center, radius, &loops, interior, 0.05).unwrap();
        assert!(
            (m.surface_area() - exact).abs() / exact < 0.01,
            "large-region area {} within 1% of {exact}",
            m.surface_area()
        );
        assert_eq!(m.num_nonmanifold_edges(), n, "single boundary loop");
    }
}
