//! SES assembler — stitch the analytic patches (contact caps, toric and spheric
//! faces) into one watertight mesh, sharing boundaries through the [`registry`].
//!
//! Built bottom-up: this first step meshes a single atom's **contact face** end to
//! end, tying together the validated pieces — [`elements::buried_cap`] (one cap
//! per neighbour), [`arrangement::boundary_loops`] (the exposed-region boundary),
//! and [`chart::fill_spherical_region`] (the multi-hole interior fill). The toric
//! and spheric faces and the full multi-atom assembly follow.

use super::arrangement::{boundary_loops, sample_loop, SphereCircle};
use super::chart::fill_spherical_region;
use super::elements::{arc_on_sphere, buried_cap};
use super::geom::{Sphere, Vec3};
use super::mesh::Mesh;
use anyhow::{ensure, Context, Result};

/// Mesh `atom`'s contact face: its sphere outside the union of the buried caps
/// carved by each of `neighbours`. `grid` is the interior chart-plane spacing
/// (≈ angular spacing); `n_boundary` the samples per boundary arc.
///
/// Returns an open patch whose boundary is exactly the contact-circle arcs
/// (later shared with the toric faces); every vertex lies on `atom`'s sphere.
pub fn contact_cap_mesh(
    atom: Sphere,
    neighbours: &[Sphere],
    probe: f64,
    grid: f64,
    n_boundary: usize,
) -> Result<Mesh> {
    let caps: Vec<SphereCircle> = neighbours
        .iter()
        .filter_map(|&b| buried_cap(atom, b, probe))
        .collect();
    ensure!(
        !caps.is_empty(),
        "atom has no buried caps — a free atom's contact face is the whole sphere"
    );
    let loops = boundary_loops(&caps)?;
    let pole = pick_chart_pole(&caps).context(
        "no single chart pole with enough clearance — contact face needs \
         multi-chart handling (e.g. a band around an atom with opposite neighbours)",
    )?;

    let world_loops: Vec<Vec<Vec3>> = loops
        .iter()
        .map(|lp| {
            sample_loop(lp, &caps, n_boundary)
                .into_iter()
                .map(|d| atom.center + d * atom.radius)
                .collect()
        })
        .collect();

    fill_spherical_region(atom.center, atom.radius, &world_loops, pole, grid)
}

/// Choose an azimuthal-chart pole deep inside the exposed region: the candidate
/// (away from all neighbours, or the antipode of one neighbour) maximizing the
/// **minimum angular clearance** from every buried-cap boundary. Returns `None`
/// when no candidate clears the margin — a genuinely multi-chart region (e.g. a
/// band around an atom with two opposite neighbours), which the `-Σ axis`
/// heuristic would have silently mishandled (codex-review).
fn pick_chart_pole(caps: &[SphereCircle]) -> Option<Vec3> {
    let mut cands: Vec<Vec3> = caps.iter().map(|c| -c.axis).collect();
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    for c in caps {
        sum = sum - c.axis;
    }
    if let Some(s) = sum.normalized() {
        cands.push(s);
    }
    // Clearance = min over caps of (angle from the cap axis − its half-angle):
    // positive ⇒ exposed, larger ⇒ deeper in the exposed region.
    let clearance = |cand: Vec3| {
        caps.iter()
            .map(|c| cand.dot(c.axis).clamp(-1.0, 1.0).acos() - c.half_angle)
            .fold(f64::INFINITY, f64::min)
    };
    cands
        .into_iter()
        .map(|c| (clearance(c), c))
        .filter(|(s, _)| s.is_finite())
        .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
        .filter(|(s, _)| *s > 0.05) // require real margin (rad) from every boundary
        .map(|(_, c)| c)
}

/// The probe centre whose contact point on `a` is `t_a`: along the ray
/// `a.center → t_a`, at distance `r_a + probe`. Normalizing the direction (rather
/// than scaling `t_a − a.center` by `(r_a+probe)/r_a`) avoids amplifying any
/// radial drift in `t_a` by that factor (codex-review).
fn probe_center_from_contact(t_a: Vec3, a: Sphere, probe: f64) -> Option<Vec3> {
    let dir = (t_a - a.center).normalized()?;
    Some(a.center + dir * (a.radius + probe))
}

/// Mesh a toric (reentrant) face between atoms `a` and `b` from its two φ-rim
/// chains: `rim_a`/`rim_b` are the **same-θ** contact points on `a`/`b` (each θ a
/// probe position around the roll circle; shared with the two contact caps). At
/// each θ the reentrant surface is the probe-sphere arc from `rim_a[t]` to
/// `rim_b[t]`, sampled with `n_phi` interior points. `wrap` closes the θ-ring for
/// a *free* toric face (no spheric faces); a bounded face leaves the two θ-end
/// columns (the concave arcs) as its open boundary, shared with the spheric faces.
///
/// **Contract** (codex-review): the two rims must be the *same* probe positions —
/// `rim_a[t]` and `rim_b[t]` are the contacts of one probe centred at `P_t`. This
/// is enforced (`|rim_b[t] − P_t| ≈ probe`); the assembler must therefore derive
/// both rims from one shared roll-circle (probe-centre) chain, not sample the two
/// contact circles independently. Every vertex then lies exactly on its primitive.
pub fn toric_face_mesh(
    a: Sphere,
    rim_a: &[Vec3],
    rim_b: &[Vec3],
    probe: f64,
    n_phi: usize,
    wrap: bool,
) -> Result<Mesh> {
    ensure!(
        rim_a.len() == rim_b.len(),
        "φ-rims must be θ-aligned (equal length)"
    );
    let n_theta = rim_a.len();
    ensure!(
        n_theta >= if wrap { 3 } else { 2 },
        "toric face needs ≥{} θ columns",
        if wrap { 3 } else { 2 }
    );
    let row = n_phi + 2; // rim_a + n_phi interior + rim_b
    let mut verts = Vec::with_capacity(n_theta * row);
    for t in 0..n_theta {
        let p = probe_center_from_contact(rim_a[t], a, probe)
            .context("toric rim point coincides with its atom centre")?;
        // Misaligned rims (rim_b[t] not on P_t's probe sphere) would tear the
        // torus — refuse rather than emit twisted geometry.
        ensure!(
            (rim_a[t].distance(a.center) - a.radius).abs() < 1e-6,
            "toric rim_a[{t}] is not on atom a"
        );
        ensure!(
            (rim_b[t].distance(p) - probe).abs() < 1e-6,
            "toric rim_a/rim_b at θ={t} are not the same probe position"
        );
        verts.push(rim_a[t]);
        verts.extend(arc_on_sphere(p, probe, rim_a[t], rim_b[t], n_phi));
        verts.push(rim_b[t]);
    }
    let mut tris = Vec::new();
    let cols = if wrap { n_theta } else { n_theta - 1 };
    for t in 0..cols {
        let t2 = (t + 1) % n_theta;
        for p in 0..row - 1 {
            let (a0, b0) = ((t * row + p) as u32, (t * row + p + 1) as u32);
            let (a1, b1) = ((t2 * row + p) as u32, (t2 * row + p + 1) as u32);
            tris.push([a0, b0, b1]);
            tris.push([a0, b1, a1]);
        }
    }
    Ok(Mesh {
        verts,
        normals: Vec::new(),
        tris,
    })
}

/// The full SES of **two atoms** (one free toric ring + two contact caps, no
/// spheric faces) — the smallest end-to-end analytic assembly, and the proof that
/// the patches stitch watertight.
///
/// Watertightness by **bit-identical shared samples**: the toric face's two
/// φ-rims (`rim_a`/`rim_b`, sampled once per probe position) are passed *verbatim*
/// as the two caps' boundary loops, so the welded vertices coincide exactly rather
/// than within a tolerance. `n_theta` probe positions, `n_phi` toric φ-samples,
/// `grid` cap-chart spacing.
pub fn two_atom_ses(
    a: Sphere,
    b: Sphere,
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    grid: f64,
) -> Result<Mesh> {
    use super::elements::contact_circle;
    use super::geom::plane_basis;
    use std::f64::consts::TAU;

    let circle_a = contact_circle(a, b, probe).context("atoms share no toric face")?;
    let (u, v) = plane_basis(circle_a.normal);
    // One probe-position sweep defines BOTH rims (same probe → θ-aligned).
    let mut rim_a = Vec::with_capacity(n_theta);
    let mut rim_b = Vec::with_capacity(n_theta);
    for t in 0..n_theta {
        let th = TAU * t as f64 / n_theta as f64;
        let ta = circle_a.center + (u * th.cos() + v * th.sin()) * circle_a.radius;
        let p = probe_center_from_contact(ta, a, probe).context("degenerate rim point")?;
        rim_a.push(ta);
        rim_b.push(b.center + (p - b.center).normalized().context("probe at b centre")? * b.radius);
    }

    let toward_b = (b.center - a.center)
        .normalized()
        .context("coincident atoms")?;
    let mut mesh = toric_face_mesh(a, &rim_a, &rim_b, probe, n_phi, true)?;
    // Caps: boundary = the *same* rim Vec the toric used (→ exact weld). Pole away
    // from the neighbour (the buried cap's antipode), deep in the exposed region.
    mesh.append(&fill_spherical_region(
        a.center,
        a.radius,
        &[rim_a],
        -toward_b,
        grid,
    )?);
    mesh.append(&fill_spherical_region(
        b.center,
        b.radius,
        &[rim_b],
        toward_b,
        grid,
    )?);

    let mut mesh = mesh.welded(); // fuse the bit-identical shared rims
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip();
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::super::elements::buried_cap;
    use super::*;
    use std::f64::consts::PI;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// One neighbour ⇒ the contact face is the sphere minus one buried cap (a
    /// spherical zone). Its area is `2πr²(1+cos half_angle)` — the analytic check
    /// that the buried_cap → boundary_loops → chart-fill pipeline is correct, end
    /// to end, on a real atom.
    #[test]
    fn single_neighbour_contact_face_matches_the_analytic_zone() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let neighbour = sph(3.0, 0.0, 0.0, 1.6);
        let probe = 1.4;
        let cap = buried_cap(atom, neighbour, probe).unwrap();
        let exact = 2.0 * PI * atom.radius * atom.radius * (1.0 + cap.half_angle.cos());

        let coarse = contact_cap_mesh(atom, &[neighbour], probe, 0.12, 48).unwrap();
        let fine = contact_cap_mesh(atom, &[neighbour], probe, 0.06, 96).unwrap();
        let (ac, af) = (coarse.surface_area(), fine.surface_area());
        assert!(
            (af - exact).abs() < (ac - exact).abs() + 1e-9,
            "contact-face area converges {ac} → {af} vs {exact}"
        );
        assert!(
            (af - exact).abs() / exact < 0.01,
            "fine contact-face area {af} within 1% of {exact}"
        );
        // Open patch with a single boundary loop (the one contact circle).
        assert!(
            fine.num_nonmanifold_edges() > 0,
            "contact face has a boundary"
        );
        for v in &fine.verts {
            assert!(
                (v.distance(atom.center) - atom.radius).abs() < 1e-9,
                "every vertex on the atom sphere"
            );
        }
    }

    /// Two neighbours (the triangle3 atom) ⇒ the contact face is the sphere minus
    /// two buried caps. It must still mesh — a closed boundary, all vertices on
    /// the sphere — exercising the multi-cap arrangement + chart fill.
    #[test]
    fn two_neighbour_contact_face_meshes_on_the_sphere() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let n1 = sph(2.6, 0.0, 0.0, 1.7);
        let n2 = sph(1.3, 2.1, 0.0, 1.7);
        let m = contact_cap_mesh(atom, &[n1, n2], 1.4, 0.07, 48).unwrap();
        assert!(!m.tris.is_empty(), "non-empty mesh");
        assert!(m.num_nonmanifold_edges() > 0, "open patch with a boundary");
        for v in &m.verts {
            assert!((v.distance(atom.center) - atom.radius).abs() < 1e-9);
        }
    }

    /// The first watertight ANALYTIC SES assembled from contact caps + a toric
    /// face, gated against `ball-py 0.1.0a6 ses_area` — the proof the patches
    /// stitch closed and on-surface. Within 1% of BALL's analytic area/volume.
    #[test]
    fn two_atom_ses_is_watertight_and_matches_ball() {
        // (atom_a, atom_b, probe, ball area, ball volume)
        let cases = [
            (
                sph(0.0, 0.0, 0.0, 1.8),
                sph(2.5, 0.0, 0.0, 1.8),
                1.4,
                67.7959,
                46.6207,
            ),
            (
                sph(0.0, 0.0, 0.0, 2.0),
                sph(3.0, 0.0, 0.0, 1.2),
                1.4,
                64.3406,
                42.1575,
            ),
        ];
        for (a, b, probe, ball_area, ball_vol) in cases {
            let m = two_atom_ses(a, b, probe, 96, 10, 0.05).unwrap();
            assert!(m.is_watertight(), "assembled SES must be closed");
            assert!(m.is_consistently_oriented());
            assert_eq!(m.euler_characteristic(), 2, "sphere topology");
            let (area, vol) = (m.surface_area(), m.signed_volume());
            assert!(vol > 0.0, "outward oriented");
            assert!(
                (area - ball_area).abs() / ball_area < 0.01,
                "SES area {area} within 1% of ball {ball_area}"
            );
            assert!(
                (vol - ball_vol).abs() / ball_vol < 0.01,
                "SES volume {vol} within 1% of ball {ball_vol}"
            );
        }
    }

    /// Two opposite neighbours leave a band (annulus) around the atom — no single
    /// azimuthal pole charts it (its antipode is always in the region). The pole
    /// search must report this rather than silently producing a degenerate chart.
    #[test]
    fn opposite_neighbours_need_multi_chart_and_error() {
        let atom = sph(0.0, 0.0, 0.0, 1.7);
        let n1 = sph(2.6, 0.0, 0.0, 1.7);
        let n2 = sph(-2.6, 0.0, 0.0, 1.7);
        assert!(contact_cap_mesh(atom, &[n1, n2], 1.4, 0.1, 24).is_err());
    }

    /// A *free* toric face (two atoms, probe rolls all the way around) from
    /// θ-aligned φ-rims: every interior vertex is exactly `probe` from its rolling
    /// probe centre, the rims lie on the atoms, and the ring is a clean grid whose
    /// only boundary is the two φ-rim circles.
    #[test]
    fn free_toric_face_lies_on_the_probe_surface() {
        use super::super::elements::contact_circle;
        use super::super::geom::plane_basis;
        use std::f64::consts::TAU;
        let a = sph(0.0, 0.0, 0.0, 1.8);
        let b = sph(2.5, 0.0, 0.0, 1.8);
        let probe = 1.4;
        let circle_a = contact_circle(a, b, probe).unwrap();
        let (u, v) = plane_basis(circle_a.normal);
        let n_theta = 48;
        let n_phi = 6;
        // rim_a sampled around contact circle A; rim_b is the same probe's contact
        // point on B (θ-aligned by construction).
        let mut rim_a = Vec::new();
        let mut rim_b = Vec::new();
        for t in 0..n_theta {
            let th = TAU * t as f64 / n_theta as f64;
            let ta = circle_a.center + (u * th.cos() + v * th.sin()) * circle_a.radius;
            let p = probe_center_from_contact(ta, a, probe).unwrap();
            rim_a.push(ta);
            rim_b.push(b.center + (p - b.center).normalized().unwrap() * b.radius);
        }
        let m = toric_face_mesh(a, &rim_a, &rim_b, probe, n_phi, true).unwrap();
        for t in 0..n_theta {
            let p = probe_center_from_contact(rim_a[t], a, probe).unwrap();
            let row = n_phi + 2;
            for q in 1..=n_phi {
                let pt = m.verts[t * row + q];
                assert!(
                    (pt.distance(p) - probe).abs() < 1e-9,
                    "interior on probe sphere"
                );
            }
            assert!((rim_a[t].distance(a.center) - a.radius).abs() < 1e-9);
            assert!((rim_b[t].distance(b.center) - b.radius).abs() < 1e-9);
        }
        // A wrapped ring: boundary = the two φ-rim circles (2·n_theta edges).
        assert_eq!(m.num_nonmanifold_edges(), 2 * n_theta);
        // Consistent winding (open patch): no directed edge is traversed twice.
        let mut seen = std::collections::HashSet::new();
        for t in &m.tris {
            for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                assert!(seen.insert(e), "a directed edge repeats — winding flipped");
            }
        }
    }
}
