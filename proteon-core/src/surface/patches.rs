//! L4 reentrant-face meshing — the concave SES patches.
//!
//! - **Spheric face** (RS face / probe on 3 atoms): the concave spherical
//!   triangle on the probe sphere, bounded by the three points where the probe
//!   touches the atoms. Its outward (solvent-facing) normal points *toward* the
//!   probe center, since the patch is concave.
//! - **Toric face** (RS edge / probe rolling over 2 atoms): the reentrant torus
//!   swept by the probe. Every surface point is exactly `probe_radius` from the
//!   roll-circle (the locus of probe centers) — the defining invariant we gate
//!   on. Normals point toward the generating probe center (concave).
//!
//! Watertight stitching across patch boundaries is the next step. Each patch is
//! gated on a closed-form quantity independent of the triangulation (spheric:
//! `radius² × spherical excess`; toric: distance-to-roll-circle + Pappus area).

use super::geom::{Circle3, Vec3};
use super::mesh::Mesh;
use std::collections::HashMap;

/// Solid angle (steradians) subtended by the spherical triangle of three unit
/// direction vectors — Van Oosterom & Strackee. Times `radius²` it is the area
/// of that triangle on a sphere of that radius.
pub fn spherical_excess(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let num = a.dot(b.cross(c)).abs();
    let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * num.atan2(den)
}

/// Geodesic mesh of the spherical triangle (`da`,`db`,`dc` are unit directions
/// from `center`) on the sphere of `radius`, `subdivisions` levels deep. If
/// `inward`, per-vertex normals point toward `center` (concave/reentrant);
/// otherwise outward.
fn geodesic_triangle(
    center: Vec3,
    radius: f64,
    da: Vec3,
    db: Vec3,
    dc: Vec3,
    subdivisions: u32,
    inward: bool,
) -> Mesh {
    let mut dirs = vec![da, db, dc];
    let mut tris = vec![[0u32, 1, 2]];
    let mut midpoint: HashMap<(u32, u32), u32> = HashMap::new();
    for _ in 0..subdivisions {
        let mut next = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = mid(a, b, &mut dirs, &mut midpoint);
            let bc = mid(b, c, &mut dirs, &mut midpoint);
            let ca = mid(c, a, &mut dirs, &mut midpoint);
            next.push([a, ab, ca]);
            next.push([b, bc, ab]);
            next.push([c, ca, bc]);
            next.push([ab, bc, ca]);
        }
        tris = next;
        midpoint.clear();
    }
    let normals = dirs.iter().map(|&d| if inward { -d } else { d }).collect();
    let verts = dirs.iter().map(|&d| center + d * radius).collect();
    Mesh {
        verts,
        normals,
        tris,
    }
}

fn mid(a: u32, b: u32, dirs: &mut Vec<Vec3>, cache: &mut HashMap<(u32, u32), u32>) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) {
        return i;
    }
    let m = (dirs[a as usize] + dirs[b as usize]).normalized().unwrap();
    let i = dirs.len() as u32;
    dirs.push(m);
    cache.insert(key, i);
    i
}

/// Mesh the spheric (concave probe-cap) SES face: the probe sphere centered at
/// `probe_center` with `probe_radius`, resting on three atoms at `atom_centers`.
/// The patch is the spherical triangle whose vertices are the contact points
/// (probe → atom directions); normals point inward (reentrant/concave).
pub fn spheric_face_mesh(
    probe_center: Vec3,
    probe_radius: f64,
    atom_centers: [Vec3; 3],
    subdivisions: u32,
) -> Mesh {
    // Direction from the probe center toward each atom = the contact direction.
    let dirs: Vec<Vec3> = atom_centers
        .iter()
        .map(|&a| {
            (a - probe_center)
                .normalized()
                .expect("atom coincident with probe")
        })
        .collect();
    geodesic_triangle(
        probe_center,
        probe_radius,
        dirs[0],
        dirs[1],
        dirs[2],
        subdivisions,
        true,
    )
}

/// Spherical linear interpolation between two unit vectors (`t` in `[0,1]`).
fn slerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    let ang = dot.acos();
    if ang < 1e-9 {
        return a;
    }
    let s = ang.sin();
    a * (((1.0 - t) * ang).sin() / s) + b * ((t * ang).sin() / s)
}

/// Mesh the toric (reentrant) SES face: the surface swept by the probe rolling
/// over atoms `atom_i`/`atom_j`. `roll_circle` is the locus of probe centers
/// (the contact circle of the two probe-inflated atoms). The probe center
/// sweeps `theta_range` around that circle; at each position the surface arc
/// runs from the contact with `atom_i` to the contact with `atom_j`. `wrap`
/// closes the θ direction for a free edge (full revolution). `n_theta`/`n_phi`
/// set the grid resolution. Normals point toward the probe center (concave).
pub fn toric_face_mesh(
    roll_circle: Circle3,
    probe_radius: f64,
    atom_i: Vec3,
    atom_j: Vec3,
    theta_range: (f64, f64),
    n_theta: usize,
    n_phi: usize,
    wrap: bool,
) -> Mesh {
    let (u, v) = super::geom::plane_basis(roll_circle.normal);
    let theta_steps = if wrap { n_theta } else { n_theta + 1 };
    let mut verts = Vec::with_capacity(theta_steps * (n_phi + 1));
    let mut normals = Vec::with_capacity(verts.capacity());
    for ti in 0..theta_steps {
        let f = ti as f64 / n_theta as f64;
        let theta = theta_range.0 + f * (theta_range.1 - theta_range.0);
        let p = roll_circle.center + (u * theta.cos() + v * theta.sin()) * roll_circle.radius;
        let di = (atom_i - p).normalized().unwrap();
        let dj = (atom_j - p).normalized().unwrap();
        for pj in 0..=n_phi {
            let d = slerp(di, dj, pj as f64 / n_phi as f64);
            verts.push(p + d * probe_radius);
            normals.push(-d); // concave: toward the probe center
        }
    }

    let row = n_phi + 1;
    let mut tris = Vec::new();
    // n_theta quads in θ for both modes: when wrapping, the last quad closes
    // back to row 0 (via %); otherwise rows 0..=n_theta exist and ti+1 is valid.
    for ti in 0..n_theta {
        let t0 = ti % theta_steps;
        let t1 = (ti + 1) % theta_steps;
        for pj in 0..n_phi {
            let a = (t0 * row + pj) as u32;
            let b = (t1 * row + pj) as u32;
            let c = (t1 * row + pj + 1) as u32;
            let d = (t0 * row + pj + 1) as u32;
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    Mesh {
        verts,
        normals,
        tris,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::geom::{intersect_two_spheres, Sphere};
    use std::f64::consts::PI;

    /// Min distance from a point to the circle (center, unit normal, radius).
    fn dist_to_circle(p: Vec3, c: &Circle3) -> f64 {
        let rel = p - c.center;
        let h = rel.dot(c.normal);
        let radial = (rel - c.normal * h).norm();
        ((radial - c.radius).powi(2) + h * h).sqrt()
    }

    #[test]
    fn spherical_excess_octant_is_half_pi() {
        // Three orthogonal directions span 1/8 of the sphere → solid angle π/2.
        let e = spherical_excess(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!((e - PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn spheric_face_area_matches_spherical_excess() {
        // Probe at origin, three atoms along the axes → the spheric face is the
        // octant triangle: analytic area = R² · (π/2). Mesh area must converge.
        let r = 1.4;
        let c = Vec3::new(2.0, -1.0, 0.5); // off-origin to exercise translation
        let atoms = [
            c + Vec3::new(3.0, 0.0, 0.0),
            c + Vec3::new(0.0, 3.0, 0.0),
            c + Vec3::new(0.0, 0.0, 3.0),
        ];
        let exact = r * r * (PI / 2.0);

        let coarse = spheric_face_mesh(c, r, atoms, 2).surface_area();
        let fine = spheric_face_mesh(c, r, atoms, 5).surface_area();
        assert!(
            (fine - exact).abs() < (coarse - exact).abs(),
            "area converges with subdivision"
        );
        assert!((fine - exact).abs() / exact < 0.005, "fine within 0.5%");

        // Reentrant face: normals point toward the probe center.
        let m = spheric_face_mesh(c, r, atoms, 1);
        for (v, n) in m.verts.iter().zip(&m.normals) {
            let toward_center = (c - *v).normalized().unwrap();
            assert!(n.dot(toward_center) > 0.9, "normal must point inward");
        }
        // It is an open patch (a triangle has a boundary), so not watertight.
        assert!(!m.is_watertight());
    }

    #[test]
    fn toric_face_lies_on_the_probe_surface() {
        // Two equal atoms on the z-axis; the probe rolls fully around them →
        // a free toric ring. roll-circle = contact circle of the inflated atoms.
        let (r_atom, probe) = (1.5, 1.4);
        let ai = Vec3::new(0.0, 0.0, -2.0);
        let aj = Vec3::new(0.0, 0.0, 2.0);
        let roll = intersect_two_spheres(
            Sphere::new(ai, r_atom + probe),
            Sphere::new(aj, r_atom + probe),
        )
        .unwrap();
        // center on the axis, radius √((r+probe)² − 2²) = √(2.9²−4) = 2.1.
        assert!((roll.radius - 2.1).abs() < 1e-9);

        let m = toric_face_mesh(roll, probe, ai, aj, (0.0, 2.0 * PI), 64, 16, true);

        // DEFINING INVARIANT: every toric surface point is exactly `probe` away
        // from the roll-circle (the locus of probe centers).
        for &v in &m.verts {
            assert!(
                (dist_to_circle(v, &roll) - probe).abs() < 1e-9,
                "toric vertex not on the probe surface"
            );
        }
        // θ is closed (free ring) but the φ ends are open → 2 boundary loops.
        assert!(!m.is_watertight());

        // Area cross-check via Pappus (surface of revolution about the axis):
        // integrate the generator arc at high resolution as ground truth.
        let (u, _v) = crate::surface::geom::plane_basis(roll.normal);
        let p0 = roll.center + u * roll.radius;
        let di = (ai - p0).normalized().unwrap();
        let dj = (aj - p0).normalized().unwrap();
        let steps = 4000;
        let mut pappus = 0.0;
        let mut prev = p0 + slerp(di, dj, 0.0) * probe;
        for s in 1..=steps {
            let pt = p0 + slerp(di, dj, s as f64 / steps as f64) * probe;
            let seg = (pt - prev).norm();
            let x_mid = {
                let m = (pt + prev) * 0.5 - roll.center;
                (m - roll.normal * m.dot(roll.normal)).norm() // axis distance
            };
            pappus += 2.0 * PI * x_mid * seg;
            prev = pt;
        }
        let rel = (m.surface_area() - pappus).abs() / pappus;
        assert!(rel < 0.01, "toric mesh area within 1% of Pappus: {rel}");
    }
}
