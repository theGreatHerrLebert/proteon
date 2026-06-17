//! L4 assembly — stitch SES patches into a closed mesh.
//!
//! First target: the SES of **two atoms** — a free toric ring between them plus
//! one single-hole contact cap on each. The toric face's two φ-rims are reused
//! as the caps' rims (so the shared contact circles coincide); the patches are
//! welded, oriented consistently (flood-fill), and turned outward. Gated
//! end-to-end against `ball-py` (area / volume / watertightness).
//!
//! This is the smallest assembly that exercises cross-patch stitching. The
//! general assembler (the shared-index registry + the per-atom spherical-circle
//! arrangement for ≥2 incident holes) is the next step — see
//! `devdocs/TO_SES_STITCHING.md`.

use super::geom::{intersect_two_spheres, Sphere, Vec3};
use super::mesh::Mesh;
use super::patches::{contact_cap_mesh, toric_face_mesh};
use std::f64::consts::TAU;

/// Stitch the two-atom SES into one closed, consistently-oriented, outward mesh.
/// `n_theta`/`n_phi` set the toric grid; `n_lat` the contact-cap rings.
/// Panics if the atoms are too far apart to share a probe (no roll-circle).
pub fn ses_mesh_two_atoms(
    ai: Sphere,
    aj: Sphere,
    probe: f64,
    n_theta: usize,
    n_phi: usize,
    n_lat: usize,
) -> Mesh {
    let roll = intersect_two_spheres(ai.inflated(probe), aj.inflated(probe))
        .expect("atoms do not share a probe roll-circle");

    let toric = toric_face_mesh(
        roll,
        probe,
        ai.center,
        aj.center,
        (0.0, TAU),
        n_theta,
        n_phi,
        true,
    );

    // The toric grid is θ-major with row = n_phi+1; φ=0 is the contact circle on
    // atom i, φ=n_phi the one on atom j. Reuse them as the cap rims.
    let row = n_phi + 1;
    let rim_i: Vec<Vec3> = (0..n_theta).map(|t| toric.verts[t * row]).collect();
    let rim_j: Vec<Vec3> = (0..n_theta).map(|t| toric.verts[t * row + n_phi]).collect();

    let cap_i = cap_for(ai, aj.center, &rim_i, n_lat);
    let cap_j = cap_for(aj, ai.center, &rim_j, n_lat);

    let mut mesh = toric;
    mesh.append(&cap_i);
    mesh.append(&cap_j);
    let mut mesh = mesh.welded(); // fuses the bit-identical shared rims
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip(); // outward (positive enclosed volume)
    }
    mesh
}

/// Single-hole contact cap for `atom`, its hole facing `toward` (the other
/// atom), with the shared `rim` from the toric face. The contact circle is at a
/// constant polar angle about the axis, so all rim points share one `alpha`; we
/// average over the whole rim rather than sampling a single vertex.
fn cap_for(atom: Sphere, toward: Vec3, rim: &[Vec3], n_lat: usize) -> Mesh {
    let axis = (toward - atom.center).normalized().unwrap();
    let alpha = rim
        .iter()
        .map(|&p| {
            (p - atom.center)
                .normalized()
                .unwrap()
                .dot(axis)
                .clamp(-1.0, 1.0)
                .acos()
        })
        .sum::<f64>()
        / rim.len() as f64;
    contact_cap_mesh(atom.center, atom.radius, axis, alpha, rim, n_lat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    /// Two-atom SES gated against `ball-py 0.1.0a6 ses_area` across several
    /// configs — symmetric, asymmetric radii, and off-axis (exercises the
    /// circle-plane basis, not just x-axis symmetry).
    #[test]
    fn two_atom_ses_is_closed_and_matches_ball() {
        // (atom_i, atom_j, probe, ball area, ball volume)
        let cases = [
            (
                s(0.0, 0.0, 0.0, 1.8),
                s(2.5, 0.0, 0.0, 1.8),
                1.4,
                67.7959,
                46.6207,
            ),
            (
                s(0.0, 0.0, 0.0, 2.0),
                s(3.0, 0.0, 0.0, 1.2),
                1.4,
                64.3406,
                42.1575,
            ),
            (
                s(0.0, 0.0, 0.0, 1.7),
                s(1.4, 1.4, 0.9, 1.7),
                1.4,
                58.7211,
                38.2107,
            ),
        ];
        for (ai, aj, probe, ball_area, ball_vol) in cases {
            let mesh = ses_mesh_two_atoms(ai, aj, probe, 128, 24, 24);

            // Closed, consistently-oriented, sphere-topology, outward.
            assert!(mesh.is_watertight(), "stitched SES must be closed");
            assert!(mesh.is_consistently_oriented());
            assert_eq!(mesh.euler_characteristic(), 2);
            let area = mesh.surface_area();
            let vol = mesh.signed_volume();
            assert!(vol > 0.0, "outward orientation");

            // No degenerate triangles survive the weld (would otherwise hide a
            // pole/seam triangulation bug).
            for &t in &mesh.tris {
                let (a, b, c) = (
                    mesh.verts[t[0] as usize],
                    mesh.verts[t[1] as usize],
                    mesh.verts[t[2] as usize],
                );
                assert!(
                    0.5 * (b - a).cross(c - a).norm() > 1e-12,
                    "degenerate triangle"
                );
            }

            assert!(
                (area - ball_area).abs() / ball_area < 0.01,
                "SES area within 1% of ball-py: got {area} vs {ball_area}"
            );
            assert!(
                (vol - ball_vol).abs() / ball_vol < 0.01,
                "SES volume within 1% of ball-py: got {vol} vs {ball_vol}"
            );
        }
    }

    /// The weld depends on the cap rim being *bit-identical* to the toric rim
    /// (not merely close). Guard that contract directly.
    #[test]
    fn cap_rim_is_bit_identical_to_toric_rim() {
        let (ai, aj, probe) = (s(0.0, 0.0, 0.0, 1.7), s(1.4, 1.4, 0.9, 1.7), 1.4);
        let (n_theta, n_phi) = (32, 8);
        let roll = intersect_two_spheres(ai.inflated(probe), aj.inflated(probe)).unwrap();
        let toric = toric_face_mesh(
            roll,
            probe,
            ai.center,
            aj.center,
            (0.0, TAU),
            n_theta,
            n_phi,
            true,
        );
        let row = n_phi + 1;
        let rim_i: Vec<Vec3> = (0..n_theta).map(|t| toric.verts[t * row]).collect();
        let cap_i = cap_for(ai, aj.center, &rim_i, 8);
        for k in 0..n_theta {
            assert_eq!(
                cap_i.verts[k], rim_i[k],
                "cap ring-0 must equal the toric rim exactly"
            );
        }
    }
}
