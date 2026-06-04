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
//! `TO_SES_STITCHING.md`.

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
    let mut mesh = mesh.welded(1e-6);
    mesh.orient_consistently();
    if mesh.signed_volume() < 0.0 {
        mesh.flip(); // outward (positive enclosed volume)
    }
    mesh
}

/// Single-hole contact cap for `atom`, its hole facing `toward` (the other
/// atom), with the shared `rim` from the toric face.
fn cap_for(atom: Sphere, toward: Vec3, rim: &[Vec3], n_lat: usize) -> Mesh {
    let axis = (toward - atom.center).normalized().unwrap();
    let alpha = (rim[0] - atom.center)
        .normalized()
        .unwrap()
        .dot(axis)
        .clamp(-1.0, 1.0)
        .acos();
    contact_cap_mesh(atom.center, atom.radius, axis, alpha, rim, n_lat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sph(x: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, 0.0, 0.0), r)
    }

    #[test]
    fn two_atom_ses_is_closed_and_matches_ball() {
        // pair2: two atoms r=1.8 at x=0 and x=2.5, probe 1.4. ball-py 0.1.0a6:
        // ses_area area=67.7959, volume=46.6207.
        let ai = sph(0.0, 1.8);
        let aj = sph(2.5, 1.8);
        let mesh = ses_mesh_two_atoms(ai, aj, 1.4, 128, 24, 24);

        // Closed, consistently-oriented manifold of sphere topology.
        assert!(mesh.is_watertight(), "stitched SES must be closed");
        assert!(mesh.is_consistently_oriented());
        assert_eq!(mesh.euler_characteristic(), 2);

        // Area + enclosed volume converge to the BALL oracle.
        let area = mesh.surface_area();
        let vol = mesh.signed_volume();
        assert!(vol > 0.0, "outward orientation");
        assert!(
            (area - 67.7959).abs() / 67.7959 < 0.01,
            "SES area within 1% of ball-py: got {area}"
        );
        assert!(
            (vol - 46.6207).abs() / 46.6207 < 0.01,
            "SES volume within 1% of ball-py: got {vol}"
        );
    }
}
