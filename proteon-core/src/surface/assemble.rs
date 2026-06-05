//! SES assembler — stitch the analytic patches (contact caps, toric and spheric
//! faces) into one watertight mesh, sharing boundaries through the [`registry`].
//!
//! Built bottom-up: this first step meshes a single atom's **contact face** end to
//! end, tying together the validated pieces — [`elements::buried_cap`] (one cap
//! per neighbour), [`arrangement::boundary_loops`] (the exposed-region boundary),
//! and [`chart::fill_spherical_region`] (the multi-hole interior fill). The toric
//! and spheric faces and the full multi-atom assembly follow.

use super::arrangement::{boundary_loops, is_exposed, sample_loop, SphereCircle};
use super::chart::fill_spherical_region;
use super::elements::buried_cap;
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

    // Projection pole: a direction inside the exposed region, away from every
    // neighbour (so its antipode is buried, keeping the chart well-posed).
    let mut pole = Vec3::new(0.0, 0.0, 0.0);
    for c in &caps {
        pole = pole - c.axis;
    }
    let pole = pole
        .normalized()
        .context("cap axes cancel — no clear interior direction")?;
    ensure!(
        is_exposed(pole, &caps),
        "interior pole is buried — contact face needs multi-chart handling"
    );

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
}
