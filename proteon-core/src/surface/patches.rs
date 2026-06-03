//! L4 reentrant-face meshing — the concave SES patches.
//!
//! - **Spheric face** (RS face / probe on 3 atoms): the concave spherical
//!   triangle on the probe sphere, bounded by the three points where the probe
//!   touches the atoms. Its outward (solvent-facing) normal points *toward* the
//!   probe center, since the patch is concave.
//!
//! Toric faces (probe rolling over 2 atoms) and watertight stitching across
//! patch boundaries are the next steps. Each patch is meshed by geodesic
//! subdivision of a spherical triangle and gated on its closed-form area
//! (`radius² × spherical excess`), independent of the triangulation.

use super::geom::Vec3;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
}
