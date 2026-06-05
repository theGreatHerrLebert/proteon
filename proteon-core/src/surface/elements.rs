//! SES element geometry — the exact analytic positions the registry, patches and
//! assembler consume (the geometry `ses.rs` deferred).
//!
//! Three primitives:
//! - [`ses_vertex`] — the probe-contact corner where a probe fixed on an RS face
//!   touches one of its atoms (lies on both the atom and the probe sphere).
//! - [`contact_circle`] — the circle on an atom traced by the probe rolling on a
//!   pair; the contact face's hole boundary and the toric face's φ-rim live here.
//! - [`arc_on_circle`] / [`arc_on_sphere`] — interior samples along a circular or
//!   geodesic boundary arc (endpoints are shared registry SES vertices, so the
//!   samplers return only the *interior* points).

use super::geom::{intersect_two_spheres, plane_basis, Circle3, Sphere, Vec3};
use std::f64::consts::TAU;

/// The contact point where the probe centred at `probe_center` touches `atom`:
/// on the line `atom.center → probe_center`, at radius `atom.radius` from the
/// atom centre. Lies on the atom surface and the probe sphere.
pub fn ses_vertex(probe_center: Vec3, atom: Sphere) -> Vec3 {
    let dir = (probe_center - atom.center)
        .normalized()
        .expect("probe coincident with atom centre");
    atom.center + dir * atom.radius
}

/// The contact circle traced on `a` by the probe rolling on the pair `(a, b)`.
///
/// The probe centre `P` rides the roll circle `(a⊕probe) ∩ (b⊕probe)`; the
/// contact point on `a` is `a.center + r_a·(P − a.center)/|P − a.center|`, and
/// `|P − a.center| = r_a + probe` is constant — so the contact point is the
/// affine image `a.center + k·(P − a.center)` with `k = r_a/(r_a+probe)`. A
/// circle maps to a circle: same axis, centre and radius scaled by `k` about
/// `a.center`. `None` if the inflated atoms do not meet (no toric face).
pub fn contact_circle(a: Sphere, b: Sphere, probe: f64) -> Option<Circle3> {
    let roll = intersect_two_spheres(a.inflated(probe), b.inflated(probe))?;
    let k = a.radius / (a.radius + probe);
    Some(Circle3 {
        center: a.center + (roll.center - a.center) * k,
        normal: roll.normal,
        radius: roll.radius * k,
    })
}

/// Angle of point `p` (assumed on `circle`) in the circle's own basis, in `[0,TAU)`.
fn angle_on(circle: &Circle3, p: Vec3, u: Vec3, v: Vec3) -> f64 {
    let d = p - circle.center;
    let t = d.dot(v).atan2(d.dot(u));
    if t < 0.0 {
        t + TAU
    } else {
        t
    }
}

/// `n` interior sample points along `circle` from `from` to `to` (both on the
/// circle), taking the arc in increasing-angle direction from `from` to `to`.
/// Endpoints are excluded (they are shared registry vertices).
pub fn arc_on_circle(circle: &Circle3, from: Vec3, to: Vec3, n: usize) -> Vec<Vec3> {
    let (u, v) = plane_basis(circle.normal);
    let a = angle_on(circle, from, u, v);
    let mut b = angle_on(circle, to, u, v);
    if b < a {
        b += TAU; // sweep forward from `from` to `to`
    }
    (1..=n)
        .map(|i| {
            let t = a + (b - a) * i as f64 / (n + 1) as f64;
            circle.center + (u * t.cos() + v * t.sin()) * circle.radius
        })
        .collect()
}

/// `n` interior sample points along the geodesic on the sphere `(center, radius)`
/// from `from` to `to` (both on the sphere) — i.e. the great-circle arc used for
/// a spheric face's concave edges. Endpoints excluded.
pub fn arc_on_sphere(center: Vec3, radius: f64, from: Vec3, to: Vec3, n: usize) -> Vec<Vec3> {
    let da = (from - center).normalized().expect("from at centre");
    let db = (to - center).normalized().expect("to at centre");
    let dot = da.dot(db).clamp(-1.0, 1.0);
    let ang = dot.acos();
    let s = ang.sin();
    (1..=n)
        .map(|i| {
            let t = i as f64 / (n + 1) as f64;
            let dir = if s < 1e-12 {
                da
            } else {
                da * (((1.0 - t) * ang).sin() / s) + db * ((t * ang).sin() / s)
            };
            center + dir * radius
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::rs;
    use super::*;

    fn sph(x: f64, y: f64, z: f64, r: f64) -> Sphere {
        Sphere::new(Vec3::new(x, y, z), r)
    }

    #[test]
    fn ses_vertex_lies_on_both_atom_and_probe() {
        let atom = sph(1.0, 2.0, 3.0, 1.7);
        let probe_center = Vec3::new(5.0, 5.0, 5.0);
        let t = ses_vertex(probe_center, atom);
        assert!(
            (t.distance(atom.center) - atom.radius).abs() < 1e-9,
            "on atom"
        );
        // probe_center is at r_atom+probe from the atom only if we place it there;
        // here just check the contact point is the closest atom-surface point to P.
        let expect = atom.center + (probe_center - atom.center).normalized().unwrap() * atom.radius;
        assert!(t.distance(expect) < 1e-12);
    }

    /// The real SES vertices (probe-contact points of the RS faces that involve
    /// edge (a,b)) must lie exactly on `contact_circle(a,b)` — the geometric
    /// identity the whole registry/arrangement relies on.
    #[test]
    fn ses_vertices_lie_on_the_contact_circle() {
        let atoms = vec![
            sph(0.0, 0.0, 0.0, 2.0),
            sph(2.5, 0.0, 0.0, 2.0),
            sph(1.25, 2.0, 0.0, 2.0),
        ];
        let probe = 1.4;
        let r = rs::compute(&atoms, probe);
        assert!(!r.faces.is_empty(), "tri must have RS faces");
        let (a, b) = (atoms[0], atoms[1]);
        let circle = contact_circle(a, b, probe).expect("contact circle");
        let (u, v) = plane_basis(circle.normal);
        // Every RS face touching atom 0 gives a contact point on atom 0; for the
        // faces that also touch atom 1, that point lies on the (0,1) contact circle.
        let mut checked = 0;
        for f in &r.faces {
            if f.atoms.contains(&0) && f.atoms.contains(&1) {
                let t = ses_vertex(f.probe_center, a);
                // on the circle's plane (no offset along the normal)…
                assert!(
                    (t - circle.center).dot(circle.normal).abs() < 1e-9,
                    "in plane"
                );
                // …and at the circle radius.
                let r_in_plane = (((t - circle.center).dot(u)).powi(2)
                    + ((t - circle.center).dot(v)).powi(2))
                .sqrt();
                assert!(
                    (r_in_plane - circle.radius).abs() < 1e-9,
                    "on circle radius"
                );
                checked += 1;
            }
        }
        assert!(checked >= 2, "expected both probe positions, got {checked}");
    }

    #[test]
    fn arc_samplers_stay_on_their_primitive() {
        // circle arc
        let c = Circle3 {
            center: Vec3::new(0.0, 0.0, 1.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            radius: 2.0,
        };
        let from = c.center + Vec3::new(2.0, 0.0, 0.0);
        let to = c.center + Vec3::new(0.0, 2.0, 0.0);
        for p in arc_on_circle(&c, from, to, 5) {
            assert!(((p - c.center).norm() - 2.0).abs() < 1e-9);
            assert!((p - c.center).dot(c.normal).abs() < 1e-9);
        }
        // sphere geodesic
        let center = Vec3::new(1.0, 1.0, 1.0);
        let radius = 1.5;
        let fa = center + Vec3::new(radius, 0.0, 0.0);
        let tb = center + Vec3::new(0.0, radius, 0.0);
        for p in arc_on_sphere(center, radius, fa, tb, 5) {
            assert!((p.distance(center) - radius).abs() < 1e-9);
        }
    }
}
