//! L0 geometry kernel for the molecular-surface pipeline.
//!
//! Vectors, spheres, circles, planes, lines, and the probe/contact predicates
//! the reduced-surface algorithm rolls on. The two load-bearing predicates —
//! probe sphere touching three atoms, and the contact circle of two atoms — are
//! ported directly from BALL's `analyticalGeometry.h` (`GetIntersection` for
//! three / two `TSphere3`), so the downstream reduced-surface combinatorics
//! match BALL's. No oracle dependency: every function here has a closed-form
//! unit test.

/// Comparison tolerance. BALL uses `Maths::EPSILON` (1e-6 by default) for the
/// `isZero`/`isLess`/`isEqual` guards in these predicates; we mirror that.
/// (Revisit when L1 parity against `reduced_surface_stats` is wired — the RS
/// ambiguity perturbation depends on the exact epsilon.)
pub const EPSILON: f64 = 1e-6;

/// A 3D point / vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn square_distance(self, o: Vec3) -> f64 {
        (self - o).norm_sq()
    }

    pub fn distance(self, o: Vec3) -> f64 {
        (self - o).norm()
    }

    /// Unit vector, or `None` if (near-)zero length.
    pub fn normalized(self) -> Option<Vec3> {
        let n = self.norm();
        if n < EPSILON {
            None
        } else {
            Some(self * (1.0 / n))
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// A sphere: center + radius. The atom/probe primitive of the RS algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

impl Sphere {
    pub const fn new(center: Vec3, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Inflate the radius by `dr` (the probe-radius offset, when rolling).
    pub fn inflated(self, dr: f64) -> Sphere {
        Sphere::new(self.center, self.radius + dr)
    }
}

/// A circle in 3D: center, unit normal, radius. The contact/torus circles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle3 {
    pub center: Vec3,
    pub normal: Vec3,
    pub radius: f64,
}

/// A plane `n · x + d = 0` (so `n · x = -d`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane3 {
    pub normal: Vec3,
    pub d: f64,
}

/// A line `p + t·dir`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line3 {
    pub p: Vec3,
    pub dir: Vec3,
}

/// Real roots of `a·x² + b·x + c = 0`, ascending. `None` if no real root
/// (or no equation). Degenerate `a≈0` falls back to the linear root (returned
/// as a pair of equal roots), matching BALL's `SolveQuadraticEquation`.
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Option<(f64, f64)> {
    if a.abs() < EPSILON {
        if b.abs() < EPSILON {
            return None;
        }
        let r = -c / b;
        return Some((r, r));
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let r1 = (-b - s) / (2.0 * a);
    let r2 = (-b + s) / (2.0 * a);
    Some(if r1 <= r2 { (r1, r2) } else { (r2, r1) })
}

/// Line of intersection of two planes, or `None` if (near-)parallel.
///
/// Direction is `n1 × n2`; the returned base point satisfies both planes (its
/// exact position along the line is immaterial — callers parameterize by `dir`).
pub fn intersect_planes(a: Plane3, b: Plane3) -> Option<Line3> {
    let dir = a.normal.cross(b.normal);
    let denom = dir.norm_sq();
    if denom < EPSILON * EPSILON {
        return None;
    }
    // Planes as n·x = c (c = -d). Standard closed form for a point on the line:
    //   p = (c1 (n2 × dir) + c2 (dir × n1)) / |dir|²
    let c1 = -a.d;
    let c2 = -b.d;
    let p = (b.normal.cross(dir) * c1 + dir.cross(a.normal) * c2) * (1.0 / denom);
    Some(Line3 { p, dir })
}

/// Contact circle of two spheres (the set of points on both), or `None` if they
/// do not intersect in a circle (disjoint, contained, or coincident centers).
///
/// Ported from BALL `GetIntersection(TSphere3 a, TSphere3 b, TCircle3&)`.
pub fn intersect_two_spheres(a: Sphere, b: Sphere) -> Option<Circle3> {
    let norm = b.center - a.center;
    let square_dist = norm.norm_sq();
    if square_dist < EPSILON {
        return None;
    }
    let dist = square_dist.sqrt();
    if a.radius + b.radius < dist {
        return None;
    }
    if (a.radius - b.radius).abs() >= dist {
        return None;
    }
    let r1_sq = a.radius * a.radius;
    let r2_sq = b.radius * b.radius;
    let u = r1_sq - r2_sq + square_dist;
    let length = u / (2.0 * square_dist);
    let square_radius = r1_sq - u * length / 2.0;
    if square_radius < 0.0 {
        return None;
    }
    Some(Circle3 {
        center: a.center + norm * length,
        normal: norm * (1.0 / dist),
        radius: square_radius.sqrt(),
    })
}

/// The (up to) two points touched by a sphere lying on three spheres — i.e. the
/// common intersection points of three spheres. For probe placement, pass the
/// three atoms inflated by the probe radius; the two results are the probe
/// centers. `None` if the three spheres have no common point.
///
/// Ported from BALL `GetIntersection(TSphere3 s1, s2, s3, p1, p2)`: two radical
/// planes → their line → intersect with sphere 1.
pub fn intersect_three_spheres(s1: Sphere, s2: Sphere, s3: Sphere) -> Option<(Vec3, Vec3)> {
    let r1_sq = s1.radius * s1.radius;
    let r2_sq = s2.radius * s2.radius;
    let r3_sq = s3.radius * s3.radius;
    let p1_sq = s1.center.norm_sq();
    let p2_sq = s2.center.norm_sq();
    let p3_sq = s3.center.norm_sq();
    let u = (r2_sq - r1_sq - p2_sq + p1_sq) / 2.0;
    let v = (r3_sq - r1_sq - p3_sq + p1_sq) / 2.0;

    let n1 = s2.center - s1.center;
    let n2 = s3.center - s1.center;
    if n1.norm_sq() < EPSILON || n2.norm_sq() < EPSILON {
        return None;
    }
    // Radical plane of (s1,s2): n1·x + u = 0 (i.e. n1·x = -u). Likewise (s1,s3).
    let plane1 = Plane3 { normal: n1, d: u };
    let plane2 = Plane3 { normal: n2, d: v };
    let line = intersect_planes(plane1, plane2)?;

    let diff = s1.center - line.p;
    let (x1, x2) = solve_quadratic(
        line.dir.dot(line.dir),
        -diff.dot(line.dir) * 2.0,
        diff.dot(diff) - r1_sq,
    )?;
    Some((line.p + line.dir * x1, line.p + line.dir * x2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn vec_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(b), 32.0));
        assert_eq!(a.cross(b), Vec3::new(-3.0, 6.0, -3.0));
        assert!(approx(a.cross(b).dot(a), 0.0)); // cross ⟂ both
        assert!(approx(a.cross(b).dot(b), 0.0));
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).norm(), 5.0));
    }

    #[test]
    fn quadratic_roots() {
        // x² - 3x + 2 = 0 → roots 1, 2
        let (r1, r2) = solve_quadratic(1.0, -3.0, 2.0).unwrap();
        assert!(approx(r1, 1.0) && approx(r2, 2.0));
        // no real root
        assert!(solve_quadratic(1.0, 0.0, 1.0).is_none());
        // degenerate (linear): 2x - 4 = 0 → 2
        let (l1, l2) = solve_quadratic(0.0, 2.0, -4.0).unwrap();
        assert!(approx(l1, 2.0) && approx(l2, 2.0));
    }

    #[test]
    fn planes_intersect_in_axis() {
        // x = 0 and y = 0 → the z-axis.
        let px = Plane3 {
            normal: Vec3::new(1.0, 0.0, 0.0),
            d: 0.0,
        };
        let py = Plane3 {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
        };
        let line = intersect_planes(px, py).unwrap();
        // base point on both planes (x=0, y=0)
        assert!(approx(line.p.x, 0.0) && approx(line.p.y, 0.0));
        // direction parallel to z
        let d = line.dir.normalized().unwrap();
        assert!(approx(d.x, 0.0) && approx(d.y, 0.0) && approx(d.z.abs(), 1.0));
    }

    #[test]
    fn parallel_planes_no_line() {
        let p1 = Plane3 {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let p2 = Plane3 {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: -5.0,
        };
        assert!(intersect_planes(p1, p2).is_none());
    }

    #[test]
    fn two_unit_spheres_contact_circle() {
        // Unit spheres at distance 1 → midplane circle, radius √(3)/2.
        let a = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let b = Sphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let c = intersect_two_spheres(a, b).unwrap();
        assert!(approx(c.center.x, 0.5));
        assert!(approx(c.center.y, 0.0) && approx(c.center.z, 0.0));
        assert!(approx(c.radius, 0.75_f64.sqrt()));
        assert!(approx(c.normal.x, 1.0)); // along center-to-center
    }

    #[test]
    fn disjoint_spheres_no_circle() {
        let a = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let b = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(intersect_two_spheres(a, b).is_none());
        // one contained in the other
        let big = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 5.0);
        let small = Sphere::new(Vec3::new(0.5, 0.0, 0.0), 1.0);
        assert!(intersect_two_spheres(big, small).is_none());
    }

    #[test]
    fn three_spheres_probe_points_equidistant() {
        // Equilateral triangle (side 1), spheres radius 0.8 → circumradius
        // 1/√3 ≈ 0.577 < 0.8, so two common points exist (above/below plane).
        let r = 0.8;
        let s1 = Sphere::new(Vec3::new(0.0, 0.0, 0.0), r);
        let s2 = Sphere::new(Vec3::new(1.0, 0.0, 0.0), r);
        let s3 = Sphere::new(Vec3::new(0.5, 3.0_f64.sqrt() / 2.0, 0.0), r);
        let (p1, p2) = intersect_three_spheres(s1, s2, s3).unwrap();
        for s in [s1, s2, s3] {
            assert!((p1.distance(s.center) - r).abs() < 1e-9, "p1 not on sphere");
            assert!((p2.distance(s.center) - r).abs() < 1e-9, "p2 not on sphere");
        }
        // The two solutions are mirror images across the z=0 triangle plane.
        assert!(approx(p1.z, -p2.z));
        assert!(p1.z.abs() > 1e-6); // genuinely off-plane
    }

    #[test]
    fn three_spheres_no_common_point() {
        // Far apart → no common intersection.
        let s1 = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 0.5);
        let s2 = Sphere::new(Vec3::new(10.0, 0.0, 0.0), 0.5);
        let s3 = Sphere::new(Vec3::new(0.0, 10.0, 0.0), 0.5);
        assert!(intersect_three_spheres(s1, s2, s3).is_none());
    }
}
