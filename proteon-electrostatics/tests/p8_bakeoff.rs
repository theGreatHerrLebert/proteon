//! P8.1 bake-off + single-panel error sweeps (plan `devdocs/TO_ELECTROSTATICS_P8.md`
//! §3.2 / §5).
//!
//! Head-to-head of the two panel-aware cluster expansions — barycentric-Lagrange
//! (BLTC, `(p+1)³` tensor terms, kernel-independent) vs Cartesian multipole
//! (`C(p+3,3)` terms, Coulomb-specific recurrence) — on the cost that drives a
//! treecode: the expansion order `p` (hence term count = kernel evaluations) needed
//! to reach a target far-field accuracy. Plus the error sweeps codex asked for:
//! distance, **aspect ratio**, and **orientation**, for the single layer (both
//! methods) and the dipole (BLTC vector moments).
//!
//! Run with `--nocapture` to see the comparison table; the asserts pin the
//! conclusions the operator architecture choice rests on.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::fastsum::{cartesian, expansion};
use proteon_electrostatics::Tri;

const REF_ORDER: usize = 24; // direct-cubature reference order
const TARGET: f64 = 1e-6; // bake-off accuracy target

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// Minimal `p ∈ 1..=pmax` for which BLTC single-layer hits `TARGET` relative error
/// (or `None` if it never does in range).
fn min_p_bltc(tri: &Tri, xi: Vec3, reference: f64, pmax: usize) -> Option<usize> {
    let (lo, hi) = expansion::tri_bbox(tri);
    (1..=pmax).find(|&p| {
        let c = expansion::Cluster::new(lo, hi, p);
        let q = expansion::single_layer_moments(&c, &[(*tri, 1.0)], p);
        rel(expansion::eval_single_layer(&c, &q, xi), reference) < TARGET
    })
}

/// Minimal `p` for Cartesian single-layer.
fn min_p_cartesian(tri: &Tri, xi: Vec3, reference: f64, pmax: usize) -> Option<usize> {
    let (lo, hi) = expansion::tri_bbox(tri);
    let center = (lo + hi) * 0.5;
    (1..=pmax).find(|&p| {
        let m = cartesian::single_layer_moments(center, &[(*tri, 1.0)], p);
        rel(cartesian::eval_single_layer(center, &m, xi, p), reference) < TARGET
    })
}

/// A triangle of a given aspect ratio (`base` along x, height `base/aspect`), then
/// tilted out of plane so its bounding box is non-degenerate, and rotated by `rot`.
fn make_tri(aspect: f64, rot_deg: f64) -> Tri {
    let h = 1.0 / aspect;
    // In-plane base triangle, lifted in z for a 3-D bbox.
    let mut v = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.1),
        Vec3::new(0.5, h, 0.05),
    ];
    let a = rot_deg.to_radians();
    let (c, s) = (a.cos(), a.sin());
    for p in &mut v {
        // Rotate about the z axis then a bit about x, to exercise orientation.
        let x = p.x * c - p.y * s;
        let y = p.x * s + p.y * c;
        let z = p.z;
        // tilt about x
        *p = Vec3::new(x, y * c - z * s, y * s + z * c);
    }
    Tri::new(v[0], v[1], v[2])
}

#[test]
fn bakeoff_cartesian_costs_no_more_than_bltc() {
    eprintln!("\n{:>8} {:>6} {:>8} {:>8} {:>8} {:>8}", "sep", "aspect", "p_bltc", "p_cart", "t_bltc", "t_cart");
    let mut all_ok = true;
    for &sep in &[3.0_f64, 5.0, 8.0] {
        for &aspect in &[1.0_f64, 2.0, 4.0] {
            let tri = make_tri(aspect, 25.0);
            // Target down the +xyz diagonal at distance ~sep from the unit-ish panel.
            let xi = Vec3::new(sep, sep * 0.8, sep * 1.1);
            let reference = expansion::direct_single_layer(&[(tri, 1.0)], xi, REF_ORDER);
            let pb = min_p_bltc(&tri, xi, reference, 14);
            let pc = min_p_cartesian(&tri, xi, reference, 14);
            let (Some(pb), Some(pc)) = (pb, pc) else {
                eprintln!("{sep:>8.1} {aspect:>6.1}   (did not converge by p=14)");
                all_ok = false;
                continue;
            };
            let tb = (pb + 1).pow(3);
            let tc = cartesian::n_terms(pc);
            eprintln!("{sep:>8.1} {aspect:>6.1} {pb:>8} {pc:>8} {tb:>8} {tc:>8}");
            // The headline: at matched accuracy Cartesian never costs MORE terms than BLTC.
            assert!(tc <= tb, "sep={sep} aspect={aspect}: cartesian {tc} > bltc {tb}");
        }
    }
    assert!(all_ok, "every config should converge within p≤14");
}

#[test]
fn error_decreases_monotonically_with_distance() {
    // Far-field error falls as the target recedes — for BOTH methods.
    let tri = make_tri(1.0, 15.0);
    let (lo, hi) = expansion::tri_bbox(&tri);
    let center = (lo + hi) * 0.5;
    let c = expansion::Cluster::new(lo, hi, 6);
    let qb = expansion::single_layer_moments(&c, &[(tri, 1.0)], 6);
    let mc = cartesian::single_layer_moments(center, &[(tri, 1.0)], 6);

    let mut prev_b = f64::INFINITY;
    let mut prev_c = f64::INFINITY;
    for &d in &[3.0_f64, 5.0, 8.0, 12.0] {
        let xi = Vec3::new(d, d * 0.6, d * 0.9);
        let reference = expansion::direct_single_layer(&[(tri, 1.0)], xi, REF_ORDER);
        let eb = rel(expansion::eval_single_layer(&c, &qb, xi), reference);
        let ec = rel(cartesian::eval_single_layer(center, &mc, xi, 6), reference);
        assert!(eb < prev_b, "BLTC error should fall with distance at d={d}: {eb:.2e}");
        assert!(ec < prev_c, "Cartesian error should fall with distance at d={d}: {ec:.2e}");
        prev_b = eb;
        prev_c = ec;
    }
}

#[test]
fn accuracy_robust_to_aspect_ratio() {
    // Slivers up to 8:1 still converge in the far field (graceful min-p growth, not a
    // floor) — the panel-aware moments don't degrade on high aspect like a centroid
    // collapse would.
    let xi = Vec3::new(6.0, 5.0, 7.0);
    for &aspect in &[1.0_f64, 2.0, 4.0, 8.0] {
        let tri = make_tri(aspect, 20.0);
        let reference = expansion::direct_single_layer(&[(tri, 1.0)], xi, REF_ORDER);
        let p = min_p_bltc(&tri, xi, reference, 14);
        assert!(p.is_some(), "aspect={aspect} should converge by p=14");
        assert!(p.unwrap() <= 10, "aspect={aspect} needed p={} (>10)", p.unwrap());
    }
}

#[test]
fn accuracy_invariant_under_orientation() {
    // Rotating the panel (rigid motion of the source) leaves the far-field accuracy
    // essentially unchanged — a sanity check that no axis is privileged.
    let xi = Vec3::new(7.0, 6.0, 5.0);
    let p = 7;
    let mut errs = Vec::new();
    for &rot in &[0.0_f64, 30.0, 75.0, 120.0] {
        let tri = make_tri(2.0, rot);
        let (lo, hi) = expansion::tri_bbox(&tri);
        let c = expansion::Cluster::new(lo, hi, p);
        let q = expansion::single_layer_moments(&c, &[(tri, 1.0)], p);
        let reference = expansion::direct_single_layer(&[(tri, 1.0)], xi, REF_ORDER);
        errs.push(rel(expansion::eval_single_layer(&c, &q, xi), reference));
    }
    let max = errs.iter().copied().fold(0.0_f64, f64::max);
    assert!(max < 1e-5, "orientation sweep errors should all be small: {errs:?}");
}

#[test]
fn dipole_converges_across_aspect_and_orientation() {
    // The double layer (BLTC vector moments) is also robust to aspect/orientation.
    let xi = Vec3::new(6.0, 7.0, 5.0);
    for &aspect in &[1.0_f64, 3.0] {
        for &rot in &[10.0_f64, 95.0] {
            let tri = make_tri(aspect, rot);
            let (lo, hi) = expansion::tri_bbox(&tri);
            let c = expansion::Cluster::new(lo, hi, 8);
            let q = expansion::double_layer_moments(&c, &[(tri, 1.0)], 8);
            let reference = expansion::direct_double_layer(&[(tri, 1.0)], xi, REF_ORDER);
            let e = rel(expansion::eval_double_layer(&c, &q, xi), reference);
            assert!(e < 1e-6, "dipole aspect={aspect} rot={rot}: error {e:.2e}");
        }
    }
}
