//! P2 primary gate: Rjasanow analytic Laplace collocation vs NESSie `collocation_dump`.
//!
//! Rebuilds NESSie's `collocation_na.json` matrices entry-by-entry from
//! [`proteon_electrostatics::laplace::laplace_collocation`] on byte-identical geometry
//! (the fixture's own vertices + normals) and asserts agreement to libm precision.
//! This is the cross-implementation parity gate; `laplace_quadrature.rs` is the
//! independent (non-NESSie) check.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::laplace::{laplace_collocation, Tri};
use proteon_electrostatics::model::PotentialKind;
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/collocation_na.json"
    );
    let bytes = std::fs::read(path).expect("read collocation_na.json");
    serde_json::from_slice(&bytes).expect("parse collocation_na.json")
}

fn vec3(v: &Value) -> Vec3 {
    let a = v.as_array().expect("xyz array");
    Vec3::new(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

/// Element triangles, consuming the fixture's stored normal verbatim (so geometry is
/// bit-identical to NESSie — no recomputed normal to drift the projection/sign).
fn tris(fix: &Value) -> Vec<Tri> {
    fix["elements"]
        .as_array()
        .expect("elements")
        .iter()
        .map(|e| {
            Tri::with_normal(
                vec3(&e["v1"]),
                vec3(&e["v2"]),
                vec3(&e["v3"]),
                vec3(&e["normal"]),
            )
        })
        .collect()
}

fn obs_points(fix: &Value) -> Vec<Vec3> {
    fix["observation_points"]
        .as_array()
        .expect("observation_points")
        .iter()
        .map(vec3)
        .collect()
}

/// Reference matrix `matrices[layer][oidx][eidx]`.
fn matrix<'a>(fix: &'a Value, layer: &str) -> &'a Vec<Value> {
    fix["matrices"][layer].as_array().expect("matrix rows")
}

fn check_layer(layer: &str, kind: PotentialKind) {
    let fix = fixture();
    let tris = tris(&fix);
    let xis = obs_points(&fix);
    let reference = matrix(&fix, layer);

    assert_eq!(reference.len(), xis.len(), "row count = #obs points");

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut worst = (0usize, 0usize, 0.0_f64, 0.0_f64);

    for (oidx, xi) in xis.iter().enumerate() {
        let row = reference[oidx].as_array().expect("matrix row");
        assert_eq!(row.len(), tris.len(), "col count = #elements");
        for (eidx, tri) in tris.iter().enumerate() {
            let want = row[eidx].as_f64().unwrap();
            let got = laplace_collocation(kind, *xi, tri);
            // A NaN would slip past the max-tracking below (every `>` is false), so
            // the maxima could stay finite and the test pass on garbage. Gate it.
            assert!(
                got.is_finite() && want.is_finite(),
                "{layer}[{oidx},{eidx}]: non-finite value got={got} want={want}"
            );
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-300);
            if abs > max_abs {
                max_abs = abs;
            }
            // Only track relative error where the reference is not ~0 (rel of a
            // near-zero entry is meaningless; the absolute gate covers those).
            if want.abs() > 1e-9 && rel > max_rel {
                max_rel = rel;
                worst = (oidx, eidx, want, got);
            }
        }
    }

    // Same formulas in the same order as NESSie ⟹ only asin/log/sqrt libm divergence
    // (≤1 ULP, openlibm vs Rust libm) remains. These bounds are ~6 orders of
    // magnitude looser than observed, but catch any structural error.
    assert!(
        max_abs < 1e-10,
        "{layer}: max abs error {max_abs:.3e} (worst rel at [{},{}] want={} got={})",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    assert!(
        max_rel < 1e-11,
        "{layer}: max rel error {max_rel:.3e} at [{},{}] want={} got={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
}

#[test]
fn single_layer_matches_nessie() {
    check_layer("single", PotentialKind::Single);
}

#[test]
fn double_layer_matches_nessie() {
    check_layer("double", PotentialKind::Double);
}
