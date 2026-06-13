//! P3 primary gate: Radon regular-Yukawa collocation vs NESSie `yukawa_dump`.
//!
//! Rebuilds NESSie's `yukawa_na.json` matrices entry-by-entry from
//! [`proteon_electrostatics::yukawa::regular_yukawa_collocation`] on byte-identical
//! geometry (the fixture's own vertices + normals) at the fixture's `yukawa`
//! exponent, and asserts agreement to libm precision. Cross-implementation parity;
//! the in-crate `yukawa` tests provide the independent (non-NESSie) checks.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::model::{PotentialKind, Tri};
use proteon_electrostatics::yukawa::regular_yukawa_collocation;
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/yukawa_na.json"
    );
    let bytes = std::fs::read(path).expect("read yukawa_na.json");
    serde_json::from_slice(&bytes).expect("parse yukawa_na.json")
}

fn vec3(v: &Value) -> Vec3 {
    let a = v.as_array().expect("xyz array");
    Vec3::new(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

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

fn check_layer(layer: &str, kind: PotentialKind) {
    let fix = fixture();
    let yukawa = fix["yukawa"].as_f64().expect("yukawa exponent");
    let tris = tris(&fix);
    let xis = obs_points(&fix);
    let reference = fix["matrices"][layer].as_array().expect("matrix rows");

    assert_eq!(reference.len(), xis.len(), "row count = #obs points");

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut worst = (0usize, 0usize, 0.0_f64, 0.0_f64);

    for (oidx, xi) in xis.iter().enumerate() {
        let row = reference[oidx].as_array().expect("matrix row");
        assert_eq!(row.len(), tris.len(), "col count = #elements");
        for (eidx, tri) in tris.iter().enumerate() {
            let want = row[eidx].as_f64().unwrap();
            let got = regular_yukawa_collocation(kind, *xi, tri, yukawa);
            assert!(
                got.is_finite() && want.is_finite(),
                "{layer}[{oidx},{eidx}]: non-finite got={got} want={want}"
            );
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-300);
            if abs > max_abs {
                max_abs = abs;
            }
            if want.abs() > 1e-9 && rel > max_rel {
                max_rel = rel;
                worst = (oidx, eidx, want, got);
            }
        }
    }

    // Same operations as NESSie (same Radon points/weights, same series guards), so
    // only libm exp/asin/sqrt divergence (≤1 ULP) remains. The bands are far looser
    // than observed but catch any structural error (wrong factor, branch, series).
    assert!(
        max_abs < 1e-10,
        "{layer}: max abs error {max_abs:.3e} (worst at [{},{}] want={} got={})",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    assert!(
        max_rel < 1e-10,
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
