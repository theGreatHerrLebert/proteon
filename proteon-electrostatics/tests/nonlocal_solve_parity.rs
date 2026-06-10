//! P6 headline gate: nonlocal Cauchy data `(u, q, w)` vs NESSie `solve_NonlocalES`.
//!
//! Solves the coupled 3-block nonlocal system on the fixture geometry and asserts the
//! full Cauchy data matches NESSie's `:blas` (exact LU) fixture. The third block `w`
//! (`γ₀ext Ψ`) is the genuinely nonlocal unknown — the differentiator over local PB.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{solve_nonlocal_elements, Charge, Params, SolveConfig, Tri};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/solve_NonlocalES_blas_na.json"
    );
    serde_json::from_slice(&std::fs::read(path).expect("read")).expect("parse")
}

fn vec3(v: &Value) -> Vec3 {
    let a = v.as_array().unwrap();
    Vec3::new(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn rel(got: &[f64], want: &[f64]) -> f64 {
    let wn: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    let dn: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w) * (g - w))
        .sum::<f64>()
        .sqrt();
    dn / wn
}

#[test]
fn nonlocal_cauchy_data_matches_nessie() {
    let fix = fixture();
    let elements: Vec<Tri> = fix["elements"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            Tri::with_normal(
                vec3(&e["v1"]),
                vec3(&e["v2"]),
                vec3(&e["v3"]),
                vec3(&e["normal"]),
            )
        })
        .collect();
    let charges: Vec<Charge> = fix["charges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Charge {
            pos: vec3(&c["pos"]),
            val: c["val"].as_f64().unwrap(),
        })
        .collect();
    let p = &fix["params"];
    let params = Params {
        eps_omega: p["eps_omega"].as_f64().unwrap(),
        eps_sigma: p["eps_sigma"].as_f64().unwrap(),
        eps_inf: p["eps_inf"].as_f64().unwrap(),
        lambda: p["lambda"].as_f64().unwrap(),
    };

    let cfg = SolveConfig {
        tol: 1e-10,
        ..Default::default()
    };
    let (res, stats) = solve_nonlocal_elements(&elements, &charges, &params, &cfg).expect("solve");

    assert!(rel(&res.umol, &floats(&fix["umol"])) < 1e-10, "umol");
    assert!(rel(&res.qmol, &floats(&fix["qmol"])) < 1e-10, "qmol");
    assert!(rel(&res.u, &floats(&fix["u"])) < 1e-6, "u");
    assert!(rel(&res.q, &floats(&fix["q"])) < 1e-6, "q");
    assert!(rel(&res.w, &floats(&fix["w"])) < 1e-6, "w");
    assert!(stats.converged && stats.residual < 1e-8, "stats {stats:?}");
}
