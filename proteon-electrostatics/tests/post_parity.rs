//! P5 gate (part 2): reaction-field energy + potentials vs NESSie `post_dump`.
//!
//! `post_dump` records `rfenergy` and `espotential` over Ω/Σ/Γ sample sets but not
//! `u`/`q`, so we solve the local system on the fixture geometry first, then compute
//! the post-processing outputs and compare. (`post_dump` itself uses the exact `:blas`
//! solve, so the only gap is our GMRES convergence carried through.)

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    espotential, rfenergy, solve_local_elements, Charge, Domain, Params, SolveConfig, Tri,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/post_local_na.json"
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

fn points(v: &Value) -> Vec<Vec3> {
    v.as_array().unwrap().iter().map(vec3).collect()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Vector relative error to the reference norm (entries span orders of magnitude).
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
fn rfenergy_and_potentials_match_nessie() {
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
        tol: 1e-11,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&elements, &charges, &params, &cfg).expect("solve");

    // rfenergy (kJ/mol).
    let e = rfenergy(&elements, &charges, &res);
    let want_e = fix["rfenergy"].as_f64().unwrap();
    assert!(
        (e - want_e).abs() / want_e.abs() < 1e-6,
        "rfenergy {e} vs {want_e}"
    );

    // espotential over each domain at its sample set.
    for (key, domain) in [
        ("Ω", Domain::Omega),
        ("Σ", Domain::Sigma),
        ("Γ", Domain::Gamma),
    ] {
        let pts = points(&fix["sample_points"][key]);
        let want = floats(&fix["espotential"][key]);
        let got: Vec<f64> = pts
            .iter()
            .map(|&xi| espotential(domain, xi, &elements, &charges, &params, &res))
            .collect();
        let r = rel(&got, &want);
        assert!(r < 1e-6, "espotential {key}: rel {r:.2e}");
    }
}
