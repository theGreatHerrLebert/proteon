//! P4 headline gate: local Cauchy data vs NESSie `solve_dump` + LU-vs-GMRES.
//!
//! Solves the local BEM system on the fixture's byte-identical geometry and asserts:
//!  1. proteon's GMRES `u, q, umol, qmol` match NESSie's `:blas` (exact LU) fixture;
//!  2. a dense LU of proteon's *own* explicit `M` / `V` matches both the fixture
//!     (so proteon's assembly reproduces NESSie's) and proteon's GMRES;
//!  3. the true relative residual `‖M·u − b‖/‖b‖` is tiny.

use nalgebra::{DMatrix, DVector};
use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    laplace_matrices, mol_potentials, solve_local_elements, Charge, Params, SolveConfig, Tri,
    TWO_PI,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/solve_LocalES_blas_na.json"
    );
    serde_json::from_slice(&std::fs::read(path).expect("read solve fixture")).expect("parse")
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

fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    // Scale-invariant: relative to the reference vector's own norm (the entries span
    // several orders of magnitude, so per-entry relative error is noisy near zeros).
    let wnorm: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    let diff: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w) * (g - w))
        .sum::<f64>()
        .sqrt();
    diff / wnorm
}

fn load() -> (Vec<Tri>, Vec<Charge>, Params, Value) {
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
    (elements, charges, params, fix)
}

#[test]
fn gmres_cauchy_data_matches_nessie() {
    let (elements, charges, params, fix) = load();
    let cfg = SolveConfig {
        tol: 1e-11,
        ..Default::default()
    };
    let (res, stats) = solve_local_elements(&elements, &charges, &params, &cfg).expect("solve");

    // umol/qmol are direct evaluations — should match to libm precision.
    assert!(max_rel(&res.umol, &floats(&fix["umol"])) < 1e-10, "umol");
    assert!(max_rel(&res.qmol, &floats(&fix["qmol"])) < 1e-10, "qmol");

    // u/q vs the exact (:blas LU) fixture — GMRES-converged, so a touch looser.
    let u_rel = max_rel(&res.u, &floats(&fix["u"]));
    let q_rel = max_rel(&res.q, &floats(&fix["q"]));
    assert!(u_rel < 1e-7, "u rel error {u_rel:.2e}");
    assert!(q_rel < 1e-7, "q rel error {q_rel:.2e}");

    // True residual gate (not iteration count).
    assert!(stats.converged, "stats not converged: {stats:?}");
    assert!(
        stats.residual < 1e-8,
        "true residual {:.2e}",
        stats.residual
    );
}

#[test]
fn lu_matches_fixture_and_gmres() {
    // Dense LU of proteon's OWN explicit M and V. Matching the fixture validates that
    // proteon's assembly reproduces NESSie's (the fixture is NESSie's LU solution);
    // matching GMRES is the LU-vs-GMRES solver-parity check.
    let (elements, charges, params, fix) = load();
    let n = elements.len();
    let frac = params.eps_omega / params.eps_sigma;

    let (v, k) = laplace_matrices(&elements);
    let (umol, qmol) = mol_potentials(&elements, &charges, params.eps_omega);

    let vmat = DMatrix::from_row_slice(n, n, &v.data);
    let kmat = DMatrix::from_row_slice(n, n, &k.data);
    let umol_v = DVector::from_row_slice(&umol);
    let qmol_v = DVector::from_row_slice(&qmol);

    // Stage 1: M = 2π(1+frac)I + (frac−1)K ; b1 = K·umol − 2π·umol − frac·V·qmol.
    let m = &kmat * (frac - 1.0) + DMatrix::identity(n, n) * (TWO_PI * (1.0 + frac));
    let b1 = &kmat * &umol_v - &umol_v * TWO_PI - &vmat * &qmol_v * frac;
    let u = m.clone().lu().solve(&b1).expect("M LU");

    // Stage 2: b2 = (2π·I + K)·u ; V·q = b2.
    let b2 = &kmat * &u + &u * TWO_PI;
    let q = vmat.clone().lu().solve(&b2).expect("V LU");

    let u_lu: Vec<f64> = u.iter().copied().collect();
    let q_lu: Vec<f64> = q.iter().copied().collect();

    // LU vs the exact fixture — assembly parity (only collocation libm + reorder).
    assert!(
        max_rel(&u_lu, &floats(&fix["u"])) < 1e-9,
        "u: LU vs fixture"
    );
    assert!(
        max_rel(&q_lu, &floats(&fix["q"])) < 1e-9,
        "q: LU vs fixture"
    );

    // LU vs GMRES.
    let cfg = SolveConfig {
        tol: 1e-11,
        ..Default::default()
    };
    let (res, _) = solve_local_elements(&elements, &charges, &params, &cfg).expect("solve");
    assert!(max_rel(&res.u, &u_lu) < 1e-7, "u: GMRES vs LU");
    assert!(max_rel(&res.q, &q_lu) < 1e-7, "q: GMRES vs LU");
}
