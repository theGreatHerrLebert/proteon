//! P5 science gate (part 1): the closed-form Born energy vs NESSie `analytic.json`.
//!
//! The Born ion has an *exact* reaction-field energy — no BEM, no mesh. This gates
//! [`proteon_electrostatics::born_rfenergy`] (local + nonlocal) against NESSie's
//! `rfenergy(LocalES|NonlocalES, ion)` for all nine built-in ions. Pure closed form,
//! so the only residual is libm (`sinh`/`exp`/`sqrt`).

use proteon_electrostatics::{born_rfenergy, Locality, Params};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/analytic.json"
    );
    serde_json::from_slice(&std::fs::read(path).expect("read analytic.json")).expect("parse")
}

#[test]
fn born_energies_match_nessie() {
    let fix = fixture();
    let born = fix["born"].as_object().expect("born table");
    assert_eq!(born.len(), 9, "expected nine built-in ions");

    for (name, ion) in born {
        let charge = ion["charge"].as_f64().unwrap();
        let radius = ion["radius"].as_f64().unwrap();
        let p = &ion["params"];
        let params = Params {
            eps_omega: p["eps_omega"].as_f64().unwrap(),
            eps_sigma: p["eps_sigma"].as_f64().unwrap(),
            eps_inf: p["eps_inf"].as_f64().unwrap(),
            lambda: p["lambda"].as_f64().unwrap(),
        };

        let want_local = ion["local"].as_f64().unwrap();
        let got_local = born_rfenergy(charge, radius, &params, Locality::Local);
        let rel_l = (got_local - want_local).abs() / want_local.abs();
        assert!(
            rel_l < 1e-12,
            "{name} local: {got_local} vs {want_local} (rel {rel_l:.2e})"
        );

        let want_nl = ion["nonlocal"].as_f64().unwrap();
        let got_nl = born_rfenergy(charge, radius, &params, Locality::Nonlocal);
        let rel_n = (got_nl - want_nl).abs() / want_nl.abs();
        assert!(
            rel_n < 1e-12,
            "{name} nonlocal: {got_nl} vs {want_nl} (rel {rel_n:.2e})"
        );

        // Sanity: solvation energy is negative, and the nonlocal magnitude is below
        // the local one (the structured solvent screens less).
        assert!(got_local < 0.0 && got_nl < 0.0);
        assert!(
            got_nl.abs() < got_local.abs(),
            "{name}: nonlocal not < local"
        );
    }
}
