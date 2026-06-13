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

#[test]
fn nonlocal_approaches_local_as_correlation_length_shrinks() {
    // The nonlocal → local limit (plan invariant), at the *analytic* level where it is
    // clean: as the correlation length λ → 0, sinh(ν)/ν·e^(−ν) → 0 (ν = √(εΣ/ε∞)·R/λ),
    // so the nonlocal Born factor → (1/εΣ − 1) = the local one. (Stays in λ ≥ 0.1, the
    // numerically-safe regime — ν > 709 overflows sinh, and λ = 0 is unreachable by the
    // closed form, in NESSie too.)
    let radius = 2.0;
    let mut params = Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    };
    let local = born_rfenergy(1.0, radius, &params, Locality::Local);

    let mut prev_gap = f64::INFINITY;
    for &lambda in &[20.0_f64, 5.0, 1.0, 0.1] {
        params.lambda = lambda;
        let nl = born_rfenergy(1.0, radius, &params, Locality::Nonlocal);
        let gap = (nl - local).abs();
        assert!(
            gap.is_finite() && gap < prev_gap,
            "not approaching local as λ→0: λ={lambda} gap={gap}"
        );
        prev_gap = gap;
    }
    // By λ = 0.1 the nonlocal energy is within ~1% of the local one.
    params.lambda = 0.1;
    let nl = born_rfenergy(1.0, radius, &params, Locality::Nonlocal);
    assert!((nl - local).abs() / local.abs() < 0.01, "{nl} vs {local}");
}
