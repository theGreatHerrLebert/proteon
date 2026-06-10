//! P4 entrywise-assembly gate (codex review + formulation spec §5).
//!
//! The solve-parity test proves proteon's assembled system *has* NESSie's solution;
//! it cannot rule out a compensating error in `M` and `b₁` that preserves the
//! solution vector. This test gates the assembly **entrywise** against NESSie's
//! `assembly_kernels_local` dump: the molecular-potential traces `umol`/`qmol`, the
//! system matrix `M = 2π(1+εΩ/εΣ)I + (εΩ/εΣ−1)K`, and the stage-1 RHS
//! `b₁ = K·umol − 2π·umol − (εΩ/εΣ)·V·qmol`, on the 32-element subset.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    laplace_matrices, mol_potentials, Charge, LinearOperator, LocalOperator, TWO_PI,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/assembly_kernels_local_na.json"
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

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn assembly_matches_nessie_entrywise() {
    use proteon_electrostatics::Tri;
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
    let eps_omega = p["eps_omega"].as_f64().unwrap();
    let frac = eps_omega / p["eps_sigma"].as_f64().unwrap();
    let n = elements.len();

    // umol / qmol entrywise.
    let (umol, qmol) = mol_potentials(&elements, &charges, eps_omega);
    assert!(
        max_abs(&umol, &floats(&fix["umol"])) < 1e-11,
        "umol entrywise"
    );
    assert!(
        max_abs(&qmol, &floats(&fix["qmol"])) < 1e-11,
        "qmol entrywise"
    );

    // V, K, then M = 2π(1+frac)I + (frac−1)K — compare M entrywise (built via the
    // implicit operator's column responses to unit vectors).
    let (v, k) = laplace_matrices(&elements);
    let m_op = LocalOperator { k: k.clone(), frac };
    let m_ref = fix["M"].as_array().unwrap();
    let mut e_j = vec![0.0; n];
    let mut col = vec![0.0; n];
    let mut max_m = 0.0_f64;
    for j in 0..n {
        e_j[j] = 1.0;
        m_op.matvec(&e_j, &mut col); // column j of M
        e_j[j] = 0.0;
        for (i, ci) in col.iter().enumerate() {
            let want = m_ref[i].as_array().unwrap()[j].as_f64().unwrap();
            max_m = max_m.max((ci - want).abs());
        }
    }
    assert!(max_m < 1e-10, "M entrywise: max abs {max_m:.3e}");

    // b1 = K·umol − 2π·umol − frac·V·qmol entrywise.
    let mut k_umol = vec![0.0; n];
    k.matvec(&umol, &mut k_umol);
    let mut v_qmol = vec![0.0; n];
    v.matvec(&qmol, &mut v_qmol);
    let b1: Vec<f64> = (0..n)
        .map(|i| k_umol[i] - TWO_PI * umol[i] - frac * v_qmol[i])
        .collect();
    assert!(max_abs(&b1, &floats(&fix["b1"])) < 1e-10, "b1 entrywise");
}
