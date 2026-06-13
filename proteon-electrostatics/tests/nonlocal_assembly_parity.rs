//! P6 entrywise-assembly gate (codex review): the nonlocal 3-block operator + RHS vs
//! NESSie, element by element — independent of the solve.
//!
//! The solve-parity test proves the assembled system *has* NESSie's solution; it
//! cannot rule out a compensating `A`/`b` error that preserves the solution. This
//! materializes proteon's `NonlocalOperator` (column by column) and asserts every
//! entry of the `3n×3n` matrix `M`, the RHS `b = [b1;0;0]`, and `umol`/`qmol` match
//! NESSie's `assembly_kernels_nonlocal` dump.

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    laplace_matrices, mol_potentials, yukawa_matrices, Charge, LinearOperator, NonlocalOperator,
    Tri, TWO_PI,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nessie/assembly_kernels_nonlocal_na.json"
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
fn nonlocal_assembly_matches_nessie_entrywise() {
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
    let (eo, es, ei, lam) = (
        p["eps_omega"].as_f64().unwrap(),
        p["eps_sigma"].as_f64().unwrap(),
        p["eps_inf"].as_f64().unwrap(),
        p["lambda"].as_f64().unwrap(),
    );
    let yuk = (es / ei).sqrt() / lam;
    let n = elements.len();

    // umol / qmol entrywise.
    let (umol, qmol) = mol_potentials(&elements, &charges, eo);
    assert!(
        max_abs(&umol, &floats(&fix["nonlocal_umol"])) < 1e-11,
        "umol"
    );
    assert!(
        max_abs(&qmol, &floats(&fix["nonlocal_qmol"])) < 1e-11,
        "qmol"
    );

    let (v, k) = laplace_matrices(&elements);
    let (vy, ky) = yukawa_matrices(&elements, yuk);

    // RHS b = [b1; 0; 0].
    let mv = |op: &proteon_electrostatics::DenseOperator, x: &[f64]| {
        let mut o = vec![0.0; n];
        op.matvec(x, &mut o);
        o
    };
    let (k_um, ky_um) = (mv(&k, &umol), mv(&ky, &umol));
    let (v_qm, vy_qm) = (mv(&v, &qmol), mv(&vy, &qmol));
    let mut b = vec![0.0; 3 * n];
    for i in 0..n {
        b[i] = k_um[i] + (1.0 - eo / es) * ky_um[i] - TWO_PI * umol[i] - (eo / ei) * v_qm[i]
            + (eo / es - eo / ei) * vy_qm[i];
    }
    assert!(
        max_abs(&b, &floats(&fix["nonlocal_b"])) < 1e-10,
        "b entrywise"
    );

    // Materialize the 3n×3n operator one column at a time and compare to NESSie's M.
    let op = NonlocalOperator {
        v,
        k,
        vy,
        ky,
        eps_omega: eo,
        eps_sigma: es,
        eps_inf: ei,
    };
    let dim = 3 * n;
    let m_ref = fix["nonlocal_M"].as_array().unwrap();
    let mut e_j = vec![0.0; dim];
    let mut col = vec![0.0; dim];
    let mut max_m = 0.0_f64;
    for j in 0..dim {
        e_j[j] = 1.0;
        op.matvec(&e_j, &mut col); // column j of M
        e_j[j] = 0.0;
        for (i, ci) in col.iter().enumerate() {
            let want = m_ref[i].as_array().unwrap()[j].as_f64().unwrap();
            max_m = max_m.max((ci - want).abs());
        }
    }
    assert!(max_m < 1e-10, "nonlocal M entrywise: max abs {max_m:.3e}");
}
