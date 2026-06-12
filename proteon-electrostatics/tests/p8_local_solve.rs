//! P8.2 end-to-end gate: a real local GMRES solve driven by the **treecode** V/K
//! operators, validated against the dense solve, the analytic Born energy, and — the
//! codex "true-dense-residual" gate — by checking the treecode solution actually solves
//! the *dense* system (not just the tree operator's own residual).

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::{
    analytic_sphere_mesh, born_rfenergy, laplace_matrices, rfenergy, solve_local_elements,
    solve_local_elements_treecode, solve_surface, Charge, FastSummation, LinearOperator, Locality,
    LocalOperator, Params, SolveConfig, SurfaceSolveOptions, Tri, TWO_PI,
};

fn sphere_elements(radius: f64, subdiv: u32) -> Vec<Tri> {
    let mesh = analytic_sphere_mesh(radius, subdiv);
    mesh.tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn treecode_local_solve_matches_dense_and_born() {
    let (radius, eo, es) = (2.0, 1.0, 78.0);
    let els = sphere_elements(radius, 3);
    let charges = [Charge { pos: Vec3::new(0.0, 0.0, 0.0), val: 1.0 }];
    let params = Params { eps_omega: eo, eps_sigma: es, eps_inf: 1.8, lambda: 20.0 };
    let cfg = SolveConfig { tol: 1e-9, ..Default::default() };

    let (rd, sd) = solve_local_elements(&els, &charges, &params, &cfg).expect("dense solve");
    let e_dense = rfenergy(&els, &charges, &rd);

    let (rt, st) =
        solve_local_elements_treecode(&els, &charges, &params, &cfg, 8, 0.45).expect("tc solve");
    let e_tree = rfenergy(&els, &charges, &rt);

    assert!(sd.converged && st.converged, "both solves must converge");

    // The treecode solve reproduces the dense reaction-field energy.
    let r_de = rel(e_tree, e_dense);
    eprintln!("rfenergy: dense {e_dense:.4}  treecode {e_tree:.4}  rel {r_de:.3e}");
    assert!(r_de < 2e-3, "treecode rfenergy off dense by {r_de:.3e}");

    // And both land near the analytic Born energy (the discretization-limited science
    // gate — independent of any BEM path).
    let born = born_rfenergy(1.0, radius, &params, Locality::Local);
    assert!(rel(e_dense, born) < 0.03 && rel(e_tree, born) < 0.03, "should track Born");
    assert!(e_tree < 0.0, "solvation energy is negative");
}

#[test]
fn treecode_solution_solves_the_dense_system() {
    // Codex's true-dense-residual gate: the treecode-computed u must solve the DENSE
    // stage-1 system M_dense·u = b1_dense (mol_potentials are exact and shared), not just
    // the tree operator's own residual. A small dense residual proves the fast-summation
    // solve found the true solution, not a self-consistent-but-wrong one.
    let els = sphere_elements(2.0, 3);
    let n = els.len();
    let charges = [Charge { pos: Vec3::new(0.3, -0.2, 0.1), val: 1.0 }];
    let params = Params { eps_omega: 1.0, eps_sigma: 78.0, eps_inf: 1.8, lambda: 20.0 };
    let cfg = SolveConfig { tol: 1e-9, ..Default::default() };

    let (rt, _) =
        solve_local_elements_treecode(&els, &charges, &params, &cfg, 8, 0.45).expect("tc solve");

    // Reconstruct the dense stage-1 system from the (exact, shared) molecular traces.
    let (v_dense, k_dense) = laplace_matrices(&els);
    let frac = params.eps_omega / params.eps_sigma;
    let mut k_umol = vec![0.0; n];
    k_dense.matvec(&rt.umol, &mut k_umol);
    let mut v_qmol = vec![0.0; n];
    v_dense.matvec(&rt.qmol, &mut v_qmol);
    let b1: Vec<f64> = (0..n)
        .map(|i| k_umol[i] - TWO_PI * rt.umol[i] - frac * v_qmol[i])
        .collect();

    let m_dense = LocalOperator { k: k_dense, frac };
    let mut mu = vec![0.0; n];
    m_dense.matvec(&rt.u, &mut mu);
    let resid: Vec<f64> = (0..n).map(|i| mu[i] - b1[i]).collect();
    let rel_resid = l2(&resid) / l2(&b1).max(1e-300);
    eprintln!("treecode u dense-residual: {rel_resid:.3e}");
    assert!(rel_resid < 2e-3, "treecode u doesn't solve the dense system: {rel_resid:.3e}");
}

#[test]
fn solve_surface_fast_summation_matches_dense() {
    // The opt-in SurfaceSolveOptions.fast_summation routes the local solve through the
    // treecode and must reproduce the dense surface_potential pipeline's energy + per-
    // vertex potential.
    let mesh = analytic_sphere_mesh(2.0, 3);
    let charges = [Charge { pos: Vec3::new(0.0, 0.0, 0.0), val: 1.0 }];
    let params = Params { eps_omega: 1.0, eps_sigma: 78.0, eps_inf: 1.8, lambda: 20.0 };
    let cfg = SolveConfig { tol: 1e-9, ..Default::default() };

    let dense_opts = SurfaceSolveOptions { params, cfg, ..Default::default() };
    let out_dense = solve_surface(mesh.clone(), &charges, &dense_opts).result.expect("dense");

    let tc_opts = SurfaceSolveOptions {
        params,
        cfg,
        fast_summation: Some(FastSummation { p: 8, theta: 0.45 }),
        ..Default::default()
    };
    let out_tc = solve_surface(mesh, &charges, &tc_opts).result.expect("treecode");

    assert!(out_tc.converged);
    let re = rel(out_tc.rfenergy, out_dense.rfenergy);
    eprintln!("surface_solve fast_summation rfenergy rel: {re:.3e}");
    assert!(re < 2e-3, "fast_summation rfenergy off dense by {re:.3e}");
    // Per-vertex potentials agree too.
    let dp = l2(&out_dense.potential);
    let diff: Vec<f64> = out_tc
        .potential
        .iter()
        .zip(&out_dense.potential)
        .map(|(a, b)| a - b)
        .collect();
    assert!(l2(&diff) / dp.max(1e-300) < 5e-3, "per-vertex potential drift too large");
}

#[test]
fn treecode_accuracy_tightens_with_p() {
    // The end-to-end energy error is controllable by the expansion order.
    let els = sphere_elements(2.0, 3);
    let charges = [Charge { pos: Vec3::new(0.0, 0.0, 0.0), val: 1.0 }];
    let params = Params { eps_omega: 1.0, eps_sigma: 78.0, eps_inf: 1.8, lambda: 20.0 };
    let cfg = SolveConfig { tol: 1e-9, ..Default::default() };
    let (rd, _) = solve_local_elements(&els, &charges, &params, &cfg).unwrap();
    let e_dense = rfenergy(&els, &charges, &rd);

    let err_at = |p: usize| {
        let (rt, _) =
            solve_local_elements_treecode(&els, &charges, &params, &cfg, p, 0.5).unwrap();
        rel(rfenergy(&els, &charges, &rt), e_dense)
    };
    let e4 = err_at(4);
    let e9 = err_at(9);
    eprintln!("energy rel-err: p=4 {e4:.3e}  p=9 {e9:.3e}");
    assert!(e9 < e4, "energy error should fall with p: {e4:.3e} -> {e9:.3e}");
}
