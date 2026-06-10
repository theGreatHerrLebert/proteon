//! GPU vs CPU parity for the Laplace collocation assembly (feature `cuda`).
//!
//! The CUDA kernel mirrors `laplace.rs` operation-for-operation, so the matrices must
//! agree with the rayon CPU path to libm precision (CUDA libdevice vs Rust libm for
//! asin/log/sqrt). Skips gracefully if no device is present.
#![cfg(feature = "cuda")]

use proteon_core::surface::geom::Vec3;
use proteon_electrostatics::system::{laplace_matrices_cpu, LinearOperator};
use proteon_electrostatics::{
    analytic_sphere_mesh,
    gpu::{laplace_matrices_gpu, laplace_matvec_gpu, solve_nonlocal_gpu, yukawa_matvec_gpu},
    solve_local_elements, solve_local_gpu, solve_nonlocal_elements, yukawa_matrices, Charge, Params,
    SolveConfig, Tri,
};

/// Relative L2 error `‖a − b‖ / ‖b‖`.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
    num / den.max(1e-300)
}

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

fn local_params() -> Params {
    Params {
        eps_omega: 1.0,
        eps_sigma: 78.0,
        eps_inf: 1.8,
        lambda: 20.0,
    }
}

#[test]
fn gpu_laplace_matrices_match_cpu() {
    let mesh = analytic_sphere_mesh(2.0, 3); // 1280 triangles
    let elements: Vec<Tri> = mesh
        .tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect();

    let Some((vg, kg)) = laplace_matrices_gpu(&elements) else {
        eprintln!("no CUDA device — skipping GPU parity");
        return;
    };
    let (vc, kc) = laplace_matrices_cpu(&elements);

    let max = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    let dv = max(&vg.data, &vc.data);
    let dk = max(&kg.data, &kc.data);
    eprintln!("GPU vs CPU max abs: V {dv:.2e}, K {dk:.2e}");
    assert!(dv < 1e-9, "single-layer GPU vs CPU max abs {dv:.2e}");
    assert!(dk < 1e-9, "double-layer GPU vs CPU max abs {dk:.2e}");
}

/// Deterministic non-trivial test vector (no `rand` dep): a spread of signs and
/// magnitudes so the matvec exercises every column.
fn test_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (0.37 * t).sin() * 1.3 - (0.11 * t).cos() * 0.7 + 0.05 * ((i % 7) as f64 - 3.0)
        })
        .collect()
}

/// Kernel-isolation gate: the matrix-free GPU `V·x` / `K·x` must equal the CPU dense
/// matrices applied to the same `x`, *independent of GMRES*. This pins down kernel
/// indexing (row i / column j) and accumulation order — a solve-level test could mask
/// an indexing bug behind GMRES convergence.
#[test]
fn gpu_matrix_free_matvec_matches_cpu_dense() {
    let elements = sphere_elements(2.0, 2); // 320 triangles
    let (vc, kc) = laplace_matrices_cpu(&elements);
    let x = test_vector(elements.len());

    let (Some(vx_gpu), Some(kx_gpu)) = (
        laplace_matvec_gpu(&elements, 0, &x),
        laplace_matvec_gpu(&elements, 1, &x),
    ) else {
        eprintln!("no CUDA device — skipping matrix-free matvec parity");
        return;
    };
    let mut vx_cpu = vec![0.0; x.len()];
    let mut kx_cpu = vec![0.0; x.len()];
    vc.matvec(&x, &mut vx_cpu);
    kc.matvec(&x, &mut kx_cpu);

    let rel = |a: &[f64], b: &[f64]| {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
        num / den.max(1e-300)
    };
    let rv = rel(&vx_gpu, &vx_cpu);
    let rk = rel(&kx_gpu, &kx_cpu);
    eprintln!("matrix-free matvec rel err: V·x {rv:.2e}, K·x {rk:.2e}");
    // GPU FMA/libdevice vs CPU libm — same op order, so this is rounding only.
    assert!(rv < 1e-11, "V·x rel err {rv:.2e}");
    assert!(rk < 1e-11, "K·x rel err {rk:.2e}");
}

/// The matrix-free GPU local solve must land on the same Cauchy data as the dense
/// two-stage solve — same GMRES, same RHS; only `K·x`/`V·x` differ (recomputed on the
/// GPU vs read from a stored matrix). Note: under the `cuda` feature
/// `solve_local_elements` itself assembles `V`/`K` on the GPU (the dense build), so
/// this is matrix-free-GPU vs dense-GPU; the CPU-dense kernel is gated separately by
/// [`gpu_matrix_free_matvec_matches_cpu_dense`]. Run twice with a central and an
/// off-center charge so the RHS is non-degenerate.
#[test]
fn gpu_matrix_free_local_solve_matches_cpu_dense() {
    let elements = sphere_elements(2.0, 2); // 320 triangles
    let params = local_params();
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };

    for charge_pos in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.6, -0.4, 0.3)] {
        let charges = [Charge {
            pos: charge_pos,
            val: 1.0,
        }];

        let Some(gpu_res) = solve_local_gpu(&elements, &charges, &params, &cfg) else {
            eprintln!("no CUDA device — skipping matrix-free solve parity");
            return;
        };
        let (gpu, gstats) = gpu_res.expect("GPU matrix-free solve");
        let (cpu, _) =
            solve_local_elements(&elements, &charges, &params, &cfg).expect("dense solve");

        let rel = |a: &[f64], b: &[f64]| {
            let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            let den: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
            num / den.max(1e-300)
        };
        let ru = rel(&gpu.u, &cpu.u);
        let rq = rel(&gpu.q, &cpu.q);
        eprintln!(
            "matrix-free vs dense @ {charge_pos:?}: rel|Δu| {ru:.2e}, rel|Δq| {rq:.2e}, residual {:.2e}",
            gstats.residual
        );
        // Both converge GMRES to the same tol on the same system; the matvec path
        // differs only by GPU-vs-CPU rounding, so the solutions agree well within it.
        assert!(ru < 1e-6, "u rel err {ru:.2e} @ {charge_pos:?}");
        assert!(rq < 1e-6, "q rel err {rq:.2e} @ {charge_pos:?}");
        assert!(gstats.converged, "GPU solve did not converge @ {charge_pos:?}");
    }
}

/// Kernel-isolation gate for the regular-Yukawa kernel: matrix-free GPU `Vy·x` / `Ky·x`
/// must equal the CPU dense Yukawa matrices applied to the same `x`, independent of
/// GMRES. Pins down the Radon cubature, the series guard, and the `×2·area` factor on
/// the GPU against the CPU port.
#[test]
fn gpu_matrix_free_yukawa_matvec_matches_cpu_dense() {
    let elements = sphere_elements(2.0, 2); // 320 triangles
    let yukawa = local_params().yukawa();
    let (vy, ky) = yukawa_matrices(&elements, yukawa);
    let x = test_vector(elements.len());

    let (Some(vyx_gpu), Some(kyx_gpu)) = (
        yukawa_matvec_gpu(&elements, 0, yukawa, &x),
        yukawa_matvec_gpu(&elements, 1, yukawa, &x),
    ) else {
        eprintln!("no CUDA device — skipping Yukawa matvec parity");
        return;
    };
    let mut vyx_cpu = vec![0.0; x.len()];
    let mut kyx_cpu = vec![0.0; x.len()];
    vy.matvec(&x, &mut vyx_cpu);
    ky.matvec(&x, &mut kyx_cpu);

    let rvy = rel(&vyx_gpu, &vyx_cpu);
    let rky = rel(&kyx_gpu, &kyx_cpu);
    eprintln!("matrix-free Yukawa matvec rel err: Vy·x {rvy:.2e}, Ky·x {rky:.2e}");
    assert!(rvy < 1e-11, "Vy·x rel err {rvy:.2e}");
    assert!(rky < 1e-11, "Ky·x rel err {rky:.2e}");
}

/// The matrix-free GPU nonlocal solve must land on the same `(u, q, w)` Cauchy data as
/// the dense 3-block solve — same GMRES, same RHS, only the five matvecs differ
/// (recomputed on the GPU vs read from stored matrices).
#[test]
fn gpu_matrix_free_nonlocal_solve_matches_cpu_dense() {
    let elements = sphere_elements(2.0, 2); // 320 triangles → 960 unknowns
    let params = local_params();
    let cfg = SolveConfig {
        tol: 1e-8,
        ..Default::default()
    };

    for charge_pos in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.6, -0.4, 0.3)] {
        let charges = [Charge {
            pos: charge_pos,
            val: 1.0,
        }];

        let Some(gpu_res) = solve_nonlocal_gpu(&elements, &charges, &params, &cfg) else {
            eprintln!("no CUDA device — skipping nonlocal matrix-free parity");
            return;
        };
        let (gpu, gstats) = gpu_res.expect("GPU nonlocal solve");
        let (cpu, _) =
            solve_nonlocal_elements(&elements, &charges, &params, &cfg).expect("dense nonlocal");

        let ru = rel(&gpu.u, &cpu.u);
        let rq = rel(&gpu.q, &cpu.q);
        let rw = rel(&gpu.w, &cpu.w);
        eprintln!(
            "nonlocal matrix-free vs dense @ {charge_pos:?}: rel|Δu| {ru:.2e}, rel|Δq| {rq:.2e}, rel|Δw| {rw:.2e}, residual {:.2e}",
            gstats.residual
        );
        assert!(ru < 1e-6, "u rel err {ru:.2e} @ {charge_pos:?}");
        assert!(rq < 1e-6, "q rel err {rq:.2e} @ {charge_pos:?}");
        assert!(rw < 1e-6, "w rel err {rw:.2e} @ {charge_pos:?}");
        assert!(gstats.converged, "nonlocal GPU solve did not converge @ {charge_pos:?}");
    }
}

/// The size-aware dispatcher must be identical to the explicit dense solve on a mesh
/// that fits the dense budget — i.e. the common path is untouched, the GPU branch only
/// engages above [`proteon_electrostatics::DENSE_MATRIX_BUDGET`].
#[test]
fn dispatcher_matches_dense_below_budget() {
    use proteon_electrostatics::{dense_matrix_bytes, solve_local_elements_auto, DENSE_MATRIX_BUDGET};

    let elements = sphere_elements(2.0, 2); // 320 triangles — well under budget
    assert!(
        dense_matrix_bytes(elements.len()) <= DENSE_MATRIX_BUDGET,
        "test mesh must be under the dense budget"
    );
    let charges = [Charge {
        pos: Vec3::new(0.3, 0.2, -0.1),
        val: 1.0,
    }];
    let params = local_params();
    let cfg = SolveConfig {
        tol: 1e-9,
        ..Default::default()
    };

    let (auto, _) = solve_local_elements_auto(&elements, &charges, &params, &cfg).expect("auto");
    let (dense, _) = solve_local_elements(&elements, &charges, &params, &cfg).expect("dense");
    // Below budget the dispatcher *is* the dense path — bit-identical, not just close.
    assert_eq!(auto.u, dense.u, "u must be identical below budget");
    assert_eq!(auto.q, dense.q, "q must be identical below budget");
}

/// Empty input takes the `None` fallback (the CPU path then reports `Empty`), with or
/// without a device — exercising the boundary the solver promises (both localities).
#[test]
fn gpu_matrix_free_empty_falls_back() {
    let cfg = SolveConfig::default();
    let params = local_params();
    assert!(
        solve_local_gpu(&[], &[], &params, &cfg).is_none(),
        "empty mesh must fall back to CPU (None) — local"
    );
    assert!(
        solve_nonlocal_gpu(&[], &[], &params, &cfg).is_none(),
        "empty mesh must fall back to CPU (None) — nonlocal"
    );
}

/// Nonlocal scale demonstration: the four-matrix dense system needs `4·N²·8` bytes
/// (~13 GiB at 20,480 triangles), while the matrix-free path is O(N) GPU memory.
/// Honest framing: five matvecs per GMRES step make it slow per solve, but it removes
/// the dense ceiling entirely.
#[test]
#[ignore = "large mesh — run explicitly with --ignored on a CUDA box"]
fn gpu_matrix_free_nonlocal_scales_past_dense_budget() {
    let elements = sphere_elements(20.0, 5); // 20_480 triangles → 61_440 unknowns
    let n = elements.len();
    // 4·N²·8 exceeds the dense budget (so the auto-dispatcher would pick this path).
    assert!(4 * (n as u128).pow(2) * 8 > proteon_electrostatics::DENSE_MATRIX_BUDGET);
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let params = local_params();
    let cfg = SolveConfig {
        tol: 1e-7,
        ..Default::default()
    };

    let Some(res) = solve_nonlocal_gpu(&elements, &charges, &params, &cfg) else {
        eprintln!("no CUDA device — skipping nonlocal scale demo");
        return;
    };
    let (_, stats) = res.expect("matrix-free nonlocal solve on a mesh too large for dense host RAM");
    eprintln!(
        "matrix-free nonlocal solved {n} elements (dense ~13 GiB): {} iters, residual {:.2e}",
        stats.iterations, stats.residual
    );
    assert!(stats.converged, "matrix-free nonlocal large solve did not converge");
}

/// Scale demonstration: the matrix-free path solves a mesh whose dense `V`+`K` would
/// exceed [`proteon_electrostatics::gpu`]'s GPU memory budget. The dense GPU build
/// declines (returns `None`) past ~30k elements (`2·N²·8` bytes), while matrix-free —
/// O(N) device memory — still solves. Honest framing: slower *per solve* (the kernel
/// is recomputed every GMRES step), but unbounded in mesh size.
#[test]
#[ignore = "large mesh — run explicitly with --ignored on a CUDA box"]
fn gpu_matrix_free_scales_past_dense_budget() {
    let elements = sphere_elements(20.0, 6); // 81_920 triangles
    let n = elements.len();
    let charges = [Charge {
        pos: Vec3::new(0.0, 0.0, 0.0),
        val: 1.0,
    }];
    let params = local_params();
    let cfg = SolveConfig {
        tol: 1e-8,
        ..Default::default()
    };

    // The dense GPU build must decline this size (2·N²·8 B ≈ 107 GB ≫ budget).
    assert!(
        laplace_matrices_gpu(&elements).is_none(),
        "dense build unexpectedly accepted {n} elements"
    );

    let Some(res) = solve_local_gpu(&elements, &charges, &params, &cfg) else {
        eprintln!("no CUDA device — skipping scale demo");
        return;
    };
    let (_, stats) = res.expect("matrix-free solve on a mesh too large for the dense path");
    eprintln!(
        "matrix-free solved {n} elements (dense build OOMs): {} iters, residual {:.2e}",
        stats.iterations, stats.residual
    );
    assert!(stats.converged, "matrix-free large solve did not converge");
}
