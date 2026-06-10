//! GPU vs CPU parity for the Laplace collocation assembly (feature `cuda`).
//!
//! The CUDA kernel mirrors `laplace.rs` operation-for-operation, so the matrices must
//! agree with the rayon CPU path to libm precision (CUDA libdevice vs Rust libm for
//! asin/log/sqrt). Skips gracefully if no device is present.
#![cfg(feature = "cuda")]

use proteon_electrostatics::system::laplace_matrices_cpu;
use proteon_electrostatics::{analytic_sphere_mesh, gpu::laplace_matrices_gpu, Tri};

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
