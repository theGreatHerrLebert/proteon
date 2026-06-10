//! GPU build of the Laplace collocation matrices (`V`, `K`) — feature `cuda`.
//!
//! The dense BEM's cost is the O(N²) collocation assembly; this offloads it to a
//! CUDA/NVRTC kernel (`laplace_kernel.cu`) that mirrors `laplace.rs`. The GMRES then
//! runs on CPU over the returned dense matrices. CuNESSie's lever — a large constant
//! speedup, not a change in asymptotics. Silent CPU fallback when there is no device,
//! the matrices would exceed GPU memory, or any CUDA call fails.

use std::sync::OnceLock;

#[allow(clippy::wildcard_imports)] // cudarc::driver is a prelude-style module
use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::model::Tri;
use crate::system::DenseOperator;

const KERNEL_SRC: &str = include_str!("laplace_kernel.cu");

/// GPU device matrices larger than this fall back to CPU (RTX-class cards are ~8 GB;
/// `V` + `K` is `2·N²·8` bytes).
const GPU_MEM_BUDGET: u128 = 7 * (1 << 30);

struct Gpu {
    ctx: std::sync::Arc<CudaContext>,
    laplace: CudaFunction,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(|| match init() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("proteon-electrostatics: GPU init failed, using CPU ({e})");
            None
        }
    })
    .as_ref()
}

fn init() -> Result<Gpu, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)?;
    let module = ctx.load_module(ptx)?;
    let laplace = module.load_function("laplace_matrices")?;
    Ok(Gpu { ctx, laplace })
}

/// Build `(V, K)` on the GPU, or `None` to signal a CPU fallback (no device, the
/// matrices exceed [`GPU_MEM_BUDGET`], or a CUDA error).
pub fn laplace_matrices_gpu(elements: &[Tri]) -> Option<(DenseOperator, DenseOperator)> {
    let g = gpu()?;
    let n = elements.len();
    if n == 0 || (n as u128).pow(2) > u128::from(u32::MAX) {
        return None; // n²>u32 can't index for_num_elems; n>65535 is GPU-OOM anyway
    }
    if 2 * (n as u128).pow(2) * 8 > GPU_MEM_BUDGET {
        return None;
    }

    // Flatten the element geometry (centroids computed exactly as the CPU path).
    let mut verts = Vec::with_capacity(n * 9);
    let mut normals = Vec::with_capacity(n * 3);
    let mut distorig = Vec::with_capacity(n);
    let mut cent = Vec::with_capacity(n * 3);
    for e in elements {
        verts.extend_from_slice(&[
            e.v1.x, e.v1.y, e.v1.z, e.v2.x, e.v2.y, e.v2.z, e.v3.x, e.v3.y, e.v3.z,
        ]);
        normals.extend_from_slice(&[e.normal.x, e.normal.y, e.normal.z]);
        distorig.push(e.distorig);
        let c = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
        cent.extend_from_slice(&[c.x, c.y, c.z]);
    }

    let run = || -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let stream = g.ctx.new_stream()?;
        let d_verts = stream.clone_htod(&verts)?;
        let d_norm = stream.clone_htod(&normals)?;
        let d_dist = stream.clone_htod(&distorig)?;
        let d_cent = stream.clone_htod(&cent)?;
        let mut d_v = stream.alloc_zeros::<f64>(n * n)?;
        let mut d_k = stream.alloc_zeros::<f64>(n * n)?;
        let nf = n as i32;

        let mut b = stream.launch_builder(&g.laplace);
        b.arg(&d_verts);
        b.arg(&d_norm);
        b.arg(&d_dist);
        b.arg(&d_cent);
        b.arg(&nf);
        b.arg(&mut d_v);
        b.arg(&mut d_k);
        // The collocation kernel is register-heavy (many local double[3] + nested
        // device calls), so a 1024-thread block over-subscribes registers
        // (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). 128 threads/block leaves headroom.
        let total = (n * n) as u32;
        const BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            b.launch(cfg)?;
        }
        stream.synchronize()?;
        Ok((stream.clone_dtoh(&d_v)?, stream.clone_dtoh(&d_k)?))
    };

    let (v, k) = match run() {
        Ok(vk) => vk,
        Err(e) => {
            eprintln!("proteon-electrostatics: GPU run failed, using CPU ({e})");
            return None;
        }
    };
    Some((DenseOperator { n, data: v }, DenseOperator { n, data: k }))
}
