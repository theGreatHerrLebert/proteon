//! GPU seed for the SDF SES path: nearest exposed surface point per boundary
//! node, computed on the GPU and reused by `volume::distance_field`.
//!
//! This is the production wiring of the GPU-K1 spike (`seed_kernel.cu`,
//! benchmarked by [`super::volume::seed_bench`]): one brute-force kernel launch
//! over the *boundary* nodes (compacted on the CPU), mirroring
//! `AtomGrid::nearest_surface_point` exactly. The CUDA context + compiled kernel
//! are cached in a `OnceLock` (compiled once via NVRTC), and any GPU failure
//! returns `None` so the caller silently falls back to the CPU path — the same
//! auto-dispatch contract as the force-field / SASA / OBC GPU kernels.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::geom::{Sphere, Vec3};

const SEED_KERNEL_SRC: &str = include_str!("seed_kernel.cu");

/// Cached CUDA context + compiled `seed_brute` kernel (compiled once).
struct SeedGpu {
    ctx: Arc<CudaContext>,
    seed: CudaFunction,
}

static SEED_GPU: OnceLock<Option<SeedGpu>> = OnceLock::new();

impl SeedGpu {
    fn try_global() -> Option<&'static SeedGpu> {
        SEED_GPU.get_or_init(|| Self::init().ok()).as_ref()
    }

    fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SEED_KERNEL_SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let seed = module.load_function("seed_brute")?;
        Ok(Self { ctx, seed })
    }
}

/// Nearest exposed surface point on the union of inflated `spheres` for each
/// position in `positions` (the SES boundary nodes), computed on the GPU.
///
/// Returns one `[x, y, z]` per input position (NaN where no exposed point), in
/// input order — identical to mapping `AtomGrid::nearest_surface_point` over the
/// same positions. Returns `None` if there is no usable GPU or any launch step
/// fails, so the caller falls back to the CPU seed.
pub(super) fn seed_boundary_gpu(positions: &[Vec3], spheres: &[Sphere]) -> Option<Vec<[f64; 3]>> {
    if positions.is_empty() {
        return Some(Vec::new());
    }
    let g = SeedGpu::try_global()?;
    launch(g, positions, spheres).ok()
}

fn launch(
    g: &SeedGpu,
    positions: &[Vec3],
    spheres: &[Sphere],
) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    let stream = g.ctx.default_stream();
    let n = positions.len();

    let nodes_flat: Vec<f64> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let atoms_flat: Vec<f64> = spheres
        .iter()
        .flat_map(|s| [s.center.x, s.center.y, s.center.z, s.radius])
        .collect();
    let n_i32 = n as i32;
    let m_i32 = spheres.len() as i32;

    let d_nodes = stream.clone_htod(&nodes_flat)?;
    let d_atoms = stream.clone_htod(&atoms_flat)?;
    let mut d_feat = stream.alloc_zeros::<f64>(n * 3)?;
    {
        let mut a = stream.launch_builder(&g.seed);
        a.arg(&d_nodes);
        a.arg(&d_atoms);
        a.arg(&n_i32);
        a.arg(&m_i32);
        a.arg(&mut d_feat);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    let flat = stream.clone_dtoh(&d_feat)?;
    stream.synchronize()?;

    Ok((0..n)
        .map(|i| [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]])
        .collect())
}
