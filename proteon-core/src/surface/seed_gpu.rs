//! GPU seed for the SDF SES path: nearest exposed surface point per boundary
//! node, computed on the GPU and reused by `volume::distance_field`.
//!
//! This is the production wiring of the GPU-K1 spike (`seed_kernel.cu`,
//! benchmarked by [`super::volume::seed_bench`]): one brute-force kernel launch
//! over the *boundary* nodes (compacted on the CPU), reproducing
//! `AtomGrid::nearest_surface_point`'s nearest distance (ties aside; see
//! [`seed_boundary_gpu`]). The CUDA context + compiled kernel
//! are cached in a `OnceLock` (compiled once via NVRTC), and any GPU failure
//! returns `None` so the caller silently falls back to the CPU path — the same
//! auto-dispatch contract as the force-field / SASA / OBC GPU kernels.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::geom::{Sphere, Vec3};

const SEED_KERNEL_SRC: &str = include_str!("seed_kernel.cu");
const JFA_KERNEL_SRC: &str = include_str!("jfa_kernel.cu");

/// Cached CUDA context + compiled SDF-field kernels (`seed_brute`, `jfa_pass`),
/// compiled once via NVRTC and shared across all callers.
struct SurfaceGpu {
    ctx: Arc<CudaContext>,
    /// Compacted seed kernel — loaded always (the kernel ships in the same PTX),
    /// but only read by the kernel-level parity test via `seed_boundary_gpu`.
    #[cfg_attr(not(test), allow(dead_code))]
    seed: CudaFunction,
    /// Scatter variant of `seed`: writes into a full-grid buffer at each node's
    /// grid index (the fused seed->flood path; see [`seed_and_flood_gpu`]).
    seed_scatter: CudaFunction,
    /// Fills a flat device buffer with NaN ("no feature yet").
    fill_nan: CudaFunction,
    jfa: CudaFunction,
}

static SURFACE_GPU: OnceLock<Option<SurfaceGpu>> = OnceLock::new();

impl SurfaceGpu {
    fn try_global() -> Option<&'static SurfaceGpu> {
        SURFACE_GPU.get_or_init(|| Self::init().ok()).as_ref()
    }

    fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let seed_mod = ctx.load_module(compile_ptx_with_opts(SEED_KERNEL_SRC, opts.clone())?)?;
        let seed = seed_mod.load_function("seed_brute")?;
        let seed_scatter = seed_mod.load_function("seed_scatter")?;
        let fill_nan = seed_mod.load_function("fill_nan")?;
        let jfa = ctx
            .load_module(compile_ptx_with_opts(JFA_KERNEL_SRC, opts)?)?
            .load_function("jfa_pass")?;
        Ok(Self {
            ctx,
            seed,
            seed_scatter,
            fill_nan,
            jfa,
        })
    }
}

/// Nearest exposed surface point on the union of inflated `spheres` for each
/// position in `positions` (the SES boundary nodes), computed on the GPU.
///
/// Returns one `[x, y, z]` per input position (NaN where no exposed point), in
/// input order. Equivalent to mapping `AtomGrid::nearest_surface_point` over the
/// same positions: same exposed/none status and same nearest *distance* (the
/// quantity the distance field uses), with the same degenerate-direction guard
/// (`|dir| < 1e-6`); when two exposed projections are exactly equidistant the
/// chosen point may differ (CPU hash order vs GPU array order) but the distance
/// is identical. Returns `None` if there is no usable GPU or any launch step
/// fails, so the caller falls back to the CPU seed.
///
/// Retained as the kernel-level entry point for `gpu_seed_matches_cpu_seed_on_
/// boundary_nodes`; the production path uses the fused [`seed_and_flood_gpu`].
#[cfg(test)]
pub(super) fn seed_boundary_gpu(positions: &[Vec3], spheres: &[Sphere]) -> Option<Vec<[f64; 3]>> {
    if positions.is_empty() {
        return Some(Vec::new());
    }
    let g = SurfaceGpu::try_global()?;
    launch(g, positions, spheres).ok()
}

#[cfg(test)]
fn launch(
    g: &SurfaceGpu,
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
    // Checked conversions: an out-of-range size returns Err → CPU fallback,
    // never a silently-wrapped launch/param.
    let n_i32 = i32::try_from(n)?;
    let m_i32 = i32::try_from(spheres.len())?;
    let n_u32 = u32::try_from(n)?;
    let n3 = n.checked_mul(3).ok_or("seed buffer size overflow")?;

    let d_nodes = stream.clone_htod(&nodes_flat)?;
    let d_atoms = stream.clone_htod(&atoms_flat)?;
    let mut d_feat = stream.alloc_zeros::<f64>(n3)?;
    {
        let mut a = stream.launch_builder(&g.seed);
        a.arg(&d_nodes);
        a.arg(&d_atoms);
        a.arg(&n_i32);
        a.arg(&m_i32);
        a.arg(&mut d_feat);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(n_u32))?;
        }
    }
    let flat = stream.clone_dtoh(&d_feat)?;
    stream.synchronize()?;

    Ok((0..n)
        .map(|i| [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]])
        .collect())
}

/// GPU jump-flooding distance transform — the production port of
/// `volume::jump_flood`. Takes the seeded feature grid (`feat`, length
/// `dims[0]·dims[1]·dims[2]`, NaN where unseeded) and floods it with the same
/// JFA+1 halving schedule (`next_pow2(reach) … 2, 1, 1`) and the same
/// 27-neighbour nearest-by-squared-distance rule (strict `<`, same scan order),
/// ping-ponging two device buffers. Node positions are `origin + (i,j,k)·spacing`.
/// Returns the flooded grid, or `None` if there is no usable GPU.
///
/// Retained as the kernel-level entry point for `gpu_jump_flood_matches_cpu`;
/// the production path uses the fused [`seed_and_flood_gpu`].
#[cfg(test)]
pub(super) fn jump_flood_gpu(
    feat: &[[f64; 3]],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
) -> Option<Vec<[f64; 3]>> {
    if dims.iter().product::<usize>() == 0 {
        return Some(Vec::new());
    }
    let g = SurfaceGpu::try_global()?;
    launch_jfa(g, feat, dims, reach, origin, spacing).ok()
}

#[cfg(test)]
fn launch_jfa(
    g: &SurfaceGpu,
    feat: &[[f64; 3]],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let stream = g.ctx.default_stream();

    // Checked conversions: an out-of-range grid returns Err → CPU fallback,
    // never a silently-wrapped launch size or kernel param.
    let n_u32 = u32::try_from(n)?;
    let n3 = n.checked_mul(3).ok_or("jfa buffer size overflow")?;
    let nx_i = i32::try_from(nx)?;
    let ny_i = i32::try_from(ny)?;
    let nz_i = i32::try_from(nz)?;

    let flat: Vec<f64> = feat.iter().flat_map(|f| [f[0], f[1], f[2]]).collect();
    let mut src = stream.clone_htod(&flat)?;
    let mut dst = stream.alloc_zeros::<f64>(n3)?;

    // Identical schedule to the CPU jump_flood: next_pow2(reach) … 2, 1, then a
    // final unit pass (JFA+1).
    let mut schedule: Vec<usize> = Vec::new();
    let mut step = reach.max(1).next_power_of_two();
    while step >= 1 {
        schedule.push(step);
        step /= 2;
    }
    schedule.push(1);

    let (ox, oy, oz) = (origin.x, origin.y, origin.z);
    for step in schedule {
        let step_i = i32::try_from(step)?;
        {
            let mut a = stream.launch_builder(&g.jfa);
            a.arg(&src);
            a.arg(&mut dst);
            a.arg(&nx_i);
            a.arg(&ny_i);
            a.arg(&nz_i);
            a.arg(&step_i);
            a.arg(&ox);
            a.arg(&oy);
            a.arg(&oz);
            a.arg(&spacing);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(n_u32))?;
            }
        }
        // src ← freshly-flooded dst for the next (smaller-step) pass. Stream
        // ordering serialises the passes; we synchronise once at the end.
        std::mem::swap(&mut src, &mut dst);
    }
    let out = stream.clone_dtoh(&src)?;
    stream.synchronize()?;
    Ok((0..n)
        .map(|i| [out[3 * i], out[3 * i + 1], out[3 * i + 2]])
        .collect())
}

/// Fused seed + jump-flood, entirely on-device: the production SDF-field path.
///
/// Equivalent to `seed_boundary_gpu` (scattered into a full grid) followed by
/// `jump_flood_gpu`, but the seeded feature buffer never leaves the GPU between
/// the two stages — dropping the boundary-feature download, the host scatter,
/// and the full-grid re-upload that the two separate calls incur. This is where
/// the seed's speedup and the on-device JFA actually compound.
///
/// `boundary_idx[t]` is the flat grid index (`i + nx*(j + ny*k)`) of boundary
/// node `t`, whose position is `boundary_pos[t]`. Returns the flooded full grid
/// (`dims` product entries, NaN where JFA never reached), or `None` on any GPU
/// failure so the caller falls back to the CPU seed + CPU jump-flood.
#[allow(clippy::too_many_arguments)]
pub(super) fn seed_and_flood_gpu(
    boundary_pos: &[Vec3],
    boundary_idx: &[usize],
    spheres: &[Sphere],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
) -> Option<Vec<[f64; 3]>> {
    if dims.iter().product::<usize>() == 0 {
        return Some(Vec::new());
    }
    let g = SurfaceGpu::try_global()?;
    launch_seed_and_flood(
        g,
        boundary_pos,
        boundary_idx,
        spheres,
        dims,
        reach,
        origin,
        spacing,
    )
    .ok()
}

#[allow(clippy::too_many_arguments)]
fn launch_seed_and_flood(
    g: &SurfaceGpu,
    boundary_pos: &[Vec3],
    boundary_idx: &[usize],
    spheres: &[Sphere],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    debug_assert_eq!(boundary_pos.len(), boundary_idx.len());
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let stream = g.ctx.default_stream();

    // Checked conversions: an out-of-range grid/index returns Err → CPU
    // fallback, never a silently-wrapped launch size or scatter index.
    let n3 = n.checked_mul(3).ok_or("seed+flood buffer size overflow")?;
    let n_u32 = u32::try_from(n)?;
    let n3_u32 = u32::try_from(n3)?;
    let n3_i64 = i64::try_from(n3)?;
    let nx_i = i32::try_from(nx)?;
    let ny_i = i32::try_from(ny)?;
    let nz_i = i32::try_from(nz)?;
    // Grid indices must fit i32 for the scatter kernel; the largest is n-1.
    i32::try_from(n.saturating_sub(1))?;

    // Full-grid feature buffer, NaN-initialised so non-boundary nodes read as
    // "no feature" for the jump-flood.
    let mut src = stream.alloc_zeros::<f64>(n3)?;
    {
        let mut a = stream.launch_builder(&g.fill_nan);
        a.arg(&mut src);
        a.arg(&n3_i64);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(n3_u32))?;
        }
    }

    // Seed the boundary nodes directly into the full grid (no compaction round
    // trip). Empty boundary → all-NaN field, valid (no surface in range).
    let nb = boundary_pos.len();
    if nb > 0 {
        let nodes_flat: Vec<f64> = boundary_pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let atoms_flat: Vec<f64> = spheres
            .iter()
            .flat_map(|s| [s.center.x, s.center.y, s.center.z, s.radius])
            .collect();
        let out_idx: Vec<i32> = boundary_idx
            .iter()
            .map(|&i| i32::try_from(i))
            .collect::<Result<_, _>>()?;
        let nb_i32 = i32::try_from(nb)?;
        let m_i32 = i32::try_from(spheres.len())?;
        let nb_u32 = u32::try_from(nb)?;

        let d_nodes = stream.clone_htod(&nodes_flat)?;
        let d_atoms = stream.clone_htod(&atoms_flat)?;
        let d_idx = stream.clone_htod(&out_idx)?;
        {
            let mut a = stream.launch_builder(&g.seed_scatter);
            a.arg(&d_nodes);
            a.arg(&d_atoms);
            a.arg(&d_idx);
            a.arg(&nb_i32);
            a.arg(&m_i32);
            a.arg(&mut src);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(nb_u32))?;
            }
        }
    }

    // Jump-flood on the seeded full grid — identical schedule/rule to
    // `launch_jfa`, ping-ponging two device buffers.
    let mut dst = stream.alloc_zeros::<f64>(n3)?;
    let mut schedule: Vec<usize> = Vec::new();
    let mut step = reach.max(1).next_power_of_two();
    while step >= 1 {
        schedule.push(step);
        step /= 2;
    }
    schedule.push(1);

    let (ox, oy, oz) = (origin.x, origin.y, origin.z);
    for step in schedule {
        let step_i = i32::try_from(step)?;
        {
            let mut a = stream.launch_builder(&g.jfa);
            a.arg(&src);
            a.arg(&mut dst);
            a.arg(&nx_i);
            a.arg(&ny_i);
            a.arg(&nz_i);
            a.arg(&step_i);
            a.arg(&ox);
            a.arg(&oy);
            a.arg(&oz);
            a.arg(&spacing);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(n_u32))?;
            }
        }
        std::mem::swap(&mut src, &mut dst);
    }

    let out = stream.clone_dtoh(&src)?;
    stream.synchronize()?;
    Ok((0..n)
        .map(|i| [out[3 * i], out[3 * i + 1], out[3 * i + 2]])
        .collect())
}
