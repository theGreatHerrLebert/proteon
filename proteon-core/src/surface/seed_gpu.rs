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

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
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
    /// Brute scatter variant of `seed`: writes into a full-grid buffer at each
    /// node's grid index. The fallback when the spatial-hash grid can't be built
    /// (huge bounding box); also the parity reference for `seed_hashed`.
    seed_scatter: CudaFunction,
    /// Spatial-hash scatter seed — the production path. Same result as
    /// `seed_scatter` but prunes the O(atoms²) exposure with a uniform cell grid.
    seed_hashed: CudaFunction,
    /// Fills a flat device buffer with NaN ("no feature yet").
    fill_nan: CudaFunction,
    jfa: CudaFunction,
    /// Turns the flooded feature grid into the signed distance field on-device,
    /// so the host downloads one f64/node instead of the 3-feature grid.
    finalize: CudaFunction,
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
        let seed_hashed = seed_mod.load_function("seed_hashed_scatter")?;
        let fill_nan = seed_mod.load_function("fill_nan")?;
        let jfa_mod = ctx.load_module(compile_ptx_with_opts(JFA_KERNEL_SRC, opts)?)?;
        let jfa = jfa_mod.load_function("jfa_pass")?;
        let finalize = jfa_mod.load_function("finalize_field")?;
        Ok(Self {
            ctx,
            seed,
            seed_scatter,
            seed_hashed,
            fill_nan,
            jfa,
            finalize,
        })
    }
}

/// Launch config for the register-heavy `seed_hashed_scatter` kernel: explicit
/// 64-thread blocks. Its nested expanding-ring + exposure loops carry many live
/// f64/int locals per thread, so the default 1024-thread blocks from
/// `for_num_elems` exceed the 65536-registers-per-block budget on CC 7.5
/// (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`). Same trick the forcefield OBC/torsion
/// kernels use for the same reason.
fn hashed_seed_cfg(nb: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (nb.div_ceil(64), 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
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

/// The spatial-hash seed alone (no flood), in compact form, for the kernel-level
/// parity tests `gpu_hashed_seed_matches_brute`/`_cpu`. Runs the production
/// `seed_hashed_scatter` over `positions` with identity output indices into a
/// NaN-filled compact buffer, so it returns one feature per node in input order
/// — directly comparable to `seed_boundary_gpu` and `nearest_surface_point`.
/// `None` if there's no GPU or the cell grid can't be built (the brute fallback
/// regime, which this test path doesn't exercise).
#[cfg(test)]
pub(super) fn seed_hashed_boundary(
    positions: &[Vec3],
    spheres: &[Sphere],
) -> Option<Vec<[f64; 3]>> {
    if positions.is_empty() {
        return Some(Vec::new());
    }
    let g = SurfaceGpu::try_global()?;
    let grid = CellGrid::build(spheres)?;
    launch_hashed_compact(g, &grid, positions).ok()
}

#[cfg(test)]
fn launch_hashed_compact(
    g: &SurfaceGpu,
    grid: &CellGrid,
    positions: &[Vec3],
) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
    let stream = g.ctx.default_stream();
    let nb = positions.len();
    let n3 = nb.checked_mul(3).ok_or("hashed seed buffer overflow")?;
    let nb_i32 = i32::try_from(nb)?;
    let nb_u32 = u32::try_from(nb)?;
    let n3_u32 = u32::try_from(n3)?;
    let n3_i64 = i64::try_from(n3)?;

    let nodes_flat: Vec<f64> = positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let out_idx: Vec<i32> = (0..nb_i32).collect(); // identity → compact output

    let d_nodes = stream.clone_htod(&nodes_flat)?;
    let d_idx = stream.clone_htod(&out_idx)?;
    let d_atoms = stream.clone_htod(&grid.atoms_sorted)?;
    let d_cell_start = stream.clone_htod(&grid.cell_start)?;
    let mut d_feat = stream.alloc_zeros::<f64>(n3)?;
    {
        let mut a = stream.launch_builder(&g.fill_nan);
        a.arg(&mut d_feat);
        a.arg(&n3_i64);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(n3_u32))?;
        }
    }
    {
        let mut a = stream.launch_builder(&g.seed_hashed);
        a.arg(&d_nodes);
        a.arg(&d_idx);
        a.arg(&nb_i32);
        a.arg(&d_atoms);
        a.arg(&d_cell_start);
        a.arg(&grid.kxmin);
        a.arg(&grid.kymin);
        a.arg(&grid.kzmin);
        a.arg(&grid.dimx);
        a.arg(&grid.dimy);
        a.arg(&grid.dimz);
        a.arg(&grid.cell);
        a.arg(&mut d_feat);
        unsafe {
            a.launch(hashed_seed_cfg(nb_u32))?;
        }
    }
    let flat = stream.clone_dtoh(&d_feat)?;
    stream.synchronize()?;
    Ok((0..nb)
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

/// A uniform cell grid over the inflated spheres, counting-sorted into cell
/// order for the `seed_hashed_scatter` kernel. Mirrors `volume::AtomGrid`:
/// `cell` = the largest inflated radius, key = `floor(coord / cell)`, dense cell
/// index `c = (kz·dimy + ky)·dimx + kx` in `(key − kmin)` coordinates, and
/// `cell_start[c]..cell_start[c+1]` is cell `c`'s atom range in `atoms_sorted`.
struct CellGrid {
    /// `natoms·4` — (cx, cy, cz, inflated_radius) in cell order.
    atoms_sorted: Vec<f64>,
    /// `ncells + 1` prefix sum (counting-sort offsets).
    cell_start: Vec<i32>,
    kxmin: i32,
    kymin: i32,
    kzmin: i32,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    cell: f64,
}

impl CellGrid {
    /// Hard ceiling on the dense cell count (`16 M` → 64 MB of `i32` offsets).
    const MAX_CELLS: i128 = 16 * 1024 * 1024;
    /// Density ceiling: cells per atom. A grid much emptier than this is a sparse
    /// bounding box (a fibre, a translated multi-domain complex) where the dense
    /// representation wastes memory and the brute kernel is the better choice —
    /// so bound the allocation to the atom count, not just an absolute cap.
    const MAX_CELLS_PER_ATOM: i128 = 64;
    /// Margin (in cells) kept between the key range and the `i32` bounds, so the
    /// kernel's `kmax = kmin + dim − 1`, the ±2 exposure gather, and node/
    /// projection keys (a few cells outside the atom grid) can never overflow
    /// `int`. Tiny vs `i32::MAX`; only a pathologically translated structure
    /// (keys near ±2 billion) trips it → brute fallback.
    const KEY_MARGIN: i64 = 8;

    /// Build the grid from the inflated `spheres`. Returns `None` (→ brute
    /// fallback) if empty, if the dense grid would exceed the absolute or
    /// per-atom cell caps, or if the cell keys sit too close to the `i32` bounds
    /// for the kernel's 32-bit arithmetic to be safe.
    fn build(spheres: &[Sphere]) -> Option<CellGrid> {
        if spheres.is_empty() {
            return None;
        }
        let cell = spheres.iter().map(|s| s.radius).fold(1.0_f64, f64::max);
        let key = |c: f64| (c / cell).floor() as i64;
        let mut kmin = (i64::MAX, i64::MAX, i64::MAX);
        let mut kmax = (i64::MIN, i64::MIN, i64::MIN);
        for s in spheres {
            let (kx, ky, kz) = (key(s.center.x), key(s.center.y), key(s.center.z));
            kmin = (kmin.0.min(kx), kmin.1.min(ky), kmin.2.min(kz));
            kmax = (kmax.0.max(kx), kmax.1.max(ky), kmax.2.max(kz));
        }
        // Both key bounds must sit a margin inside i32 so every key the kernel
        // forms (kmax, ±2 exposure, node/projection cells) stays in range.
        let lo = i64::from(i32::MIN) + Self::KEY_MARGIN;
        let hi = i64::from(i32::MAX) - Self::KEY_MARGIN;
        for (mn, mx) in [(kmin.0, kmax.0), (kmin.1, kmax.1), (kmin.2, kmax.2)] {
            if mn < lo || mx > hi {
                return None;
            }
        }
        let (dx, dy, dz) = (
            kmax.0 - kmin.0 + 1,
            kmax.1 - kmin.1 + 1,
            kmax.2 - kmin.2 + 1,
        );
        let ncells = (dx as i128) * (dy as i128) * (dz as i128);
        // Absolute ceiling AND density ceiling (tie the allocation to atom count
        // so a sparse-but-huge bounding box can't exhaust host memory before the
        // brute fallback).
        let density_cap = (spheres.len() as i128).saturating_mul(Self::MAX_CELLS_PER_ATOM);
        if ncells <= 0 || ncells > Self::MAX_CELLS || ncells > density_cap {
            return None;
        }
        let ncells = ncells as usize;
        // Kernel params + the +1 prefix entry must fit i32 (guaranteed by the
        // margin check above for the keys; dims/len are bounded by the caps).
        let kxmin = i32::try_from(kmin.0).ok()?;
        let kymin = i32::try_from(kmin.1).ok()?;
        let kzmin = i32::try_from(kmin.2).ok()?;
        let dimx = i32::try_from(dx).ok()?;
        let dimy = i32::try_from(dy).ok()?;
        let dimz = i32::try_from(dz).ok()?;
        i32::try_from(spheres.len()).ok()?;

        let cidx = |s: &Sphere| -> usize {
            let kx = key(s.center.x) - kmin.0;
            let ky = key(s.center.y) - kmin.1;
            let kz = key(s.center.z) - kmin.2;
            ((kz * dy + ky) * dx + kx) as usize
        };
        // Counting sort: count → prefix sum (cell_start) → scatter atoms.
        let cells: Vec<usize> = spheres.iter().map(cidx).collect();
        let mut cell_start = vec![0i32; ncells + 1];
        for &c in &cells {
            cell_start[c + 1] += 1;
        }
        for i in 1..=ncells {
            cell_start[i] += cell_start[i - 1];
        }
        let mut cursor = cell_start.clone();
        let mut atoms_sorted = vec![0.0f64; spheres.len() * 4];
        for (i, s) in spheres.iter().enumerate() {
            let c = cells[i];
            let pos = cursor[c] as usize;
            cursor[c] += 1;
            atoms_sorted[4 * pos] = s.center.x;
            atoms_sorted[4 * pos + 1] = s.center.y;
            atoms_sorted[4 * pos + 2] = s.center.z;
            atoms_sorted[4 * pos + 3] = s.radius;
        }
        Some(CellGrid {
            atoms_sorted,
            cell_start,
            kxmin,
            kymin,
            kzmin,
            dimx,
            dimy,
            dimz,
            cell,
        })
    }
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
///
/// Retained as the seed+flood entry for `gpu_fused_seed_flood_matches_unfused`;
/// the production path is `field_gpu` (which also finalizes on-device).
#[cfg(test)]
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

/// Seed + jump-flood the field into a device buffer, returning the flooded
/// feature grid (`n*3`, NaN where unreached) **without downloading it**. Shared
/// by `launch_seed_and_flood` (which downloads the features) and `launch_field`
/// (which finalizes to the signed distance on-device and downloads that).
#[allow(clippy::too_many_arguments)]
fn flood_into_device(
    g: &SurfaceGpu,
    stream: &Arc<CudaStream>,
    boundary_pos: &[Vec3],
    boundary_idx: &[usize],
    spheres: &[Sphere],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
) -> Result<CudaSlice<f64>, Box<dyn std::error::Error>> {
    debug_assert_eq!(boundary_pos.len(), boundary_idx.len());
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;

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
        let out_idx: Vec<i32> = boundary_idx
            .iter()
            .map(|&i| i32::try_from(i))
            .collect::<Result<_, _>>()?;
        let nb_i32 = i32::try_from(nb)?;
        let nb_u32 = u32::try_from(nb)?;
        let d_nodes = stream.clone_htod(&nodes_flat)?;
        let d_idx = stream.clone_htod(&out_idx)?;

        // Prefer the spatial-hash seed (prunes the O(atoms²) exposure with a
        // uniform cell grid built on the host). When the dense grid would be too
        // large to be worth it (huge bounding box — usually few, sparse atoms),
        // fall back to the brute scatter, which is fine in that regime. Both
        // write the same result into `src`.
        if let Some(grid) = CellGrid::build(spheres) {
            let d_atoms = stream.clone_htod(&grid.atoms_sorted)?;
            let d_cell_start = stream.clone_htod(&grid.cell_start)?;
            let mut a = stream.launch_builder(&g.seed_hashed);
            a.arg(&d_nodes);
            a.arg(&d_idx);
            a.arg(&nb_i32);
            a.arg(&d_atoms);
            a.arg(&d_cell_start);
            a.arg(&grid.kxmin);
            a.arg(&grid.kymin);
            a.arg(&grid.kzmin);
            a.arg(&grid.dimx);
            a.arg(&grid.dimy);
            a.arg(&grid.dimz);
            a.arg(&grid.cell);
            a.arg(&mut src);
            unsafe {
                a.launch(hashed_seed_cfg(nb_u32))?;
            }
        } else {
            // Huge bounding box (usually sparse, low atom count) — brute is fine.
            let atoms_flat: Vec<f64> = spheres
                .iter()
                .flat_map(|s| [s.center.x, s.center.y, s.center.z, s.radius])
                .collect();
            let m_i32 = i32::try_from(spheres.len())?;
            let d_atoms = stream.clone_htod(&atoms_flat)?;
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

    // `src` now holds the flooded field (the last swap leaves the result there).
    Ok(src)
}

#[cfg(test)]
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
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let stream = g.ctx.default_stream();
    let src = flood_into_device(
        g,
        &stream,
        boundary_pos,
        boundary_idx,
        spheres,
        dims,
        reach,
        origin,
        spacing,
    )?;
    let out = stream.clone_dtoh(&src)?;
    stream.synchronize()?;
    Ok((0..n)
        .map(|i| [out[3 * i], out[3 * i + 1], out[3 * i + 2]])
        .collect())
}

/// Full GPU field: seed + jump-flood + **finalize** to the signed distance,
/// entirely on-device. Returns `f` (one f64 per node) directly, so the host
/// downloads `n` f64 instead of the `3n` feature grid. `inside[node]` is the
/// host-computed occupancy (1 = node inside any inflated atom) that already
/// drives boundary detection; it's uploaded so the finalize matches the CPU
/// `distance_field` finalize exactly. `None` on any GPU failure → CPU fallback.
#[allow(clippy::too_many_arguments)]
pub(super) fn field_gpu(
    boundary_pos: &[Vec3],
    boundary_idx: &[usize],
    spheres: &[Sphere],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
    inside: &[u8],
    probe: f64,
) -> Option<Vec<f64>> {
    if dims.iter().product::<usize>() == 0 {
        return Some(Vec::new());
    }
    let g = SurfaceGpu::try_global()?;
    launch_field(
        g,
        boundary_pos,
        boundary_idx,
        spheres,
        dims,
        reach,
        origin,
        spacing,
        inside,
        probe,
    )
    .ok()
}

#[allow(clippy::too_many_arguments)]
fn launch_field(
    g: &SurfaceGpu,
    boundary_pos: &[Vec3],
    boundary_idx: &[usize],
    spheres: &[Sphere],
    dims: [usize; 3],
    reach: usize,
    origin: Vec3,
    spacing: f64,
    inside: &[u8],
    probe: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    if inside.len() != n {
        return Err("inside length != node count".into());
    }
    let stream = g.ctx.default_stream();
    let src = flood_into_device(
        g,
        &stream,
        boundary_pos,
        boundary_idx,
        spheres,
        dims,
        reach,
        origin,
        spacing,
    )?;

    // Finalize on-device: f = (inside ? dist : -dist) - probe, from `src`.
    let n_u32 = u32::try_from(n)?;
    let nx_i = i32::try_from(nx)?;
    let ny_i = i32::try_from(ny)?;
    let nz_i = i32::try_from(nz)?;
    let (ox, oy, oz) = (origin.x, origin.y, origin.z);

    let d_inside = stream.clone_htod(inside)?;
    let mut f = stream.alloc_zeros::<f64>(n)?;
    {
        let mut a = stream.launch_builder(&g.finalize);
        a.arg(&src);
        a.arg(&d_inside);
        a.arg(&nx_i);
        a.arg(&ny_i);
        a.arg(&nz_i);
        a.arg(&ox);
        a.arg(&oy);
        a.arg(&oz);
        a.arg(&spacing);
        a.arg(&probe);
        a.arg(&mut f);
        unsafe {
            a.launch(LaunchConfig::for_num_elems(n_u32))?;
        }
    }
    let out = stream.clone_dtoh(&f)?;
    stream.synchronize()?;
    Ok(out)
}
