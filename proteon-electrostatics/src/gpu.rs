//! GPU build of the Laplace collocation matrices (`V`, `K`) — feature `cuda`.
//!
//! The dense BEM's cost is the O(N²) collocation assembly; this offloads it to a
//! CUDA/NVRTC kernel (`laplace_kernel.cu`) that mirrors `laplace.rs`. The GMRES then
//! runs on CPU over the returned dense matrices. CuNESSie's lever — a large constant
//! speedup, not a change in asymptotics. Silent CPU fallback when there is no device,
//! the matrices would exceed GPU memory, or any CUDA call fails.

use std::cell::Cell;
use std::sync::{Arc, OnceLock};

#[allow(clippy::wildcard_imports)] // cudarc::driver is a prelude-style module
use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::laplace::laplace_collocation;
use crate::model::{Charge, Params, PotentialKind, Tri};
use crate::solve::{gmres, true_residual, LocalResult, SolveConfig, SolveError, SolveStats};
use crate::system::{
    mol_potentials, DenseOperator, JacobiPreconditioner, LinearOperator, TWO_PI,
};

/// Threads per block for the register-heavy collocation kernels (the dense build and
/// the matrix-free matvec). A full 1024-thread block over-subscribes registers and
/// trips `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`; 128 leaves headroom.
const BLOCK: u32 = 128;

/// Flatten element geometry into the flat device-upload layout shared by the dense
/// build and the matrix-free matvec: `verts` (9N: v1,v2,v3), `normals` (3N),
/// `distorig` (N), `cent` (3N, centroids computed exactly as the CPU path).
fn flatten_geometry(elements: &[Tri]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = elements.len();
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
    (verts, normals, distorig, cent)
}

const KERNEL_SRC: &str = include_str!("laplace_kernel.cu");

/// GPU device matrices larger than this fall back to CPU (RTX-class cards are ~8 GB;
/// `V` + `K` is `2·N²·8` bytes).
const GPU_MEM_BUDGET: u128 = 7 * (1 << 30);

struct Gpu {
    ctx: std::sync::Arc<CudaContext>,
    laplace: CudaFunction,
    matvec: CudaFunction,
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
    let matvec = module.load_function("laplace_matvec")?;
    Ok(Gpu {
        ctx,
        laplace,
        matvec,
    })
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

    let (verts, normals, distorig, cent) = flatten_geometry(elements);

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
        let total = (n * n) as u32;
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

// ---- matrix-free GPU matvec --------------------------------------------------------
//
// The dense build above stores the full N² collocation matrices, so it OOMs the GPU
// past ~30k elements (2·N²·8 bytes > budget). The matrix-free path uploads only the
// O(N) element geometry once and recomputes each collocation on the fly inside the
// `laplace_matvec` kernel — O(N) memory, O(N²) work per matvec (same as CuNESSie.jl).
// It lifts the memory ceiling at the cost of recomputing the kernel every GMRES step;
// per-solve it is slower than the cached dense matvec, but it scales to meshes the
// dense path cannot hold.

/// Element geometry resident on the GPU for repeated matrix-free matvecs. Uploaded
/// once (verts/normals/distorig/centroids, all O(N)); each [`Self::matvec`] streams a
/// fresh `x` up, launches `laplace_matvec`, and reads `y` back.
struct GpuLaplaceGeometry {
    g: &'static Gpu,
    stream: Arc<CudaStream>,
    n: usize,
    d_verts: CudaSlice<f64>,
    d_norm: CudaSlice<f64>,
    d_dist: CudaSlice<f64>,
    d_cent: CudaSlice<f64>,
    /// Sticky flag: set when a matvec hits a CUDA error during the GMRES iteration
    /// (where the infallible [`LinearOperator`] trait cannot return one). The solver
    /// checks it after each GMRES run and abandons the GPU — recomputing on the CPU,
    /// which yields a correct answer — rather than panicking mid-iteration. `Cell` is
    /// sound here: the operators are driven single-threaded by `gmres`.
    failed: Cell<bool>,
}

impl GpuLaplaceGeometry {
    /// Upload the element geometry to the device once.
    fn upload(g: &'static Gpu, elements: &[Tri]) -> Result<Self, Box<dyn std::error::Error>> {
        let n = elements.len();
        let (verts, normals, distorig, cent) = flatten_geometry(elements);
        let stream = g.ctx.new_stream()?;
        let d_verts = stream.clone_htod(&verts)?;
        let d_norm = stream.clone_htod(&normals)?;
        let d_dist = stream.clone_htod(&distorig)?;
        let d_cent = stream.clone_htod(&cent)?;
        stream.synchronize()?;
        Ok(Self {
            g,
            stream,
            n,
            d_verts,
            d_norm,
            d_dist,
            d_cent,
            failed: Cell::new(false),
        })
    }

    /// `y = A·x` where `A` is the single- (`kind = 0`) or double-layer (`kind = 1`)
    /// Laplace collocation matrix, computed on the fly without storing `A`.
    fn matvec(&self, kind: i32, x: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let d_x = self.stream.clone_htod(x)?;
        let mut d_y = self.stream.alloc_zeros::<f64>(self.n)?;
        let nf = self.n as i32;
        let mut b = self.stream.launch_builder(&self.g.matvec);
        b.arg(&self.d_verts);
        b.arg(&self.d_norm);
        b.arg(&self.d_dist);
        b.arg(&self.d_cent);
        b.arg(&d_x);
        b.arg(&nf);
        b.arg(&kind);
        b.arg(&mut d_y);
        let cfg = LaunchConfig {
            grid_dim: ((self.n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            b.launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&d_y)?)
    }

    /// Infallible matvec for the [`LinearOperator`] trait. A CUDA failure mid-solve
    /// cannot propagate through the trait, so instead of panicking (which would unwind
    /// an entire batch) it sets [`Self::failed`] and returns zeros; `solve_local_gpu`
    /// checks the flag after GMRES and falls back to the CPU. Returning zeros only
    /// corrupts an already-doomed iteration whose result is discarded.
    fn matvec_trait(&self, kind: i32, x: &[f64]) -> Vec<f64> {
        match self.matvec(kind, x) {
            Ok(y) => y,
            Err(e) => {
                eprintln!(
                    "proteon-electrostatics: GPU matvec failed mid-solve, abandoning GPU ({e})"
                );
                self.failed.set(true);
                vec![0.0; self.n]
            }
        }
    }
}

/// `M = 2π(1 + εΩ/εΣ)I + (εΩ/εΣ − 1)·K`, matrix-free over the GPU `K·x` — the GPU
/// twin of [`crate::system::LocalOperator`].
struct GpuLocalOp<'a> {
    geom: &'a GpuLaplaceGeometry,
    frac: f64,
}

impl LinearOperator for GpuLocalOp<'_> {
    fn dim(&self) -> usize {
        self.geom.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let kx = self.geom.matvec_trait(1, x); // double layer
        let diag = TWO_PI * (1.0 + self.frac);
        let off = self.frac - 1.0;
        for i in 0..y.len() {
            y[i] = diag * x[i] + off * kx[i];
        }
    }
    fn diagonal(&self) -> Vec<f64> {
        vec![TWO_PI * (1.0 + self.frac); self.geom.n]
    }
}

/// `V` (single-layer) as a matrix-free GPU operator. The Jacobi preconditioner needs
/// `diag(V)`, which the matvec alone cannot supply, so the self-term collocation is
/// precomputed once on the CPU (O(N), cheap).
struct GpuSingleOp<'a> {
    geom: &'a GpuLaplaceGeometry,
    diag: Vec<f64>,
}

impl LinearOperator for GpuSingleOp<'_> {
    fn dim(&self) -> usize {
        self.geom.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let vx = self.geom.matvec_trait(0, x); // single layer
        y.copy_from_slice(&vx);
    }
    fn diagonal(&self) -> Vec<f64> {
        self.diag.clone()
    }
}

/// `diag(V)` — the single-layer self-term `V[i][i]` at each element centroid,
/// computed on the CPU with the same `laplace_collocation` the kernel mirrors.
fn single_layer_diagonal(elements: &[Tri]) -> Vec<f64> {
    elements
        .iter()
        .map(|e| {
            let xi = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
            laplace_collocation(PotentialKind::Single, xi, e)
        })
        .collect()
}

/// One matrix-free GPU matvec `y = A·x`, with `A` the single- (`kind = 0`) or
/// double-layer (`kind = 1`) Laplace collocation matrix — computed on the fly, never
/// stored. Uploads the geometry, runs one kernel launch, returns `y`. `None` on no
/// device, a shape mismatch, or a CUDA error.
///
/// Exposed primarily to gate the kernel directly against the CPU dense matrices
/// (`V·x` / `K·x`) — isolating kernel indexing/accumulation from the GMRES path.
#[must_use]
pub fn laplace_matvec_gpu(elements: &[Tri], kind: i32, x: &[f64]) -> Option<Vec<f64>> {
    let g = gpu()?;
    if elements.is_empty() || x.len() != elements.len() {
        return None;
    }
    let geom = GpuLaplaceGeometry::upload(g, elements).ok()?;
    geom.matvec(kind, x).ok()
}

/// Matrix-free GPU local (Poisson) solve — the two-stage local solve of
/// [`crate::solve::solve_local_elements`], but with the O(N²) `K`/`V` never
/// materialized: GMRES runs over [`GpuLocalOp`] / [`GpuSingleOp`], whose matvecs
/// recompute the collocation on the GPU each step.
///
/// Return contract:
/// - `None` → fall back to the CPU. No device, empty model, or *any* CUDA error
///   (upload, a direct RHS matvec, or a matvec inside GMRES via the poison flag). A
///   CPU redo of the whole solve then produces a correct answer; the GPU failure is
///   logged, never silent.
/// - `Some(Err(..))` → a genuine **numerical** failure (non-convergence / non-finite)
///   that the CPU would hit too — surface it, do not retry on CPU.
pub fn solve_local_gpu(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Option<Result<(LocalResult, SolveStats), SolveError>> {
    let g = gpu()?;
    let n = elements.len();
    if n == 0 {
        return None; // let the CPU path report SolveError::Empty
    }
    let geom = GpuLaplaceGeometry::upload(g, elements)
        .map_err(|e| {
            eprintln!("proteon-electrostatics: GPU geometry upload failed, using CPU ({e})");
        })
        .ok()?;
    let frac = params.eps_omega / params.eps_sigma;

    // A direct (pre/inter-stage) matvec; on CUDA error log and fall back to CPU (None).
    let direct = |kind: i32, x: &[f64]| -> Option<Vec<f64>> {
        geom.matvec(kind, x).map_err(|e| {
            eprintln!("proteon-electrostatics: GPU matvec failed, using CPU ({e})");
        }).ok()
    };

    let (umol, qmol) = mol_potentials(elements, charges, params.eps_omega);

    // Stage 1: b₁ = K·umol − 2π·umol − frac·(V·qmol); M·u = b₁.
    let k_umol = direct(1, &umol)?;
    let v_qmol = direct(0, &qmol)?;
    let b1: Vec<f64> = (0..n)
        .map(|i| k_umol[i] - TWO_PI * umol[i] - frac * v_qmol[i])
        .collect();

    let m_op = GpuLocalOp { geom: &geom, frac };
    let m_pre = JacobiPreconditioner::from_operator(&m_op);
    let u_sol = match gmres(&m_op, &b1, &m_pre, cfg) {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };
    if geom.failed.get() {
        return None; // a matvec failed inside GMRES — CPU fallback (already logged)
    }
    let u = u_sol.x;
    let res_u = true_residual(&m_op, &u, &b1);
    if geom.failed.get() {
        return None;
    }

    // Stage 2: b₂ = 2π·u + K·u; V·q = b₂.
    let k_u = direct(1, &u)?;
    let b2: Vec<f64> = (0..n).map(|i| TWO_PI * u[i] + k_u[i]).collect();
    let v_op = GpuSingleOp {
        geom: &geom,
        diag: single_layer_diagonal(elements),
    };
    let v_pre = JacobiPreconditioner::from_operator(&v_op);
    let q_sol = match gmres(&v_op, &b2, &v_pre, cfg) {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };
    if geom.failed.get() {
        return None;
    }
    let q = q_sol.x;
    let res_q = true_residual(&v_op, &q, &b2);
    if geom.failed.get() {
        return None;
    }

    if !u.iter().chain(&q).all(|x| x.is_finite()) {
        return Some(Err(SolveError::NonFinite));
    }

    let stats = SolveStats {
        iterations: u_sol.iterations + q_sol.iterations,
        residual: res_u.max(res_q),
        per_block_residual: vec![res_u, res_q],
        converged: res_u <= cfg.tol && res_q <= cfg.tol,
    };
    Some(Ok((LocalResult { u, q, umol, qmol }, stats)))
}
