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
use crate::solve::{
    gmres, true_residual, LocalResult, NonlocalResult, SolveConfig, SolveError, SolveStats,
};
use crate::system::{
    mol_potentials, DenseOperator, JacobiPreconditioner, LinearOperator, Quadrature, TWO_PI,
};
use crate::yukawa::regular_yukawa_collocation;

/// Threads per block for the register-heavy collocation kernels (the dense build and
/// the matrix-free matvec). A full 1024-thread block over-subscribes registers and
/// trips `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`; 128 leaves headroom.
const BLOCK: u32 = 128;

/// Flatten element geometry into the flat device-upload layout shared by the dense
/// build and the matrix-free matvecs: `verts` (9N: v1,v2,v3), `normals` (3N),
/// `distorig` (N, Laplace), `cent` (3N, centroids computed exactly as the CPU path),
/// `area` (N, the regular-Yukawa `×2·area` factor).
fn flatten_geometry(elements: &[Tri]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = elements.len();
    let mut verts = Vec::with_capacity(n * 9);
    let mut normals = Vec::with_capacity(n * 3);
    let mut distorig = Vec::with_capacity(n);
    let mut cent = Vec::with_capacity(n * 3);
    let mut area = Vec::with_capacity(n);
    for e in elements {
        verts.extend_from_slice(&[
            e.v1.x, e.v1.y, e.v1.z, e.v2.x, e.v2.y, e.v2.z, e.v3.x, e.v3.y, e.v3.z,
        ]);
        normals.extend_from_slice(&[e.normal.x, e.normal.y, e.normal.z]);
        distorig.push(e.distorig);
        let c = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
        cent.extend_from_slice(&[c.x, c.y, c.z]);
        area.push(e.area);
    }
    (verts, normals, distorig, cent, area)
}

const KERNEL_SRC: &str = include_str!("laplace_kernel.cu");

/// GPU device matrices larger than this fall back to CPU (RTX-class cards are ~8 GB;
/// `V` + `K` is `2·N²·8` bytes).
const GPU_MEM_BUDGET: u128 = 7 * (1 << 30);

struct Gpu {
    ctx: std::sync::Arc<CudaContext>,
    laplace: CudaFunction,
    matvec: CudaFunction,
    yukawa_matvec: CudaFunction,
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
    let yukawa_matvec = module.load_function("yukawa_matvec")?;
    Ok(Gpu {
        ctx,
        laplace,
        matvec,
        yukawa_matvec,
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

    let (verts, normals, distorig, cent, _area) = flatten_geometry(elements);

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
// `laplace_matvec` / `yukawa_matvec` kernels — O(N) memory, O(N²) work per matvec
// (same as CuNESSie.jl). It lifts the memory ceiling at the cost of recomputing the
// kernel every GMRES step; per-solve it is slower than the cached dense matvec, but it
// scales to meshes the dense path cannot hold.

/// Element geometry resident on the GPU for repeated matrix-free matvecs. Uploaded
/// once (verts/normals/distorig/centroids/area, all O(N)); each matvec streams a fresh
/// `x` up, launches a collocation kernel, and reads `y` back. Backs both the Laplace
/// (`V`/`K`) and regular-Yukawa (`Vy`/`Ky`) matvecs, so one upload serves the nonlocal
/// solve's five matvecs per application.
struct GpuBemGeometry {
    g: &'static Gpu,
    stream: Arc<CudaStream>,
    n: usize,
    d_verts: CudaSlice<f64>,
    d_norm: CudaSlice<f64>,
    d_dist: CudaSlice<f64>,
    d_cent: CudaSlice<f64>,
    d_area: CudaSlice<f64>,
    /// Sticky flag: set when a matvec hits a CUDA error during the GMRES iteration
    /// (where the infallible [`LinearOperator`] trait cannot return one). The solver
    /// checks it after each GMRES run and abandons the GPU — recomputing on the CPU,
    /// which yields a correct answer — rather than panicking mid-iteration. `Cell` is
    /// sound here: the operators are driven single-threaded by `gmres`.
    failed: Cell<bool>,
}

impl GpuBemGeometry {
    /// Upload the element geometry to the device once.
    fn upload(g: &'static Gpu, elements: &[Tri]) -> Result<Self, Box<dyn std::error::Error>> {
        let n = elements.len();
        let (verts, normals, distorig, cent, area) = flatten_geometry(elements);
        let stream = g.ctx.new_stream()?;
        let d_verts = stream.clone_htod(&verts)?;
        let d_norm = stream.clone_htod(&normals)?;
        let d_dist = stream.clone_htod(&distorig)?;
        let d_cent = stream.clone_htod(&cent)?;
        let d_area = stream.clone_htod(&area)?;
        stream.synchronize()?;
        Ok(Self {
            g,
            stream,
            n,
            d_verts,
            d_norm,
            d_dist,
            d_cent,
            d_area,
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
        unsafe {
            b.launch(self.launch_cfg())?;
        }
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&d_y)?)
    }

    /// `y = A·x` where `A` is the single- (`kind = 0`) or double-layer (`kind = 1`)
    /// regular-Yukawa collocation matrix at exponent `yukawa`, computed on the fly.
    fn matvec_yukawa(
        &self,
        kind: i32,
        yukawa: f64,
        x: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let d_x = self.stream.clone_htod(x)?;
        let mut d_y = self.stream.alloc_zeros::<f64>(self.n)?;
        let nf = self.n as i32;
        let mut b = self.stream.launch_builder(&self.g.yukawa_matvec);
        b.arg(&self.d_verts);
        b.arg(&self.d_norm);
        b.arg(&self.d_area);
        b.arg(&self.d_cent);
        b.arg(&d_x);
        b.arg(&nf);
        b.arg(&kind);
        b.arg(&yukawa);
        b.arg(&mut d_y);
        unsafe {
            b.launch(self.launch_cfg())?;
        }
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&d_y)?)
    }

    /// One-thread-per-row launch config (128-thread blocks, see [`BLOCK`]).
    fn launch_cfg(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((self.n as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Infallible Laplace matvec for the [`LinearOperator`] trait. A CUDA failure
    /// mid-solve cannot propagate through the trait, so instead of panicking (which
    /// would unwind an entire batch) it sets [`Self::failed`] and returns zeros; the
    /// solver checks the flag after GMRES and falls back to the CPU. Returning zeros
    /// only corrupts an already-doomed iteration whose result is discarded.
    fn matvec_trait(&self, kind: i32, x: &[f64]) -> Vec<f64> {
        self.guard(self.matvec(kind, x))
    }

    /// Infallible regular-Yukawa matvec for the trait (see [`Self::matvec_trait`]).
    fn matvec_yukawa_trait(&self, kind: i32, yukawa: f64, x: &[f64]) -> Vec<f64> {
        self.guard(self.matvec_yukawa(kind, yukawa, x))
    }

    /// Convert a fallible matvec into the trait's infallible contract: on error, poison
    /// the geometry and hand back zeros for the (doomed, discarded) iteration.
    fn guard(&self, r: Result<Vec<f64>, Box<dyn std::error::Error>>) -> Vec<f64> {
        match r {
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
    geom: &'a GpuBemGeometry,
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
    geom: &'a GpuBemGeometry,
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
    let geom = GpuBemGeometry::upload(g, elements).ok()?;
    geom.matvec(kind, x).ok()
}

/// One matrix-free GPU regular-Yukawa matvec `y = A·x`, with `A` the single-
/// (`kind = 0`) / double-layer (`kind = 1`) collocation at exponent `yukawa` — computed
/// on the fly, never stored. `None` on no device / shape mismatch / CUDA error. Exposed
/// to gate the Yukawa kernel against the CPU dense matrices, isolated from GMRES.
#[must_use]
pub fn yukawa_matvec_gpu(elements: &[Tri], kind: i32, yukawa: f64, x: &[f64]) -> Option<Vec<f64>> {
    let g = gpu()?;
    if elements.is_empty() || x.len() != elements.len() {
        return None;
    }
    let geom = GpuBemGeometry::upload(g, elements).ok()?;
    geom.matvec_yukawa(kind, yukawa, x).ok()
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
    let geom = GpuBemGeometry::upload(g, elements)
        .map_err(|e| {
            eprintln!("proteon-electrostatics: GPU geometry upload failed, using CPU ({e})");
        })
        .ok()?;
    let frac = params.eps_omega / params.eps_sigma;

    // A direct (pre/inter-stage) matvec; on CUDA error log and fall back to CPU (None).
    let direct = |kind: i32, x: &[f64]| -> Option<Vec<f64>> {
        geom.matvec(kind, x)
            .map_err(|e| {
                eprintln!("proteon-electrostatics: GPU matvec failed, using CPU ({e})");
            })
            .ok()
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
        // A CUDA error poisons the geometry and feeds GMRES zeros, which can itself
        // surface as NonFinite/NotConverged — that is a GPU failure, not a numerical
        // one, so fall back to CPU rather than reporting a spurious solve error.
        Err(_) if geom.failed.get() => return None,
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
        Err(_) if geom.failed.get() => return None, // GPU failure, not numerical
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
        quadrature: Quadrature::Fixed,
        capped_panels: 0,
    };
    Some(Ok((LocalResult { u, q, umol, qmol }, stats)))
}

// ---- matrix-free nonlocal (Yukawa) solve -------------------------------------------

/// The nonlocal 3-block operator `A·[x1;x2;x3]` (the GPU twin of
/// [`crate::system::NonlocalOperator`], formulation spec §6), matrix-free over the GPU
/// Laplace (`V`/`K`) and regular-Yukawa (`Vy`/`Ky`) matvecs. One application is five
/// matvecs: `K·x1`, `K·x3`, `V·x2`, `Vy·x2`, and `Ky·((ε∞/εΣ)x3 − x1)`.
struct GpuNonlocalOp<'a> {
    geom: &'a GpuBemGeometry,
    yukawa: f64,
    eps_omega: f64,
    eps_sigma: f64,
    eps_inf: f64,
    /// `diag(V)` (Laplace single-layer self-term) — the middle preconditioner block.
    v_diag: Vec<f64>,
    /// `diag(Ky)` (regular-Yukawa double-layer self-term) — the first block uses
    /// `2π − diag(Ky)`.
    ky_diag: Vec<f64>,
}

impl LinearOperator for GpuNonlocalOp<'_> {
    fn dim(&self) -> usize {
        3 * self.geom.n
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.geom.n;
        let (x1, x2, x3) = (&x[0..n], &x[n..2 * n], &x[2 * n..3 * n]);
        let (eo, es, ei) = (self.eps_omega, self.eps_sigma, self.eps_inf);

        let kx1 = self.geom.matvec_trait(1, x1);
        let kx3 = self.geom.matvec_trait(1, x3);
        let vx2 = self.geom.matvec_trait(0, x2);
        let vyx2 = self.geom.matvec_yukawa_trait(0, self.yukawa, x2);
        let comb: Vec<f64> = (0..n).map(|i| (ei / es) * x3[i] - x1[i]).collect();
        let kycomb = self.geom.matvec_yukawa_trait(1, self.yukawa, &comb);

        for i in 0..n {
            y[i] = kycomb[i] - kx1[i]
                + (eo / ei - eo / es) * vyx2[i]
                + (eo / ei) * vx2[i]
                + TWO_PI * x1[i];
            y[n + i] = kx1[i] - vx2[i] + TWO_PI * x1[i];
            y[2 * n + i] = (eo / ei) * vx2[i] - kx3[i] + TWO_PI * x3[i];
        }
    }
    /// NESSie's preconditioner diagonal `[2π − diag(Ky); +diag(V); 2π]`
    /// (`bem/nonlocal.jl:210`) — the middle block is `+diag(V)`, *not* the algebraic
    /// `−diag(V)`; see the formulation spec §6. Because GMRES converges on the true
    /// residual, the preconditioner only affects speed, so this faithful choice is safe.
    fn diagonal(&self) -> Vec<f64> {
        let n = self.geom.n;
        let mut d = Vec::with_capacity(3 * n);
        d.extend((0..n).map(|i| TWO_PI - self.ky_diag[i]));
        d.extend_from_slice(&self.v_diag);
        d.extend(std::iter::repeat(TWO_PI).take(n));
        d
    }
}

/// `diag(Ky)` — the regular-Yukawa double-layer self-term at each element centroid,
/// computed on the CPU with the same `regular_yukawa_collocation` the kernel mirrors.
fn yukawa_double_diagonal(elements: &[Tri], yukawa: f64) -> Vec<f64> {
    elements
        .iter()
        .map(|e| {
            let xi = (e.v1 + e.v2 + e.v3) * (1.0 / 3.0);
            regular_yukawa_collocation(PotentialKind::Double, xi, e, yukawa)
        })
        .collect()
}

/// Matrix-free GPU nonlocal (Yukawa) solve — the coupled 3-block `(u,q,w)` solve of
/// [`crate::solve::solve_nonlocal_elements`] with the four O(N²) matrices never
/// materialized: GMRES runs over [`GpuNonlocalOp`], whose five matvecs per application
/// recompute the collocations on the GPU.
///
/// Same return contract as [`solve_local_gpu`]: `None` → CPU fallback (no device /
/// CUDA error, always logged); `Some(Err)` → a genuine numerical failure to surface.
pub fn solve_nonlocal_gpu(
    elements: &[Tri],
    charges: &[Charge],
    params: &Params,
    cfg: &SolveConfig,
) -> Option<Result<(NonlocalResult, SolveStats), SolveError>> {
    let g = gpu()?;
    let n = elements.len();
    if n == 0 {
        return None; // let the CPU path report SolveError::Empty
    }
    let geom = GpuBemGeometry::upload(g, elements)
        .map_err(|e| {
            eprintln!("proteon-electrostatics: GPU geometry upload failed, using CPU ({e})");
        })
        .ok()?;
    let (eo, es, ei) = (params.eps_omega, params.eps_sigma, params.eps_inf);
    let yuk = params.yukawa();

    let lap = |kind: i32, x: &[f64]| -> Option<Vec<f64>> {
        geom.matvec(kind, x)
            .map_err(|e| eprintln!("proteon-electrostatics: GPU matvec failed, using CPU ({e})"))
            .ok()
    };
    let yuka = |kind: i32, x: &[f64]| -> Option<Vec<f64>> {
        geom.matvec_yukawa(kind, yuk, x)
            .map_err(|e| eprintln!("proteon-electrostatics: GPU matvec failed, using CPU ({e})"))
            .ok()
    };

    let (umol, qmol) = mol_potentials(elements, charges, eo);

    // RHS first block (b = [b1; 0; 0]):
    // b1 = K·umol + (1 − εΩ/εΣ)·Ky·umol − 2π·umol − (εΩ/ε∞)·V·qmol + (εΩ/εΣ − εΩ/ε∞)·Vy·qmol.
    let k_um = lap(1, &umol)?;
    let ky_um = yuka(1, &umol)?;
    let v_qm = lap(0, &qmol)?;
    let vy_qm = yuka(0, &qmol)?;
    let mut b = vec![0.0; 3 * n];
    for i in 0..n {
        b[i] = k_um[i] + (1.0 - eo / es) * ky_um[i] - TWO_PI * umol[i] - (eo / ei) * v_qm[i]
            + (eo / es - eo / ei) * vy_qm[i];
    }

    let op = GpuNonlocalOp {
        geom: &geom,
        yukawa: yuk,
        eps_omega: eo,
        eps_sigma: es,
        eps_inf: ei,
        v_diag: single_layer_diagonal(elements),
        ky_diag: yukawa_double_diagonal(elements, yuk),
    };
    let pre = JacobiPreconditioner::from_operator(&op);
    let sol = match gmres(&op, &b, &pre, cfg) {
        Ok(s) => s,
        Err(_) if geom.failed.get() => return None, // GPU failure, not numerical
        Err(e) => return Some(Err(e)),
    };
    if geom.failed.get() {
        return None; // a matvec failed inside GMRES — CPU fallback (already logged)
    }
    let res = true_residual(&op, &sol.x, &b);
    if geom.failed.get() {
        return None;
    }

    if !sol.x.iter().all(|x| x.is_finite()) {
        return Some(Err(SolveError::NonFinite));
    }
    let (u, q, w) = (
        sol.x[0..n].to_vec(),
        sol.x[n..2 * n].to_vec(),
        sol.x[2 * n..3 * n].to_vec(),
    );
    let stats = SolveStats {
        iterations: sol.iterations,
        residual: res,
        per_block_residual: vec![res],
        converged: res <= cfg.tol,
        // The GPU matrix-free Yukawa matvec is fixed 7-point (adaptive is CPU-only); the
        // reported mode makes that explicit (review [R6]).
        quadrature: Quadrature::Fixed,
        capped_panels: 0,
    };
    Some(Ok((
        NonlocalResult {
            u,
            q,
            w,
            umol,
            qmol,
        },
        stats,
    )))
}
