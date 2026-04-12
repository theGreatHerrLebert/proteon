//! CUDA-accelerated energy + forces evaluation.
//!
//! Behind the `cuda` feature flag. Provides a GPU backend for the
//! energy+forces evaluation that NbCache can dispatch to instead of the
//! CPU path. All CUDA interaction is encapsulated here; the rest of
//! ferritin never touches cudarc directly.
//!
//! Architecture:
//! - `GpuContext`: singleton (OnceLock), owns the CUDA device context +
//!   compiled kernels. Created lazily on first use; returns None if no
//!   GPU is available (silent CPU fallback).
//! - `GpuStructState`: per-structure GPU data. Owns topology buffers
//!   (immutable between LBFGS steps) + a CUDA stream + mutable
//!   coords/forces/energy arrays. One per NbCache.
//!
//! The kernels themselves live in `kernel.cu` (nonbonded) and
//! `bonded_kernels.cu` (bonds, angles, torsions), compiled at runtime
//! via nvrtc for the detected GPU architecture.

use std::sync::{Arc, OnceLock};

use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::energy::EnergyResult;
use super::neighbor_list::NeighborList;
use super::params::ForceField;
use super::topology::Topology;

const NONBONDED_KERNEL_SRC: &str = include_str!("kernel.cu");
const BONDED_KERNEL_SRC: &str = include_str!("bonded_kernels.cu");
const SASA_KERNEL_SRC: &str = include_str!("sasa_kernel.cu");

// ---------------------------------------------------------------------------
// GpuContext: global singleton (one per process)
// ---------------------------------------------------------------------------

/// Compiled CUDA kernels, shared across all structures.
struct GpuKernels {
    nonbonded: CudaFunction,
    bond: CudaFunction,
    angle: CudaFunction,
    torsion: CudaFunction,
    sasa: CudaFunction,
}

/// Global GPU context. Created once, shared across all rayon threads.
pub(crate) struct GpuContext {
    ctx: Arc<CudaContext>,
    kernels: GpuKernels,
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Get the global GPU context, initializing on first call.
    /// Returns None if no GPU is available or initialization fails.
    pub fn try_global() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(|| {
                match Self::init() {
                    Ok(ctx) => {
                        eprintln!(
                            "[ferritin-gpu] CUDA initialized: {} (CC {}.{})",
                            ctx.ctx.name().unwrap_or_default(),
                            ctx.ctx.compute_capability().map(|c| c.0).unwrap_or(0),
                            ctx.ctx.compute_capability().map(|c| c.1).unwrap_or(0),
                        );
                        Some(ctx)
                    }
                    Err(e) => {
                        eprintln!("[ferritin-gpu] No GPU available, using CPU: {}", e);
                        None
                    }
                }
            })
            .as_ref()
    }

    fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str =
            Box::leak(format!("sm_{}{}", major, minor).into_boxed_str());
        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };

        let nb_ptx = compile_ptx_with_opts(NONBONDED_KERNEL_SRC, opts.clone())?;
        let nb_mod = ctx.load_module(nb_ptx)?;
        let nonbonded = nb_mod.load_function("nonbonded_energy_forces")?;

        let bonded_ptx = compile_ptx_with_opts(BONDED_KERNEL_SRC, opts.clone())?;
        let bonded_mod = ctx.load_module(bonded_ptx)?;
        let bond = bonded_mod.load_function("bond_energy_forces")?;
        let angle = bonded_mod.load_function("angle_energy_forces")?;
        let torsion = bonded_mod.load_function("torsion_energy_forces")?;

        let sasa_ptx = compile_ptx_with_opts(SASA_KERNEL_SRC, opts)?;
        let sasa_mod = ctx.load_module(sasa_ptx)?;
        let sasa = sasa_mod.load_function("sasa_shrake_rupley")?;

        Ok(Self {
            ctx,
            kernels: GpuKernels {
                nonbonded,
                bond,
                angle,
                torsion,
                sasa,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// GpuStructState: per-structure GPU data
// ---------------------------------------------------------------------------

/// Immutable topology data on GPU (uploaded once per structure).
struct GpuTopology {
    // NBL pairs
    pair_i: CudaSlice<i32>,
    pair_j: CudaSlice<i32>,
    pair_14: CudaSlice<i32>,
    n_pairs: i32,
    // Per-atom
    atom_types: CudaSlice<i32>,
    charges: CudaSlice<f64>,
    is_h: CudaSlice<i32>,
    // Per-type LJ
    lj_r: CudaSlice<f64>,
    lj_eps: CudaSlice<f64>,
    // EEF1 per-atom
    eef1_dg_free: CudaSlice<f64>,
    eef1_volume: CudaSlice<f64>,
    eef1_sigma: CudaSlice<f64>,
    eef1_r_min: CudaSlice<f64>,
    // Bonded
    bond_i: CudaSlice<i32>,
    bond_j: CudaSlice<i32>,
    bond_k: CudaSlice<f64>,
    bond_r0: CudaSlice<f64>,
    n_bonds: i32,
    angle_i: CudaSlice<i32>,
    angle_j: CudaSlice<i32>,
    angle_k: CudaSlice<i32>,
    angle_kf: CudaSlice<f64>,
    angle_t0: CudaSlice<f64>,
    n_angles: i32,
    tor_i: CudaSlice<i32>,
    tor_j: CudaSlice<i32>,
    tor_k: CudaSlice<i32>,
    tor_l: CudaSlice<i32>,
    tor_v: CudaSlice<f64>,
    tor_f: CudaSlice<f64>,
    tor_p: CudaSlice<f64>,
    n_torsion_terms: i32,
    // Constants
    n_atoms: usize,
    self_solvation: f64,
    cutoff_sq: f64,
    cuton_sq: f64,
    eef1_cutoff_sq: f64,
    coulomb_factor: f64,
    scee_inv: f64,
    scnb_inv: f64,
    pi_sqrt_pi: f64,
}

/// Per-structure mutable GPU state: coords, forces, energy output arrays.
pub(crate) struct GpuStructState {
    stream: Arc<CudaStream>,
    topo_gpu: GpuTopology,
    // Mutable per-step
    coords: CudaSlice<f64>,
    forces: CudaSlice<f64>,
    pair_vdw: CudaSlice<f64>,
    pair_elec: CudaSlice<f64>,
    pair_solv: CudaSlice<f64>,
    bond_energies: CudaSlice<f64>,
    angle_energies: CudaSlice<f64>,
    tor_energies: CudaSlice<f64>,
}

impl GpuStructState {
    /// Upload a structure's topology to GPU and allocate output buffers.
    pub fn new<F: ForceField>(
        gpu: &GpuContext,
        topo: &Topology,
        nbl: &NeighborList,
        ff: &F,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = gpu.ctx.new_stream()?;
        let n_atoms = topo.atoms.len();
        let n_pairs = nbl.pairs.len();

        // Marshal NBL pairs
        let pi: Vec<i32> = nbl.pairs.iter().map(|p| p.i as i32).collect();
        let pj: Vec<i32> = nbl.pairs.iter().map(|p| p.j as i32).collect();
        let p14: Vec<i32> = nbl
            .pairs
            .iter()
            .map(|p| if p.is_14 { 1 } else { 0 })
            .collect();

        // Atom type index mapping
        let mut type_names: Vec<String> = Vec::new();
        let mut type_idx: Vec<i32> = Vec::with_capacity(n_atoms);
        for a in &topo.atoms {
            let pos = type_names
                .iter()
                .position(|t| t == &a.amber_type)
                .unwrap_or_else(|| {
                    type_names.push(a.amber_type.clone());
                    type_names.len() - 1
                });
            type_idx.push(pos as i32);
        }
        let n_types = type_names.len();
        let mut lj_r = vec![0.0f64; n_types];
        let mut lj_eps = vec![0.0f64; n_types];
        for (i, t) in type_names.iter().enumerate() {
            if let Some(lj) = ff.get_lj(t) {
                lj_r[i] = lj.r;
                lj_eps[i] = lj.epsilon;
            }
        }

        let charges: Vec<f64> = topo.atoms.iter().map(|a| a.charge).collect();
        let is_h: Vec<i32> = topo
            .atoms
            .iter()
            .map(|a| if a.is_hydrogen { 1 } else { 0 })
            .collect();

        let mut eef1_dg_free = vec![0.0f64; n_atoms];
        let mut eef1_volume = vec![0.0f64; n_atoms];
        let mut eef1_sigma = vec![3.5f64; n_atoms];
        let mut eef1_r_min_v = vec![1.6f64; n_atoms];
        let mut self_solv = 0.0f64;
        for (i, a) in topo.atoms.iter().enumerate() {
            if let Some(eef) = ff.get_eef1(&a.amber_type) {
                eef1_dg_free[i] = eef.dg_free;
                eef1_volume[i] = eef.volume;
                eef1_sigma[i] = eef.sigma;
                eef1_r_min_v[i] = eef.r_min;
                if !a.is_hydrogen {
                    self_solv += eef.dg_ref;
                }
            }
        }

        // Bonded marshalling
        let mut bi = Vec::new();
        let mut bj = Vec::new();
        let mut bk = Vec::new();
        let mut br0 = Vec::new();
        for bond in &topo.bonds {
            let ti = &topo.atoms[bond.i].amber_type;
            let tj = &topo.atoms[bond.j].amber_type;
            if let Some(bp) = ff.get_bond(ti, tj) {
                bi.push(bond.i as i32);
                bj.push(bond.j as i32);
                bk.push(bp.k);
                br0.push(bp.r0);
            }
        }

        let mut ai = Vec::new();
        let mut aj = Vec::new();
        let mut ak = Vec::new();
        let mut akf = Vec::new();
        let mut at0 = Vec::new();
        for angle in &topo.angles {
            let ti = &topo.atoms[angle.i].amber_type;
            let tj = &topo.atoms[angle.j].amber_type;
            let tk = &topo.atoms[angle.k].amber_type;
            if let Some(ap) = ff.get_angle(ti, tj, tk) {
                ai.push(angle.i as i32);
                aj.push(angle.j as i32);
                ak.push(angle.k as i32);
                akf.push(ap.k);
                at0.push(ap.theta0);
            }
        }

        let mut ti_arr = Vec::new();
        let mut tj_arr = Vec::new();
        let mut tk_arr = Vec::new();
        let mut tl_arr = Vec::new();
        let mut tv = Vec::new();
        let mut tf = Vec::new();
        let mut tp = Vec::new();
        for torsion in &topo.torsions {
            let t_i = &topo.atoms[torsion.i].amber_type;
            let t_j = &topo.atoms[torsion.j].amber_type;
            let t_k = &topo.atoms[torsion.k].amber_type;
            let t_l = &topo.atoms[torsion.l].amber_type;
            if let Some(terms) = ff.get_torsion(t_i, t_j, t_k, t_l) {
                for term in terms {
                    ti_arr.push(torsion.i as i32);
                    tj_arr.push(torsion.j as i32);
                    tk_arr.push(torsion.k as i32);
                    tl_arr.push(torsion.l as i32);
                    tv.push(term.v / term.div);
                    tf.push(term.f);
                    tp.push(term.phi0);
                }
            }
        }
        for torsion in &topo.improper_torsions {
            let t_i = &topo.atoms[torsion.i].amber_type;
            let t_j = &topo.atoms[torsion.j].amber_type;
            let t_k = &topo.atoms[torsion.k].amber_type;
            let t_l = &topo.atoms[torsion.l].amber_type;
            if let Some(terms) = ff.get_improper_torsion(t_i, t_j, t_k, t_l) {
                for term in terms {
                    ti_arr.push(torsion.i as i32);
                    tj_arr.push(torsion.j as i32);
                    tk_arr.push(torsion.k as i32);
                    tl_arr.push(torsion.l as i32);
                    tv.push(term.v / term.div);
                    tf.push(term.f);
                    tp.push(term.phi0);
                }
            }
        }

        let cutoff = ff.nonbonded_cutoff();
        let cuton = ff.switching_on();

        let topo_gpu = GpuTopology {
            pair_i: stream.clone_htod(&pi)?,
            pair_j: stream.clone_htod(&pj)?,
            pair_14: stream.clone_htod(&p14)?,
            n_pairs: n_pairs as i32,
            atom_types: stream.clone_htod(&type_idx)?,
            charges: stream.clone_htod(&charges)?,
            is_h: stream.clone_htod(&is_h)?,
            lj_r: stream.clone_htod(&lj_r)?,
            lj_eps: stream.clone_htod(&lj_eps)?,
            eef1_dg_free: stream.clone_htod(&eef1_dg_free)?,
            eef1_volume: stream.clone_htod(&eef1_volume)?,
            eef1_sigma: stream.clone_htod(&eef1_sigma)?,
            eef1_r_min: stream.clone_htod(&eef1_r_min_v)?,
            bond_i: stream.clone_htod(&bi)?,
            bond_j: stream.clone_htod(&bj)?,
            bond_k: stream.clone_htod(&bk)?,
            bond_r0: stream.clone_htod(&br0)?,
            n_bonds: bi.len() as i32,
            angle_i: stream.clone_htod(&ai)?,
            angle_j: stream.clone_htod(&aj)?,
            angle_k: stream.clone_htod(&ak)?,
            angle_kf: stream.clone_htod(&akf)?,
            angle_t0: stream.clone_htod(&at0)?,
            n_angles: ai.len() as i32,
            tor_i: stream.clone_htod(&ti_arr)?,
            tor_j: stream.clone_htod(&tj_arr)?,
            tor_k: stream.clone_htod(&tk_arr)?,
            tor_l: stream.clone_htod(&tl_arr)?,
            tor_v: stream.clone_htod(&tv)?,
            tor_f: stream.clone_htod(&tf)?,
            tor_p: stream.clone_htod(&tp)?,
            n_torsion_terms: ti_arr.len() as i32,
            n_atoms,
            self_solvation: self_solv,
            cutoff_sq: cutoff * cutoff,
            cuton_sq: cuton * cuton,
            eef1_cutoff_sq: 81.0,
            coulomb_factor: 332.0,
            scee_inv: 1.0 / ff.scee(),
            scnb_inv: 1.0 / ff.scnb(),
            pi_sqrt_pi: 5.568_327_996_831_708,
        };

        let np = n_pairs.max(1);
        let nb = (bi.len()).max(1);
        let na = (ai.len()).max(1);
        let nt = (ti_arr.len()).max(1);

        Ok(Self {
            coords: stream.alloc_zeros::<f64>(n_atoms * 3)?,
            forces: stream.alloc_zeros::<f64>(n_atoms * 3)?,
            pair_vdw: stream.alloc_zeros::<f64>(np)?,
            pair_elec: stream.alloc_zeros::<f64>(np)?,
            pair_solv: stream.alloc_zeros::<f64>(np)?,
            bond_energies: stream.alloc_zeros::<f64>(nb)?,
            angle_energies: stream.alloc_zeros::<f64>(na)?,
            tor_energies: stream.alloc_zeros::<f64>(nt)?,
            stream,
            topo_gpu,
        })
    }

    /// Re-upload NBL pairs after a neighbor list rebuild.
    pub fn refresh_nbl(&mut self, nbl: &NeighborList) -> Result<(), Box<dyn std::error::Error>> {
        let pi: Vec<i32> = nbl.pairs.iter().map(|p| p.i as i32).collect();
        let pj: Vec<i32> = nbl.pairs.iter().map(|p| p.j as i32).collect();
        let p14: Vec<i32> = nbl.pairs.iter().map(|p| if p.is_14 { 1 } else { 0 }).collect();
        let np = pi.len();

        self.topo_gpu.pair_i = self.stream.clone_htod(&pi)?;
        self.topo_gpu.pair_j = self.stream.clone_htod(&pj)?;
        self.topo_gpu.pair_14 = self.stream.clone_htod(&p14)?;
        self.topo_gpu.n_pairs = np as i32;

        // Reallocate energy output buffers if pair count changed
        let np = np.max(1);
        self.pair_vdw = self.stream.alloc_zeros::<f64>(np)?;
        self.pair_elec = self.stream.alloc_zeros::<f64>(np)?;
        self.pair_solv = self.stream.alloc_zeros::<f64>(np)?;

        Ok(())
    }

    /// Launch all kernels, sync, return energy only (no forces download).
    /// Used for line-search energy evaluations (~20 per LBFGS step).
    pub fn energy(
        &mut self,
        gpu: &GpuContext,
        coords_flat: &[f64],
    ) -> Result<EnergyResult, Box<dyn std::error::Error>> {
        self.launch_kernels(gpu, coords_flat)?;
        self.stream.synchronize()?;
        self.read_energy_only()
    }

    /// Launch all kernels, sync, return energy + forces.
    /// Used for the initial and post-step force evaluations.
    pub fn energy_and_forces(
        &mut self,
        gpu: &GpuContext,
        coords_flat: &[f64],
    ) -> Result<(EnergyResult, Vec<[f64; 3]>), Box<dyn std::error::Error>> {
        self.launch_kernels(gpu, coords_flat)?;
        self.stream.synchronize()?;
        self.read_energy_and_forces()
    }

    // --- Internal helpers ---

    fn launch_kernels(
        &mut self,
        gpu: &GpuContext,
        coords_flat: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t = &self.topo_gpu;

        // Upload coords + zero forces
        self.coords = self.stream.clone_htod(coords_flat)?;
        self.forces = self.stream.alloc_zeros::<f64>(t.n_atoms * 3)?;

        // Nonbonded
        {
            let mut a = self.stream.launch_builder(&gpu.kernels.nonbonded);
            a.arg(&self.coords); a.arg(&t.pair_i); a.arg(&t.pair_j); a.arg(&t.pair_14);
            a.arg(&t.lj_r); a.arg(&t.lj_eps); a.arg(&t.atom_types);
            a.arg(&t.charges); a.arg(&t.is_h);
            a.arg(&t.eef1_dg_free); a.arg(&t.eef1_volume);
            a.arg(&t.eef1_sigma); a.arg(&t.eef1_r_min);
            a.arg(&t.n_pairs); a.arg(&t.cutoff_sq); a.arg(&t.cuton_sq);
            a.arg(&t.eef1_cutoff_sq);
            a.arg(&t.coulomb_factor); a.arg(&t.scee_inv); a.arg(&t.scnb_inv);
            a.arg(&t.pi_sqrt_pi);
            a.arg(&mut self.pair_vdw); a.arg(&mut self.pair_elec);
            a.arg(&mut self.pair_solv); a.arg(&mut self.forces);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(t.n_pairs as u32))?;
            }
        }

        // Bonds
        if t.n_bonds > 0 {
            let mut a = self.stream.launch_builder(&gpu.kernels.bond);
            a.arg(&self.coords); a.arg(&t.bond_i); a.arg(&t.bond_j);
            a.arg(&t.bond_k); a.arg(&t.bond_r0);
            a.arg(&t.n_bonds); a.arg(&mut self.bond_energies);
            a.arg(&mut self.forces);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(t.n_bonds as u32))?;
            }
        }

        // Angles
        if t.n_angles > 0 {
            let mut a = self.stream.launch_builder(&gpu.kernels.angle);
            a.arg(&self.coords); a.arg(&t.angle_i); a.arg(&t.angle_j);
            a.arg(&t.angle_k); a.arg(&t.angle_kf); a.arg(&t.angle_t0);
            a.arg(&t.n_angles); a.arg(&mut self.angle_energies);
            a.arg(&mut self.forces);
            unsafe {
                a.launch(LaunchConfig::for_num_elems(t.n_angles as u32))?;
            }
        }

        // Torsions (smaller block size for register pressure)
        if t.n_torsion_terms > 0 {
            let mut a = self.stream.launch_builder(&gpu.kernels.torsion);
            a.arg(&self.coords); a.arg(&t.tor_i); a.arg(&t.tor_j);
            a.arg(&t.tor_k); a.arg(&t.tor_l);
            a.arg(&t.tor_v); a.arg(&t.tor_f); a.arg(&t.tor_p);
            a.arg(&t.n_torsion_terms); a.arg(&mut self.tor_energies);
            a.arg(&mut self.forces);
            let cfg = LaunchConfig {
                grid_dim: (((t.n_torsion_terms as u32) + 63) / 64, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                a.launch(cfg)?;
            }
        }

        Ok(())
    }

    fn read_energy_only(&self) -> Result<EnergyResult, Box<dyn std::error::Error>> {
        let t = &self.topo_gpu;
        let vdw: f64 = self.stream.clone_dtoh(&self.pair_vdw)?.iter().sum();
        let elec: f64 = self.stream.clone_dtoh(&self.pair_elec)?.iter().sum();
        let solv_pair: f64 = self.stream.clone_dtoh(&self.pair_solv)?.iter().sum();
        let bond: f64 = self.stream.clone_dtoh(&self.bond_energies)?.iter().sum();
        let angle: f64 = self.stream.clone_dtoh(&self.angle_energies)?.iter().sum();
        let torsion: f64 = self.stream.clone_dtoh(&self.tor_energies)?.iter().sum();
        let solvation = t.self_solvation + solv_pair;

        let total = bond + angle + torsion + vdw + elec + solvation;
        Ok(EnergyResult {
            bond_stretch: bond,
            angle_bend: angle,
            torsion,
            improper_torsion: 0.0, // merged into torsion on GPU
            vdw,
            electrostatic: elec,
            solvation,
            total,
        })
    }

    fn read_energy_and_forces(
        &self,
    ) -> Result<(EnergyResult, Vec<[f64; 3]>), Box<dyn std::error::Error>> {
        let energy = self.read_energy_only()?;
        let forces_flat = self.stream.clone_dtoh(&self.forces)?;
        let forces: Vec<[f64; 3]> = forces_flat
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        Ok((energy, forces))
    }
}

// ---------------------------------------------------------------------------
// GPU-accelerated SASA (Shrake-Rupley with neighbor prefilter)
// ---------------------------------------------------------------------------

/// GPU-accelerated Shrake-Rupley SASA with precomputed neighbor lists.
///
/// Builds neighbor lists on CPU (using the same distance-based filtering
/// as ferritin's sasa.rs CellList), uploads flat neighbor arrays to GPU,
/// and launches one thread per atom. Each thread loops over N test points
/// × k neighbors (where k is the prefiltered count, typically 20-50).
///
/// Returns the same `Vec<f64>` as `sasa::shrake_rupley` — per-atom SASA
/// in Å². Falls back to CPU if GPU initialization fails.
pub(crate) fn gpu_shrake_rupley(
    coords: &[[f64; 3]],
    radii: &[f64],
    probe: f64,
    n_points: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let gpu = GpuContext::try_global()
        .ok_or("no GPU available")?;

    let n_atoms = coords.len();
    let stream = gpu.ctx.default_stream();

    // Pre-compute expanded radii
    let expanded: Vec<f64> = radii.iter().map(|r| r + probe).collect();
    let expanded_sq: Vec<f64> = expanded.iter().map(|r| r * r).collect();

    // Build neighbor lists on CPU (same logic as sasa.rs)
    let mut neighbor_offsets = Vec::with_capacity(n_atoms);
    let mut neighbor_counts = Vec::with_capacity(n_atoms);
    let mut neighbor_indices = Vec::new();

    for i in 0..n_atoms {
        let ri = expanded[i];
        neighbor_offsets.push(neighbor_indices.len() as i32);
        let mut count = 0i32;
        for j in 0..n_atoms {
            if j == i { continue; }
            let dx = coords[j][0] - coords[i][0];
            let dy = coords[j][1] - coords[i][1];
            let dz = coords[j][2] - coords[i][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let sum_r = ri + expanded[j];
            if dist_sq < sum_r * sum_r {
                neighbor_indices.push(j as i32);
                count += 1;
            }
        }
        neighbor_counts.push(count);
    }

    // Generate golden spiral points (same as sasa.rs)
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let dz = 2.0 / n_points as f64;
    let mut unit_points = Vec::with_capacity(n_points * 3);
    let mut longitude = 0.0f64;
    let mut z = 1.0 - dz / 2.0;
    for _ in 0..n_points {
        let r = (1.0 - z * z).max(0.0).sqrt();
        unit_points.push(longitude.cos() * r);
        unit_points.push(longitude.sin() * r);
        unit_points.push(z);
        z -= dz;
        longitude += golden_angle;
    }

    let inv_n = 1.0 / n_points as f64;
    let four_pi = 4.0 * std::f64::consts::PI;
    let n_atoms_i32 = n_atoms as i32;
    let n_points_i32 = n_points as i32;

    // Upload
    let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
    let d_coords = stream.clone_htod(&coords_flat)?;
    let d_expanded = stream.clone_htod(&expanded)?;
    let d_expanded_sq = stream.clone_htod(&expanded_sq)?;
    let d_unit_points = stream.clone_htod(&unit_points)?;
    let d_nb_offsets = stream.clone_htod(&neighbor_offsets)?;
    let d_nb_counts = stream.clone_htod(&neighbor_counts)?;
    // Ensure at least 1 element for the neighbor array (empty structures)
    let nb_padded = if neighbor_indices.is_empty() { vec![0i32] } else { neighbor_indices };
    let d_nb_indices = stream.clone_htod(&nb_padded)?;
    let mut d_sasa = stream.alloc_zeros::<f64>(n_atoms)?;

    // Launch
    let cfg = LaunchConfig::for_num_elems(n_atoms as u32);
    {
        let mut a = stream.launch_builder(&gpu.kernels.sasa);
        a.arg(&d_coords); a.arg(&d_expanded); a.arg(&d_expanded_sq);
        a.arg(&d_unit_points);
        a.arg(&d_nb_offsets); a.arg(&d_nb_counts); a.arg(&d_nb_indices);
        a.arg(&n_atoms_i32); a.arg(&n_points_i32);
        a.arg(&inv_n); a.arg(&four_pi);
        a.arg(&mut d_sasa);
        unsafe { a.launch(cfg)?; }
    }
    stream.synchronize()?;

    // Download
    let sasa = stream.clone_dtoh(&d_sasa)?;
    Ok(sasa)
}
