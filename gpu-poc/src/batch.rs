//! Multi-stream batched GPU energy evaluation.
//!
//! Processes N structures concurrently using N CUDA streams. Each structure
//! gets its own stream, so kernel launches and data transfers overlap
//! automatically via the GPU's hardware scheduler.
//!
//! Usage pattern for LBFGS:
//!   1. Upload all structures' topology data once (NBL, types, charges, etc.)
//!   2. Per LBFGS step: upload coords → launch all kernels → download forces
//!   3. CPU does LBFGS direction computation per structure (serial or rayon)
//!   4. Repeat until converged

use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
pub use std::sync::Arc;

use proteon_connector::forcefield::{
    neighbor_list::NeighborList,
    params::ForceField,
    topology::Topology,
};

/// Pre-uploaded GPU buffers for one structure's topology (immutable between
/// LBFGS steps — only coords change).
pub struct GpuTopology {
    // NBL pairs
    pub pair_i: CudaSlice<i32>,
    pub pair_j: CudaSlice<i32>,
    pub pair_14: CudaSlice<i32>,
    pub n_pairs: i32,
    // Per-atom
    pub atom_types: CudaSlice<i32>,
    pub charges: CudaSlice<f64>,
    pub is_h: CudaSlice<i32>,
    // Per-type LJ
    pub lj_r: CudaSlice<f64>,
    pub lj_eps: CudaSlice<f64>,
    // EEF1 per-atom
    pub eef1_dg_free: CudaSlice<f64>,
    pub eef1_volume: CudaSlice<f64>,
    pub eef1_sigma: CudaSlice<f64>,
    pub eef1_r_min: CudaSlice<f64>,
    // Bonded
    pub bond_i: CudaSlice<i32>,
    pub bond_j: CudaSlice<i32>,
    pub bond_k: CudaSlice<f64>,
    pub bond_r0: CudaSlice<f64>,
    pub n_bonds: i32,
    pub angle_i: CudaSlice<i32>,
    pub angle_j: CudaSlice<i32>,
    pub angle_k: CudaSlice<i32>,
    pub angle_kf: CudaSlice<f64>,
    pub angle_t0: CudaSlice<f64>,
    pub n_angles: i32,
    pub tor_i: CudaSlice<i32>,
    pub tor_j: CudaSlice<i32>,
    pub tor_k: CudaSlice<i32>,
    pub tor_l: CudaSlice<i32>,
    pub tor_v: CudaSlice<f64>,
    pub tor_f: CudaSlice<f64>,
    pub tor_p: CudaSlice<f64>,
    pub n_torsion_terms: i32,
    // Constants
    pub n_atoms: usize,
    pub self_solvation: f64, // Σ dG_ref (computed on CPU, constant)
    // FF constants
    pub cutoff_sq: f64,
    pub cuton_sq: f64,
    pub eef1_cutoff_sq: f64,
    pub coulomb_factor: f64,
    pub scee_inv: f64,
    pub scnb_inv: f64,
    pub pi_sqrt_pi: f64,
}

/// Compiled GPU kernels (shared across all structures).
pub struct GpuKernels {
    pub nonbonded: CudaFunction,
    pub bond: CudaFunction,
    pub angle: CudaFunction,
    pub torsion: CudaFunction,
}

/// Per-structure mutable state (coords + output buffers), one per stream.
pub struct GpuState {
    pub stream: Arc<CudaStream>,
    pub coords: CudaSlice<f64>,
    pub forces: CudaSlice<f64>,
    pub pair_vdw: CudaSlice<f64>,
    pub pair_elec: CudaSlice<f64>,
    pub pair_solv: CudaSlice<f64>,
    pub bond_energies: CudaSlice<f64>,
    pub angle_energies: CudaSlice<f64>,
    pub tor_energies: CudaSlice<f64>,
}

impl GpuKernels {
    /// Compile all kernels once for the detected GPU architecture.
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{}{}", major, minor).into_boxed_str());
        let opts = CompileOptions { arch: Some(arch), ..Default::default() };

        let nb_ptx = compile_ptx_with_opts(include_str!("kernel.cu"), opts.clone())?;
        let nb_mod = ctx.load_module(nb_ptx)?;
        let nonbonded = nb_mod.load_function("nonbonded_energy_forces")?;

        let bonded_ptx = compile_ptx_with_opts(include_str!("bonded_kernels.cu"), opts)?;
        let bonded_mod = ctx.load_module(bonded_ptx)?;
        let bond = bonded_mod.load_function("bond_energy_forces")?;
        let angle = bonded_mod.load_function("angle_energy_forces")?;
        let torsion = bonded_mod.load_function("torsion_energy_forces")?;

        Ok(Self { nonbonded, bond, angle, torsion })
    }
}

/// Upload a structure's topology to GPU. Call once per structure.
pub fn upload_topology<F: ForceField>(
    stream: &Arc<CudaStream>,
    topo: &Topology,
    nbl: &NeighborList,
    ff: &F,
) -> Result<GpuTopology, Box<dyn std::error::Error>> {
    let n_atoms = topo.atoms.len();
    let n_pairs = nbl.pairs.len();

    // NBL pairs
    let pi: Vec<i32> = nbl.pairs.iter().map(|p| p.i as i32).collect();
    let pj: Vec<i32> = nbl.pairs.iter().map(|p| p.j as i32).collect();
    let p14: Vec<i32> = nbl.pairs.iter().map(|p| if p.is_14 { 1 } else { 0 }).collect();

    // Atom type index mapping
    let mut type_names: Vec<String> = Vec::new();
    let mut type_idx: Vec<i32> = Vec::with_capacity(n_atoms);
    for a in &topo.atoms {
        let pos = type_names.iter().position(|t| t == &a.amber_type)
            .unwrap_or_else(|| { type_names.push(a.amber_type.clone()); type_names.len() - 1 });
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
    let is_h: Vec<i32> = topo.atoms.iter().map(|a| if a.is_hydrogen { 1 } else { 0 }).collect();

    let mut eef1_dg_free = vec![0.0f64; n_atoms];
    let mut eef1_volume = vec![0.0f64; n_atoms];
    let mut eef1_sigma = vec![3.5f64; n_atoms];
    let mut eef1_r_min = vec![1.6f64; n_atoms];
    let mut self_solv = 0.0f64;
    for (i, a) in topo.atoms.iter().enumerate() {
        if let Some(eef) = ff.get_eef1(&a.amber_type) {
            eef1_dg_free[i] = eef.dg_free;
            eef1_volume[i] = eef.volume;
            eef1_sigma[i] = eef.sigma;
            eef1_r_min[i] = eef.r_min;
            if !a.is_hydrogen {
                self_solv += eef.dg_ref;
            }
        }
    }

    // Bonded marshalling
    let mut bi = Vec::new(); let mut bj = Vec::new();
    let mut bk = Vec::new(); let mut br0 = Vec::new();
    for bond in &topo.bonds {
        let ti = &topo.atoms[bond.i].amber_type;
        let tj = &topo.atoms[bond.j].amber_type;
        if let Some(bp) = ff.get_bond(ti, tj) {
            bi.push(bond.i as i32); bj.push(bond.j as i32);
            bk.push(bp.k); br0.push(bp.r0);
        }
    }

    let mut ai = Vec::new(); let mut aj = Vec::new(); let mut ak = Vec::new();
    let mut akf = Vec::new(); let mut at0 = Vec::new();
    for angle in &topo.angles {
        let ti = &topo.atoms[angle.i].amber_type;
        let tj = &topo.atoms[angle.j].amber_type;
        let tk = &topo.atoms[angle.k].amber_type;
        if let Some(ap) = ff.get_angle(ti, tj, tk) {
            ai.push(angle.i as i32); aj.push(angle.j as i32); ak.push(angle.k as i32);
            akf.push(ap.k); at0.push(ap.theta0);
        }
    }

    let mut ti_arr = Vec::new(); let mut tj_arr = Vec::new();
    let mut tk_arr = Vec::new(); let mut tl_arr = Vec::new();
    let mut tv = Vec::new(); let mut tf = Vec::new(); let mut tp = Vec::new();
    for torsion in &topo.torsions {
        let t_i = &topo.atoms[torsion.i].amber_type;
        let t_j = &topo.atoms[torsion.j].amber_type;
        let t_k = &topo.atoms[torsion.k].amber_type;
        let t_l = &topo.atoms[torsion.l].amber_type;
        if let Some(terms) = ff.get_torsion(t_i, t_j, t_k, t_l) {
            for term in terms {
                ti_arr.push(torsion.i as i32); tj_arr.push(torsion.j as i32);
                tk_arr.push(torsion.k as i32); tl_arr.push(torsion.l as i32);
                tv.push(term.v / term.div); tf.push(term.f); tp.push(term.phi0);
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
                ti_arr.push(torsion.i as i32); tj_arr.push(torsion.j as i32);
                tk_arr.push(torsion.k as i32); tl_arr.push(torsion.l as i32);
                tv.push(term.v / term.div); tf.push(term.f); tp.push(term.phi0);
            }
        }
    }

    let cutoff = ff.nonbonded_cutoff();
    let cuton = ff.switching_on();

    Ok(GpuTopology {
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
        eef1_r_min: stream.clone_htod(&eef1_r_min)?,
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
    })
}

/// Allocate per-structure mutable buffers on a dedicated stream.
pub fn alloc_state(
    ctx: &Arc<CudaContext>,
    topo_gpu: &GpuTopology,
) -> Result<GpuState, Box<dyn std::error::Error>> {
    let stream = ctx.new_stream()?;
    let n = topo_gpu.n_atoms;
    let np = topo_gpu.n_pairs as usize;
    Ok(GpuState {
        coords: stream.alloc_zeros::<f64>(n * 3)?,
        forces: stream.alloc_zeros::<f64>(n * 3)?,
        pair_vdw: stream.alloc_zeros::<f64>(np)?,
        pair_elec: stream.alloc_zeros::<f64>(np)?,
        pair_solv: stream.alloc_zeros::<f64>(np)?,
        bond_energies: stream.alloc_zeros::<f64>(topo_gpu.n_bonds as usize)?,
        angle_energies: stream.alloc_zeros::<f64>(topo_gpu.n_angles as usize)?,
        tor_energies: stream.alloc_zeros::<f64>(topo_gpu.n_torsion_terms.max(1) as usize)?,
        stream,
    })
}

/// Launch all kernels for one structure on its stream. Non-blocking.
/// Call `state.stream.synchronize()` before reading results.
pub fn launch_energy_forces(
    kernels: &GpuKernels,
    topo: &GpuTopology,
    state: &mut GpuState,
    coords_host: &[f64], // N*3 flat
) -> Result<(), Box<dyn std::error::Error>> {
    // Upload coords
    state.coords = state.stream.clone_htod(coords_host)?;
    // Zero forces
    state.forces = state.stream.alloc_zeros::<f64>(topo.n_atoms * 3)?;

    // Nonbonded
    {
        let mut a = state.stream.launch_builder(&kernels.nonbonded);
        a.arg(&state.coords); a.arg(&topo.pair_i); a.arg(&topo.pair_j); a.arg(&topo.pair_14);
        a.arg(&topo.lj_r); a.arg(&topo.lj_eps); a.arg(&topo.atom_types); a.arg(&topo.charges);
        a.arg(&topo.is_h);
        a.arg(&topo.eef1_dg_free); a.arg(&topo.eef1_volume); a.arg(&topo.eef1_sigma); a.arg(&topo.eef1_r_min);
        a.arg(&topo.n_pairs); a.arg(&topo.cutoff_sq); a.arg(&topo.cuton_sq); a.arg(&topo.eef1_cutoff_sq);
        a.arg(&topo.coulomb_factor); a.arg(&topo.scee_inv); a.arg(&topo.scnb_inv); a.arg(&topo.pi_sqrt_pi);
        a.arg(&mut state.pair_vdw); a.arg(&mut state.pair_elec); a.arg(&mut state.pair_solv);
        a.arg(&mut state.forces);
        unsafe { a.launch(LaunchConfig::for_num_elems(topo.n_pairs as u32)) }?;
    }

    // Bonds
    if topo.n_bonds > 0 {
        let mut a = state.stream.launch_builder(&kernels.bond);
        a.arg(&state.coords); a.arg(&topo.bond_i); a.arg(&topo.bond_j);
        a.arg(&topo.bond_k); a.arg(&topo.bond_r0);
        a.arg(&topo.n_bonds); a.arg(&mut state.bond_energies); a.arg(&mut state.forces);
        unsafe { a.launch(LaunchConfig::for_num_elems(topo.n_bonds as u32)) }?;
    }

    // Angles
    if topo.n_angles > 0 {
        let mut a = state.stream.launch_builder(&kernels.angle);
        a.arg(&state.coords); a.arg(&topo.angle_i); a.arg(&topo.angle_j); a.arg(&topo.angle_k);
        a.arg(&topo.angle_kf); a.arg(&topo.angle_t0);
        a.arg(&topo.n_angles); a.arg(&mut state.angle_energies); a.arg(&mut state.forces);
        unsafe { a.launch(LaunchConfig::for_num_elems(topo.n_angles as u32)) }?;
    }

    // Torsions
    if topo.n_torsion_terms > 0 {
        let mut a = state.stream.launch_builder(&kernels.torsion);
        a.arg(&state.coords); a.arg(&topo.tor_i); a.arg(&topo.tor_j);
        a.arg(&topo.tor_k); a.arg(&topo.tor_l);
        a.arg(&topo.tor_v); a.arg(&topo.tor_f); a.arg(&topo.tor_p);
        a.arg(&topo.n_torsion_terms); a.arg(&mut state.tor_energies); a.arg(&mut state.forces);
        let cfg = LaunchConfig {
            grid_dim: (((topo.n_torsion_terms as u32) + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { a.launch(cfg) }?;
    }

    Ok(())
}

/// Read back results after synchronization. Returns (total_energy, forces).
pub fn read_results(
    topo: &GpuTopology,
    state: &GpuState,
) -> Result<(f64, Vec<[f64; 3]>), Box<dyn std::error::Error>> {
    let vdw: f64 = state.stream.clone_dtoh(&state.pair_vdw)?.iter().sum();
    let elec: f64 = state.stream.clone_dtoh(&state.pair_elec)?.iter().sum();
    let solv_pair: f64 = state.stream.clone_dtoh(&state.pair_solv)?.iter().sum();
    let bond: f64 = state.stream.clone_dtoh(&state.bond_energies)?.iter().sum();
    let angle: f64 = state.stream.clone_dtoh(&state.angle_energies)?.iter().sum();
    let torsion: f64 = state.stream.clone_dtoh(&state.tor_energies)?.iter().sum();

    let total = bond + angle + torsion + vdw + elec + topo.self_solvation + solv_pair;

    let forces_flat = state.stream.clone_dtoh(&state.forces)?;
    let forces: Vec<[f64; 3]> = forces_flat.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    Ok((total, forces))
}
