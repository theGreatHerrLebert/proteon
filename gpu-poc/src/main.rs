//! GPU POC: full nonbonded kernel on REAL crambin data.
//!
//! Loads 1crn.pdb via pdbtbx, builds CHARMM19+EEF1 topology via proteon,
//! marshals real pairs/types/charges/EEF1 params to GPU, runs the kernel,
//! and compares to proteon's CPU energy to 1e-4 kcal/mol.
//!
//! Tests both f64 and f32 precision to characterize the tradeoff.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::time::Instant;

use proteon_connector::forcefield::{
    energy::compute_energy_nbl,
    neighbor_list::NeighborList,
    params::{charmm19_eef1, ForceField},
    topology::build_topology,
};

const NONBONDED_KERNEL: &str = include_str!("kernel.cu");
const BONDED_KERNELS: &str = include_str!("bonded_kernels.cu");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Nonbonded on REAL Crambin (CHARMM19+EEF1) ===\n");

    // --- Load 1crn and build real topology ---
    let pdb_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-pdbs/1crn.pdb");
    println!("Loading {}...", pdb_path);
    let (pdb, _) = pdbtbx::ReadOptions::default()
        .set_level(pdbtbx::StrictnessLevel::Loose)
        .read(pdb_path)
        .expect("failed to load PDB");

    let ff = charmm19_eef1();
    let topo = build_topology(&pdb, &ff);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let n_atoms = coords.len();
    println!("  {} atoms in topology", n_atoms);

    // Build NBL with FF-aware cutoff
    let cutoff = ff.nonbonded_cutoff();
    let cuton = ff.switching_on();
    let nbl = NeighborList::build(&coords, cutoff, &topo.excluded_pairs, &topo.pairs_14);
    let n_pairs = nbl.pairs.len();
    println!("  {} NBL pairs within {} Å", n_pairs, cutoff);

    // --- CPU reference via proteon ---
    let cpu_e = compute_energy_nbl(&coords, &topo, &ff, &nbl);
    println!("\nCPU reference (kcal/mol):");
    println!("  vdw  = {:+.6}", cpu_e.vdw);
    println!("  elec = {:+.6}", cpu_e.electrostatic);
    println!("  solv = {:+.6}", cpu_e.solvation);
    println!("  total= {:+.6}", cpu_e.total);

    // --- Marshal topology to flat GPU arrays ---
    let coords_flat: Vec<f64> = coords.iter().flat_map(|c: &[f64; 3]| c.iter().copied()).collect();

    let mut pair_i_arr = Vec::with_capacity(n_pairs);
    let mut pair_j_arr = Vec::with_capacity(n_pairs);
    let mut pair_14_arr = Vec::with_capacity(n_pairs);
    for p in &nbl.pairs {
        pair_i_arr.push(p.i as i32);
        pair_j_arr.push(p.j as i32);
        pair_14_arr.push(if p.is_14 { 1i32 } else { 0 });
    }

    // Build per-atom arrays from topology + FF lookups
    // Type index: map each unique amber_type string to an integer
    let mut type_set: Vec<String> = Vec::new();
    let mut atom_type_idx: Vec<i32> = Vec::with_capacity(n_atoms);
    for a in &topo.atoms {
        let pos = type_set.iter().position(|t| t == &a.amber_type)
            .unwrap_or_else(|| { type_set.push(a.amber_type.clone()); type_set.len() - 1 });
        atom_type_idx.push(pos as i32);
    }
    let n_types = type_set.len();
    println!("  {} unique atom types: {:?}", n_types, type_set);

    // Per-type LJ params
    let mut lj_r_arr = vec![0.0f64; n_types];
    let mut lj_eps_arr = vec![0.0f64; n_types];
    for (i, t) in type_set.iter().enumerate() {
        if let Some(lj) = ff.get_lj(t) {
            lj_r_arr[i] = lj.r;
            lj_eps_arr[i] = lj.epsilon;
        }
    }

    // Per-atom charges, is_hydrogen, EEF1 params
    let charges: Vec<f64> = topo.atoms.iter().map(|a| a.charge).collect();
    let is_h: Vec<i32> = topo.atoms.iter().map(|a| if a.is_hydrogen { 1 } else { 0 }).collect();
    let mut eef1_dg_free = vec![0.0f64; n_atoms];
    let mut eef1_volume = vec![0.0f64; n_atoms];
    let mut eef1_sigma = vec![3.5f64; n_atoms];
    let mut eef1_r_min_arr = vec![1.6f64; n_atoms];
    for (i, a) in topo.atoms.iter().enumerate() {
        if let Some(eef) = ff.get_eef1(&a.amber_type) {
            eef1_dg_free[i] = eef.dg_free;
            eef1_volume[i] = eef.volume;
            eef1_sigma[i] = eef.sigma;
            eef1_r_min_arr[i] = eef.r_min;
        }
    }

    let cutoff_sq = cutoff * cutoff;
    let cuton_sq = cuton * cuton;
    let eef1_cutoff_sq = 81.0f64; // 9 Å
    let coulomb_factor = 332.0f64;
    let scee_inv = 1.0 / ff.scee();
    let scnb_inv = 1.0 / ff.scnb();
    let pi_sqrt_pi = 5.568_327_996_831_708f64;

    // --- GPU ---
    println!("\nInitializing CUDA...");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("  Device: {}", ctx.name()?);

    let (major, minor) = ctx.compute_capability()?;
    let arch: &'static str = Box::leak(format!("sm_{}{}", major, minor).into_boxed_str());
    println!("  Arch: {} (CC {}.{})", arch, major, minor);

    println!("Compiling f64 kernel...");
    let ptx = compile_ptx_with_opts(NONBONDED_KERNEL, CompileOptions {
        arch: Some(arch), ..Default::default()
    })?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("nonbonded_energy_forces")?;

    // Upload
    let d_coords = stream.clone_htod(&coords_flat)?;
    let d_pair_i = stream.clone_htod(&pair_i_arr)?;
    let d_pair_j = stream.clone_htod(&pair_j_arr)?;
    let d_pair_14 = stream.clone_htod(&pair_14_arr)?;
    let d_lj_r = stream.clone_htod(&lj_r_arr)?;
    let d_lj_eps = stream.clone_htod(&lj_eps_arr)?;
    let d_types = stream.clone_htod(&atom_type_idx)?;
    let d_charges = stream.clone_htod(&charges)?;
    let d_is_h = stream.clone_htod(&is_h)?;
    let d_eef1_dg_free = stream.clone_htod(&eef1_dg_free)?;
    let d_eef1_volume = stream.clone_htod(&eef1_volume)?;
    let d_eef1_sigma = stream.clone_htod(&eef1_sigma)?;
    let d_eef1_r_min = stream.clone_htod(&eef1_r_min_arr)?;
    let mut d_pair_vdw = stream.alloc_zeros::<f64>(n_pairs)?;
    let mut d_pair_elec = stream.alloc_zeros::<f64>(n_pairs)?;
    let mut d_pair_solv = stream.alloc_zeros::<f64>(n_pairs)?;
    let mut d_forces = stream.alloc_zeros::<f64>(n_atoms * 3)?;

    let cfg = LaunchConfig::for_num_elems(n_pairs as u32);
    let n_pairs_i32 = n_pairs as i32;

    // Warm up
    {
        let mut a = stream.launch_builder(&func);
        a.arg(&d_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
        a.arg(&d_lj_r); a.arg(&d_lj_eps); a.arg(&d_types); a.arg(&d_charges); a.arg(&d_is_h);
        a.arg(&d_eef1_dg_free); a.arg(&d_eef1_volume); a.arg(&d_eef1_sigma); a.arg(&d_eef1_r_min);
        a.arg(&n_pairs_i32); a.arg(&cutoff_sq); a.arg(&cuton_sq); a.arg(&eef1_cutoff_sq);
        a.arg(&coulomb_factor); a.arg(&scee_inv); a.arg(&scnb_inv); a.arg(&pi_sqrt_pi);
        a.arg(&mut d_pair_vdw); a.arg(&mut d_pair_elec); a.arg(&mut d_pair_solv); a.arg(&mut d_forces);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;

    // Timed run
    let n_runs = 1000;
    let t0 = Instant::now();
    for _ in 0..n_runs {
        d_forces = stream.alloc_zeros::<f64>(n_atoms * 3)?;
        let mut a = stream.launch_builder(&func);
        a.arg(&d_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
        a.arg(&d_lj_r); a.arg(&d_lj_eps); a.arg(&d_types); a.arg(&d_charges); a.arg(&d_is_h);
        a.arg(&d_eef1_dg_free); a.arg(&d_eef1_volume); a.arg(&d_eef1_sigma); a.arg(&d_eef1_r_min);
        a.arg(&n_pairs_i32); a.arg(&cutoff_sq); a.arg(&cuton_sq); a.arg(&eef1_cutoff_sq);
        a.arg(&coulomb_factor); a.arg(&scee_inv); a.arg(&scnb_inv); a.arg(&pi_sqrt_pi);
        a.arg(&mut d_pair_vdw); a.arg(&mut d_pair_elec); a.arg(&mut d_pair_solv); a.arg(&mut d_forces);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

    // Read back and sum
    let vdw_pairs = stream.clone_dtoh(&d_pair_vdw)?;
    let elec_pairs = stream.clone_dtoh(&d_pair_elec)?;
    let solv_pairs = stream.clone_dtoh(&d_pair_solv)?;
    let gpu_vdw: f64 = vdw_pairs.iter().sum();
    let gpu_elec: f64 = elec_pairs.iter().sum();
    let gpu_solv: f64 = solv_pairs.iter().sum();

    // EEF1 self-solvation (done on CPU — trivial loop, not worth GPU)
    let mut self_solv = 0.0f64;
    for a in &topo.atoms {
        if a.is_hydrogen { continue; }
        if let Some(eef) = ff.get_eef1(&a.amber_type) {
            self_solv += eef.dg_ref;
        }
    }
    let gpu_solv_total = self_solv + gpu_solv; // self + pair correction

    println!("\nGPU f64 ({:.3} ms/eval, {} launches):", gpu_ms, n_runs);
    println!("  vdw  = {:+.6}  (CPU: {:+.6}  diff: {:.2e})", gpu_vdw, cpu_e.vdw, (gpu_vdw - cpu_e.vdw).abs());
    println!("  elec = {:+.6}  (CPU: {:+.6}  diff: {:.2e})", gpu_elec, cpu_e.electrostatic, (gpu_elec - cpu_e.electrostatic).abs());
    println!("  solv = {:+.6}  (CPU: {:+.6}  diff: {:.2e})", gpu_solv_total, cpu_e.solvation, (gpu_solv_total - cpu_e.solvation).abs());

    // CPU timing for the same computation
    let t_cpu = Instant::now();
    for _ in 0..100 {
        let _ = compute_energy_nbl(&coords, &topo, &ff, &nbl);
    }
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0 / 100.0;

    println!("\n  CPU: {:.3} ms/eval", cpu_ms);
    println!("  GPU f64: {:.3} ms/eval", gpu_ms);
    println!("  Speedup: {:.1}x", cpu_ms / gpu_ms);

    let vdw_ok = (gpu_vdw - cpu_e.vdw).abs() < 1e-4;
    let elec_ok = (gpu_elec - cpu_e.electrostatic).abs() < 1e-4;
    let solv_ok = (gpu_solv_total - cpu_e.solvation).abs() < 1e-4;
    println!("\n  vdw:  {}  elec: {}  solv: {}",
        if vdw_ok { "PASS" } else { "FAIL" },
        if elec_ok { "PASS" } else { "FAIL" },
        if solv_ok { "PASS" } else { "FAIL" });

    if vdw_ok && elec_ok && solv_ok {
        println!("\n=== F64 KERNEL: SUCCESS ===");
    } else {
        println!("\n=== F64 KERNEL: MISMATCH ===");
    }

    // ================================================================
    // F32 comparison on real crambin data
    // ================================================================
    println!("\n\n=== F32 PRECISION ON REAL CRAMBIN ===\n");

    let f32_src = include_str!("kernel.cu")
        .replace("double", "float")
        .replace("void nonbonded_energy_forces", "void nonbonded_f32");

    println!("Compiling f32 kernel...");
    let ptx32 = compile_ptx_with_opts(&f32_src, CompileOptions {
        arch: Some(arch), ..Default::default()
    })?;
    let mod32 = ctx.load_module(ptx32)?;
    let func32 = mod32.load_function("nonbonded_f32")?;

    // Convert to f32
    let coords_f32: Vec<f32> = coords_flat.iter().map(|&v| v as f32).collect();
    let lj_r_f32: Vec<f32> = lj_r_arr.iter().map(|&v| v as f32).collect();
    let lj_eps_f32: Vec<f32> = lj_eps_arr.iter().map(|&v| v as f32).collect();
    let charges_f32: Vec<f32> = charges.iter().map(|&v| v as f32).collect();
    let eef1_dg_free_f32: Vec<f32> = eef1_dg_free.iter().map(|&v| v as f32).collect();
    let eef1_volume_f32: Vec<f32> = eef1_volume.iter().map(|&v| v as f32).collect();
    let eef1_sigma_f32: Vec<f32> = eef1_sigma.iter().map(|&v| v as f32).collect();
    let eef1_r_min_f32: Vec<f32> = eef1_r_min_arr.iter().map(|&v| v as f32).collect();

    let d32_coords = stream.clone_htod(&coords_f32)?;
    let d32_lj_r = stream.clone_htod(&lj_r_f32)?;
    let d32_lj_eps = stream.clone_htod(&lj_eps_f32)?;
    let d32_charges = stream.clone_htod(&charges_f32)?;
    let d32_eef1_dg = stream.clone_htod(&eef1_dg_free_f32)?;
    let d32_eef1_vol = stream.clone_htod(&eef1_volume_f32)?;
    let d32_eef1_sig = stream.clone_htod(&eef1_sigma_f32)?;
    let d32_eef1_rm = stream.clone_htod(&eef1_r_min_f32)?;
    let mut d32_vdw = stream.alloc_zeros::<f32>(n_pairs)?;
    let mut d32_elec = stream.alloc_zeros::<f32>(n_pairs)?;
    let mut d32_solv = stream.alloc_zeros::<f32>(n_pairs)?;
    let mut d32_forces = stream.alloc_zeros::<f32>(n_atoms * 3)?;

    let cutoff_sq_f32 = cutoff_sq as f32;
    let cuton_sq_f32 = cuton_sq as f32;
    let eef1_cutoff_sq_f32 = eef1_cutoff_sq as f32;
    let coulomb_f32 = coulomb_factor as f32;
    let scee_f32 = scee_inv as f32;
    let scnb_f32 = scnb_inv as f32;
    let pisp_f32 = pi_sqrt_pi as f32;

    // Warm up f32
    {
        let mut a = stream.launch_builder(&func32);
        a.arg(&d32_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
        a.arg(&d32_lj_r); a.arg(&d32_lj_eps); a.arg(&d_types); a.arg(&d32_charges); a.arg(&d_is_h);
        a.arg(&d32_eef1_dg); a.arg(&d32_eef1_vol); a.arg(&d32_eef1_sig); a.arg(&d32_eef1_rm);
        a.arg(&n_pairs_i32); a.arg(&cutoff_sq_f32); a.arg(&cuton_sq_f32); a.arg(&eef1_cutoff_sq_f32);
        a.arg(&coulomb_f32); a.arg(&scee_f32); a.arg(&scnb_f32); a.arg(&pisp_f32);
        a.arg(&mut d32_vdw); a.arg(&mut d32_elec); a.arg(&mut d32_solv); a.arg(&mut d32_forces);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;

    // Timed f32
    let t32 = Instant::now();
    for _ in 0..n_runs {
        d32_forces = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let mut a = stream.launch_builder(&func32);
        a.arg(&d32_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
        a.arg(&d32_lj_r); a.arg(&d32_lj_eps); a.arg(&d_types); a.arg(&d32_charges); a.arg(&d_is_h);
        a.arg(&d32_eef1_dg); a.arg(&d32_eef1_vol); a.arg(&d32_eef1_sig); a.arg(&d32_eef1_rm);
        a.arg(&n_pairs_i32); a.arg(&cutoff_sq_f32); a.arg(&cuton_sq_f32); a.arg(&eef1_cutoff_sq_f32);
        a.arg(&coulomb_f32); a.arg(&scee_f32); a.arg(&scnb_f32); a.arg(&pisp_f32);
        a.arg(&mut d32_vdw); a.arg(&mut d32_elec); a.arg(&mut d32_solv); a.arg(&mut d32_forces);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;
    let f32_ms = t32.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

    let f32_vdw_v = stream.clone_dtoh(&d32_vdw)?;
    let f32_elec_v = stream.clone_dtoh(&d32_elec)?;
    let f32_solv_v = stream.clone_dtoh(&d32_solv)?;
    let f32_forces_v = stream.clone_dtoh(&d32_forces)?;
    let f32_vdw: f64 = f32_vdw_v.iter().map(|&v| v as f64).sum();
    let f32_elec: f64 = f32_elec_v.iter().map(|&v| v as f64).sum();
    let f32_solv_pair: f64 = f32_solv_v.iter().map(|&v| v as f64).sum();
    let f32_solv_total = self_solv + f32_solv_pair;

    println!("GPU f32 ({:.3} ms/eval):", f32_ms);
    println!("  vdw  = {:+.6}  (CPU f64: {:+.6})", f32_vdw, cpu_e.vdw);
    println!("  elec = {:+.6}  (CPU f64: {:+.6})", f32_elec, cpu_e.electrostatic);
    println!("  solv = {:+.6}  (CPU f64: {:+.6})", f32_solv_total, cpu_e.solvation);

    // Force comparison: GPU f32 vs CPU f64
    let cpu_forces = {
        let (_, forces) = proteon_connector::forcefield::energy::compute_energy_and_forces_nbl(
            &coords, &topo, &ff, &nbl,
        );
        forces
    };
    let cpu_forces_flat: Vec<f64> = cpu_forces.iter().flat_map(|f: &[f64; 3]| f.iter().copied()).collect();
    let f32_forces_f64: Vec<f64> = f32_forces_v.iter().map(|&v| v as f64).collect();
    let max_fdiff = cpu_forces_flat.iter().zip(f32_forces_f64.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);
    let cpu_fmax = cpu_forces_flat.chunks(3)
        .map(|f| (f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).sqrt())
        .fold(0.0_f64, f64::max);

    println!("\n=== F64 vs F32 precision on REAL crambin ===");
    println!("  vdw:   delta={:.2e} kcal/mol  rel={:.2e}",
        (cpu_e.vdw - f32_vdw).abs(),
        if cpu_e.vdw.abs() > 1e-10 { (cpu_e.vdw - f32_vdw).abs() / cpu_e.vdw.abs() } else { 0.0 });
    println!("  elec:  delta={:.2e} kcal/mol  rel={:.2e}",
        (cpu_e.electrostatic - f32_elec).abs(),
        (cpu_e.electrostatic - f32_elec).abs() / cpu_e.electrostatic.abs());
    println!("  solv:  delta={:.2e} kcal/mol  rel={:.2e}",
        (cpu_e.solvation - f32_solv_total).abs(),
        (cpu_e.solvation - f32_solv_total).abs() / cpu_e.solvation.abs());
    println!("  force: max_diff={:.2e}  max_cpu_force={:.2e}  rel={:.2e}",
        max_fdiff, cpu_fmax, max_fdiff / cpu_fmax.max(1e-10));

    println!("\n=== Nonbonded timing summary ===");
    println!("  CPU f64:  {:.3} ms/eval", cpu_ms);
    println!("  GPU f64:  {:.3} ms/eval  ({:.1}x vs CPU)", gpu_ms, cpu_ms / gpu_ms);
    println!("  GPU f32:  {:.3} ms/eval  ({:.1}x vs CPU, {:.1}x vs GPU f64)",
        f32_ms, cpu_ms / f32_ms, gpu_ms / f32_ms);

    // ================================================================
    // FULL GPU ENERGY: bonded + nonbonded in one evaluation
    // ================================================================
    println!("\n\n=== FULL GPU ENERGY EVAL (ALL TERMS) ===\n");

    println!("Compiling bonded kernels...");
    let bonded_ptx = compile_ptx_with_opts(BONDED_KERNELS, CompileOptions {
        arch: Some(arch), ..Default::default()
    })?;
    let bonded_mod = ctx.load_module(bonded_ptx)?;
    let bond_func = bonded_mod.load_function("bond_energy_forces")?;
    let angle_func = bonded_mod.load_function("angle_energy_forces")?;
    let torsion_func = bonded_mod.load_function("torsion_energy_forces")?;

    // Marshal bonded data from topology
    let n_bonds = topo.bonds.len();
    let n_angles = topo.angles.len();
    println!("  {} bonds, {} angles", n_bonds, n_angles);

    // Bonds: look up k, r0 for each bond
    let mut b_i = Vec::with_capacity(n_bonds);
    let mut b_j = Vec::with_capacity(n_bonds);
    let mut b_k_arr = Vec::with_capacity(n_bonds);
    let mut b_r0 = Vec::with_capacity(n_bonds);
    for bond in &topo.bonds {
        let ti = &topo.atoms[bond.i].amber_type;
        let tj = &topo.atoms[bond.j].amber_type;
        if let Some(bp) = ff.get_bond(ti, tj) {
            b_i.push(bond.i as i32);
            b_j.push(bond.j as i32);
            b_k_arr.push(bp.k);
            b_r0.push(bp.r0);
        }
    }
    let n_bonds_actual = b_i.len();

    // Angles: look up k, theta0
    let mut a_i = Vec::new();
    let mut a_j = Vec::new();
    let mut a_k = Vec::new();
    let mut a_kf = Vec::new();
    let mut a_t0 = Vec::new();
    for angle in &topo.angles {
        let ti = &topo.atoms[angle.i].amber_type;
        let tj = &topo.atoms[angle.j].amber_type;
        let tk = &topo.atoms[angle.k].amber_type;
        if let Some(ap) = ff.get_angle(ti, tj, tk) {
            a_i.push(angle.i as i32);
            a_j.push(angle.j as i32);
            a_k.push(angle.k as i32);
            a_kf.push(ap.k);
            a_t0.push(ap.theta0);
        }
    }
    let n_angles_actual = a_i.len();

    // Torsions: expand to individual Fourier terms
    let mut t_i = Vec::new();
    let mut t_j = Vec::new();
    let mut t_k = Vec::new();
    let mut t_l = Vec::new();
    let mut t_v = Vec::new();
    let mut t_f = Vec::new();
    let mut t_p = Vec::new();
    // Proper torsions: use get_torsion only
    for torsion in &topo.torsions {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;
        if let Some(terms) = ff.get_torsion(ti, tj, tk, tl) {
            for term in terms {
                t_i.push(torsion.i as i32);
                t_j.push(torsion.j as i32);
                t_k.push(torsion.k as i32);
                t_l.push(torsion.l as i32);
                t_v.push(term.v / term.div);
                t_f.push(term.f);
                t_p.push(term.phi0);
            }
        }
    }
    // Improper torsions: use get_improper_torsion only
    for torsion in &topo.improper_torsions {
        let ti = &topo.atoms[torsion.i].amber_type;
        let tj = &topo.atoms[torsion.j].amber_type;
        let tk = &topo.atoms[torsion.k].amber_type;
        let tl = &topo.atoms[torsion.l].amber_type;
        if let Some(terms) = ff.get_improper_torsion(ti, tj, tk, tl) {
            for term in terms {
                t_i.push(torsion.i as i32);
                t_j.push(torsion.j as i32);
                t_k.push(torsion.k as i32);
                t_l.push(torsion.l as i32);
                t_v.push(term.v / term.div);
                t_f.push(term.f);
                t_p.push(term.phi0);
            }
        }
    }
    let n_torsion_terms = t_i.len();
    println!("  {} bond params, {} angle params, {} torsion terms",
        n_bonds_actual, n_angles_actual, n_torsion_terms);

    // Upload bonded data
    let db_i = stream.clone_htod(&b_i)?;
    let db_j = stream.clone_htod(&b_j)?;
    let db_k = stream.clone_htod(&b_k_arr)?;
    let db_r0 = stream.clone_htod(&b_r0)?;
    let da_i = stream.clone_htod(&a_i)?;
    let da_j = stream.clone_htod(&a_j)?;
    let da_k = stream.clone_htod(&a_k)?;
    let da_kf = stream.clone_htod(&a_kf)?;
    let da_t0 = stream.clone_htod(&a_t0)?;
    let dt_i = stream.clone_htod(&t_i)?;
    let dt_j = stream.clone_htod(&t_j)?;
    let dt_k = stream.clone_htod(&t_k)?;
    let dt_l = stream.clone_htod(&t_l)?;
    let dt_v = stream.clone_htod(&t_v)?;
    let dt_f = stream.clone_htod(&t_f)?;
    let dt_p = stream.clone_htod(&t_p)?;

    let mut db_energies = stream.alloc_zeros::<f64>(n_bonds_actual)?;
    let mut da_energies = stream.alloc_zeros::<f64>(n_angles_actual)?;
    let mut dt_energies = stream.alloc_zeros::<f64>(n_torsion_terms)?;

    // Full GPU evaluation: nonbonded + bonded
    let n_bonds_i32 = n_bonds_actual as i32;
    let n_angles_i32 = n_angles_actual as i32;
    let n_torsions_i32 = n_torsion_terms as i32;

    // Warm up all kernels
    d_forces = stream.alloc_zeros::<f64>(n_atoms * 3)?;
    {
        let mut a = stream.launch_builder(&func);
        a.arg(&d_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
        a.arg(&d_lj_r); a.arg(&d_lj_eps); a.arg(&d_types); a.arg(&d_charges); a.arg(&d_is_h);
        a.arg(&d_eef1_dg_free); a.arg(&d_eef1_volume); a.arg(&d_eef1_sigma); a.arg(&d_eef1_r_min);
        a.arg(&n_pairs_i32); a.arg(&cutoff_sq); a.arg(&cuton_sq); a.arg(&eef1_cutoff_sq);
        a.arg(&coulomb_factor); a.arg(&scee_inv); a.arg(&scnb_inv); a.arg(&pi_sqrt_pi);
        a.arg(&mut d_pair_vdw); a.arg(&mut d_pair_elec); a.arg(&mut d_pair_solv); a.arg(&mut d_forces);
        unsafe { a.launch(cfg) }?;
    }
    {
        let mut a = stream.launch_builder(&bond_func);
        a.arg(&d_coords); a.arg(&db_i); a.arg(&db_j); a.arg(&db_k); a.arg(&db_r0);
        a.arg(&n_bonds_i32); a.arg(&mut db_energies); a.arg(&mut d_forces);
        unsafe { a.launch(LaunchConfig::for_num_elems(n_bonds_actual as u32)) }?;
    }
    {
        let mut a = stream.launch_builder(&angle_func);
        a.arg(&d_coords); a.arg(&da_i); a.arg(&da_j); a.arg(&da_k); a.arg(&da_kf); a.arg(&da_t0);
        a.arg(&n_angles_i32); a.arg(&mut da_energies); a.arg(&mut d_forces);
        unsafe { a.launch(LaunchConfig::for_num_elems(n_angles_actual as u32)) }?;
    }
    if n_torsion_terms > 0 {
        let mut a = stream.launch_builder(&torsion_func);
        a.arg(&d_coords); a.arg(&dt_i); a.arg(&dt_j); a.arg(&dt_k); a.arg(&dt_l);
        a.arg(&dt_v); a.arg(&dt_f); a.arg(&dt_p);
        a.arg(&n_torsions_i32); a.arg(&mut dt_energies); a.arg(&mut d_forces);
        // Use smaller block size for torsion (many f64 registers per thread)
            let tor_cfg = LaunchConfig {
                grid_dim: (((n_torsion_terms as u32) + 63) / 64, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { a.launch(tor_cfg) }?;
    }
    stream.synchronize()?;

    // Timed full evaluation (100 runs)
    let n_full_runs = 100;
    let t_full = Instant::now();
    for _ in 0..n_full_runs {
        d_forces = stream.alloc_zeros::<f64>(n_atoms * 3)?;
        // Nonbonded
        {
            let mut a = stream.launch_builder(&func);
            a.arg(&d_coords); a.arg(&d_pair_i); a.arg(&d_pair_j); a.arg(&d_pair_14);
            a.arg(&d_lj_r); a.arg(&d_lj_eps); a.arg(&d_types); a.arg(&d_charges); a.arg(&d_is_h);
            a.arg(&d_eef1_dg_free); a.arg(&d_eef1_volume); a.arg(&d_eef1_sigma); a.arg(&d_eef1_r_min);
            a.arg(&n_pairs_i32); a.arg(&cutoff_sq); a.arg(&cuton_sq); a.arg(&eef1_cutoff_sq);
            a.arg(&coulomb_factor); a.arg(&scee_inv); a.arg(&scnb_inv); a.arg(&pi_sqrt_pi);
            a.arg(&mut d_pair_vdw); a.arg(&mut d_pair_elec); a.arg(&mut d_pair_solv); a.arg(&mut d_forces);
            unsafe { a.launch(cfg) }?;
        }
        // Bonds
        {
            let mut a = stream.launch_builder(&bond_func);
            a.arg(&d_coords); a.arg(&db_i); a.arg(&db_j); a.arg(&db_k); a.arg(&db_r0);
            a.arg(&n_bonds_i32); a.arg(&mut db_energies); a.arg(&mut d_forces);
            unsafe { a.launch(LaunchConfig::for_num_elems(n_bonds_actual as u32)) }?;
        }
        // Angles
        {
            let mut a = stream.launch_builder(&angle_func);
            a.arg(&d_coords); a.arg(&da_i); a.arg(&da_j); a.arg(&da_k); a.arg(&da_kf); a.arg(&da_t0);
            a.arg(&n_angles_i32); a.arg(&mut da_energies); a.arg(&mut d_forces);
            unsafe { a.launch(LaunchConfig::for_num_elems(n_angles_actual as u32)) }?;
        }
        // Torsions
        if n_torsion_terms > 0 {
            let mut a = stream.launch_builder(&torsion_func);
            a.arg(&d_coords); a.arg(&dt_i); a.arg(&dt_j); a.arg(&dt_k); a.arg(&dt_l);
            a.arg(&dt_v); a.arg(&dt_f); a.arg(&dt_p);
            a.arg(&n_torsions_i32); a.arg(&mut dt_energies); a.arg(&mut d_forces);
            // Use smaller block size for torsion (many f64 registers per thread)
            let tor_cfg = LaunchConfig {
                grid_dim: (((n_torsion_terms as u32) + 63) / 64, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { a.launch(tor_cfg) }?;
        }
    }
    stream.synchronize()?;
    let full_ms = t_full.elapsed().as_secs_f64() * 1000.0 / n_full_runs as f64;

    // Read back all component energies
    let bond_e: f64 = stream.clone_dtoh(&db_energies)?.iter().sum();
    let angle_e: f64 = stream.clone_dtoh(&da_energies)?.iter().sum();
    let torsion_e: f64 = stream.clone_dtoh(&dt_energies)?.iter().sum();
    let nb_vdw: f64 = stream.clone_dtoh(&d_pair_vdw)?.iter().sum();
    let nb_elec: f64 = stream.clone_dtoh(&d_pair_elec)?.iter().sum();
    let nb_solv: f64 = stream.clone_dtoh(&d_pair_solv)?.iter().sum();
    let gpu_total = bond_e + angle_e + torsion_e + nb_vdw + nb_elec + self_solv + nb_solv;

    println!("GPU full energy eval ({:.3} ms, {} runs):", full_ms, n_full_runs);
    println!("  bond_stretch = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        bond_e, cpu_e.bond_stretch, (bond_e - cpu_e.bond_stretch).abs());
    println!("  angle_bend   = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        angle_e, cpu_e.angle_bend, (angle_e - cpu_e.angle_bend).abs());
    println!("  torsion      = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        torsion_e, cpu_e.torsion + cpu_e.improper_torsion,
        (torsion_e - cpu_e.torsion - cpu_e.improper_torsion).abs());
    println!("  vdw          = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        nb_vdw, cpu_e.vdw, (nb_vdw - cpu_e.vdw).abs());
    println!("  electrostatic= {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        nb_elec, cpu_e.electrostatic, (nb_elec - cpu_e.electrostatic).abs());
    println!("  solvation    = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        self_solv + nb_solv, cpu_e.solvation, (self_solv + nb_solv - cpu_e.solvation).abs());
    println!("  TOTAL        = {:+.6}  (CPU: {:+.6}  diff: {:.2e})",
        gpu_total, cpu_e.total, (gpu_total - cpu_e.total).abs());

    println!("\n=== Full GPU vs CPU timing ===");
    println!("  CPU (full energy+forces): {:.3} ms", cpu_ms);
    println!("  GPU (all kernels):        {:.3} ms  ({:.1}x speedup)", full_ms, cpu_ms / full_ms);

    let total_match = (gpu_total - cpu_e.total).abs() < 1.0; // 1 kcal/mol tolerance for bonded FP
    if total_match {
        println!("\n=== FULL GPU ENERGY EVAL: SUCCESS ===");
    } else {
        println!("\n=== FULL GPU ENERGY EVAL: TOTAL MISMATCH ===");
    }

    Ok(())
}
