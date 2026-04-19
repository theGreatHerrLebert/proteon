//! GPU SASA benchmark: Shrake-Rupley on real crambin.
//! Compares GPU kernel to proteon's CPU implementation.

use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::time::Instant;

use proteon_connector::forcefield::params::{charmm19_eef1, ForceField};
use proteon_connector::forcefield::topology::build_topology;
use proteon_connector::sasa;

mod batch; // for Arc import

const SASA_KERNEL_SRC: &str = include_str!("sasa_kernel.cu");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU SASA Benchmark (Shrake-Rupley) ===\n");

    // Load crambin
    let pdb_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-pdbs/1crn.pdb");
    let (pdb, _) = pdbtbx::ReadOptions::default()
        .set_level(pdbtbx::StrictnessLevel::Loose)
        .read(pdb_path)
        .expect("failed to load PDB");

    let n_points = 960;
    let probe = 1.4;

    // Get coords and radii from proteon's SASA module
    // (need to replicate the atom extraction since sasa_from_pdb is high-level)
    let mut coords = Vec::new();
    let mut radii = Vec::new();
    let first_model = pdb.models().next().unwrap();
    for chain in first_model.chains() {
        for residue in chain.residues() {
            for atom in proteon_connector::altloc::residue_atoms_primary(residue) {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);
                let elem = atom.element()
                    .map(|e| e.symbol().to_string())
                    .unwrap_or_else(|| "C".to_string());
                let r = sasa::vdw_radius(&elem).unwrap_or(1.7);
                radii.push(r);
            }
        }
    }
    let n_atoms = coords.len();
    println!("  {} atoms, {} test points/atom, probe={} Å", n_atoms, n_points, probe);

    // --- CPU reference ---
    let t_cpu = Instant::now();
    let n_cpu_runs = 100;
    let mut cpu_sasa = vec![0.0f64; n_atoms];
    for _ in 0..n_cpu_runs {
        cpu_sasa = sasa::shrake_rupley(&coords, &radii, probe, n_points);
    }
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0 / n_cpu_runs as f64;
    let cpu_total: f64 = cpu_sasa.iter().sum();
    println!("\nCPU ({:.2} ms/eval, {} runs):", cpu_ms, n_cpu_runs);
    println!("  Total SASA = {:.2} Å²", cpu_total);
    println!("  Per-atom range: {:.2} – {:.2} Å²",
        cpu_sasa.iter().copied().fold(f64::INFINITY, f64::min),
        cpu_sasa.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    // --- GPU ---
    println!("\nInitializing CUDA...");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("  Device: {}", ctx.name()?);

    let (major, minor) = ctx.compute_capability()?;
    let arch: &'static str = Box::leak(format!("sm_{}{}", major, minor).into_boxed_str());

    println!("Compiling SASA kernel...");
    let ptx = compile_ptx_with_opts(SASA_KERNEL_SRC, CompileOptions {
        arch: Some(arch), ..Default::default()
    })?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("sasa_shrake_rupley")?;

    // Prepare data
    let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
    let expanded: Vec<f64> = radii.iter().map(|r| r + probe).collect();
    let expanded_sq: Vec<f64> = expanded.iter().map(|r| r * r).collect();

    // Golden spiral points (same as CPU)
    let unit_points = golden_spiral(n_points);
    let unit_flat: Vec<f64> = unit_points.iter().flat_map(|p| p.iter().copied()).collect();

    let inv_n = 1.0 / n_points as f64;
    let four_pi = 4.0 * std::f64::consts::PI;
    let n_atoms_i32 = n_atoms as i32;
    let n_points_i32 = n_points as i32;

    // Upload
    let d_coords = stream.clone_htod(&coords_flat)?;
    let d_expanded = stream.clone_htod(&expanded)?;
    let d_expanded_sq = stream.clone_htod(&expanded_sq)?;
    let d_unit_points = stream.clone_htod(&unit_flat)?;
    let mut d_sasa = stream.alloc_zeros::<f64>(n_atoms)?;

    let cfg = LaunchConfig::for_num_elems(n_atoms as u32);

    // Warm up
    {
        let mut a = stream.launch_builder(&func);
        a.arg(&d_coords); a.arg(&d_expanded); a.arg(&d_expanded_sq);
        a.arg(&d_unit_points);
        a.arg(&n_atoms_i32); a.arg(&n_points_i32);
        a.arg(&inv_n); a.arg(&four_pi);
        a.arg(&mut d_sasa);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;

    // Timed run
    let n_gpu_runs = 100;
    let t_gpu = Instant::now();
    for _ in 0..n_gpu_runs {
        let mut a = stream.launch_builder(&func);
        a.arg(&d_coords); a.arg(&d_expanded); a.arg(&d_expanded_sq);
        a.arg(&d_unit_points);
        a.arg(&n_atoms_i32); a.arg(&n_points_i32);
        a.arg(&inv_n); a.arg(&four_pi);
        a.arg(&mut d_sasa);
        unsafe { a.launch(cfg) }?;
    }
    stream.synchronize()?;
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0 / n_gpu_runs as f64;

    // Read back
    let gpu_sasa = stream.clone_dtoh(&d_sasa)?;
    let gpu_total: f64 = gpu_sasa.iter().sum();

    println!("\nGPU ({:.3} ms/eval, {} runs):", gpu_ms, n_gpu_runs);
    println!("  Total SASA = {:.2} Å²", gpu_total);

    // Compare
    println!("\n=== Comparison ===");
    let total_diff = (gpu_total - cpu_total).abs();
    let total_rel = total_diff / cpu_total;
    println!("  Total: GPU={:.2} CPU={:.2} diff={:.2e} rel={:.2e}",
        gpu_total, cpu_total, total_diff, total_rel);

    // Per-atom comparison
    let max_diff = gpu_sasa.iter().zip(cpu_sasa.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f64, f64::max);
    let max_rel = gpu_sasa.iter().zip(cpu_sasa.iter())
        .filter(|(_, c)| c.abs() > 1.0)
        .map(|(g, c)| (g - c).abs() / c)
        .fold(0.0f64, f64::max);
    println!("  Per-atom max diff: {:.4} Å²", max_diff);
    println!("  Per-atom max rel:  {:.2e}", max_rel);

    println!("\n=== Timing ===");
    println!("  CPU: {:.2} ms/eval", cpu_ms);
    println!("  GPU: {:.3} ms/eval", gpu_ms);
    println!("  Speedup: {:.1}x", cpu_ms / gpu_ms);

    let match_ok = total_rel < 1e-6;
    if match_ok {
        println!("\n=== GPU SASA: SUCCESS ===");
    } else {
        println!("\n=== GPU SASA: MISMATCH (total rel = {:.2e}) ===", total_rel);
    }

    Ok(())
}

/// Golden spiral points on unit sphere — EXACT copy of proteon's sasa.rs
fn golden_spiral(n: usize) -> Vec<[f64; 3]> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let dz = 2.0 / n as f64;
    let mut points = Vec::with_capacity(n);
    let mut longitude = 0.0_f64;
    let mut z = 1.0 - dz / 2.0;
    for _ in 0..n {
        let r = (1.0 - z * z).max(0.0).sqrt();
        points.push([longitude.cos() * r, longitude.sin() * r, z]);
        z -= dz;
        longitude += golden_angle;
    }
    points
}
