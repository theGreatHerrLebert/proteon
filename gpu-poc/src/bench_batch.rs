//! Benchmark: multi-stream batched GPU energy evaluation.
//!
//! Loads crambin N times (simulating N independent structures), uploads each
//! to its own CUDA stream, launches all kernels concurrently, synchronizes,
//! and measures throughput. Compares to sequential (single-stream) baseline.

use cudarc::driver::CudaContext;
use std::time::Instant;

use proteon_connector::forcefield::{
    neighbor_list::NeighborList,
    params::{charmm19_eef1, ForceField},
    topology::build_topology,
};

mod batch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_structures: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    println!("=== Multi-Stream Batch Benchmark ({} structures) ===\n", n_structures);

    // Load crambin once, replicate N times
    let pdb_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-pdbs/1crn.pdb");
    let (pdb, _) = pdbtbx::ReadOptions::default()
        .set_level(pdbtbx::StrictnessLevel::Loose)
        .read(pdb_path)
        .expect("failed to load PDB");

    let ff = charmm19_eef1();
    let topo = build_topology(&pdb, &ff);
    let coords: Vec<[f64; 3]> = topo.atoms.iter().map(|a| a.pos).collect();
    let coords_flat: Vec<f64> = coords.iter().flat_map(|c| c.iter().copied()).collect();
    let nbl = NeighborList::build(&coords, ff.nonbonded_cutoff(), &topo.excluded_pairs, &topo.pairs_14);
    println!("  {} atoms, {} NBL pairs per structure", coords.len(), nbl.pairs.len());

    // Init GPU
    let ctx = CudaContext::new(0)?;
    println!("  Device: {}", ctx.name()?);
    let kernels = batch::GpuKernels::compile(&ctx)?;
    println!("  Kernels compiled.\n");

    // Upload topology N times (in production each structure has different topology)
    let default_stream = ctx.default_stream();
    let mut topos = Vec::with_capacity(n_structures);
    let mut states = Vec::with_capacity(n_structures);
    for _ in 0..n_structures {
        let gt = batch::upload_topology(&default_stream, &topo, &nbl, &ff)?;
        let gs = batch::alloc_state(&ctx, &gt)?;
        topos.push(gt);
        states.push(gs);
    }
    println!("  {} topologies uploaded, {} streams created.\n", n_structures, n_structures);

    // --- Sequential baseline: one stream, N structures one-at-a-time ---
    let n_evals = 10; // simulate 10 LBFGS steps worth
    println!("Sequential ({} structures × {} evals)...", n_structures, n_evals);
    let t_seq = Instant::now();
    for _ in 0..n_evals {
        for (gt, gs) in topos.iter().zip(states.iter_mut()) {
            batch::launch_energy_forces(&kernels, gt, gs, &coords_flat)?;
            gs.stream.synchronize()?;
            let (_energy, _forces) = batch::read_results(gt, gs)?;
        }
    }
    let seq_ms = t_seq.elapsed().as_secs_f64() * 1000.0;
    let seq_per_eval = seq_ms / (n_structures * n_evals) as f64;
    println!("  Total: {:.1} ms  Per eval: {:.3} ms  Throughput: {:.0} evals/s",
        seq_ms, seq_per_eval, 1000.0 / seq_per_eval);

    // --- Multi-stream: launch all N, then sync all ---
    println!("\nMulti-stream ({} structures × {} evals)...", n_structures, n_evals);
    let t_multi = Instant::now();
    for _ in 0..n_evals {
        // Launch ALL structures' kernels (non-blocking, different streams)
        for (gt, gs) in topos.iter().zip(states.iter_mut()) {
            batch::launch_energy_forces(&kernels, gt, gs, &coords_flat)?;
        }
        // Sync ALL streams
        for gs in states.iter() {
            gs.stream.synchronize()?;
        }
        // Read results (could overlap with next step's launches)
        for (gt, gs) in topos.iter().zip(states.iter()) {
            let (_energy, _forces) = batch::read_results(gt, gs)?;
        }
    }
    let multi_ms = t_multi.elapsed().as_secs_f64() * 1000.0;
    let multi_per_eval = multi_ms / (n_structures * n_evals) as f64;
    println!("  Total: {:.1} ms  Per eval: {:.3} ms  Throughput: {:.0} evals/s",
        multi_ms, multi_per_eval, 1000.0 / multi_per_eval);

    println!("\n=== Results ===");
    println!("  Sequential: {:.3} ms/eval  ({:.0} evals/s)", seq_per_eval, 1000.0 / seq_per_eval);
    println!("  Multi-stream: {:.3} ms/eval  ({:.0} evals/s)", multi_per_eval, 1000.0 / multi_per_eval);
    println!("  Speedup: {:.1}x", seq_per_eval / multi_per_eval);
    println!("  50K projection (50 steps): {:.1}s",
        50_000.0 * 50.0 * multi_per_eval / 1000.0);

    Ok(())
}
