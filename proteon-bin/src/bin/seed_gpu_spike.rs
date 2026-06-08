//! GPU-K1 spike: benchmark the SES SDF seed stage (nearest exposed surface point
//! per boundary node) on serial CPU vs 16-core CPU vs GPU brute-force kernel, and
//! report parity. Build/run with the `cuda` feature:
//!
//!   cargo run --release --features cuda --bin seed_gpu_spike -- test-pdbs/1crn.pdb 0.4 0.2 0.15

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("seed_gpu_spike requires the `cuda` feature: cargo run --features cuda --bin seed_gpu_spike -- <pdb> [spacing...]");
}

#[cfg(feature = "cuda")]
fn main() {
    use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
    use proteon_core::surface::geom::{Sphere, Vec3};
    use proteon_core::surface::volume::seed_bench;

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: seed_gpu_spike <pdb> [spacing ...]");
    let spacings: Vec<f64> = {
        let v: Vec<f64> = args.filter_map(|a| a.parse().ok()).collect();
        if v.is_empty() {
            vec![0.4, 0.2, 0.15]
        } else {
            v
        }
    };
    let probe = 1.4;
    let pdb = proteon_io::pdb_io::load(&path).expect("load pdb");
    let model = pdb.models().next().expect("no models");
    let atoms: Vec<Sphere> = model
        .chains()
        .flat_map(|c| c.residues())
        .filter(|r| !matches!(r.name().unwrap_or(""), "HOH" | "WAT" | "DOD"))
        .flat_map(|r| r.atoms())
        .map(|a| {
            let (x, y, z) = a.pos();
            let elem = a.element().map(|e| e.symbol()).unwrap_or("");
            let r = vdw_radius(elem).unwrap_or(DEFAULT_RADIUS);
            Sphere::new(Vec3::new(x, y, z), r)
        })
        .collect();
    println!("{path}: {} atoms, probe {probe}", atoms.len());
    println!(
        "{:>5} {:>9} {:>11} {:>12} {:>11} {:>10} {:>10} {:>9}",
        "h", "boundary", "cpu_1core", "cpu_16core", "gpu_kernel", "gpu_total", "speedup", "maxdiff"
    );
    for h in spacings {
        match seed_bench(&atoms, probe, h) {
            Ok(b) => {
                let speedup = b.cpu_parallel_ms / b.gpu_kernel_ms.max(1e-9);
                println!(
                    "{h:>5} {:>9} {:>9.1}ms {:>10.1}ms {:>9.1}ms {:>8.1}ms {:>9.1}x {:>9.0e}",
                    b.n_boundary,
                    b.cpu_serial_ms,
                    b.cpu_parallel_ms,
                    b.gpu_kernel_ms,
                    b.gpu_total_ms,
                    speedup,
                    b.max_feature_diff,
                );
                println!(
                    "        parity: gpu-vs-hash {} | gpu-vs-CPUbrute {} | hash-vs-CPUbrute {} (of which DISTANCE bug {}, max dist err {:.3} Å; rest = equidistant ties)",
                    b.mismatched,
                    b.gpu_vs_cpubrute_mismatch,
                    b.hash_vs_cpubrute_mismatch,
                    b.hash_vs_brute_distance_bug,
                    b.max_distance_error,
                );
            }
            Err(e) => println!("{h:>5}  ERR: {e}"),
        }
    }
}
