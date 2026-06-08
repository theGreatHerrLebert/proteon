//! Diagnostic: run the *analytic* cleaned+welded SES path directly (no grid
//! fallback) and print, per protein, whether it succeeded and — on failure — the
//! exact error string. Unlike `ses_corpus` (which uses the hybrid `ses_mesh` and
//! silently falls back to a numerical grid), this surfaces the real failure class
//! so robustness work can be prioritized.
//!
//!   cargo run --release --bin ses_diag -- a.pdb b.pdb ...

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::ses_mesh_cleaned_welded;
use proteon_core::surface::geom::{Sphere, Vec3};
use std::time::Instant;

fn load_spheres(path: &str) -> Vec<Sphere> {
    let Ok(pdb) = proteon_io::pdb_io::load(path) else {
        return Vec::new();
    };
    let Some(model) = pdb.models().next() else {
        return Vec::new();
    };
    model
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
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = 1.4;
    let (mut ok, mut err) = (0usize, 0usize);
    for path in &args {
        let name = path.rsplit('/').next().unwrap_or(path);
        let atoms = load_spheres(path);
        if atoms.len() < 2 {
            println!("{name:<14} LOAD/EMPTY");
            continue;
        }
        let t = Instant::now();
        // Same parameters as the hybrid's analytic attempt in ses_corpus/ses_export.
        match ses_mesh_cleaned_welded(&atoms, probe, 48, 10, 0.04, 1e-4) {
            Ok(m) => {
                ok += 1;
                println!(
                    "{name:<14} {:>5} atoms  OK   wt={} open={} area={:.1}  {:.1}s",
                    atoms.len(),
                    m.is_watertight(),
                    {
                        // count boundary (degree-1) edges
                        m.num_nonmanifold_edges()
                    },
                    m.surface_area(),
                    t.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                err += 1;
                // Root cause is the innermost context.
                let chain: Vec<String> = e.chain().map(|c| c.to_string()).collect();
                println!(
                    "{name:<14} {:>5} atoms  ERR  {}  ({:.1}s)",
                    atoms.len(),
                    chain.last().cloned().unwrap_or_default(),
                    t.elapsed().as_secs_f64()
                );
            }
        }
    }
    println!("\n{ok} ok / {err} err / {} total", args.len());
}
