//! CPU stage profiler for the SDF SES mesher (GPU-target de-risking).
//! Runs `ses_mesh_sdf` at several grid spacings and prints the per-stage
//! breakdown (set via SES_SDF_PROF inside volume.rs).
//!
//!   SES_SDF_PROF=1 cargo run --release --bin sdf_prof -- test-pdbs/1crn.pdb 0.4 0.2 0.15 0.1

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::geom::{Sphere, Vec3};
use proteon_core::surface::volume::ses_mesh_sdf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: sdf_prof <pdb> [spacing ...]");
    let spacings: Vec<f64> = args.filter_map(|a| a.parse().ok()).collect();
    let spacings = if spacings.is_empty() {
        vec![0.4, 0.3, 0.2, 0.15]
    } else {
        spacings
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
    for h in spacings {
        println!("--- spacing h={h} ---");
        let t = Instant::now();
        let m = ses_mesh_sdf(&atoms, probe, h);
        println!(
            "TOTAL h={h}: {:.1}ms  verts={} tris={} area={:.2}",
            t.elapsed().as_secs_f64() * 1e3,
            m.num_vertices(),
            m.num_triangles(),
            m.surface_area()
        );
    }
}
