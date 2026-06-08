//! Generate a protein's solvent-excluded-surface mesh and write it to OBJ/PLY
//! (open in MeshLab, Blender, PyMOL, or any web viewer). Uses the hybrid
//! `ses_mesh` (exact analytic + perturbation, numerical grid fallback) by
//! default; `--sdf <h>` forces the signed-distance grid mesher at spacing `h`.
//!
//!   cargo run --release --bin ses_export -- test-pdbs/1crn.pdb crambin.obj
//!   cargo run --release --bin ses_export -- 1ubq.pdb ubq.ply --sdf 0.2

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::ses_mesh;
use proteon_core::surface::geom::{Sphere, Vec3};
use proteon_core::surface::volume::ses_mesh_sdf;
use std::io::BufWriter;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let sdf_h = args
        .iter()
        .position(|a| a == "--sdf")
        .map(|i| args.remove(i + 1).parse::<f64>().expect("--sdf <spacing>"));
    args.retain(|a| a != "--sdf");
    if args.len() < 2 {
        eprintln!("usage: ses_export <pdb> <out.obj|out.ply> [--sdf <spacing>]");
        std::process::exit(2);
    }
    let (path, out) = (&args[0], &args[1]);
    let probe = 1.4;

    let pdb = proteon_io::pdb_io::load(path).expect("load pdb");
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

    let (mesh, method) = if let Some(h) = sdf_h {
        (ses_mesh_sdf(&atoms, probe, h), format!("grid h={h}"))
    } else {
        let (m, meth) = ses_mesh(&atoms, probe, 48, 10, 0.04, 1e-4, 0.30);
        (m, format!("{meth:?}"))
    };

    let name = path.rsplit('/').next().unwrap_or("surface");
    let file = std::fs::File::create(out).expect("create output");
    let mut w = BufWriter::new(file);
    if std::path::Path::new(out)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("ply"))
    {
        mesh.write_ply(&mut w).expect("write ply");
    } else {
        mesh.write_obj(&mut w, name).expect("write obj");
    }

    println!(
        "{path}: {} atoms -> {out}\n  {} verts, {} tris, area {:.1} Å², method {method}",
        atoms.len(),
        mesh.num_vertices(),
        mesh.num_triangles(),
        mesh.surface_area(),
    );
}
