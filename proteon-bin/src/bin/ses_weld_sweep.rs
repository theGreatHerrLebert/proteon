//! Diagnose the SES cross-face weld gap: mesh the raw cleaned patches ONCE
//! (the expensive analytic step), then weld at a sweep of `weld_eps` values and
//! report open edges + area at each. Tells us whether the open seams are a
//! tolerance-tuning issue (open edges drop to 0 at some modest eps without
//! moving the area) or structural (open edges persist / area moves first).
//!
//!   cargo run --release --bin ses_weld_sweep -- a.pdb [b.pdb ...]

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::ses_mesh_cleaned;
use proteon_core::surface::geom::{Sphere, Vec3};

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
    // Sweep weld_eps across 4 decades; grid spacing is 0.04, so anything ≥ ~0.02
    // risks false-merging real features (watch the area column for that).
    let epss = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2];
    for path in &args {
        let name = path.rsplit('/').next().unwrap_or(path);
        let atoms = load_spheres(path);
        if atoms.len() < 2 {
            println!("{name}: LOAD/EMPTY");
            continue;
        }
        let raw = match ses_mesh_cleaned(&atoms, probe, 48, 10, 0.04) {
            Ok(m) => m,
            Err(e) => {
                println!("{name}: analytic ERR {}", e.chain().last().unwrap());
                continue;
            }
        };
        println!(
            "{name} ({} atoms): raw {} verts, {} tris, area {:.3}",
            atoms.len(),
            raw.num_vertices(),
            raw.num_triangles(),
            raw.surface_area()
        );
        println!(
            "    {:>8}  {:>6}  {:>6}  {:>10}",
            "weld_eps", "open", "verts", "area"
        );
        for &eps in &epss {
            let w = raw.welded_within(eps);
            println!(
                "    {eps:>8.0e}  {:>6}  {:>6}  {:>10.3}{}",
                w.num_nonmanifold_edges(),
                w.num_vertices(),
                w.surface_area(),
                if w.is_watertight() {
                    "  <- WATERTIGHT"
                } else {
                    ""
                },
            );
        }
    }
}
