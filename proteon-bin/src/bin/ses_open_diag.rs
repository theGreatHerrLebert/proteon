//! Locate the SES weld gap: mesh a protein (analytic cleaned+welded at a tight
//! eps), find the boundary (degree-1) edges, group them into connected chains,
//! and report each chain's size + location. A few small chains ⇒ localized seam
//! defects (a specific patch-adjacency not wired); many scattered ⇒ pervasive.
//!
//!   cargo run --release --bin ses_open_diag -- a.pdb

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::ses_mesh_cleaned_welded;
use proteon_core::surface::geom::{Sphere, Vec3};
use std::collections::HashMap;

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
    for path in &args {
        let name = path.rsplit('/').next().unwrap_or(path);
        let atoms = load_spheres(path);
        if atoms.len() < 2 {
            println!("{name}: LOAD/EMPTY");
            continue;
        }
        let m = match ses_mesh_cleaned_welded(&atoms, 1.4, 48, 10, 0.04, 1e-5) {
            Ok(m) => m,
            Err(e) => {
                println!("{name}: ERR {}", e.chain().last().unwrap());
                continue;
            }
        };
        // Undirected edge use counts.
        let mut uc: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &m.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *uc.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        // Boundary edges = degree 1; non-manifold = degree >= 3.
        let bnd: Vec<(u32, u32)> = uc
            .iter()
            .filter(|(_, &c)| c == 1)
            .map(|(&e, _)| e)
            .collect();
        let nm = uc.values().filter(|&&c| c >= 3).count();
        // Union-find over boundary-edge endpoints to group chains.
        let mut parent: HashMap<u32, u32> = HashMap::new();
        fn find(p: &mut HashMap<u32, u32>, x: u32) -> u32 {
            let mut r = x;
            while p[&r] != r {
                r = p[&r];
            }
            let mut c = x;
            while p[&c] != r {
                let n = p[&c];
                p.insert(c, r);
                c = n;
            }
            r
        }
        for &(a, b) in &bnd {
            parent.entry(a).or_insert(a);
            parent.entry(b).or_insert(b);
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            parent.insert(ra, rb);
        }
        let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
        let keys: Vec<u32> = parent.keys().copied().collect();
        for v in keys {
            let r = find(&mut parent, v);
            groups.entry(r).or_default().push(v);
        }
        let verts = &m.verts;
        println!(
            "{name} ({} atoms): {} boundary edges, {nm} nonmanifold, {} chains",
            atoms.len(),
            bnd.len(),
            groups.len()
        );
        let mut chains: Vec<&Vec<u32>> = groups.values().collect();
        chains.sort_by_key(|g| std::cmp::Reverse(g.len()));
        for (i, g) in chains.iter().enumerate().take(20) {
            let n = g.len() as f64;
            let c = g
                .iter()
                .fold(Vec3::new(0.0, 0.0, 0.0), |a, &v| a + verts[v as usize])
                * (1.0 / n);
            let r = g
                .iter()
                .map(|&v| {
                    let d = verts[v as usize] - c;
                    (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
                })
                .fold(0.0_f64, f64::max);
            println!(
                "    chain {i}: {} verts, centroid=({:+.2},{:+.2},{:+.2}) radius={:.2}",
                g.len(),
                c.x,
                c.y,
                c.z,
                r
            );
            if std::env::var("DUMP_VERTS").is_ok() && i < 4 {
                // Sort verts by azimuth around the chain centroid (in the chain's
                // best-fit plane) so two samplings of the same circle line up for
                // comparison.
                let mut sorted: Vec<(f64, Vec3)> = g
                    .iter()
                    .map(|&v| {
                        let p = verts[v as usize];
                        let d = p - c;
                        (d.y.atan2(d.x), p)
                    })
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (_, p) in sorted {
                    println!("        ({:+.4},{:+.4},{:+.4})", p.x, p.y, p.z);
                }
            }
        }
    }
}
