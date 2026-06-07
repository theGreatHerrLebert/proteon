//! Watertightness generalization check for the cleaned SES weld across a set of
//! proteins. For each PDB: build the cleaned welded mesh and report
//! watertight/open/nonmanifold/euler/components/area/volume. Fast — no analytic
//! baseline, no self-intersection scan (pass --si to add it).
//!
//!   cargo run --release --bin ses_corpus -- a.pdb b.pdb ...
//!   cargo run --release --bin ses_corpus -- --si a.pdb     # + self-intersection

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::assemble::{ses_mesh, SesMethod};
use proteon_core::surface::geom::{Sphere, Vec3};
use proteon_core::surface::mesh::Mesh;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

fn components(m: &Mesh) -> usize {
    let nv = m.num_vertices();
    let mut parent: Vec<u32> = (0..nv as u32).collect();
    fn find(p: &mut [u32], x: u32) -> u32 {
        let mut r = x;
        while p[r as usize] != r {
            r = p[r as usize];
        }
        p[x as usize] = r;
        r
    }
    for t in &m.tris {
        let a = find(&mut parent, t[0]);
        let b = find(&mut parent, t[1]);
        let c = find(&mut parent, t[2]);
        parent[b as usize] = a;
        parent[c as usize] = a;
    }
    let mut seen = HashSet::new();
    for t in &m.tris {
        seen.insert(find(&mut parent, t[0]));
    }
    seen.len()
}

fn edge_stats(m: &Mesh) -> (usize, usize) {
    let mut uc: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &m.tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *uc.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let open = uc.values().filter(|&&c| c == 1).count();
    let nonmanifold = uc.values().filter(|&&c| c >= 3).count();
    (open, nonmanifold)
}

fn load_spheres(path: &str) -> Vec<Sphere> {
    let pdb = match proteon_io::pdb_io::load(path) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let Some(model) = pdb.models().next() else {
        return Vec::new();
    };
    model
        .chains()
        .flat_map(|c| c.residues())
        // Drop waters — isolated solvent spheres are their own trivial SES
        // components and only add noise to a protein-surface generalization check.
        .filter(|r| {
            let n = r.name().unwrap_or("");
            n != "HOH" && n != "WAT" && n != "DOD"
        })
        .flat_map(|r| r.atoms())
        .map(|a| {
            let (x, y, z) = a.pos();
            let elem = a.element().map(|e| e.symbol()).unwrap_or("");
            let r = vdw_radius(elem).unwrap_or(DEFAULT_RADIUS);
            Sphere::new(Vec3::new(x, y, z), r)
        })
        .collect()
}

fn dump_spheres(path: &str, atoms: &[Sphere]) {
    let rows: Vec<String> = atoms
        .iter()
        .map(|s| {
            format!(
                "[{},{},{},{}]",
                s.center.x, s.center.y, s.center.z, s.radius
            )
        })
        .collect();
    let _ = std::fs::write(path, format!("[{}]", rows.join(",")));
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let with_si = args.iter().any(|a| a == "--si");
    args.retain(|a| a != "--si");
    // --dump-dir <dir>: write each protein's exact spheres to <dir>/<name>.json
    // (the SAME input BALL must get) and emit a machine-readable REC line.
    let dump_dir = args
        .iter()
        .position(|a| a == "--dump-dir")
        .map(|i| args.remove(i + 1));
    args.retain(|a| a != "--dump-dir");
    if let Some(d) = &dump_dir {
        let _ = std::fs::create_dir_all(d);
    }
    let probe = 1.4;
    let eps = 1e-4;

    println!(
        "{:<22} {:>5} {:>4} {:>5} {:>5} {:>5} {:>4} {:>10} {:>10} {:>9} {:>6}",
        "pdb", "atoms", "WT?", "open", "nm>=3", "euler", "comp", "area", "volume", "method", "secs"
    );
    let (mut wt_n, mut exact_n, mut processed, mut total) = (0usize, 0usize, 0usize, 0usize);
    for path in &args {
        total += 1;
        let atoms = load_spheres(path);
        let name = path.rsplit('/').next().unwrap_or(path);
        if atoms.len() < 2 {
            println!("{name:<22} {:>5} LOAD/EMPTY", atoms.len());
            continue;
        }
        processed += 1;
        if let Some(d) = &dump_dir {
            dump_spheres(&format!("{d}/{name}.json"), &atoms);
        }
        let t = Instant::now();
        // The hybrid: exact analytic (+ perturbation retry) first, numerical grid
        // fallback otherwise. Always returns a mesh.
        let (m, method) = ses_mesh(&atoms, probe, 48, 10, 0.04, eps, 0.30);
        let (open, nm) = edge_stats(&m);
        let wt = m.is_watertight();
        if wt {
            wt_n += 1;
        }
        if method.is_exact() {
            exact_n += 1;
        }
        let tag = match method {
            SesMethod::Analytic => "exact".to_string(),
            SesMethod::AnalyticPerturbed(n) => format!("pert{n}"),
            SesMethod::NumericalGrid(_) => "grid".to_string(),
        };
        let si = if with_si {
            format!(
                " si={}",
                proteon_core::surface::intersect::self_intersections(&m, 1.0, 5000)
            )
        } else {
            String::new()
        };
        println!(
            "{name:<22} {:>5} {:>4} {:>5} {:>5} {:>5} {:>4} {:>10.3} {:>10.3} {tag:>9} {:>6.1}{si}",
            atoms.len(),
            if wt { "yes" } else { "NO" },
            open,
            nm,
            m.euler_characteristic(),
            components(&m),
            m.surface_area(),
            m.signed_volume(),
            t.elapsed().as_secs_f64(),
        );
        if dump_dir.is_some() {
            println!(
                "REC,{name},{},{},{},{},{:.4},{tag}",
                atoms.len(),
                i32::from(wt),
                open,
                nm,
                m.surface_area()
            );
        }
    }
    println!(
        "\n{wt_n}/{processed} watertight | {exact_n}/{processed} exact-analytic | {} grid-fallback | {} skipped",
        processed - exact_n,
        total - processed
    );
}
