//! Scratch harness: run the SES SDF mesher on a real protein and report
//! watertightness / topology / area / volume / timing, and dump the exact
//! spheres used so `ball.ses_area` can be compared apples-to-apples.
//!
//!   cargo run --release --bin ses_protein -- test-pdbs/1crn.pdb 0.3 [0.25 ...]

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::geom::{Sphere, Vec3};
use proteon_core::surface::volume::ses_mesh_sdf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: ses_protein <pdb> [spacing ...]");
    let spacings: Vec<f64> = {
        let v: Vec<f64> = args.filter_map(|a| a.parse().ok()).collect();
        if v.is_empty() {
            vec![0.4, 0.3]
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
        .flat_map(|r| r.atoms())
        .map(|a| {
            let (x, y, z) = a.pos();
            let elem = a.element().map(|e| e.symbol()).unwrap_or("");
            let r = vdw_radius(elem).unwrap_or(DEFAULT_RADIUS);
            Sphere::new(Vec3::new(x, y, z), r)
        })
        .collect();
    println!("{path}: {} atoms, probe {probe}", atoms.len());

    // Dump spheres for the ball-py oracle (exact same input).
    let json: String = {
        let rows: Vec<String> = atoms
            .iter()
            .map(|s| {
                format!(
                    "[{},{},{},{}]",
                    s.center.x, s.center.y, s.center.z, s.radius
                )
            })
            .collect();
        format!("[{}]", rows.join(","))
    };
    std::fs::write("/tmp/ses_spheres.json", &json).expect("write spheres");
    println!("wrote /tmp/ses_spheres.json ({} spheres)", atoms.len());

    // Diagnostic: try the analytic mesher (general-N, non-singular) and report
    // how far it gets (it errors loud on a singular config — the next piece).
    {
        let t = Instant::now();
        match proteon_core::surface::assemble::ses_mesh_analytic(&atoms, probe, 48, 10, 0.04) {
            Ok(m) => {
                let si = proteon_core::surface::intersect::self_intersections(&m, 1.0, 5000);
                println!(
                    "ANALYTIC: ok  verts={} tris={} watertight={} self_intersect(cap5000)={si} area={:.3} {:.1}s",
                    m.num_vertices(),
                    m.num_triangles(),
                    m.is_watertight(),
                    m.surface_area(),
                    t.elapsed().as_secs_f64(),
                );
            }
            Err(e) => println!("ANALYTIC: ERR after {:.1}s: {e}", t.elapsed().as_secs_f64()),
        }
        let tc = Instant::now();
        match proteon_core::surface::assemble::ses_mesh_cleaned(&atoms, probe, 48, 10, 0.04) {
            Ok(m) => println!(
                "CLEANED:  area={:.3} tris={} {:.1}s (cleaner active; unwelded patches)",
                m.surface_area(),
                m.num_triangles(),
                tc.elapsed().as_secs_f64(),
            ),
            Err(e) => println!("CLEANED: ERR after {:.1}s: {e}", tc.elapsed().as_secs_f64()),
        }
        // Tolerance-weld sweep: smallest eps that closes the open edges without
        // moving the area is the right one. Self-intersection scan (slow) only on
        // the last eps.
        let eps_list = [1e-4];
        let last = eps_list.len() - 1;
        for (idx, eps) in eps_list.into_iter().enumerate() {
            let tw = Instant::now();
            match proteon_core::surface::assemble::ses_mesh_cleaned_welded(
                &atoms, probe, 48, 10, 0.04, eps,
            ) {
                Ok(m) => {
                    let mut use_count: std::collections::HashMap<(u32, u32), u32> =
                        std::collections::HashMap::new();
                    for tri in &m.tris {
                        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                            *use_count.entry((a.min(b), a.max(b))).or_default() += 1;
                        }
                    }
                    let (mut e1, mut e3) = (0u32, 0u32);
                    for &c in use_count.values() {
                        match c {
                            1 => e1 += 1,
                            2 => {}
                            _ => e3 += 1,
                        }
                    }
                    let si = if idx == last {
                        proteon_core::surface::intersect::self_intersections(&m, 1.0, 5000) as i64
                    } else {
                        -1
                    };
                    println!(
                        "WELD eps={eps:.0e}: area={:.3} vol={:.3} verts={} tris={} euler={} watertight={} open={e1} nonmanifold={e3} si={si} {:.1}s",
                        m.surface_area(),
                        m.signed_volume(),
                        m.num_vertices(),
                        m.num_triangles(),
                        m.euler_characteristic(),
                        m.is_watertight(),
                        tw.elapsed().as_secs_f64(),
                    );
                }
                Err(e) => println!("WELD eps={eps:.0e}: ERR {e}"),
            }
        }
    }

    for h in spacings {
        let t = Instant::now();
        let mesh = ses_mesh_sdf(&atoms, probe, h);
        let dt = t.elapsed().as_secs_f64();
        // Edge-usage histogram: how many undirected edges are used 1× (open
        // boundary), 2× (manifold), ≥3× (non-manifold)?
        let mut use_count: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for tri in &mesh.tris {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *use_count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let (mut e1, mut e3) = (0u32, 0u32);
        for &c in use_count.values() {
            match c {
                1 => e1 += 1,
                2 => {}
                _ => e3 += 1,
            }
        }
        // Connected components (union-find over triangle vertices).
        let nv = mesh.num_vertices();
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        fn find(p: &mut [u32], x: u32) -> u32 {
            let mut r = x;
            while p[r as usize] != r {
                r = p[r as usize];
            }
            p[x as usize] = r;
            r
        }
        for tri in &mesh.tris {
            let a = find(&mut parent, tri[0]);
            let b = find(&mut parent, tri[1]);
            let c = find(&mut parent, tri[2]);
            parent[b as usize] = a;
            parent[c as usize] = a;
        }
        let mut comps = std::collections::HashSet::new();
        for tri in &mesh.tris {
            comps.insert(find(&mut parent, tri[0]));
        }
        let ncomp = comps.len();
        println!(
            "h={h:.2}  area={:.3}  volume={:.3}  verts={}  tris={}  euler={}  components={ncomp}  watertight={}  | edges 1x(open)={e1} 3x+(nonmanifold)={e3}  {:.2}s",
            mesh.surface_area(),
            mesh.signed_volume(),
            mesh.num_vertices(),
            mesh.num_triangles(),
            mesh.euler_characteristic(),
            mesh.is_watertight(),
            dt,
        );
    }
}
