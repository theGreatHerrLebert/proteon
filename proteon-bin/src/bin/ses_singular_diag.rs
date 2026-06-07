//! Diagnostic: on a real protein, how much of the SES self-intersection is
//! *spindle* (radial: one toric face's roll circle < probe) vs *probe-probe*
//! (nonradial: two distinct fixed probes < 2·probe apart)? Codex flagged that
//! `probe_clear` excludes only atom overlap, not probe overlap — so the crambin
//! excess may be dominated by the nonradial class, which the spindle clip cannot
//! touch. This measures the split before we commit to a resolver scope.
//!
//!   cargo run --release --bin ses_singular_diag -- test-pdbs/1crn.pdb

use proteon_core::sasa::{vdw_radius, DEFAULT_RADIUS};
use proteon_core::surface::cleaner::{clip_spheric_face, toric_trim_mesh};
use proteon_core::surface::elements::ses_vertex;
use proteon_core::surface::geom::{intersect_two_spheres, plane_basis, Sphere, Vec3};
use proteon_core::surface::graph::{build_graph, enumerate_rs_faces, enumerate_toric_faces};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-pdbs/1crn.pdb".into());
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

    // --- Spindle (radial) class: toric faces with roll-circle radius < probe.
    let toric = enumerate_toric_faces(&atoms, probe).expect("toric faces");
    let mut singular = 0usize;
    let mut min_roll = f64::INFINITY;
    for tf in &toric {
        let (i, j) = (tf.edge[0], tf.edge[1]);
        if let Some(roll) =
            intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
        {
            min_roll = min_roll.min(roll.radius);
            if roll.radius < probe {
                singular += 1;
            }
        }
    }
    println!(
        "SPINDLE: {singular}/{} toric faces singular (R_roll < {probe}); min R_roll = {min_roll:.3}",
        toric.len()
    );

    // --- Probe-probe (nonradial) class: distinct fixed probes < 2·probe apart.
    let rs = enumerate_rs_faces(&atoms, probe);
    let centers: Vec<Vec3> = rs.iter().map(|f| f.probe).collect();
    let lim = (2.0 * probe) * (2.0 * probe);
    let mut dup = 0usize; // near-coincident (duplicate placement, dist < 0.05)
    let mut genuine = 0usize; // genuine overlap (0.05 ≤ dist < 2·probe)
    for a in 0..centers.len() {
        for b in (a + 1)..centers.len() {
            let d2 = centers[a].square_distance(centers[b]);
            if d2 < lim - 1e-9 {
                if d2 < 0.05 * 0.05 {
                    dup += 1;
                } else {
                    genuine += 1;
                }
            }
        }
    }
    println!(
        "PROBE-PROBE: {genuine} genuine overlaps + {dup} near-coincident duplicates (< 2·probe={:.2}, of {} RS faces)",
        2.0 * probe,
        rs.len()
    );
    let close_pairs = genuine;
    println!(
        "VERDICT: spindle faces={singular}, probe-probe overlaps={close_pairs} -> {}",
        if close_pairs > singular {
            "NONRADIAL dominates: arrangement rewrite is the real work"
        } else if singular > 0 {
            "spindle dominates: free-edge clip may close most of the gap"
        } else {
            "neither: excess is elsewhere"
        }
    );

    // --- How much area does the SPHERIC clip remove? (Diagnostic, per codex: a
    // spheric-only result is not watertight, but per-face area is meaningful.)
    // Sum, over RS faces with a colliding neighbour, plain spheric area − clipped.
    let g = build_graph(&atoms, probe).expect("graph");
    let probe_centers: Vec<Vec3> = g.rs_faces.iter().map(|f| f.probe).collect();
    let (mut plain_sum, mut clip_sum) = (0.0f64, 0.0f64);
    let (mut clipped_faces, mut clip_err) = (0usize, 0usize);
    for (idx, f) in g.rs_faces.iter().enumerate() {
        let cs = [
            ses_vertex(f.probe, atoms[f.atoms[0]]),
            ses_vertex(f.probe, atoms[f.atoms[1]]),
            ses_vertex(f.probe, atoms[f.atoms[2]]),
        ];
        // Plain spheric face = clip with only itself (no neighbours).
        let plain = clip_spheric_face(f.probe, cs, &[f.probe], 0, probe, 0.1, 8);
        let clipped = clip_spheric_face(f.probe, cs, &probe_centers, idx, probe, 0.1, 8);
        match (plain, clipped) {
            (Ok(p), Ok(c)) => {
                let (pa, ca) = (p.surface_area(), c.surface_area());
                plain_sum += pa;
                clip_sum += ca;
                if pa - ca > 1e-6 {
                    clipped_faces += 1;
                }
            }
            _ => clip_err += 1,
        }
    }
    println!(
        "SPHERIC-CLIP: removed {:.3} A^2 over {clipped_faces} faces ({clip_err} clip errors); plain={plain_sum:.1} clipped={clip_sum:.1}",
        plain_sum - clip_sum
    );

    // --- How much area does the TORIC trim remove? Same idea, over toric arcs:
    // full toric patch (no neighbours) vs trimmed (all fixed probes except this
    // arc's own end faces).
    let (n_theta, n_phi) = (24usize, 8usize);
    let (mut t_plain, mut t_clip) = (0.0f64, 0.0f64);
    let (mut t_trimmed, mut t_err) = (0usize, 0usize);
    for arc in &g.toric {
        let [i, j] = arc.edge;
        let Some(roll) = intersect_two_spheres(atoms[i].inflated(probe), atoms[j].inflated(probe))
        else {
            continue;
        };
        let (u, v) = plane_basis(roll.normal);
        let wrap = arc.end_faces[0].is_none();
        let (s, e) = arc.theta;
        let count = if wrap { n_theta } else { n_theta + 1 };
        let (mut cen, mut ra, mut rb) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..count {
            let p = if !wrap && t == 0 {
                g.rs_faces[arc.end_faces[0].unwrap()].probe
            } else if !wrap && t == n_theta {
                g.rs_faces[arc.end_faces[1].unwrap()].probe
            } else {
                let th = s + (e - s) * t as f64 / n_theta as f64;
                roll.center + (u * th.cos() + v * th.sin()) * roll.radius
            };
            cen.push(p);
            ra.push(ses_vertex(p, atoms[i]));
            rb.push(ses_vertex(p, atoms[j]));
        }
        // Neighbours = all fixed probes except this arc's own end faces.
        let ends: Vec<usize> = arc.end_faces.iter().flatten().copied().collect();
        let nbrs: Vec<Vec3> = probe_centers
            .iter()
            .enumerate()
            .filter(|(k, _)| !ends.contains(k))
            .map(|(_, &c)| c)
            .collect();
        match (
            toric_trim_mesh(&cen, &ra, &rb, probe, &[], None, wrap, n_phi),
            toric_trim_mesh(&cen, &ra, &rb, probe, &nbrs, Some(roll), wrap, n_phi),
        ) {
            (Ok(f), Ok(c)) => {
                let (fa, ca) = (f.surface_area(), c.surface_area());
                t_plain += fa;
                t_clip += ca;
                if fa - ca > 1e-6 {
                    t_trimmed += 1;
                }
            }
            _ => t_err += 1,
        }
    }
    println!(
        "TORIC-TRIM: removed {:.3} A^2 over {t_trimmed} arcs ({t_err} errors); plain={t_plain:.1} trimmed={t_clip:.1}",
        t_plain - t_clip
    );
    println!(
        "CLEANER TOTAL removed = {:.3} A^2 (spheric {:.3} + toric {:.3}); crambin excess vs BALL ~26",
        (plain_sum - clip_sum) + (t_plain - t_clip),
        plain_sum - clip_sum,
        t_plain - t_clip
    );
}
