//! P8.3 scaling benchmark — dense O(N²) vs treecode matvec, measured (plan §5).
//!
//! Honest characterization of the v1 treecode: it has the exact analytic near field and
//! a panel-aware far field, but the per-matvec moment rebuild is the direct
//! `O(N·depth·p³)` (the M2M upward pass that would make it linear is deferred). This
//! reports build + matvec wall-clock and the matvec accuracy across a sphere-mesh ladder,
//! so the crossover N (where the treecode beats dense) — or the lack of one at these
//! sizes, which motivates the M2M follow-up — is a measured fact, not a claim.
//!
//! Run: `cargo run --release --example p8_scaling`

use std::time::Instant;

use proteon_electrostatics::system::{laplace_matrices, LinearOperator};
use proteon_electrostatics::fastsum::operator::CollocationTreecode;
use proteon_electrostatics::{analytic_sphere_mesh, PotentialKind, Tri};

fn sphere_elements(subdiv: u32) -> Vec<Tri> {
    let mesh = analytic_sphere_mesh(2.0, subdiv);
    mesh.tris
        .iter()
        .map(|t| {
            Tri::new(
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
            )
        })
        .collect()
}

fn time_ms(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|q| q * q).sum::<f64>().sqrt().max(1e-300);
    num / den
}

fn main() {
    let (p, theta) = (6usize, 0.5_f64);
    let n_reps = 5;
    // Dense K is 2·N²·8 bytes; cap the dense side so it stays in memory.
    let dense_cap_n = 12_000usize;

    println!("# P8.3 scaling: dense vs treecode matvec (double-layer K, p={p}, θ={theta})");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>9}",
        "N", "dense_bld", "dense_mv", "tree_bld", "tree_mv", "mv_relL2", "speedup"
    );

    for subdiv in 2..=5u32 {
        let els = sphere_elements(subdiv);
        let n = els.len();
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();

        // Treecode (always).
        let mut tree = None;
        let tree_bld = time_ms(|| {
            tree = Some(CollocationTreecode::new(&els, PotentialKind::Double, p, theta));
        });
        let tree = tree.unwrap();
        let mut y_tree = vec![0.0; n];
        let mut tmv = f64::INFINITY;
        for _ in 0..n_reps {
            tmv = tmv.min(time_ms(|| tree.matvec(&x, &mut y_tree)));
        }

        if n <= dense_cap_n {
            let mut dense = None;
            let dense_bld = time_ms(|| {
                dense = Some(laplace_matrices(&els).1); // K
            });
            let dense = dense.unwrap();
            let mut y_dense = vec![0.0; n];
            let mut dmv = f64::INFINITY;
            for _ in 0..n_reps {
                dmv = dmv.min(time_ms(|| dense.matvec(&x, &mut y_dense)));
            }
            let err = rel_l2(&y_tree, &y_dense);
            println!(
                "{n:>8} {dense_bld:>11.1}ms {dmv:>11.3}ms {tree_bld:>11.1}ms {tmv:>11.3}ms {err:>10.2e} {:>8.2}x",
                dmv / tmv
            );
        } else {
            println!(
                "{n:>8} {:>13} {:>13} {tree_bld:>11.1}ms {tmv:>11.3}ms {:>10} {:>9}",
                "(over cap)", "(over cap)", "—", "—"
            );
        }
    }

    println!(
        "\nNote: speedup < 1 means the v1 direct moment rebuild (O(N·depth·p³)) is slower\n\
         per matvec than the tight dense O(N²) loop at this N; the treecode's win is\n\
         O(N) MEMORY (dense is capped out past ~{dense_cap_n} triangles) and the matvec\n\
         crossover moves in once the M2M upward pass replaces the direct rebuild."
    );
}
