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
        let mut trebuild = f64::INFINITY;
        for _ in 0..n_reps {
            tmv = tmv.min(time_ms(|| tree.matvec(&x, &mut y_tree)));
            trebuild = trebuild.min(time_ms(|| tree.bench_rebuild(&x)));
        }
        let (near, far) = tree.traversal_counts();
        let mut t_near = f64::INFINITY;
        let mut t_far = f64::INFINITY;
        for _ in 0..n_reps {
            t_near = t_near.min(time_ms(|| tree.bench_near_only(&x)));
            t_far = t_far.min(time_ms(|| tree.bench_far_only(&x)));
        }
        // NOTE: these are ISOLATED timings, not an additive decomposition of `matvec` —
        // bench_near_only/bench_far_only each re-pay allocation + parallel scheduling +
        // tree-walk overhead, and bench_far_only includes the moment rebuild. They are a
        // valid relative comparison of the near vs far traversal cost (which dominates),
        // not a partition that sums to `matvec`.
        let t_far_pure = (t_far - trebuild).max(0.0);
        eprintln!(
            "  [N={n}] matvec {tmv:.1}ms | isolated: rebuild {trebuild:.1}, near-only {t_near:.1}, \
             far-only(−rebuild) {t_far_pure:.1}ms | counts {near} near / {far} far \
             | far ≳ {:.0}% of traversal cost",
            100.0 * t_far_pure / (t_near + t_far_pure).max(1e-9)
        );

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
        "\nNote: the M2M upward pass makes the moment rebuild linear — but the breakdown\n\
         above shows it is only ~12–14% of the matvec; the rest is TRAVERSAL (per-target\n\
         far-field Taylor evals + exact near-field collocations). Each treecode pair costs\n\
         far more than dense's single FMA, so dense wins on speed wherever its O(N²)\n\
         matrix still fits (capped ~{dense_cap_n} triangles here). The treecode's realized\n\
         win is O(N) MEMORY — it solves meshes dense cannot hold (350 ms/matvec at 20k,\n\
         where dense needs 6.7 GiB). A speed crossover needs traversal-constant work\n\
         (cheaper kernel eval / level-batching / the FMM L2L downward pass), not more M2M."
    );
}
