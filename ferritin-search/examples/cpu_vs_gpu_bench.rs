//! CPU vs GPU throughput bench for the SearchEngine dispatch path.
//!
//! Builds an engine over a randomly generated protein corpus, runs a
//! fixed set of queries twice (use_gpu = false, then true), and prints
//! per-path wall time plus speedup. Uses a deterministic PRNG so two
//! invocations on the same host are directly comparable.
//!
//! Run with:
//!   cargo run --release -p ferritin-search --features cuda \
//!       --example cpu_vs_gpu_bench -- <n_targets> <n_queries> <target_len_min> <target_len_max>
//!
//! Defaults: 5000 targets, 50 queries, lengths 80-320.

use std::time::Instant;

use ferritin_search::alphabet::Alphabet;
use ferritin_search::matrix::SubstitutionMatrix;
use ferritin_search::search::{SearchEngine, SearchOptions};
use ferritin_search::sequence::Sequence;

fn parse_arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_targets: usize = parse_arg(&args, 1, 5000);
    let n_queries: usize = parse_arg(&args, 2, 50);
    let t_min: usize = parse_arg(&args, 3, 80);
    let t_max: usize = parse_arg(&args, 4, 320);

    #[cfg(feature = "cuda")]
    let gpu_present = ferritin_search::gpu::is_available();
    #[cfg(not(feature = "cuda"))]
    let gpu_present = false;

    eprintln!(
        "[bench] n_targets={n_targets}, n_queries={n_queries}, target_len={t_min}..={t_max}, cuda_feature={}, gpu_detected={}",
        cfg!(feature = "cuda"),
        gpu_present,
    );

    let protein_bytes: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

    let mut rng: u32 = 0xC0FF_EE42;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };

    let targets: Vec<(u32, Sequence)> = (0..n_targets)
        .map(|id| {
            let len = t_min + (next() as usize) % (t_max - t_min + 1);
            let bytes: Vec<u8> = (0..len)
                .map(|_| protein_bytes[(next() as usize) % protein_bytes.len()])
                .collect();
            let alpha = Alphabet::protein();
            (id as u32, Sequence::from_ascii(alpha, &bytes))
        })
        .collect();

    let queries: Vec<Sequence> = (0..n_queries)
        .map(|_| {
            let pick = (next() as usize) % n_targets;
            targets[pick].1.clone()
        })
        .collect();

    let alpha = Alphabet::protein();
    let matrix = SubstitutionMatrix::blosum62();

    let prefilter_cap = env_usize("BENCH_PREFILTER_CAP", 1000);
    let result_cap = env_usize("BENCH_RESULT_CAP", 50);
    let min_score = env_usize("BENCH_MIN_SCORE", 0) as i32;
    eprintln!(
        "[bench] prefilter_cap={prefilter_cap}, result_cap={result_cap}, min_score={min_score}"
    );

    let make_opts = |use_gpu: bool| SearchOptions {
        k: 6,
        reduce_to: Some(13),
        bit_factor: 2.0,
        diagonal_score_threshold: 0,
        max_prefilter_hits: Some(prefilter_cap),
        gap_open: -11,
        gap_extend: -1,
        min_score,
        max_results: Some(result_cap),
        use_gpu,
    };

    let t_build = Instant::now();
    let engine_cpu = SearchEngine::build(targets.clone(), &matrix, alpha.clone(), make_opts(false))
        .expect("cpu engine build");
    let engine_gpu = SearchEngine::build(targets.clone(), &matrix, alpha.clone(), make_opts(true))
        .expect("gpu engine build");
    eprintln!("[bench] engines built in {:.2}s", t_build.elapsed().as_secs_f64());

    // Warm up each path — in particular, first GPU call pays NVRTC
    // compile cost. Warmup excluded from timing.
    let _ = engine_cpu.search(&queries[0]);
    let _ = engine_gpu.search(&queries[0]);

    let t_cpu = Instant::now();
    let mut cpu_hit_count: usize = 0;
    for q in &queries {
        let hits = engine_cpu.search(q);
        cpu_hit_count += hits.len();
    }
    let cpu_elapsed = t_cpu.elapsed().as_secs_f64();

    let t_gpu = Instant::now();
    let mut gpu_hit_count: usize = 0;
    for q in &queries {
        let hits = engine_gpu.search(q);
        gpu_hit_count += hits.len();
    }
    let gpu_elapsed = t_gpu.elapsed().as_secs_f64();

    let cpu_qps = n_queries as f64 / cpu_elapsed;
    let gpu_qps = n_queries as f64 / gpu_elapsed;
    let speedup = cpu_elapsed / gpu_elapsed;

    println!("=== SearchEngine::search() CPU vs GPU ===");
    println!("corpus:          {n_targets} targets");
    println!("queries:         {n_queries}");
    println!("cpu total:       {cpu_elapsed:.3} s   ({cpu_qps:.1} queries/s)   hits={cpu_hit_count}");
    println!("gpu total:       {gpu_elapsed:.3} s   ({gpu_qps:.1} queries/s)   hits={gpu_hit_count}");
    println!("gpu speedup:     {speedup:.2}x");

    if cpu_hit_count != gpu_hit_count {
        eprintln!(
            "[bench] WARNING: hit count differs: cpu={cpu_hit_count} gpu={gpu_hit_count}"
        );
    }
}
