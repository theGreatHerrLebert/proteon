//! Isolated CPU-vs-GPU benchmark for the diagonal-voting k-mer prefilter.
//!
//! Unlike `cpu_vs_gpu_bench` (which times the whole `SearchEngine::search`
//! path and so conflates prefilter + ungapped + SW), this benches ONLY the
//! prefilter: CPU [`diagonal_prefilter`] vs the resident GPU
//! [`GpuPrefilterIndex::prefilter_batch`], on a synthetic reduced-alphabet
//! (13-letter) corpus indexed at k=6 — the configuration `SearchEngine` uses
//! by default. It reports the one-time GPU upload cost separately from steady-
//! state query throughput, and verifies GPU results are bit-exact vs CPU.
//!
//! Run with:
//!   cargo run --release -p proteon-search --features cuda \
//!       --example prefilter_bench -- <n_targets> <n_queries> <len_min> <len_max>
//!
//! Defaults: 5000 targets, 1000 queries, lengths 80..=320, alphabet 13, k=6.
//! A deterministic PRNG (seed 0xC0FF_EE42) makes runs comparable.

use std::time::Instant;

use proteon_search::kmer::{KmerEncoder, KmerIndex};
use proteon_search::prefilter::{diagonal_prefilter, PrefilterOptions};

fn parse_arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_targets: usize = parse_arg(&args, 1, 5000);
    let n_queries: usize = parse_arg(&args, 2, 1000);
    let len_min: usize = parse_arg(&args, 3, 80);
    let len_max: usize = parse_arg(&args, 4, 320);
    let alphabet_size: u32 = 13;
    let k: usize = 6;
    // Out-of-range "skip" index: valid symbols are 0..=12, so 13 never
    // appears and no k-mer window is ever skipped.
    let skip_idx: u8 = 13;

    #[cfg(feature = "cuda")]
    let gpu_present = proteon_search::gpu::is_available();
    #[cfg(not(feature = "cuda"))]
    let gpu_present = false;

    eprintln!(
        "[bench] n_targets={n_targets} n_queries={n_queries} len={len_min}..={len_max} \
         alphabet={alphabet_size} k={k} cuda_feature={} gpu_detected={gpu_present}",
        cfg!(feature = "cuda"),
    );

    // Deterministic xorshift PRNG.
    let mut rng: u32 = 0xC0FF_EE42;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };

    // Synthetic corpus: each sequence is alphabet indices in 0..alphabet_size.
    let targets: Vec<Vec<u8>> = (0..n_targets)
        .map(|_| {
            let span = len_max - len_min + 1;
            let len = len_min + (next() as usize) % span;
            (0..len).map(|_| (next() % alphabet_size) as u8).collect()
        })
        .collect();

    // Queries are random existing targets (guarantees self + homolog hits).
    let queries: Vec<Vec<u8>> = (0..n_queries)
        .map(|_| targets[(next() as usize) % n_targets].clone())
        .collect();

    let encoder = KmerEncoder::new(alphabet_size, k);
    let t_idx = Instant::now();
    let index = KmerIndex::build(
        encoder,
        targets
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_slice())),
        skip_idx,
    )
    .expect("index build");
    eprintln!(
        "[bench] index built in {:.3}s  table_size={}  total_hits={}  distinct_kmers={}",
        t_idx.elapsed().as_secs_f64(),
        index.encoder.table_size(),
        index.total_hits(),
        index.distinct_kmers(),
    );

    let opts = PrefilterOptions::default();

    // ---- CPU path ----
    // Warm up (page in, branch predictor) on the first query, untimed.
    let _ = diagonal_prefilter(&index, &queries[0], skip_idx, &opts);
    let t_cpu = Instant::now();
    let mut cpu_results: Vec<Vec<_>> = Vec::with_capacity(n_queries);
    for q in &queries {
        cpu_results.push(diagonal_prefilter(&index, q, skip_idx, &opts));
    }
    let cpu_elapsed = t_cpu.elapsed().as_secs_f64();
    let cpu_hits: usize = cpu_results.iter().map(Vec::len).sum();
    let cpu_qps = n_queries as f64 / cpu_elapsed;
    println!("=== diagonal prefilter: CPU vs GPU ===");
    println!("corpus:        {n_targets} targets, {n_queries} queries");
    println!(
        "cpu:           {cpu_elapsed:.3} s   ({cpu_qps:.1} queries/s)   total_hits={cpu_hits}"
    );

    #[cfg(feature = "cuda")]
    {
        use proteon_search::gpu::prefilter::GpuPrefilterIndex;

        if !gpu_present {
            eprintln!("[bench] cuda feature on but no GPU detected — skipping GPU path");
            return;
        }

        // One-time upload, timed separately (it is amortized across queries).
        let t_up = Instant::now();
        let gpu = GpuPrefilterIndex::upload(&index).expect("gpu upload");
        let upload_elapsed = t_up.elapsed().as_secs_f64();

        let query_slices: Vec<&[u8]> = queries.iter().map(Vec::as_slice).collect();

        // Warm up: first launch pays NVRTC compile + module load, untimed.
        let _ = gpu
            .prefilter_batch(&query_slices[..1], skip_idx, &opts)
            .expect("gpu warmup");

        let t_gpu = Instant::now();
        let gpu_results = gpu
            .prefilter_batch(&query_slices, skip_idx, &opts)
            .expect("gpu prefilter_batch");
        let gpu_elapsed = t_gpu.elapsed().as_secs_f64();
        let gpu_hits: usize = gpu_results.iter().map(Vec::len).sum();
        let gpu_qps = n_queries as f64 / gpu_elapsed;

        println!(
            "gpu upload:    {upload_elapsed:.3} s   (one-time, {} resident entries)",
            index.total_hits()
        );
        println!(
            "gpu:           {gpu_elapsed:.3} s   ({gpu_qps:.1} queries/s)   total_hits={gpu_hits}"
        );
        println!(
            "gpu speedup:   {:.2}x  (steady-state, upload excluded)",
            cpu_elapsed / gpu_elapsed
        );
        println!(
            "gpu speedup:   {:.2}x  (amortized, upload included)",
            cpu_elapsed / (gpu_elapsed + upload_elapsed)
        );

        // ---- Single-query path (what SearchEngine::search actually does) ----
        // search() calls prefilter() once per query, each making its OWN scratch
        // (no batch amortization). This is the crossover that gates the search()
        // wiring, NOT the batched number above.
        let _ = gpu
            .prefilter(query_slices[0], skip_idx, &opts)
            .expect("gpu single warmup");
        let t_sq = Instant::now();
        let mut sq_hits = 0usize;
        for q in &query_slices {
            sq_hits += gpu.prefilter(q, skip_idx, &opts).expect("gpu single").len();
        }
        let sq_elapsed = t_sq.elapsed().as_secs_f64();
        let sq_qps = n_queries as f64 / sq_elapsed;
        println!(
            "gpu single:    {sq_elapsed:.3} s   ({sq_qps:.1} queries/s)   total_hits={sq_hits}   \
             speedup={:.2}x  (fresh scratch/query = search() path)",
            cpu_elapsed / sq_elapsed
        );

        // ---- Cached single-query path (search() with a REUSED scratch) ----
        // What search() does once we cache one PrefilterScratch on the engine:
        // single-query calls, but the device buffers + stream are reused across
        // calls. This is the crossover that the search() wiring should gate on.
        use proteon_search::gpu::prefilter::PrefilterScratch;
        let mut scratch = PrefilterScratch::new().expect("scratch");
        let _ = gpu
            .prefilter_with(&mut scratch, query_slices[0], skip_idx, &opts)
            .expect("gpu cached warmup");
        let t_cs = Instant::now();
        let mut cs_hits = 0usize;
        for q in &query_slices {
            cs_hits += gpu
                .prefilter_with(&mut scratch, q, skip_idx, &opts)
                .expect("gpu cached single")
                .len();
        }
        let cs_elapsed = t_cs.elapsed().as_secs_f64();
        let cs_qps = n_queries as f64 / cs_elapsed;
        println!(
            "gpu cached:    {cs_elapsed:.3} s   ({cs_qps:.1} queries/s)   total_hits={cs_hits}   \
             speedup={:.2}x  (reused scratch/query = wired search() path)",
            cpu_elapsed / cs_elapsed
        );

        // ---- Bit-exact verification ----
        let mut mismatches = 0usize;
        for (qi, (c, g)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            if c != g {
                mismatches += 1;
                if mismatches <= 3 {
                    eprintln!(
                        "[bench] MISMATCH query {qi}: cpu={} hits, gpu={} hits",
                        c.len(),
                        g.len()
                    );
                }
            }
        }
        if mismatches == 0 {
            println!("verify:        OK — GPU bit-exact vs CPU on all {n_queries} queries");
        } else {
            eprintln!("[bench] FAIL: {mismatches}/{n_queries} queries differ");
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = cpu_results;
        eprintln!("[bench] built without --features cuda — CPU path only");
    }
}
