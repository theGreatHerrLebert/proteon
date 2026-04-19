//! External-memory k-mer index builder: walks an mmseqs DB and writes
//! a `.kmi` file without materializing the full in-memory KmerIndex.
//!
//! Peak resident memory is bounded by the offsets table
//! (`~8 × (alphabet_size ^ k + 1)` bytes), not by `n_entries × 8`.
//! At UniRef50 scale (50M sequences, ~12 × 10^9 entries) the in-memory
//! build path needs ~100 GB just for the postings; this binary does
//! the same work in ~1.4 GB (full alphabet) or ~77 MB (reduced).
//!
//! Usage:
//!   build_kmi <db_prefix> <out.kmi> [--k 6] [--reduce-to 13]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use proteon_search::alphabet::Alphabet;
use proteon_search::db::DBReader;
use proteon_search::kmer_index_file::{build_kmi_external, BuildExternalOptions};
use proteon_search::matrix::SubstitutionMatrix;
use proteon_search::reduced_alphabet::ReducedAlphabet;

#[derive(Parser, Debug)]
#[command(name = "build_kmi", about = "External-memory .kmi builder")]
struct Args {
    /// Path to an mmseqs-compatible DB (prefix — three files live at
    /// `<prefix>`, `<prefix>.index`, `<prefix>.dbtype`).
    db_prefix: PathBuf,
    /// Output `.kmi` path.
    out_kmi: PathBuf,
    /// K-mer size; default matches the Python/Rust engine default.
    #[arg(long, default_value_t = 6)]
    k: usize,
    /// Reduced alphabet size. `None` (via --no-reduce) keeps the full
    /// 21-letter protein alphabet; table_size grows to 21^k.
    #[arg(long, default_value_t = 13)]
    reduce_to: usize,
    /// Skip alphabet reduction; k-mer index will be in the full alphabet.
    #[arg(long)]
    no_reduce: bool,
    /// Number of hash-range passes over the DB during the write phase.
    /// At K=1 the builder writes to the full output in a single scan,
    /// which thrashes the OS page cache for outputs larger than RAM.
    /// K × full-DB scans trades CPU for write locality: each pass
    /// touches 1/K of the output file. Rule of thumb: pick K so that
    /// `output_size / K ≤ ~1 GB`. At UniRef50 scale, K = 60–100 is
    /// reasonable.
    #[arg(long, default_value_t = 1)]
    hash_range_passes: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let db = DBReader::open(&args.db_prefix)
        .with_context(|| format!("open DB at {}", args.db_prefix.display()))?;
    eprintln!(
        "DB opened: {} entries, data blob = {} bytes (mmapped)",
        db.len(),
        db.data().len(),
    );

    let alphabet = Alphabet::protein();
    let reducer_owned: Option<ReducedAlphabet> = if args.no_reduce {
        None
    } else {
        let matrix = SubstitutionMatrix::blosum62();
        let x_full = alphabet.encode(b'X');
        Some(
            ReducedAlphabet::from_matrix(&matrix, args.reduce_to, Some(x_full))
                .context("ReducedAlphabet::from_matrix failed (check reduce_to and matrix)")?,
        )
    };
    let reducer = reducer_owned.as_ref();
    let effective_alphabet = reducer
        .map(|r| r.reduced_size)
        .unwrap_or_else(|| alphabet.size());
    eprintln!(
        "build params: k={} alphabet={} (reduced={})",
        args.k,
        effective_alphabet,
        reducer.is_some(),
    );
    let table_size = (effective_alphabet as u64).pow(args.k as u32);
    eprintln!(
        "table_size={} (offsets buffer ≈ {} MB resident during build)",
        table_size,
        8 * (table_size + 1) / 1024 / 1024,
    );

    eprintln!("hash_range_passes = {}", args.hash_range_passes);
    build_kmi_external(
        &db,
        alphabet,
        reducer,
        BuildExternalOptions {
            k: args.k,
            hash_range_passes: args.hash_range_passes,
        },
        &args.out_kmi,
    )
    .context("build_kmi_external failed")?;

    let elapsed = t0.elapsed().as_secs_f64();
    let out_size = std::fs::metadata(&args.out_kmi)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "done: wrote {} ({:.1} GB) in {:.1}s",
        args.out_kmi.display(),
        out_size as f64 / 1e9,
        elapsed,
    );
    println!("{}", out_size);
    Ok(())
}
