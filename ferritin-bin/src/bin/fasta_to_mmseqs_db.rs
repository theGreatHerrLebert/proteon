//! Convert a FASTA file into an MMseqs2-compatible on-disk DB.
//!
//! Byte-compatible with `mmseqs createdb`: the resulting prefix
//! is directly consumable by [`SearchEngine::build_from_mmseqs_db`]
//! (see `ferritin-search::search`) and, in principle, by upstream
//! `mmseqs search` as well.
//!
//! Why this exists:
//! installing upstream `mmseqs` just to `createdb` large reference
//! corpora (UniRef50 at ~27 GB, BFD, etc.) is a real dep-management
//! burden for end users and CI. We already own a byte-level
//! [`DBWriter`] for the mmseqs DB format; wiring a FASTA iterator
//! to it is a few dozen lines and removes the external dependency.
//!
//! Usage:
//!   fasta_to_mmseqs_db <in.fasta> <out_prefix> [--max-records N]
//!
//! Output: `<out_prefix>`, `<out_prefix>.index`, `<out_prefix>.dbtype`
//! — the three files DBReader needs. Header DB (`_h`, `_h.index`,
//! `_h.dbtype`) and lookup/source auxiliaries are NOT written; they
//! aren't required for search and would double the disk footprint.
//! Add them later if a downstream tool needs them.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use ferritin_search::db::{DBWriter, Dbtype};

#[derive(Parser, Debug)]
#[command(
    name = "fasta_to_mmseqs_db",
    about = "Convert FASTA to an MMseqs2-compatible DB (amino acids).",
)]
struct Args {
    /// Input FASTA file (plain text, not gzipped).
    fasta: PathBuf,
    /// Output DB prefix. Three files will be written:
    /// `<prefix>`, `<prefix>.index`, `<prefix>.dbtype`.
    out_prefix: PathBuf,
    /// Stop after this many records (useful for subset tests).
    #[arg(long)]
    max_records: Option<usize>,
    /// Log progress every N records.
    #[arg(long, default_value_t = 100_000)]
    progress_every: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let file = File::open(&args.fasta)
        .with_context(|| format!("open FASTA {}", args.fasta.display()))?;
    let reader = BufReader::with_capacity(1 << 20, file); // 1 MiB buffer

    let mut writer = DBWriter::create(&args.out_prefix, Dbtype::AMINO_ACIDS)
        .with_context(|| format!("create DB at {}", args.out_prefix.display()))?;

    // Key assignment: sequential starting at 1 so we never emit key=0,
    // which some downstream tools treat as a sentinel. MMseqs itself
    // uses 0-based keys but our reader doesn't care.
    let mut key: u32 = 1;
    let mut current_seq: Vec<u8> = Vec::with_capacity(4096);
    let mut n_written: usize = 0;

    for line_res in reader.lines() {
        let line = line_res?;
        if line.starts_with('>') {
            // FASTA header: flush the previous record (if any).
            if !current_seq.is_empty() {
                writer.write_entry(key, &current_seq)?;
                current_seq.clear();
                key = key.checked_add(1).context("key overflow: >2^32 records")?;
                n_written += 1;
                if args.progress_every > 0 && n_written % args.progress_every == 0 {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = n_written as f64 / elapsed;
                    eprintln!(
                        "written {:>12} records  ({:>9.0} rec/s, {:>6.1} s elapsed)",
                        n_written, rate, elapsed,
                    );
                }
                if let Some(limit) = args.max_records {
                    if n_written >= limit {
                        break;
                    }
                }
            }
            // We intentionally do NOT carry the FASTA header through; the
            // headers DB (`_h`) is optional for search and we keep this
            // writer minimal.
        } else {
            // Sequence line: strip whitespace, append residues to the
            // current record. Mixed-case is preserved — the alphabet
            // encoder uppercases on read, and we don't want to edit the
            // bytes here in case a downstream tool cares about case.
            current_seq.extend(line.trim().as_bytes());
        }
    }

    // Flush the trailing record.
    if !current_seq.is_empty() {
        let under_limit = args.max_records.map_or(true, |limit| n_written < limit);
        if under_limit {
            writer.write_entry(key, &current_seq)?;
            n_written += 1;
        }
    }
    writer.finish()?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "done: {} records written to {} in {:.1}s ({:.0} rec/s)",
        n_written,
        args.out_prefix.display(),
        elapsed,
        n_written as f64 / elapsed.max(1e-9),
    );
    println!("{}", n_written);
    Ok(())
}
