//! Proteon Data Engine — bulk structure → Parquet pipeline.
//!
//! Reads PDB/mmCIF files, writes per-atom Arrow/Parquet tables.
//!
//! Usage:
//!     proteon-ingest input/ --out features.parquet
//!     proteon-ingest *.pdb --out features.parquet
//!     proteon-ingest input/ --out output/ --per-structure

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;

use proteon_arrow::convert::pdb_to_atom_batch;

#[derive(Parser)]
#[command(
    name = "proteon-ingest",
    about = "Bulk protein structure → Parquet pipeline",
    long_about = "Reads PDB/mmCIF files and writes per-atom Parquet tables.\n\n\
                  Output is columnar and ready for pandas, polars, DuckDB, Spark,\n\
                  or PyTorch Geometric."
)]
struct Args {
    /// Input: directory of PDB/mmCIF files, or individual file paths.
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Output Parquet file, or directory (with --per-structure).
    #[arg(short, long, default_value = "output.parquet")]
    out: PathBuf,

    /// Write one Parquet file per input structure (output must be a directory).
    #[arg(long)]
    per_structure: bool,

    /// Maximum structures to process (0 = all).
    #[arg(short = 'n', long, default_value = "0")]
    max_structures: usize,

    /// Number of threads (0 = all available cores).
    #[arg(short = 'j', long, default_value = "0")]
    threads: usize,

    /// Flush to Parquet every N structures (single-file mode, minimum 1).
    /// Keeps memory bounded for large corpora.
    #[arg(long, default_value = "500")]
    chunk_size: usize,
}

/// Collect input file paths from args (handles directories).
fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            for entry in std::fs::read_dir(input)
                .with_context(|| format!("reading directory {}", input.display()))?
            {
                let entry = entry?;
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if matches!(ext, "pdb" | "cif" | "mmcif" | "ent") {
                        files.push(path);
                    }
                }
            }
        } else if input.exists() {
            files.push(input.clone());
        } else {
            eprintln!("Warning: {} not found, skipping", input.display());
        }
    }
    files.sort();
    Ok(files)
}

/// Derive a collision-safe structure ID from a file path.
/// Uses parent directory + stem to avoid overwrites when files
/// from different directories share the same name.
fn structure_id_from_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // If there's a parent directory, include it for uniqueness
    if let Some(parent) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|p| p.to_str())
    {
        // Only prepend parent if it's not "." or a root-level path
        if parent != "." && parent != "/" {
            return format!("{parent}/{stem}");
        }
    }
    stem.to_string()
}

/// Permissive PDB loading matching the Python connector's behavior.
/// Skips CRYST1 and MASTER parsing to maximize compatibility with
/// non-standard PDB files from the archive.
fn load_pdb_permissive(path: &Path) -> Result<pdbtbx::PDB> {
    let path_str = path.to_str().context("non-UTF8 path")?;

    let mut parsing = pdbtbx::ParsingLevel::all();
    parsing.set_cryst1(false);
    parsing.set_master(false);

    let mut opts = pdbtbx::ReadOptions::new();
    opts.set_level(pdbtbx::StrictnessLevel::Loose)
        .set_parsing_level(&parsing);

    let (pdb, _warnings) = opts.read(path_str).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::anyhow!(msg)
    })?;
    Ok(pdb)
}

fn main() -> Result<()> {
    let args = Args::parse();

    anyhow::ensure!(args.chunk_size >= 1, "--chunk-size must be at least 1");

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    // Collect input files
    let mut files = collect_inputs(&args.input)?;
    if args.max_structures > 0 && files.len() > args.max_structures {
        files.truncate(args.max_structures);
    }

    if files.is_empty() {
        eprintln!("No input files found.");
        return Ok(());
    }

    eprintln!(
        "proteon-ingest: {} structures, {} threads",
        files.len(),
        rayon::current_num_threads()
    );

    let t0 = Instant::now();
    let n_done = AtomicUsize::new(0);
    let n_failed = AtomicUsize::new(0);

    if args.per_structure {
        // One Parquet file per structure
        std::fs::create_dir_all(&args.out)
            .with_context(|| format!("creating output directory {}", args.out.display()))?;

        // Build collision-safe output names: stem, stem_2, stem_3, ...
        let mut out_names: Vec<String> = Vec::with_capacity(files.len());
        {
            let mut seen: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for path in &files {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let count = seen.entry(stem.clone()).or_insert(0);
                *count += 1;
                if *count == 1 {
                    out_names.push(stem);
                } else {
                    out_names.push(format!("{stem}_{count}"));
                }
            }
        }

        files
            .par_iter()
            .zip(out_names.par_iter())
            .for_each(|(path, out_name)| {
                let sid = structure_id_from_path(path);
                match load_and_write_single(path, &sid, &args.out, out_name) {
                    Ok(_) => {
                        let done = n_done.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 100 == 0 {
                            let elapsed = t0.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed;
                            eprintln!("  [{done}/{}] {rate:.1} structs/s", files.len());
                        }
                    }
                    Err(e) => {
                        n_failed.fetch_add(1, Ordering::Relaxed);
                        eprintln!("  SKIP {}: {e:#}", path.display());
                    }
                }
            });
    } else {
        // Streaming chunked writes — process chunk_size structures at a time,
        // flush each chunk to the Parquet writer to keep memory bounded.
        let schema = proteon_arrow::atom::atom_schema();
        let file = std::fs::File::create(&args.out)
            .with_context(|| format!("creating {}", args.out.display()))?;

        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
            .build();
        let schema_arc = std::sync::Arc::new(schema);
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema_arc, Some(props))?;

        for chunk in files.chunks(args.chunk_size) {
            let batches: Vec<_> = chunk
                .par_iter()
                .filter_map(|path| {
                    let sid = structure_id_from_path(path);
                    match load_pdb_permissive(path).and_then(|pdb| pdb_to_atom_batch(&pdb, &sid)) {
                        Ok(batch) => {
                            n_done.fetch_add(1, Ordering::Relaxed);
                            Some(batch)
                        }
                        Err(e) => {
                            n_failed.fetch_add(1, Ordering::Relaxed);
                            eprintln!("  SKIP {}: {e:#}", path.display());
                            None
                        }
                    }
                })
                .collect();

            // Flush this chunk to disk
            for batch in &batches {
                writer.write(batch)?;
            }
            writer.flush()?;

            let done = n_done.load(Ordering::Relaxed);
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            eprintln!("  [{done}/{}] {rate:.1} structs/s (flushed)", files.len());
        }

        writer.close()?;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let done = n_done.load(Ordering::Relaxed);
    let failed = n_failed.load(Ordering::Relaxed);
    eprintln!(
        "Done: {done} structures in {elapsed:.1}s ({:.1} structs/s), {failed} failed",
        done as f64 / elapsed
    );
    if !args.per_structure {
        let size = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "Output: {} ({:.1} MB)",
            args.out.display(),
            size as f64 / 1e6
        );
    }

    Ok(())
}

fn load_and_write_single(path: &Path, sid: &str, out_dir: &Path, out_name: &str) -> Result<usize> {
    let pdb = load_pdb_permissive(path)?;
    let total_atoms: usize = pdb.models().map(|m| m.atom_count()).sum();
    let batch = pdb_to_atom_batch(&pdb, sid)?;
    let schema = proteon_arrow::atom::atom_schema();
    let out_path = out_dir.join(format!("{out_name}.parquet"));
    proteon_arrow::writer::write_parquet(&out_path, &schema, &[batch])?;
    Ok(total_atoms)
}
