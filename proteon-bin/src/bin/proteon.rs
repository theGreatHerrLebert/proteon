//! `proteon` — unified CLI for the molecular-mechanics / analysis surface.
//!
//! Phase 1 (read-only analysis): `sasa`, `dssp`, `hbond`. These are the
//! Python-only surfaces given a command-line door. Each subcommand calls the
//! EXACT same pure-Rust entry points the Python API calls
//! (`sasa::sasa_from_pdb`, `dssp::dssp_from_pdb`, `hbond::backbone_hbonds` /
//! `geometric_hbonds`) — it is deliberately not a second implementation, so it
//! cannot drift from the Python path. Numeric defaults mirror the Python
//! signatures (`probe=1.4`, `n_points=960`, `radii="bondi"`,
//! `energy_cutoff=-0.5`, `dist_cutoff=3.5`).
//!
//! Implemented via the "Option A" path (TO_RUST_CLI.md): `proteon-bin` depends
//! on the pyo3 `extension-module` connector as an rlib and calls only the
//! pure-Rust paths. Verified to link + run on Linux including release/LTO; the
//! eventual clean home is a pyo3-free `proteon-core` crate (Option B), tracked
//! as a follow-up.
//!
//! Batch: every subcommand accepts one or more files or a directory (files
//! gathered non-recursively, sorted for determinism), fans out with rayon,
//! isolates per-file failures (bad file → stderr line, run continues), keeps
//! output in input order regardless of thread count, and exits nonzero if any
//! input failed.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rayon::prelude::*;

use proteon_connector::dssp;
use proteon_connector::hbond;
use proteon_connector::sasa;

#[derive(Parser)]
#[command(
    name = "proteon",
    about = "proteon — structure analysis & molecular-mechanics CLI",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Solvent-accessible surface area (Shrake-Rupley).
    Sasa(SasaArgs),
    /// DSSP secondary-structure assignment (Kabsch-Sander).
    Dssp(DsspArgs),
    /// Backbone (Kabsch-Sander energy) or geometric (distance) H-bonds.
    Hbond(HbondArgs),
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Format {
    Tsv,
    Json,
}

/// Inputs + batch knobs shared by every subcommand.
#[derive(Args)]
struct Common {
    /// One or more structures (PDB/mmCIF), or a single directory of them.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Tsv)]
    format: Format,
    /// Worker threads for batch (0 = all available cores).
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,
}

#[derive(Args)]
struct SasaArgs {
    #[command(flatten)]
    common: Common,
    /// Emit per-residue SASA instead of the per-structure total.
    #[arg(long)]
    per_residue: bool,
    /// Probe radius in Angstroms (water = 1.4).
    #[arg(long, default_value_t = 1.4)]
    probe: f64,
    /// Test points per sphere.
    #[arg(long = "points", default_value_t = 960)]
    n_points: usize,
    /// vdW radii set: bondi | protor (protor = naccess/freesasa).
    #[arg(long, default_value = "bondi")]
    radii: String,
}

#[derive(Args)]
struct DsspArgs {
    #[command(flatten)]
    common: Common,
}

#[derive(Args)]
struct HbondArgs {
    #[command(flatten)]
    common: Common,
    /// Use the geometric (donor-acceptor distance) criterion instead of the
    /// Kabsch-Sander backbone energy criterion.
    #[arg(long)]
    geometric: bool,
    /// Backbone energy cutoff in kcal/mol (default mode).
    #[arg(long, default_value_t = -0.5)]
    energy_cutoff: f64,
    /// Donor-acceptor distance cutoff in Angstroms (--geometric mode).
    #[arg(long, default_value_t = 3.5)]
    dist_cutoff: f64,
}

/// One output row, as ordered (column, value) pairs. The first column is always
/// `file`, so the schema is identical for single-structure and batch runs and
/// for both TSV and JSON.
type Row = Vec<(&'static str, Value)>;

enum Value {
    Str(String),
    Int(i64),
    /// f64 rendered to 4 dp in TSV; full precision in JSON.
    F64(f64),
}

/// Permissive load matching the Python connector (Loose strictness, CRYST1 /
/// MASTER parsing off) so the CLI accepts the same archive files the Python
/// path does.
fn load_pdb(path: &Path) -> Result<pdbtbx::PDB> {
    let path_str = path.to_str().context("non-UTF8 path")?;

    let mut parsing = pdbtbx::ParsingLevel::all();
    parsing.set_cryst1(false);
    parsing.set_master(false);

    let mut opts = pdbtbx::ReadOptions::new();
    opts.set_level(pdbtbx::StrictnessLevel::Loose)
        .set_parsing_level(&parsing);

    let (pdb, _warnings) = opts.read(path_str).map_err(|errs| {
        anyhow!(errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; "))
    })?;
    Ok(pdb)
}

fn parse_radii(s: &str) -> Result<sasa::RadiiSet> {
    match s.to_lowercase().as_str() {
        "protor" | "naccess" | "freesasa" => Ok(sasa::RadiiSet::ProtOr),
        "bondi" => Ok(sasa::RadiiSet::Bondi),
        other => Err(anyhow!(
            "unknown radii set '{other}'. Use 'bondi' or 'protor'."
        )),
    }
}

/// Expand inputs into a sorted file list. A single directory argument is
/// expanded to its `.pdb`/`.cif`/`.ent` children (non-recursive); otherwise the
/// inputs are taken verbatim. Sorted for deterministic output order.
fn gather_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    if inputs.len() == 1 && inputs[0].is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&inputs[0])
            .with_context(|| format!("reading directory {}", inputs[0].display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(str::to_lowercase)
                        .as_deref(),
                    Some("pdb" | "cif" | "ent")
                )
            })
            .collect();
        files.sort();
        return Ok(files);
    }
    let mut files = inputs.to_vec();
    files.sort();
    Ok(files)
}

fn run<F>(common: &Common, compute: F) -> Result<()>
where
    F: Fn(&pdbtbx::PDB) -> Vec<Row> + Sync,
{
    let files = gather_inputs(&common.inputs)?;
    if files.is_empty() {
        return Err(anyhow!("no input structures found"));
    }

    let compute_one = |path: &Path| -> std::result::Result<Vec<Row>, String> {
        let pdb = load_pdb(path).map_err(|e| e.to_string())?;
        Ok(compute(&pdb))
    };

    // Fan out with the requested thread budget; collect preserves input order.
    let results: Vec<std::result::Result<Vec<Row>, String>> = if common.threads == 1 {
        files.iter().map(|p| compute_one(p)).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(common.threads) // 0 => rayon picks all cores
            .build()
            .context("building thread pool")?;
        pool.install(|| files.par_iter().map(|p| compute_one(p)).collect())
    };

    let mut rows: Vec<(String, Row)> = Vec::new();
    let mut had_error = false;
    for (path, res) in files.iter().zip(results) {
        let file = path.to_string_lossy().into_owned();
        match res {
            Ok(file_rows) => {
                for mut row in file_rows {
                    row.insert(0, ("file", Value::Str(file.clone())));
                    rows.push((file.clone(), row));
                }
            }
            Err(e) => {
                had_error = true;
                eprintln!("ERROR {file}: {e}");
            }
        }
    }

    emit(common.format, &rows)?;
    if had_error {
        std::process::exit(1);
    }
    Ok(())
}

/// Write all rows, treating a broken downstream pipe (`… | head`) as a clean
/// exit rather than a panic.
fn emit(format: Format, rows: &[(String, Row)]) -> Result<()> {
    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    let res = match format {
        Format::Tsv => emit_tsv(&mut w, rows),
        Format::Json => emit_json(&mut w, rows),
    };
    match res.and_then(|()| w.flush()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn emit_tsv<W: Write>(w: &mut W, rows: &[(String, Row)]) -> io::Result<()> {
    let mut header_written = false;
    for (_file, row) in rows {
        if !header_written {
            let header: Vec<&str> = row.iter().map(|(k, _)| *k).collect();
            writeln!(w, "{}", header.join("\t"))?;
            header_written = true;
        }
        let cells: Vec<String> = row.iter().map(|(_, v)| v.to_tsv()).collect();
        writeln!(w, "{}", cells.join("\t"))?;
    }
    Ok(())
}

fn emit_json<W: Write>(w: &mut W, rows: &[(String, Row)]) -> io::Result<()> {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|(_file, row)| {
            let map: serde_json::Map<String, serde_json::Value> = row
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.to_json()))
                .collect();
            serde_json::Value::Object(map)
        })
        .collect();
    writeln!(w, "{}", serde_json::to_string_pretty(&arr).unwrap())
}

impl Value {
    fn to_tsv(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::F64(f) => format!("{f:.4}"),
        }
    }
    fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::Int(i) => serde_json::Value::from(*i),
            Value::F64(f) => serde_json::Value::from(*f),
        }
    }
}

fn run_sasa(args: &SasaArgs) -> Result<()> {
    if args.n_points == 0 {
        return Err(anyhow!("--points must be > 0"));
    }
    let rs = parse_radii(&args.radii)?;
    let probe = args.probe;
    let n_points = args.n_points;
    let per_residue = args.per_residue;

    run(&args.common, move |pdb| {
        let atom_areas = sasa::sasa_from_pdb(pdb, probe, n_points, rs);
        if !per_residue {
            let total: f64 = atom_areas.iter().sum();
            return vec![vec![("total_sasa", Value::F64(total))]];
        }
        // residue_sasa() walks (first model → chains → residues) in the same
        // order we re-walk here for identifiers, so the vectors align.
        let res_areas = sasa::residue_sasa(pdb, &atom_areas);
        let mut out = Vec::new();
        if let Some(model) = pdb.models().next() {
            let mut idx = 0usize;
            for chain in model.chains() {
                let chain_id = chain.id().to_string();
                for residue in chain.residues() {
                    if idx >= res_areas.len() {
                        break;
                    }
                    let (resnum, icode) = residue.id();
                    out.push(vec![
                        ("chain", Value::Str(chain_id.clone())),
                        ("resnum", Value::Int(resnum as i64)),
                        ("icode", Value::Str(icode.unwrap_or("").to_string())),
                        (
                            "resname",
                            Value::Str(residue.name().unwrap_or("").to_string()),
                        ),
                        ("sasa", Value::F64(res_areas[idx])),
                    ]);
                    idx += 1;
                }
            }
        }
        out
    })
}

fn run_dssp(args: &DsspArgs) -> Result<()> {
    // Same entry point as py_dssp::compute_dssp — parity is by construction.
    run(&args.common, |pdb| {
        vec![vec![("dssp", Value::Str(dssp::dssp_from_pdb(pdb)))]]
    })
}

fn run_hbond(args: &HbondArgs) -> Result<()> {
    let geometric = args.geometric;
    let energy_cutoff = args.energy_cutoff;
    let dist_cutoff = args.dist_cutoff;
    run(&args.common, move |pdb| {
        if geometric {
            hbond::geometric_hbonds(pdb, dist_cutoff)
                .into_iter()
                .map(|b| {
                    vec![
                        ("donor_atom", Value::Int(b.donor_atom as i64)),
                        ("acceptor_atom", Value::Int(b.acceptor_atom as i64)),
                        ("distance", Value::F64(b.distance)),
                    ]
                })
                .collect()
        } else {
            hbond::backbone_hbonds(pdb, energy_cutoff)
                .into_iter()
                .map(|b| {
                    vec![
                        ("acceptor_res", Value::Int(b.acceptor as i64)),
                        ("donor_res", Value::Int(b.donor as i64)),
                        ("energy", Value::F64(b.energy)),
                        ("dist_on", Value::F64(b.dist_on)),
                    ]
                })
                .collect()
        }
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Sasa(a) => run_sasa(a),
        Cmd::Dssp(a) => run_dssp(a),
        Cmd::Hbond(a) => run_hbond(a),
    }
}
