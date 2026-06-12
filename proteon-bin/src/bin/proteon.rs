//! `proteon` — unified CLI for the molecular-mechanics / analysis surface.
//!
//! Read-only analysis: `sasa`, `dssp`, `hbond`, `energy`. Write commands
//! (emit prepared structures): `prepare`, `protonate`, `minimize`. These are
//! the Python-only surfaces given a command-line door. Each subcommand calls
//! the EXACT same pure-Rust entry points the Python API calls
//! (`sasa::sasa_from_pdb`, `dssp::dssp_from_pdb`, `hbond::backbone_hbonds` /
//! `geometric_hbonds`, `forcefield::api::energy_from_pdb`,
//! `prepare::prepare_from_pdb`) — deliberately not a second implementation, so
//! it cannot drift from the Python path. Numeric defaults mirror the Python
//! signatures (`probe=1.4`, `n_points=960`, `radii="bondi"`,
//! `energy_cutoff=-0.5`, `dist_cutoff=3.5`; energy defaults to `charmm19_eef1`
//! in kJ/mol like `proteon.compute_energy`; the write commands are presets over
//! the `batch_prepare` pipeline).
//!
//! Write-command I/O contract: see TO_RUST_CLI.md. In brief — first model only;
//! highest-occupancy altloc; missing heavy *atoms* reconstructed, missing
//! *residues* never fabricated; non-protein residues pass through but are not
//! parameterized (>50% non-water-unassigned ⇒ skipped_no_protein, minimize
//! skipped); output format follows the `-o`/`--out-dir` path extension.
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

use proteon_core::dssp;
use proteon_core::forcefield::api as ff_api;
use proteon_core::hbond;
use proteon_core::prepare::{self, PrepareOptions, PrepareReport};
use proteon_core::sasa;
use proteon_electrostatics as electro;

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
    /// Force-field potential energy + per-term breakdown (read-only).
    Energy(EnergyArgs),
    /// Full preparation: reconstruct missing atoms, place H, minimize (writes structures).
    Prepare(PrepareArgs),
    /// Place hydrogens only (writes structures).
    Protonate(ProtonateArgs),
    /// Energy-minimize the input as-is (writes structures).
    Minimize(MinimizeArgs),
    /// Continuum-electrostatics BEM solve on a surface mesh + point charges (NESSie port).
    Electrostatics(ElectrostaticsArgs),
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

#[derive(Args)]
struct EnergyArgs {
    #[command(flatten)]
    common: Common,
    /// Force field: charmm19_eef1 | amber96 | amber96_obc. Defaults to
    /// charmm19_eef1 (proteon's production force field; united-atom, so it
    /// evaluates on heavy-atom-only inputs — amber96 expects placed hydrogens,
    /// e.g. via `proteon prepare`).
    #[arg(long, default_value = "charmm19_eef1")]
    ff: String,
    /// Override the nonbonded cutoff distance in Angstroms (default: the force
    /// field's own value). Mainly for cross-tool oracle comparison.
    #[arg(long)]
    nonbonded_cutoff: Option<f64>,
    /// Energy units: kJ/mol (default, matches proteon.compute_energy) or
    /// kcal/mol (the force field's internal unit).
    #[arg(long, default_value = "kJ/mol")]
    units: String,
}

/// Output destination + force field shared by the write commands. Exactly one
/// of `-o`/`--out-dir` is required; `-o` is single-input only.
#[derive(Args)]
struct WriteCommon {
    #[command(flatten)]
    common: Common,
    /// Write the single prepared structure to this file (single input only).
    #[arg(short = 'o', long = "output")]
    out_file: Option<PathBuf>,
    /// Write each prepared structure into this directory, keeping its filename.
    #[arg(long = "out-dir")]
    out_dir: Option<PathBuf>,
    /// Force field for topology/minimization: charmm19_eef1 (default) or amber96.
    #[arg(long, default_value = "charmm19_eef1")]
    ff: String,
}

#[derive(Args)]
struct PrepareArgs {
    #[command(flatten)]
    write: WriteCommon,
    /// Hydrogen placement: all (default) | backbone | general | none.
    #[arg(long, default_value = "all")]
    hydrogens: String,
    /// Minimizer: lbfgs (default) | cg | sd.
    #[arg(long = "method", default_value = "lbfgs")]
    minimize_method: String,
    /// Maximum minimizer steps.
    #[arg(long, default_value_t = 500)]
    minimize_steps: usize,
    /// Skip the minimization stage (reconstruct + protonate only).
    #[arg(long)]
    no_minimize: bool,
    /// Skip reconstruction of missing heavy atoms.
    #[arg(long)]
    no_reconstruct: bool,
}

#[derive(Args)]
struct ProtonateArgs {
    #[command(flatten)]
    write: WriteCommon,
    /// Hydrogen placement: all (default) | backbone | general.
    #[arg(long, default_value = "all")]
    hydrogens: String,
    /// Keep pre-existing hydrogens instead of stripping and re-placing.
    #[arg(long)]
    keep_existing_h: bool,
}

#[derive(Args)]
struct MinimizeArgs {
    #[command(flatten)]
    write: WriteCommon,
    /// Minimizer: lbfgs (default) | cg | sd.
    #[arg(long = "method", default_value = "lbfgs")]
    minimize_method: String,
    /// Maximum minimizer steps.
    #[arg(long, default_value_t = 500)]
    minimize_steps: usize,
}

/// Continuum-electrostatics solve. Unlike the other subcommands this is not a
/// PDB-batch command: it takes a surface mesh + a charge set (from NESSie-style
/// format files) and emits one summary row, optionally writing the per-vertex
/// potential. Input is one of: `--hmo` (mesh + charges in one file), `--off` +
/// `--pqr`, or `--msms PREFIX` (reads `PREFIX.vert`/`PREFIX.face`) + `--pqr`.
#[derive(Args)]
struct ElectrostaticsArgs {
    /// OFF surface mesh (requires --pqr for charges).
    #[arg(long, conflicts_with_all = ["hmo", "msms"])]
    off: Option<PathBuf>,
    /// MSMS mesh prefix; reads <prefix>.vert and <prefix>.face (requires --pqr).
    #[arg(long, conflicts_with_all = ["hmo", "off"])]
    msms: Option<PathBuf>,
    /// HMO file providing BOTH the mesh and the charges.
    #[arg(long, conflicts_with_all = ["off", "msms", "pqr"])]
    hmo: Option<PathBuf>,
    /// PQR charge set (used with --off / --msms).
    #[arg(long)]
    pqr: Option<PathBuf>,
    /// Solute (interior) dielectric.
    #[arg(long, default_value_t = 1.0)]
    eps_omega: f64,
    /// Solvent (bulk) dielectric.
    #[arg(long, default_value_t = 78.0)]
    eps_sigma: f64,
    /// High-frequency (optical) dielectric — nonlocal only.
    #[arg(long, default_value_t = 1.8)]
    eps_inf: f64,
    /// Solvent correlation length in Angstroms — nonlocal only.
    #[arg(long = "lambda", default_value_t = 20.0)]
    lambda: f64,
    /// Nonlocal (Lorentz/Yukawa) solve instead of local Poisson.
    #[arg(long)]
    nonlocal: bool,
    /// Regular-Yukawa cubature for the nonlocal solve: fixed (fast) | adaptive (accurate).
    #[arg(long, default_value = "fixed")]
    quadrature: String,
    /// GMRES relative-residual tolerance.
    #[arg(long, default_value_t = 1e-7)]
    tol: f64,
    /// GMRES restart length.
    #[arg(long, default_value_t = 200)]
    restart: usize,
    /// GMRES maximum iterations.
    #[arg(long, default_value_t = 10000)]
    max_iter: usize,
    /// Override the ~6 GiB dense-matrix memory guard.
    #[arg(long)]
    allow_large: bool,
    /// Solve despite Error-severity mesh/charge quality issues (still reported).
    #[arg(long)]
    allow_low_quality: bool,
    /// Write the per-vertex potential to this file as TSV (`x\ty\tz\tpotential`).
    #[arg(long)]
    potential_out: Option<PathBuf>,
    /// Summary output format.
    #[arg(long, value_enum, default_value_t = Format::Tsv)]
    format: Format,
}

/// kcal/mol → kJ/mol. Mirrors `_KCAL_TO_KJ` in the Python wrapper
/// (packages/proteon/src/proteon/forcefield.py); the connector computes in
/// kcal/mol and both front-ends convert for display.
const KCAL_TO_KJ: f64 = 4.184;

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
    F: Fn(&pdbtbx::PDB) -> std::result::Result<Vec<Row>, String> + Sync,
{
    let files = gather_inputs(&common.inputs)?;
    if files.is_empty() {
        return Err(anyhow!("no input structures found"));
    }

    // The compute closure is fallible so a per-structure failure (e.g. an
    // unparameterized residue for `energy`) isolates to that file rather than
    // aborting the batch — same path as a load failure.
    let compute_one = |path: &Path| -> std::result::Result<Vec<Row>, String> {
        let pdb = load_pdb(path).map_err(|e| e.to_string())?;
        compute(&pdb)
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
            return Ok(vec![vec![("total_sasa", Value::F64(total))]]);
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
        Ok(out)
    })
}

fn run_dssp(args: &DsspArgs) -> Result<()> {
    // Same entry point as py_dssp::compute_dssp — parity is by construction.
    run(&args.common, |pdb| {
        Ok(vec![vec![("dssp", Value::Str(dssp::dssp_from_pdb(pdb)))]])
    })
}

fn run_hbond(args: &HbondArgs) -> Result<()> {
    let geometric = args.geometric;
    let energy_cutoff = args.energy_cutoff;
    let dist_cutoff = args.dist_cutoff;
    run(&args.common, move |pdb| {
        let rows = if geometric {
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
        };
        Ok(rows)
    })
}

fn run_energy(args: &EnergyArgs) -> Result<()> {
    // Validate the force field once up front so an unknown name fails the whole
    // command cleanly, rather than emitting one identical per-file error.
    if !ff_api::is_known_force_field(&args.ff) {
        return Err(anyhow!(
            "unknown force field '{}'. Use charmm19_eef1, amber96, or amber96_obc.",
            args.ff
        ));
    }
    let (unit_label, factor) = match args.units.to_lowercase().as_str() {
        "kj/mol" | "kj" => ("kJ/mol", KCAL_TO_KJ),
        "kcal/mol" | "kcal" => ("kcal/mol", 1.0),
        other => {
            return Err(anyhow!(
                "unknown units '{other}'. Use 'kJ/mol' or 'kcal/mol'."
            ))
        }
    };
    let ff = args.ff.clone();
    let cutoff = args.nonbonded_cutoff;
    run(&args.common, move |pdb| {
        // Same entry point as py_forcefield::compute_energy — parity by
        // construction (the connector returns kcal/mol; we scale for display
        // exactly as the Python wrapper does). A per-structure failure
        // (unparameterized residue) becomes a String error and isolates to
        // this file.
        let rep = ff_api::energy_from_pdb(pdb, &ff, None, cutoff)?;
        let e = &rep.energy;
        Ok(vec![vec![
            ("ff", Value::Str(ff.clone())),
            ("units", Value::Str(unit_label.to_string())),
            ("bond_stretch", Value::F64(e.bond_stretch * factor)),
            ("angle_bend", Value::F64(e.angle_bend * factor)),
            ("torsion", Value::F64(e.torsion * factor)),
            ("improper_torsion", Value::F64(e.improper_torsion * factor)),
            ("vdw", Value::F64(e.vdw * factor)),
            ("electrostatic", Value::F64(e.electrostatic * factor)),
            ("solvation", Value::F64(e.solvation * factor)),
            ("total", Value::F64(e.total * factor)),
            (
                "n_unassigned_atoms",
                Value::Int(rep.n_unassigned_atoms as i64),
            ),
        ]])
    })
}

enum OutputDest {
    File(PathBuf),
    Dir(PathBuf),
}

fn resolve_output(w: &WriteCommon, n_inputs: usize) -> Result<OutputDest> {
    match (&w.out_file, &w.out_dir) {
        (Some(_), Some(_)) => Err(anyhow!("use -o/--output or --out-dir, not both")),
        (None, None) => Err(anyhow!("specify -o FILE (single input) or --out-dir DIR")),
        (Some(f), None) => {
            if n_inputs != 1 {
                return Err(anyhow!(
                    "-o/--output takes a single input; use --out-dir for multiple"
                ));
            }
            Ok(OutputDest::File(f.clone()))
        }
        (None, Some(d)) => Ok(OutputDest::Dir(d.clone())),
    }
}

/// Per-structure stat row from a preparation report. Energies are the
/// minimizer's native kcal/mol (column names say so) — distinct from the
/// `energy` command, which reports kJ/mol by default.
fn prepare_report_row(report: &PrepareReport, ff: &str) -> Row {
    vec![
        ("ff", Value::Str(ff.to_string())),
        ("reconstructed", Value::Int(report.reconstructed as i64)),
        ("h_added", Value::Int(report.h_added as i64)),
        ("h_skipped", Value::Int(report.h_skipped as i64)),
        ("n_unassigned", Value::Int(report.n_unassigned as i64)),
        (
            "skipped_no_protein",
            Value::Str(report.skipped_no_protein.to_string()),
        ),
        ("minimized", Value::Str(report.minimized.to_string())),
        ("init_e_kcal", Value::F64(report.init_e)),
        ("final_e_kcal", Value::F64(report.final_e)),
        ("steps", Value::Int(report.steps as i64)),
        ("converged", Value::Str(report.converged.to_string())),
    ]
}

/// Shared driver for the write commands: load each structure, run the
/// preparation pipeline in place, save the result, and emit a stat row. Per-file
/// failures isolate (stderr + nonzero exit); the structure file is written only
/// on success.
fn run_write(w: &WriteCommon, opts: PrepareOptions) -> Result<()> {
    if !prepare::is_prepare_force_field(&w.ff) {
        return Err(anyhow!(
            "unknown force field '{}'. Use charmm19_eef1 or amber96.",
            w.ff
        ));
    }
    let files = gather_inputs(&w.common.inputs)?;
    if files.is_empty() {
        return Err(anyhow!("no input structures found"));
    }
    let dest = resolve_output(w, files.len())?;
    if let OutputDest::Dir(d) = &dest {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    let ff = w.ff.clone();

    let out_path_for = |input: &Path| -> PathBuf {
        match &dest {
            OutputDest::File(f) => f.clone(),
            OutputDest::Dir(d) => {
                d.join(input.file_name().unwrap_or(std::ffi::OsStr::new("out.pdb")))
            }
        }
    };

    let compute_one = |path: &Path| -> std::result::Result<Row, String> {
        let mut pdb = load_pdb(path).map_err(|e| e.to_string())?;
        let report = prepare::prepare_from_pdb(&mut pdb, &ff, &opts)?;
        let outp = out_path_for(path);
        let outp_str = outp.to_str().ok_or("non-UTF8 output path")?;
        pdbtbx::save(&pdb, outp_str, pdbtbx::StrictnessLevel::Loose).map_err(|errs| {
            format!(
                "save failed: {}",
                errs.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })?;
        let mut row = prepare_report_row(&report, &ff);
        row.insert(
            0,
            ("output", Value::Str(outp.to_string_lossy().into_owned())),
        );
        Ok(row)
    };

    let results: Vec<std::result::Result<Row, String>> = if w.common.threads == 1 {
        files.iter().map(|p| compute_one(p)).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(w.common.threads)
            .build()
            .context("building thread pool")?;
        pool.install(|| files.par_iter().map(|p| compute_one(p)).collect())
    };

    let mut rows: Vec<(String, Row)> = Vec::new();
    let mut had_error = false;
    for (path, res) in files.iter().zip(results) {
        let file = path.to_string_lossy().into_owned();
        match res {
            Ok(mut row) => {
                row.insert(0, ("file", Value::Str(file.clone())));
                rows.push((file, row));
            }
            Err(e) => {
                had_error = true;
                eprintln!("ERROR {file}: {e}");
            }
        }
    }
    emit(w.common.format, &rows)?;
    if had_error {
        std::process::exit(1);
    }
    Ok(())
}

fn run_prepare(a: &PrepareArgs) -> Result<()> {
    run_write(
        &a.write,
        PrepareOptions {
            reconstruct: !a.no_reconstruct,
            hydrogens: a.hydrogens.clone(),
            include_water: false,
            minimize: !a.no_minimize,
            minimize_method: a.minimize_method.clone(),
            minimize_steps: a.minimize_steps,
            gradient_tolerance: 0.1,
            strip_hydrogens: true,
            constrain_heavy: None, // FF-aware
        },
    )
}

fn run_protonate(a: &ProtonateArgs) -> Result<()> {
    run_write(
        &a.write,
        PrepareOptions {
            reconstruct: false,
            hydrogens: a.hydrogens.clone(),
            include_water: false,
            minimize: false,
            minimize_method: "lbfgs".to_string(),
            minimize_steps: 0,
            gradient_tolerance: 0.1,
            strip_hydrogens: !a.keep_existing_h,
            constrain_heavy: None,
        },
    )
}

fn run_minimize(a: &MinimizeArgs) -> Result<()> {
    // Minimize the input as-is: no reconstruct, no H placement, keep existing
    // atoms. If the input has no hydrogens the pipeline reports minimized=false
    // (run `protonate` or `prepare` first to add them).
    run_write(
        &a.write,
        PrepareOptions {
            reconstruct: false,
            hydrogens: "none".to_string(),
            include_water: false,
            minimize: true,
            minimize_method: a.minimize_method.clone(),
            minimize_steps: a.minimize_steps,
            gradient_tolerance: 0.1,
            strip_hydrogens: false,
            constrain_heavy: None,
        },
    )
}

/// Resolve the (mesh, charges) pair from whichever input mode the user selected.
fn load_mesh_and_charges(
    args: &ElectrostaticsArgs,
) -> Result<(proteon_core::surface::mesh::Mesh, Vec<electro::Charge>)> {
    if let Some(hmo) = &args.hmo {
        return electro::read_hmo(hmo)
            .with_context(|| format!("reading HMO {}", hmo.display()));
    }
    let charges_path = args
        .pqr
        .as_ref()
        .ok_or_else(|| anyhow!("--pqr is required with --off / --msms"))?;
    let charges =
        electro::read_pqr(charges_path).with_context(|| format!("reading PQR {}", charges_path.display()))?;
    let mesh = if let Some(off) = &args.off {
        electro::read_off(off).with_context(|| format!("reading OFF {}", off.display()))?
    } else if let Some(prefix) = &args.msms {
        let vert = prefix.with_extension("vert");
        let face = prefix.with_extension("face");
        electro::read_msms(&vert, &face)
            .with_context(|| format!("reading MSMS {}.{{vert,face}}", prefix.display()))?
    } else {
        return Err(anyhow!("specify a mesh: --off FILE, --msms PREFIX, or --hmo FILE"));
    };
    Ok((mesh, charges))
}

fn run_electrostatics(args: &ElectrostaticsArgs) -> Result<()> {
    let quad = match args.quadrature.to_lowercase().as_str() {
        "fixed" => electro::Quadrature::Fixed,
        "adaptive" => electro::Quadrature::Adaptive(electro::AdaptiveConfig::default()),
        other => {
            return Err(anyhow!(
                "quadrature must be 'fixed' or 'adaptive', got '{other}'"
            ))
        }
    };
    let (mesh, charges) = load_mesh_and_charges(args)?;
    // Vertex order is preserved across the solve (auto-orient only re-winds triangles),
    // so this snapshot aligns 1:1 with the returned per-vertex potential.
    let verts = mesh.verts.clone();

    let opts = electro::SurfaceSolveOptions {
        params: electro::Params {
            eps_omega: args.eps_omega,
            eps_sigma: args.eps_sigma,
            eps_inf: args.eps_inf,
            lambda: args.lambda,
        },
        cfg: electro::SolveConfig {
            tol: args.tol,
            restart: args.restart,
            max_iter: args.max_iter,
        },
        nonlocal: args.nonlocal,
        quadrature: quad,
        allow_large: args.allow_large,
        allow_low_quality: args.allow_low_quality,
    };

    let sol = electro::solve_surface(mesh, &charges, &opts).map_err(|e| anyhow!(e.to_string()))?;

    // Advisories (auto-flip, quality, size) to stderr — the summary stays clean on stdout.
    for w in &sol.warnings {
        eprintln!("WARN {w}");
    }

    if let Some(path) = &args.potential_out {
        let mut w = io::BufWriter::new(
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
        );
        writeln!(w, "x\ty\tz\tpotential")?;
        for (v, phi) in verts.iter().zip(&sol.potential) {
            writeln!(w, "{:.6}\t{:.6}\t{:.6}\t{:.8}", v.x, v.y, v.z, phi)?;
        }
        w.flush()?;
    }

    let n_self = match sol.topology.num_self_intersections {
        Some(k) => k.to_string(),
        None => "inconclusive".to_string(),
    };
    let row: Row = vec![
        ("rfenergy_kj_mol", Value::F64(sol.rfenergy)),
        ("n_elements", Value::Int(sol.n_elements as i64)),
        ("converged", Value::Str(sol.converged.to_string())),
        ("iterations", Value::Int(sol.iterations as i64)),
        ("residual", Value::F64(sol.residual)),
        ("quadrature", Value::Str(sol.quadrature.to_string())),
        ("capped_panels", Value::Int(sol.capped_panels as i64)),
        ("watertight", Value::Str(sol.topology.watertight.to_string())),
        ("oriented", Value::Str(sol.topology.consistently_oriented.to_string())),
        ("is_outward", Value::Str(sol.topology.is_outward.to_string())),
        ("n_components", Value::Int(sol.topology.num_components as i64)),
        ("n_cavities", Value::Int(sol.topology.num_cavities as i64)),
        ("flipped_to_outward", Value::Str(sol.flipped_to_outward.to_string())),
        ("n_self_intersections", Value::Str(n_self)),
        ("min_angle_deg", Value::F64(sol.quality.min_angle_deg)),
        ("max_aspect_ratio", Value::F64(sol.quality.max_aspect_ratio)),
    ];
    emit(args.format, &[(String::new(), row)])
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::Sasa(a) => run_sasa(a),
        Cmd::Dssp(a) => run_dssp(a),
        Cmd::Hbond(a) => run_hbond(a),
        Cmd::Energy(a) => run_energy(a),
        Cmd::Prepare(a) => run_prepare(a),
        Cmd::Protonate(a) => run_protonate(a),
        Cmd::Minimize(a) => run_minimize(a),
        Cmd::Electrostatics(a) => run_electrostatics(a),
    }
}
