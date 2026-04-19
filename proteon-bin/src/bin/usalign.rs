use std::io::Write;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;

use proteon_align::core::align::cpalign::cpalign;
use proteon_align::core::align::tmalign::tmalign;
use proteon_align::core::types::{AlignOptions, AlignResult, StructureData, Transform};
use proteon_align::ext::flexalign::{flexalign_main, FlexOptions};
use proteon_align::ext::mmalign::{mmalign_complex, ChainData};
use proteon_align::ext::soialign::{soialign_main, SoiOptions};
use proteon_io::alignment::read_alignment;
use proteon_io::chain_list::read_chain_list;
use proteon_io::loader::{load_structure, InputFormat, LoadOptions};
use proteon_io::output::{output_results, output_rotation_matrix, OutputOptions};
use proteon_io::superpose::output_superpose;

const VERSION: &str = "20260329";

fn print_version() {
    println!(
        "\n\
         ********************************************************************\n\
         * US-align (Rust) (Version {VERSION})                              *\n\
         * Universal Structure Alignment of Proteins and Nucleic Acids      *\n\
         * Reference: C Zhang, L Freddolino, Y Zhang. (2026) Nat Protoc    *\n\
         *            C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Nat Methods*\n\
         *            C Zhang, AM Pyle (2022) iScience.                     *\n\
         ********************************************************************"
    );
}

#[derive(Parser, Debug)]
#[command(
    name = "USalign",
    about = "US-align: universal structure alignment of proteins and nucleic acids (Rust)"
)]
struct Cli {
    /// First structure (PDB/mmCIF), or chain list file when used with -dir/-dir1
    #[arg(index = 1)]
    pdb1: Option<String>,

    /// Second structure (PDB/mmCIF), or chain list file when used with -dir2
    #[arg(index = 2)]
    pdb2: Option<String>,

    /// Print version
    #[arg(short = 'v', long = "version")]
    print_version: bool,

    /// TM-score normalized by user-assigned length
    #[arg(short = 'u', long = "user-length")]
    user_length: Option<f64>,

    /// TM-score normalized by average length (T/F/-1/-2)
    #[arg(short = 'a', long = "average")]
    average: Option<String>,

    /// Start with an alignment from a FASTA file
    #[arg(short = 'i', long = "init-align")]
    init_align: Option<String>,

    /// Stick to the alignment from a FASTA file
    #[arg(short = 'I', long = "strict-align")]
    strict_align: Option<String>,

    /// Output rotation matrix to file
    #[arg(short = 'm', long = "matrix")]
    matrix: Option<String>,

    /// TM-score scaled by assigned d0
    #[arg(short = 'd', long = "d0-scale")]
    d0_scale: Option<f64>,

    /// Output superposition prefix
    #[arg(short = 'o', long = "output")]
    output_super: Option<String>,

    /// Fast but slightly inaccurate alignment
    #[arg(long = "fast")]
    fast: bool,

    /// Circular permutation alignment
    #[arg(long = "cp")]
    cp: bool,

    /// Multimeric alignment option (0-7)
    #[arg(long = "mm", default_value = "0")]
    mm: i32,

    /// Chain termination option (0-3, default auto)
    #[arg(long = "ter", default_value = "-1")]
    ter: i32,

    /// Split PDB into chains (0-2, default auto)
    #[arg(long = "split", default_value = "-1")]
    split: i32,

    /// Atom name (4-char, e.g., " CA " or "auto")
    #[arg(long = "atom", default_value = "auto")]
    atom: String,

    /// Input format for structure 1 (PDB, mmCIF, or auto)
    #[arg(long = "infmt1", default_value = "auto")]
    infmt1: String,

    /// Input format for structure 2
    #[arg(long = "infmt2", default_value = "auto")]
    infmt2: String,

    /// Output format (0=full, 1=fasta, 2=tabular, -1=no header)
    #[arg(long = "outfmt", default_value = "0")]
    outfmt: i32,

    /// TM-score cutoff for early termination (-1 to disable)
    #[arg(long = "TMcut", default_value = "-1")]
    tmcut: f64,

    /// Align mirror image of structure 1 (0 or 1)
    #[arg(long = "mirror", default_value = "0")]
    mirror: u8,

    /// Include HETATM residues (0=no, 1=yes, 2=MSE only)
    #[arg(long = "het", default_value = "0")]
    het: u8,

    /// Residue index matching mode (0-7)
    #[arg(long = "byresi", default_value = "0")]
    byresi: u8,

    /// Show full pairwise alignment of individual chains
    #[arg(long = "full")]
    full: bool,

    /// Skip superposition (extract alignment from pre-superposed structures)
    #[arg(long = "se")]
    se: bool,

    /// Number of closest atoms for SOI alignment (default auto)
    #[arg(long = "closeK", default_value = "-1")]
    close_k: i32,

    /// Maximum number of hinges for flexible alignment
    #[arg(long = "hinge", default_value = "9")]
    hinge: usize,

    /// All-against-all: directory + chain list file
    #[arg(long = "dir")]
    dir: Option<String>,

    /// One-against-many: chain1 directory + chain1 list file, chain2 is pdb2
    #[arg(long = "dir1")]
    dir1: Option<String>,

    /// One-against-many: chain2 directory + chain2 list file, chain1 is pdb1
    #[arg(long = "dir2")]
    dir2: Option<String>,

    /// File suffix for batch modes
    #[arg(long = "suffix", default_value = "")]
    suffix: String,

    /// Molecule type (auto, protein, RNA)
    #[arg(long = "mol", default_value = "auto")]
    mol: String,

    /// Output distance of aligned pairs
    #[arg(long = "do")]
    do_opt: bool,

    /// Chain mapping file (for -mm 1)
    #[arg(long = "chainmap")]
    chainmap: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_input_format(s: &str) -> InputFormat {
    match s.to_lowercase().as_str() {
        "pdb" => InputFormat::Pdb,
        "mmcif" | "cif" => InputFormat::Mmcif,
        _ => InputFormat::Auto,
    }
}

fn make_load_opts(cli: &Cli) -> LoadOptions {
    let ter_opt = if cli.ter < 0 { 3u8 } else { cli.ter as u8 };
    let split_opt = if cli.split < 0 { 0u8 } else { cli.split as u8 };
    LoadOptions {
        ter_opt,
        split_opt,
        atom_name: cli.atom.clone(),
        het_opt: cli.het > 0,
        mirror: cli.mirror > 0,
        infmt: parse_input_format(&cli.infmt1),
        byresi: cli.byresi,
    }
}

fn parse_a_opt(cli: &Cli) -> i32 {
    match cli.average.as_deref() {
        Some("T" | "t" | "1") => 1,
        Some("-1") => -1,
        Some("-2") => -2,
        _ => 0,
    }
}

/// Compact tabular output line for one alignment pair.
fn format_tabular(
    name1: &str,
    chain1: &str,
    name2: &str,
    chain2: &str,
    result: &AlignResult,
    xlen: usize,
    ylen: usize,
) -> String {
    let id_ali = if result.n_aligned > 0 {
        result.seq_identity
    } else {
        0.0
    };
    let id1 = result.seq_identity * result.n_aligned as f64 / xlen.max(1) as f64;
    let id2 = result.seq_identity * result.n_aligned as f64 / ylen.max(1) as f64;
    format!(
        "{}:{}\t{}:{}\t{:.4}\t{:.4}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}",
        name1,
        chain1,
        name2,
        chain2,
        result.tm_score_chain1,
        result.tm_score_chain2,
        result.rmsd,
        id1,
        id2,
        id_ali,
        xlen,
        ylen,
        result.n_aligned,
    )
}

// ---------------------------------------------------------------------------
// Single-pair alignment
// ---------------------------------------------------------------------------

fn align_pair(
    s1: &StructureData,
    s2: &StructureData,
    fast: bool,
    cp: bool,
    tmcut: f64,
    a_opt: i32,
) -> Result<AlignResult> {
    let opts = AlignOptions {
        i_opt: 0,
        a_opt,
        u_opt: false,
        lnorm_ass: 0.0,
        d_opt: false,
        d0_scale: 0.0,
        fast_opt: fast,
        mol_type: s1.mol_type,
        tm_cut: tmcut,
        user_alignment: None,
    };

    if cp {
        cpalign(
            &s1.coords,
            &s2.coords,
            &s1.sequence,
            &s2.sequence,
            &s1.sec_structure,
            &s2.sec_structure,
            &opts,
        )
    } else {
        tmalign(
            &s1.coords,
            &s2.coords,
            &s1.sequence,
            &s2.sequence,
            &s1.sec_structure,
            &s2.sec_structure,
            &opts,
        )
    }
}

// ---------------------------------------------------------------------------
// Single-pair mode (mm=0, no batch)
// ---------------------------------------------------------------------------

fn run_single_pair(cli: &Cli) -> Result<()> {
    let pdb1 = cli.pdb1.as_ref().context("Missing first structure")?;
    let pdb2 = cli.pdb2.as_ref().context("Missing second structure")?;
    let load_opts = make_load_opts(cli);
    let load_opts2 = LoadOptions {
        infmt: parse_input_format(&cli.infmt2),
        ..load_opts.clone()
    };

    let structures1 =
        load_structure(pdb1, &load_opts).with_context(|| format!("Failed to load {}", pdb1))?;
    let structures2 =
        load_structure(pdb2, &load_opts2).with_context(|| format!("Failed to load {}", pdb2))?;

    if structures1.is_empty() || structures2.is_empty() {
        bail!("No chains found in input structures");
    }

    let s1 = &structures1[0];
    let s2 = &structures2[0];
    let a_opt = parse_a_opt(cli);

    let mut i_opt: i32 = 0;
    let mut user_alignment = None;
    if let Some(ref path) = cli.strict_align {
        i_opt = 3;
        user_alignment = Some(read_alignment(path)?);
    } else if let Some(ref path) = cli.init_align {
        i_opt = 1;
        user_alignment = Some(read_alignment(path)?);
    }

    let opts = AlignOptions {
        i_opt,
        a_opt,
        u_opt: cli.user_length.is_some(),
        lnorm_ass: cli.user_length.unwrap_or(0.0),
        d_opt: cli.d0_scale.is_some(),
        d0_scale: cli.d0_scale.unwrap_or(0.0),
        fast_opt: cli.fast,
        mol_type: s1.mol_type,
        tm_cut: cli.tmcut,
        user_alignment,
    };

    let result = if cli.cp {
        cpalign(
            &s1.coords,
            &s2.coords,
            &s1.sequence,
            &s2.sequence,
            &s1.sec_structure,
            &s2.sec_structure,
            &opts,
        )?
    } else {
        tmalign(
            &s1.coords,
            &s2.coords,
            &s1.sequence,
            &s2.sequence,
            &s1.sec_structure,
            &s2.sec_structure,
            &opts,
        )?
    };

    let out_opts = OutputOptions {
        outfmt: cli.outfmt,
        xname: pdb1.clone(),
        yname: pdb2.clone(),
        chain_id1: s1.chain_id.clone(),
        chain_id2: s2.chain_id.clone(),
        xlen: s1.coords.len(),
        ylen: s2.coords.len(),
        a_opt,
        u_opt: cli.user_length.is_some(),
        d_opt: cli.d0_scale.is_some(),
        i_opt,
        mirror_opt: cli.mirror > 0,
        lnorm_ass: cli.user_length.unwrap_or(0.0),
        d0_scale: cli.d0_scale.unwrap_or(0.0),
        tm_ali: 0.0,
        l_ali: 0,
        rmsd_ali: 0.0,
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    if cli.outfmt <= 0 {
        print_version();
        println!();
    }

    output_results(&mut out, &result, &out_opts)?;

    if let Some(ref matrix_file) = cli.matrix {
        let mut f = std::fs::File::create(matrix_file)
            .with_context(|| format!("Failed to create {}", matrix_file))?;
        output_rotation_matrix(&mut f, &result)?;
    }

    if let Some(ref super_prefix) = cli.output_super {
        let pdb_path = format!("{super_prefix}.pdb");
        let mut f = std::fs::File::create(&pdb_path)
            .with_context(|| format!("Failed to create {pdb_path}"))?;
        output_superpose(&mut f, &result, s1, s2)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Batch modes with rayon parallelism
// ---------------------------------------------------------------------------

/// Build list of (path1, path2) pairs from dir/dir1/dir2 flags.
fn build_pair_list(cli: &Cli) -> Result<Vec<(String, String)>> {
    let pdb1 = cli.pdb1.as_ref().context("Missing first argument")?;

    if let Some(ref dir) = cli.dir {
        // All-against-all: pdb1 is the chain list file, dir is the folder
        let chains = read_chain_list(pdb1, dir, &cli.suffix)?;
        let n = chains.len();
        let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((chains[i].clone(), chains[j].clone()));
            }
        }
        eprintln!("All-against-all: {} structures, {} pairs", n, pairs.len());
        Ok(pairs)
    } else if let Some(ref dir1) = cli.dir1 {
        // dir1 mode: pdb1 is the chain list, pdb2 is the query
        let pdb2 = cli
            .pdb2
            .as_ref()
            .context("Missing second structure for -dir1")?;
        let chains = read_chain_list(pdb1, dir1, &cli.suffix)?;
        eprintln!("Query {} against {} structures", pdb2, chains.len());
        Ok(chains.into_iter().map(|c| (c, pdb2.clone())).collect())
    } else if let Some(ref dir2) = cli.dir2 {
        // dir2 mode: pdb2 (second arg) is the chain list, pdb1 is the query
        let pdb2 = cli.pdb2.as_ref().context("Missing chain list for -dir2")?;
        let chains = read_chain_list(pdb2, dir2, &cli.suffix)?;
        eprintln!("Query {} against {} structures", pdb1, chains.len());
        Ok(chains.into_iter().map(|c| (pdb1.clone(), c)).collect())
    } else {
        bail!("No batch mode specified")
    }
}

fn run_batch_tmalign(cli: &Cli) -> Result<()> {
    let pairs = build_pair_list(cli)?;
    let load_opts = make_load_opts(cli);
    let a_opt = parse_a_opt(cli);

    // Force tabular output for batch
    if cli.outfmt < 2 {
        println!("#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali");
    }

    // Parallel alignment of all pairs
    let results: Vec<String> = pairs
        .par_iter()
        .filter_map(|(path1, path2)| {
            let s1_vec = load_structure(path1, &load_opts).ok()?;
            let s2_vec = load_structure(
                path2,
                &LoadOptions {
                    infmt: parse_input_format(&cli.infmt2),
                    ..load_opts.clone()
                },
            )
            .ok()?;

            if s1_vec.is_empty() || s2_vec.is_empty() {
                return None;
            }
            let s1 = &s1_vec[0];
            let s2 = &s2_vec[0];

            let result = align_pair(s1, s2, cli.fast, cli.cp, cli.tmcut, a_opt).ok()?;
            Some(format_tabular(
                path1,
                &s1.chain_id,
                path2,
                &s2.chain_id,
                &result,
                s1.coords.len(),
                s2.coords.len(),
            ))
        })
        .collect();

    // Print results in order
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in &results {
        writeln!(out, "{}", line)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// SOI-align mode
// ---------------------------------------------------------------------------

fn run_soialign_mode(cli: &Cli) -> Result<()> {
    let pdb1 = cli.pdb1.as_ref().context("Missing first structure")?;
    let pdb2 = cli.pdb2.as_ref().context("Missing second structure")?;
    let load_opts = make_load_opts(cli);

    let structures1 = load_structure(pdb1, &load_opts)?;
    let structures2 = load_structure(
        pdb2,
        &LoadOptions {
            infmt: parse_input_format(&cli.infmt2),
            ..load_opts.clone()
        },
    )?;

    if structures1.is_empty() || structures2.is_empty() {
        bail!("No chains found");
    }
    let s1 = &structures1[0];
    let s2 = &structures2[0];

    let secx: Vec<u8> = s1.sec_structure.iter().map(|c| *c as u8).collect();
    let secy: Vec<u8> = s2.sec_structure.iter().map(|c| *c as u8).collect();
    let seqx: Vec<u8> = s1.sequence.iter().map(|c| *c as u8).collect();
    let seqy: Vec<u8> = s2.sequence.iter().map(|c| *c as u8).collect();

    let identity = Transform {
        t: [0.0; 3],
        u: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    let soi_opts = SoiOptions {
        mol_type: s1.mol_type,
        close_k_opt: if cli.close_k < 0 {
            5
        } else {
            cli.close_k as usize
        },
        fast_opt: cli.fast,
        outfmt_opt: cli.outfmt,
        ..Default::default()
    };

    let result = soialign_main(
        &s1.coords, &s2.coords, &seqx, &seqy, &secx, &secy, &identity, &soi_opts,
    );

    if cli.outfmt <= 0 {
        print_version();
        println!();
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if cli.outfmt >= 2 {
        writeln!(
            out,
            "{}\t{}\t{:.4}\t{:.4}\t{:.2}\t{:.3}\t{:.3}\t{}\t{}\t{}",
            pdb1,
            pdb2,
            result.tm1,
            result.tm2,
            result.rmsd,
            if !s1.coords.is_empty() {
                result.liden / s1.coords.len() as f64
            } else {
                0.0
            },
            if !s2.coords.is_empty() {
                result.liden / s2.coords.len() as f64
            } else {
                0.0
            },
            s1.coords.len(),
            s2.coords.len(),
            result.n_ali8
        )?;
    } else {
        writeln!(out, "Name of Structure_1: {}:{}", pdb1, s1.chain_id)?;
        writeln!(out, "Name of Structure_2: {}:{}", pdb2, s2.chain_id)?;
        writeln!(out, "Length of Structure_1: {} residues", s1.coords.len())?;
        writeln!(out, "Length of Structure_2: {} residues", s2.coords.len())?;
        writeln!(out)?;
        writeln!(
            out,
            "Aligned length= {}, RMSD= {:6.2}, Seq_ID=n_identical/n_aligned= {:.3}",
            result.n_ali8,
            result.rmsd,
            if result.n_ali8 > 0 {
                result.liden / result.n_ali8 as f64
            } else {
                0.0
            }
        )?;
        writeln!(
            out,
            "TM-score= {:.5} (normalized by length of Structure_1: L={}, d0={:.2})",
            result.tm1,
            s1.coords.len(),
            result.d0a
        )?;
        writeln!(
            out,
            "TM-score= {:.5} (normalized by length of Structure_2: L={}, d0={:.2})",
            result.tm2,
            s2.coords.len(),
            result.d0b
        )?;
        if !result.seq_x_aligned.is_empty() {
            writeln!(out)?;
            writeln!(out, "{}", result.seq_x_aligned)?;
            writeln!(out, "{}", result.seq_m)?;
            writeln!(out, "{}", result.seq_y_aligned)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MM-align mode (-mm 1) — multi-chain complex alignment
// ---------------------------------------------------------------------------

fn run_mmalign_mode(cli: &Cli) -> Result<()> {
    let pdb1 = cli.pdb1.as_ref().context("Missing first structure")?;
    let pdb2 = cli.pdb2.as_ref().context("Missing second structure")?;

    // Force per-chain split regardless of CLI's --split flag — mmalign
    // operates on a vector of chains, so each StructureData must be one chain.
    let mut load_opts = make_load_opts(cli);
    load_opts.split_opt = 2;
    let load_opts2 = LoadOptions {
        infmt: parse_input_format(&cli.infmt2),
        ..load_opts.clone()
    };

    let structures1 =
        load_structure(pdb1, &load_opts).with_context(|| format!("Failed to load {pdb1}"))?;
    let structures2 =
        load_structure(pdb2, &load_opts2).with_context(|| format!("Failed to load {pdb2}"))?;

    if structures1.is_empty() || structures2.is_empty() {
        bail!("MM-align needs at least one chain in each structure");
    }

    let x_chains = structures_to_chains(&structures1);
    let y_chains = structures_to_chains(&structures2);
    let result = mmalign_complex(&x_chains, &y_chains)?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if cli.outfmt <= 0 {
        print_version();
        writeln!(out)?;
    }
    write_mmalign_result(&mut out, cli, pdb1, pdb2, &x_chains, &y_chains, &result)?;
    Ok(())
}

fn structures_to_chains(structures: &[StructureData]) -> Vec<ChainData> {
    structures
        .iter()
        .map(|s| ChainData {
            coords: s.coords.clone(),
            sequence: s.sequence.iter().map(|&c| c as u8).collect(),
            sec_structure: s.sec_structure.iter().map(|&c| c as u8).collect(),
            chain_id: s.chain_id.clone(),
            mol_type: s.mol_type,
        })
        .collect()
}

fn write_mmalign_result<W: Write>(
    w: &mut W,
    cli: &Cli,
    pdb1: &str,
    pdb2: &str,
    x_chains: &[ChainData],
    y_chains: &[ChainData],
    result: &proteon_align::ext::mmalign::MMAlignResult,
) -> Result<()> {
    let chain_ids = |chains: &[ChainData]| {
        chains
            .iter()
            .map(|c| c.chain_id.clone())
            .collect::<Vec<_>>()
            .join(":")
    };
    let total_len = |chains: &[ChainData]| chains.iter().map(ChainData::len).sum::<usize>();

    let xlen = total_len(x_chains);
    let ylen = total_len(y_chains);

    if cli.outfmt <= 0 {
        writeln!(
            w,
            "Name of Structure_1: {pdb1}:{} (to be superimposed onto Structure_2)",
            chain_ids(x_chains)
        )?;
        writeln!(w, "Name of Structure_2: {pdb2}:{}", chain_ids(y_chains))?;
        writeln!(w, "Length of Structure_1: {xlen} residues")?;
        writeln!(w, "Length of Structure_2: {ylen} residues")?;
        writeln!(w)?;

        // Aggregate aligned-pair count + RMSD across the assigned chain pairs.
        let mut total_n_ali = 0usize;
        let mut total_sq_dev = 0.0_f64;
        let mut total_iden = 0.0_f64;
        for r in &result.per_chain_results {
            total_n_ali += r.n_ali8;
            total_sq_dev += r.rmsd * r.rmsd * (r.n_ali8 as f64);
            total_iden += r.liden;
        }
        let rmsd = if total_n_ali == 0 {
            0.0
        } else {
            (total_sq_dev / total_n_ali as f64).sqrt()
        };
        let seq_id = if total_n_ali == 0 {
            0.0
        } else {
            total_iden / total_n_ali as f64
        };
        writeln!(
            w,
            "Aligned length= {total_n_ali}, RMSD= {rmsd:>5.2}, Seq_ID=n_identical/n_aligned= {seq_id:.3}"
        )?;
        // Length-weighted mean of per-chain TM-scores (each already normalized
        // by the chain's reference length). Not identical to C++ `TMave_mat`
        // output, which re-scores the concatenated aligned pairs against d0
        // of the whole complex; this is a cheaper proxy in the same [0,1]
        // range. `total_score` is the chain-assignment search objective and
        // is reported separately so both views are visible.
        let mut weighted_tm = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for (&(_, j), r) in result
            .chain_assignments
            .iter()
            .zip(&result.per_chain_results)
        {
            let w_j = y_chains[j].len() as f64;
            weighted_tm += r.tm1 * w_j;
            weight_sum += w_j;
        }
        let tm_complex = if weight_sum > 0.0 {
            weighted_tm / weight_sum
        } else {
            0.0
        };
        writeln!(
            w,
            "Complex TM-score= {tm_complex:.5} (length-weighted mean over assigned chain pairs)"
        )?;
        writeln!(w, "Assignment objective= {:.5}", result.total_score)?;
        writeln!(w)?;

        writeln!(w, "Chain assignments (Structure_1 → Structure_2):")?;
        for &(i, j) in &result.chain_assignments {
            let xc = &x_chains[i];
            let yc = &y_chains[j];
            let pair_score = result
                .per_chain_results
                .get(
                    result
                        .chain_assignments
                        .iter()
                        .position(|&p| p == (i, j))
                        .unwrap_or(0),
                )
                .map(|r| r.tm1)
                .unwrap_or(0.0);
            writeln!(
                w,
                "  {}({}) ↔ {}({})  TM-score= {:.5}",
                xc.chain_id,
                xc.len(),
                yc.chain_id,
                yc.len(),
                pair_score
            )?;
        }
    } else {
        // outfmt >= 2: tabular
        writeln!(
            w,
            "{pdb1}:{}\t{pdb2}:{}\t{xlen}\t{ylen}\t{:.5}",
            chain_ids(x_chains),
            chain_ids(y_chains),
            result.total_score
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Flex-align mode
// ---------------------------------------------------------------------------

fn run_flexalign_mode(cli: &Cli) -> Result<()> {
    let pdb1 = cli.pdb1.as_ref().context("Missing first structure")?;
    let pdb2 = cli.pdb2.as_ref().context("Missing second structure")?;
    let load_opts = make_load_opts(cli);

    let structures1 = load_structure(pdb1, &load_opts)?;
    let structures2 = load_structure(
        pdb2,
        &LoadOptions {
            infmt: parse_input_format(&cli.infmt2),
            ..load_opts.clone()
        },
    )?;

    if structures1.is_empty() || structures2.is_empty() {
        bail!("No chains found");
    }
    let s1 = &structures1[0];
    let s2 = &structures2[0];

    let secx: Vec<u8> = s1.sec_structure.iter().map(|c| *c as u8).collect();
    let secy: Vec<u8> = s2.sec_structure.iter().map(|c| *c as u8).collect();
    let seqx: Vec<u8> = s1.sequence.iter().map(|c| *c as u8).collect();
    let seqy: Vec<u8> = s2.sequence.iter().map(|c| *c as u8).collect();

    let flex_opts = FlexOptions {
        hinge_opt: cli.hinge,
        mol_type: s1.mol_type,
        ..Default::default()
    };

    let result = flexalign_main(
        &s1.coords,
        &s2.coords,
        &seqx,
        &seqy,
        &secx,
        &secy,
        &flex_opts,
        &[],
    )?;

    if cli.outfmt <= 0 {
        print_version();
        println!();
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "Name of Structure_1: {}:{}", pdb1, s1.chain_id)?;
    writeln!(out, "Name of Structure_2: {}:{}", pdb2, s2.chain_id)?;
    writeln!(out, "Length of Structure_1: {} residues", s1.coords.len())?;
    writeln!(out, "Length of Structure_2: {} residues", s2.coords.len())?;
    writeln!(out)?;
    writeln!(out, "Number of hinges: {}", result.hinge_count)?;
    writeln!(
        out,
        "Aligned length= {}, RMSD= {:6.2}",
        result.se_result.n_ali8, result.se_result.rmsd
    )?;
    writeln!(
        out,
        "TM-score= {:.5} (normalized by length of Structure_1)",
        result.se_result.tm1
    )?;
    writeln!(
        out,
        "TM-score= {:.5} (normalized by length of Structure_2)",
        result.se_result.tm2
    )?;

    if !result.se_result.seq_x_aligned.is_empty() {
        writeln!(out)?;
        writeln!(out, "{}", result.se_result.seq_x_aligned)?;
        writeln!(out, "{}", result.se_result.seq_m)?;
        writeln!(out, "{}", result.se_result.seq_y_aligned)?;
    }

    if let Some(ref matrix_file) = cli.matrix {
        use proteon_align::ext::flexalign::format_rotation_matrices;
        let mut f = std::fs::File::create(matrix_file)?;
        write!(f, "{}", format_rotation_matrices(&result.transforms))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main dispatch
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.print_version {
        print_version();
        std::process::exit(1);
    }

    if cli.pdb1.is_none() {
        Cli::parse_from(["USalign", "--help"]);
        return Ok(());
    }

    let t1 = Instant::now();
    let is_batch = cli.dir.is_some() || cli.dir1.is_some() || cli.dir2.is_some();

    match cli.mm {
        0 if is_batch => run_batch_tmalign(&cli)?,
        0 => run_single_pair(&cli)?,
        1 => run_mmalign_mode(&cli)?,
        2 => eprintln!("MM-dock mode (-mm 2) -- not yet fully wired in Rust port"),
        4 => eprintln!("mTM-align mode (-mm 4) -- not yet fully wired in Rust port"),
        5 | 6 => run_soialign_mode(&cli)?,
        7 => run_flexalign_mode(&cli)?,
        _ => eprintln!("WARNING! -mm {} not implemented", cli.mm),
    }

    let elapsed = t1.elapsed();
    if cli.outfmt < 2 || !is_batch {
        println!("#Total CPU time is {:5.2} seconds", elapsed.as_secs_f64());
    }

    Ok(())
}
