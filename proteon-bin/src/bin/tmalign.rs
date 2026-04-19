use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use proteon_align::core::align::cpalign::cpalign;
use proteon_align::core::align::tmalign::tmalign;
use proteon_align::core::types::{AlignOptions, MolType};
use proteon_io::alignment::read_alignment;
use proteon_io::chain_list::read_chain_list;
use proteon_io::loader::{load_structure, InputFormat, LoadOptions};
use proteon_io::output::{output_results, output_rotation_matrix, OutputOptions};
use proteon_io::superpose::output_superpose;

#[derive(Parser, Debug)]
#[command(
    name = "TMalign",
    about = "TM-align: sequence-independent protein structure alignment (Rust)"
)]
struct Cli {
    /// First structure (PDB/mmCIF)
    #[arg(index = 1)]
    pdb1: Option<String>,

    /// Second structure (PDB/mmCIF)
    #[arg(index = 2)]
    pdb2: Option<String>,

    /// TM-score normalized by user-assigned length
    #[arg(short = 'u', long = "user-length")]
    user_length: Option<f64>,

    /// TM-score normalized by average length (T or F)
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

    /// Print version
    #[arg(short = 'v', long = "version")]
    print_version: bool,

    /// Chain termination option (0-3)
    #[arg(long = "ter", default_value = "3")]
    ter: u8,

    /// Split PDB into chains (0-2)
    #[arg(long = "split", default_value = "0")]
    split: u8,

    /// Atom name (4-char, e.g., " CA ")
    #[arg(long = "atom", default_value = "auto")]
    atom: String,

    /// Input format for chain1 (-1=auto, 0=PDB, 1=SPICKER, 2=XYZ, 3=mmCIF)
    #[arg(long = "infmt1", default_value = "-1")]
    infmt1: i32,

    /// Input format for chain2
    #[arg(long = "infmt2", default_value = "-1")]
    infmt2: i32,

    /// Output format (0=full, 1=fasta, 2=tabular, -1=full-no-version)
    #[arg(long = "outfmt", default_value = "0")]
    outfmt: i32,

    /// TMcut threshold (-1 to disable)
    #[arg(long = "TMcut", default_value = "-1")]
    tmcut: f64,

    /// Align by residue index (0-3)
    #[arg(long = "byresi", default_value = "0")]
    byresi: u8,

    /// Align mirror image
    #[arg(long = "mirror", default_value = "0")]
    mirror: u8,

    /// Include HETATM residues
    #[arg(long = "het", default_value = "0")]
    het: u8,

    /// All-against-all in directory
    #[arg(long = "dir")]
    dir: Option<String>,

    /// Search chain2 against list in directory
    #[arg(long = "dir1")]
    dir1: Option<String>,

    /// Search chain1 against list in directory
    #[arg(long = "dir2")]
    dir2: Option<String>,

    /// File suffix for batch mode
    #[arg(long = "suffix", default_value = "")]
    suffix: String,
}

fn parse_infmt(val: i32) -> InputFormat {
    match val {
        0 => InputFormat::Pdb,
        1 => InputFormat::Spicker,
        2 => InputFormat::Xyz,
        3 => InputFormat::Mmcif,
        _ => InputFormat::Auto,
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.print_version {
        println!("TMalign-rs version 0.1.0 (Rust port)");
        return Ok(());
    }

    // Parse -a option
    let a_opt = match cli.average.as_deref() {
        Some("T" | "t" | "1") => 1,
        Some("-1") => -1,
        Some("-2") => -2,
        _ => 0,
    };

    // Handle alignment options
    let (i_opt, user_alignment) = if let Some(ref path) = cli.strict_align {
        let seqs = read_alignment(path)
            .with_context(|| format!("Failed to read strict alignment from {path}"))?;
        (3, Some(seqs))
    } else if let Some(ref path) = cli.init_align {
        let seqs = read_alignment(path)
            .with_context(|| format!("Failed to read initial alignment from {path}"))?;
        (1, Some(seqs))
    } else {
        (0, None)
    };

    let u_opt = cli.user_length.is_some();
    let lnorm_ass = cli.user_length.unwrap_or(0.0);
    let d_opt = cli.d0_scale.is_some();
    let d0_scale = cli.d0_scale.unwrap_or(0.0);

    // Build chain lists
    let (chain1_list, chain2_list) = build_chain_lists(&cli)?;

    // Print tabular header
    if cli.outfmt == 2 {
        println!("#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali");
    }

    let start = Instant::now();
    let mut stdout = std::io::stdout().lock();

    // Main alignment loop
    for (ci, chain1_path) in chain1_list.iter().enumerate() {
        let load_opts1 = LoadOptions {
            ter_opt: cli.ter,
            split_opt: cli.split,
            het_opt: cli.het > 0,
            atom_name: cli.atom.clone(),
            infmt: parse_infmt(cli.infmt1),
            mirror: cli.mirror > 0,
            byresi: cli.byresi,
        };

        let structures1 = load_structure(chain1_path, &load_opts1)
            .with_context(|| format!("Failed to load {chain1_path}"))?;

        for s1 in &structures1 {
            let j_start = if cli.dir.is_some() { ci + 1 } else { 0 };

            for chain2_path in chain2_list.iter().skip(j_start) {
                let load_opts2 = LoadOptions {
                    ter_opt: cli.ter,
                    split_opt: cli.split,
                    het_opt: cli.het > 0,
                    atom_name: cli.atom.clone(),
                    infmt: parse_infmt(cli.infmt2),
                    mirror: false,
                    byresi: cli.byresi,
                };

                let structures2 = load_structure(chain2_path, &load_opts2)
                    .with_context(|| format!("Failed to load {chain2_path}"))?;

                for s2 in &structures2 {
                    if s1.is_empty() || s2.is_empty() {
                        continue;
                    }

                    // Determine molecule type
                    let mol_type = if s1.mol_type == MolType::RNA || s2.mol_type == MolType::RNA {
                        MolType::RNA
                    } else {
                        MolType::Protein
                    };

                    let opts = AlignOptions {
                        i_opt,
                        a_opt,
                        u_opt,
                        lnorm_ass,
                        d_opt,
                        d0_scale,
                        fast_opt: cli.fast,
                        mol_type,
                        tm_cut: cli.tmcut,
                        user_alignment: user_alignment.clone(),
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
                        xname: chain1_path.clone(),
                        yname: chain2_path.clone(),
                        chain_id1: s1.chain_id.clone(),
                        chain_id2: s2.chain_id.clone(),
                        xlen: s1.len(),
                        ylen: s2.len(),
                        outfmt: cli.outfmt,
                        a_opt,
                        u_opt,
                        d_opt,
                        i_opt,
                        mirror_opt: cli.mirror > 0,
                        lnorm_ass,
                        d0_scale,
                        tm_ali: 0.0,
                        l_ali: 0,
                        rmsd_ali: 0.0,
                    };

                    output_results(&mut stdout, &result, &out_opts)?;

                    // Write rotation matrix if requested
                    if let Some(ref matrix_path) = cli.matrix {
                        let mut f = std::fs::File::create(matrix_path)
                            .with_context(|| format!("Failed to create {matrix_path}"))?;
                        output_rotation_matrix(&mut f, &result)?;
                    }

                    // Write superposed PDB if requested
                    if let Some(ref super_prefix) = cli.output_super {
                        let pdb_path = format!("{super_prefix}.pdb");
                        let mut f = std::fs::File::create(&pdb_path)
                            .with_context(|| format!("Failed to create {pdb_path}"))?;
                        output_superpose(&mut f, &result, s1, s2)?;
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();
    if cli.outfmt != 2 {
        writeln!(
            stdout,
            "\nTotal CPU time is {:.2} seconds",
            elapsed.as_secs_f64()
        )?;
    }

    Ok(())
}

fn build_chain_lists(cli: &Cli) -> Result<(Vec<String>, Vec<String>)> {
    if let Some(ref dir) = cli.dir {
        // All-vs-all mode
        let pdb1 = cli
            .pdb1
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("chain list file required with -dir"))?;
        let list = read_chain_list(pdb1, dir, &cli.suffix)?;
        Ok((list.clone(), list))
    } else if cli.dir1.is_some() || cli.dir2.is_some() {
        let mut list1 = Vec::new();
        let mut list2 = Vec::new();

        if let Some(ref dir1) = cli.dir1 {
            let pdb1 = cli
                .pdb1
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("chain list file required with -dir1"))?;
            list1 = read_chain_list(pdb1, dir1, &cli.suffix)?;
        } else if let Some(ref path) = cli.pdb1 {
            list1.push(path.clone());
        }

        if let Some(ref dir2) = cli.dir2 {
            let pdb2 = cli
                .pdb2
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("chain list file required with -dir2"))?;
            list2 = read_chain_list(pdb2, dir2, &cli.suffix)?;
        } else if let Some(ref path) = cli.pdb2 {
            list2.push(path.clone());
        }

        Ok((list1, list2))
    } else {
        // Single pair mode
        let pdb1 = cli.pdb1.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Two structure files required. Run with --help for usage.")
        })?;
        let pdb2 = cli.pdb2.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Two structure files required. Run with --help for usage.")
        })?;
        Ok((vec![pdb1.clone()], vec![pdb2.clone()]))
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
