//! Structure loading via pdbtbx.
//!
//! Bridges pdbtbx's PDB/mmCIF parser to TMAlign's internal `StructureData`.

use anyhow::{bail, Context, Result};

use proteon_align::core::residue_map::three_to_one;
use proteon_align::core::secondary_structure::make_sec;
use proteon_align::core::types::{Coord3D, MolType, StructureData};

/// Input format specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Auto,
    Pdb,
    Mmcif,
    Spicker,
    Xyz,
}

/// Options controlling structure loading.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Chain termination mode (0-3). Currently maps to pdbtbx behavior.
    pub ter_opt: u8,
    /// Chain splitting mode (0=whole file, 1=by MODEL, 2=by chain).
    pub split_opt: u8,
    /// Include HETATM records.
    pub het_opt: bool,
    /// Atom name to extract (e.g., "CA" for protein, "C3'" for RNA).
    /// "auto" means detect from molecule type.
    pub atom_name: String,
    /// Input format.
    pub infmt: InputFormat,
    /// Negate z-coordinates (mirror).
    pub mirror: bool,
    /// Residue index correspondence mode (0-3).
    pub byresi: u8,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            ter_opt: 3,
            split_opt: 0,
            het_opt: false,
            atom_name: "auto".to_string(),
            infmt: InputFormat::Auto,
            mirror: false,
            byresi: 0,
        }
    }
}

/// Nucleotide residue names used to detect RNA/DNA.
const NUCLEOTIDE_NAMES: &[&str] = &[
    "A", "T", "G", "C", "U", "DA", "DT", "DG", "DC", "DU", "ADE", "THY", "GUA", "CYT", "URA",
];

fn is_nucleotide(name: &str) -> bool {
    NUCLEOTIDE_NAMES.contains(&name)
}

/// Load structure(s) from a PDB or mmCIF file using pdbtbx.
///
/// Returns one `StructureData` per chain (or per model, depending on `split_opt`).
pub fn load_structure(path: &str, opts: &LoadOptions) -> Result<Vec<StructureData>> {
    match opts.infmt {
        InputFormat::Spicker => {
            return crate::spicker::load_spicker(path);
        }
        InputFormat::Xyz => {
            return crate::xyz::load_xyz(path);
        }
        _ => {}
    }

    // Determine pdbtbx format
    let format = match opts.infmt {
        InputFormat::Pdb => pdbtbx::Format::Pdb,
        InputFormat::Mmcif => pdbtbx::Format::Mmcif,
        _ => pdbtbx::Format::Auto,
    };

    // Use minimal parsing level: we only need ATOM/HETATM coordinates.
    // This avoids errors from invalid space groups, ANISOU records, etc.
    // that are irrelevant to structure alignment.
    let mut parsing_level = pdbtbx::ParsingLevel::none();
    parsing_level
        .set_atom(true)
        .set_hetatm(opts.het_opt)
        .set_model(true);

    let mut read_opts = pdbtbx::ReadOptions::new();
    read_opts
        .set_format(format)
        .set_level(pdbtbx::StrictnessLevel::Loose)
        .set_parsing_level(&parsing_level)
        .set_discard_hydrogens(true);

    // Only first model if split_opt == 0 and ter_opt >= 1
    if opts.split_opt == 0 && opts.ter_opt >= 1 {
        read_opts.set_only_first_model(true);
    }

    let (pdb, _errors) = read_opts
        .read(path)
        .map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            anyhow::anyhow!(msg)
        })
        .with_context(|| format!("Failed to read {path}"))?;

    let mut results = Vec::new();

    // Determine atom name to search for
    let target_atom = if opts.atom_name == "auto" {
        // Will be determined per-chain based on molecule type
        None
    } else {
        Some(opts.atom_name.trim().to_string())
    };

    match opts.split_opt {
        0 => {
            // Treat entire structure as one chain
            let data = extract_structure_data(&pdb, None, None, &target_atom, opts, path);
            if !data.is_empty() {
                results.push(data);
            }
        }
        1 => {
            // Split by MODEL
            for model in pdb.models() {
                let data = extract_structure_data(
                    &pdb,
                    Some(model.serial_number()),
                    None,
                    &target_atom,
                    opts,
                    path,
                );
                if !data.is_empty() {
                    results.push(data);
                }
            }
        }
        2 => {
            // Split by chain — only first model
            if let Some(model) = pdb.models().next() {
                for chain in model.chains() {
                    let data = extract_structure_data(
                        &pdb,
                        Some(model.serial_number()),
                        Some(chain.id()),
                        &target_atom,
                        opts,
                        path,
                    );
                    if !data.is_empty() {
                        results.push(data);
                    }
                }
            }
        }
        _ => {}
    }

    if results.is_empty() {
        bail!("No atoms found in {path}");
    }

    Ok(results)
}

/// Extract a StructureData from a pdbtbx PDB, optionally filtering by model/chain.
fn extract_structure_data(
    pdb: &pdbtbx::PDB,
    model_filter: Option<usize>,
    chain_filter: Option<&str>,
    target_atom: &Option<String>,
    opts: &LoadOptions,
    path: &str,
) -> StructureData {
    let mut coords: Vec<Coord3D> = Vec::new();
    let mut sequence: Vec<char> = Vec::new();
    let mut resi_ids: Vec<String> = Vec::new();
    let mut pdb_lines: Vec<String> = Vec::new();
    let mut chain_id = String::new();
    let mut nuc_count = 0i32;
    let mut prot_count = 0i32;

    for model in pdb.models() {
        if let Some(m) = model_filter {
            if model.serial_number() != m {
                continue;
            }
        }

        for chain in model.chains() {
            if let Some(cf) = chain_filter {
                if chain.id() != cf {
                    continue;
                }
            }

            if chain_id.is_empty() {
                chain_id = chain.id().to_string();
            }

            for residue in chain.residues() {
                // Get the first conformer (altloc ' ' or 'A')
                let conformer = residue
                    .conformers()
                    .find(|c| match c.alternative_location() {
                        None => true,
                        Some(loc) => loc == "A",
                    });

                let conformer = match conformer {
                    Some(c) => c,
                    None => {
                        // Fall back to first conformer
                        match residue.conformers().next() {
                            Some(c) => c,
                            None => continue,
                        }
                    }
                };

                let res_name = conformer.name();
                let is_nuc = is_nucleotide(res_name);

                // Determine target atom name
                // C++ auto mode: always looks for CA first. Only uses C3'
                // if explicitly set via -atom or if mol type is pre-determined.
                let atom_name = match target_atom {
                    Some(ref name) => name.as_str(),
                    None => "CA",
                };

                // Find the target atom
                let atom = conformer.atoms().find(|a| {
                    let name_match = a.name().trim() == atom_name;
                    let het_ok = !a.hetero() || opts.het_opt;
                    name_match && het_ok
                });

                if let Some(atom) = atom {
                    let (x, y, z) = atom.pos();
                    let coord = if opts.mirror { [x, y, -z] } else { [x, y, z] };
                    coords.push(coord);

                    // Sequence
                    let aa = if is_nuc {
                        nuc_count += 1;
                        // Use lowercase for nucleotides (matches C++ behavior)
                        res_name.chars().next().unwrap_or('x').to_ascii_lowercase()
                    } else {
                        prot_count += 1;
                        three_to_one(res_name)
                    };
                    sequence.push(aa);

                    // Residue ID for byresi
                    let resi_id = match opts.byresi {
                        2 | 3 => {
                            let ins = residue.insertion_code().unwrap_or("");
                            format!("{:>4}{}{}", residue.serial_number(), ins, chain.id())
                        }
                        1 => {
                            let ins = residue.insertion_code().unwrap_or("");
                            format!("{:>4}{}", residue.serial_number(), ins)
                        }
                        _ => String::new(),
                    };
                    resi_ids.push(resi_id);

                    // Build a PDB-format line for superpose output
                    let het_tag = if atom.hetero() { "HETATM" } else { "ATOM  " };
                    let line = format!(
                        "{}{:>5} {:^4}{}{:>3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}",
                        het_tag,
                        atom.serial_number(),
                        atom.name(),
                        conformer.alternative_location().unwrap_or(" "),
                        res_name,
                        chain.id().chars().next().unwrap_or(' '),
                        residue.serial_number(),
                        residue.insertion_code().unwrap_or(" "),
                        x,
                        y,
                        z,
                    );
                    pdb_lines.push(line);
                }
            }

            // ter_opt chain termination logic (matches C++ behavior):
            // C++ reads through chains, skipping those with no matching atoms.
            // It only stops at TER/chain-change AFTER finding at least one atom.
            // So: once we've found atoms in a chain, stop at the next chain boundary.
            if chain_filter.is_none() && opts.ter_opt >= 2 && !coords.is_empty() {
                // We found atoms in this chain — don't continue to next chain
                break;
            }
        }

        // Only process first model if split_opt == 0
        if model_filter.is_none() {
            break;
        }
    }

    let mol_type = if nuc_count > prot_count {
        MolType::RNA
    } else {
        MolType::Protein
    };

    let sec_structure = make_sec(&coords);

    StructureData {
        coords,
        sequence,
        sec_structure,
        resi_ids,
        chain_id,
        mol_type,
        source_path: path.to_string(),
        pdb_lines,
    }
}
