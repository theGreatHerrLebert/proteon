//! General PDB/mmCIF I/O via pdbtbx.
//!
//! This module provides thin convenience wrappers around pdbtbx for
//! loading and saving structural files, used by both the CLI and the
//! PyO3 connector.

use anyhow::{bail, Context, Result};

use proteon_align::core::residue_map::three_to_one;
use proteon_align::core::secondary_structure::make_sec;
use proteon_align::core::types::{Coord3D, MolType, StructureData};

/// Load a PDB/mmCIF file with loose strictness (auto-detect format).
/// Returns just the PDB struct; warnings are discarded.
pub fn load(path: &str) -> Result<pdbtbx::PDB> {
    let (pdb, _warnings) = pdbtbx::ReadOptions::new()
        .set_level(pdbtbx::StrictnessLevel::Loose)
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
    Ok(pdb)
}

/// Load forcing PDB format.
pub fn load_pdb(path: &str) -> Result<pdbtbx::PDB> {
    let (pdb, _warnings) = pdbtbx::ReadOptions::new()
        .set_format(pdbtbx::Format::Pdb)
        .set_level(pdbtbx::StrictnessLevel::Loose)
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
    Ok(pdb)
}

/// Load forcing mmCIF format.
pub fn load_mmcif(path: &str) -> Result<pdbtbx::PDB> {
    let (pdb, _warnings) = pdbtbx::ReadOptions::new()
        .set_format(pdbtbx::Format::Mmcif)
        .set_level(pdbtbx::StrictnessLevel::Loose)
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
    Ok(pdb)
}

/// Save a PDB structure (format auto-detected from extension).
pub fn save(pdb: &pdbtbx::PDB, path: &str) -> Result<()> {
    pdbtbx::save(pdb, path, pdbtbx::StrictnessLevel::Loose).map_err(|errs| {
        let msg = errs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::anyhow!("Failed to save {path}: {msg}")
    })
}

// ---------------------------------------------------------------------------
// Structure extraction for alignment
// ---------------------------------------------------------------------------

/// Nucleotide residue names used to detect RNA/DNA.
const NUCLEOTIDE_NAMES: &[&str] = &[
    "A", "T", "G", "C", "U", "DA", "DT", "DG", "DC", "DU", "ADE", "THY", "GUA", "CYT", "URA",
];

/// Extract CA (or C3') coordinates and sequence from a loaded PDB for alignment.
///
/// Uses the first model, optionally filtered to a single chain.
/// Auto-detects protein (CA) vs RNA (C3') from residue composition.
///
/// Returns a `StructureData` suitable for `tmalign()` and friends.
pub fn extract_for_alignment(pdb: &pdbtbx::PDB, chain: Option<&str>) -> Result<StructureData> {
    let mut coords: Vec<Coord3D> = Vec::new();
    let mut sequence: Vec<char> = Vec::new();
    let mut chain_id = String::new();
    let mut nuc_count = 0u32;
    let mut prot_count = 0u32;

    let model = pdb.models().next();
    let model = match model {
        Some(m) => m,
        None => bail!("PDB has no models"),
    };

    for ch in model.chains() {
        if let Some(cf) = chain {
            if ch.id() != cf {
                continue;
            }
        }

        if chain_id.is_empty() {
            chain_id = ch.id().to_string();
        }

        for residue in ch.residues() {
            // Pick primary conformer (altloc blank or 'A')
            let conformer = residue
                .conformers()
                .find(|c| match c.alternative_location() {
                    None => true,
                    Some(loc) => loc == "A",
                })
                .or_else(|| residue.conformers().next());

            let conformer = match conformer {
                Some(c) => c,
                None => continue,
            };

            let res_name = conformer.name();
            let is_nuc = NUCLEOTIDE_NAMES.contains(&res_name);

            // Auto-detect atom name: CA for protein, C3' for nucleotides
            let atom_name = if is_nuc { "C3'" } else { "CA" };

            let atom = conformer
                .atoms()
                .find(|a| a.name().trim() == atom_name && !a.hetero());

            if let Some(atom) = atom {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);

                let aa = if is_nuc {
                    nuc_count += 1;
                    res_name.chars().next().unwrap_or('x').to_ascii_lowercase()
                } else {
                    prot_count += 1;
                    three_to_one(res_name)
                };
                sequence.push(aa);
            }
        }
    }

    if coords.is_empty() {
        let id = pdb.identifier.as_deref().unwrap_or("unknown");
        bail!(
            "No CA/C3' atoms found in {id}{}",
            chain.map_or(String::new(), |c| format!(" chain {c}"))
        );
    }

    let mol_type = if nuc_count > prot_count {
        MolType::RNA
    } else {
        MolType::Protein
    };

    let sec_structure = make_sec(&coords);

    Ok(StructureData {
        coords,
        sequence,
        sec_structure,
        resi_ids: Vec::new(),
        chain_id,
        mol_type,
        source_path: pdb.identifier.as_deref().unwrap_or("").to_string(),
        pdb_lines: Vec::new(),
    })
}

/// Extract per-chain `ChainData` from a loaded PDB for multi-chain alignment.
///
/// Returns one `ChainData` per chain in the first model.
pub fn extract_chains_for_alignment(
    pdb: &pdbtbx::PDB,
) -> Result<Vec<proteon_align::ext::mmalign::ChainData>> {
    use proteon_align::ext::mmalign::ChainData;

    let model = pdb.models().next();
    let model = match model {
        Some(m) => m,
        None => bail!("PDB has no models"),
    };

    let mut chains = Vec::new();

    for ch in model.chains() {
        let mut coords: Vec<Coord3D> = Vec::new();
        let mut sequence: Vec<u8> = Vec::new();
        let mut nuc_count = 0u32;
        let mut prot_count = 0u32;

        for residue in ch.residues() {
            let conformer = residue
                .conformers()
                .find(|c| match c.alternative_location() {
                    None => true,
                    Some(loc) => loc == "A",
                })
                .or_else(|| residue.conformers().next());

            let conformer = match conformer {
                Some(c) => c,
                None => continue,
            };

            let res_name = conformer.name();
            let is_nuc = NUCLEOTIDE_NAMES.contains(&res_name);
            let atom_name = if is_nuc { "C3'" } else { "CA" };

            let atom = conformer
                .atoms()
                .find(|a| a.name().trim() == atom_name && !a.hetero());

            if let Some(atom) = atom {
                let (x, y, z) = atom.pos();
                coords.push([x, y, z]);

                let aa = if is_nuc {
                    nuc_count += 1;
                    res_name.chars().next().unwrap_or(b'x' as char) as u8
                } else {
                    prot_count += 1;
                    three_to_one(res_name) as u8
                };
                sequence.push(aa);
            }
        }

        if coords.is_empty() {
            continue;
        }

        let mol_type = if nuc_count > prot_count {
            MolType::RNA
        } else {
            MolType::Protein
        };

        let sec_structure: Vec<u8> = make_sec(&coords).iter().map(|c| *c as u8).collect();

        chains.push(ChainData {
            coords,
            sequence,
            sec_structure,
            chain_id: ch.id().to_string(),
            mol_type,
        });
    }

    if chains.is_empty() {
        let id = pdb.identifier.as_deref().unwrap_or("unknown");
        bail!("No chains with CA/C3' atoms found in {id}");
    }

    Ok(chains)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_path(name: &str) -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        // Try test-pdbs in repo first, fall back to pdbtbx example-pdbs
        let repo_path = format!("{manifest}/../test-pdbs/{name}");
        if std::path::Path::new(&repo_path).exists() {
            return repo_path;
        }
        format!("{manifest}/../../pdbtbx/example-pdbs/{name}")
    }

    #[test]
    fn test_load_pdb_auto() {
        let pdb = load(&example_path("1ubq.pdb")).expect("should load 1ubq.pdb");
        assert!(pdb.model_count() >= 1);
        assert!(pdb.chain_count() >= 1);
        assert!(pdb.atom_count() > 0);
    }

    #[test]
    fn test_load_pdb_explicit_format() {
        let pdb = load_pdb(&example_path("1ubq.pdb")).expect("should load as PDB");
        assert!(pdb.atom_count() > 0);
    }

    #[test]
    fn test_load_mmcif() {
        let pdb = load_mmcif(&example_path("1ubq.cif")).expect("should load 1ubq.cif");
        assert!(pdb.atom_count() > 0);
        assert!(pdb.chain_count() >= 1);
    }

    #[test]
    fn test_pdb_and_cif_both_load() {
        let pdb_struct = load_pdb(&example_path("1ubq.pdb")).expect("load pdb");
        let cif_struct = load_mmcif(&example_path("1ubq.cif")).expect("load cif");
        // Both should load successfully with atoms
        assert!(pdb_struct.atom_count() > 0);
        assert!(cif_struct.atom_count() > 0);
        // Chain counts should match (same structure)
        assert_eq!(
            pdb_struct.chain_count(),
            cif_struct.chain_count(),
            "PDB and CIF chain counts should match for 1ubq"
        );
    }

    #[test]
    fn test_hierarchy_counts() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        // 1UBQ: 1 model, 1 chain (A), 76 residues, 660 atoms (no H)
        assert_eq!(pdb.model_count(), 1);
        assert!(pdb.chain_count() >= 1);
        assert!(pdb.residue_count() > 50, "expected >50 residues for 1ubq");
        assert!(pdb.atom_count() > 500, "expected >500 atoms for 1ubq");
    }

    #[test]
    fn test_hierarchy_traversal() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let mut atom_count_manual = 0usize;
        for model in pdb.models() {
            for chain in model.chains() {
                assert!(!chain.id().is_empty(), "chain id should not be empty");
                for residue in chain.residues() {
                    assert!(residue.name().is_some(), "residue should have a name");
                    for _atom in residue.atoms() {
                        atom_count_manual += 1;
                    }
                }
            }
        }
        assert_eq!(
            atom_count_manual,
            pdb.total_atom_count(),
            "manual traversal atom count should match total_atom_count"
        );
    }

    #[test]
    fn test_atom_properties() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let first_atom = pdb.atoms().next().expect("should have at least one atom");
        let (x, y, z) = first_atom.pos();
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
        assert!(first_atom.element().is_some());
        assert!(first_atom.b_factor() >= 0.0);
        assert!(first_atom.occupancy() >= 0.0);
        assert!(first_atom.occupancy() <= 1.0);
        assert!(!first_atom.name().trim().is_empty());
    }

    #[test]
    fn test_multi_model() {
        let pdb = load(&example_path("models.pdb")).expect("load models.pdb");
        assert!(
            pdb.model_count() > 1,
            "models.pdb should have multiple models"
        );
        let first_count = pdb.models().next().unwrap().atom_count();
        for model in pdb.models() {
            assert_eq!(
                model.atom_count(),
                first_count,
                "all models should have same atom count"
            );
        }
    }

    #[test]
    fn test_save_roundtrip() {
        let original = load(&example_path("1ubq.pdb")).expect("load");
        let tmp_path = "/tmp/proteon_test_roundtrip.pdb";
        save(&original, tmp_path).expect("save should succeed");
        let reloaded = load(tmp_path).expect("reload should succeed");
        assert_eq!(
            original.atom_count(),
            reloaded.atom_count(),
            "roundtrip should preserve atom count"
        );
        let orig_atoms: Vec<_> = original.atoms().collect();
        let reload_atoms: Vec<_> = reloaded.atoms().collect();
        for (o, r) in orig_atoms.iter().zip(reload_atoms.iter()) {
            let (ox, oy, oz) = o.pos();
            let (rx, ry, rz) = r.pos();
            assert!((ox - rx).abs() < 0.001, "x mismatch");
            assert!((oy - ry).abs() < 0.001, "y mismatch");
            assert!((oz - rz).abs() < 0.001, "z mismatch");
        }
        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn test_coordinate_extraction() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let coords: Vec<(f64, f64, f64)> = pdb.atoms().map(|a| a.pos()).collect();
        assert_eq!(coords.len(), pdb.atom_count());
        for &(x, y, z) in &coords {
            assert!(x.is_finite() && y.is_finite() && z.is_finite());
        }
    }

    #[test]
    fn test_b_factor_extraction() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let bfactors: Vec<f64> = pdb.atoms().map(|a| a.b_factor()).collect();
        assert_eq!(bfactors.len(), pdb.atom_count());
        for &b in &bfactors {
            assert!(b.is_finite());
        }
    }

    #[test]
    fn test_residue_names() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        for chain in pdb.chains() {
            for residue in chain.residues() {
                let name = residue.name();
                assert!(name.is_some(), "residue should have a name");
                assert!(
                    !name.unwrap().is_empty(),
                    "residue name should not be empty"
                );
            }
        }
    }

    #[test]
    fn test_chain_ids() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        for chain in pdb.chains() {
            assert!(!chain.id().is_empty(), "chain id should not be empty");
        }
    }

    #[test]
    fn test_element_symbols() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let mut has_carbon = false;
        let mut has_nitrogen = false;
        for atom in pdb.atoms() {
            if let Some(elem) = atom.element() {
                match elem.symbol() {
                    "C" => has_carbon = true,
                    "N" => has_nitrogen = true,
                    _ => {}
                }
            }
        }
        assert!(has_carbon, "protein should contain carbon");
        assert!(has_nitrogen, "protein should contain nitrogen");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load("/nonexistent/path.pdb");
        assert!(result.is_err());
    }

    // -- extract_for_alignment tests ----------------------------------------

    #[test]
    fn test_extract_for_alignment_basic() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let sd = extract_for_alignment(&pdb, None).expect("extract");
        assert!(!sd.is_empty());
        assert_eq!(sd.coords.len(), sd.sequence.len());
        assert_eq!(sd.coords.len(), sd.sec_structure.len());
        assert!(!sd.chain_id.is_empty());
    }

    #[test]
    fn test_extract_for_alignment_residue_count() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let sd = extract_for_alignment(&pdb, None).expect("extract");
        // 1UBQ has 76 residues with CA atoms
        assert_eq!(sd.coords.len(), 76, "1ubq should have 76 CA atoms");
    }

    #[test]
    fn test_extract_for_alignment_chain_filter() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let sd = extract_for_alignment(&pdb, Some("A")).expect("extract chain A");
        assert_eq!(sd.chain_id, "A");
        assert!(!sd.is_empty());
    }

    #[test]
    fn test_extract_for_alignment_nonexistent_chain() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let result = extract_for_alignment(&pdb, Some("Z"));
        assert!(result.is_err(), "chain Z should not exist");
    }

    #[test]
    fn test_extract_for_alignment_mol_type() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let sd = extract_for_alignment(&pdb, None).expect("extract");
        assert!(
            matches!(sd.mol_type, proteon_align::core::types::MolType::Protein),
            "1ubq should be detected as protein"
        );
    }

    #[test]
    fn test_extract_for_alignment_sec_structure() {
        let pdb = load(&example_path("1ubq.pdb")).expect("load");
        let sd = extract_for_alignment(&pdb, None).expect("extract");
        // Secondary structure should be one of H, E, T, C
        for &c in &sd.sec_structure {
            assert!("HETC".contains(c), "unexpected sec structure char: {c}");
        }
    }
}
