//! XYZ format parser.
//!
//! XYZ format: first line is atom count L, second is description,
//! then L lines of "atom x y z".

use anyhow::{bail, Result};
use std::fs;

use proteon_align::core::residue_map::three_to_one;
use proteon_align::core::secondary_structure::make_sec;
use proteon_align::core::types::{MolType, StructureData};

pub fn load_xyz(path: &str) -> Result<Vec<StructureData>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();

    let first = lines.next().unwrap_or("").trim();
    let len: usize = first
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid XYZ atom count: {first}"))?;
    if len == 0 {
        bail!("Empty XYZ file");
    }

    // Skip description line
    let _ = lines.next();

    let mut coords = Vec::with_capacity(len);
    let mut sequence = Vec::with_capacity(len);

    for _ in 0..len {
        let line = lines.next().unwrap_or("").trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            bail!("Invalid XYZ coordinate line: {line}");
        }
        // First field is atom name or residue code
        let aa = if parts[0].len() == 3 {
            three_to_one(parts[0])
        } else {
            parts[0].chars().next().unwrap_or('X')
        };
        let x: f64 = parts[1].parse()?;
        let y: f64 = parts[2].parse()?;
        let z: f64 = parts[3].parse()?;
        coords.push([x, y, z]);
        sequence.push(aa);
    }

    let sec = make_sec(&coords);
    Ok(vec![StructureData {
        coords,
        sequence,
        sec_structure: sec,
        resi_ids: Vec::new(),
        chain_id: String::new(),
        mol_type: MolType::Protein,
        source_path: path.to_string(),
        pdb_lines: Vec::new(),
    }])
}
