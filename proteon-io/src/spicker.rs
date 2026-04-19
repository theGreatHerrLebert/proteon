//! SPICKER format parser.
//!
//! SPICKER format: first line is residue count L,
//! then L lines of "x y z" coordinates.

use anyhow::{bail, Result};
use std::fs;

use proteon_align::core::secondary_structure::make_sec;
use proteon_align::core::types::{MolType, StructureData};

pub fn load_spicker(path: &str) -> Result<Vec<StructureData>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines();

    let first = lines.next().unwrap_or("").trim();
    let len: usize = first
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid SPICKER length: {first}"))?;
    if len == 0 {
        bail!("Empty SPICKER file");
    }

    let mut coords = Vec::with_capacity(len);
    let mut sequence = Vec::with_capacity(len);

    for _ in 0..len {
        let line = lines.next().unwrap_or("").trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            bail!("Invalid SPICKER coordinate line: {line}");
        }
        let x: f64 = parts[0].parse()?;
        let y: f64 = parts[1].parse()?;
        let z: f64 = parts[2].parse()?;
        coords.push([x, y, z]);
        sequence.push('A'); // SPICKER doesn't have sequence info
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
