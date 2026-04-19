//! FASTA alignment file reader for -i/-I options.

use anyhow::{bail, Result};
use std::fs;

/// Read a pairwise FASTA alignment file.
///
/// Returns two aligned sequences (with gaps as '-').
pub fn read_alignment(path: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let mut sequences: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('>') {
            if !current.is_empty() {
                sequences.push(current.clone());
                current.clear();
            }
            if sequences.len() >= 2 {
                break;
            }
        } else {
            current.push_str(line);
        }
    }
    if !current.is_empty() {
        sequences.push(current);
    }

    if sequences.len() != 2 {
        bail!(
            "Alignment file must contain exactly 2 sequences, found {}",
            sequences.len()
        );
    }
    if sequences[0].len() != sequences[1].len() {
        bail!(
            "Alignment sequences must have equal length: {} vs {}",
            sequences[0].len(),
            sequences[1].len()
        );
    }

    Ok(sequences)
}
