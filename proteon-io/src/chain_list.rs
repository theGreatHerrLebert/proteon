//! Chain list file reader for -dir/-dir1/-dir2 batch modes.

use anyhow::Result;
use std::fs;

/// Read a chain list file and construct full paths.
///
/// Each non-empty line is a chain entry name.
/// Returns full paths: `dir + entry + suffix`.
pub fn read_chain_list(list_path: &str, dir: &str, suffix: &str) -> Result<Vec<String>> {
    let content = fs::read_to_string(list_path)?;
    let chains: Vec<String> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|entry| format!("{}{}{}", dir, entry, suffix))
        .collect();
    Ok(chains)
}
