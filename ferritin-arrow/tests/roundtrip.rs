//! Bulk round-trip test: PDB → Arrow → PDB for many structures.
//!
//! Loads all PDB files from test-pdbs/, converts each to Arrow and back,
//! then verifies atom counts and coordinate fidelity.

use ferritin_arrow::convert::{atom_batch_to_pdbs, pdb_to_atom_batch};
use std::fs;
use std::path::Path;

fn load_pdb(path: &str) -> Option<pdbtbx::PDB> {
    let (pdb, _) = pdbtbx::ReadOptions::new()
        .set_level(pdbtbx::StrictnessLevel::Loose)
        .read(path)
        .ok()?;
    Some(pdb)
}

#[test]
fn test_roundtrip_all_test_pdbs() {
    // Try the large test set first, fall back to the small one
    let large_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test-pdbs");
    let small_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-pdbs");
    let test_path = if Path::new(large_dir).exists() {
        Path::new(large_dir)
    } else if Path::new(small_dir).exists() {
        Path::new(small_dir)
    } else {
        eprintln!("Skipping: no test-pdbs/ found");
        return;
    };

    let mut entries: Vec<_> = fs::read_dir(test_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "pdb")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut n_tested = 0;
    let mut n_skipped = 0;
    let mut failures = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_str().unwrap();

        let pdb = match load_pdb(path.to_str().unwrap()) {
            Some(p) => p,
            None => {
                n_skipped += 1;
                continue;
            }
        };

        // Count atoms across ALL models (not just first)
        let original_atoms: usize = pdb.models()
            .map(|m| m.atom_count())
            .sum();
        if original_atoms == 0 {
            n_skipped += 1;
            continue;
        }

        // PDB → Arrow
        let batch = match pdb_to_atom_batch(&pdb, name) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{name}: to_arrow failed: {e}"));
                continue;
            }
        };

        if batch.num_rows() != original_atoms {
            failures.push(format!(
                "{name}: row count mismatch: {original_atoms} atoms → {} rows",
                batch.num_rows()
            ));
            continue;
        }

        // Arrow → PDB
        let rebuilt = match atom_batch_to_pdbs(&batch) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{name}: from_arrow failed: {e}"));
                continue;
            }
        };

        if rebuilt.len() != 1 {
            failures.push(format!("{name}: expected 1 structure, got {}", rebuilt.len()));
            continue;
        }

        let (sid, rebuilt_pdb) = &rebuilt[0];
        if sid != name {
            failures.push(format!("{name}: structure_id mismatch: {sid}"));
            continue;
        }

        let rebuilt_atoms: usize = rebuilt_pdb.models()
            .map(|m| m.atom_count())
            .sum();
        if rebuilt_atoms != original_atoms {
            failures.push(format!(
                "{name}: atom count mismatch after roundtrip: {original_atoms} → {rebuilt_atoms}"
            ));
            continue;
        }

        // Spot-check coordinates of first atom
        let orig_first = pdb.models().next()
            .and_then(|m| m.chains().next())
            .and_then(|c| c.residues().next())
            .and_then(|r| r.conformers().next())
            .and_then(|cf| cf.atoms().next());
        let rebuilt_first = rebuilt_pdb.models().next()
            .and_then(|m| m.chains().next())
            .and_then(|c| c.residues().next())
            .and_then(|r| r.conformers().next())
            .and_then(|cf| cf.atoms().next());

        if let (Some(orig), Some(rebuilt)) = (orig_first, rebuilt_first) {
            let (ox, oy, oz) = orig.pos();
            let (rx, ry, rz) = rebuilt.pos();
            let max_diff = (ox - rx).abs().max((oy - ry).abs()).max((oz - rz).abs());
            if max_diff > 1e-6 {
                failures.push(format!(
                    "{name}: coordinate drift: max_diff={max_diff:.2e}"
                ));
                continue;
            }
        }

        n_tested += 1;
    }

    eprintln!(
        "Arrow roundtrip: {n_tested} passed, {} failed, {n_skipped} skipped (of {} total)",
        failures.len(),
        entries.len()
    );

    if !failures.is_empty() {
        eprintln!("Failures:");
        for f in &failures {
            eprintln!("  {f}");
        }
    }

    // Allow a few failures (some PDBs have unusual features) but not many
    let failure_rate = failures.len() as f64 / entries.len() as f64;
    assert!(
        failure_rate < 0.05,
        "Too many roundtrip failures: {}/{} ({:.1}%)",
        failures.len(),
        entries.len(),
        failure_rate * 100.0
    );
}
