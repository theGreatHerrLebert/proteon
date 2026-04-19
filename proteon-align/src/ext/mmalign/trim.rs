//! Complex trimming for multi-chain alignment.
//!
//! When chains are very large, keep only the residues closest to partner chains
//! (interfacial residues) to reduce alignment complexity.
//!
//! Ported from C++ USAlign `MMalign.h`: `trimComplex`.

use crate::core::types::{dist_squared, MolType};

use super::ChainData;

/// Trim large chains in a complex, keeping residues nearest to partner chains.
///
/// For each chain longer than the maximum allowed length (scaled by 2x expansion),
/// keeps only the `Lchain_max` residues with the smallest distance to any residue
/// in any other chain.
///
/// Returns a new `Vec<ChainData>` with trimmed chains and the count of chains
/// that were trimmed.
///
/// Corresponds to C++ `trimComplex`.
pub fn trim_complex(
    chains: &[ChainData],
    lchain_aa_max: usize,
    lchain_na_max: usize,
) -> (Vec<ChainData>, usize) {
    let chain_num = chains.len();
    let expand = 2.0f64;
    let mut trim_count = 0usize;
    let mut result = Vec::with_capacity(chain_num);

    for i in 0..chain_num {
        let xlen = chains[i].len();
        let lchain_max_raw = match chains[i].mol_type {
            MolType::RNA => (lchain_na_max as f64 * expand) as usize,
            MolType::Protein => (lchain_aa_max as f64 * expand) as usize,
        };
        let lchain_max = lchain_max_raw.max(3);

        if xlen <= lchain_max || xlen <= 3 {
            // No trimming needed
            result.push(chains[i].clone());
            continue;
        }

        trim_count += 1;

        // Compute minimum distance from each residue to any residue in other chains
        let mut dinter_vec: Vec<(f64, usize)> = Vec::with_capacity(xlen);

        for r1 in 0..xlen {
            let x = &chains[i].coords[r1];
            let mut dmin = f64::MAX;

            for j in 0..chain_num {
                if i == j {
                    continue;
                }
                for r2 in 0..chains[j].len() {
                    let d = dist_squared(x, &chains[j].coords[r2]);
                    if d < dmin {
                        dmin = d;
                    }
                }
            }
            dinter_vec.push((dmin, r1));
        }

        // Sort by distance (ascending), keep the closest lchain_max residues
        dinter_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut include = vec![false; xlen];
        for k in 0..lchain_max.min(dinter_vec.len()) {
            include[dinter_vec[k].1] = true;
        }

        // Build trimmed chain (preserving residue order)
        let mut new_coords = Vec::with_capacity(lchain_max);
        let mut new_seq = Vec::with_capacity(lchain_max);
        let mut new_sec = Vec::with_capacity(lchain_max);

        for r1 in 0..xlen {
            if !include[r1] {
                continue;
            }
            new_coords.push(chains[i].coords[r1]);
            new_seq.push(chains[i].sequence[r1]);
            new_sec.push(chains[i].sec_structure[r1]);
        }

        result.push(ChainData {
            coords: new_coords,
            sequence: new_seq,
            sec_structure: new_sec,
            chain_id: chains[i].chain_id.clone(),
            mol_type: chains[i].mol_type,
        });
    }

    (result, trim_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(n: usize, offset: f64, mol: MolType) -> ChainData {
        ChainData {
            coords: (0..n).map(|i| [offset + i as f64, 0.0, 0.0]).collect(),
            sequence: vec![b'A'; n],
            sec_structure: vec![b'C'; n],
            chain_id: "X".to_string(),
            mol_type: mol,
        }
    }

    #[test]
    fn test_trim_no_trimming_needed() {
        let chains = vec![
            make_chain(10, 0.0, MolType::Protein),
            make_chain(5, 20.0, MolType::Protein),
        ];
        let (trimmed, count) = trim_complex(&chains, 100, 100);
        assert_eq!(count, 0);
        assert_eq!(trimmed[0].len(), 10);
        assert_eq!(trimmed[1].len(), 5);
    }

    #[test]
    fn test_trim_large_chain() {
        // Chain A has 20 residues, chain B has 5 residues far away
        // With lchain_aa_max=5, expanded=10, chain A (20 > 10) should be trimmed
        let chains = vec![
            make_chain(20, 0.0, MolType::Protein),
            make_chain(5, 100.0, MolType::Protein),
        ];
        let (trimmed, count) = trim_complex(&chains, 5, 5);
        assert_eq!(count, 1);
        // Chain A trimmed to 10 (5 * 2)
        assert_eq!(trimmed[0].len(), 10);
        // Chain B untouched (5 <= 10)
        assert_eq!(trimmed[1].len(), 5);
    }

    #[test]
    fn test_trim_keeps_nearest_residues() {
        // Chain A at x=[0..20], chain B at x=5 (only 1 residue)
        // After trimming, residues nearest to x=5 should be kept
        let chain_a = ChainData {
            coords: (0..20).map(|i| [i as f64, 0.0, 0.0]).collect(),
            sequence: vec![b'A'; 20],
            sec_structure: vec![b'C'; 20],
            chain_id: "A".to_string(),
            mol_type: MolType::Protein,
        };
        let chain_b = ChainData {
            coords: vec![[5.0, 0.0, 0.0]],
            sequence: vec![b'A'],
            sec_structure: vec![b'C'],
            chain_id: "B".to_string(),
            mol_type: MolType::Protein,
        };
        let chains = vec![chain_a, chain_b];
        // lchain_aa_max=3 => expanded=6, chain A (20>6) trimmed to 6
        let (trimmed, count) = trim_complex(&chains, 3, 3);
        assert_eq!(count, 1);
        assert_eq!(trimmed[0].len(), 6);
        // The kept residues should be near x=5 (residues 2..8 or similar)
        for c in &trimmed[0].coords {
            assert!(
                (c[0] - 5.0).abs() <= 4.0,
                "residue at x={} is too far from partner",
                c[0]
            );
        }
    }
}
