//! Multi-chain (MM) alignment algorithm.
//!
//! Ported from C++ USAlign `MMalign.h`.
//! Aligns two protein/RNA complexes by determining optimal chain-to-chain
//! correspondence, superposing concatenated chains, and iteratively refining.

pub mod chain_assign;
pub mod complex_score;
pub mod dimer;
pub mod iter;
pub mod trim;

use crate::core::types::{Coord3D, MolType, Transform};

use crate::ext::se::SeResult;

/// Per-chain structure data for multi-chain alignment.
#[derive(Debug, Clone)]
pub struct ChainData {
    /// CA (or C3') coordinates.
    pub coords: Vec<Coord3D>,
    /// One-letter amino acid or nucleotide codes.
    pub sequence: Vec<u8>,
    /// Secondary structure assignment.
    pub sec_structure: Vec<u8>,
    /// Chain identifier.
    pub chain_id: String,
    /// Molecule type (protein vs RNA).
    pub mol_type: MolType,
}

impl ChainData {
    /// Number of residues in this chain.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether this chain has no residues.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Result of multi-chain alignment.
#[derive(Debug, Clone)]
pub struct MMAlignResult {
    /// Total TM-score of the complex alignment.
    pub total_score: f64,
    /// Chain assignment pairs: `(chain_i_in_complex1, chain_j_in_complex2)`.
    pub chain_assignments: Vec<(usize, usize)>,
    /// Per-chain SE alignment results (one per assigned chain pair).
    pub per_chain_results: Vec<SeResult>,
    /// Per-chain transforms (one per assigned chain pair).
    pub transforms: Vec<Transform>,
}

/// Count the number of nucleic acid and protein chains.
///
/// Returns `(na_count, aa_count)`.
pub fn count_na_aa_chains(mol_vec: &[MolType]) -> (usize, usize) {
    let mut na = 0usize;
    let mut aa = 0usize;
    for m in mol_vec {
        match m {
            MolType::RNA => na += 1,
            MolType::Protein => aa += 1,
        }
    }
    (na, aa)
}

/// Compute total chain lengths split by molecule type.
///
/// Returns `(len_aa, len_na)`.
pub fn total_chain_lengths(chains: &[ChainData]) -> (usize, usize) {
    let mut len_aa = 0usize;
    let mut len_na = 0usize;
    for c in chains {
        match c.mol_type {
            MolType::Protein => len_aa += c.len(),
            MolType::RNA => len_na += c.len(),
        }
    }
    (len_aa, len_na)
}

/// Concatenate assigned chain pairs into single coordinate/sequence arrays.
///
/// Corresponds to C++ `copy_chain_pair_data`.
/// Returns `(xa, ya, seqx, seqy, secx, secy, mol_type_sum)`.
pub fn concatenate_assigned_chains(
    x_chains: &[ChainData],
    y_chains: &[ChainData],
    assign1: &[i32],
    seqx_a_mat: &[Vec<String>],
    seqy_a_mat: &[Vec<String>],
) -> (
    Vec<Coord3D>,
    Vec<Coord3D>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    i32,
    Vec<String>,
) {
    let mut xa = Vec::new();
    let mut ya = Vec::new();
    let mut seqx = Vec::new();
    let mut seqy = Vec::new();
    let mut secx = Vec::new();
    let mut secy = Vec::new();
    let mut mol_type_sum = 0i32;
    let mut sequence = vec![String::new(), String::new()];

    for (i, &j) in assign1.iter().enumerate() {
        if j < 0 {
            continue;
        }
        let ju = j as usize;
        let xc = &x_chains[i];
        let yc = &y_chains[ju];

        xa.extend_from_slice(&xc.coords);
        seqx.extend_from_slice(&xc.sequence);
        secx.extend_from_slice(&xc.sec_structure);

        ya.extend_from_slice(&yc.coords);
        seqy.extend_from_slice(&yc.sequence);
        secy.extend_from_slice(&yc.sec_structure);

        let mol_i = match xc.mol_type {
            MolType::Protein => -1,
            MolType::RNA => 1,
        };
        let mol_j = match yc.mol_type {
            MolType::Protein => -1,
            MolType::RNA => 1,
        };
        mol_type_sum += mol_i + mol_j;

        if i < seqx_a_mat.len() && ju < seqx_a_mat[i].len() {
            sequence[0].push_str(&seqx_a_mat[i][ju]);
        }
        if i < seqy_a_mat.len() && ju < seqy_a_mat[i].len() {
            sequence[1].push_str(&seqy_a_mat[i][ju]);
        }
    }

    (xa, ya, seqx, seqy, secx, secy, mol_type_sum, sequence)
}

use anyhow::{bail, Result};

use crate::core::align::tmalign::tmalign;
use crate::core::types::AlignOptions;
use crate::ext::se::{se_main, SeOptions};

use chain_assign::{check_heterooligomer, enhanced_greedy_search, homo_refined_greedy_search};
use complex_score::calculate_centroids;
use iter::copy_chain_assign_data;

/// Align two multi-chain complexes using MM-align.
///
/// This is the top-level orchestrator that:
/// 1. Computes pairwise per-chain TM-scores via SE alignment
/// 2. Determines chain-to-chain assignment (homo vs hetero detection)
/// 3. Iteratively refines the assignment
///
/// Returns an `MMAlignResult` with chain assignments, per-chain results,
/// and an overall complex TM-score.
pub fn mmalign_complex(x_chains: &[ChainData], y_chains: &[ChainData]) -> Result<MMAlignResult> {
    let chain1_num = x_chains.len();
    let chain2_num = y_chains.len();

    if chain1_num == 0 || chain2_num == 0 {
        bail!("Both complexes must have at least one chain");
    }

    // Step 1: Pairwise per-chain alignment → TM matrix + alignment strings + transforms.
    //
    // C++ USalign.cpp:874-919 runs full TMalign_main per pair (the default path)
    // and stores the resulting u/t in ut_mat. Only the explicit `-se` flag
    // shortcuts to se_main with identity matrices. We follow the default path
    // here; downstream `homo_refined_greedy_search` uses the transforms to
    // superpose centroids when refining homo-oligomer chain assignments —
    // without them it can't see internal symmetries and picks bad pairings.
    let mut tm_mat = vec![vec![0.0f64; chain2_num]; chain1_num];
    let mut seqx_a_mat = vec![vec![String::new(); chain2_num]; chain1_num];
    let mut seqy_a_mat = vec![vec![String::new(); chain2_num]; chain1_num];
    let mut ut_mat: Vec<Vec<f64>> = vec![vec![0.0; 12]; chain1_num * chain2_num];

    let pair_align_opts = AlignOptions {
        i_opt: 0,
        a_opt: 0,
        u_opt: false,
        lnorm_ass: 0.0,
        d_opt: false,
        d0_scale: 0.0,
        fast_opt: true,
        mol_type: x_chains[0].mol_type,
        tm_cut: -1.0,
        user_alignment: None,
    };

    for i in 0..chain1_num {
        for j in 0..chain2_num {
            let xc = &x_chains[i];
            let yc = &y_chains[j];
            if xc.is_empty() || yc.is_empty() || xc.len() < 3 || yc.len() < 3 {
                // Identity for chains too short for TM-align (matches C++ early-out).
                ut_mat[i * chain2_num + j] =
                    vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
                continue;
            }

            let seqx_chars: Vec<char> = xc.sequence.iter().map(|&b| b as char).collect();
            let seqy_chars: Vec<char> = yc.sequence.iter().map(|&b| b as char).collect();
            let secx_chars: Vec<char> = xc.sec_structure.iter().map(|&b| b as char).collect();
            let secy_chars: Vec<char> = yc.sec_structure.iter().map(|&b| b as char).collect();

            let result = tmalign(
                &xc.coords,
                &yc.coords,
                &seqx_chars,
                &seqy_chars,
                &secx_chars,
                &secy_chars,
                &pair_align_opts,
            )?;

            tm_mat[i][j] = result.tm_score_chain1;
            seqx_a_mat[i][j] = result.aligned_seq_x.clone();
            seqy_a_mat[i][j] = result.aligned_seq_y.clone();

            let u = result.transform.u;
            let t = result.transform.t;
            ut_mat[i * chain2_num + j] = vec![
                u[0][0], u[0][1], u[0][2], u[1][0], u[1][1], u[1][2], u[2][0], u[2][1], u[2][2],
                t[0], t[1], t[2],
            ];
        }
    }
    let se_opts = SeOptions::default();

    // Step 2: Calculate centroids
    let x_coords_vec: Vec<Vec<Coord3D>> = x_chains.iter().map(|c| c.coords.clone()).collect();
    let y_coords_vec: Vec<Vec<Coord3D>> = y_chains.iter().map(|c| c.coords.clone()).collect();
    let (x_centroids, _) = calculate_centroids(&x_coords_vec);
    let (y_centroids, d0mm) = calculate_centroids(&y_coords_vec);
    let (len_aa, len_na) = total_chain_lengths(y_chains);
    let total_len = len_aa + len_na;

    // Step 3: Check heterooligomer
    let het_deg = check_heterooligomer(&tm_mat, chain1_num, chain2_num);
    let is_hetero = het_deg > 0.3;

    // Step 4: Initial chain assignment
    let (mut assign1, mut assign2, mut best_score) = if is_hetero || chain1_num <= 2 {
        enhanced_greedy_search(&tm_mat, chain1_num, chain2_num)
    } else {
        homo_refined_greedy_search(
            &tm_mat,
            chain1_num,
            chain2_num,
            &x_centroids,
            &y_centroids,
            d0mm,
            total_len,
            &ut_mat,
        )
    };

    // Step 5: Iterative refinement (inlined to avoid lifetime issues with callbacks)
    let max_iter = chain1_num.min(chain2_num) * 2 + 1;
    for _round in 0..max_iter {
        // Refine: re-align each assigned chain pair using SE with prior invmap
        for (i, &j) in assign1.iter().enumerate() {
            if j < 0 {
                continue;
            }
            let ju = j as usize;
            let xc = &x_chains[i];
            let yc = &y_chains[ju];
            if xc.is_empty() || yc.is_empty() {
                continue;
            }

            let prior = invmap_from_alignment(&seqx_a_mat[i][ju], &seqy_a_mat[i][ju], yc.len());

            let result = se_main(
                &xc.coords,
                &yc.coords,
                &xc.sequence,
                &yc.sequence,
                xc.len(),
                yc.len(),
                &se_opts,
                Some(&prior),
                0,
            );

            tm_mat[i][ju] = result.tm1;
            seqx_a_mat[i][ju] = result.seq_x_aligned;
            seqy_a_mat[i][ju] = result.seq_y_aligned;
        }

        // Re-assign chains
        let (new_a1, new_a2, new_score) = enhanced_greedy_search(&tm_mat, chain1_num, chain2_num);

        if new_score <= best_score {
            break; // no improvement
        }
        assign1 = new_a1;
        assign2 = new_a2;
        best_score = new_score;
    }

    // Step 6: Save best state
    let best_state = copy_chain_assign_data(
        chain1_num,
        chain2_num,
        &seqx_a_mat,
        &seqy_a_mat,
        &assign1,
        &assign2,
        &tm_mat,
    );

    // Step 7: Build result — final SE alignment for each assigned pair
    let mut chain_assignments = Vec::new();
    let mut per_chain_results = Vec::new();
    let mut transforms = Vec::new();

    for (i, &j) in best_state.assign1.iter().enumerate() {
        if j < 0 {
            continue;
        }
        let ju = j as usize;
        chain_assignments.push((i, ju));

        let xc = &x_chains[i];
        let yc = &y_chains[ju];
        let result = se_main(
            &xc.coords,
            &yc.coords,
            &xc.sequence,
            &yc.sequence,
            xc.len(),
            yc.len(),
            &se_opts,
            None,
            0,
        );
        transforms.push(Transform::default());
        per_chain_results.push(result);
    }

    Ok(MMAlignResult {
        total_score: best_score,
        chain_assignments,
        per_chain_results,
        transforms,
    })
}

/// Reconstruct invmap from alignment strings.
/// `invmap[j] = i` means y[j] aligns to x[i], -1 = gap.
fn invmap_from_alignment(seq_x: &str, seq_y: &str, ylen: usize) -> Vec<i32> {
    let sx = seq_x.as_bytes();
    let sy = seq_y.as_bytes();
    let len = sx.len().min(sy.len());
    let mut invmap = vec![-1i32; ylen];
    let mut xi: i32 = -1;
    let mut yj: i32 = -1;
    for k in 0..len {
        if sx[k] != b'-' {
            xi += 1;
        }
        if sy[k] != b'-' {
            yj += 1;
        }
        if sx[k] != b'-' && sy[k] != b'-' && (yj as usize) < ylen {
            invmap[yj as usize] = xi;
        }
    }
    invmap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_na_aa_chains() {
        let mol_vec = vec![MolType::Protein, MolType::RNA, MolType::Protein];
        let (na, aa) = count_na_aa_chains(&mol_vec);
        assert_eq!(na, 1);
        assert_eq!(aa, 2);
    }

    #[test]
    fn test_total_chain_lengths() {
        let chains = vec![
            ChainData {
                coords: vec![[0.0; 3]; 10],
                sequence: vec![b'A'; 10],
                sec_structure: vec![b'C'; 10],
                chain_id: "A".to_string(),
                mol_type: MolType::Protein,
            },
            ChainData {
                coords: vec![[0.0; 3]; 5],
                sequence: vec![b'a'; 5],
                sec_structure: vec![b'C'; 5],
                chain_id: "B".to_string(),
                mol_type: MolType::RNA,
            },
        ];
        let (len_aa, len_na) = total_chain_lengths(&chains);
        assert_eq!(len_aa, 10);
        assert_eq!(len_na, 5);
    }

    #[test]
    fn test_chain_data_len() {
        let cd = ChainData {
            coords: vec![[1.0, 2.0, 3.0]; 7],
            sequence: vec![b'G'; 7],
            sec_structure: vec![b'H'; 7],
            chain_id: "X".to_string(),
            mol_type: MolType::Protein,
        };
        assert_eq!(cd.len(), 7);
        assert!(!cd.is_empty());
    }
}
