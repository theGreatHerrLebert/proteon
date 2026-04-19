//! Iterative refinement for multi-chain alignment.
//!
//! Ported from C++ USAlign `MMalign.h`: `MMalign_iter`, `copy_chain_assign_data`.

use super::chain_assign::enhanced_greedy_search;

/// State snapshot of chain assignment data, used for save/restore during iteration.
///
/// Corresponds to C++ `copy_chain_assign_data`.
#[derive(Debug, Clone)]
pub struct ChainAssignState {
    /// Sequence alignment strings for each chain pair.
    pub seqx_a_mat: Vec<Vec<String>>,
    /// Sequence alignment strings for each chain pair.
    pub seqy_a_mat: Vec<Vec<String>>,
    /// Forward assignment: `assign1[i]` = chain in complex 2 assigned to chain i.
    pub assign1: Vec<i32>,
    /// Reverse assignment: `assign2[j]` = chain in complex 1 assigned to chain j.
    pub assign2: Vec<i32>,
    /// TM-score matrix.
    pub tm_mat: Vec<Vec<f64>>,
    /// Concatenated alignment sequences.
    pub sequence: Vec<String>,
}

/// Copy (snapshot) chain assignment data from source to destination state.
///
/// Builds the concatenated `sequence` from the alignment matrices and assignments.
///
/// Corresponds to C++ `copy_chain_assign_data`.
pub fn copy_chain_assign_data(
    chain1_num: usize,
    chain2_num: usize,
    seqx_a_mat: &[Vec<String>],
    seqy_a_mat: &[Vec<String>],
    assign1: &[i32],
    assign2: &[i32],
    tm_mat: &[Vec<f64>],
) -> ChainAssignState {
    let mut sequence = vec![String::new(), String::new()];

    let mut new_seqx_a = vec![vec![String::new(); chain2_num]; chain1_num];
    let mut new_seqy_a = vec![vec![String::new(); chain2_num]; chain1_num];
    let mut new_tm = vec![vec![0.0f64; chain2_num]; chain1_num];

    for i in 0..chain1_num {
        for j in 0..chain2_num {
            if i < seqx_a_mat.len() && j < seqx_a_mat[i].len() {
                new_seqx_a[i][j] = seqx_a_mat[i][j].clone();
            }
            if i < seqy_a_mat.len() && j < seqy_a_mat[i].len() {
                new_seqy_a[i][j] = seqy_a_mat[i][j].clone();
            }
            if i < tm_mat.len() && j < tm_mat[i].len() {
                new_tm[i][j] = tm_mat[i][j];
            }
            if assign1[i] == j as i32 {
                if i < seqx_a_mat.len() && j < seqx_a_mat[i].len() {
                    sequence[0].push_str(&seqx_a_mat[i][j]);
                }
                if i < seqy_a_mat.len() && j < seqy_a_mat[i].len() {
                    sequence[1].push_str(&seqy_a_mat[i][j]);
                }
            }
        }
    }

    ChainAssignState {
        seqx_a_mat: new_seqx_a,
        seqy_a_mat: new_seqy_a,
        assign1: assign1.to_vec(),
        assign2: assign2.to_vec(),
        tm_mat: new_tm,
        sequence,
    }
}

/// Callback type for the search function used in iterative refinement.
///
/// Takes: current TM matrix, alignment matrices, assignment lists, sequence,
/// and returns the total score after search + SE refinement.
pub type SearchFn = dyn Fn(
    &mut Vec<Vec<f64>>,    // tm_mat (mutable for update)
    &mut Vec<Vec<String>>, // seqx_a_mat
    &mut Vec<Vec<String>>, // seqy_a_mat
    &mut Vec<i32>,         // assign1
    &mut Vec<i32>,         // assign2
    &mut Vec<String>,      // sequence
) -> f64;

/// Iterative chain-pairing refinement.
///
/// Alternates between:
/// 1. Running a search function (analogous to `MMalign_search`) to refine
///    per-chain alignments under the current assignment.
/// 2. Re-running `enhanced_greedy_search` on the updated TM matrix.
///
/// Continues until the total score stops improving or `max_iter` is reached.
///
/// Corresponds to C++ `MMalign_iter`.
///
/// `chain_map`: if non-empty, restricts which (i,j) pairs are valid.
///
/// Returns the updated state with the best assignment found.
pub fn mmalign_iter(
    max_total_score: f64,
    max_iter: usize,
    chain1_num: usize,
    chain2_num: usize,
    initial_state: &ChainAssignState,
    chain_map: &[(usize, usize)],
    search_fn: &SearchFn,
) -> (f64, ChainAssignState) {
    let mut best_score = max_total_score;
    let mut best_state = initial_state.clone();

    // Working copy
    let mut tmp_state = initial_state.clone();

    for _iter in 0..max_iter {
        // Run the search/refinement step
        let _total_score = search_fn(
            &mut tmp_state.tm_mat,
            &mut tmp_state.seqx_a_mat,
            &mut tmp_state.seqy_a_mat,
            &mut tmp_state.assign1,
            &mut tmp_state.assign2,
            &mut tmp_state.sequence,
        );

        // Apply chain map restrictions
        if !chain_map.is_empty() {
            for i in 0..chain1_num {
                for j in 0..chain2_num {
                    let allowed = chain_map.iter().any(|&(ci, cj)| ci == i && cj == j);
                    if !allowed {
                        tmp_state.tm_mat[i][j] = -1.0;
                    }
                }
            }
        }

        // Re-assign chains based on updated TM matrix
        let (new_assign1, new_assign2, total_score) =
            enhanced_greedy_search(&tmp_state.tm_mat, chain1_num, chain2_num);
        tmp_state.assign1 = new_assign1;
        tmp_state.assign2 = new_assign2;

        if total_score <= best_score {
            break;
        }

        best_score = total_score;

        // Save best state
        best_state = copy_chain_assign_data(
            chain1_num,
            chain2_num,
            &tmp_state.seqx_a_mat,
            &tmp_state.seqy_a_mat,
            &tmp_state.assign1,
            &tmp_state.assign2,
            &tmp_state.tm_mat,
        );
    }

    (best_score, best_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_chain_assign_data() {
        let seqx_a = vec![vec!["ACG".to_string(), "DEF".to_string()]];
        let seqy_a = vec![vec!["acg".to_string(), "def".to_string()]];
        let assign1 = vec![1i32]; // chain 0 -> chain 1
        let assign2 = vec![-1i32, 0]; // chain 1 <- chain 0
        let tm_mat = vec![vec![0.5, 0.8]];

        let state = copy_chain_assign_data(1, 2, &seqx_a, &seqy_a, &assign1, &assign2, &tm_mat);
        assert_eq!(state.assign1, vec![1]);
        assert_eq!(state.assign2, vec![-1, 0]);
        assert_eq!(state.sequence[0], "DEF");
        assert_eq!(state.sequence[1], "def");
        assert!((state.tm_mat[0][1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_mmalign_iter_no_improvement() {
        let state = ChainAssignState {
            seqx_a_mat: vec![vec!["A".to_string()]],
            seqy_a_mat: vec![vec!["A".to_string()]],
            assign1: vec![0],
            assign2: vec![0],
            tm_mat: vec![vec![0.5]],
            sequence: vec!["A".to_string(), "A".to_string()],
        };

        // Search function that returns a lower score
        let search = |tm: &mut Vec<Vec<f64>>,
                      _sx: &mut Vec<Vec<String>>,
                      _sy: &mut Vec<Vec<String>>,
                      _a1: &mut Vec<i32>,
                      _a2: &mut Vec<i32>,
                      _seq: &mut Vec<String>|
         -> f64 {
            tm[0][0] = 0.3; // worse
            0.3
        };

        let (score, result) = mmalign_iter(0.5, 5, 1, 1, &state, &[], &search);
        assert!((score - 0.5).abs() < 1e-10); // no improvement
        assert_eq!(result.assign1, vec![0]);
    }
}
