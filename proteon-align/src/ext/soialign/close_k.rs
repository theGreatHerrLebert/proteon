//! K-nearest neighbor precomputation for SOI alignment.
//!
//! Ported from C++ USAlign `SOIalign.h` `getCloseK`.
//! For each residue, finds the K closest CA atoms and stores their coordinates
//! in a flat array. This provides local structural context for the initial
//! SOI assignment step.

use crate::core::types::Coord3D;

/// Euclidean distance between two 3D points (not squared).
#[inline]
fn dist(a: &Coord3D, b: &Coord3D) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
}

/// Precompute K-nearest neighbors for each residue.
///
/// For each residue `i` in `xa` (length `xlen`), finds the `close_k_opt` nearest
/// residues (including self) sorted by distance. Their coordinates are stored
/// in the returned flat vector at positions `[i * close_k_opt .. (i+1) * close_k_opt)`.
///
/// If `close_k_opt > xlen`, wraps around (the C++ code uses `k % xlen`).
///
/// Corresponds to C++ `getCloseK`.
pub fn get_close_k(xa: &[Coord3D], close_k_opt: usize) -> Vec<Coord3D> {
    let xlen = xa.len();
    let mut xk = vec![[0.0; 3]; xlen * close_k_opt];

    // Build pairwise distance matrix (1-indexed in C++, we use 0-indexed)
    let mut score = vec![vec![0.0_f64; xlen]; xlen];
    for i in 0..xlen {
        score[i][i] = 0.0;
        for j in (i + 1)..xlen {
            let d = dist(&xa[i], &xa[j]);
            score[i][j] = d;
            score[j][i] = d;
        }
    }

    // For each residue, sort neighbors by distance and pick K closest
    let mut close_idx_vec: Vec<(f64, usize)> = vec![(0.0, 0); xlen];
    for i in 0..xlen {
        for j in 0..xlen {
            close_idx_vec[j] = (score[i][j], j);
        }
        close_idx_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for k in 0..close_k_opt {
            let j = close_idx_vec[k % xlen].1;
            xk[i * close_k_opt + k] = xa[j];
        }
    }

    xk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_close_k_self_is_closest() {
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let k = 2;
        let xk = get_close_k(&coords, k);
        // For residue 0, closest is self (0,0,0), then (1,0,0)
        assert_eq!(xk[0], [0.0, 0.0, 0.0]);
        assert_eq!(xk[1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_get_close_k_output_length() {
        let n = 5;
        let k = 3;
        let coords: Vec<Coord3D> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
        let xk = get_close_k(&coords, k);
        assert_eq!(xk.len(), n * k);
    }

    #[test]
    fn test_get_close_k_wraps_when_k_exceeds_n() {
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let k = 5;
        let xk = get_close_k(&coords, k);
        assert_eq!(xk.len(), 2 * k);
        // Should wrap around without panic
    }

    #[test]
    fn test_get_close_k_ordering() {
        let coords: Vec<Coord3D> = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let k = 3;
        let xk = get_close_k(&coords, k);
        // For residue 0: self(0,0,0) dist=0, (1,0,0) dist=1, (10,0,0) dist=10
        assert_eq!(xk[0], [0.0, 0.0, 0.0]);
        assert_eq!(xk[1], [1.0, 0.0, 0.0]);
        assert_eq!(xk[2], [10.0, 0.0, 0.0]);
    }
}
