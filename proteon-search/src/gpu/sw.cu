// Batched Smith-Waterman local alignment: score + endpoint only.
//
// One thread per (query, target) pair. Per-thread two-row affine-gap
// DP (Gotoh M/X/Y) with score-clamp at 0 for local alignment. Returns
// max M score and the (query_end, target_end) of the cell that
// achieved it. **No traceback** — CIGARs come from CPU on the
// post-ranking top-N hits, matching how upstream's GPU prefilter
// hands off to CPU for output.
//
// Memory: each thread gets a slice of `scratch` of size
// 6 * (target_len + 1) i32. Layout per thread:
//
//   M_prev | M_cur | X_prev | X_cur | Y_prev | Y_cur
//
// At t_len=1000 that's 24 KB per thread; with 256 threads/block,
// 6 MB per block in global memory. Fine for the single-query × N-
// targets prefilter use case. Long-target tuning is a 4.4 concern.
//
// Same algorithm as src/gapped.rs::smith_waterman, line-for-line —
// the parity test compares algorithms, not implementations.

extern "C" __global__ void smith_waterman_score_batch(
    // Inputs
    const unsigned char* __restrict__ query,
    int query_len,
    const unsigned char* __restrict__ targets,
    const int* __restrict__ target_offsets,
    const int* __restrict__ target_lens,
    const int* __restrict__ scores,
    int alphabet_size,
    int gap_open,
    int gap_extend,
    int n_pairs,
    int max_target_len,
    // Per-thread scratch: 6 i32 buffers of length (max_target_len + 1)
    int* __restrict__ scratch,
    // Outputs
    int* __restrict__ out_score,
    int* __restrict__ out_qend,
    int* __restrict__ out_tend
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    const int target_offset = target_offsets[tid];
    const int target_len = target_lens[tid];

    if (query_len == 0 || target_len == 0) {
        out_score[tid] = 0;
        out_qend[tid] = 0;
        out_tend[tid] = 0;
        return;
    }

    // Slice this thread's scratch buffer into six rows.
    const int row_stride = max_target_len + 1;
    int* base = scratch + (long long) tid * 6 * row_stride;
    int* M_prev = base + 0 * row_stride;
    int* M_cur  = base + 1 * row_stride;
    int* X_prev = base + 2 * row_stride;
    int* X_cur  = base + 3 * row_stride;
    int* Y_prev = base + 4 * row_stride;
    int* Y_cur  = base + 5 * row_stride;

    // Sentinel for unreachable gap states (matches CPU's i32::MIN/4).
    const int NEG_SENT = (int) 0xC0000000; // ~ -1.07e9, well above wrap risk

    // Row 0 boundary: M=0, X/Y=NEG_SENT (no gap can exist with empty prefix).
    for (int j = 0; j <= target_len; j++) {
        M_prev[j] = 0;
        X_prev[j] = NEG_SENT;
        Y_prev[j] = NEG_SENT;
    }

    int best_score = 0;
    int best_qend = 0;
    int best_tend = 0;

    for (int i = 1; i <= query_len; i++) {
        const unsigned char qb = query[i - 1];
        // Column 0 boundary on the current row.
        M_cur[0] = 0;
        X_cur[0] = NEG_SENT;
        Y_cur[0] = NEG_SENT;

        for (int j = 1; j <= target_len; j++) {
            const unsigned char tb = targets[target_offset + (j - 1)];
            const int sub = scores[(int) qb * alphabet_size + (int) tb];

            // X[i,j] = gap in query column (consume target j-1).
            //   max(M[i, j-1] + gap_open, X[i, j-1] + gap_extend)
            int x_from_m = M_cur[j - 1] + gap_open;
            int x_from_x = X_cur[j - 1] + gap_extend;
            int x_val = x_from_m >= x_from_x ? x_from_m : x_from_x;
            X_cur[j] = x_val;

            // Y[i,j] = gap in target column (consume query i-1).
            //   max(M[i-1, j] + gap_open, Y[i-1, j] + gap_extend)
            int y_from_m = M_prev[j] + gap_open;
            int y_from_y = Y_prev[j] + gap_extend;
            int y_val = y_from_m >= y_from_y ? y_from_m : y_from_y;
            Y_cur[j] = y_val;

            // M[i,j] = best of three diagonal predecessors + sub, clamped at 0.
            int m_diag = M_prev[j - 1];
            int m_from_x = X_prev[j - 1];
            int m_from_y = Y_prev[j - 1];
            int best_pred = m_diag;
            if (m_from_x > best_pred) best_pred = m_from_x;
            if (m_from_y > best_pred) best_pred = m_from_y;
            int m_val = best_pred + sub;
            if (m_val < 0) m_val = 0;
            M_cur[j] = m_val;

            if (m_val > best_score) {
                best_score = m_val;
                best_qend = i;
                best_tend = j;
            }
        }

        // Swap prev and cur by pointer, classical two-row pattern.
        int* tmp;
        tmp = M_prev; M_prev = M_cur; M_cur = tmp;
        tmp = X_prev; X_prev = X_cur; X_cur = tmp;
        tmp = Y_prev; Y_prev = Y_cur; Y_cur = tmp;
    }

    out_score[tid] = best_score;
    out_qend[tid] = best_qend;
    out_tend[tid] = best_tend;
}
