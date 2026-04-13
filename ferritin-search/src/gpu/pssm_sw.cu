// PSSM Smith-Waterman on PaddedDb (coalesced + shared memory).
//
// Combines the two libmarv perf primitives:
//   - PSSM staged into block-shared memory (from 4.4a pssm_diagonal)
//   - Targets in transposed-coalesced PaddedDb layout (from 4.4b)
//
// Thread assignment: one thread per (bucket, slot). Thread index
// `tid = bucket_id * bucket_size + slot`. Within a warp whose threads
// cover the same bucket, the target-byte read at DP column j issues a
// single coalesced global load across all `bucket_size` slots at the
// same position — which is what makes this kernel meaningfully faster
// than gpu/sw.cu on large batches.
//
// Score lookup is `s_pssm[row * alphabet_size + target_byte]` from
// shared memory; no global-memory traffic for scores during the
// inner loop.
//
// Same Gotoh affine-gap recurrences as src/gapped.rs::smith_waterman
// and gpu/sw.cu — the parity test asserts bit-equal output against
// both the CPU and non-PSSM/non-padded GPU paths.

extern "C" __global__ void pssm_sw_padded_batch(
    // Inputs
    const int* __restrict__ pssm,            // query_len × alphabet_size, row-major
    int query_len,
    int alphabet_size,
    const unsigned char* __restrict__ padded_data,
    const int* __restrict__ bucket_starts,       // byte offsets per bucket
    const int* __restrict__ bucket_padded_lens,  // per bucket
    const int* __restrict__ slot_real_lens,      // n_buckets * bucket_size, flat
    int bucket_size,
    int n_buckets,
    int gap_open,
    int gap_extend,
    int max_padded_len,
    int n_threads_total,
    // Per-thread scratch: 6 i32 buffers of length (max_padded_len + 1)
    int* __restrict__ scratch,
    // Outputs in (bucket, slot) order — n_buckets * bucket_size entries
    int* __restrict__ out_score,
    int* __restrict__ out_qend,
    int* __restrict__ out_tend
) {
    extern __shared__ int s_pssm[];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Stage PSSM into shared memory cooperatively.
    const int pssm_total = query_len * alphabet_size;
    for (int idx = local_tid; idx < pssm_total; idx += block_size) {
        s_pssm[idx] = pssm[idx];
    }
    __syncthreads();

    if (tid >= n_threads_total) return;

    const int bucket_id = tid / bucket_size;
    const int slot = tid % bucket_size;
    if (bucket_id >= n_buckets) return;

    const int real_len = slot_real_lens[tid];
    // Inactive slot (padding-only) — zero outputs and done.
    if (real_len == 0 || query_len == 0) {
        out_score[tid] = 0;
        out_qend[tid] = 0;
        out_tend[tid] = 0;
        return;
    }

    const int bucket_start = bucket_starts[bucket_id];
    const int padded_len = bucket_padded_lens[bucket_id];

    // Per-thread scratch slice (six rows of length max_padded_len+1).
    const int row_stride = max_padded_len + 1;
    int* base = scratch + (long long) tid * 6 * row_stride;
    int* M_prev = base + 0 * row_stride;
    int* M_cur  = base + 1 * row_stride;
    int* X_prev = base + 2 * row_stride;
    int* X_cur  = base + 3 * row_stride;
    int* Y_prev = base + 4 * row_stride;
    int* Y_cur  = base + 5 * row_stride;

    const int NEG_SENT = (int) 0xC0000000;

    // We only need entries up through `real_len` in the inner loops;
    // initialize that range. (max_padded_len >= padded_len >= real_len.)
    for (int j = 0; j <= real_len; j++) {
        M_prev[j] = 0;
        X_prev[j] = NEG_SENT;
        Y_prev[j] = NEG_SENT;
    }

    int best_score = 0;
    int best_qend = 0;
    int best_tend = 0;

    for (int i = 1; i <= query_len; i++) {
        // PSSM row for query position i-1; lookup is
        // `s_pssm[(i-1) * alphabet_size + target_byte]` per column.
        const int* pssm_row = &s_pssm[(i - 1) * alphabet_size];

        M_cur[0] = 0;
        X_cur[0] = NEG_SENT;
        Y_cur[0] = NEG_SENT;

        for (int j = 1; j <= real_len; j++) {
            // Coalesced read: thread with `slot` reads position `j-1`
            // out of its bucket, and neighbor threads in the same warp
            // (same bucket, different slots) read the adjacent bytes.
            const unsigned char tb =
                padded_data[bucket_start + (j - 1) * bucket_size + slot];
            const int sub = pssm_row[(int) tb];

            // X[i,j]: gap in query column.
            int x_from_m = M_cur[j - 1] + gap_open;
            int x_from_x = X_cur[j - 1] + gap_extend;
            int x_val = x_from_m >= x_from_x ? x_from_m : x_from_x;
            X_cur[j] = x_val;

            // Y[i,j]: gap in target column.
            int y_from_m = M_prev[j] + gap_open;
            int y_from_y = Y_prev[j] + gap_extend;
            int y_val = y_from_m >= y_from_y ? y_from_m : y_from_y;
            Y_cur[j] = y_val;

            // M[i,j]: best diagonal predecessor + sub, clamped at 0.
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

        // Swap prev/cur by pointer.
        int* tmp;
        tmp = M_prev; M_prev = M_cur; M_cur = tmp;
        tmp = X_prev; X_prev = X_cur; X_cur = tmp;
        tmp = Y_prev; Y_prev = Y_cur; Y_cur = tmp;
    }

    out_score[tid] = best_score;
    out_qend[tid] = best_qend;
    out_tend[tid] = best_tend;
}
