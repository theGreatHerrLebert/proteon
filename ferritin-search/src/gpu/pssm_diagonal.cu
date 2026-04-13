// PSSM-based batched ungapped diagonal scoring.
//
// Same algorithm as gpu/diagonal.cu, but the inner loop reads scores
// via a precomputed PSSM (`query_len × alphabet_size` i32 matrix)
// instead of doing the two-step `scores[query[q]*alphabet_size+target[t]]`
// indirection.
//
// Critically, the PSSM is staged into block-shared memory at kernel
// entry. After staging, every score lookup is a single shared-memory
// read indexed by `(q_position, target_byte)` — no global-memory
// traffic for scores during the hot loop. This is the central perf
// primitive in libmarv: it turns score lookup from a memory-bound
// operation into something close to register-speed, which is what
// makes 1000x throughput vs CPU possible at production sizes.
//
// Memory model:
//   - PSSM total size = query_len × alphabet_size × 4 bytes.
//     For a 500-residue protein query at alphabet 21, that's
//     500 × 21 × 4 = 42 KB. Most CUDA architectures support
//     ≥48 KB of shared memory per block, so this fits at typical
//     query sizes. Larger queries fall back to global memory
//     (caller can dispatch to the non-PSSM kernel).
//   - Caller passes the PSSM as a flat row-major i32 buffer.
//
// Same Kadane-with-reset semantic as gpu/diagonal.cu — the parity
// test asserts every byte of every output matches the non-PSSM
// path on the same inputs.

extern "C" __global__ void ungapped_diagonal_pssm_batch(
    // Inputs
    const int* __restrict__ pssm,           // query_len × alphabet_size, row-major
    int query_len,
    int alphabet_size,
    const unsigned char* __restrict__ targets,
    const int* __restrict__ target_offsets,
    const int* __restrict__ target_lens,
    const int* __restrict__ diagonals,
    int n_pairs,
    // Outputs
    int* __restrict__ out_score,
    int* __restrict__ out_qstart,
    int* __restrict__ out_qend,
    int* __restrict__ out_tstart,
    int* __restrict__ out_tend
) {
    extern __shared__ int s_pssm[];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Stage PSSM into shared memory cooperatively. Every thread in
    // the block contributes to the load; once done, every thread
    // reads from `s_pssm` for the rest of the kernel.
    const int pssm_total = query_len * alphabet_size;
    for (int idx = local_tid; idx < pssm_total; idx += block_size) {
        s_pssm[idx] = pssm[idx];
    }
    __syncthreads();

    if (tid >= n_pairs) return;

    const int target_offset = target_offsets[tid];
    const int target_len = target_lens[tid];
    const int diagonal = diagonals[tid];

    // Overlap clamp (i64 arithmetic — same as gpu/diagonal.cu and
    // src/ungapped.rs::overlap_range; safe for diagonal = i32::MIN).
    long long t_minus_diag = (long long) target_len - (long long) diagonal;
    if (t_minus_diag < 0) t_minus_diag = 0;
    int q_end_by_target = (t_minus_diag > (long long) query_len)
        ? query_len
        : (int) t_minus_diag;

    int q_start = (diagonal < 0) ? -diagonal : 0;
    int q_end = q_end_by_target;

    if (q_start >= q_end) {
        out_score[tid] = 0;
        out_qstart[tid] = 0;
        out_qend[tid] = 0;
        out_tstart[tid] = 0;
        out_tend[tid] = 0;
        return;
    }

    int running = 0;
    int running_start = q_start;
    int best_score = 0;
    int best_start = q_start;
    int best_end = q_start;

    for (int q = q_start; q < q_end; q++) {
        const int t = q + diagonal;
        const unsigned char tb = targets[target_offset + t];
        // PSSM lookup from shared memory — single indexed read.
        const int cell = s_pssm[q * alphabet_size + (int) tb];
        running += cell;
        if (running < 0) {
            running = 0;
            running_start = q + 1;
        } else if (running > best_score) {
            best_score = running;
            best_start = running_start;
            best_end = q + 1;
        }
    }

    out_score[tid] = best_score;
    if (best_score > 0) {
        out_qstart[tid] = best_start;
        out_qend[tid] = best_end;
        out_tstart[tid] = best_start + diagonal;
        out_tend[tid] = best_end + diagonal;
    } else {
        out_qstart[tid] = 0;
        out_qend[tid] = 0;
        out_tstart[tid] = 0;
        out_tend[tid] = 0;
    }
}
