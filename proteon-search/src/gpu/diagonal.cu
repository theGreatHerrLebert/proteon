// Batched ungapped diagonal scoring.
//
// One thread per (target, diagonal) pair. Walks the diagonal serially
// using Kadane-style max running sum with reset-on-negative — the
// exact same algorithm as the CPU `ungapped_alignment` in
// src/ungapped.rs, ported line-for-line so the parity test compares
// algorithms not implementations.
//
// Memory layout for variable-length targets: targets are concatenated
// into one flat byte array; per-pair `target_offsets[i]` and
// `target_lens[i]` describe where pair i's target lives. The query
// is shared across all pairs (one query × N targets per launch),
// matching upstream libmarv's batching pattern.

extern "C" __global__ void ungapped_diagonal_batch(
    // Inputs (read-only)
    const unsigned char* __restrict__ query,
    int query_len,
    const unsigned char* __restrict__ targets,
    const int* __restrict__ target_offsets,
    const int* __restrict__ target_lens,
    const int* __restrict__ diagonals,
    const int* __restrict__ scores,
    int alphabet_size,
    int n_pairs,
    // Outputs (one entry per pair)
    int* __restrict__ out_score,
    int* __restrict__ out_qstart,
    int* __restrict__ out_qend,
    int* __restrict__ out_tstart,
    int* __restrict__ out_tend
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    const int target_offset = target_offsets[tid];
    const int target_len = target_lens[tid];
    const int diagonal = diagonals[tid];

    // Overlap range: q in [max(0, -diag), min(query_len, target_len - diag)).
    // Guards i32 overflow on extreme diagonal values by promoting to
    // 64-bit before subtraction (matches src/ungapped.rs::overlap_range).
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
        const unsigned char qb = query[q];
        const unsigned char tb = targets[target_offset + t];
        const int cell = scores[(int) qb * alphabet_size + (int) tb];
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
