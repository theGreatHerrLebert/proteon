// Phase 4.5a: Warp-collaborative PSSM Smith-Waterman (singletile variant).
//
// One warp (32 threads) per alignment pair. Each thread owns NUM_ITEMS
// consecutive query rows in per-thread register arrays. Threads cooperate
// via anti-diagonal wavefront: at time step t, lane k processes target
// column (t - k). Boundary values between adjacent lanes flow through
// __shfl_up_sync, eliminating shared-memory traffic in the inner loop.
//
// Libmarv shape (pssmkernels_smithwaterman.cuh, *_singletile), adapted
// to the **3-state** Gotoh recurrence used by the rest of this crate —
//
//   X[i,j] = max(M[i,   j-1] + gap_open,  X[i,   j-1] + gap_extend)
//   Y[i,j] = max(M[i-1, j]   + gap_open,  Y[i-1, j]   + gap_extend)
//   M[i,j] = max(M[i-1,j-1], X[i-1,j-1], Y[i-1,j-1]) + sub(q[i], t[j]);
//            clamp M at 0 (local alignment).
//
// — matching src/gapped.rs::smith_waterman and src/gpu/pssm_sw.cu bit
// for bit. That's the parity target: this kernel must produce exactly
// the same (score, query_end, target_end) tuple as both, for any
// (query, target) with query_len <= 256.
//
// Eligibility: query_len <= GROUPSIZE * NUM_ITEMS = 32 * 8 = 256.
// Longer queries need the multitile variant (phase 4.5b) or fall back
// to the thread-per-pair kernel in pssm_sw.cu.

#define GROUPSIZE 32
#define NUM_ITEMS 8
#define MAX_QUERY (GROUPSIZE * NUM_ITEMS)

#define FULL_MASK 0xffffffff

extern "C" __global__ void pssm_sw_warp_batch(
    const int* __restrict__ pssm,              // query_len × alphabet_size, row-major
    int query_len,
    int alphabet_size,
    const unsigned char* __restrict__ targets_flat,
    const int* __restrict__ target_offsets,    // per-pair into targets_flat
    const int* __restrict__ target_lens,       // per-pair
    int gap_open,
    int gap_extend,
    int n_pairs,
    int* __restrict__ out_score,
    int* __restrict__ out_qend,
    int* __restrict__ out_tend
) {
    extern __shared__ int s_pssm[];

    const int block_size = blockDim.x;
    const int local_tid = threadIdx.x;
    const int warp_in_block = local_tid / 32;
    const int lane = local_tid & 31;
    const int warps_per_block = block_size / 32;
    const int pair_id = blockIdx.x * warps_per_block + warp_in_block;

    // Stage PSSM into shared memory cooperatively across the whole block.
    // Layout: row-major, row = query position, col = alphabet index.
    // All warps in the block share the same PSSM (one query per kernel launch).
    const int pssm_total = query_len * alphabet_size;
    for (int idx = local_tid; idx < pssm_total; idx += block_size) {
        s_pssm[idx] = pssm[idx];
    }
    __syncthreads();

    if (pair_id >= n_pairs) return;

    const int target_len = target_lens[pair_id];

    // Early out for empty inputs. Lane 0 writes the zero result; other
    // lanes just return (no shuffles have been issued yet).
    if (query_len == 0 || target_len == 0) {
        if (lane == 0) {
            out_score[pair_id] = 0;
            out_qend[pair_id] = 0;
            out_tend[pair_id] = 0;
        }
        return;
    }

    const int target_offset = target_offsets[pair_id];
    const unsigned char* target = targets_flat + target_offset;

    const int base = lane * NUM_ITEMS;  // first query row owned by this lane

    // Sentinel mirrors gapped.rs: big-negative with headroom so it stays
    // negative under `+ gap_extend` accumulation without wrapping.
    const int NEG_SENT = (int) 0xC0000000;

    // Per-thread DP state. scoresM/X/Y[i] holds (M, X, Y)[base + i, current_col].
    // At loop entry for col j these hold the col-(j-1) values; the inner
    // loop overwrites them in place with col-j values.
    int scoresM[NUM_ITEMS];
    int scoresX[NUM_ITEMS];
    int scoresY[NUM_ITEMS];
    #pragma unroll
    for (int i = 0; i < NUM_ITEMS; i++) {
        scoresM[i] = 0;
        scoresX[i] = NEG_SENT;
        scoresY[i] = NEG_SENT;
    }

    // Boundary state from the lane above (k-1) covering the row base-1.
    // Two generations:
    //   *_N  : most-recent shfl result  = values at col = this lane's current col
    //   *_P  : previous shfl result     = values at col = current - 1
    // They cycle each time step: *_P := *_N ; *_N := shfl_up(neighbor's last row).
    //
    // For lane 0, "row base-1" = row -1 which is outside the matrix: M
    // there is 0, X/Y are unreachable so NEG_SENT.
    int M_N = 0, X_N = NEG_SENT, Y_N = NEG_SENT;
    int M_P = 0, X_P = NEG_SENT, Y_P = NEG_SENT;

    int best_score = 0;
    int best_qend = 0;   // 1-indexed, CPU convention
    int best_tend = 0;

    // Total wavefront length = target_len + GROUPSIZE - 1. Lane k is
    // active during steps [k, k + target_len), idle outside that window
    // but still participates in every shfl_up so the wavefront remains
    // consistent across the warp.
    const int n_steps = target_len + GROUPSIZE - 1;

    for (int t = 0; t < n_steps; t++) {
        const int j = t - lane;                     // 0-indexed target col
        const bool active = (j >= 0) && (j < target_len);

        // Active lanes fetch the target byte for column j. Inactive
        // lanes still run the inner loop (to keep shfl lanes aligned)
        // but with an inert byte.
        unsigned char tb = 0;
        if (active) tb = target[j];

        // Diag state for i=0 comes from the lane above at col j-1.
        int prev_M = M_P;
        int prev_X = X_P;
        int prev_Y = Y_P;

        #pragma unroll
        for (int i = 0; i < NUM_ITEMS; i++) {
            const int row = base + i;

            // Save OLD col-(j-1) values before overwrite — these become
            // the diag predecessors for the next i inside this inner loop.
            const int saved_M = scoresM[i];
            const int saved_X = scoresX[i];
            const int saved_Y = scoresY[i];

            if (active && row < query_len) {
                // "Up" neighbors (row-1 at col j):
                //   i==0 → from lane above, at THIS col → (M_N, Y_N)
                //   i>0  → from scoresM[i-1] / scoresY[i-1] which were
                //          just overwritten this iter with col-j values.
                const int up_M = (i == 0) ? M_N : scoresM[i - 1];
                const int up_Y = (i == 0) ? Y_N : scoresY[i - 1];

                const int sub = s_pssm[row * alphabet_size + (int) tb];

                // X[row, j] = max(M[row, j-1] + go, X[row, j-1] + ge)
                const int x_from_m = saved_M + gap_open;
                const int x_from_x = saved_X + gap_extend;
                int x_val = x_from_m >= x_from_x ? x_from_m : x_from_x;
                scoresX[i] = x_val;

                // Y[row, j] = max(M[row-1, j] + go, Y[row-1, j] + ge)
                const int y_from_m = up_M + gap_open;
                const int y_from_y = up_Y + gap_extend;
                int y_val = y_from_m >= y_from_y ? y_from_m : y_from_y;
                scoresY[i] = y_val;

                // M[row, j] = max(M, X, Y at row-1, col-1) + sub, clamp ≥0.
                int best_pred = prev_M;
                if (prev_X > best_pred) best_pred = prev_X;
                if (prev_Y > best_pred) best_pred = prev_Y;
                int m_val = best_pred + sub;
                if (m_val < 0) m_val = 0;
                scoresM[i] = m_val;

                // CPU's `smith_waterman` scans row-major (outer i, inner
                // j) with strict `>`, so ties resolve to the smallest
                // (row, col) pair. The wavefront visits cells in a
                // DIFFERENT order (col-major within each lane's rows),
                // so strict `>` here would pick a later (row, col) on
                // ties. Explicitly tie-break to preserve CPU semantics.
                const int new_qend = row + 1;
                const int new_tend = j + 1;
                bool take = m_val > best_score;
                if (!take && m_val == best_score && best_score > 0) {
                    if (new_qend < best_qend) {
                        take = true;
                    } else if (new_qend == best_qend && new_tend < best_tend) {
                        take = true;
                    }
                }
                if (take) {
                    best_score = m_val;
                    best_qend = new_qend;
                    best_tend = new_tend;
                }
            }
            // Inactive / out-of-matrix rows: leave scoresM/X/Y[i]
            // unchanged. They hold the previous col's values which
            // will continue to shift diagonally in subsequent iters.

            // Shift diag for next i: (prev_M, prev_X, prev_Y) becomes
            // the (M, X, Y)[row, j-1] we just saved.
            prev_M = saved_M;
            prev_X = saved_X;
            prev_Y = saved_Y;
        }

        // End-of-step boundary exchange: send this lane's last-row
        // (M, X, Y) at col j to lane+1. We always shfl, even for
        // inactive lanes, so the collective shuffle stays in sync.
        const int new_M = __shfl_up_sync(FULL_MASK, scoresM[NUM_ITEMS - 1], 1);
        const int new_X = __shfl_up_sync(FULL_MASK, scoresX[NUM_ITEMS - 1], 1);
        const int new_Y = __shfl_up_sync(FULL_MASK, scoresY[NUM_ITEMS - 1], 1);

        // Cycle generations: previous *_N becomes *_P; fresh shfl result
        // becomes *_N. Lane 0 receives its own value from shfl_up; force
        // to the outside-matrix sentinels.
        M_P = M_N; X_P = X_N; Y_P = Y_N;
        if (lane == 0) {
            M_N = 0;
            X_N = NEG_SENT;
            Y_N = NEG_SENT;
        } else {
            M_N = new_M;
            X_N = new_X;
            Y_N = new_Y;
        }
    }

    // Reduce (best_score, best_qend, best_tend) across the warp. Keep
    // the triple together so we return a coherent endpoint, not a
    // score from one lane paired with an endpoint from another.
    int warp_best = best_score;
    int warp_qend = best_qend;
    int warp_tend = best_tend;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_best = __shfl_down_sync(FULL_MASK, warp_best, offset);
        int other_qend = __shfl_down_sync(FULL_MASK, warp_qend, offset);
        int other_tend = __shfl_down_sync(FULL_MASK, warp_tend, offset);
        // Tie-break: prefer the lane that reached the higher score;
        // if tied, prefer the lane that matches CPU's "first-seen
        // maximum" ordering. CPU scans rows in increasing i and for
        // each i increasing j, so ties break by smaller (i, j). Lane 0
        // covers rows [0, NUM_ITEMS), so it already has the earliest
        // rows — when two lanes tie on score, prefer the one with
        // smaller qend (equivalent to smaller query_end coord).
        bool take = false;
        if (other_best > warp_best) {
            take = true;
        } else if (other_best == warp_best) {
            if (other_qend < warp_qend
                || (other_qend == warp_qend && other_tend < warp_tend)) {
                take = true;
            }
        }
        if (take) {
            warp_best = other_best;
            warp_qend = other_qend;
            warp_tend = other_tend;
        }
    }

    if (lane == 0) {
        out_score[pair_id] = warp_best;
        out_qend[pair_id] = warp_qend;
        out_tend[pair_id] = warp_tend;
    }
}
