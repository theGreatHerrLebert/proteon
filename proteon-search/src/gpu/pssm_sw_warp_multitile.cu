// Phase 4.5b: Multitile warp-collaborative PSSM Smith-Waterman.
//
// Extension of pssm_sw_warp.cu to queries longer than TILE_SIZE (= 256).
// The query is partitioned along rows into tiles of TILE_SIZE rows
// each, processed sequentially by the same warp. Between tiles, the
// last row's (M, X, Y) state at every target column is captured to a
// global-memory scratch buffer; the next tile's top-row boundary reads
// from that buffer at each wavefront time step.
//
// Within each tile: identical wavefront to the singletile kernel —
// one warp per pair, 32 lanes × NUM_ITEMS=8 rows per lane, shuffle-
// based anti-diagonal propagation, 3-state Gotoh recurrence. See the
// singletile header for the full algorithm derivation.
//
// best_score / best_qend / best_tend persist in per-thread registers
// across tile iterations (same kernel launch, same register file).
// Final warp-level reduction after all tiles — same shape as singletile.
//
// Eligibility: query_len ≤ MAX_MULTITILE_TILES * TILE_SIZE. With
// MAX_MULTITILE_TILES=8 that's up to 2048-residue queries, covering
// essentially all single-domain and most full-protein queries.

#define GROUPSIZE 32
#define NUM_ITEMS 8
#define TILE_SIZE (GROUPSIZE * NUM_ITEMS)  // 256 rows per tile

#define FULL_MASK 0xffffffff

// Ping-pong border layout in global memory:
//   border_scratch is laid out as  [2][n_pairs][3][target_len] ints
//   interleaved per-pair as [buffer_idx][pair_id][channel][col].
//
// `channel`: 0 = M, 1 = X, 2 = Y.
//
// For tile t (0-indexed):
//   border_in  lives in buffer ((t - 1) & 1), i.e. written by tile t-1.
//   border_out lives in buffer (t & 1).
// Tile 0 ignores border_in; the final tile may ignore border_out (but
// we always write for uniformity — the extra traffic is negligible
// and the code stays branchless at the critical path).
extern "C" __global__ void pssm_sw_warp_multitile_batch(
    const int* __restrict__ pssm,              // query_len × alphabet_size, row-major (global memory)
    int query_len,
    int alphabet_size,
    const unsigned char* __restrict__ targets_flat,
    const int* __restrict__ target_offsets,    // per-pair into targets_flat
    const int* __restrict__ target_lens,       // per-pair
    int gap_open,
    int gap_extend,
    int n_pairs,
    int n_tiles,
    int max_target_len,                        // used for border_scratch stride
    int* __restrict__ border_scratch,          // 2 × n_pairs × 3 × max_target_len ints
    int* __restrict__ out_score,
    int* __restrict__ out_qend,
    int* __restrict__ out_tend
) {
    extern __shared__ int s_tile_pssm[];       // TILE_SIZE × alphabet_size ints per block

    const int block_size = blockDim.x;
    const int local_tid = threadIdx.x;
    const int warp_in_block = local_tid / 32;
    const int lane = local_tid & 31;
    const int warps_per_block = block_size / 32;
    const int pair_id = blockIdx.x * warps_per_block + warp_in_block;

    if (pair_id >= n_pairs) {
        // Still must participate in __syncthreads below, so fall through
        // the load loops but no output write. Easier to just return —
        // but that can deadlock the PSSM __syncthreads across warps.
        // Workaround: participate in syncs but gate the DP work by
        // an `is_active` pair flag.
    }
    const bool is_active_pair = (pair_id < n_pairs);

    const int target_len = is_active_pair ? target_lens[pair_id] : 0;
    const int target_offset = is_active_pair ? target_offsets[pair_id] : 0;
    const unsigned char* target = targets_flat + target_offset;

    const int NEG_SENT = (int) 0xC0000000;

    // Per-thread DP state — reset at each tile boundary.
    int scoresM[NUM_ITEMS];
    int scoresX[NUM_ITEMS];
    int scoresY[NUM_ITEMS];

    // Best triple persists across tile iterations.
    int best_score = 0;
    int best_qend = 0;
    int best_tend = 0;

    const int tile_channel_stride = max_target_len;                 // per-channel col count
    const int tile_pair_stride = 3 * tile_channel_stride;            // per-pair (M,X,Y stacked)
    const int tile_buffer_stride = n_pairs * tile_pair_stride;       // per buffer (ping or pong)

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base_row = tile * TILE_SIZE;
        // This tile covers rows [tile_base_row, tile_base_row + TILE_SIZE),
        // capped at query_len. The kernel handles the cap via the inner
        // `row < query_len` guard — same pattern as singletile.

        // Stage this tile's PSSM rows into shared memory. Rows outside
        // the tile aren't needed; load only [tile_base_row, tile_base_row + tile_row_count).
        const int tile_row_count = (query_len - tile_base_row) < TILE_SIZE
                                    ? (query_len - tile_base_row)
                                    : TILE_SIZE;
        const int pssm_tile_total = tile_row_count * alphabet_size;
        for (int idx = local_tid; idx < pssm_tile_total; idx += block_size) {
            // Source: pssm[(tile_base_row + local_row) * alphabet_size + col]
            // Dest:   s_tile_pssm[local_row * alphabet_size + col]
            const int local_row = idx / alphabet_size;
            const int col = idx % alphabet_size;
            s_tile_pssm[idx] =
                pssm[(tile_base_row + local_row) * alphabet_size + col];
        }
        __syncthreads();

        if (!is_active_pair || target_len == 0 || query_len == 0) {
            // Skip DP for idle warps in partial blocks. Still __syncthreads
            // with the rest at the next tile boundary.
            __syncthreads();
            continue;
        }

        // Reset per-tile DP state (scoresM/X/Y and lane-boundary).
        #pragma unroll
        for (int i = 0; i < NUM_ITEMS; i++) {
            scoresM[i] = 0;
            scoresX[i] = NEG_SENT;
            scoresY[i] = NEG_SENT;
        }
        int M_N = 0, X_N = NEG_SENT, Y_N = NEG_SENT;
        int M_P = 0, X_P = NEG_SENT, Y_P = NEG_SENT;

        // Set up border I/O pointers for this tile.
        // border_in: previous tile's last-row state. Absent for tile 0.
        // border_out: this tile's last-row state. Unused by the final
        // tile but we always write — uniform code path.
        const int in_buf_idx = (tile - 1) & 1;
        const int out_buf_idx = tile & 1;
        const int* border_in_M = (tile > 0)
            ? (border_scratch + in_buf_idx * tile_buffer_stride
                + pair_id * tile_pair_stride + 0 * tile_channel_stride)
            : nullptr;
        const int* border_in_X = (tile > 0)
            ? (border_scratch + in_buf_idx * tile_buffer_stride
                + pair_id * tile_pair_stride + 1 * tile_channel_stride)
            : nullptr;
        const int* border_in_Y = (tile > 0)
            ? (border_scratch + in_buf_idx * tile_buffer_stride
                + pair_id * tile_pair_stride + 2 * tile_channel_stride)
            : nullptr;
        int* border_out_M = border_scratch + out_buf_idx * tile_buffer_stride
            + pair_id * tile_pair_stride + 0 * tile_channel_stride;
        int* border_out_X = border_scratch + out_buf_idx * tile_buffer_stride
            + pair_id * tile_pair_stride + 1 * tile_channel_stride;
        int* border_out_Y = border_scratch + out_buf_idx * tile_buffer_stride
            + pair_id * tile_pair_stride + 2 * tile_channel_stride;

        const int base = lane * NUM_ITEMS;              // lane's first row within tile
        const int abs_base = tile_base_row + base;      // absolute query row for PSSM / qend

        const int n_steps = target_len + GROUPSIZE - 1;

        for (int t = 0; t < n_steps; t++) {
            const int j = t - lane;
            const bool active = (j >= 0) && (j < target_len);

            // For tile > 0 at lane 0, the "up" boundary is the previous
            // tile's bottom row at col j — read from border_in.
            if (tile > 0 && lane == 0 && active) {
                M_N = border_in_M[j];
                X_N = border_in_X[j];
                Y_N = border_in_Y[j];
            }

            unsigned char tb = 0;
            if (active) tb = target[j];

            int prev_M = M_P;
            int prev_X = X_P;
            int prev_Y = Y_P;

            #pragma unroll
            for (int i = 0; i < NUM_ITEMS; i++) {
                const int local_row = base + i;                // within-tile row index
                const int abs_row = abs_base + i;              // absolute query row

                const int saved_M = scoresM[i];
                const int saved_X = scoresX[i];
                const int saved_Y = scoresY[i];

                if (active && abs_row < query_len) {
                    const int up_M = (i == 0) ? M_N : scoresM[i - 1];
                    const int up_Y = (i == 0) ? Y_N : scoresY[i - 1];

                    const int sub = s_tile_pssm[local_row * alphabet_size + (int) tb];

                    const int x_from_m = saved_M + gap_open;
                    const int x_from_x = saved_X + gap_extend;
                    int x_val = x_from_m >= x_from_x ? x_from_m : x_from_x;
                    scoresX[i] = x_val;

                    const int y_from_m = up_M + gap_open;
                    const int y_from_y = up_Y + gap_extend;
                    int y_val = y_from_m >= y_from_y ? y_from_m : y_from_y;
                    scoresY[i] = y_val;

                    int best_pred = prev_M;
                    if (prev_X > best_pred) best_pred = prev_X;
                    if (prev_Y > best_pred) best_pred = prev_Y;
                    int m_val = best_pred + sub;
                    if (m_val < 0) m_val = 0;
                    scoresM[i] = m_val;

                    // CPU-convention tie-break on (score, smallest abs_row,
                    // smallest col).
                    const int new_qend = abs_row + 1;
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

                prev_M = saved_M;
                prev_X = saved_X;
                prev_Y = saved_Y;
            }

            // Capture lane 31's last-row state (the tile-bottom row) to
            // border_out for this target col. Only when lane 31's row
            // is still within the matrix.
            if (lane == 31 && active) {
                const int abs_last_row = abs_base + (NUM_ITEMS - 1);
                if (abs_last_row < query_len) {
                    border_out_M[j] = scoresM[NUM_ITEMS - 1];
                    border_out_X[j] = scoresX[NUM_ITEMS - 1];
                    border_out_Y[j] = scoresY[NUM_ITEMS - 1];
                } else {
                    // Partial final tile: lane 31's last row is beyond
                    // query_len. The next tile won't exist (this IS the
                    // final tile), so the border write is moot. Write
                    // a defined value anyway for determinism.
                    border_out_M[j] = 0;
                    border_out_X[j] = NEG_SENT;
                    border_out_Y[j] = NEG_SENT;
                }
            }

            // Lane boundary shuffles (same as singletile).
            const int new_M_shfl = __shfl_up_sync(FULL_MASK, scoresM[NUM_ITEMS - 1], 1);
            const int new_X_shfl = __shfl_up_sync(FULL_MASK, scoresX[NUM_ITEMS - 1], 1);
            const int new_Y_shfl = __shfl_up_sync(FULL_MASK, scoresY[NUM_ITEMS - 1], 1);

            M_P = M_N; X_P = X_N; Y_P = Y_N;
            if (lane == 0) {
                // Lane 0's boundary for the NEXT time step (col j+1) —
                // for tile 0 this is outside the matrix; for tile > 0
                // we refresh from border_in at the top of the next
                // iteration, so the value we set here is overwritten.
                // Leaving it at 0 / NEG_SENT keeps the singletile case
                // correct.
                M_N = 0;
                X_N = NEG_SENT;
                Y_N = NEG_SENT;
            } else {
                M_N = new_M_shfl;
                X_N = new_X_shfl;
                Y_N = new_Y_shfl;
            }
        }

        // Tile-boundary sync: make border_out visible + align all warps
        // in the block before reloading the next tile's PSSM.
        __syncthreads();
    }

    if (!is_active_pair) return;

    // Final warp-level reduction over (best_score, best_qend, best_tend).
    int warp_best = best_score;
    int warp_qend = best_qend;
    int warp_tend = best_tend;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_best = __shfl_down_sync(FULL_MASK, warp_best, offset);
        int other_qend = __shfl_down_sync(FULL_MASK, warp_qend, offset);
        int other_tend = __shfl_down_sync(FULL_MASK, warp_tend, offset);
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
