// GPU k-mer diagonal-voting prefilter kernels.
//
// Parity oracle: crate::prefilter::diagonal_prefilter. Produces the SAME
// PrefilterHit set (bit-exact — integer-only). Two kernels:
//
//   1. prefilter_vote  — one block per query k-mer; threads stride that
//      k-mer's posting list (entries[offsets[h]..offsets[h+1]]), compute
//      diagonal = pos - q_pos, and insert-or-increment a count keyed by
//      (seq_id, diagonal) into an open-addressing hash table.
//   2. prefilter_reduce — one thread per hash slot; reduces each occupied
//      slot into best[seq_id] via atomicMax over a packed key that encodes
//      the CPU tie-break (max count, then SMALLEST diagonal).
//
// Key packing: key = (seq << 32) | diag_biased, diag_biased = diag + DIAG_BIAS.
// The host sets DIAG_BIAS = qlen so diag_biased >= 1 ⇒ a real key is never 0,
// so EMPTY = 0 is an unambiguous empty-slot sentinel. The host guarantees
// diag_biased < 2^DIAG_BITS.

#define EMPTY 0ULL
#define DIAG_BITS 20

// splitmix64 finalizer — spread keys before masking so linear probing on
// structured (seq, diag) keys doesn't cluster.
__device__ __forceinline__ unsigned long long hash_mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

extern "C" __global__ void prefilter_vote(
    const unsigned long long* __restrict__ offsets,        // table_size + 1
    const unsigned int* __restrict__ entries_seq_id,       // n_entries
    const unsigned int* __restrict__ entries_pos,          // n_entries (u16 widened)
    const int* __restrict__ kmer_qpos,                     // n_kmers
    const unsigned long long* __restrict__ kmer_hash,      // n_kmers
    int n_kmers,
    int diag_bias,
    unsigned long long* __restrict__ table_keys,           // table_size (zeroed)
    unsigned int* __restrict__ table_counts,               // table_size (zeroed)
    unsigned int table_mask,                               // table_size - 1 (pow2)
    unsigned int* __restrict__ error_flag                  // set to 1 on probe exhaustion
) {
    int k = blockIdx.x;
    if (k >= n_kmers) return;

    unsigned long long h = kmer_hash[k];
    int q_pos = kmer_qpos[k];
    unsigned long long start = offsets[h];
    unsigned long long end = offsets[h + 1];

    for (unsigned long long e = start + threadIdx.x; e < end; e += blockDim.x) {
        unsigned int seq = entries_seq_id[e];
        int diag = (int)entries_pos[e] - q_pos;
        unsigned int diag_b = (unsigned int)(diag + diag_bias);
        unsigned long long key = ((unsigned long long)seq << 32) | (unsigned long long)diag_b;

        // Insert-or-increment. The CAS claims an empty slot; if it returns
        // EMPTY (we won) or our own key (someone else claimed this slot for
        // the SAME key), increment that slot's count. Otherwise the slot
        // holds a different key — linear-probe to the next.
        unsigned int slot = (unsigned int)hash_mix(key) & table_mask;
        for (unsigned int probe = 0;; ++probe) {
            unsigned long long old = atomicCAS(&table_keys[slot], EMPTY, key);
            if (old == EMPTY || old == key) {
                atomicAdd(&table_counts[slot], 1u);
                break;
            }
            if (probe >= table_mask) {
                // Probed every slot without finding room — capacity bug; the
                // host sizes for >= 2x distinct keys so this should be
                // unreachable. Flag it rather than miscount silently.
                atomicExch(error_flag, 1u);
                break;
            }
            slot = (slot + 1) & table_mask;
        }
    }
}

extern "C" __global__ void prefilter_reduce(
    const unsigned long long* __restrict__ table_keys,
    const unsigned int* __restrict__ table_counts,
    unsigned int table_size,
    unsigned long long diag_max,        // (1 << DIAG_BITS) - 1
    unsigned long long* __restrict__ best,   // best_len (zeroed)
    unsigned int best_len
) {
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= table_size) return;

    unsigned long long key = table_keys[s];
    if (key == EMPTY) return;
    unsigned int count = table_counts[s];
    if (count == 0) return;

    unsigned int seq = (unsigned int)(key >> 32);
    if (seq >= best_len) return; // guard malformed seq_id (claudex)
    unsigned long long diag_b = key & 0xffffffffULL;

    // Pack so atomicMax(u64) yields max count, then (on tie) the largest
    // (diag_max - diag_b) = the SMALLEST diag_b = smallest diagonal — exactly
    // the CPU tie-break. count >= 1 ⇒ packed >= (1<<DIAG_BITS) > 0, so best==0
    // unambiguously means "no vote".
    unsigned long long packed = ((unsigned long long)count << DIAG_BITS) | (diag_max - diag_b);
    atomicMax(&best[seq], packed);
}
