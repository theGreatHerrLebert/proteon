// GPU k-mer diagonal-voting prefilter kernels.
//
// Parity oracle: crate::prefilter::diagonal_prefilter. Produces the SAME
// PrefilterHit set (bit-exact — integer-only). Three kernels:
//
//   1. prefilter_vote          — one block per query k-mer; threads stride that
//      k-mer's posting list (entries[offsets[h]..offsets[h+1]]), compute
//      diagonal = pos - q_pos, and insert-or-increment a count keyed by
//      (seq_id, diagonal) into an open-addressing hash table.
//   2. prefilter_reduce_seqhash — one thread per VOTE slot; reduces each
//      occupied slot into a SECOND open-addressing hash table keyed by seq_id
//      only, via atomicMax over a packed value encoding the CPU tie-break (max
//      count, then SMALLEST diagonal). Replaces the old dense best[#targets]
//      array so every buffer is O(query postings), not O(#targets).
//   3. prefilter_compact       — one thread per SEQ-HASH slot; stream-compacts
//      each occupied (seq, packed) into a dense output list via an atomic
//      counter, so the host copies back O(hits), not O(#targets).
//
// Key packing (vote table): key = (seq << 32) | diag_biased, diag_biased =
// diag + DIAG_BIAS. The host sets DIAG_BIAS = qlen so diag_biased >= 1 ⇒ a real
// key is never 0, so EMPTY = 0 is an unambiguous empty-slot sentinel. The host
// guarantees diag_biased < 2^DIAG_BITS.
//
// Seq-hash table: key = seq + 1 (so seq 0 is never EMPTY=0); value = packed
// (count, diag_max - diag_biased). atomicMax is commutative ⇒ the hashed
// per-seq max equals the old dense-index max, bit-for-bit.

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

extern "C" __global__ void prefilter_reduce_seqhash(
    const unsigned long long* __restrict__ table_keys,    // vote table (zeroed)
    const unsigned int* __restrict__ table_counts,        // vote table
    unsigned int table_size,                              // vote table cap
    unsigned long long diag_max,                          // (1 << DIAG_BITS) - 1
    unsigned long long* __restrict__ best_keys,           // seq-hash keys (zeroed)
    unsigned long long* __restrict__ best_vals,           // seq-hash packed vals (zeroed)
    unsigned int best_mask,                               // best cap - 1 (pow2)
    unsigned int* __restrict__ error_flag                 // set to 1 on probe exhaustion
) {
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= table_size) return;

    unsigned long long key = table_keys[s];
    if (key == EMPTY) return;
    unsigned int count = table_counts[s];
    if (count == 0) return;

    unsigned int seq = (unsigned int)(key >> 32);
    unsigned long long diag_b = key & 0xffffffffULL;

    // Pack so atomicMax(u64) yields max count, then (on tie) the largest
    // (diag_max - diag_b) = the SMALLEST diag_b = smallest diagonal — exactly
    // the CPU tie-break. count >= 1 ⇒ packed >= (1<<DIAG_BITS) > 0, so an unset
    // best_vals slot (0) unambiguously means "no vote".
    unsigned long long packed = ((unsigned long long)count << DIAG_BITS) | (diag_max - diag_b);

    // Insert-or-max into the seq-keyed table. Key = seq + 1 so seq 0 never
    // collides with EMPTY=0. CAS claims the slot; both "we won" (old==EMPTY) and
    // "already ours" (old==skey) proceed to the atomicMax — converging every
    // thread that sees the same seq onto one slot (claudex).
    unsigned long long skey = (unsigned long long)seq + 1ULL;
    unsigned int slot = (unsigned int)hash_mix(skey) & best_mask;
    for (unsigned int probe = 0;; ++probe) {
        unsigned long long old = atomicCAS(&best_keys[slot], EMPTY, skey);
        if (old == EMPTY || old == skey) {
            atomicMax(&best_vals[slot], packed);
            break;
        }
        if (probe >= best_mask) {
            atomicExch(error_flag, 1u);
            break;
        }
        slot = (slot + 1) & best_mask;
    }
}

extern "C" __global__ void prefilter_compact(
    const unsigned long long* __restrict__ best_keys,     // seq-hash keys
    const unsigned long long* __restrict__ best_vals,     // seq-hash packed vals
    unsigned int best_size,                               // seq-hash cap
    unsigned int* __restrict__ out_seq,                   // out_cap
    unsigned long long* __restrict__ out_packed,          // out_cap
    unsigned int out_cap,
    unsigned int* __restrict__ out_count,                 // global counter (zeroed)
    unsigned int* __restrict__ overflow_flag              // set to 1 if out_cap exceeded
) {
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= best_size) return;

    unsigned long long skey = best_keys[s];
    if (skey == EMPTY) return; // key off the KEY, not the value (claudex)

    unsigned int idx = atomicAdd(out_count, 1u);
    if (idx >= out_cap) {
        atomicExch(overflow_flag, 1u);
        return;
    }
    out_seq[idx] = (unsigned int)(skey - 1ULL);
    out_packed[idx] = best_vals[s];
}
