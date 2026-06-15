# GAPPED_SW_MEMORY_PLAN — cut gapped Smith-Waterman memory ~15× (exact, no result change)

Status: DRAFT (pre-implementation). Tier-2 weak-spot fix. Scope: the memory wall in
`proteon-search/src/gapped.rs::smith_waterman` — **only** the storage layout; the DP and
its results are unchanged.

## 1. Problem

`smith_waterman` (Gotoh 3-matrix affine local SW) allocates **6 full
`(q+1)×(t+1)` arrays**: `m`, `x`, `y` (i32) + `tb_m`, `tb_x`, `tb_y` (u8) =
**15 bytes/cell** (`gapped.rs:167-179`). A 3000×3000 protein pair is ~135 MB; the header
admits it is "sized for ~few-thousand residues … without memory pressure" — past that it
is a wall, and it allocates this per pair.

## 2. Hard constraint: results must NOT change

`smith_waterman` is the **CPU oracle** for every GPU SW kernel (`gpu/sw.rs`,
`gpu/pssm_sw*.rs` compare GPU output to it) AND is called by the search gapped stage
(`search.rs:631,809`). The unit tests pin exact scores/CIGARs/coordinates. So the fix
**must return byte-identical `GappedAlignment`** for every input. This rules out banding /
X-drop (they change which alignment is optimal and would diverge from the GPU kernels).
A pure storage-layout change is the only safe lever here.

## 3. Fix — rolling score rows + single-byte packed traceback

**Score matrices → rolling 2 rows.** Each cell `(i,j)` reads only row `i-1` and the
already-computed part of row `i`:
- `m[i,j]` needs `m[i-1,j-1]`, `x[i-1,j-1]`, `y[i-1,j-1]` (prev row) and nothing from row
  `i` for M itself;
- `x[i,j]` needs `m[i,j-1]`, `x[i,j-1]` (current row, already computed);
- `y[i,j]` needs `m[i-1,j]`, `y[i-1,j]` (prev row).

So keep `prev_{m,x,y}` and `cur_{m,x,y}` each of length `cols` (O(t)), swap after each `i`.
This removes the three O(q·t) i32 arrays entirely.

**Traceback → one u8 per cell.** The traceback still needs a per-cell record (O(q·t) is
unavoidable for exact traceback without Myers-Miller D&C — out of scope). Today that is 3
bytes; pack into 1:
- `tb_m ∈ {0,1,2,3}` (stop / from-M / from-X / from-Y) → bits 0-1;
- `tb_x ∈ {4,5}` (from-M / from-X) → 1 bit (bit 2): `0 ⇒ came from M, 1 ⇒ from X`;
- `tb_y ∈ {6,7}` (from-M / from-Y) → 1 bit (bit 3).

`packed = tb_m | (x_bit << 2) | (y_bit << 3)`. Traceback unpacks: in state M read bits 0-1;
in state X read bit 2 (0→M, 1→X); in state Y read bit 3 (0→M, 1→Y). One `tb: Vec<u8>` of
`rows*cols`.

**Result:** 15 bytes/cell → **1 byte/cell + O(t)** scratch (~15×). 3000² pair 135 MB → ~9 MB.
Better cache locality too (working set = a few rows, not the whole grid), so likely also
faster for large pairs. Same DP order, same tie-breaks (keep the existing `>=`/`>`
comparisons verbatim so the optimal-path selection is bit-identical), same outputs.

## 4. Edge cases to preserve exactly
- `neg_sent = i32::MIN/4` boundary for `x`/`y` row 0 and column 0 — the rolling `prev`/`cur`
  rows must be initialized to the SAME boundary the full matrix had (m row/col 0 = 0;
  x/y = neg_sent) so the first row/column arithmetic matches.
- `best_score`/`best_i`/`best_j` tracking is unchanged (scan during fill).
- Local clamp at 0 with `tb_m = 0` (stop) unchanged.
- Empty input / no-positive-alignment `None` paths unchanged.

## 5. Tests
- All existing `gapped.rs` unit tests must pass **unchanged** (they pin exact results) —
  the primary correctness guard.
- The GPU-oracle parity tests (`gpu/sw.rs`, `gpu/pssm_sw*.rs`) compare GPU vs this CPU
  function; they must still pass (run the non-GPU ones; GPU ones skip without a device).
- **New:** a randomized differential test — for many random (q, t, matrix, gaps), assert
  the new implementation's `GappedAlignment` equals a reference computed by the OLD
  full-matrix code (vendored into the test, or a brute-force small-case checker). Cover
  q<t, q>t, q==t, all-gap, no-alignment, length-1.
- **New:** a memory-shape assertion is impractical to unit-test directly; instead a large
  pair (e.g. 4000×4000) that would have been ~240 MB completes — a smoke that it no longer
  allocates the full grid (guard via a generous size that the old code would strain).

## 6. Non-goals
- SIMD (Farrar striped SW) — a separate SPEED change; complex with the exact-traceback +
  byte-exact-oracle constraint; deferred.
- Myers-Miller / Hirschberg O(min(q,t)) traceback — would remove the last O(q·t) term but
  is far more complex and error-prone; the 1-byte traceback already lifts the wall to
  ~12k residues. Deferred unless a real need appears.
- Banding / X-drop — would change results (see §2). Not applicable to the shared exact
  aligner.

## 7. Files
- `proteon-search/src/gapped.rs` — `smith_waterman` body + traceback; new differential test.

## 8. Review log (claudex)
Approach confirmed sound. Adopted: swap all three prev/cur pairs TOGETHER (never alias),
explicit column-0 reset each row; pack traceback FROM PREDICATES (`x_bit = x_pred==5`,
`y_bit = y_pred==7`); preserve exact op order / tie-breaks (`>=` gap open-vs-extend, `>`
M predecessors and best-cell, clamp at `m_val<=0`); differential test against the
**preserved OLD full-matrix** implementation (brute-force checks score only, not the
exact path) with tie-heavy cases (M=X, M=Y, X=Y, all-equal; open=extend ties; multiple
equal best cells; `m_val==0`; length-1; asymmetric matrices; zero gap penalties);
4000² test is `#[ignore]`'d stress, not normal CI. **Bug found:** the existing
`gapped.rs:172` traceback comment swaps X/Y ("up"/"left") vs the recurrence — X is the
current-row LEFT dependency (Delete, `j-=1`), Y is the prev-row UP dependency (Insert,
`i-=1`); follow the recurrence and fix the comment. No findings rejected.
