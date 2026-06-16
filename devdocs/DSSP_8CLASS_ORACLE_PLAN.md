# DSSP_8CLASS_ORACLE_PLAN — canonical 8-class DSSP oracle (RELIABILITY §12)

Status: DRAFT (for claudex). Builds on the existing `tests/oracle/test_dssp_oracle.py`
(pydssp, **3-class only** H/E/-). proteon emits the full 8-class DSSP alphabet
(H,G,I,E,B,T,S,C), and pydssp's 3-class collapse cannot validate G/I/B/T/S. This
adds a **canonical 8-class oracle** so the helix-flavor (G/I), beta-bridge (B),
and turn/bend (T/S) classes pydssp can't see are actually checked.

## 1. Why a second DSSP oracle
- pydssp is an independent NumPy reimplementation but only emits **3 classes**.
- The canonical DSSP (Kabsch–Sander, as shipped by `mkdssp` and `gmx dssp`)
  emits all **8 classes** — the ground truth proteon's full alphabet needs.
- The new disagreements an 8-class compare surfaces are exactly the interesting
  ones: 3-10 vs alpha helix, isolated bridge vs ladder, turn vs bend.

## 2. Backend abstraction (the crux)
A `_reference_dssp8(pdb_path) -> dict[(chain,resnum,icode) -> ss8] | None` that
tries, in order:
1. **mkdssp** via Biopython `Bio.PDB.DSSP` (already a CI dep). Returns per-residue
   `(chain, (' ', resseq, icode)) -> ss` keyed by residue ID ⇒ **ID-based
   alignment** (robust on multi-chain / filtered residues). CI installs it with
   `apt-get install dssp` (package `dssp` 4.0.4 → `/usr/bin/mkdssp`, ~2 MB).
2. **`gmx dssp`** (GROMACS) if a `gmx` binary is found (`PROTEON_GMX` env, then
   `shutil.which`, then the sibling `gromacs-*/build/bin/gmx`). Run
   `gmx dssp -s <pdb> -o <dat> -hmode dssp -polypro no`; parse the positional
   one-char-per-residue `.dat` (`~`→`C`). **Positional** alignment (no residue
   IDs in the `.dat`) ⇒ used only when the residue COUNT matches proteon's.
3. Neither present ⇒ `pytest.skip`.

Both normalise to proteon's residue order; the comparison body is shared.
**Local validation uses gmx** (built at `gromacs-2026.1/build/bin/gmx`); **CI uses
mkdssp** (light apt install). Two canonical DSSP implementations of the same paper
is, like pydssp, the right oracle shape.

### `-polypro no`
`gmx dssp` defaults to emitting a `P` (polyproline-II) class proteon does not
model; `-polypro no` disables it so the alphabets line up (measured: removes all
spurious `C->P` disagreements; 1enh goes 94→100%).

## 3. Alignment
- **mkdssp (id-based):** iterate proteon residues `(chain_id, resnum, icode)` in
  DSSP order, look each up in the Biopython dict. Residues absent from one side
  (HETATM, gmx/mkdssp filtering) are reported as `coverage` and excluded from the
  agreement denominator — NOT a hard failure (handles the multi-chain off-by-N
  seen on 1ake +1, 4hhb +3).
- **gmx (positional):** require `len(ref) == len(proteon)`; skip with a clear
  message otherwise (single-chain structures only under this backend).

## 4. Comparison + discrepancy categorization
For the aligned residues:
- **8-class agreement** `mean(ref8 == proteon8)`.
- **3-class agreement** (collapse H/G/I→H, E/B→E, rest→L) — should ≈ the pydssp
  numbers.
- **Confusion categorization:** bucket each disagreement as
  - `helix-flavor` (both in {H,G,I}) — alpha/3-10/pi boundary, benign
  - `strand-flavor` (both in {E,B}) — ladder vs isolated bridge, benign
  - `loop-flavor` (both in {T,S,C}) — turn/bend/coil, benign
  - `cross` (helix↔strand, or SS↔loop) — the real ones
  Report the full counter in the failure message.

## 5. Thresholds (measured on the local gmx run, with margin)
proteon vs `gmx dssp -polypro no`, 8-class: 1crn 93.5%, 1ubq 98.7%, 1enh 100%
(1ake/4hhb length-mismatch under positional → covered by mkdssp id-align in CI).
- **8-class agreement ≥ 0.90** per structure (1crn's 93.5% has margin; the gap is
  a genuine 3-10-helix boundary call).
- **3-class agreement ≥ 0.95** per structure (matches the pydssp oracle).
- **No gross confusion:** `cross` (helix↔strand) disagreements ≤ 2% of residues.

## 6. Test set
Reuse the pydssp oracle's five: 1crn, 1ubq, 1enh (single-chain, gmx-validated
locally), 1ake, 4hhb (multi-chain, mkdssp id-aligned in CI). Each
`@pytest.mark.oracle("dssp8")`, skips cleanly when no backend present.

## 7. CI wiring
Add to the `test.yml` oracle job, before pytest:
```yaml
- name: Install mkdssp (canonical DSSP oracle)
  run: sudo apt-get update && sudo apt-get install -y dssp
```
The existing pydssp oracle stays (3-class, every PR, pip-light); this 8-class
oracle runs alongside it when `mkdssp` is present.

## 8. Non-goals
- Replacing the pydssp oracle (kept — independent NumPy impl, no system dep).
- Validating the `P` (PP-II) class (proteon doesn't model it; `-polypro no`).
- GROMACS in CI (heavy); gmx is the LOCAL validation backend only.

## 9. Review log (claudex + in-container validation) — adopted
1. **mkdssp is the required CI backend; gmx is a clearly-marked LOCAL diagnostic**
   (codex). Both normalise to a `{(chain,resnum,icode): ss8}` dict so the
   comparison is **id-based for both** (gmx builds the dict by zipping its
   positional string with proteon's AA-residue ids, requiring a length match).
2. **Assert coverage SEPARATELY from agreement** (codex): `coverage = matched /
   proteon_AA_residues`, `agreement = equal / matched`. Measured 100% coverage on
   all five via mkdssp id-align ⇒ assert `coverage ≥ 0.98`; list unmatched
   residues in the failure message.
3. **Validated mkdssp in an ubuntu:24.04 container** (codex's top risk) — caught
   two real traps before shipping:
   - mkdssp 4.2.2 treats any file NOT starting with `HEADER` as mmCIF and fails
     (1ubq starts with `REMARK`). Fix: **prepend a synthetic `HEADER` line** to a
     temp copy before handing it to Biopython/mkdssp.
   - mkdssp emits a `P` (polyproline-II) class proteon does not model. Normalise
     mkdssp `-` (loop) AND `P` (PP-II) → `C`, same spirit as `-polypro no` for gmx.
4. **Thresholds re-derived from REAL mkdssp 4.2.2** (not gmx): 8-class 93.0–100%,
   3-class 93.5–100%, helix↔strand cross = 0 on all five. Set:
   - 8-class agreement ≥ **0.90** (4hhb floor 93.0%)
   - 3-class agreement ≥ **0.92** (1crn floor 93.5% — the 3-10-helix-vs-turn
     region is a genuine DSSP-4.x convention difference, not a bug)
   - helix↔strand confusion **== 0** (codex: a swap is far more suspicious than a
     boundary wobble; measured 0 everywhere)
   - report the full confusion counter in the failure message.
5. mkdssp counts **match proteon exactly** (1ake 428, 4hhb 574) where gmx differed
   (+1, +3) — Biopython+proteon filter HETATM consistently ⇒ multi-chain
   id-alignment is clean (gmx's positional path skips those).

## 10. Open questions (resolved)
1. Is ID-based alignment excluding unmatched residues from the denominator the
   right call, or should a large coverage gap (say <90% residues matched) be a
   hard failure (it usually means a real filtering bug, not a boundary wobble)?
2. 8-class ≥ 0.90 — is that too loose? 1crn is the floor at 93.5%; the others are
   ≥98.7%. Could set 0.90 globally but assert the *3-class* tighter at 0.95.
3. Should the gmx-positional backend be dropped entirely (mkdssp-only, simpler)
   given CI is mkdssp? Keeping gmx is purely for LOCAL validation without
   installing mkdssp — worth the second code path?
4. Biopython `DSSP` needs the model + a file path; it re-parses the PDB itself.
   Any risk its residue set diverges from proteon's enough to make ID-alignment
   coverage low on the multi-chain cases? (Mitigated by reporting coverage.)
