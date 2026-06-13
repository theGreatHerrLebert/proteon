# RESULT — Analytic SES surface: robustness, performance, and a BALL oracle

Summary of the work that landed on `main` via PR #110 (squash commit `83cfeee`,
2026-06-10). It hardens the analytic solvent-excluded-surface (SES / Connolly)
mesher, makes it tractable on real proteins, and puts a CI-gated parity oracle
against BALL behind it.

Design/diagnosis detail lives in `devdocs/SES_CDT_CROSSING.md` and
`devdocs/SES_DIRECTED_PERTURBATION.md`; this is the results view.

---

## TL;DR

| Area | Before | After |
|------|--------|-------|
| `build_graph` (1558 atoms) | 9.2 s | **0.73 s** (12.6×) |
| 1sq9 (2981 atoms) | couldn't finish in 120 s | **full mesh in 411 s** |
| CDT-crossing subset (22 proteins) | 5 OK / 15 timeout | **11 OK / 7 timeout** |
| SES area vs BALL (5 structures) | hand-run, one input | CI-gated, **+0.04 %..+0.08 %** |
| BALL SES oracle | none | `evident` claim, green in CI |

---

## 1. Robustness — the #1 analytic failure fixed

On a 317-protein RCSB corpus, the dominant hard failure of the analytic mesher
was a constrained-Delaunay **"boundary crossing"** — **22 of 36 hard failures
(61 %)**.

**Diagnosis (instrumented).** The *sampled* contact-cap boundary self-intersects
at zero-clearance near-tangencies in the reduced-surface arrangement — not a
chart-projection artifact. Proven by a 2.4° sliver case and a 76°
within-hemisphere case both still crossing, where projection distortion is
negligible.

**What we tried, and what won.** Three candidate fixes were implemented and
measured:
- **D — boundary-resolution escalation**: refine `n_theta` on a crossing. Could
  not clear zero-clearance tangencies; 2–8× the (slow) whole-protein remesh.
  Reverted.
- **A — atom-perturbation retry** (shipped): route the crossing through the
  existing deterministic atom-jitter retry. A ≤1e-2 Å nudge opens the
  near-tangency at the *arrangement* level so the boundary is simple again.
- This is exactly what BALL does in its own triangulation
  (`surfaceProcessor.C:64`: probe-radius jitter ±0.01 Å, ≤10×) — independent
  confirmation that bounded geometric perturbation is the standard remedy.

**Validated.** Recovered surfaces match BALL's analytic area to **≤0.32 %**.
Measured recovery depth is **5–11 perturbations** (median 9) — the perturbation is
an undirected random walk, so a budget cap is not viable (it loses recoveries).

---

## 2. Performance — the real bottleneck was `build_graph`, O(N³) → O(N·k²)

Profiling (env `SES_PROFILE`) overturned the latency story. The cost was **not**
the retry count — it was that `build_graph` re-runs an O(N³) reduced-surface
enumeration **every** attempt:
- 9hfa (1558 atoms): `build_graph` **9.2 s** + toric 1.1 s + caps 4.3 s.
- 1sq9 (2981 atoms): `build_graph` **alone > 120 s**.

`enumerate_rs_faces` / `enumerate_toric_faces` brute-forced **all O(N³) atom
triples/pairs** with an O(N) clearance/blocker scan — no spatial acceleration. (This
is why BALL, which uses spatial acceleration, was seconds where proteon was
minutes — the gap was algorithmic, not surface quality.)

**Fix.** A uniform `NeighborGrid` (cell = the interaction cutoff `2·(r_max+probe)`
+ headroom). An RS face's three atoms are pairwise within the cutoff, and any
clearance/blocker atom is within the cutoff of the probe/roll-centre, so the
27-cell stencil is provably complete — the enumerated faces are **bit-identical**
to brute force (a parity test asserts faces, order, probe bits, and toric θ-bits;
all BALL-gated tests pass).

**Measured:** `build_graph` 9.2 s → **0.73 s** (12.6×); 1sq9 from "couldn't finish
in 120 s" to a full mesh in 411 s. On the 22-protein crossing subset, analytic
recoveries **5 → 11** and timeouts **15 → 7**. This attacks the general
large-protein slowness behind the original 163/317 corpus timeouts, not just the
crossing cases. The remaining 7 timeouts are 4000–6000-atom proteins where the
cap-loop (chart fill) × retries is now the dominant cost.

Also shipped: a fast perturbation magnitude schedule for crossings and a
determinism fix (contact-cap iteration `HashMap` → `BTreeMap`, so the first
reported crossing — and thus the retry — is reproducible run-to-run).

---

## 3. The BALL SES oracle (evident)

A CI-gated `evident` claim — `surface_ses_ball` — compares proteon's analytic SES
mesh against BALL's analytic `ses_area`/`ses_volume` on **byte-identical spheres**
(same centres, same per-element vdW radii), so it isolates the meshing algorithm.

**5 fixtures, 327 → 3804 atoms** (`1crn`, `1bpi`, `1aaj`, `1ubq`, `1ake`); `1aaj`
is a former degenerate-crash case, now a regression guard. Tolerances: area
< 0.5 %, volume < 1.0 %, watertight + analytic path.

| fixture | atoms | Δ area vs BALL |
|---------|------:|---------------:|
| 1crn | 327 | +0.037 % |
| 1bpi | 626 | +0.055 % |
| 1aaj | 905 | +0.052 % |
| 1ubq | 1271 | +0.058 % |
| 1ake | 3804 | +0.079 % |

`tests/oracle/test_ses_ball_oracle.py`; claim registered in `evident/evident.yaml`
(vocabularies extended: subsystem `surface.ses`, capability
`surface-ses-area-parity`). **Runs green in CI across Python 3.11/3.12/3.13.**

**What we learned about BALL.** Its *analytic* `ses_area` is robust on all 22 of
proteon's crossing-failure inputs (it's the right oracle). Its *triangulation*
(`ball.ses_mesh`) hits `DivisionByZero` on the same near-tangencies (e.g. 1ijp) —
triangulating these surfaces is genuinely hard for everyone, which is why the
analytic path is the reference.

---

## 4. ball-py — already released

`ball-py==0.1.0a6` is **already on PyPI** (cp311/cp312/cp313 manylinux wheels) with
the SES bindings `ses_area` / `ses_mesh` / `ses_graph` / `reduced_surface_stats`.
Verified the published wheel exports them; proteon's oracle passes against it.
proteon's CI pin is bumped `0.1.0a4 → 0.1.0a6`. No new release needed.

A `tools/oracle/ball_ses/` C++ driver over `libBALL` exists as a fallback, now
superseded by `ball.ses_area`.

---

## 5. Provenance

Reviewed throughout: the perturbation fix and the grid acceleration each had an
independent `codex` code review (the grid review confirmed the cutoff/stencil
geometry is sound and flagged a cell-headroom + a stronger parity test, both
applied). The diagnosis was driven by instrumented measurement, not assumption —
each pivot (refinement → perturbation → grid) was forced by data.

PR #110, all 11 CI gates green (Lint, Version Sync, Rust, CLI smoke, MMseqs2
oracle, Python 3.11/3.12/3.13, evident slim+cuda, site).

---

## 6. Open follow-ups

- **Large-protein latency tail.** 7 of the 22 crossing-subset proteins (4000–6000
  atoms) still time out: the cap-loop (chart fill) × perturbation-retries is now
  the bottleneck. A cheaper per-retry (incremental graph reuse) or a directed
  perturbation would help, but both are non-trivial.
- **Re-measure full 317-corpus coverage** post-grid-acceleration — the original
  163/317 timeouts were dominated by the O(N³) `build_graph` that is now fixed, so
  coverage should be substantially higher; worth quantifying.
- **`ball.ses_mesh` DivisionByZero** on degenerate inputs is an upstream BALL
  triangulation robustness gap — fixable with the same perturbation approach if a
  future ball-py release wants it.
- **Release-tier oracle** over a large diverse corpus that exercises the
  perturbation-recovery path (the current claim is 5 clean fixtures).
