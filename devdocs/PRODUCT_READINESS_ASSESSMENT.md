# proteon — product readiness assessment (candid)

> Status: **living record.** Originally written 2026-06-24 as an internal
> assessment for adversarial Codex review; **updated 2026-06-25** to record what
> shipped and to correct one materially wrong claim. The original analysis below
> is preserved (it drove the work); corrections are marked inline and summarized
> in the update section. Still written to be pressure-tested, not to sell.

## Update (2026-06-25) — status after v0.4.0

Three of the four gaps below were actioned within a day; one was **factually
wrong** and is corrected here.

- **Gap 1 ("cannot be published") was WRONG.** proteon was already on PyPI
  (v0.3.0, published 2026-06-17 via a complete, working release workflow);
  `pip install proteon` worked the whole time. The author *and* Codex treated the
  `pdbtbx` git-dep as a hard gate on "product" without checking PyPI: it only
  ever blocked **crates.io** (the Rust crates), never the **PyPI wheel** path
  (maturin resolves the git dep at build time and bundles the compiled
  connector). The crates.io blocker is intentionally left as-is — upstream
  `pdbtbx` is not under our control. **Lesson: verify the distribution channel
  before calling something unpublishable.**
- **Gap 2 (flat, under-hardened surface) — addressed (#206, #207).** The public
  surface is now tiered into `proteon.__stable__` (81 oracle-validated symbols
  with a back-compat promise) and `proteon.__experimental__`, with a canonical
  `proteon.experimental.*` namespace and a CI snapshot guard that freezes the
  stable surface against accidental growth. The stable tier's **validity
  boundaries** are now proven, not assumed — a boundary suite found and fixed a
  real bug (`compute_energy` returned a misleading `0.0` for unparameterizable
  input; now raises `ParameterizationError`). `STABILITY.md` documents the 5
  stability gates.
- **Gap 3 (breadth > depth) — mitigated, not closed.** The research frontiers
  (electrostatics BEM, GPU SES, Vina, advanced search GPU, `run_md`) are still
  present, but are now explicitly in the **experimental** tier rather than
  exposed at equal endorsement. That is the intended treatment; they remain
  frontiers.
- **Gap 4 (dev-facing docs) — partially addressed.** Added `STABILITY.md` (tier
  table + gates) and a Keep-a-Changelog `CHANGELOG.md` with versioned 0.4.0
  notes. A getting-started tutorial / published API reference for the stable tier
  is still open.

**Shipped:** v0.4.0 on PyPI (tiered surface + validity fix), PRs #206/#207/#208.
**Corrected author's read:** proteon is a **published 0.x library with a 1.0-grade,
now-tier-frozen-and-boundary-tested core.** The remaining distance to a 1.0
*label* is user-facing docs + completing the deprecation cycle for the
experimental flat-name shims — not a publishing blocker, which never really
existed for the Python product. The crates.io/Rust distribution stays gated by
upstream `pdbtbx` and is out of scope.

---

> The original 2026-06-24 assessment follows, with gap #1 corrected inline.

## What proteon is

A Rust workspace (9 crates) + a PyO3 connector + a pure-Python wrapper package,
positioned explicitly as **"a library, not a platform."** It ports and unifies
several reference tools (TM-align/US-align, MMseqs2, OpenMM-class MM/analysis)
into one Rust core with Python bindings, plus a geometric-DL supervision export
pipeline (AlphaFold/OpenFold-compatible features).

Crates: `proteon-align`, `proteon-search`, `proteon-io`, `proteon-arrow`,
`proteon-core`, `proteon-connector`, `proteon-bin`, `proteon-electrostatics`,
`proteon-vina`. Python package: `packages/proteon/` (52 modules).

## The core claim being assessed

**Is proteon ready to be presented/used as a product (a published, dependable
library others build on) — or is it still a strong personal research toolkit
that *looks* like a product?**

## Evidence FOR readiness (the strong core)

The load → prepare → analysis → forcefield → align → supervision path is
oracle-gated and now volume-tested:

- TM-align vs USAlign: 0.003 median TM drift on 4,656 pairs.
- SASA vs Biopython: 0.17% median deviation (1,000 PDBs).
- AMBER96 vs OpenMM: ≤0.5% all components at NoCutoff (218/218 invariants).
- OBC GB vs OpenMM: ≤5% GB / ≤1% total; GPU matches CPU to 1e-11.
- Fold preservation (1,000 PDBs): median TM 0.9945, 30× faster than OpenMM.
- **New (today): 47,183-structure label-safe supervision sweep on 120 cores —
  ZERO pipeline crashes, 64.1% label-safe export yield at coverage floor 0.8,
  96.4% prepared OK, slowest single export 0.74 s (4,128 residues), no perf
  cliff. 6.1% of exported residues trust-masked; masking is localized (only 48
  of 30,240 exports mask >10% of residues).**
- 161 Rust test files, 107 Python test files; CI on `main` is green across 8
  required jobs (Lint, Version Sync, Rust, CLI smoke, MMseqs2 byte-exact oracle,
  Python 3.11/3.12/3.13).

Read: the **numerical core is genuinely dependable** and has now survived the
messy real-PDB tail, not just curated benchmark sets.

## Evidence AGAINST readiness (the honest gaps)

### 1. Cannot be published / installed by anyone else

> **CORRECTED (2026-06-25): this gap was wrong.** `pip install proteon` worked
> the whole time — proteon was already on PyPI. The claim below conflated
> crates.io with the actual (PyPI wheel) distribution channel. See the update
> section at the top. Kept for the record as a cautionary example of asserting a
> blocker without checking the channel.

- ~~4 of 6 publishable crates depend on a **patched `pdbtbx` fork via `git = …`
  with no `version`** → crates.io rejects the manifest. There is no
  `pip install proteon` / `cargo add proteon-io` today.~~ The `pdbtbx` git-dep
  blocks **crates.io** (Rust crates) only; the **PyPI wheel** is built by maturin
  from the git dep and was already published. (`proteon-search` dry-runs clean;
  `proteon-align` needs a 3-line `include!`-path fix — relevant only if/when
  crates.io distribution is wanted.)
- ~~This is a **hard gate on the word "product."**~~ It is a gate on *crates.io
  Rust distribution* only, which is intentionally deferred (upstream `pdbtbx` not
  under our control). It was never a gate on the Python product.

### 2. The public Python surface is large, flat, and under-hardened
- ~52 Python modules re-exported through one flat `proteon.` namespace; the
  API surface is in the hundreds of symbols.
- As of **one week ago**, a real footgun was found by accident: the
  `load_and_*` batch convenience functions silently iterated a single path
  *string* character-by-character. It was fixed with a `normalize_paths`
  decorator + deprecation warning — but it was found by luck, not by a
  systematic surface audit. One audit pass has since happened; it is not a
  proven-stable, contract-tested API.
- For a **library, the API surface IS the product.** A surface this large that
  has had exactly one hardening pass is a stability risk: there is no declared
  stable-vs-experimental tier, no documented deprecation policy, no API
  freeze. Everything is exposed at the same level of apparent endorsement.

### 3. Breadth may exceed depth (research frontiers exposed as product)
- `proteon-electrostatics` (BEM) ships with `ELECTROSTATICS_FORMULATION.md`
  marked **DRAFT**; FMM/treecode work is partially closed-as-wrong-lever.
- GPU SES has open kernels (K3 dual-contour not default; needs field-residency
  wiring to be a win).
- `proteon-vina` has a roadmap of remaining docking work.
- Several search GPU phases (batch, sensitive, on-disk) are explicitly deferred.
- These are legitimate **research frontiers**, but they sit in the same
  workspace and (partly) the same Python namespace as the dependable core, with
  no signal to a user about which is load-bearing.

### 4. Documentation is dev-facing, not user-facing
- A 566-line README exists, plus extensive `devdocs/` roadmaps — but these are
  *internal* planning/design docs. There is no evidence of: published API docs,
  a getting-started tutorial path for an external user, or versioned changelog
  discipline beyond commit history.

## The author's actual read (to be challenged)

proteon is a **strong 0.x library with a 1.0-grade core**, not a 1.0 product.
The fastest path to "product" is **not more features** — it is **scope
narrowing + shipping discipline**:

1. **Declare tiers.** Mark a *stable* set (load/io, prepare, sasa, dssp, hbond,
   align, forcefield/minimize, supervision export) vs *experimental*
   (electrostatics BEM, GPU SES, vina, advanced search GPU). Stop exposing them
   at equal endorsement.
2. **Unblock publishing** (pdbtbx fork → crates.io or vendor in-tree;
   `proteon-align` include! fix). Without this, "product" is aspirational.
3. **Harden + freeze the stable Python surface**: full audit pass (the footgun
   was surely not unique), document it, add a deprecation policy, version it.
4. **Add user-facing docs** for the stable tier only.

Claim: doing (1)–(4) on the *existing* core would make proteon a credible,
publishable 1.0 **without writing a single new algorithm.** The core is done;
the productization is not.

> **Outcome (2026-06-25):** (1) was already done (PyPI), (2) and the tiering half
> of (3) shipped in #206/#207 (v0.4.0), (4) is partially done (`STABILITY.md` +
> `CHANGELOG.md`). The "scope narrowing + shipping discipline" thesis held: no
> new algorithm was written. Remaining for a 1.0 *label*: user-facing docs and
> the experimental-shim deprecation cycle.

## Questions for the reviewer (Codex)

1. Is the stable/experimental tiering the right primary lever, or is there a
   more fundamental blocker the author is underweighting (e.g. the pdbtbx fork
   dependency is a deeper architectural liability than "just publish it")?
2. Is "zero crashes on 47 k PDB + oracle parity" sufficient evidence of *core*
   readiness, or does it conflate *robustness* (doesn't crash) with
   *correctness* (right answer on the hard cases)? What's the missing test?
3. Is the author over-rating the core and under-rating the integration/API
   risk? For a library, where does product-readiness actually fail first —
   the algorithms or the surface?
4. What is the single highest-leverage thing to do *before* declaring a 1.0,
   and what is being treated as required that is actually optional?
5. Any failure mode of "ship the core, defer the frontiers" that the author is
   not seeing (e.g. the frontiers are load-bearing for the core in some way)?
