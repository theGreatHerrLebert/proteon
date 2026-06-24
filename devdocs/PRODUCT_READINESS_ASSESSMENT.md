# proteon — product readiness assessment (candid)

> Status: **internal assessment for external review** (2026-06-24). Written to be
> pressure-tested, not to sell. The author (Claude) wrote it; it is going to
> Codex for an adversarial second opinion before the maintainer acts on it.
> Where the author is rationalizing a gap away, say so.

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
- 4 of 6 publishable crates depend on a **patched `pdbtbx` fork via `git = …`
  with no `version`** → crates.io rejects the manifest. There is no
  `pip install proteon` / `cargo add proteon-io` today. (`proteon-search`
  dry-runs clean; `proteon-align` needs a 3-line `include!`-path fix.)
- This is a **hard gate on the word "product."** A library nobody can depend on
  is, by definition, not yet shipped. Resolution is known (publish the pdbtbx
  fork or vendor it in-tree as `proteon-pdbtbx`) but not done.

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
