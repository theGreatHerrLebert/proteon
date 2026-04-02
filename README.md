# Ferritin

**Fast structural bioinformatics in Python, powered by Rust.**

One `pip install`, Rust speed, clean API. Load structures, align proteins, compute SASA, assign secondary structure, detect hydrogen bonds — all with batch parallelism out of the box.

```python
import ferritin

# Load and analyze
s = ferritin.load("1crn.pdb")
sasa = ferritin.atom_sasa(s)                          # Shrake-Rupley SASA
ss = ferritin.dssp(s)                                  # Kabsch-Sander DSSP
phi, psi, _ = ferritin.backbone_dihedrals(s)           # Ramachandran angles
hbonds = ferritin.backbone_hbonds(s)                   # H-bond detection
ca = ferritin.extract_ca_coords(s)                     # CA coordinates
cm = ferritin.contact_map(ca, cutoff=8.0)              # Contact map
df = ferritin.to_dataframe(s)                          # → pandas DataFrame

# Align structures
result = ferritin.tm_align(s1, s2)
print(f"TM-score: {result.tm_score_chain1:.4f}")

# Batch: process 1000 structures on all cores
results = ferritin.load_and_analyze(pdb_files, n_threads=-1)
```

## Installation

```bash
pip install ferritin
```

## Why Ferritin?

The structural bioinformatics landscape is fragmented: Biopython is slow, Gemmi is C++-focused, MDAnalysis is trajectory-oriented. Nobody has a single library where you `pip install` and get loading, alignment, SASA, DSSP, and batch parallelism.

Ferritin fills that gap:

- **Rust core** — 3.9x faster than Biopython for SASA, 34x for dihedrals
- **Rayon parallelism** — batch operations scale to all cores automatically
- **Zero-GIL pipelines** — `load_and_analyze()` runs entirely in Rust, 6x faster than a Python loop
- **Oracle-tested** — validated against Biopython, Gemmi, and C++ USAlign to 4-5 decimal places
- **316 tests** — comprehensive test suite including cross-tool oracle validation

## Features

### Structure I/O
```python
s = ferritin.load("protein.pdb")        # auto-detect PDB/mmCIF
s = ferritin.load("protein.cif")        # mmCIF
ferritin.save(s, "output.pdb")          # save

# Batch load (parallel)
structures = ferritin.batch_load(pdb_files, n_threads=-1)
```

### Structural Alignment
Four alignment algorithms with single-pair, one-to-many, and many-to-many variants:
```python
r = ferritin.tm_align(s1, s2)                   # TM-align
r = ferritin.soi_align(s1, s2)                   # sequence-order independent
r = ferritin.flex_align(s1, s2)                   # flexible (hinge-based)
r = ferritin.mm_align(complex1, complex2)         # multi-chain complex

# Batch: all pairs, parallel
results = ferritin.tm_align_many_to_many(queries, targets, n_threads=-1)
```

### SASA (Solvent Accessible Surface Area)
Shrake-Rupley algorithm, 0.18% agreement with Biopython:
```python
sasa = ferritin.atom_sasa(s)             # per-atom (Å²)
res_sasa = ferritin.residue_sasa(s)      # per-residue
rsa = ferritin.relative_sasa(s)          # relative (0-1, burial classification)
total = ferritin.total_sasa(s)           # total

# Batch (rayon parallel)
totals = ferritin.batch_total_sasa(structures, n_threads=-1)
```

### DSSP (Secondary Structure)
Native Kabsch-Sander implementation — no external binary needed:
```python
ss = ferritin.dssp(s)          # → "CEEEEEETTTCEEEEECHHHHHHHH..."
ss_arr = ferritin.dssp_array(s) # → numpy uint8 array

# Batch
all_ss = ferritin.batch_dssp(structures, n_threads=-1)
```

### Hydrogen Bonds
```python
hb = ferritin.backbone_hbonds(s)          # Kabsch-Sander energy criterion
ghb = ferritin.geometric_hbonds(s)        # distance-based, all polar atoms
counts = ferritin.hbond_count(s)          # per-residue participation
```

### Geometry & Analysis
```python
phi, psi, omega = ferritin.backbone_dihedrals(s)     # Ramachandran
ca = ferritin.extract_ca_coords(s)                    # CA coordinates
dm = ferritin.distance_matrix(ca)                     # pairwise distances
cm = ferritin.contact_map(ca, cutoff=8.0)             # contact map
rg = ferritin.radius_of_gyration(s)                   # radius of gyration
ss = ferritin.assign_secondary_structure(ca)          # quick SS (CA-distance)
rmsd, R, t = ferritin.kabsch_superpose(x, y)          # optimal superposition
```

### Atom Selection Language
```python
mask = ferritin.select(s, "CA and chain A")          # → boolean mask
mask = ferritin.select(s, "backbone and resid 1-50") # backbone of region
mask = ferritin.select(s, "protein and heavy")       # heavy atoms only
mask = ferritin.select(s, "(chain A or chain B) and not water")

coords = s.coords[mask]                              # index any per-atom array
bfactors = s.b_factors[mask]
```

### DataFrame Export
```python
df = ferritin.to_dataframe(s)                  # pandas
df = ferritin.to_dataframe(s, engine="polars")  # polars
```

### Zero-GIL Pipelines
Load files and compute everything in one Rust call — no Python overhead:
```python
# Load + full analysis (CA coords, distance matrix, contact map, dihedrals, Rg)
results = ferritin.load_and_analyze(pdb_files, n_threads=-1)
for r in results:
    print(f"{r['path']}: {r['n_ca']} CA, Rg={r['rg']:.1f}")

# Load + SASA
results = ferritin.load_and_sasa(pdb_files, n_threads=-1)

# Load + DSSP
results = ferritin.load_and_dssp(pdb_files, n_threads=-1)
```

## Performance

Benchmarks on 97 PDB files (327 to 58,870 atoms):

| Operation | Ferritin | vs Biopython | Rayon speedup |
|-----------|----------|-------------|---------------|
| SASA (single) | 12ms | **3.9x faster** | — |
| Dihedrals (single) | 3ms | **34x faster** | — |
| Batch SASA (97 structs) | 17s | — | **2.5x** (16 threads) |
| load_and_analyze (97) | 585ms | **6x faster** | pipeline |

## Architecture

```
Pure Rust (no Python dependency)
├── ferritin-align    — TM-align, SOI-align, FlexAlign, MM-align
├── ferritin-io       — PDB/mmCIF I/O via pdbtbx
└── ferritin-bin      — CLI binaries (tmalign, usalign)

PyO3 connector (cdylib, rayon, GIL-released)
└── ferritin-connector — analysis, SASA, DSSP, H-bonds, alignment

Python package
└── ferritin           — clean Pythonic API
```

## CLI Tools

```bash
# Align two structures
cargo run --release --bin usalign -- s1.pdb s2.pdb

# Batch all-against-all (parallel)
cargo run --release --bin usalign -- chain_list.txt --dir /pdbs/ --outfmt 2
```

## Examples

See [`examples/`](examples/) for runnable scripts:
- `01_load_and_explore.py` — I/O, hierarchy, DataFrame
- `02_structural_alignment.py` — TM-align, batch, superposition
- `03_contact_map.py` — distance matrices, contacts
- `04_ramachandran.py` — backbone dihedrals, SS correlation
- `05_sasa_analysis.py` — SASA, RSA, burial classification

## Agent-Aware Documentation

Ferritin includes structured **Agent Notes** in every public function's docstring — guidance specifically for AI agents and LLM-powered tools that call ferritin programmatically.

```python
def atom_sasa(structure, probe=1.4, n_points=960):
    """Compute per-atom SASA using Shrake-Rupley algorithm.

    Args / Returns / Examples...

    Agent Notes:
        DEFAULTS: probe=1.4 is the standard water probe radius. Don't change
        unless you have a specific reason.

        PRECISION: n_points=960 gives ~0.2% accuracy. Use 100 for screening,
        960 for publication quality.

        PREFER: For many structures, use batch_total_sasa() with n_threads=-1.

        COST: Crambin (327 atoms): ~12ms. Large complex (58k atoms): ~230ms.
    """
```

The convention uses keyword prefixes to categorize guidance:

| Prefix | Meaning |
|--------|---------|
| `PREREQUISITE` | What must be true before calling |
| `INTERPRET` | How to read the output correctly |
| `WATCH` | Common gotchas that are NOT bugs |
| `PREFER` | Better alternatives for common use cases |
| `AVOID` | Anti-patterns that produce bad results |
| `CAUTION` | Destructive or misleading behaviors |
| `COST` | Computational complexity and timing |
| `VERIFY` | What to check in the result |

This ensures AI agents that read function signatures before calling them always see the most critical usage guidance — preventing the most common mistakes in automated structural biology workflows.

## References

- Zhang & Skolnick. "TM-align." *Nucleic Acids Research* 33, 2302-9 (2005)
- Zhang et al. "US-align." *Nature Methods* 19(9), 1109-1115 (2022)
- Kabsch & Sander. "DSSP." *Biopolymers* 22, 2577-2637 (1983)
- Shrake & Rupley. "Environment and exposure to solvent." *J Mol Biol* 79(2), 351-71 (1973)
- Tien et al. "Maximum allowed solvent accessibilities." *PLoS ONE* 8(11), e80635 (2013)
- Hildebrandt et al. "BALL — Biochemical Algorithms Library 1.3." *BMC Bioinformatics* 11, 531 (2010)
- Schulte, D. "pdbtbx: A Rust library for reading, editing, and saving crystallographic PDB/mmCIF files." *JOSS* 7(77), 4377 (2022)

## License

MIT — see [LICENSE](LICENSE).

Clean-room Rust reimplementation. The original algorithms are from published papers. Please cite the relevant papers if you use Ferritin in published work.
