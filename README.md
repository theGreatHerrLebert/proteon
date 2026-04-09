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

- **Rust core** — 24.6x faster than Biopython for SASA, 34x for dihedrals
- **Rayon parallelism** — batch operations scale to all cores automatically
- **Zero-GIL pipelines** — `load_and_analyze()` runs entirely in Rust, 6x faster than a Python loop
- **Oracle-validated on 1000 random PDB structures** — SASA within 0.17% of Biopython (median), 96.3% PDB load rate
- **Alignment accuracy** — TM-scores match C++ USAlign to 0.003 median difference (4,656 pair benchmark)
- **342 tests** — comprehensive test suite including cross-tool oracle validation

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
Shrake-Rupley algorithm, within 0.2% of Biopython:
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

### Force Field & Energy Minimization
AMBER96 force field with steepest descent minimization — fix hydrogen clashes after loading or editing:
```python
# Energy breakdown
e = ferritin.compute_energy(s)
print(f"Total: {e['total']:.1f} kcal/mol")
print(f"  bonds={e['bond_stretch']:.1f}  angles={e['angle_bend']:.1f}")
print(f"  torsions={e['torsion']:.1f}  vdw={e['vdw']:.1f}  elec={e['electrostatic']:.1f}")

# Minimize hydrogens only (heavy atoms frozen)
result = ferritin.minimize_hydrogens(s)
print(f"{result['initial_energy']:.1f} → {result['final_energy']:.1f} in {result['steps']} steps")

# Full structure minimization
result = ferritin.minimize_structure(s, max_steps=1000, gradient_tolerance=0.1)

# Batch parallel (rayon, GIL-released)
results = ferritin.batch_minimize_hydrogens(structures, n_threads=-1)

# Zero-GIL: load from disk + minimize in one Rust call
results = ferritin.load_and_minimize_hydrogens(pdb_files, n_threads=-1)
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

Validated on 1,000 randomly sampled PDB structures (diverse sizes, all experimental methods):

| Metric | Value |
|--------|-------|
| **SASA accuracy** | Within 0.17% of Biopython (median), 99.8% within 5% |
| **SASA speed** | **25.7x faster** than Biopython (median, up to 100x on large structures) |
| **TM-align accuracy** | Median 0.003 TM-score diff vs C++ USAlign (4,656 pairs) |
| **TM-align speed** | Parity with C++ (25.9 ms vs 22.5 ms median per pair) |
| **Dihedrals speed** | **34x faster** than Python |
| **Load + analyze** | 101 structures/second (zero-GIL pipeline) |
| **PDB loading** | 96.3% of random PDB archive loads successfully |
| **Batch parallelism** | Rayon-based, scales to all cores automatically |

## Architecture

```
Pure Rust (no Python dependency)
├── ferritin-align    — TM-align, SOI-align, FlexAlign, MM-align
│   ├── core/         — Kabsch, TM-score, DP, secondary structure
│   ├── ext/          — US-align extensions (BLOSUM, SOI, Flex, MM-align)
│   └── search/       — structural alphabet (3Di-style, VQ-VAE, WIP)
├── ferritin-io       — PDB/mmCIF I/O via pdbtbx
└── ferritin-bin      — CLI binaries (tmalign, usalign)

PyO3 connector (cdylib, rayon, GIL-released)
└── ferritin-connector — analysis, SASA, DSSP, H-bonds, force field, alignment

Python package
└── ferritin           — clean Pythonic API
```

## Ferritin Data Engine

Bulk structure → Parquet pipeline. One command turns raw PDB/mmCIF files into queryable datasets:

```bash
# Ingest a directory of structures into a single Parquet file
ferritin-ingest structures/ --out features.parquet

# One Parquet file per structure
ferritin-ingest structures/ --out output/ --per-structure

# Control parallelism and chunk size
ferritin-ingest structures/ --out features.parquet -j 8 --chunk-size 1000
```

Output is a columnar Parquet table (Zstd compressed) with 17 columns per atom — ready for pandas, polars, DuckDB, Spark, or PyTorch Geometric:

```python
import duckdb

# "Show me all glycine CA atoms in chain A with high B-factors"
duckdb.sql("""
    SELECT structure_id, residue_serial, b_factor
    FROM 'features.parquet'
    WHERE residue_name = 'GLY' AND atom_name = 'CA'
      AND chain_id = 'A' AND b_factor > 30
    ORDER BY b_factor DESC
""")
```

The shift:
- **Before:** load structure, parse quirks, compute features, write one-off code, repeat.
- **After:** run one ingestion command, get a usable structural dataset.

### Arrow/Parquet API (Python)
```python
import ferritin

s = ferritin.load("protein.pdb")

# Structure → Arrow IPC bytes (zero-copy to pyarrow/polars)
ipc = ferritin.to_arrow(s, "1crn")

# Arrow → back to structure (round-trip)
structures = ferritin.from_arrow(ipc)

# Direct to Parquet (Zstd compressed, from Rust)
ferritin.to_parquet(s, "output.parquet", "1crn")

# Parquet → back to structures
results = ferritin.from_parquet("output.parquet")
for structure_id, structure in results:
    print(f"{structure_id}: {structure.atom_count} atoms")
```

## CLI Tools

```bash
# Align two structures
cargo run --release --bin usalign -- s1.pdb s2.pdb

# Batch all-against-all (parallel)
cargo run --release --bin usalign -- chain_list.txt --dir /pdbs/ --outfmt 2

# Bulk ingest to Parquet
cargo run --release --bin ingest -- structures/ --out features.parquet
```

## Geometric Deep Learning Examples

Ferritin is the missing link between PDB files and GNNs. Zero glue code:

```python
# Load 100 structures, extract ALL features, build graphs → train GCN
results = ferritin.load_and_analyze(pdb_files, n_threads=-1)  # 5 seconds

for r in results:
    graph = Data(
        x=node_features(r["phi"], r["psi"], r["rsa"], r["dssp"]),  # 14 features
        edge_index=edges_from(r["contact_map"]),                     # CA-CA contacts
        pos=torch.tensor(r["ca_coords"]),                            # 3D coordinates
    )
```

| Example | Task | Result |
|---------|------|--------|
| `06_geometric_dl_pipeline.py` | Protein fold classification (GCN) | **94.7% accuracy** |
| `07_bfactor_prediction.py` | B-factor regression (GCN) | **Pearson r=0.54** |
| `08_interface_residues.py` | Interface prediction (GAT) | **F1=0.42** |
| `09_structure_similarity.py` | Siamese GNN for TM-score | **Pearson r=0.72** |

All examples use ferritin for data prep and PyTorch Geometric for training. Ferritin has no DL dependency — it produces numpy arrays, you bring your own framework.

## Examples

See [`examples/`](examples/) for runnable scripts:

**Structural analysis:**
- `01_load_and_explore.py` — I/O, hierarchy, DataFrame
- `02_structural_alignment.py` — TM-align, batch, superposition
- `03_contact_map.py` — distance matrices, contacts
- `04_ramachandran.py` — backbone dihedrals, SS correlation
- `05_sasa_analysis.py` — SASA, RSA, burial classification

**Geometric deep learning (requires `pip install torch torch-geometric`):**
- `06_geometric_dl_pipeline.py` — PDB → features → GCN fold classifier
- `07_bfactor_prediction.py` — per-residue B-factor regression
- `08_interface_residues.py` — protein-protein interface prediction
- `09_structure_similarity.py` — Siamese GNN for structure similarity

## Agent-Aware Documentation

Ferritin includes structured **Agent Notes** in selected public boundary functions — guidance specifically for AI agents and LLM-powered tools that call ferritin programmatically.

```python
def atom_sasa(structure, probe=1.4, n_points=960):
    """Compute per-atom SASA using Shrake-Rupley algorithm.

    Args / Returns / Examples...

    Agent Notes:
        WATCH: probe=1.4 is the standard water probe radius. Changing it
        changes the physical interpretation of the result.

        PREFER: For many structures, use batch_total_sasa() with n_threads=-1.

        COST: Crambin (327 atoms): ~12ms. Large complex (58k atoms): ~230ms.
    """
```

The convention uses a small set of keyword prefixes so the notes stay short and scannable:

| Prefix | Meaning |
|--------|---------|
| `WATCH` | Common gotchas that are NOT bugs |
| `PREFER` | Better alternatives for common use cases |
| `COST` | Performance or memory implications |
| `INVARIANT` | Important guarantees callers can rely on |

Policy:

- Use Agent Notes only on public APIs where misuse is likely or where a caller must make a decision.
- Prefer `WATCH`, `PREFER`, `COST`, and optionally `INVARIANT`.
- Do not restate obvious behavior already covered by the signature or docstring.
- If a note describes an important behavioral guarantee or trap, it should be backed by tests or a clear invariant in the implementation.
| `CAUTION` | Destructive or misleading behaviors |
| `COST` | Computational complexity and timing |
| `VERIFY` | What to check in the result |

This ensures AI agents that read function signatures before calling them always see the most critical usage guidance — preventing the most common mistakes in automated structural biology workflows.

## References

**Alignment:**
- Zhang & Skolnick. "TM-align: a protein structure alignment algorithm based on the TM-score." [*Nucleic Acids Research* 33, 2302-9 (2005)](https://doi.org/10.1093/nar/gki524)
- Zhang, Pagnon Braunstein, et al. "US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes." [*Nature Methods* 19(9), 1109-1115 (2022)](https://doi.org/10.1038/s41592-022-01585-1)

**Structural analysis:**
- Kabsch & Sander. "Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features." [*Biopolymers* 22, 2577-2637 (1983)](https://doi.org/10.1002/bip.360221211)
- Shrake & Rupley. "Environment and exposure to solvent of protein atoms." [*J Mol Biol* 79(2), 351-71 (1973)](https://doi.org/10.1016/0022-2836(73)90011-9)
- Tien et al. "Maximum allowed solvent accessibilities of residues in proteins." [*PLoS ONE* 8(11), e80635 (2013)](https://doi.org/10.1371/journal.pone.0080635)

**Structural search (in development):**
- van Kempen et al. "Fast and accurate protein structure search with Foldseek." [*Nature Biotechnology* 41, 243-246 (2023)](https://doi.org/10.1038/s41587-023-01773-0)

**Force field:**
- Cornell et al. "A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules." [*J. Am. Chem. Soc.* 117(19), 5179-5197 (1995)](https://doi.org/10.1021/ja00124a002)

**Infrastructure:**
- Hildebrandt et al. "BALL — Biochemical Algorithms Library 1.3." [*BMC Bioinformatics* 11, 531 (2010)](https://doi.org/10.1186/1471-2105-11-531)
- Schulte, D. "pdbtbx: A Rust library for reading, editing, and saving crystallographic PDB/mmCIF files." [*JOSS* 7(77), 4377 (2022)](https://doi.org/10.21105/joss.04377)

## License

MIT — see [LICENSE](LICENSE).

Clean-room Rust reimplementation. The original algorithms are from published papers. Please cite the relevant papers if you use Ferritin in published work.
