# Install

## From PyPI

```bash
pip install proteon
```

This pulls a prebuilt wheel of the PyO3 connector for your platform.
CPU-only — no CUDA toolkit required at install time.

## From source

You need:

- Rust (stable, 1.75+)
- Python 3.10+
- A working C/C++ toolchain (for `pdbtbx`'s native deps)

```bash
git clone https://github.com/theGreatHerrLebert/proteon
cd proteon

# Build the Rust workspace (sanity check)
cargo build --workspace

# Build and install the Python package in editable mode
cd packages/proteon
pip install maturin
maturin develop --release
```

For active development you can use the in-tree venv at
`/scratch/TMAlign/proteon/.venv` (Python 3.12) which already has `proteon`
installed; activate it before running tests or examples.

## CUDA build

The `proteon-search` crate and parts of the forcefield/MD path have CUDA
kernels gated behind the `cuda` feature.

```bash
maturin develop --release --features cuda
```

Requirements:

- CUDA 12 toolkit (driver + `nvcc`)
- An RTX 30/40/50-series GPU or equivalent compute capability
- `cudarc` will load the runtime dynamically; missing GPU silently falls back
  to CPU on most paths

## Verifying the install

```python
import proteon
print(proteon.__version__)

# Quick smoke test: load a structure, run TM-align against itself.
s = proteon.load("path/to/1crn.pdb")
print(proteon.tm_align(s, s).tm_score_chain1)  # expect ~1.0
```

## Optional: oracle dependencies

Some tests in `tests/oracle/` compare proteon against OpenMM, BALL, Biopython,
and MMseqs2. See [Oracle setup](ORACLE_SETUP.md) for the full list and
install recipe — only needed if you want to run parity tests yourself.
