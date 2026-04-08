#!/bin/bash
# Setup and run ferritin 50K benchmark on monster3.
# Usage: ssh monster3 "bash /path/to/setup_and_run.sh"

set -e

BENCH_DIR="/globalscratch/dateschn/ferritin-benchmark"
REPO_DIR="$BENCH_DIR/ferritin"
PDB_DIR="$BENCH_DIR/pdbs_50k"
RESULTS="$BENCH_DIR/benchmark_results.json"

echo "=== Ferritin 50K Benchmark Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname) ($(nproc) cores, $(free -h | awk '/^Mem:/{print $2}') RAM)"

# 1. Update repo
echo -e "\n[1/5] Updating repo..."
cd "$REPO_DIR"
git pull --ff-only

# 2. Build release
echo -e "\n[2/5] Building release binaries..."
cargo build --release -p ferritin-connector -p ferritin-bin 2>&1 | tail -3

# 3. Setup Python environment
echo -e "\n[3/5] Setting up Python environment..."
python3 -m venv "$BENCH_DIR/venv" 2>/dev/null || true
source "$BENCH_DIR/venv/bin/activate"
pip install --quiet maturin numpy pytest pandas pyarrow
cd ferritin-connector
maturin develop --release 2>&1 | tail -1
cd ..
pip install -e packages/ferritin/ --quiet

# 4. Download PDB corpus (if not already done)
echo -e "\n[4/5] Downloading 50K PDB corpus..."
mkdir -p "$PDB_DIR"
n_existing=$(ls "$PDB_DIR"/*.pdb "$PDB_DIR"/*.cif 2>/dev/null | wc -l)
if [ "$n_existing" -lt 45000 ]; then
    python3 benchmark/download_pdbs.py --n 50000 --out "$PDB_DIR" --workers 64
else
    echo "  Already have $n_existing files, skipping download."
fi

# 5. Run benchmark
echo -e "\n[5/5] Running benchmark..."
python3 benchmark/run_benchmark.py \
    --pdb-dir "$PDB_DIR" \
    --n 50000 \
    --threads 0 \
    --output "$RESULTS"

echo -e "\n=== Done ==="
echo "Results: $RESULTS"
