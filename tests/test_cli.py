"""CLI integration tests for tmalign, usalign, and ingest binaries.

Tests that binaries start, accept expected arguments, produce correct
output formats, and handle error cases gracefully.

RELIABILITY_ROADMAP P0.4: CLI Integration Tests
"""

import os
import subprocess
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "target", "release")
TMALIGN = os.path.join(BIN, "tmalign")
USALIGN = os.path.join(BIN, "usalign")
INGEST = os.path.join(BIN, "ingest")
TEST_PDBS = os.path.join(REPO, "test-pdbs")
CRAMBIN = os.path.join(TEST_PDBS, "1crn.pdb")
UBIQ = os.path.join(TEST_PDBS, "1ubq.pdb")
UBIQ_CIF = os.path.join(TEST_PDBS, "1ubq.cif")

# Skip all tests if binaries not built
pytestmark = pytest.mark.skipif(
    not os.path.isfile(TMALIGN),
    reason="Release binaries not built (run cargo build --release)",
)


def run(cmd, timeout=30):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# =========================================================================
# tmalign
# =========================================================================


class TestTMalign:
    def test_help(self):
        rc, out, err = run([TMALIGN, "--help"])
        assert rc == 0
        assert "TM-align" in out or "tmalign" in out.lower() or "Usage" in out

    def test_two_structures(self):
        rc, out, err = run([TMALIGN, UBIQ, CRAMBIN])
        assert rc == 0
        assert "TM-score" in out
        assert "RMSD" in out
        assert "Aligned length" in out

    def test_tabular_output(self):
        rc, out, err = run([TMALIGN, UBIQ, CRAMBIN, "--outfmt", "2"])
        assert rc == 0
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        assert len(lines) == 1
        fields = lines[0].split("\t")
        assert len(fields) == 11  # PDB1, PDB2, TM1, TM2, RMSD, ID1, ID2, IDali, L1, L2, Lali
        tm1 = float(fields[2])
        assert 0.0 < tm1 < 1.0

    def test_tabular_values(self):
        rc, out, err = run([TMALIGN, UBIQ, CRAMBIN, "--outfmt", "2"])
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        fields = lines[0].split("\t")
        tm1, tm2, rmsd = float(fields[2]), float(fields[3]), float(fields[4])
        l1, l2, lali = int(fields[8]), int(fields[9]), int(fields[10])
        assert l1 == 76  # ubiquitin
        assert l2 == 46  # crambin
        assert lali > 0
        assert rmsd > 0
        assert 0 < tm1 < 1
        assert 0 < tm2 < 1

    def test_fast_mode(self):
        rc, out, err = run([TMALIGN, UBIQ, CRAMBIN, "--outfmt", "2", "--fast"])
        assert rc == 0
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        assert len(lines) == 1

    def test_matrix_output(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            matrix_path = f.name
        try:
            rc, out, err = run([TMALIGN, UBIQ, CRAMBIN, "-m", matrix_path])
            assert rc == 0
            assert os.path.isfile(matrix_path)
            content = open(matrix_path).read()
            # Matrix file should have rotation/translation data
            assert len(content) > 10
        finally:
            os.unlink(matrix_path)

    def test_self_alignment(self):
        rc, out, err = run([TMALIGN, CRAMBIN, CRAMBIN, "--outfmt", "2"])
        assert rc == 0
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        fields = lines[0].split("\t")
        tm1 = float(fields[2])
        rmsd = float(fields[4])
        assert tm1 > 0.99  # self-alignment should be ~1.0
        assert rmsd < 0.01  # self-alignment RMSD should be ~0

    def test_missing_file(self):
        rc, out, err = run([TMALIGN, "nonexistent.pdb", CRAMBIN])
        assert rc != 0

    def test_circular_permutation(self):
        rc, out, err = run([TMALIGN, UBIQ, CRAMBIN, "--cp", "--outfmt", "2"])
        assert rc == 0


# =========================================================================
# usalign
# =========================================================================


class TestUSalign:
    def test_help(self):
        rc, out, err = run([USALIGN, "--help"])
        assert rc == 0

    def test_two_structures(self):
        rc, out, err = run([USALIGN, UBIQ, CRAMBIN])
        assert rc == 0
        assert "TM-score" in out

    def test_tabular_output(self):
        rc, out, err = run([USALIGN, UBIQ, CRAMBIN, "--outfmt", "2"])
        assert rc == 0
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        assert len(lines) >= 1
        fields = lines[0].split("\t")
        assert len(fields) >= 8

    def test_mm_align(self):
        """MM-align mode (--mm 1) for multi-chain alignment."""
        rc, out, err = run([USALIGN, UBIQ, CRAMBIN, "--mm", "1", "--outfmt", "2"])
        assert rc == 0

    def test_self_alignment(self):
        rc, out, err = run([USALIGN, CRAMBIN, CRAMBIN, "--outfmt", "2"])
        assert rc == 0
        lines = [l for l in out.strip().splitlines() if not l.startswith("#")]
        fields = lines[0].split("\t")
        tm1 = float(fields[2])
        assert tm1 > 0.99

    def test_mmcif_input(self):
        """Test mmCIF format input."""
        if not os.path.isfile(UBIQ_CIF):
            pytest.skip("1ubq.cif not available")
        rc, out, err = run([USALIGN, UBIQ_CIF, CRAMBIN, "--outfmt", "2"])
        assert rc == 0

    def test_missing_file(self):
        rc, out, err = run([USALIGN, "nonexistent.pdb", CRAMBIN])
        assert rc != 0


# =========================================================================
# ingest (ferritin-ingest)
# =========================================================================


class TestIngest:
    def test_help(self):
        rc, out, err = run([INGEST, "--help"])
        assert rc == 0
        assert "ingest" in out.lower() or "parquet" in out.lower() or "Usage" in out

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, CRAMBIN, "-o", outpath])
            assert rc == 0
            assert os.path.isfile(outpath)
            assert os.path.getsize(outpath) > 100

    def test_single_file_report(self):
        """Ingest should report structure count on stderr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, CRAMBIN, "-o", outpath])
            assert rc == 0
            combined = out + err
            assert "1 structures" in combined or "Done: 1" in combined

    def test_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, CRAMBIN, UBIQ, "-o", outpath])
            assert rc == 0
            assert os.path.isfile(outpath)

    def test_per_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rc, out, err = run([INGEST, CRAMBIN, UBIQ, "-o", tmpdir, "--per-structure"])
            assert rc == 0
            parquets = [f for f in os.listdir(tmpdir) if f.endswith(".parquet")]
            assert len(parquets) == 2

    def test_directory_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, TEST_PDBS, "-o", outpath, "-n", "3"])
            assert rc == 0
            assert os.path.isfile(outpath)

    def test_max_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, TEST_PDBS, "-o", outpath, "-n", "2"])
            assert rc == 0
            combined = out + err
            assert "2 structures" in combined or "Done: 2" in combined

    def test_nonexistent_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, "nonexistent_dir/", "-o", outpath])
            combined = out + err
            # Should either fail or report no input
            assert rc != 0 or "0 structures" in combined or "Done: 0" in combined or "No input" in combined

    def test_parquet_readable(self):
        """Verify output Parquet is readable by pyarrow."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "out.parquet")
            rc, out, err = run([INGEST, CRAMBIN, "-o", outpath])
            assert rc == 0

            table = pq.read_table(outpath)
            assert table.num_rows > 0
            assert "x" in table.column_names or "coords_x" in table.column_names or len(table.column_names) > 3
