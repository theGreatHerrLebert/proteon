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
FASTA_TO_MMSEQS_DB = os.path.join(BIN, "fasta_to_mmseqs_db")
BUILD_KMI = os.path.join(BIN, "build_kmi")
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


# =========================================================================
# fasta_to_mmseqs_db — FASTA → MMseqs2-compatible DB
# =========================================================================


# Three short amino-acid sequences. The two build_kmi tests below consume
# this DB, so keep the alphabet in the standard 20 + no gaps / stops.
_TINY_FASTA = """\
>seq1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>seq2
MTSESATKHNSTNSLQATETTIRKSFISTASLLSKALKNVLNSLKAALELPVFYIR
>seq3
MKAIFVLNAQHDEAVDTHLAGKAALVENVTLKFDAAPLTDPTIAQLYKHRLVSFGDNKY
"""


def _write_tiny_fasta(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "tiny.fasta")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_TINY_FASTA)
    return path


@pytest.mark.skipif(
    not os.path.isfile(FASTA_TO_MMSEQS_DB),
    reason="fasta_to_mmseqs_db binary not built",
)
class TestFastaToMmseqsDb:
    def test_help(self):
        rc, out, err = run([FASTA_TO_MMSEQS_DB, "--help"])
        assert rc == 0
        combined = out + err
        assert "fasta" in combined.lower() or "MMseqs" in combined or "Usage" in combined

    def test_writes_three_db_files(self):
        """Happy path: FASTA in, three mmseqs-compatible files out.
        The three-file layout (prefix, prefix.index, prefix.dbtype) is
        what `DBReader` expects — asserting all three exist catches any
        regression that would silently break search DB consumption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(tmpdir)
            out_prefix = os.path.join(tmpdir, "tinydb")
            rc, out, err = run([FASTA_TO_MMSEQS_DB, fasta, out_prefix])
            assert rc == 0, f"stderr: {err}"
            # Three files, byte-compatible with `mmseqs createdb`.
            assert os.path.isfile(out_prefix)
            assert os.path.isfile(out_prefix + ".index")
            assert os.path.isfile(out_prefix + ".dbtype")
            # Log line reports record count on stderr; loose contains
            # check to tolerate format tweaks.
            combined = out + err
            assert "3" in combined and "record" in combined.lower()

    def test_max_records_truncates(self):
        """`--max-records N` stops after N sequences; the DB record
        count should match the truncation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(tmpdir)
            out_prefix = os.path.join(tmpdir, "db_trunc")
            rc, out, err = run([
                FASTA_TO_MMSEQS_DB, fasta, out_prefix, "--max-records", "2",
            ])
            assert rc == 0, f"stderr: {err}"
            # Two records instead of three. The mmseqs .index file has
            # one line per record, so its line count is the record
            # count.
            with open(out_prefix + ".index", "r", encoding="utf-8") as f:
                n_records = sum(1 for line in f if line.strip())
            assert n_records == 2

    def test_missing_fasta_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "does_not_exist.fasta")
            out_prefix = os.path.join(tmpdir, "out")
            rc, out, err = run([FASTA_TO_MMSEQS_DB, missing, out_prefix])
            assert rc != 0
            # Neither prefix nor index should have been created on error.
            assert not os.path.isfile(out_prefix)
            assert not os.path.isfile(out_prefix + ".index")


# =========================================================================
# build_kmi — MMseqs2 DB → .kmi external-memory k-mer index
# =========================================================================


@pytest.mark.skipif(
    not os.path.isfile(BUILD_KMI) or not os.path.isfile(FASTA_TO_MMSEQS_DB),
    reason="build_kmi / fasta_to_mmseqs_db binary not built",
)
class TestBuildKmi:
    def test_help(self):
        rc, out, err = run([BUILD_KMI, "--help"])
        assert rc == 0
        combined = out + err
        assert "kmi" in combined.lower() or "k-mer" in combined.lower() or "Usage" in combined

    def test_fasta_to_db_to_kmi_end_to_end(self):
        """Realistic flow: FASTA → DB (via fasta_to_mmseqs_db) → .kmi
        (via build_kmi). Exercises both CLIs as a single pipeline,
        mirroring how users consume them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(tmpdir)
            db_prefix = os.path.join(tmpdir, "db")
            kmi_path = os.path.join(tmpdir, "db.kmi")

            rc, out, err = run([FASTA_TO_MMSEQS_DB, fasta, db_prefix])
            assert rc == 0, f"fasta_to_mmseqs_db stderr: {err}"

            # Small k + small reduced alphabet keeps the offsets table
            # tiny on the smoke input.
            rc, out, err = run([
                BUILD_KMI, db_prefix, kmi_path, "--k", "3", "--reduce-to", "13",
            ])
            assert rc == 0, f"build_kmi stderr: {err}"
            assert os.path.isfile(kmi_path)
            assert os.path.getsize(kmi_path) > 0

    def test_no_reduce_uses_full_alphabet(self):
        """`--no-reduce` switches to the full 21-letter alphabet.
        The produced .kmi is larger than the reduced-alphabet version
        because the offsets table is 21^k instead of 13^k entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = _write_tiny_fasta(tmpdir)
            db_prefix = os.path.join(tmpdir, "db")
            rc, _, err = run([FASTA_TO_MMSEQS_DB, fasta, db_prefix])
            assert rc == 0, f"fasta_to_mmseqs_db stderr: {err}"

            kmi_reduced = os.path.join(tmpdir, "reduced.kmi")
            rc, _, _ = run([BUILD_KMI, db_prefix, kmi_reduced, "--k", "3"])
            assert rc == 0

            kmi_full = os.path.join(tmpdir, "full.kmi")
            rc, _, err = run([BUILD_KMI, db_prefix, kmi_full, "--k", "3", "--no-reduce"])
            assert rc == 0, f"build_kmi --no-reduce stderr: {err}"
            assert os.path.isfile(kmi_full)
            # Full alphabet is a larger offsets table; the .kmi file
            # grows accordingly. Assert strictly > so any future change
            # that silently makes --no-reduce a no-op would fail.
            assert os.path.getsize(kmi_full) > os.path.getsize(kmi_reduced)

    def test_missing_db_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "nonexistent_db")
            kmi_path = os.path.join(tmpdir, "out.kmi")
            rc, out, err = run([BUILD_KMI, missing, kmi_path])
            assert rc != 0
            assert not os.path.isfile(kmi_path)
