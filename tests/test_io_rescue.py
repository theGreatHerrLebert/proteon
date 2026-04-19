from pathlib import Path

import pytest

from proteon import io


class FakeBackend:
    def load(self, path: str):
        return self.load_pdb(path)

    def load_pdb(self, path: str):
        text = Path(path).read_text(encoding="utf-8")
        if "SEQADV" in text:
            raise OSError(
                "Failed to read x.pdb: InvalidatingError: Invalid data in field\n"
                "432 | SEQADV 1BGX L GB 437099 SER 31 DELETION\n"
                "The text presented is not of the right kind (isize)."
            )
        if any(
            line.startswith(("ATOM  ", "HETATM")) and line.rstrip().endswith("N0")
            for line in text.splitlines()
        ):
            raise OSError(
                "Failed to read x.pdb: InvalidatingError: Atom charge is not correct\n"
                "ATOM      1  N   MET W   1      -7.341  14.092 113.200  1.00 56.51           N0"
            )
        if any(line.startswith("SSBOND") for line in text.splitlines()):
            raise OSError(
                "Failed to read x.pdb: InvalidatingError: Could not find a bond partner\n"
                "SSBOND   1 MET A   35    MET A   35"
            )
        if any(line.startswith("DBREF1") for line in text.splitlines()) and not any(
            line.startswith("DBREF2") for line in text.splitlines()
        ):
            raise OSError("Failed to read x.pdb: StrictWarning: Solitary DBREF1 definition")
        if "UNMATCHED_ERROR" in text:
            raise OSError("Failed to read x.pdb: unexpected parse failure")
        return {"path": path, "text": text}

    def load_mmcif(self, path: str):
        return {"path": path, "text": Path(path).read_text(encoding="utf-8")}


@pytest.fixture(autouse=True)
def patch_io_backend(monkeypatch):
    monkeypatch.setattr(io, "_io", FakeBackend())
    monkeypatch.setattr(io.Structure, "from_py_ptr", staticmethod(lambda ptr: ptr))


def test_load_with_rescue_strips_seqadv_records(tmp_path):
    path = tmp_path / "rescued_seqadv.pdb"
    path.write_text(
        "HEADER    TEST\n"
        "SEQADV 9L05     A       UNP  P0DTC2    LEU    24 DELETION\n"
        "ATOM      1  N   GLY A   1      11.104  13.207   9.125  1.00 10.00           N\n",
        encoding="utf-8",
    )

    result = io.load_with_rescue(path)

    assert result.rescued is True
    assert result.rescue_bucket is not None
    assert result.rescue_bucket.code == "seqadv_invalid_field"
    assert result.rescue_steps == ("drop_seqadv",)
    assert "SEQADV" not in result.structure["text"]


def test_load_with_rescue_clears_zero_charge_suffix(tmp_path):
    path = tmp_path / "rescued_charge.pdb"
    path.write_text(
        "ATOM      1  N   MET W   1      -7.341  14.092 113.200  1.00 56.51           N0\n",
        encoding="utf-8",
    )

    result = io.load_with_rescue(path)

    assert result.rescued is True
    assert result.rescue_bucket is not None
    assert result.rescue_bucket.code == "atom_charge_suffix"
    repaired_line = result.structure["text"].splitlines()[0].ljust(80)
    assert repaired_line[78:80] == "  "


def test_load_with_rescue_drops_ssbond_records(tmp_path):
    path = tmp_path / "rescued_ssbond.pdb"
    path.write_text(
        "SSBOND   1 MET A   35    MET A   35\n"
        "ATOM      1  N   GLY A   1      11.104  13.207   9.125  1.00 10.00           N\n",
        encoding="utf-8",
    )

    result = io.load_with_rescue(path)

    assert result.rescued is True
    assert result.rescue_bucket is not None
    assert result.rescue_bucket.code == "ssbond_missing_partner"
    assert "SSBOND" not in result.structure["text"]


def test_load_with_rescue_reraises_unmatched_errors(tmp_path):
    path = tmp_path / "broken.pdb"
    path.write_text("UNMATCHED_ERROR\n", encoding="utf-8")

    with pytest.raises(OSError, match="unexpected parse failure"):
        io.load_with_rescue(path)


def test_batch_load_tolerant_with_rescue_returns_indexed_results(tmp_path):
    good = tmp_path / "good.pdb"
    rescued = tmp_path / "rescued.pdb"
    bad = tmp_path / "bad.pdb"
    good.write_text(
        "ATOM      1  N   GLY A   1      11.104  13.207   9.125  1.00 10.00           N\n",
        encoding="utf-8",
    )
    rescued.write_text(
        "DBREF1 8A14 A    1    138  UNP    A0A0D1C5E9\n"
        "ATOM      1  N   GLY A   1      11.104  13.207   9.125  1.00 10.00           N\n",
        encoding="utf-8",
    )
    bad.write_text("UNMATCHED_ERROR\n", encoding="utf-8")

    results = io.batch_load_tolerant_with_rescue([good, rescued, bad])

    assert [index for index, _ in results] == [0, 1]
    assert results[0][1].rescued is False
    assert results[1][1].rescued is True
    assert results[1][1].rescue_bucket is not None
    assert results[1][1].rescue_bucket.code == "solitary_dbref1_definition"


def test_load_with_rescue_can_apply_multiple_passes(tmp_path, monkeypatch):
    class MultiPassBackend(FakeBackend):
        def load_pdb(self, path: str):
            text = Path(path).read_text(encoding="utf-8")
            if any(line.startswith("SEQRES") for line in text.splitlines()):
                raise OSError("Failed to read x.pdb: LooseWarning: SEQRES residue total invalid")
            if "SEQADV" in text:
                raise OSError(
                    "Failed to read x.pdb: InvalidatingError: Invalid data in field\n"
                    "432 | SEQADV 1BGX L GB 437099 SER 31 DELETION\n"
                    "The text presented is not of the right kind (isize)."
                )
            return {"path": path, "text": text}

    monkeypatch.setattr(io, "_io", MultiPassBackend())

    path = tmp_path / "rescued_multipass.pdb"
    path.write_text(
        "SEQRES   1 A    2  GLY SER\n"
        "SEQADV 9L05     A       UNP  P0DTC2    LEU    24 DELETION\n"
        "ATOM      1  N   GLY A   1      11.104  13.207   9.125  1.00 10.00           N\n",
        encoding="utf-8",
    )

    result = io.load_with_rescue(path)

    assert result.rescued is True
    assert result.rescue_bucket is not None
    assert result.rescue_bucket.code == "seqres_total_invalid"
    assert result.rescue_steps == ("drop_seqres_dbref_metadata", "drop_seqadv")
    assert "SEQRES" not in result.structure["text"]
    assert "SEQADV" not in result.structure["text"]
