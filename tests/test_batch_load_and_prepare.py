"""batch_load_and_prepare — one verdict per input path across load + prepare.

The batch counterpart of load_and_prepare. Its contract is that the result list
is the SAME LENGTH AND ORDER as the input paths, with the two failure modes
(file never loaded vs structure did not prepare cleanly) collapsed into one
`res.ready` decision so archive-scale ingestion is a single branch.
"""

import os

import pytest

import proteon
from proteon import LoadPrepResult, PrepStatus

PDBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-pdbs")
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _pdb(name):
    return os.path.join(PDBS, name)


class TestContract:
    def test_all_good_are_ready(self):
        paths = [_pdb("1crn.pdb"), _pdb("1crn.pdb")]
        results = proteon.batch_load_and_prepare(paths, minimize=False)
        assert len(results) == len(paths)
        assert all(isinstance(r, LoadPrepResult) for r in results)
        assert all(r.ready for r in results), [r.reason for r in results]
        assert all(r.loaded for r in results)
        assert all(r.structure is not None for r in results)
        assert all(r.report is not None for r in results)

    def test_length_and_order_preserved_with_failures(self, tmp_path):
        junk = tmp_path / "garbage.pdb"
        junk.write_text("this is not a structure file\n")
        paths = [_pdb("1crn.pdb"), str(junk), _pdb("1crn.pdb")]
        results = proteon.batch_load_and_prepare(paths, minimize=False)
        # one record per input, in order
        assert len(results) == 3
        assert [r.path for r in results] == [str(p) for p in paths]
        # the junk file in the middle is a load failure, not dropped
        assert results[0].ready is True
        assert results[2].ready is True
        assert results[1].loaded is False
        assert results[1].ready is False
        assert results[1].structure is None
        assert results[1].report is None

    def test_load_failure_reason(self, tmp_path):
        junk = tmp_path / "garbage.cif"
        junk.write_text("nonsense\n")
        results = proteon.batch_load_and_prepare([str(junk)], minimize=False)
        r = results[0]
        assert r.loaded is False
        assert r.ready is False
        assert r.reason.startswith("load failed:")

    def test_empty_input(self):
        assert proteon.batch_load_and_prepare([]) == []

    def test_all_failures_no_prepare_crash(self, tmp_path):
        a = tmp_path / "a.pdb"
        b = tmp_path / "b.pdb"
        a.write_text("junk\n")
        b.write_text("junk\n")
        results = proteon.batch_load_and_prepare([str(a), str(b)])
        assert len(results) == 2
        assert all(not r.ready and not r.loaded for r in results)


class TestPrepareForwarding:
    def test_minimize_false_forwarded(self):
        # minimize=False must reach batch_prepare: a ready structure with the
        # minimizer not run (no convergence claimed).
        results = proteon.batch_load_and_prepare([_pdb("1crn.pdb")], minimize=False)
        r = results[0]
        assert r.ready is True
        assert r.report.minimized is False

    def test_reason_forwards_prep_report(self):
        # A loaded-but-not-ready structure forwards PrepReport.reason verbatim.
        # Construct the record directly to exercise the delegation without
        # depending on a specific not-ready fixture.
        from proteon import PrepReport

        rep = PrepReport(skipped_no_protein=True, n_unassigned_atoms=400)
        res = LoadPrepResult(path="x.pdb", structure=object(), report=rep)
        assert res.loaded is True
        assert res.ready is False
        assert res.reason == rep.reason
        assert rep.status == PrepStatus.NOT_PROTEIN
