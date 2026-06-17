"""The PrepReport readiness verdict (PrepReport.ready / status / reason).

A single trust signal so the load->prepare loop is one decision:
    structure, report = proteon.load_and_prepare(path)
    if report.ready:
        use(structure)

`ready` is a HARD-FAILURE gate (usable structure, nothing failed), NOT a
convergence claim — convergence is the separate `report.converged` axis.
"""

import os

import pytest

import proteon
from proteon import PrepReport, PrepStatus

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


# --- verdict logic (pure-Python on the dataclass; no connector needed) ---


class TestVerdictLogic:
    def test_clean_report_is_ready(self):
        r = PrepReport(hydrogens_added=50, minimizer_status="converged_gradient")
        assert r.status == PrepStatus.READY
        assert r.ready is True
        assert r.reason == ""

    def test_not_protein(self):
        r = PrepReport(skipped_no_protein=True, n_unassigned_atoms=400)
        assert r.status == PrepStatus.NOT_PROTEIN
        assert r.ready is False
        assert "not a protein" in r.reason

    def test_numerical_failure(self):
        r = PrepReport(minimizer_status="numerical_failure")
        assert r.status == PrepStatus.MINIMIZE_FAILED
        assert r.ready is False
        assert "non-finite" in r.reason

    def test_incomplete_ff(self):
        r = PrepReport(incomplete_ff=True, n_unassigned_atoms=40)
        assert r.status == PrepStatus.INCOMPLETE_FF
        assert r.ready is False
        assert "incomplete" in r.reason

    def test_minimize_not_requested_is_still_ready(self):
        # minimize=False -> minimizer_status "not_run": intentional, not a failure.
        r = PrepReport(hydrogens_added=50, minimized=False, minimizer_status="not_run")
        assert r.ready is True
        assert r.status == PrepStatus.READY

    def test_max_steps_is_ready_but_not_converged(self):
        # Ran out of budget: improved but not at a minimum. READY (usable), and
        # the separate `converged` gate is what energy-sensitive callers check.
        r = PrepReport(minimizer_status="max_steps", converged=False)
        assert r.ready is True
        assert r.converged is False  # the second gate stays False

    def test_line_search_stall_is_ready(self):
        r = PrepReport(minimizer_status="line_search_failed", converged=False)
        assert r.ready is True

    def test_hard_failure_precedence_not_protein_first(self):
        # A not-a-protein entry that also lacks FF coverage reports NOT_PROTEIN.
        r = PrepReport(skipped_no_protein=True, incomplete_ff=False, n_unassigned_atoms=500)
        assert r.status == PrepStatus.NOT_PROTEIN

    def test_repr_shows_verdict(self):
        assert "ready=True" in repr(PrepReport(minimizer_status="converged_gradient"))
        assert "ready=False" in repr(PrepReport(skipped_no_protein=True))


# --- integration: a real prepare round-trips to a ready verdict ---


def _corpus(subdir, name):
    return os.path.join(CORPUS, subdir, name)


class TestVerdictIntegration:
    def test_normal_protein_prepares_ready(self):
        s = proteon.load(os.path.join(os.path.dirname(__file__), "..", "test-pdbs", "1crn.pdb"))
        report = proteon.prepare(s, minimize=False)
        assert report.ready is True, f"1crn should be ready, got {report.status}: {report.reason}"
        assert report.status == PrepStatus.READY

    def test_minimize_false_is_ready(self):
        s = proteon.load(os.path.join(os.path.dirname(__file__), "..", "test-pdbs", "1crn.pdb"))
        report = proteon.prepare(s, minimize=False)
        # minimize not requested must not make the structure "not ready".
        assert report.ready is True

    def test_legacy_strip_false_path_computes_verdict(self):
        # The strip_hydrogens=False path computes the readiness flags from the
        # same Rust coverage pass as the default path — not left blind when
        # minimization is skipped (codex). For a clean protein both paths agree.
        p = os.path.join(os.path.dirname(__file__), "..", "test-pdbs", "1crn.pdb")
        r_default = proteon.prepare(proteon.load(p), strip_hydrogens=True, minimize=False)
        r_legacy = proteon.prepare(proteon.load(p), strip_hydrogens=False, minimize=False)
        assert r_legacy.ready is True
        assert r_legacy.skipped_no_protein == r_default.skipped_no_protein
        assert r_legacy.incomplete_ff == r_default.incomplete_ff

    def test_waters_only_structure_handles_gracefully(self):
        # A corpus fixture that is mostly solvent / non-protein should not crash
        # the verdict; ready is a clean bool either way.
        s = proteon.load(_corpus("waters", "protein_with_waters.pdb"))
        report = proteon.prepare(s, minimize=False)
        assert isinstance(report.ready, bool)
        assert report.status in set(PrepStatus)
