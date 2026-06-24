import os

import pytest
import proteon

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "data", "stable_api_snapshot.txt")


def _load_stable_snapshot():
    names = set()
    with open(_SNAPSHOT_PATH) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.add(stripped)
    return frozenset(names)


class TestStabilityTiers:
    """The stable/experimental split is the public contract (see STABILITY.md).

    The STABLE set is frozen against a checked-in snapshot so that growing it is
    a deliberate, reviewed act — not an import side effect. EXPERIMENTAL is free
    to churn but must stay disjoint, so a frontier symbol can never silently
    become 'stable'.
    """

    def test_tiers_exist_and_are_frozensets(self):
        assert isinstance(proteon.__stable__, frozenset)
        assert isinstance(proteon.__experimental__, frozenset)
        assert proteon.__stable__
        assert proteon.__experimental__

    def test_tiers_are_disjoint(self):
        overlap = proteon.__stable__ & proteon.__experimental__
        assert not overlap, f"symbols in BOTH tiers: {sorted(overlap)}"

    def test_tiers_partition_all_exactly(self):
        # No symbol may be exported without being tiered, and no tiered symbol
        # may be missing from __all__.
        exported = set(proteon.__all__) - {"__version__"}
        tiered = proteon.__stable__ | proteon.__experimental__
        assert exported == tiered, (
            f"untiered exports: {sorted(exported - tiered)}; "
            f"tiered-but-not-exported: {sorted(tiered - exported)}"
        )

    def test_stable_surface_matches_frozen_snapshot(self):
        # THE GUARD. If this fails, the stable (promised) API changed. That is a
        # contract change: update tests/data/stable_api_snapshot.txt in the same
        # commit and note it in the changelog. See STABILITY.md.
        snapshot = _load_stable_snapshot()
        added = sorted(proteon.__stable__ - snapshot)
        removed = sorted(snapshot - proteon.__stable__)
        assert not added and not removed, (
            f"STABLE API drift — added: {added}; removed: {removed}. "
            "This is a public-contract change; update the snapshot deliberately."
        )

    def test_every_stable_name_resolves(self):
        missing = [n for n in proteon.__stable__ if not hasattr(proteon, n)]
        assert not missing

    def test_known_frontiers_are_experimental_not_stable(self):
        # Guard against accidentally promising research frontiers. These must
        # NEVER be in the stable tier.
        for name in ("prepare", "dock", "run_md", "search", "born_energy",
                     "build_structure_supervision_example", "to_parquet"):
            assert name in proteon.__experimental__, f"{name} should be experimental"
            assert name not in proteon.__stable__

    def test_oracle_validated_core_is_stable(self):
        # The pure-compute core we DO promise.
        for name in ("tm_align", "total_sasa", "dssp", "compute_energy",
                     "minimize_structure", "kabsch_superpose", "load", "Structure"):
            assert name in proteon.__stable__, f"{name} should be stable"


class TestExperimentalNamespace:
    """Experimental symbols are reachable via the canonical proteon.experimental.*
    namespace (PR1: flat top-level access is also retained, non-breaking)."""

    def test_experimental_submodule_importable(self):
        import proteon.experimental as ex

        assert ex is proteon.experimental

    def test_experimental_namespace_exposes_every_experimental_symbol(self):
        ex = proteon.experimental
        missing = [n for n in proteon.__experimental__ if not hasattr(ex, n)]
        assert not missing

    def test_experimental_namespace_excludes_stable_symbols(self):
        ex = proteon.experimental
        leaked = [n for n in proteon.__stable__ if hasattr(ex, n)]
        assert not leaked, f"stable symbols leaked into experimental: {leaked}"

    def test_flat_access_still_works_pr1_non_breaking(self):
        # PR1 is additive: experimental names remain bound on the top level.
        assert proteon.prepare is proteon.experimental.prepare


class TestTopLevelExports:
    def test___all___exists_and_is_unique(self):
        assert isinstance(proteon.__all__, tuple)
        assert proteon.__all__
        assert len(proteon.__all__) == len(set(proteon.__all__))

    def test___all___entries_resolve(self):
        missing = [name for name in proteon.__all__ if not hasattr(proteon, name)]
        assert not missing

    def test_star_import_exposes_core_symbols(self):
        ns = {}
        exec("from proteon import *", ns)

        expected = {
            "__version__",
            "load",
            "save",
            "Structure",
            "tm_align",
            "compute_energy",
            "prepare",
            "build_search_db",
            "build_sequence_example",
            "batch_build_structure_supervision_examples",
        }
        assert expected <= set(ns)

    def test_build_sequence_dataset_is_top_level_canonical_entry_point(self):
        """v0.3.0 Phase A reversed an earlier v0.2.1-era decision to keep
        `build_sequence_dataset` submodule-only. Per codex review on
        TO_V030_TRAINING_CORPUS_FACTORY.md, the MSA-wired sequence-release
        builder is the canonical "structures + MSA -> sequence release"
        path and belongs in the top-level surface alongside
        `build_structure_supervision_dataset_from_prepared`. The earlier
        assertion that it must NOT be exposed is replaced by the
        opposite — it MUST be exposed, by name, in `proteon.__all__`.
        """
        assert "build_sequence_dataset" in proteon.__all__
        assert callable(proteon.build_sequence_dataset)
