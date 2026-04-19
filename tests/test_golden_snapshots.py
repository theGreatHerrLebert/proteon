"""Frozen numerical snapshot regression canary.

The premise
-----------
For a small set of (structure, force_field) pairs we record the
current value of every energy component to 6 decimal places in a
JSON file under tests/fixtures/golden/. Any code change that alters
a recorded value by more than TOL_KJ_MOL fails this test and MUST
be accompanied by an explicit snapshot update via the env var
``PROTEON_UPDATE_SNAPSHOTS=1``.

Why
---
Invariant tests catch structural bugs (sign, finiteness, accounting),
oracle tests catch parameter bugs (values diverge from BALL), and
cross-path parity catches divergence between code paths. None of
those catches *unintentional drift*: an innocent-looking refactor
that changes the numerical output by ~1 kJ/mol is invisible to all
of them. Frozen snapshots close that gap — they're the numerical
equivalent of a diff-friendly golden-output test.

Tolerance
---------
TOL_KJ_MOL = 1e-4 kJ/mol (absolute) or 1e-9 (relative), whichever
is looser. This is strict enough to catch real drift (a 1 kJ/mol
delta on a 1000 kJ/mol bond_stretch is O(10³) above threshold) and
loose enough to tolerate the pre-existing 1 ULP FP reordering on
bond_stretch in release builds.

Updating snapshots
------------------
On purpose:
  PROTEON_UPDATE_SNAPSHOTS=1 pytest tests/test_golden_snapshots.py

Never commit a snapshot update without a matching explanation in
the commit message. The point of this test is that changes to these
numbers are intentional and auditable.
"""

from __future__ import annotations

import json
import os
import math

import pytest

import proteon

from conftest import ENERGY_COMPONENTS, STRUCTURES


GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "golden")

#: Force fields to snapshot. Keep this small — each snapshot is a
#: maintenance tax, so only the two production FFs.
SNAPSHOT_FFS = ("amber96", "charmm19_eef1")

#: 1e-4 kJ/mol absolute floor; 1e-9 relative. A ~1 kJ/mol drift on
#: a 1000 kJ/mol component is three orders of magnitude above this,
#: which is what we want to catch. Well below the "irrelevant FP
#: reordering" floor of ~1e-10 in release builds.
TOL_ABS_KJ_MOL = 1e-4
TOL_REL = 1e-9


def _snapshot_path(pdb: str, ff: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{pdb}_{ff}.json")


def _compute_snapshot(pdb_name: str, absolute_path: str, ff: str) -> dict:
    """Compute the canonical snapshot for a (pdb, ff) pair.

    Uses the default code path (nbl_threshold=None) because cross-path
    parity is tested separately — here we record ONE canonical value
    per (pdb, ff). If cross-path parity breaks, the parity test will
    flag it directly; this file only needs to pin one number.
    """
    s = proteon.load(absolute_path)
    e = proteon.compute_energy(s, ff=ff, units="kJ/mol")
    return {
        "pdb": pdb_name,
        "ff": ff,
        "units": "kJ/mol",
        "components": {k: float(e[k]) for k in ENERGY_COMPONENTS},
        "total": float(e["total"]),
    }


@pytest.fixture(scope="session", autouse=False)
def update_snapshots():
    return os.environ.get("PROTEON_UPDATE_SNAPSHOTS", "0") == "1"


@pytest.mark.parametrize(
    "spec",
    STRUCTURES,
    ids=[s.name for s in STRUCTURES],
)
@pytest.mark.parametrize("ff", SNAPSHOT_FFS)
def test_snapshot_matches(spec, ff, update_snapshots):
    path = _snapshot_path(spec.name, ff)
    actual = _compute_snapshot(spec.name, spec.absolute_path, ff)

    if update_snapshots:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")
        pytest.skip(f"snapshot written: {path}")

    if not os.path.exists(path):
        pytest.fail(
            f"no snapshot at {path}. Run with PROTEON_UPDATE_SNAPSHOTS=1 "
            f"to create it."
        )

    with open(path) as f:
        expected = json.load(f)

    # Schema check first — catches a component rename or a new
    # component being added without a corresponding snapshot update.
    assert set(actual["components"].keys()) == set(expected["components"].keys()), (
        f"{spec.name}/{ff}: component set changed. Actual: "
        f"{sorted(actual['components'])}, expected: "
        f"{sorted(expected['components'])}. Run with "
        f"PROTEON_UPDATE_SNAPSHOTS=1 to update."
    )

    # Value check — each component plus the total.
    drifts = []
    for comp in list(expected["components"].keys()) + ["total"]:
        if comp == "total":
            expected_val = expected["total"]
            actual_val = actual["total"]
        else:
            expected_val = expected["components"][comp]
            actual_val = actual["components"][comp]

        if not (math.isfinite(expected_val) and math.isfinite(actual_val)):
            continue
        diff = abs(actual_val - expected_val)
        tol = max(TOL_ABS_KJ_MOL, TOL_REL * abs(expected_val))
        if diff > tol:
            drifts.append((comp, expected_val, actual_val, diff, tol))

    if drifts:
        lines = [
            f"{spec.name}/{ff}: snapshot drift detected. If intentional, "
            "rerun with PROTEON_UPDATE_SNAPSHOTS=1 and commit the updated JSON."
        ]
        for comp, expected_val, actual_val, diff, tol in drifts:
            lines.append(
                f"  {comp:<18} expected={expected_val:+.6f} "
                f"actual={actual_val:+.6f} diff={diff:.3e} tol={tol:.3e}"
            )
        pytest.fail("\n".join(lines))
