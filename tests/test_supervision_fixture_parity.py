"""Guards against schema/fixture drift in supervision parquet tests.

When ``TENSOR_FIELDS`` in ``supervision_export.py`` grows a new field, every
fixture that builds a ``StructureSupervisionExample`` for round-trip tests
must populate it with an array of the right shape and dtype. Otherwise the
parquet writer silently casts ``None`` to NaN, the flat values array stops
being a multiple of the inner-shape dims, and every streaming/training test
downstream fails with a confusing ``ArrowInvalid`` error.

Commit ``c4e8574`` is the symptom that motivated this test: ``atom14_alt_gt_*``
were added to ``TENSOR_FIELDS`` but neither test fixture was updated, and
``main`` stayed red for six days before anyone noticed. This test is the
structural fix — it catches the *category* of drift, not just that one
instance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from proteon.supervision_export import TENSOR_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


# Every fixture in the suite that builds a ``StructureSupervisionExample``
# for round-trip tests. Adding a new fixture here makes it subject to the
# same drift checks; forgetting to add one will only silence the check for
# that fixture, never break the export path.
SUPERVISION_FIXTURES = [
    (
        "test_supervision_parquet_streaming",
        "_fake_supervision",
        {"record_id": "fixture-r", "L": 7, "seed": 1},
    ),
    (
        "test_training_parquet",
        "_fake_struc",
        {"record_id": "fixture-r", "chain_id": "A", "L": 7, "seed": 1},
    ),
]


def _load_fixture(module_name: str, fixture_name: str):
    """Load a fixture function from a sibling test module by file path.

    Going through ``importlib.util`` (rather than ``from … import …``)
    keeps the parity test independent of pytest collection order and of
    whether ``tests/`` happens to be on ``sys.path``.
    """
    path = TESTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, fixture_name)


@pytest.mark.parametrize(
    ("module_name", "fixture_name", "kwargs"),
    SUPERVISION_FIXTURES,
    ids=[f"{m}::{n}" for m, n, _ in SUPERVISION_FIXTURES],
)
def test_fixture_populates_every_tensor_field(module_name, fixture_name, kwargs):
    factory = _load_fixture(module_name, fixture_name)
    example = factory(**kwargs)
    expected_length = int(example.length)

    issues: list[str] = []
    for name, inner_shape, dtype, attr in TENSOR_FIELDS:
        value = getattr(example, attr, None)
        if value is None:
            issues.append(
                f"  {name}: attr `{attr}` is None — fixture must populate every TENSOR_FIELDS entry"
            )
            continue

        arr = np.asarray(value)
        expected_shape = (expected_length,) + tuple(inner_shape)
        if arr.shape != expected_shape:
            issues.append(
                f"  {name}: shape {arr.shape} does not match expected {expected_shape} "
                f"(L={expected_length}, inner_shape={tuple(inner_shape)})"
            )
            continue

        try:
            np.asarray(arr, dtype=dtype)
        except (TypeError, ValueError) as err:
            issues.append(
                f"  {name}: dtype {arr.dtype} not castable to {dtype.__name__}: {err}"
            )

    assert not issues, (
        f"{module_name}.{fixture_name} drifted from TENSOR_FIELDS — every entry "
        f"must be populated with the right shape/dtype:\n" + "\n".join(issues)
    )
