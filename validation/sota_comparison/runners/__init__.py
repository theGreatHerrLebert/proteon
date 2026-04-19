"""Runner registry for the SOTA comparison harness.

Each `runners/<op>.py` module exports one function per implementation
(e.g. `def proteon(path)`, `def freesasa(path)`). They register themselves
in the global `OPS` dict via the `register()` decorator from `_base`.

Importing a runner module registers all of its implementations as a side
effect. The driver imports every runner module up front so the registry is
populated before walking it.
"""

from . import _base
from ._base import BATCH_RUNNERS, IMPORT_FAILURES, OPS, RunnerResult, register, register_batch

# Side-effect imports: each module registers its implementations on import.
# Wrap in try/except so a missing optional dependency (e.g. freesasa not yet
# installed in the venv) doesn't break the whole registry.
def _safe_import(modname: str) -> None:
    try:
        __import__(f"validation.sota_comparison.runners.{modname}", fromlist=["_"])
    except ImportError as e:
        # Record the failure so the driver can report it without crashing.
        _base.IMPORT_FAILURES[modname] = str(e)


for _mod in ("sasa", "energy", "energy_charmm"):
    _safe_import(_mod)

__all__ = [
    "OPS",
    "BATCH_RUNNERS",
    "RunnerResult",
    "register",
    "register_batch",
    "IMPORT_FAILURES",
]
