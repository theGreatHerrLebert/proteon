"""Tests for the Agent Notes docstring convention.

These tests keep the boundary-layer guidance narrow and consistent:
only a small set of prefixes is allowed, and the README example should
follow the same convention as the code.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_API_ROOT = REPO_ROOT / "packages" / "ferritin" / "src" / "ferritin"
README = REPO_ROOT / "README.md"
SCAN_ROOTS = [
    REPO_ROOT / "packages" / "ferritin" / "src",
    REPO_ROOT / "ferritin-connector" / "src",
    REPO_ROOT / "ferritin-align" / "src",
    REPO_ROOT / "ferritin-io" / "src",
    REPO_ROOT / "ferritin-arrow" / "src",
]

ALLOWED_PREFIXES = {"WATCH", "PREFER", "COST", "INVARIANT"}
PREFIX_RE = re.compile(r"^\s*([A-Z_]+):", re.MULTILINE)


def _agent_note_prefixes(path: Path) -> set[str]:
    text = path.read_text()
    if "Agent Notes:" not in text:
        return set()
    return set(PREFIX_RE.findall(text))


class TestAgentNotesPolicy:
    def test_python_api_agent_notes_use_allowed_prefixes(self):
        offenders: list[str] = []

        for path in sorted(PYTHON_API_ROOT.glob("*.py")):
            prefixes = _agent_note_prefixes(path)
            invalid = sorted(p for p in prefixes if p not in ALLOWED_PREFIXES)
            if invalid:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(invalid)}")

        assert not offenders, "Unsupported Agent Notes prefixes found:\n" + "\n".join(offenders)

    def test_readme_agent_notes_example_uses_allowed_prefixes(self):
        prefixes = _agent_note_prefixes(README)
        invalid = sorted(p for p in prefixes if p not in ALLOWED_PREFIXES)
        assert not invalid, f"README contains unsupported Agent Notes prefixes: {', '.join(invalid)}"

    def test_readme_describes_actual_scope(self):
        text = README.read_text()
        assert "selected public boundary functions" in text
        assert "every public function" not in text

    def test_agent_notes_stay_in_public_python_boundary_layer(self):
        offenders: list[str] = []

        for root in SCAN_ROOTS:
            for path in sorted(root.rglob("*")):
                if path.suffix not in {".py", ".rs"}:
                    continue
                text = path.read_text()
                if "Agent Notes:" not in text:
                    continue
                if not path.is_relative_to(PYTHON_API_ROOT):
                    offenders.append(str(path.relative_to(REPO_ROOT)))

        assert not offenders, (
            "Agent Notes should stay in the public Python boundary layer:\n"
            + "\n".join(offenders)
        )
