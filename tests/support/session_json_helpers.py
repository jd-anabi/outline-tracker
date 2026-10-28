"""What the tests of core/session.py share: the SPEC 8.10 example, a strict JSON comparison, a JSON file, a video name.
They also share what docs/design/formats.md says about session.json version 2: its examples and its table.

Imported by name from tests/core/test_session.py, tests/core/test_session_file.py,
tests/core/test_session_video.py and tests/core/test_session_v1.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from layout_helpers import path as table_path


VIDEO_NAME = "groupB_2026-10-06_1325_main_tracker.mp4"


def spec_example() -> dict:
    """The session.json example of SPEC 8.10, parsed from docs/design/SPEC-v0.1.md."""
    text = table_path("spec").read_text(encoding="utf-8")
    section = text.split("### 8.10 session.json", 1)[1]
    return json.loads(section.split("```json", 1)[1].split("```", 1)[0])


def formats_section() -> str:
    """The section of docs/design/formats.md on session.json version 2, as text."""
    text = table_path("formats").read_text(encoding="utf-8")
    return text.split("## session.json, version 2", 1)[1].split("\n## ", 1)[0]


def formats_example(number: int, version: int) -> dict:
    """Example `number` of docs/design/formats.md (1: made in the window, 2: made by from-tracker) as the
    session.json of `version` (1 or 2), parsed."""
    example = formats_section().split(f"### Example {number}:", 1)[1].split("\n### ", 1)[0]
    block = example.split(f"#### Version {version}\n", 1)[1]
    return json.loads(block.split("```json", 1)[1].split("```", 1)[0])


@dataclass(frozen=True)
class FormatRow:
    """One row of the table of docs/design/formats.md. v1, v2: the key path in each version (dots, and
    `[]` for the items of a list), None where the table has a dash. reading, default, step: the other
    three cells as written."""

    v1: str | None
    v2: str | None
    reading: str
    default: str
    step: str

    def json_default(self):
        """The cell "Default" as a JSON value when it is one in backticks, else the `NotImplemented`
        marker (the words "below", "required", "left out" and the like)."""
        cell = self.default
        if cell.startswith("`") and cell.endswith("`"):
            return json.loads(cell.strip("`"))
        return NotImplemented


def formats_rows() -> list[FormatRow]:
    """Every row of the tables of the section "The table" of docs/design/formats.md, in their order."""
    tables = formats_section().split("### The table", 1)[1].split("\n### ", 1)[0]
    rows = []
    for line in tables.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or cells[0] in ("Version 1", "---"):
            continue
        keys = [None if cell == "-" else cell.strip("`") for cell in cells[:2]]
        rows.append(FormatRow(*keys, *cells[2:]))
    return rows


def key_paths(data: dict, leaves=frozenset(), prefix: str = "") -> set[str]:
    """Every key path of a parsed session file, written as the table of docs/design/formats.md writes it:
    dots, and `[]` for the items of a list. An object at a path in `leaves` is one key, and so is a list
    whose items are not objects."""
    paths = set()
    for key, value in data.items():
        path = prefix + key
        paths.add(path)
        if path in leaves:
            continue
        if isinstance(value, dict):
            paths |= key_paths(value, leaves, path + ".")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    paths |= key_paths(item, leaves, path + "[].")
    return paths


def assert_same_json(a, b, where="$"):
    """Equal as JSON, and strictly: the same keys in the same order, and 0 is not 0.0."""
    assert type(a) is type(b), f"{where}: {a!r} is not {b!r}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{where}: keys {list(a)} are not {list(b)}"
        for key in a:
            assert_same_json(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{where}: {a!r} is not {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_json(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} is not {b!r}"


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
