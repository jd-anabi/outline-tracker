"""What the tests of core/session.py share: the SPEC 8.10 example, a strict JSON comparison, a JSON file, a video name.

Imported by name from tests/core/test_session.py, tests/core/test_session_file.py and
tests/core/test_session_video.py.
"""

import json
from pathlib import Path

from layout_helpers import path as table_path


VIDEO_NAME = "groupB_2026-10-06_1325_main_tracker.mp4"


def spec_example() -> dict:
    """The session.json example of SPEC 8.10, parsed from docs/design/SPEC-v0.1.md."""
    text = table_path("spec").read_text(encoding="utf-8")
    section = text.split("### 8.10 session.json", 1)[1]
    return json.loads(section.split("```json", 1)[1].split("```", 1)[0])


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
