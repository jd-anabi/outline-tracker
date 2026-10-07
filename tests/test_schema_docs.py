"""Tests of outline_tracker/schema_docs.py: the text builders of README.txt and docs/OUTPUTS.md (SPEC 8.11, 16).

The builders used to live in schema.py. They moved out (task B6a) because schema.py had grown to 591
lines and SPEC 12 says to split past about 400. The first group of tests pins what must not change in
the move; the later groups test the wording and the Markdown contract. Expected values are typed from
the specification or computed from the schema tables, never copied from the output under test.
"""

import hashlib
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from outline_tracker import schema

REPO = Path(__file__).resolve().parents[1]

# SHA-256 of the UTF-8 bytes of readme_text(), computed with
#   uv run python -c "import hashlib; from outline_tracker import schema; \
#       print(hashlib.sha256(schema.readme_text().encode('utf-8')).hexdigest())"
# Commit 1 of task B6a moved the builders out of schema.py without changing a character. Its pinned value
# was computed from the unmodified schema.py of commit e49a088, before the move:
#   9cede8f4c9ba9cc98f4ee3ebbdb1907aa992700e9bde43c2013131705109b57b   (13751 characters, 249 lines)
# Commit 2 changed the wording of four places (the intro, Units, Rows and the core_frac entry, see the
# tests below). The value pinned now was computed after that change, and the diff between the two texts was
# read: those four places and nothing else differ (14468 characters, 258 lines). Any later change of the
# README text has to update this value on purpose.
README_SHA256 = "154a055faef113bdedafa9ed9e77b13f6a66f25a610170db164ade19795ae7ae"

# Every public name that schema.py had before the move (dir() of the unmodified module, without the
# imported helpers). Other modules import these by name, so each must still be importable.
PUBLIC_NAMES_OF_SCHEMA = [
    "ArrayKey", "CSV_ENCODING", "CSV_NEWLINE", "Column", "FILES", "FLAGS", "FLAG_INFO", "FLAG_SEPARATOR",
    "FileSpec", "FlagInfo", "OUTLINES_KEYS", "OUTLINES_META_FIELDS", "OUTLINES_META_KEY", "OUTLINES_NPZ",
    "OUTLINE_POINTS", "OVERLAY_MP4", "POSITIONS", "POSITIONS_CSV", "POSITION_FLAGS", "PROBES", "PROBES_CSV",
    "RADIAL", "RADIAL_ANGLES", "RADIAL_CSV", "RADIAL_STEP_DEG", "README_TXT", "RESULTS_KEYS", "RESULTS_NPZ",
    "RESULTS_VERSION", "RESULTS_VERSION_KEY", "RUN_LOG", "SESSION_JSON", "SESSION_SCHEMA_VERSION", "SHAPES",
    "SHAPES_CSV", "SHAPE_FLAGS", "STORED_OUTLINE_POINTS", "TRACKER_COLUMNS", "TRACKER_FILE", "csv_text",
    "format_row", "header_line", "join_flags", "npz_key", "readme_text", "split_npz_key",
]

LIGHT_CHECK = (
    "import sys; {imports}; "
    "bad = sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph', 'pandas') if m in sys.modules); "
    "print(bad)"
)


@pytest.fixture(scope="module")
def schema_docs():
    """The module under test, imported here so that a missing module fails the tests that need it."""
    return importlib.import_module("outline_tracker.schema_docs")


def python(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter (pytest-qt has already imported PySide6 into this one)."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=60)


# ---------------------------------------------------------------------------------------------
# The move: same text, same names, smaller modules

def test_readme_text_is_the_pinned_text():
    text = schema.readme_text()  # through the old import path, as every existing caller uses it
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == README_SHA256


def test_readme_text_from_schema_docs_is_the_pinned_text(schema_docs):
    assert hashlib.sha256(schema_docs.readme_text().encode("utf-8")).hexdigest() == README_SHA256


def test_schema_re_exports_readme_text(schema_docs):
    from outline_tracker.schema import readme_text

    assert readme_text is schema_docs.readme_text
    assert schema.readme_text is schema_docs.readme_text


@pytest.mark.parametrize("name", PUBLIC_NAMES_OF_SCHEMA)
def test_every_public_name_of_schema_is_still_importable(name):
    namespace: dict = {}
    exec(f"from outline_tracker.schema import {name}", namespace)  # noqa: S102
    assert namespace[name] is getattr(schema, name)


def test_schema_has_no_name_that_does_not_exist():
    with pytest.raises(AttributeError, match="no_such_name"):
        schema.no_such_name  # noqa: B018


def test_the_two_modules_are_each_under_420_lines(schema_docs):
    for module in (schema, schema_docs):
        lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
        assert len(lines) < 420, f"{Path(module.__file__).name} has {len(lines)} lines (SPEC 12: about 400)"


@pytest.mark.parametrize(
    "imports",
    [
        "import outline_tracker.schema",
        "import outline_tracker.schema_docs",
        "import outline_tracker.schema_docs, outline_tracker.schema",
        "from outline_tracker.schema import readme_text",
    ],
)
def test_importing_either_module_loads_neither_pandas_torch_nor_qt(imports):
    done = python(LIGHT_CHECK.format(imports=imports))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "imports",
    [
        "import outline_tracker.schema_docs as d; import outline_tracker.schema as s",
        "import outline_tracker.schema as s; import outline_tracker.schema_docs as d",
    ],
)
def test_import_order_does_not_matter(imports):
    """schema_docs reads the tables of schema, and schema hands out readme_text from schema_docs:
    neither import order may fail on the circle."""
    done = python(f"{imports}; assert s.readme_text is d.readme_text; print('ok')")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# ---------------------------------------------------------------------------------------------
# Wording slips that the review of A05 noted in the README text (fixed in the second commit of B6a)

NAN = float("nan")


@pytest.fixture(scope="module")
def readme(schema_docs) -> str:
    return schema_docs.readme_text()


def normalized(text: str) -> str:
    """Whitespace collapsed to single spaces, so a wrapped paragraph can be searched."""
    return " ".join(text.split())


def convention(readme: str, label: str) -> str:
    """The item of the CONVENTIONS section that starts with `label:`, wrapped lines joined."""
    lines = readme.splitlines()
    start = lines.index("== CONVENTIONS ==") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("== ") and i > start)
    items: list[str] = []
    for line in lines[start:end]:
        if line.startswith("  ") and not line.startswith("    "):  # an item starts at two spaces
            items.append(line.strip())
        elif line.strip() and items:
            items[-1] += " " + line.strip()
    [item] = [i for i in items if i.startswith(f"{label}:")]
    return item


def has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) is not None


def lost_row(columns: list[schema.Column]) -> dict:
    """A lost frame, built by hand as SPEC 8.2 and 8.4 describe it: floats empty (NaN), integers 0."""
    row: dict = {c.name: NAN for c in columns if c.dtype == "float"}
    row.update({c.name: 0 for c in columns if c.dtype == "int"})
    row.update(track_id="B", frame=40, t_s=0.1666667, mode="coarse", flags="LOST")
    return row


def test_the_readme_says_that_a_lost_row_keeps_frame_and_t_s(readme):
    """The numbers that a lost row keeps are found from the real cells, not typed: format_row writes the
    row and the cells that are not empty are the kept ones. The Rows convention must name each one."""
    rows = convention(readme, "Rows")
    kept_somewhere = set()
    for columns in (schema.POSITIONS, schema.SHAPES):
        cells = dict(zip((c.name for c in columns), schema.format_row(columns, lost_row(columns)).split(",")))
        kept = [c.name for c in columns if c.dtype in ("int", "float") and cells[c.name] != ""]
        assert {"frame", "t_s"} <= set(kept)  # SPEC 8.2: a lost frame keeps its row
        for name in kept:
            assert has_word(rows, name), f"{name} stays filled in a lost row but Rows does not say so"
        kept_somewhere.update(kept)
    assert kept_somewhere == {"frame", "t_s", "visible", "n_components", "shape_ok"}
    assert "every number is an empty cell" not in normalized(readme)


def test_the_readme_does_not_say_that_every_number_has_its_unit_in_its_name(readme):
    flat = normalized(readme)
    assert "The unit of every number is in its column name" not in flat
    units = convention(readme, "Units")
    for name in ("r", "g", "b", "gray"):  # SPEC 8.7: probe means on the 0-255 scale
        assert has_word(units, name), name
    assert "0-255" in units
    assert "ratio" in units and "no unit" in units
    for name in ("eccentricity", "core_frac", "solidity", "circularity", "largest_fraction"):
        assert has_word(units, name), name  # the ratios, which have no unit


def test_core_frac_is_described_for_the_fallback_case(readme):
    """SPEC 7.3: if the opening removes more than half the area the full mask is the core, and the stored
    core_frac stays the fraction the opening left (tests/test_measure_core.py), so it is below 0.5."""
    shapes = {c.name: c.meaning for c in schema.SHAPES}["core_frac"]
    stored = {k.name: k.meaning for k in schema.RESULTS_KEYS}["core_frac"]
    for meaning in (shapes, stored):
        text = normalized(meaning)
        assert "fallback" in text and "0.5" in text and "not 1" in text, text
    assert normalized(shapes) != "core area / full mask area"
    assert normalized(stored) != "core area / mask area"
    assert normalized(shapes) in normalized(readme)  # the README is built from the table
