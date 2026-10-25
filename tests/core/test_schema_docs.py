"""Tests of outline_tracker/core/schema_docs.py: what the move out of core/schema.py must not change (SPEC 8.11).

The builders used to live in schema.py (now core/schema.py). They moved out (task B6a) because it had grown to 591
lines and SPEC 12 says to split past about 400. The tests here pin what must not change in the move (the
text, the names, the sizes, the imports). Expected values are typed from the specification or computed from the
schema tables, never copied from the output under test.

The rest of the tests of the module are beside this file: the wording of the README text and the units it
gives in tests/core/test_schema_docs_readme.py, outputs_markdown() and the committed docs/OUTPUTS.md in
tests/core/test_schema_docs_outputs.py. The fixtures `schema_docs`, `readme` and `outputs` are in
tests/core/conftest.py; what the last two files share is in tests/support/schema_docs_helpers.py.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from layout_helpers import REPO, module

from outline_tracker.core import schema


# SHA-256 of the UTF-8 bytes of readme_text(), computed with
#   uv run python -c "import hashlib; from outline_tracker.core import schema; \
#       print(hashlib.sha256(schema.readme_text().encode('utf-8')).hexdigest())"
# Commit 1 of task B6a moved the builders out of core/schema.py without changing a character. Its pinned value
# was computed from the unmodified schema.py (now core/schema.py) of commit e49a088, before the move:
#   9cede8f4c9ba9cc98f4ee3ebbdb1907aa992700e9bde43c2013131705109b57b   (13751 characters, 249 lines)
# Commit 2 changed the wording of four places (the intro, Units, Rows and the core_frac entry, see the
# tests below). The value pinned now was computed after that change, and the diff between the two texts was
# read: those four places and nothing else differ (14468 characters, 258 lines; hash 154a055f...).
# Commit 3 (docs/OUTPUTS.md) turned "(see FLAGS)" into "(see the quality flags)" in the two flags columns, so
# that the table text also points right in OUTPUTS.md, which has no section called FLAGS; again the diff was
# read and shows those two lines only (14498 characters, 259 lines). Its refactor of the builder, to share the
# prose with OUTPUTS.md, left the text unchanged: the value above was checked before that edit. Any later
# change of the README text has to update this value on purpose.
# Fix round 1 (review of B6a) changed two places, and the diff of the old and the new text was read: the intro
# now limits its claim to the CSV files, and the Units convention lists every column whose name does not show
# its unit (the Tracker-format columns t, x, y, pixelx, pixely; px_along_major, cells_along_major; the 72
# radius columns; the probe means), read from the tables. The text before this change had the hash
# 662e5f73c9a1dd0331dba2cfc656c20c957bbde680a0f916cd2d6d379d5ba1df (14498 characters, 259 lines); the text
# then had 14831 characters and 262 lines.
# Decision 26 of docs/ROADMAP.md (W1 step 3) changed one description: px_along_major is of "the largest piece
# of the mask", no longer of "the full mask". The diff of the old and the new text was read and shows that
# entry and nothing else (its two lines, wrapped again). The text before this change had the hash
# 174e5877b06aea49446e8119a18822f08f475a051ef7885507b799ca6a1f94c0 (14831 characters, 262 lines); the text
# then had 14847 characters and 262 lines.
# The review of that change found that "(L1 / k)" in px_along_major and cells_along_major borrowed the name
# that major_mm gives to the full mask's axis. The two descriptions now give the formula of the largest piece
# and say that cells_along_major is px_along_major / cell. The diff of the old and the new text was read and
# shows those two entries and nothing else (three lines). The text before this change had the hash
# 0a3d0042970e10beaaee704d18ef648c6203ee8ac68af5600c0ccdc4dccfe098 (14847 characters, 262 lines); the text
# now has 14877 characters and 262 lines.
README_SHA256 = "20ec79686c3b77a80a5966f7d1e929274861aaf07c3956d8d88a8a5435ce7c73"

# Every public name that core/schema.py had before the move of the builders (dir() of the unmodified module,
# without the imported helpers), except the three that left it later (NAMES_THAT_LEFT_SCHEMA). Other modules
# import these by name, so each must still be importable.
PUBLIC_NAMES_OF_SCHEMA = [
    "ArrayKey", "CSV_ENCODING", "CSV_NEWLINE", "Column", "FILES", "FLAGS", "FLAG_INFO", "FLAG_SEPARATOR",
    "FileSpec", "FlagInfo", "OUTLINES_KEYS", "OUTLINES_META_FIELDS", "OUTLINES_META_KEY", "OUTLINES_NPZ",
    "OUTLINE_POINTS", "OVERLAY_MP4", "POSITIONS", "POSITIONS_CSV", "POSITION_FLAGS", "PROBES", "PROBES_CSV",
    "RADIAL", "RADIAL_ANGLES", "RADIAL_CSV", "RADIAL_STEP_DEG", "README_TXT", "RESULTS_KEYS", "RESULTS_NPZ",
    "RESULTS_VERSION", "RESULTS_VERSION_KEY", "RUN_LOG", "SESSION_JSON", "SESSION_SCHEMA_VERSION", "SHAPES",
    "SHAPES_CSV", "SHAPE_FLAGS", "STORED_OUTLINE_POINTS", "TRACKER_COLUMNS", "TRACKER_FILE", "join_flags",
    "npz_key", "readme_text", "split_npz_key",
]

# The names that left core/schema.py in W2 step 6 (docs/ROADMAP.md), each with the module it went to. They are
# imported from there; schema does not hand them out any more, so that no import is kept working by a front.
NAMES_THAT_LEFT_SCHEMA = {"csv_text": "csv_format", "format_row": "csv_format", "header_line": "csv_format"}

LIGHT_CHECK = (
    "import sys; {imports}; "
    "bad = sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph', 'pandas') if m in sys.modules); "
    "print(bad)"
)


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
    from outline_tracker.core.schema import readme_text

    assert readme_text is schema_docs.readme_text
    assert schema.readme_text is schema_docs.readme_text


@pytest.mark.parametrize("name", PUBLIC_NAMES_OF_SCHEMA)
def test_every_public_name_of_schema_is_still_importable(name):
    namespace: dict = {}
    exec(f"from outline_tracker.core.schema import {name}", namespace)  # noqa: S102
    assert namespace[name] is getattr(schema, name)


@pytest.mark.parametrize("name", NAMES_THAT_LEFT_SCHEMA)
def test_a_name_that_left_schema_is_importable_from_its_new_module_only(name):
    namespace: dict = {}
    exec(f"from outline_tracker.core.{NAMES_THAT_LEFT_SCHEMA[name]} import {name}", namespace)  # noqa: S102
    assert callable(namespace[name])
    assert not hasattr(schema, name)


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
        "import outline_tracker.core.schema",
        "import outline_tracker.core.schema_docs",
        "import outline_tracker.core.schema_docs, outline_tracker.core.schema",
        "from outline_tracker.core.schema import readme_text",
    ],
    ids=["schema", "schema_docs", "schema_docs and schema", "readme_text from schema"],
)
def test_importing_either_module_loads_neither_pandas_torch_nor_qt(imports):
    done = python(LIGHT_CHECK.format(imports=imports))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "imports",
    [
        "import outline_tracker.core.schema_docs as d; import outline_tracker.core.schema as s",
        "import outline_tracker.core.schema as s; import outline_tracker.core.schema_docs as d",
    ],
    ids=["schema_docs first", "schema first"],
)
def test_import_order_does_not_matter(imports):
    """schema_docs reads the tables of schema, and schema hands out readme_text from schema_docs:
    neither import order may fail on the circle."""
    done = python(f"{imports}; assert s.readme_text is d.readme_text; print('ok')")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
