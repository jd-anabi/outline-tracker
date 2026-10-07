"""Tests of outline_tracker/schema_docs.py: the text builders of README.txt and docs/OUTPUTS.md (SPEC 8.11, 16).

The builders used to live in schema.py. They moved out (task B6a) because schema.py had grown to 591
lines and SPEC 12 says to split past about 400. The first group of tests pins what must not change in
the move; the later groups test the wording and the Markdown contract. Expected values are typed from
the specification or computed from the schema tables, never copied from the output under test.
"""

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from outline_tracker import schema

REPO = Path(__file__).resolve().parents[1]

# SHA-256 of the UTF-8 bytes of readme_text(), computed BEFORE the builders moved out of schema.py:
# the unmodified schema.py of commit e49a088 (the base of task B6a), with
#   uv run python -c "import hashlib; from outline_tracker import schema; \
#       print(hashlib.sha256(schema.readme_text().encode('utf-8')).hexdigest())"
# The text has 13751 characters in 249 lines. The move must give the same text, character for character.
README_SHA256_BEFORE_THE_MOVE = "9cede8f4c9ba9cc98f4ee3ebbdb1907aa992700e9bde43c2013131705109b57b"

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

def test_readme_text_is_still_the_text_computed_before_the_move():
    text = schema.readme_text()  # through the old import path, as every existing caller uses it
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == README_SHA256_BEFORE_THE_MOVE


def test_readme_text_from_schema_docs_is_the_same_text(schema_docs):
    assert hashlib.sha256(schema_docs.readme_text().encode("utf-8")).hexdigest() == README_SHA256_BEFORE_THE_MOVE


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
