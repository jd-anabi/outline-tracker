"""Ported code keeps its behavior: its source text must equal last week's (SPEC 0, 13.3; CLAUDE.md).

`tests/reference/` holds the template's files, byte for byte (importable here as package `shrimp`
through pytest's `pythonpath`; the package under test never imports it). For every function or
class that a port task moved over unchanged, this module compares `inspect.getsource` of the new
copy with the reference copy. An unattended "improvement" of ported code makes a test here fail.

To add a port (every port task does): add one row to `VERBATIM`, with the new module, the
reference module and the names moved unchanged. A name whose text had to change (a message that
named last week's command, say) goes into `ADAPTED` instead, with the exact text replacements,
each of which must occur once in the reference source. Names that exist in both modules and are
listed in neither table fail `test_no_ported_name_is_left_unchecked`.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
REFERENCE_TESTS = TESTS / "reference" / "template_tests"

# (new module, reference module, names moved verbatim). Functions and classes only.
VERBATIM: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "outline_tracker.video",
        "shrimp.video",
        ("VideoInfo", "VideoCheck", "_open", "_to_gray", "probe", "iter_frames", "read_frame",
         "frame_changes", "_ramp_ratio"),
    ),
    ("outline_tracker.convert", "shrimp.convert", ("ffmpeg_exe", "convert_for_tracker")),
]

# (new module, reference module, name, ((old text, new text), ...)). The new source must equal the
# reference source after the replacements, nothing else.
ADAPTED: list[tuple[str, str, str, tuple[tuple[str, str], ...]]] = [
    (
        "outline_tracker.video",
        "shrimp.video",
        "check_video",
        # The note names last week's module, which students no longer have.
        (("through shrimp.video so all coordinates", "with Outline Tracker so all coordinates"),),
    ),
]

# Ported test files: (new file, reference file, (old line, new line)). Only this line may differ
# (X3: tests are ported with their import line changed).
PORTED_TESTS: list[tuple[str, str, tuple[str, str]]] = [
    ("test_video.py", "test_video.py",
     ("from shrimp import video", "from outline_tracker import video")),
    ("test_convert.py", "test_convert.py",
     ("from shrimp import convert, video", "from outline_tracker import convert, video")),
]


def _source(module_name: str, name: str) -> str:
    """The source text of a function or class of an importable module."""
    return inspect.getsource(getattr(importlib.import_module(module_name), name))


def _adapted(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Apply the listed text replacements; each old text must occur exactly once in `source`."""
    for old, new in replacements:
        assert source.count(old) == 1, f"{old!r} must occur exactly once in the reference source"
        source = source.replace(old, new)
    return source


def _top_level_names(path: Path) -> set[str]:
    """Names of the functions and classes defined at the top level of a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return {node.name for node in tree.body if isinstance(node, kinds)}


def _unchecked(new_file: Path, reference_file: Path, checked: set[str]) -> list[str]:
    """Names defined in both files that are not in `checked`: ported but not compared."""
    return sorted((_top_level_names(new_file) & _top_level_names(reference_file)) - checked)


def _module_file(module_name: str) -> Path:
    return Path(inspect.getsourcefile(importlib.import_module(module_name)))


VERBATIM_CASES = [(new, ref, name) for new, ref, names in VERBATIM for name in names]
VERBATIM_IDS = [f"{new}.{name}" for new, _, name in VERBATIM_CASES]
ADAPTED_IDS = [f"{new}.{name}" for new, _, name, _ in ADAPTED]


@pytest.mark.parametrize("new_module, reference_module, name", VERBATIM_CASES, ids=VERBATIM_IDS)
def test_moved_verbatim(new_module, reference_module, name):
    assert _source(new_module, name) == _source(reference_module, name)


@pytest.mark.parametrize("new_module, reference_module, name, replacements", ADAPTED, ids=ADAPTED_IDS)
def test_adapted_differs_only_by_the_listed_replacements(new_module, reference_module, name, replacements):
    expected = _adapted(_source(reference_module, name), replacements)
    assert _source(new_module, name) == expected


@pytest.mark.parametrize("new_module, reference_module", sorted({(n, r) for n, r, _ in VERBATIM}))
def test_no_ported_name_is_left_unchecked(new_module, reference_module):
    checked = {name for n, _, names in VERBATIM if n == new_module for name in names}
    checked |= {name for n, _, name, _ in ADAPTED if n == new_module}
    assert _unchecked(_module_file(new_module), _module_file(reference_module), checked) == []


@pytest.mark.parametrize("new_name, reference_name, change", PORTED_TESTS, ids=[t[0] for t in PORTED_TESTS])
def test_ported_tests_differ_only_in_the_import_line(new_name, reference_name, change):
    new = (TESTS / new_name).read_text(encoding="utf-8").splitlines()
    reference = (REFERENCE_TESTS / reference_name).read_text(encoding="utf-8").splitlines()
    old_line, new_line = change
    assert reference.count(old_line) == 1
    assert [new_line if line == old_line else line for line in reference] == new


# ---------------------------------------------------------------------------------------------
# The checks above must be able to fail: try them on small made-up files.


def test_adapted_demands_a_unique_match():
    assert _adapted("see a and b", (("a and", "x and"),)) == "see x and b"
    with pytest.raises(AssertionError):
        _adapted("see a and b", (("c", "d"),))  # absent: the table has rotted
    with pytest.raises(AssertionError):
        _adapted("a and a", (("a", "x"),))  # ambiguous


def test_unchecked_finds_same_named_definitions_that_nobody_compares(tmp_path):
    reference = tmp_path / "reference.py"
    reference.write_text("def probe():\n    pass\n\n\nclass Info:\n    pass\n\n\ndef main():\n    pass\n")
    new = tmp_path / "new.py"
    new.write_text("def probe():\n    return 1\n\n\nclass Info:\n    pass\n\n\ndef fresh():\n    pass\n")
    # `fresh` exists only in the new module; `main` only in the reference: neither needs a row.
    assert _unchecked(new, reference, set()) == ["Info", "probe"]
    assert _unchecked(new, reference, {"Info", "probe"}) == []
