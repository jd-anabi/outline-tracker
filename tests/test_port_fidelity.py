"""Ported code keeps its behavior: its source text must equal last week's (SPEC 0, 13.3; CLAUDE.md).

`tests/reference/` holds the template's files, byte for byte (importable here as package `shrimp`
through pytest's `pythonpath`; the package under test never imports it). For every function or
class that a port task moved over unchanged, this module compares `inspect.getsource` of the new
copy with the reference copy. An unattended "improvement" of ported code makes a test here fail.

To add a port (every port task does): add one row to `VERBATIM`, with the new module, the
reference module and the names moved unchanged. A name whose text had to change (a message that
named last week's command, say) goes into `ADAPTED` instead, with the exact text replacements,
each of which must occur once in the reference source. Module-level constants go into `CONSTANTS`
and are compared by value (for a dict, also the order of its items). Methods moved unchanged into
a class with a new name go into `VERBATIM_METHODS`; the methods of the same name that were
rewritten are named in the same row, so none is forgotten. Names that exist in both modules and
are listed in no table fail `test_no_ported_name_is_left_unchecked`.

A ported test file goes into `PORTED_TESTS` when it is one file of the template (only its import
line may differ), or into `SPLIT_TESTS` when the template's file was shared out among several new
files (each listed definition must equal the template's; the three tests that were not shared out
are named in `NOT_PORTED_HERE`).
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
    (
        "outline_tracker.tracker_io",
        "shrimp.segment",
        ("_cells", "read_tracker_export", "Calibration", "fit_calibration", "write_tracker_file",
         "_fps_from_export", "_fps_from_manifest", "Plan", "make_plan"),
    ),
    ("outline_tracker.measure", "shrimp.segment", ("mask_center",)),
    (
        "outline_tracker.segmenter.edgetam_convert",
        "shrimp._edgetam",
        ("_renumber", "convert_state_dict", "edgetam_config", "load_edgetam"),
    ),
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
    (
        "outline_tracker.segmenter.hf",
        "shrimp.segment",
        "load_model",
        (
            # An OSError while loading becomes "the first run needs internet once".
            ("def load_model(", "@needs_internet_once\ndef load_model("),
            # The package never imports the reference copy.
            ("from shrimp._edgetam import load_edgetam",
             "from outline_tracker.segmenter.edgetam_convert import load_edgetam"),
        ),
    ),
]

# (new module, reference module, names of module-level constants). Compared by value; for a dict
# also the order of its items (the EdgeTAM key renaming applies its entries one after another).
CONSTANTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("outline_tracker.segmenter.edgetam_convert", "shrimp._edgetam", ("KEYS_TO_MODIFY_MAPPING", "PERCEIVER")),
    ("outline_tracker.segmenter.hf", "shrimp.segment", ("KEEP_FRAMES", "MODELS")),
]

# (new module, new class, reference module, reference class, methods moved verbatim, methods of
# the same name that were rewritten). Every method name the two classes share is in one of the two.
VERBATIM_METHODS: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "outline_tracker.segmenter.hf", "HFSegmenter", "shrimp.segment", "TransformersSegmenter",
        ("_pick_device", "_prune"),
        # __init__ also records model_id and weights_sha256; _forward adds the prompts with one
        # call per object and returns cropped results with logits; start and step take and return
        # the records of segmenter/base.py.
        ("__init__", "_forward", "start", "step"),
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

# The template's test_segment.py went to several new files, as the plan's table (section 8) says.
# Each row: (new file, the import line that replaces the template's `from shrimp import segment`,
# the top-level definitions and assignments taken over, whether the file holds nothing else).
# Each taken-over name must have the same source text as in the template; the alias keeps the
# bodies of the tests unchanged. The three end-to-end tests are not here: they go through
# from_tracker and selftest (tasks B1 and B2), where their assertions are kept.
SPLIT_REFERENCE = "test_segment.py"
SPLIT_REFERENCE_IMPORT = "from shrimp import segment"
SPLIT_TESTS: list[tuple[str, str, tuple[str, ...], bool]] = [
    (
        "test_tracker_io.py",
        "from outline_tracker import tracker_io as segment",
        ("MM_PER_PX", "W, H", "tracker_map", "export_text",
         "test_read_one_point_mass_named_after_the_file",
         "test_read_several_point_masses_exported_together",
         "test_read_needs_the_pixel_columns",
         "test_calibration_recovers_trackers_map",
         "test_calibration_from_a_straight_track",
         "test_two_points_give_trackers_flipped_map",
         "test_calibration_needs_two_points",
         "test_written_file_reads_like_a_tracker_export",
         "test_plan_follows_the_tracker_track",
         "test_plan_for_many_shrimp_needs_fps_true",
         "test_every_shrimp_must_start_on_the_same_frame"),
        True,
    ),
    (
        "test_measure.py",
        "from outline_tracker import measure as segment",
        ("test_mask_center_uses_trackers_pixel_convention",),
        False,  # this file adds tests of its own below the template's
    ),
]
NOT_PORTED_HERE = (
    "test_whole_run_with_a_stand_in_model",
    "test_many_shrimp_from_a_start_file_in_extra",
    "test_selftest_runs_and_reports_the_time",
)


def _source(module_name: str, name: str) -> str:
    """The source text of a function or class of an importable module (`Class.method` for a method)."""
    found = importlib.import_module(module_name)
    for part in name.split("."):
        found = getattr(found, part)
    return inspect.getsource(found)


def _same_constant(new: object, reference: object) -> bool:
    """Equal values of the same type; two dicts must also list their items in the same order."""
    if type(new) is not type(reference) or new != reference:
        return False
    return not isinstance(new, dict) or list(new.items()) == list(reference.items())


def _method_names(module_name: str, class_name: str) -> set[str]:
    """Names of the functions defined in the body of a class."""
    cls = getattr(importlib.import_module(module_name), class_name)
    return {name for name, value in vars(cls).items() if inspect.isfunction(value)}


def _adapted(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Apply the listed text replacements; each old text must occur exactly once in `source`."""
    for old, new in replacements:
        assert source.count(old) == 1, f"{old!r} must occur exactly once in the reference source"
        source = source.replace(old, new)
    return source


def _top_level_names(path: Path) -> set[str]:
    """Names of the functions, classes and constants defined at the top level of a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    names = {node.name for node in tree.body if isinstance(node, kinds)}
    for node in tree.body:  # NAME = value, also NAME: type = value and A, B = values
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def _unchecked(new_file: Path, reference_file: Path, checked: set[str]) -> list[str]:
    """Names defined in both files that are not in `checked`: ported but not compared."""
    return sorted((_top_level_names(new_file) & _top_level_names(reference_file)) - checked)


def _module_file(module_name: str) -> Path:
    return Path(inspect.getsourcefile(importlib.import_module(module_name)))


def _target_name(target: ast.expr) -> str:
    """The name an assignment binds, as written: `W` or, for a tuple target, `W, H`."""
    if isinstance(target, ast.Tuple):
        return ", ".join(ast.unparse(element) for element in target.elts)
    return ast.unparse(target)


def _statement_sources(path: Path) -> dict[str, str]:
    """The top-level functions, classes and assignments of a Python file, as {name: source text}.

    The text runs from the first decorator (or the first line) to the last line of the statement.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        elif isinstance(node, ast.Assign):
            name = ", ".join(_target_name(t) for t in node.targets)
            first = node.lineno
        else:
            continue
        found[name] = "\n".join(lines[first - 1:node.end_lineno])
    return found


def _import_lines(path: Path) -> list[str]:
    """The lines of a Python file that start with `import` or `from` (the top-level imports)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith(("import ", "from "))]


VERBATIM_CASES = [(new, ref, name) for new, ref, names in VERBATIM for name in names]
VERBATIM_IDS = [f"{new}.{name}" for new, _, name in VERBATIM_CASES]
ADAPTED_IDS = [f"{new}.{name}" for new, _, name, _ in ADAPTED]
CONSTANT_CASES = [(new, ref, name) for new, ref, names in CONSTANTS for name in names]
CONSTANT_IDS = [f"{new}.{name}" for new, _, name in CONSTANT_CASES]
METHOD_CASES = [(new, cls, ref, ref_cls, name)
                for new, cls, ref, ref_cls, names, _ in VERBATIM_METHODS for name in names]
METHOD_IDS = [f"{new}.{cls}.{name}" for new, cls, _, _, name in METHOD_CASES]
MODULE_PAIRS = sorted({(row[0], row[1]) for row in [*VERBATIM, *ADAPTED, *CONSTANTS]}
                      | {(row[0], row[2]) for row in VERBATIM_METHODS})


@pytest.mark.parametrize("new_module, reference_module, name", VERBATIM_CASES, ids=VERBATIM_IDS)
def test_moved_verbatim(new_module, reference_module, name):
    assert _source(new_module, name) == _source(reference_module, name)


@pytest.mark.parametrize("new_module, reference_module, name, replacements", ADAPTED, ids=ADAPTED_IDS)
def test_adapted_differs_only_by_the_listed_replacements(new_module, reference_module, name, replacements):
    expected = _adapted(_source(reference_module, name), replacements)
    assert _source(new_module, name) == expected


@pytest.mark.parametrize("new_module, reference_module, name", CONSTANT_CASES, ids=CONSTANT_IDS)
def test_constant_has_the_reference_value(new_module, reference_module, name):
    new = getattr(importlib.import_module(new_module), name)
    reference = getattr(importlib.import_module(reference_module), name)
    assert _same_constant(new, reference)


@pytest.mark.parametrize("new_module, new_class, reference_module, reference_class, name", METHOD_CASES,
                         ids=METHOD_IDS)
def test_method_moved_verbatim(new_module, new_class, reference_module, reference_class, name):
    assert _source(new_module, f"{new_class}.{name}") == _source(reference_module, f"{reference_class}.{name}")


@pytest.mark.parametrize("new_module, new_class, reference_module, reference_class, verbatim, rewritten",
                         VERBATIM_METHODS, ids=[f"{row[0]}.{row[1]}" for row in VERBATIM_METHODS])
def test_no_ported_method_is_left_unchecked(new_module, new_class, reference_module, reference_class,
                                            verbatim, rewritten):
    shared = _method_names(new_module, new_class) & _method_names(reference_module, reference_class)
    assert not set(verbatim) & set(rewritten)
    assert shared == set(verbatim) | set(rewritten)


@pytest.mark.parametrize("new_module, reference_module", MODULE_PAIRS)
def test_no_ported_name_is_left_unchecked(new_module, reference_module):
    checked = {name for n, _, names in [*VERBATIM, *CONSTANTS] if n == new_module for name in names}
    checked |= {name for n, _, name, _ in ADAPTED if n == new_module}
    assert _unchecked(_module_file(new_module), _module_file(reference_module), checked) == []


@pytest.mark.parametrize("new_name, reference_name, change", PORTED_TESTS, ids=[t[0] for t in PORTED_TESTS])
def test_ported_tests_differ_only_in_the_import_line(new_name, reference_name, change):
    new = (TESTS / new_name).read_text(encoding="utf-8").splitlines()
    reference = (REFERENCE_TESTS / reference_name).read_text(encoding="utf-8").splitlines()
    old_line, new_line = change
    assert reference.count(old_line) == 1
    assert [new_line if line == old_line else line for line in reference] == new


@pytest.mark.parametrize("new_name, new_import, names, exact", SPLIT_TESTS, ids=[t[0] for t in SPLIT_TESTS])
def test_split_ported_tests_equal_the_templates(new_name, new_import, names, exact):
    new_file, reference_file = TESTS / new_name, REFERENCE_TESTS / SPLIT_REFERENCE
    new, reference = _statement_sources(new_file), _statement_sources(reference_file)
    for name in names:
        assert name in new, f"{new_name} lost {name}"
        assert new[name] == reference[name], f"{new_name}: {name} differs from the template"
    if exact:
        assert sorted(new) == sorted(names), "this file holds only what the template had"
        assert ast.get_docstring(ast.parse(new_file.read_text(encoding="utf-8"))) == ast.get_docstring(
            ast.parse(reference_file.read_text(encoding="utf-8")))


@pytest.mark.parametrize("new_name, new_import, names, exact", SPLIT_TESTS, ids=[t[0] for t in SPLIT_TESTS])
def test_split_ported_tests_differ_only_in_the_import_line(new_name, new_import, names, exact):
    new = _import_lines(TESTS / new_name)
    reference = _import_lines(REFERENCE_TESTS / SPLIT_REFERENCE)
    assert reference.count(SPLIT_REFERENCE_IMPORT) == 1
    assert new.count(new_import) == 1
    # Nothing but the changed line is new; a template import the file no longer needs may be gone.
    assert set(new) - {new_import} <= set(reference) - {SPLIT_REFERENCE_IMPORT}


def test_every_template_test_of_test_segment_has_a_home():
    reference = _statement_sources(REFERENCE_TESTS / SPLIT_REFERENCE)
    template_tests = {name for name in reference if name.startswith("test_")}
    homes = {name for _, _, names, _ in SPLIT_TESTS for name in names if name.startswith("test_")}
    assert template_tests == homes | set(NOT_PORTED_HERE)
    assert not homes & set(NOT_PORTED_HERE)


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


def test_unchecked_finds_constants_too(tmp_path):
    reference = tmp_path / "reference.py"
    reference.write_text("KEEP = 20\nW, H = 320, 240\nLIMIT: float = 3.0\nOLD = 1\n\n\ndef f():\n    KEEP = 1\n")
    new = tmp_path / "new.py"
    new.write_text("KEEP = 21\nW, H = 320, 240\nLIMIT: float = 3.0\nFRESH = 2\n")
    # `OLD` exists only in the reference, `FRESH` only in the new module; a name bound inside a
    # function is not a module constant.
    assert _unchecked(new, reference, set()) == ["H", "KEEP", "LIMIT", "W"]
    assert _unchecked(new, reference, {"H", "KEEP", "LIMIT", "W"}) == []


def test_same_constant_compares_value_type_and_dict_order():
    assert _same_constant(20, 20)
    assert not _same_constant(21, 20)
    assert not _same_constant(20.0, 20)  # another type
    assert _same_constant({"a": 1, "ab": 2}, {"a": 1, "ab": 2})
    assert not _same_constant({"a": 1, "ab": 3}, {"a": 1, "ab": 2})
    # Equal as dicts, but applied one after another in a different order.
    assert {"ab": 2, "a": 1} == {"a": 1, "ab": 2}
    assert not _same_constant({"ab": 2, "a": 1}, {"a": 1, "ab": 2})


def test_source_reads_a_method_through_its_class():
    assert _source(__name__, "_Sample.twice") == "    def twice(self, x):\n        return 2 * x\n"
    assert _method_names(__name__, "_Sample") == {"twice"}


class _Sample:
    limit = 3

    def twice(self, x):
        return 2 * x


def test_statement_sources_reads_decorators_tuples_and_trailing_comments(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        "import os\n"
        "W, H = 320, 240  # frame\n"
        "LIMIT = 3\n"
        "\n\n"
        "@decorated(1,\n"
        "           2)\n"
        "def f(x):\n"
        "    return x\n"
        "\n\n"
        "class C:\n"
        "    pass\n"
    )
    found = _statement_sources(source)
    assert sorted(found) == ["C", "LIMIT", "W, H", "f"]
    assert found["W, H"] == "W, H = 320, 240  # frame"
    assert found["f"] == "@decorated(1,\n           2)\ndef f(x):\n    return x"
    assert _import_lines(source) == ["import os"]
