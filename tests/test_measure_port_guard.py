"""Guard for the ported test file tests/test_measure.py: the template's test there runs with the
modules its import lines name (SPEC 0, "port them"; plan decision X3).

tests/test_port_fidelity.py compares two texts with the template: the test
`test_mask_center_uses_trackers_pixel_convention` and the import lines of tests/test_measure.py.
A name rebound further down that file (by an assignment, or by a module loaded by name) is in
neither text, and would change what the unchanged test does. The check lives in a file of its own
because tests/test_measure.py holds nothing but the tests of mask_center, and its text is compared
with the template's.
"""

import runpy
import sys
from pathlib import Path

import pytest
import test_measure

PORTED = Path(__file__).with_name("test_measure.py")
TEMPLATE_TEST = "test_mask_center_uses_trackers_pixel_convention"
# The global names the template's test uses, and the modules that the import lines of
# tests/test_measure.py bind them to (`from outline_tracker import measure as segment`).
NAMES = ("np", "pytest", "segment")
MODULES = ("numpy", "pytest", "outline_tracker.measure")


def _modules_seen_by(test_function) -> tuple:
    """For each of `NAMES`: the name of the module that `test_function` finds under it when it runs.

    None stands for a name that is missing or bound to anything but an imported module (a number,
    a class, an object that only carries a module's name).
    """
    seen = []
    for name in NAMES:
        value = test_function.__globals__.get(name)
        module = getattr(value, "__name__", None)
        imported = isinstance(module, str) and sys.modules.get(module) is value
        seen.append(module if imported else None)
    return tuple(seen)


def test_the_template_test_runs_with_the_modules_named_by_the_import_lines():
    # `test_measure` is the module pytest collects the template's test from (both come from
    # sys.modules), and it has been executed as a whole by the time this file is imported: the
    # three names are read exactly as the template's test reads them when it runs.
    assert PORTED.samefile(test_measure.__file__)
    assert _modules_seen_by(getattr(test_measure, TEMPLATE_TEST)) == MODULES


# ---------------------------------------------------------------------------------------------
# The check above must be able to fail: try it on a small made-up file that starts like the ported
# one, with one line added at its end. None of these lines is an import line, and none changes the
# text of the test, which are the two things tests/test_port_fidelity.py compares.

STARTS_LIKE_THE_PORTED_FILE = """\
import numpy as np
import pytest

from outline_tracker import measure as segment


def test_mask_center_uses_trackers_pixel_convention():
    pass
"""

ADDED_LINES = {
    "nothing": ("", MODULES),
    # Another name: the three keep their meaning.
    "another_name": ('shapes = __import__("analytic_shapes")', MODULES),
    # A module loaded by name. This one is last week's copy of the module (tests/reference): with
    # it the template's test passes, on code that is not the package's.
    "reference_copy": ('segment = __import__("shrimp.segment").segment', ("numpy", "pytest", "shrimp.segment")),
    "another_module": ('np = __import__("math")', ("math", "pytest", "outline_tracker.measure")),
    "not_a_module": ("pytest = None", ("numpy", None, "outline_tracker.measure")),
    "deleted": ("del segment", ("numpy", "pytest", None)),
    "class_of_that_name": ('np = type("numpy", (), {})', (None, "pytest", "outline_tracker.measure")),
}


@pytest.mark.parametrize("added_line, seen", ADDED_LINES.values(), ids=ADDED_LINES.keys())
def test_the_guard_sees_a_name_rebound_below_the_import_lines(tmp_path, added_line, seen):
    made_up = tmp_path / "made_up.py"
    made_up.write_text(STARTS_LIKE_THE_PORTED_FILE + added_line + "\n", encoding="utf-8")
    assert _modules_seen_by(runpy.run_path(str(made_up))[TEMPLATE_TEST]) == seen
