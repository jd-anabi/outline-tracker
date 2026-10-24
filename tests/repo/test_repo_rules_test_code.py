"""Rules about the test files: how they share helpers and fixtures, and that no test waits by a delay.

- no module under tests/ imports a conftest.py, no fixture is imported by name, and no test module imports from
  another test module: what tests share is a helper module or a fixture of a conftest.py;
- no test waits for a fixed time: a call of `sleep` or `qWait` and the like is found by a scan of the names, and the
  two calls that stay are listed with their reasons.

The other rules of the repository (the entry point, the import boundaries, no private data, when CI runs) are in
tests/repo/test_repo_rules.py.
"""

import ast
from pathlib import Path

from import_helpers import imports
from layout_helpers import PATHS, REPO, below

TESTS = REPO / "tests"


# ---------------------------------------------------------------------------------------------
# How the tests share helpers and fixtures


def _imported_paths(tree: ast.AST):
    """Yield (line, dotted path) for everything that an absolute import in a parsed file names
    (`imports`), with every part of the name: `import a.b` and `import_module("a.b")` name a.b;
    `from a.b import c, d` names a.b.c and a.b.d, because c may be a module of a.b as well as a
    name in it."""
    for line, module, names in imports(tree):
        for name in names or ("",):
            yield line, f"{module}.{name}".rstrip(".")


def _conftest_imports(folder: Path) -> list[str]:
    """Scan every .py file under `folder`; return `file:line` for each line that imports a module
    named conftest, at any depth (the three ways of `imports`). Every part of a dotted name is looked
    at (`_imported_paths`): `tests.conftest` is that module too, and so is `conftest` imported as a
    name of `tests`."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = sorted({line for line, dotted in _imported_paths(tree) if "conftest" in dotted.split(".")})
        found += [f"{path.relative_to(folder).as_posix()}:{line}" for line in lines]
    return found


def test_no_module_under_tests_imports_conftest(tmp_path):
    # A conftest.py is pytest's: it holds fixtures, and pytest finds it by its folder. Python keeps one
    # module under that name, so with a second conftest.py below tests/ an import by that name gets the
    # wrong file in a run of the whole suite, while a run of the one file passes (measured with pytest
    # 9.1.1). What tests import by name is in a helper module.
    scanned = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("*.py")}
    real_files = {"conftest.py", *below(TESTS, "test helpers", "tracker io test", "window helpers")}
    assert real_files <= scanned  # the real folder
    assert _conftest_imports(TESTS) == []
    # The scan finds every spelling, also inside a function and in a folder below, and not the word alone.
    # The name is put in here, so that a search of tests/ for such an import finds nothing in this file.
    name = "conftest"
    sources = {
        "test_a.py": f"import os\nfrom {name} import java_sci\n",
        "helpers.py": f"def late():\n    import {name}\n    return {name}\n",
        "gui/test_b.py": f"import importlib\n\nfound = importlib.import_module('{name}')\n",
        "gui/conftest.py": f"import pytest\nfrom helpers import late  # used by this {name}.py\n",
        "test_c.py": f"from {name}_helpers import late\n",
        # by way of the folder's name: the module, a name from it, the module as a name of the folder
        "slow/test_d.py": f"import tests.{name}\nfrom tests.gui.{name} import look\nfrom tests.gui import {name}\n",
        "slow/d_helpers.py": f"from tests.helpers import {name}_of, late\n",
    }
    for rel, text in sources.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert _conftest_imports(tmp_path) == ["gui/test_b.py:3", "helpers.py:2", "slow/test_d.py:1", "slow/test_d.py:2",
                                           "slow/test_d.py:3", "test_a.py:2"]


def _fixtures_of(tree: ast.Module) -> dict[str, int]:
    """The pytest fixtures a parsed file defines at its top level, {name: line}: the functions under
    `@pytest.fixture` or `@fixture`, with or without arguments."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
            if any(ast.unparse(d) in ("pytest.fixture", "fixture") for d in decorators):
                found[node.name] = node.lineno
    return found


def _fixtures_out_of_place(folder: Path) -> list[str]:
    """Scan every .py file under `folder`, where a module's name is its file's name (the test folders
    are on the import path). Return one line for each fixture that a helper module defines (a file
    that is neither a conftest.py nor a test_*.py), and one for each `from module import ...` that
    brings in a fixture of that module, by name or with a star."""
    trees = {path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
             for path in sorted(folder.rglob("*.py"))}
    fixtures: dict[str, dict[str, int]] = {}
    for path, tree in trees.items():
        fixtures.setdefault(path.stem, {}).update(_fixtures_of(tree))
    found = []
    for path, tree in trees.items():
        rel = path.relative_to(folder).as_posix()
        if path.name != "conftest.py" and not path.name.startswith("test_"):
            found += [f"{rel}:{line}: the fixture {name} is in a helper module"
                      for name, line in _fixtures_of(tree).items()]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and fixtures.get(node.module):
                theirs = fixtures[node.module]
                asked = [alias.name for alias in node.names]
                names = sorted(theirs) if "*" in asked else [name for name in asked if name in theirs]
                if names:
                    found.append(f"{rel}:{node.lineno}: imports {', '.join(names)} from {node.module}")
    return found


def test_no_fixture_is_imported_by_name(tmp_path):
    # pytest finds a fixture by its name in the test's module and in the conftest.py files of the folders
    # above it. A fixture imported into a test module is a second fixture: a session fixture is made once
    # more for each module that imports it, an autouse fixture acts wherever it is imported, and a test
    # that asks for it shadows the imported name. So a fixture is defined in a conftest.py, or in the
    # test module that uses it.
    scanned = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("*.py")}
    real_files = {"conftest.py", *below(TESTS, "test helpers", "session helpers", "theme test")}
    assert real_files <= scanned  # the real folder
    assert _fixtures_out_of_place(TESTS) == []
    # the scan finds a fixture in a helper module, and its import by name, over several lines and with a
    # star; it leaves alone the fixtures of a conftest.py and of a test module, and the import of a function
    sources = {
        "conftest.py": "import pytest\n\n\n@pytest.fixture(scope='session')\ndef clip():\n    return 1\n",
        "gui/conftest.py": "from pytest import fixture\n\n\n@fixture\ndef look():\n    yield\n",
        "gui/panel_helpers.py": ("import pytest\n\n\n@pytest.fixture\ndef own():\n    return 2\n\n\n"
                                 "def body():\n    return 3\n"),
        "gui/test_panel.py": ("from panel_helpers import body, own\nfrom test_theme import (badge,\n"
                              "                        shade)\n\n\ndef test_body(own, shade):\n    pass\n"),
        "gui/test_theme.py": ("import pytest\n\n\n@pytest.fixture\ndef shade():\n    yield\n\n\n"
                              "def badge():\n    return 4\n\n\ndef test_shade(shade, look, clip):\n    pass\n"),
        "gui/test_view.py": "from panel_helpers import body\nfrom test_theme import badge\n",
        "test_all.py": "from panel_helpers import *\n",
    }
    for rel, text in sources.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert _fixtures_out_of_place(tmp_path) == [
        "gui/panel_helpers.py:5: the fixture own is in a helper module",
        "gui/test_panel.py:1: imports own from panel_helpers",
        "gui/test_panel.py:2: imports shade from test_theme",
        "test_all.py:1: imports own from panel_helpers",
    ]


def _test_module_imports(folder: Path) -> list[str]:
    """Scan every .py file under `folder`; return `file:line: imports module` for each import of a
    module whose name begins with `test_`, at any depth (the three ways of `imports`). Every part of
    a dotted name is looked at (`_imported_paths`), and `module` is the name up to the first such
    part: `tests.test_b` for `from tests.test_b import one` and for `from tests import test_b`."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen = set()  # (line, module): `from test_b import one, two` names the module twice
        for line, dotted in _imported_paths(tree):
            parts = dotted.split(".")
            first = next((count for count, part in enumerate(parts, start=1) if part.startswith("test_")), None)
            if first is not None:
                seen.add((line, ".".join(parts[:first])))
        found += [f"{path.relative_to(folder).as_posix()}:{line}: imports {module}" for line, module in sorted(seen)]
    return found


def test_no_test_module_imports_from_a_test_module(tmp_path):
    # A test file is pytest's: it is collected and run, split when it grows long, renamed with what it
    # tests. A file that imports a name from it depends on all of that, and Python runs the whole test
    # file to hand out the one name. What a second file needs is a helper, and its home is a helper module.
    scanned = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("*.py")}
    assert below(TESTS, "test helpers", "theme test", "slow test helpers") <= scanned  # the real folder
    assert _test_module_imports(TESTS) == []
    # The scan finds every spelling, in a test file, a helper module and a conftest.py, in a folder below
    # and inside a function; a name that only holds `test_`, or begins with `test` alone, is left alone.
    sources = {
        "test_a.py": "import os\nfrom test_b import one\n",
        "test_b.py": "import pytest\nfrom b_helpers import one\nfrom latest_helpers import two\n",
        "b_helpers.py": "def late():\n    import test_a.parts\n    return test_a\n",
        "conftest.py": "import pytest\nfrom test_b import one, two\n",
        "gui/test_c.py": "import importlib\nimport testing_tools\n\nfound = importlib.import_module('test_a')\n",
        "slow/c_helpers.py": "from test_c import (found,\n                    other)\n",
        # by way of the folder's name: a name from the module, the module as a name of the folder, the module
        "slow/test_d.py": ("from tests.test_b import one\nfrom tests.gui import test_c, gui_helpers\n"
                           "import tests.gui.test_c\nimport tests.testing_tools\nfrom tests.b_helpers import late\n"),
    }
    for rel, text in sources.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert _test_module_imports(tmp_path) == [
        "b_helpers.py:2: imports test_a",
        "conftest.py:2: imports test_b",
        "gui/test_c.py:4: imports test_a",
        "slow/c_helpers.py:1: imports test_c",
        "slow/test_d.py:1: imports tests.test_b",
        "slow/test_d.py:2: imports tests.gui.test_c",
        "slow/test_d.py:3: imports tests.gui.test_c",
        "test_a.py:2: imports test_b",
    ]


# ---------------------------------------------------------------------------------------------
# No test waits by a delay


# What waits for a time when it is called, by the last part of its name: `time.sleep` and a bare `sleep`,
# `QThread.msleep` and `usleep`, `QTest.qWait` and `qSleep`, `threading.Timer` and a bare `Timer`.
DELAY_NAMES = {"sleep", "msleep", "usleep", "qWait", "qSleep", "Timer"}


# The calls that the scan finds and that stay, {file: what it calls there}, each with its reason.
ALLOWED_DELAYS = {
    # A real delay, the one exception: the steps of a mouse drag are 20 ms apart, because pyqtgraph drops a
    # mouse move that follows another one sooner (`drag`).
    PATHS["window helpers"]: ["QTest.qWait"],
    # Not a delay: the fixture `lock_file` has put its recorder in place of the wait, and the test that calls
    # it asserts that nothing was waited (`test_lock_file_skips_the_waits_between_the_tries_and_records_them`).
    PATHS["fileio test"]: ["fileio.time.sleep"],
}


def _delay_calls(source: str) -> list[tuple[int, str]]:
    """(line, what is called) for every call in the Python text `source` that waits for a time, in the
    order of the lines: a call of a name in `DELAY_NAMES`, alone or as the last part of a dotted name,
    and a call of `wait` on pytest-qt's `qtbot`, which waits for a number of ms. The `wait` of an event
    or of a thread is not one: it waits for that event or thread. The line is the one the call begins in.

    The scan reads names, not objects, so it takes a function for a wait that only has one of the names.
    When it takes a call that waits for nothing for a delay, list that call in `ALLOWED_DELAYS` with the
    reason; do not call it under another name. The scan does not see a delay that is called under another
    name, a `QTimer.singleShot` with a delay above 0, or a `QTimer` that is started.
    """
    def last(name: ast.expr) -> str | None:  # `sleep` of `fileio.time.sleep` and of `sleep`
        return name.attr if isinstance(name, ast.Attribute) else getattr(name, "id", None)

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            on_qtbot = isinstance(node.func, ast.Attribute) and last(node.func.value) == "qtbot"
            if last(node.func) in DELAY_NAMES or (last(node.func) == "wait" and on_qtbot):
                found.append((node.lineno, ast.unparse(node.func)))
    return sorted(found)


def test_no_test_waits_by_a_delay():
    # A wait for a fixed time is too long on a fast machine and too short on a busy one: the test then
    # fails, or it passes without showing what it is there for. A test waits for the thing itself: a
    # condition (`qtbot.waitUntil`), an event with a safety bound, a stand-in parked on a gate
    # (tests/support/prompt_helpers.py, `Gate`).
    files = [REPO / "conftest.py", *sorted(TESTS.rglob("*.py"))]
    found = {path.relative_to(REPO).as_posix(): _delay_calls(path.read_text(encoding="utf-8")) for path in files}
    assert {"conftest.py", "tests/conftest.py", "tests/slow/conftest.py",
            *below(REPO, "window helpers", "worker jobs test")} <= set(found)  # the real folders
    assert [f"{name}:{line}: calls {called}" for name, calls in found.items() for line, called in calls
            if called not in ALLOWED_DELAYS.get(name, [])] == []
    # a listed call is there as often as the list says: one more of its kind in that file is a finding too
    assert {name: [called for _, called in calls] for name, calls in found.items() if calls} == ALLOWED_DELAYS


def test_the_delay_scan_flags_each_kind():
    # The rule test above shows that the scan finds the two listed calls in the tests and nothing else, not
    # that it would find a wait of another kind: check it on a text that waits in each way once, and on a text
    # that only looks so. The two words are put in here, so that a search of tests/ for a call of them
    # finds nothing in this file. The lines are counted by hand.
    sleep, timer = "sleep", "Timer"
    waits = "\n".join([
        "import threading",                                   # line 1
        "import time",
        f"from threading import {timer}",
        f"from time import {sleep}",
        "",                                                   # line 5
        "from PySide6.QtCore import QThread",
        "from PySide6.QtTest import QTest",
        "",
        "",
        "def test_closing(qtbot, window, gate):",             # line 10
        f"    time.{sleep}(0.2)",
        f"    opener = threading.{timer}(0.2, gate.open)",
        "    opener.start()",
        f"    {timer}(0.2, gate.open).start()",
        "    QTest.qWait(20)",                                # line 15
        "    qtbot.wait(",
        "        200)",
        f"    QThread.m{sleep}(200)",
        "",
        "    def check(path):",                               # line 20
        f"        {sleep}(0.2)",
        f"        return [QThread.u{sleep}(200) for _ in range(2)]",
        "",
        "    QTest.qSleep(20)",
        "    window.close()",                                 # line 25
        "",
    ])
    assert _delay_calls(waits) == [
        (11, f"time.{sleep}"), (12, f"threading.{timer}"), (14, timer), (15, "QTest.qWait"), (16, "qtbot.wait"),
        (18, f"QThread.m{sleep}"), (21, sleep), (22, f"QThread.u{sleep}"), (24, "QTest.qSleep")]
    waits_for_events = "\n".join([
        f'"""No test here waits by a delay: time.{sleep}(0.2) and threading.{timer}(0.2, gate.open) are words."""',
        "import threading",
        "import time",
        "",
        "from PySide6.QtCore import QTimer",
        "",
        "from outline_tracker.core import fileio",
        "",
        "SAFETY_S = 30",
        "",
        "",
        "def test_closing(qtbot, monkeypatch, worker, thread):",
        f'    monkeypatch.setattr(fileio.time, "{sleep}", lambda seconds: None)  # replaced, and never called',
        f"    real = time.{sleep}  # named, and not called",
        "    started = threading.Event()",
        "    assert not started.wait(SAFETY_S)  # an event, with a bound",
        "    worker.stopping.wait(SAFETY_S)",
        "    thread.wait()",
        "    QTimer.singleShot(0, started.set)  # a probe that the event loop runs",
        "    qtbot.waitUntil(started.is_set)",
        f'    said = "QTest.qWait(20), qtbot.wait(200) and {sleep}(1) are words here too"',
        f"    # time.{sleep}(0.2) would wait here, and so would QTest.qWait(20)",
        "",
    ])
    assert _delay_calls(waits_for_events) == []
