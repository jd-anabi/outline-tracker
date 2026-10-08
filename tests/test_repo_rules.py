"""Rules of the repository itself: entry point, import boundaries, no private data, when CI runs and
which tests each of its jobs runs, how the tests share helpers and fixtures, and that no test waits
by a delay.

Checks that a command "loads neither torch nor Qt" run that command in a subprocess: pytest-qt has
already imported PySide6 into the test process.
"""

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from helpers import HOME_PATH, _PYTEST_TMP, _USERS

import outline_tracker

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "outline_tracker"
TESTS = REPO / "tests"
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"

# Import boundaries (SPEC 12, CLAUDE.md). Top-level module names.
QT_MODULES = {"PySide6", "pyqtgraph"}  # only inside outline_tracker/gui/
MODEL_MODULES = {"torch", "torchvision", "transformers", "timm"}  # only segmenter/ and gui/app.py
HEAVY_MODULES = sorted(QT_MODULES | MODEL_MODULES)

# Files that never go into git (SPEC 8.13, CLAUDE.md): videos, results, weights.
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".npz", ".pt", ".safetensors"}


def _run(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    """Run a command in the repo folder and capture its text output."""
    return subprocess.run(
        args, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


# ---------------------------------------------------------------------------------------------
# Entry point


def test_version_flag():
    # -X importtime prints every module this process imports to stderr, one per line.
    done = _run([sys.executable, "-X", "importtime", "-m", "outline_tracker.cli", "--version"])
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("outline-tracker 0.2.0.dev0 (commit "), done.stdout
    imported = {line.rsplit("|", 1)[-1].strip().split(".")[0] for line in done.stderr.splitlines()}
    assert "outline_tracker" in imported, done.stderr  # the listing is there and was parsed
    assert sorted(imported & set(HEAVY_MODULES)) == []


def test_the_version_is_written_the_same_in_its_three_places():
    # Two places are written by hand and `uv lock` copies the third. CI installs with `uv sync --locked`,
    # which refuses a lock that does not fit pyproject.toml: a forgotten place fails here first.
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))["package"]
    written = {
        "pyproject.toml": project["version"],
        "outline_tracker/__init__.py": outline_tracker.__version__,
        "uv.lock": [package["version"] for package in locked if package["name"] == "outline-tracker"],
    }
    assert written == {
        "pyproject.toml": "0.2.0.dev0",
        "outline_tracker/__init__.py": "0.2.0.dev0",
        "uv.lock": ["0.2.0.dev0"],
    }


# ---------------------------------------------------------------------------------------------
# Import boundaries


def _imported_top_levels(tree: ast.AST):
    """Yield (line, top-level module name) for every absolute import in a parsed file.

    Covers `import a.b`, `from a.b import c`, and `import_module("a.b")` / `__import__("a.b")`
    with a literal name, at any depth (lazy imports inside functions count too).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module.split(".")[0]
        elif isinstance(node, ast.Call) and node.args:
            func, first = node.func, node.args[0]
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
            if called in ("import_module", "__import__") and literal:
                yield node.lineno, first.value.split(".")[0]


def _import_violations(package: Path) -> list[tuple[str, int, str]]:
    """Scan every .py file under `package`; return (file, line, module) for each forbidden import."""
    found = []
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package)
        qt_allowed = rel.parts[0] == "gui"
        model_allowed = rel.parts[0] == "segmenter" or rel.as_posix() == "gui/app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, top in _imported_top_levels(tree):
            if (top in QT_MODULES and not qt_allowed) or (top in MODEL_MODULES and not model_allowed):
                found.append((rel.as_posix(), line, top))
    return found


def test_import_boundaries():
    scanned = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*.py")}
    assert {"__init__.py", "cli.py"} <= scanned  # the scan looks at the real package
    found = _import_violations(PACKAGE)
    assert [f"outline_tracker/{file}:{line}: imports {top}" for file, line, top in found] == []


def test_import_scan_flags_each_rule(tmp_path):
    # The scan above passes trivially on an empty package: check it on a package that breaks
    # every rule once, next to each allowed case.
    sources = {
        "__init__.py": "",
        "cli.py": "def run():\n    import torch\n    return torch\n",  # lazy import outside segmenter/
        "export.py": "import numpy as np\nfrom PySide6.QtWidgets import QWidget\n",
        "overlay.py": "import pyqtgraph as pg\n",
        "tracking.py": "import importlib\n\nhub = importlib.import_module('transformers.utils')\n",
        # relative imports are fine
        "measure.py": "import timm.models\nfrom . import schema\nfrom .segmenter import hf\n",
        "derive.py": "import torchvision\n",
        "segmenter/__init__.py": "",
        "segmenter/hf.py": "import torch\nimport transformers\nimport timm\nimport torchvision\n",
        "segmenter/fake.py": "from PySide6 import QtCore\n",
        "segmenter/edgetam_convert.py": "import torch\n",
        "gui/__init__.py": "",
        "gui/app.py": "import torch\nimport PySide6.QtWidgets\nimport pyqtgraph\n",
        "gui/worker.py": "from PySide6 import QtCore\nimport torch\n",
        "gui/panels/plot.py": "from PySide6 import QtWidgets\nimport pyqtgraph as pg\n",
    }
    for rel, text in sources.items():
        target = tmp_path / "pkg" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    expected = {
        ("cli.py", 2, "torch"),
        ("export.py", 2, "PySide6"),
        ("overlay.py", 1, "pyqtgraph"),
        ("tracking.py", 3, "transformers"),
        ("measure.py", 1, "timm"),
        ("derive.py", 1, "torchvision"),
        ("segmenter/fake.py", 1, "PySide6"),
        ("gui/worker.py", 2, "torch"),
    }
    assert set(_import_violations(tmp_path / "pkg")) == expected


_IMPORT_PROBE = """
import importlib, json, sys
heavy = json.loads(sys.argv[1])
loaded_by, seen = {}, set()
for name in sys.argv[2:]:
    importlib.import_module(name)
    new = sorted(m for m in heavy if m in sys.modules and m not in seen)
    if new:
        loaded_by[name] = new
        seen.update(new)
print(json.dumps(loaded_by))
"""


def test_core_does_not_load_torch():
    # Every module outside gui/ and segmenter/. A __main__.py is a script, not an importable module.
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE)
        if rel.parts[0] in ("gui", "segmenter") or rel.name == "__main__.py":
            continue
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(("outline_tracker", *parts)))
    assert {"outline_tracker", "outline_tracker.cli"} <= set(modules)
    done = _run([sys.executable, "-c", _IMPORT_PROBE, json.dumps(HEAVY_MODULES), *modules])
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {}, "module -> heavy modules its import loaded"


# ---------------------------------------------------------------------------------------------
# Nothing private, nothing big (the repo becomes public)


def _tracked_files() -> list[str]:
    done = _run(["git", "ls-files", "-z"])
    assert done.returncode == 0, done.stderr
    return [name for name in done.stdout.split("\0") if name]


def test_no_personal_paths_or_big_files():
    tracked = _tracked_files()
    assert "pyproject.toml" in tracked
    forbidden = [name for name in tracked if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]
    assert forbidden == []
    with_home_path = []
    for name in tracked:
        path = REPO / name
        if not path.is_file():  # deleted in the working tree, not yet committed
            continue
        data = path.read_bytes()
        if b"\0" in data:  # binary
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            if HOME_PATH.search(line):
                with_home_path.append(f"{name}:{number}")
    assert with_home_path == []


def test_home_path_pattern_examples():
    name = "someone"
    personal = [
        "/".join(["", _USERS, name, "Movies", "clip.mp4"]),  # macOS
        "video: " + "/".join(["", _USERS, name]),
        "\\".join(["C:", _USERS, name, "Videos"]),  # Windows
        "\\\\".join(["C:", _USERS, name, "Videos"]),  # Windows, backslashes escaped (JSON, repr)
        "/".join(["C:", _USERS, name]),  # Windows, forward slashes
        "/private/var/folders/ab/cd/T/" + _PYTEST_TMP + name + "/pytest-0/clip.mp4",
    ]
    generic = [
        "~/.cache/shrimp-models/edgetam",
        "$HOME/.venvs/outline-tracker",
        "%USERPROFILE%\\.local\\bin",
        "https://github.com/jd-anabi/outline-tracker",
        "the " + _USERS + " folder",
        "outline_tracker/segmenter/hf.py",
    ]
    assert [text for text in personal if not HOME_PATH.search(text)] == []
    assert [text for text in generic if HOME_PATH.search(text)] == []


# ---------------------------------------------------------------------------------------------
# When CI runs


def _path_filters(workflow: str) -> list[str]:
    """The lines of a workflow text that limit a trigger to some paths: `paths-ignore` anywhere, or a key
    `paths:`. The text is read as lines, not as YAML: no YAML library is a dependency."""
    return [line.strip() for line in workflow.splitlines()
            if "paths-ignore" in line or re.search(r"(?<![\w-])paths[\"']?\s*:", line)]


def test_ci_runs_for_every_push_to_main():
    # Fast tests read documents (README.md, SPEC.md, docs/), so a push that changes only documents must
    # start a run too (docs/ROADMAP.md, section 2, rule 2).
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^ *push:\n *branches: *\[main\]", workflow, re.MULTILINE)  # the trigger is there
    assert _path_filters(workflow) == []
    # the check finds both kinds of filter, as a block and inside braces, and nothing in a plain trigger
    assert _path_filters('on:\n  push:\n    branches: [main]\n    paths-ignore: ["**.md"] # no run\n') == [
        'paths-ignore: ["**.md"] # no run']
    assert _path_filters("on:\n  push:\n    paths:\n      - outline_tracker/**\n") == ["paths:"]
    assert _path_filters('on: {push: {branches: [main], "paths": ["tests/**"]}}\n') != []
    assert _path_filters("on:\n  push:\n    branches: [main]\n  pull_request:\n") == []


# The two test commands of a CI job, as its `run:` lines have them (docs/ROADMAP.md, W1 step 8).
FAST_TESTS = 'uv run pytest -m "not slow"'
SLOW_TESTS_WITHOUT_WEIGHTS = 'uv run pytest -m "slow and not weights" tests/slow'


def _job_steps(workflow: str) -> dict[str, list[str]]:
    """The jobs of a workflow text with their steps, {job: [the text of each step]}, in the order of
    the text. A job is a key two spaces deep below the line `jobs:`; a step begins with `- ` six spaces
    deep and goes on to the next step or job. The text is read as lines, not as YAML, so it has to be
    laid out as this repository's workflow is."""
    jobs: dict[str, list[str]] = {}
    steps, in_jobs = None, False
    for line in workflow.splitlines():
        if re.match(r"[\w\"']", line):  # a key at the left edge
            steps, in_jobs = None, line.rstrip() == "jobs:"
        elif in_jobs and (job := re.fullmatch(r"  ([\w-]+):\s*", line)):
            steps = jobs.setdefault(job[1], [])
        elif steps is not None and line.startswith("      - "):
            steps.append(line)
        elif steps:
            steps[-1] += "\n" + line
    return jobs


def _runs(step: str, command: str) -> bool:
    """True if the step's `run:` line is this command and nothing else: a command in a block of several
    lines or behind a `#` does not count."""
    return re.search(rf"^ +(- )?run: {re.escape(command)} *$", step, re.MULTILINE) is not None


def _test_step_gaps(workflow: str) -> list[str]:
    """What the jobs of a workflow text lack, one line each: a job without a step that runs the fast
    tests, without a step that runs the slow tests without weights, with that step before the fast
    tests, or with that step not offline (`HF_HUB_OFFLINE: "1"` among the lines of the step)."""
    found = []
    for job, steps in _job_steps(workflow).items():
        fast = [number for number, step in enumerate(steps) if _runs(step, FAST_TESTS)]
        slow = [number for number, step in enumerate(steps) if _runs(step, SLOW_TESTS_WITHOUT_WEIGHTS)]
        if not fast:
            found.append(f"{job}: no step runs {FAST_TESTS}")
        if not slow:
            found.append(f"{job}: no step runs {SLOW_TESTS_WITHOUT_WEIGHTS}")
        if fast and slow and slow[0] < fast[0]:
            found.append(f"{job}: the slow tests run before the fast tests")
        if slow and not re.search(r'^ +HF_HUB_OFFLINE: "1" *$', steps[slow[0]], re.MULTILINE):
            found.append(f"{job}: the step of the slow tests does not set HF_HUB_OFFLINE")
    return found


def test_every_ci_job_runs_the_fast_tests_and_then_the_slow_tests_without_weights():
    # The use of this test: nobody drops one of the two steps from one job unnoticed. It reads the file it
    # guards, so it does not show that a job passes; only a run on GitHub shows that. The second step is
    # offline, so that a test that reaches for the model fails there instead of downloading it
    # (tests/test_slow_selection.py holds which tests the step selects).
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert list(_job_steps(workflow)) == ["ubuntu", "windows", "macos"]
    assert _test_step_gaps(workflow) == []

    # The check finds each gap in a made-up workflow, and none in a job that has both steps.
    def job(name, *steps):
        return [f"  {name}:", "    runs-on: ubuntu-latest", "    steps:", "      - uses: actions/checkout@v7", *steps]

    fast = ["      - name: Fast tests", f"        run: {FAST_TESTS}"]
    slow = ["      - name: Slow tests without weights", "        env:", '          HF_HUB_OFFLINE: "1"',
            f"        run: {SLOW_TESTS_WITHOUT_WEIGHTS}"]
    online = [slow[0], slow[3]]
    later = [*slow[:3], f"        # run: {SLOW_TESTS_WITHOUT_WEIGHTS}", "        run: echo later"]
    made_up = "\n".join([
        "name: tests", "on:", "  push:", "    branches: [main]", "jobs:",
        *job("whole", *fast, *slow),
        *job("no-slow", *fast),
        *job("no-fast", *slow),
        *job("online", *fast, *online),
        *job("slow-first", *slow, *fast),
        *job("commented", *fast, *later),
        ""])
    assert list(_job_steps(made_up)) == ["whole", "no-slow", "no-fast", "online", "slow-first", "commented"]
    assert [len(steps) for steps in _job_steps(made_up).values()] == [3, 2, 2, 3, 3, 3]  # with the checkout
    assert _test_step_gaps(made_up) == [
        f"no-slow: no step runs {SLOW_TESTS_WITHOUT_WEIGHTS}",
        f"no-fast: no step runs {FAST_TESTS}",
        "online: the step of the slow tests does not set HF_HUB_OFFLINE",
        "slow-first: the slow tests run before the fast tests",
        f"commented: no step runs {SLOW_TESTS_WITHOUT_WEIGHTS}",
    ]


# ---------------------------------------------------------------------------------------------
# How the tests share helpers and fixtures


def _conftest_imports(folder: Path) -> list[str]:
    """Scan every .py file under `folder`; return `file:line` for each import of a module named
    conftest, at any depth (the three ways of `_imported_top_levels`)."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [f"{path.relative_to(folder).as_posix()}:{line}"
                  for line, top in _imported_top_levels(tree) if top == "conftest"]
    return found


def test_no_module_under_tests_imports_conftest(tmp_path):
    # A conftest.py is pytest's: it holds fixtures, and pytest finds it by its folder. Python keeps one
    # module under that name, so with a second conftest.py below tests/ an import by that name gets the
    # wrong file in a run of the whole suite, while a run of the one file passes (measured with pytest
    # 9.1.1). What tests import by name is in a helper module.
    scanned = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("*.py")}
    assert {"conftest.py", "helpers.py", "test_tracker_io.py", "gui/gui_helpers.py"} <= scanned  # the real folder
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
    }
    for rel, text in sources.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert _conftest_imports(tmp_path) == ["gui/test_b.py:3", "helpers.py:2", "test_a.py:2"]


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
    assert {"conftest.py", "helpers.py", "gui/session_helpers.py", "gui/test_theme.py"} <= scanned  # the real folder
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
    module whose name begins with `test_`, at any depth (the three ways of `_imported_top_levels`)."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [f"{path.relative_to(folder).as_posix()}:{line}: imports {top}"
                  for line, top in _imported_top_levels(tree) if top.startswith("test_")]
    return found


def test_no_test_module_imports_from_a_test_module(tmp_path):
    # A test file is pytest's: it is collected and run, split when it grows long, renamed with what it
    # tests. A file that imports a name from it depends on all of that, and Python runs the whole test
    # file to hand out the one name. What a second file needs is a helper, and its home is a helper module.
    scanned = {path.relative_to(TESTS).as_posix() for path in TESTS.rglob("*.py")}
    assert {"helpers.py", "gui/test_theme.py", "slow/pipeline_helpers.py"} <= scanned  # the real folder
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
    "tests/gui/gui_helpers.py": ["QTest.qWait"],
    # Not a delay: the fixture `lock_file` has put its recorder in place of the wait, and the test that calls
    # it asserts that nothing was waited (`test_lock_file_skips_the_waits_between_the_tries_and_records_them`).
    "tests/test_fileio.py": ["fileio.time.sleep"],
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
    # (tests/gui/prompt_helpers.py, `Gate`).
    files = [REPO / "conftest.py", *sorted(TESTS.rglob("*.py"))]
    found = {path.relative_to(REPO).as_posix(): _delay_calls(path.read_text(encoding="utf-8")) for path in files}
    assert {"conftest.py", "tests/conftest.py", "tests/gui/gui_helpers.py", "tests/gui/test_worker_jobs.py",
            "tests/slow/conftest.py"} <= set(found)  # the real folders
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
        "from outline_tracker import fileio",
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
