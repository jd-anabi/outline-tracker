"""Rules of the repository itself: entry point, import boundaries, no private data, when CI runs, how
the tests share helpers and fixtures.

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

# A path under a home folder: the users folder of macOS or Windows followed by a name, or pytest's
# per-user temp folder. The patterns are put together from pieces so that this file, which is
# itself scanned, does not contain what it looks for.
_USERS = "Users"
_PYTEST_TMP = "pytest-of" + "-"
HOME_PATH = re.compile(rf"[/\\]{_USERS}[/\\]+[^/\\\s]|{_PYTEST_TMP}[^/\\\s]")


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
