"""Rules of the repository itself: entry point, import boundaries, no private data, when CI runs and
which tests each of its jobs runs.

Checks that a command "loads neither torch nor Qt" run that command in a subprocess: pytest-qt has
already imported PySide6 into the test process.

The rules about the test files (how they share helpers and fixtures, and that no test waits by a delay) are in
tests/repo/test_repo_rules_test_code.py.
"""

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from helpers import HOME_PATH, PYTEST_TMP, USERS, VERSION
from import_helpers import imports
from layout_helpers import REPO, below, module

import outline_tracker

PACKAGE = REPO / "outline_tracker"
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"

# Import boundaries (SPEC 12, CLAUDE.md). Top-level module names.
QT_MODULES = {"PySide6", "pyqtgraph"}  # only inside outline_tracker/gui/
MODEL_MODULES = {"torch", "torchvision", "transformers", "timm"}  # only segmenter/ and the window's entry point
MODEL_ENTRY_POINTS = below(PACKAGE, "window entry point")  # it imports torch before PySide6 (CLAUDE.md)
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
    assert done.stdout.startswith(f"outline-tracker {VERSION} (commit "), done.stdout
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
        "pyproject.toml": VERSION,
        "outline_tracker/__init__.py": VERSION,
        "uv.lock": [VERSION],
    }


# ---------------------------------------------------------------------------------------------
# Import boundaries


def _imported_top_levels(tree: ast.AST):
    """Yield (line, top-level module name) for every absolute import in a parsed file (`imports`)."""
    for line, module, _ in imports(tree):
        yield line, module.split(".")[0]


def _import_violations(package: Path) -> list[tuple[str, int, str]]:
    """Scan every .py file under `package`; return (file, line, module) for each forbidden import."""
    found = []
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package)
        qt_allowed = rel.parts[0] == "gui"
        model_allowed = rel.parts[0] == "segmenter" or rel.as_posix() in MODEL_ENTRY_POINTS
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, top in _imported_top_levels(tree):
            if (top in QT_MODULES and not qt_allowed) or (top in MODEL_MODULES and not model_allowed):
                found.append((rel.as_posix(), line, top))
    return found


def test_import_boundaries():
    scanned = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*.py")}
    assert {"__init__.py", *below(PACKAGE, "command line")} <= scanned  # the scan looks at the real package
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
    assert {"outline_tracker", module("command line")} <= set(modules)
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
        "/".join(["", USERS, name, "Movies", "clip.mp4"]),  # macOS
        "video: " + "/".join(["", USERS, name]),
        "\\".join(["C:", USERS, name, "Videos"]),  # Windows
        "\\\\".join(["C:", USERS, name, "Videos"]),  # Windows, backslashes escaped (JSON, repr)
        "/".join(["C:", USERS, name]),  # Windows, forward slashes
        "/private/var/folders/ab/cd/T/" + PYTEST_TMP + name + "/pytest-0/clip.mp4",
    ]
    generic = [
        "~/.cache/shrimp-models/edgetam",
        "$HOME/.venvs/outline-tracker",
        "%USERPROFILE%\\.local\\bin",
        "https://github.com/jd-anabi/outline-tracker",
        "the " + USERS + " folder",
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
    # Fast tests read documents (README.md, docs/), so a push that changes only documents must
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
    # (tests/repo/test_slow_selection.py holds which tests the step selects).
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
