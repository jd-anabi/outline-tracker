"""Rules of the repository itself: entry point, reference copies, import boundaries, no private data,
when CI runs.

Checks that a command "loads neither torch nor Qt" run that command in a subprocess: pytest-qt has
already imported PySide6 into the test process.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import outline_tracker

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "outline_tracker"
REFERENCE = REPO / "tests" / "reference"
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"

# Import boundaries (SPEC 12, CLAUDE.md). Top-level module names.
QT_MODULES = {"PySide6", "pyqtgraph"}  # only inside outline_tracker/gui/
MODEL_MODULES = {"torch", "torchvision", "transformers", "timm"}  # only segmenter/ and gui/app.py
REFERENCE_MODULES = {"shrimp"}  # tests/reference: never imported by the package
HEAVY_MODULES = sorted(QT_MODULES | MODEL_MODULES)

# SHA-256 of the ten provided template files, jd-anabi/shrimp-tracker-template at commit 4ec8cd7,
# computed from the bytes stored in git (their blob ids equal the template's at that commit).
REFERENCE_SHA256 = {
    "shrimp/__init__.py": "38f759fce1c80f2b1ac2c55b86d88e32372a51ba5a7f1fa91335aa69c78a145e",
    "shrimp/_edgetam.py": "781d6a1ff16c04a9f9dee96b2196cd14a052b0b901b371956a79c67cbfb49e25",
    "shrimp/check_video.py": "9c6b07844b1a8b2bbf8a36784a256d0a5fbb81eaa027e6b33cd114f5217ca6a9",
    "shrimp/convert.py": "179d742ace115025de263914ce54aad2c11e668d629d0389b69412626339d834",
    "shrimp/segment.py": "783795a464395688b1b9acfeeb7c3c6fbe11bc2954e40c74ee5c09aebf30a9e7",
    "shrimp/video.py": "a48b6e491e30ec53d1b1ca44d3ee2ffec6c117e4345638574cfd40a6d8ea20b5",
    "template_tests/conftest.py": "58cf7c466bd3182cb0213fe06967310445c6fdc74acef603a1d0a1a5841d127d",
    "template_tests/test_convert.py": "5e535a4f0141154879e8bbe01bbf8ea9fe8a1718d900ec3c79e1de24963f1d02",
    "template_tests/test_segment.py": "ad091d1037c9238ae984fb5b40e6a15be6a063eb0d8b1628922a68db0e5e202e",
    "template_tests/test_video.py": "a01a0844fa5727279177274a9f11ada970a85e04c6f8bef7b35ea7cb72faaf4d",
}

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
# Reference copies (last week's template files)


def test_reference_copies_unmodified():
    # Bytes as checked out: .gitattributes (`tests/reference/** -text`) keeps them LF on Windows.
    actual = {rel: hashlib.sha256((REFERENCE / rel).read_bytes()).hexdigest() for rel in REFERENCE_SHA256}
    assert actual == REFERENCE_SHA256
    # Nothing was added next to them either.
    present = {p.relative_to(REFERENCE).as_posix() for p in REFERENCE.rglob("*.py")}
    assert present == set(REFERENCE_SHA256)
    # tests/conftest.py is the template's helper file, unchanged (the ported tests import from it).
    ours = hashlib.sha256((REPO / "tests" / "conftest.py").read_bytes()).hexdigest()
    assert ours == REFERENCE_SHA256["template_tests/conftest.py"]


def test_template_tests_pass_on_reference():
    # X2: last week's 27 tests against last week's code, with this week's libraries. A separate
    # pytest process, because the two test sets share file names.
    command = [sys.executable, "-m", "pytest", "tests/reference/template_tests", "-q", "--color=no"]
    done = _run([*command, "-p", "no:cacheprovider"], timeout=600)  # leave the outer run's cache alone
    lines = done.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    assert done.returncode == 0, done.stdout + done.stderr
    # Exactly 27 passed: no failure, error, skip or xfail next to them (warnings are tolerated).
    assert re.fullmatch(r"27 passed(, \d+ warnings?)? in \d.*", summary), done.stdout + done.stderr


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
            if (
                top in REFERENCE_MODULES
                or (top in QT_MODULES and not qt_allowed)
                or (top in MODEL_MODULES and not model_allowed)
            ):
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
        "video.py": "from shrimp.video import probe\n",
        "tracking.py": "import importlib\n\nhub = importlib.import_module('transformers.utils')\n",
        # relative imports and a module whose name only starts like the reference package are fine
        "measure.py": "import timm.models\nfrom . import schema\nfrom .segmenter import hf\nimport shrimps\n",
        "derive.py": "import torchvision\n",
        "segmenter/__init__.py": "",
        "segmenter/hf.py": "import torch\nimport transformers\nimport timm\nimport torchvision\n",
        "segmenter/fake.py": "from PySide6 import QtCore\n",
        "segmenter/edgetam_convert.py": "import torch\nimport shrimp\n",
        "gui/__init__.py": "",
        "gui/app.py": "import torch\nimport PySide6.QtWidgets\nimport pyqtgraph\n",
        "gui/worker.py": "from PySide6 import QtCore\nimport torch\n",
        "gui/panels/plot.py": "from PySide6 import QtWidgets\nimport pyqtgraph as pg\n",
        "gui/panels/flags.py": "from shrimp import segment\n",
    }
    for rel, text in sources.items():
        target = tmp_path / "pkg" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    expected = {
        ("cli.py", 2, "torch"),
        ("export.py", 2, "PySide6"),
        ("overlay.py", 1, "pyqtgraph"),
        ("video.py", 1, "shrimp"),
        ("tracking.py", 3, "transformers"),
        ("measure.py", 1, "timm"),
        ("derive.py", 1, "torchvision"),
        ("segmenter/fake.py", 1, "PySide6"),
        ("segmenter/edgetam_convert.py", 2, "shrimp"),
        ("gui/worker.py", 2, "torch"),
        ("gui/panels/flags.py", 1, "shrimp"),
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
        "tests/reference/shrimp/video.py",
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
