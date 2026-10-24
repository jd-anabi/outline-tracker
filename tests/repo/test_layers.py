"""The layers of the package: which subpackage may import which, kept as a rule of the repository.

The rule comes from docs/roadmap/repo.md section 3 ("Allowed direction") and from the decisions of W2 in
docs/ROADMAP.md. `RANK` is its table, from the bottom up:

- a module may import a module of its own subpackage, or of a subpackage of lower rank; two subpackages of
  the same rank do not import each other;
- `outline_tracker/__init__.py` (the version) may be imported by every module and imports nothing;
- inside gui/, a module of gui/panels/ imports no other module of gui/panels/, and a module of gui/ outside
  gui/panels/ imports no module of gui/panels/. The package `gui.panels` itself (`PANEL_MODULES`,
  `build_bodies`) may be imported by any of them.

Every import counts, at the top of a file and inside a function: `import a.b`, `from a.b import c` (where c may
be a module) and `import_module("a.b")` with a literal name (`imports` of tests/support/import_helpers.py).
A name that is not a literal cannot be read; and a relative import would not be read either, so a test below
holds that the package has none.

Some imports go against the rule today. `EXCEPTIONS` names each one with the step or workstream that removes
it. The scan of the real package must find exactly those: a new one fails, and so does an exception that no
longer occurs, so the commit that removes an import removes its entry too.

The scans are checked on small made-up packages in `tmp_path`, so that a scan that finds nothing would show up
here. A module is named by its dotted name; no units or coordinates.
"""

import ast
import re
from pathlib import Path

from import_helpers import imports
from layout_helpers import PACKAGE_NAME, REPO, module

PACKAGE = REPO / PACKAGE_NAME

# The layers, from the bottom: a subpackage (or the module launch.py) and its rank.
RANK = {
    "core": 0,
    "video": 1,
    "measure": 2,
    "segmenter": 2,
    "tracking": 3,
    "synthetic": 3,
    "export": 4,
    "workflows": 5,
    "gui": 6,
    "cli": 7,  # its `gui` command opens the window
    "launch": 8,
}
ROOT_RANK = -1  # outline_tracker/__init__.py: below every layer, so it imports nothing of the package

# The imports against the rule today: {(importing module, imported module): "what it is; removed by ..."}.
# Each commit that removes one of them removes its entry.
EXCEPTIONS: dict[tuple[str, str], str] = {
    # measurement and the segmenters depend on each other by topic
    ("outline_tracker.measure.mask", "outline_tracker.segmenter.base"):
        "takes MaskResult from the segmenters' base; removed by W6 step 2 (the registry of methods)",
    ("outline_tracker.segmenter.hf", "outline_tracker.measure.mask"):
        "takes mask_center from measurement; removed by W6 step 2 (the registry of methods)",
    ("outline_tracker.segmenter.fake", "outline_tracker.measure.mask"):
        "takes mask_center from measurement; removed by W6 step 2 (the registry of methods)",
    ("outline_tracker.segmenter.fake", "outline_tracker.synthetic.clips"):
        "the stand-in of the tests names GroundTruth, in an annotation only; removed by W6 step 2"
        " (the registry of methods)",
    # the window: a panel takes the small parts it shares from another panel
    ("outline_tracker.gui.panels.export_panel", "outline_tracker.gui.panels.track_panel"):
        "takes PRIMARY_HEIGHT; removed by W2 step 4e",
}


# ---------------------------------------------------------------------------------------------
# The scan


def _name_of(path: Path, package: Path) -> str:
    """The dotted name of the module in the file `path` of the package in the folder `package`:
    `outline_tracker/core/schema.py` is `outline_tracker.core.schema`, and an `__init__.py` names its folder."""
    parts = path.relative_to(package.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _module_names(package: Path) -> set[str]:
    """The dotted name of every module of the package in the folder `package`."""
    return {_name_of(path, package) for path in package.rglob("*.py")}


def _imported_modules(tree: ast.AST, names: set[str]):
    """Yield (line, module) for every module of the package that a parsed file imports; `names` are the modules
    of the package. `from a.b import c` imports a.b.c when that is a module, else a.b; a name that is not
    a module is cut back to the longest module it starts with. Names of other packages are left out."""
    for line, dotted, from_names in imports(tree):
        for name in from_names or (None,):
            parts = (f"{dotted}.{name}" if name else dotted).split(".")
            while parts and ".".join(parts) not in names:
                parts.pop()
            if parts:
                yield line, ".".join(parts)


def _is_panel(parts: list[str]) -> bool:
    """Whether the module (its parts below the package) is a module of gui/panels/, not the package itself."""
    return parts[:2] == ["gui", "panels"] and len(parts) > 2


def _rank(parts: list[str]) -> int:
    return RANK[parts[0]] if parts else ROOT_RANK


def _why_not(importer: str, imported: str) -> str | None:
    """Why the module `importer` may not import the module `imported`, or None if it may."""
    mine, theirs = importer.split(".")[1:], imported.split(".")[1:]  # the parts below the package
    if not theirs:
        return None  # the version
    if mine[:1] != theirs[:1]:
        mine_name, their_name = (mine or [PACKAGE_NAME])[0], theirs[0]
        if _rank(theirs) > _rank(mine):
            return f"{mine_name} (rank {_rank(mine)}) imports {their_name} (rank {_rank(theirs)}), which is above it"
        if _rank(theirs) == _rank(mine):
            return f"{mine_name} and {their_name} have the same rank ({_rank(mine)})"
        return None
    if _is_panel(theirs) and mine != theirs:
        if _is_panel(mine):
            return "a panel imports another panel"
        if mine[:2] != ["gui", "panels"]:
            return "a module of gui/ outside gui/panels/ imports a panel"
    return None


def _breaks(package: Path) -> dict[tuple[str, str], str]:
    """{(importing module, imported module): "file:line: why"} for every import in the package in the folder
    `package` that goes against the rule; the first line, if a file imports the module more than once."""
    names = _module_names(package)
    found = {}
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        importer = _name_of(path, package)
        for line, imported in sorted(_imported_modules(tree, names)):
            why = _why_not(importer, imported)
            if why:
                found.setdefault((importer, imported),
                                 f"{path.relative_to(package.parent).as_posix()}:{line}: {why}")
    return found


def _relative_imports(package: Path) -> list[str]:
    """`file:line` for every relative import (`from . import x`, `from .y import z`) in the package."""
    found = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = sorted(node.lineno for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0)
        found += [f"{path.relative_to(package.parent).as_posix()}:{line}" for line in lines]
    return found


def _breaks_of_the_real_package() -> dict[tuple[str, str], str]:
    # Without these the scan could look at an empty folder and pass.
    names = _module_names(PACKAGE)
    assert {PACKAGE_NAME, module("command line"), module("window package"), module("export panel")} <= names
    return _breaks(PACKAGE)


# ---------------------------------------------------------------------------------------------
# The real package


def test_no_import_goes_against_the_layers_except_the_ones_named_in_exceptions():
    found = _breaks_of_the_real_package()
    new = sorted(f"{where}: {importer} -> {imported}"
                 for (importer, imported), where in found.items() if (importer, imported) not in EXCEPTIONS)
    assert new == [], "imports against the layers that EXCEPTIONS does not name"


def test_every_named_exception_still_occurs():
    found = _breaks_of_the_real_package()
    gone = sorted(f"{importer} -> {imported}" for importer, imported in EXCEPTIONS if (importer, imported) not in found)
    assert gone == [], "exceptions that no longer occur: remove their entries"


def test_the_tables_are_complete_and_every_exception_names_who_removes_it():
    top_level = {name.split(".")[1] for name in _module_names(PACKAGE) if "." in name}
    # every subpackage and top-level module has a rank, and no rank is for nothing
    assert top_level == set(RANK), "RANK and the folders of the package differ"
    no_remover = sorted(f"{importer} -> {imported}" for (importer, imported), reason in EXCEPTIONS.items()
                        if not re.search(r"; removed by W\d step \d+[a-z]?\b", reason))
    assert no_remover == []


def test_no_module_of_the_package_imports_relatively(tmp_path):
    # `imports` reads absolute imports only. A relative import would pass the rule unseen.
    assert module("command line") in _module_names(PACKAGE)  # the real package
    assert _relative_imports(PACKAGE) == []
    made_up = tmp_path / PACKAGE_NAME
    (made_up / "core").mkdir(parents=True)
    (made_up / "__init__.py").write_text("")
    (made_up / "core" / "__init__.py").write_text("from . import fileio\n")
    (made_up / "core" / "schema.py").write_text("\n".join([
        "import os", "from outline_tracker.core import fileio", "", "", "def late():", "    from .. import core",
        "    return core", ""]))
    (made_up / "core" / "fileio.py").write_text("from .schema import late\n")
    assert _relative_imports(made_up) == [
        f"{PACKAGE_NAME}/core/__init__.py:1",
        f"{PACKAGE_NAME}/core/fileio.py:1",
        f"{PACKAGE_NAME}/core/schema.py:6",
    ]


# ---------------------------------------------------------------------------------------------
# The scan, on made-up packages


def _write_package(root: Path, sources: dict[str, str]) -> Path:
    """The files `sources` (path below the package, text) as a package named like the real one in `root`."""
    for rel, text in sources.items():
        target = root / PACKAGE_NAME / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root / PACKAGE_NAME


def test_an_import_is_read_as_the_module_of_the_package_that_it_names():
    names = {PACKAGE_NAME, f"{PACKAGE_NAME}.core", f"{PACKAGE_NAME}.core.schema", f"{PACKAGE_NAME}.video",
             f"{PACKAGE_NAME}.video.reader", f"{PACKAGE_NAME}.gui", f"{PACKAGE_NAME}.gui.panels",
             f"{PACKAGE_NAME}.gui.panels.export_panel"}
    source = "\n".join([
        "import outline_tracker.core.schema",  # 1
        "import outline_tracker.video.reader as reader",  # 2
        "from outline_tracker.core.schema import RESULTS_VERSION",  # 3: a name in the module
        "from outline_tracker.gui.panels import PANEL_MODULES, export_panel",  # 4: the package, and a module of it
        "from outline_tracker import __version__, core",  # 5: the top module, and a package
        "import importlib",  # 6
        "found = importlib.import_module('outline_tracker.video.reader')",  # 7
        "import numpy.linalg",  # 8: another package
        "from os import path",  # 9
        "from . import schema",  # 10: relative, not read
        "def late():",  # 11
        "    from outline_tracker.gui.panels.export_panel import build",  # 12
        "    return outline_tracker.core.schema.RESULTS_VERSION",  # 13: not an import
    ])
    assert sorted(_imported_modules(ast.parse(source), names)) == [
        (1, "outline_tracker.core.schema"),
        (2, "outline_tracker.video.reader"),
        (3, "outline_tracker.core.schema"),
        (4, "outline_tracker.gui.panels"),
        (4, "outline_tracker.gui.panels.export_panel"),
        (5, "outline_tracker"),
        (5, "outline_tracker.core"),
        (7, "outline_tracker.video.reader"),
        (12, "outline_tracker.gui.panels.export_panel"),
    ]


def test_the_scan_finds_each_kind_of_import_against_the_rule(tmp_path):
    package = _write_package(tmp_path, {
        "__init__.py": "",
        "core/__init__.py": "",
        "video/__init__.py": "",
        "measure/__init__.py": "",
        "segmenter/__init__.py": "",
        "export/__init__.py": "",
        "workflows/__init__.py": "",
        "tracking/__init__.py": "",
        "gui/__init__.py": "",
        "gui/panels/__init__.py": "",
        "cli/__init__.py": "",
        "video/reader.py": "",
        "segmenter/base.py": "",
        "workflows/from_tracker.py": "",
        "gui/worker.py": "",
        "cli/export.py": "",
        "gui/panels/calibration_panel.py": "",
        "gui/panels/dish_panel.py": "",
        "gui/panels/time_panel.py": "",
        # against the rank, by `import` and by `from`
        "core/schema.py": "import os\nimport outline_tracker.video.reader\n",
        "gui/dialogs.py": "from outline_tracker.cli.export import size_text\n",
        # two subpackages of the same rank
        "measure/mask.py": "from outline_tracker.segmenter.base import MaskResult\n",
        # inside a function, and the module named by `from package import module`
        "export/pipeline.py": ("def run():\n    from outline_tracker.workflows import from_tracker\n"
                               "    return from_tracker\n"),
        # by `import_module`, with a literal name
        "tracking/job.py": "import importlib\n\nworker = importlib.import_module('outline_tracker.gui.worker')\n",
        # a panel imports a panel: by `from package import module`, and by the module's name
        "gui/panels/track_panel.py": "from outline_tracker.gui.panels import calibration_panel\n",
        "gui/panels/review_panel.py": "from outline_tracker.gui.panels.dish_panel import DISH\nimport os\n",
        # a view module imports a panel
        "gui/overlays.py": "from outline_tracker.gui.panels.time_panel import settings\n",
    })
    assert _breaks(package) == {
        ("outline_tracker.core.schema", "outline_tracker.video.reader"):
            "outline_tracker/core/schema.py:2: core (rank 0) imports video (rank 1), which is above it",
        ("outline_tracker.gui.dialogs", "outline_tracker.cli.export"):
            "outline_tracker/gui/dialogs.py:1: gui (rank 6) imports cli (rank 7), which is above it",
        ("outline_tracker.measure.mask", "outline_tracker.segmenter.base"):
            "outline_tracker/measure/mask.py:1: measure and segmenter have the same rank (2)",
        ("outline_tracker.export.pipeline", "outline_tracker.workflows.from_tracker"):
            "outline_tracker/export/pipeline.py:2: export (rank 4) imports workflows (rank 5), which is above it",
        ("outline_tracker.tracking.job", "outline_tracker.gui.worker"):
            "outline_tracker/tracking/job.py:3: tracking (rank 3) imports gui (rank 6), which is above it",
        ("outline_tracker.gui.panels.track_panel", "outline_tracker.gui.panels.calibration_panel"):
            "outline_tracker/gui/panels/track_panel.py:1: a panel imports another panel",
        ("outline_tracker.gui.panels.review_panel", "outline_tracker.gui.panels.dish_panel"):
            "outline_tracker/gui/panels/review_panel.py:1: a panel imports another panel",
        ("outline_tracker.gui.overlays", "outline_tracker.gui.panels.time_panel"):
            "outline_tracker/gui/overlays.py:1: a module of gui/ outside gui/panels/ imports a panel",
    }


def test_the_scan_leaves_the_allowed_imports_alone(tmp_path):
    package = _write_package(tmp_path, {
        "__init__.py": '__version__ = "1"\n',
        "core/__init__.py": "",
        "core/fileio.py": "",
        # the version, and a module of its own subpackage
        "core/schema.py": "from outline_tracker import __version__\nfrom outline_tracker.core import fileio\n",
        "video/__init__.py": "",
        "video/reader.py": "import numpy\nfrom outline_tracker.core.schema import RESULTS_VERSION\n",  # lower rank
        "measure/__init__.py": "",
        "measure/mask.py": "from outline_tracker.video import reader\nimport outline_tracker\n",
        "segmenter/__init__.py": "",
        "segmenter/hf.py": "from outline_tracker.video.reader import read\n",  # beside measure, both import below
        "tracking/__init__.py": "",
        "tracking/job.py": "from outline_tracker.measure import mask\nfrom outline_tracker.segmenter import hf\n",
        "export/__init__.py": "",
        "export/pipeline.py": "from outline_tracker.tracking.job import run\n",
        "workflows/__init__.py": "",
        "workflows/from_tracker.py": ("def late():\n    from outline_tracker.export import pipeline\n"
                                      "    return pipeline\n"),
        "gui/__init__.py": "",
        "gui/theme.py": "",
        "gui/panels/__init__.py": "",
        "gui/panels/track_panel.py": ("from outline_tracker.gui.panels import PANEL_MODULES, build_bodies\n"
                                      "from outline_tracker.gui import theme\nimport outline_tracker.core.schema\n"),
        "gui/main_window.py": ("import outline_tracker.gui.panels\n"
                               "from outline_tracker.gui.panels import build_bodies\n"),
        "cli/__init__.py": "",
        "cli/gui.py": ("from outline_tracker.gui import main_window\n"
                       "from outline_tracker.workflows import from_tracker\n"),
        "launch.py": "from outline_tracker.cli import gui\n",
    })
    assert _module_names(package) >= {"outline_tracker.gui.panels", "outline_tracker.launch"}  # the scan sees them
    assert _breaks(package) == {}
    # the same package with one import turned upward breaks, so that the empty answer above is not the scan's blindness
    (package / "core" / "fileio.py").write_text("import outline_tracker.launch\n")
    assert list(_breaks(package)) == [("outline_tracker.core.fileio", "outline_tracker.launch")]

