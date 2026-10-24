"""Where the real model's factory is: `load_segmenter` is defined in segmenter/load.py and nowhere else (W2 step 4d).

The factory makes the Hugging Face segmenter, so it belongs to segmenter/. It was in workflows/from_tracker.py, and
the window took it from there. The rule test of the layers (test_layers.py) cannot hold this, because the window may
import workflows. So this test holds the place: the function says that it lives in segmenter/load.py, and no other
module of the package defines a function of that name. A second one would be a second factory, and the worker tells
the real one from a test's stand-in by identity (`before_load_for` in gui/worker.py).

The scan is checked on a made-up package in `tmp_path`, so that a scan that finds nothing would show up here.
A module is named by its dotted name; no units or coordinates.
"""

import ast
from pathlib import Path

from layout_helpers import PACKAGE_NAME, REPO, module

PACKAGE = REPO / PACKAGE_NAME


def _defined_outside(package: Path, subpackage: str, name: str) -> list[str]:
    """`file:line` of every function called `name`, at any depth (also in a function or a class), in the modules of
    the package in the folder `package` that are not in its subpackage `subpackage`."""
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.relative_to(package).parts[0] == subpackage:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [f"{path.relative_to(package.parent).as_posix()}:{node.lineno}" for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    return found


def test_the_real_models_factory_is_defined_in_segmenter_load_and_nowhere_else(tmp_path):
    from outline_tracker.segmenter.load import load_segmenter

    assert load_segmenter.__module__ == module("model factory")
    assert _defined_outside(PACKAGE, "segmenter", "load_segmenter") == []
    # the scan on a made-up package: it finds a definition at any depth, and leaves the home and an import alone
    made_up = tmp_path / PACKAGE_NAME
    for rel, text in {
        "segmenter/load.py": "def load_segmenter(model, device):\n    return model\n",  # the home
        "workflows/from_tracker.py": "from outline_tracker.segmenter.load import load_segmenter\n",  # an import
        "core/a.py": "import os\n\n\ndef load_segmenter():\n    return os\n",
        "gui/b.py": "def late():\n    def load_segmenter():\n        pass\n    return load_segmenter\n",
        "cli/c.py": "class Factory:\n    async def load_segmenter(self):\n        pass\n",
        "launch.py": "load_segmenter = None\n",  # a name, not a function
    }.items():
        (made_up / rel).parent.mkdir(parents=True, exist_ok=True)
        (made_up / rel).write_text(text, encoding="utf-8")
    assert _defined_outside(made_up, "segmenter", "load_segmenter") == [
        f"{PACKAGE_NAME}/cli/c.py:2", f"{PACKAGE_NAME}/core/a.py:4", f"{PACKAGE_NAME}/gui/b.py:2"]
