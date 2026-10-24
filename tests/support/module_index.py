"""The index of the modules of the package, docs/developer/modules.md, generated from their docstrings.

`module_index(package)` reads every .py file below the folder `package` (by default `outline_tracker/`) with
`ast`, takes the first line of the module docstring, and returns the text of the page: one section for each
folder, in the order of the layers (`FOLDERS`), with the first line of the folder's `__init__.py` under the
heading and a table `| module | what it holds |` of its modules. The package is read, never imported.

Run as a script, it writes the page: `uv run python tests/support/module_index.py docs/developer/modules.md`
(`COMMAND`). tests/repo/test_module_index.py compares the committed page with this text, byte for byte, and
fails with that command when they differ, as tests/core/test_schema_docs.py does for docs/OUTPUTS.md.

A module without a docstring, or whose first line is empty, makes the generator raise a `ValueError` that
names the module: every module has a first line. So does a folder that is not in `FOLDERS` (a new subpackage
is placed in the order of the layers on purpose) and a folder with modules and no `__init__.py`.

The text has LF line ends and one line end at its end, and holds no date and no name of a machine, so the same
package gives the same text every time. No quantities here.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from layout_helpers import PACKAGE_NAME, PATHS, REPO

# The folders of the package under `outline_tracker/`, in the order of the layers: the base first, the command
# line last. "" is `outline_tracker/` itself, with `__init__.py` and `launch.py`.
FOLDERS = ("", "core", "video", "measure", "segmenter", "tracking", "synthetic", "export", "workflows",
           "gui", "gui/view", "gui/widgets", "gui/panels", "cli")

COMMAND = f"uv run python {PATHS['module index generator']} {PATHS['module index page']}"  # writes the page
TITLE = "Outline Tracker: the modules"


def _first_line(file: Path, shown: str) -> str:
    """The first line of the module docstring of `file`, without the spaces at its ends. `shown` names the
    module in an error. Raises `ValueError` when there is no docstring or when its first line is empty."""
    docstring = ast.get_docstring(ast.parse(file.read_text(encoding="utf-8"), filename=shown), clean=False)
    if docstring is None:
        raise ValueError(f"{shown}: the module has no docstring; every module starts with one")
    lines = docstring.splitlines()
    first = lines[0].strip() if lines else ""
    if not first:
        raise ValueError(f"{shown}: the first line of the docstring is empty; every module has a first line")
    return first


def _cell(text: str) -> str:
    """`text` as the content of a cell of a table: a bar would end the cell."""
    return text.replace("|", "\\|")


def module_index(package: Path | None = None) -> str:
    """The text of docs/developer/modules.md for the package in the folder `package` (default: `outline_tracker/`
    of this repository). Raises `ValueError` as the module's text says."""
    package = REPO / PACKAGE_NAME if package is None else package

    def shown(file: Path) -> str:  # the module in an error: `outline_tracker/core/x.py`
        return file.relative_to(package.parent).as_posix()

    by_folder: dict[str, list[Path]] = {}
    for file in package.rglob("*.py"):
        folder = file.parent.relative_to(package).as_posix()
        by_folder.setdefault("" if folder == "." else folder, []).append(file)
    unplaced = sorted(set(by_folder) - set(FOLDERS))
    if unplaced:
        raise ValueError(f"{package.name}/{unplaced[0]} is not in the order of the layers: add it to FOLDERS in "
                         f"{PATHS['module index generator']}")
    out = [
        f'<!-- Generated from the docstrings of {package.name}/ by "{COMMAND}". Do not edit by hand. -->',
        "",
        f"# {TITLE}",
        "",
        f"Every module of the package `{package.name}/` with the first line of its docstring, in the order of the "
        f"layers, the base first; the paths are relative to `{package.name}/`.",
    ]
    for folder in FOLDERS:
        if folder not in by_folder:
            continue
        init = package / folder / "__init__.py"
        if not init.is_file():
            raise ValueError(f"{shown(init)}: the folder has modules and no __init__.py, which gives its section "
                             "its sentence")
        # `__init__.py` first, then the others by path
        files = sorted(by_folder[folder], key=lambda file: (file.name != "__init__.py", file.name))
        out += ["", f"## `{folder + '/' if folder else package.name + '/'}`", "", _first_line(init, shown(init)), "",
                "| module | what it holds |", "|---|---|"]
        out += [f"| `{file.relative_to(package).as_posix()}` | {_cell(_first_line(file, shown(file)))} |"
                for file in files]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """`python tests/support/module_index.py OUTPUT.md` writes `module_index()` to that file as UTF-8 with LF line
    ends on every platform. Returns the exit code: 0, or 2 (with a usage line) for a wrong call."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(f"usage: python {PATHS['module index generator']} OUTPUT.md", file=sys.stderr)
        return 2
    Path(args[0]).write_bytes(module_index().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
