"""Where the files are: the one table of paths that the tests name.

A test that needs a file of the package, a test file, a helper module or a document asks this table by
role (`path("command line")`, `module("schema docs")`) and does not write the path itself. So a move
of files changes this table and the import lines, and no test.

`PATHS` has every path that a test names and that a move might change; paths no move touches
(the README, `pyproject.toml`, the data of the tests) stay in the tests that read them. Each path is
relative to `REPO` and written with `/`.

`REPO` is found by walking up from this file to the first folder that holds pyproject.toml, so the
file can lie at any depth. A test never takes a folder of the repository from its own `__file__`
(tests/test_layout.py checks that); this is the one place that may.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

PACKAGE_NAME = "outline_tracker"  # the name of the package, and of its folder at the top of the repository

PATHS: dict[str, str] = {
    # the package
    "command line": "outline_tracker/cli/main.py",
    "schema docs": "outline_tracker/core/schema_docs.py",
    "window package": "outline_tracker/gui/__init__.py",
    "window entry point": "outline_tracker/gui/app.py",
    "main window": "outline_tracker/gui/main_window.py",
    "panel frame": "outline_tracker/gui/panel.py",
    "theme": "outline_tracker/gui/theme.py",
    "prompts": "outline_tracker/gui/prompts.py",
    "review table": "outline_tracker/gui/review_table.py",
    "worker": "outline_tracker/gui/worker.py",
    "worker engine": "outline_tracker/gui/worker_engine.py",
    "worker jobs": "outline_tracker/gui/worker_jobs.py",
    "video panel": "outline_tracker/gui/panels/video_panel.py",
    "tracking panel": "outline_tracker/gui/panels/track_panel.py",
    "export panel": "outline_tracker/gui/panels/export_panel.py",
    # the tests and their helpers
    "layout helpers": "tests/layout_helpers.py",
    "test helpers": "tests/helpers.py",
    "tracker io test": "tests/test_tracker_io.py",
    "fileio test": "tests/test_fileio.py",
    "window helpers": "tests/gui/gui_helpers.py",
    "session helpers": "tests/gui/session_helpers.py",
    "theme test": "tests/gui/test_theme.py",
    "worker jobs test": "tests/gui/test_worker_jobs.py",
    "slow test helpers": "tests/slow/pipeline_helpers.py",
    # documents
    "spec": "SPEC.md",
    "developer page": "docs/DEVELOPER.md",
    "agent rules": "CLAUDE.md",
    "roadmap": "docs/ROADMAP.md",
}


def _repo_above(file: Path) -> Path:
    """The first folder above `file` that holds pyproject.toml."""
    for folder in file.resolve().parents:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise RuntimeError(f"no folder above {file.name} holds pyproject.toml, so the repository cannot be found")


REPO: Path = _repo_above(Path(__file__))


def path(role: str) -> Path:
    """The file or folder of this role, as a path in the checkout. Raises `KeyError` for a role that
    the table does not have."""
    return REPO / PATHS[role]


def module(role: str) -> str:
    """The dotted name of the package module at the path of this role: `outline_tracker/gui/worker.py`
    is `outline_tracker.gui.worker`, and a path that ends in `__init__.py` names its package. Raises
    `ValueError` for a path that is not a .py file below `outline_tracker/`."""
    parts = PurePosixPath(PATHS[role]).parts
    if len(parts) < 2 or parts[0] != PACKAGE_NAME or not parts[-1].endswith(".py"):
        raise ValueError(f"{PATHS[role]} (role {role!r}) is not a module of the package {PACKAGE_NAME}")
    stem = parts[-1].removesuffix(".py")
    return ".".join(parts[:-1] if stem == "__init__" else (*parts[:-1], stem))


def below(folder: Path, *roles: str) -> set[str]:
    """The paths of these roles as names below `folder`, with `/`: the form that a scan of `folder` gives
    (`below(REPO / "outline_tracker", "worker")` is `{"gui/worker.py"}`). Raises `ValueError` for a path
    that does not lie below `folder`."""
    return {path(role).relative_to(folder).as_posix() for role in roles}
