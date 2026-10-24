"""Where the files are: the one table of paths that the tests name, and the scan of the links between
documents.

A test that needs a file of the package, a test file, a helper module or a document asks this table by
role (`path("command line")`, `module("schema docs")`) and does not write the path itself. So a move
of files changes this table and the import lines, and no test.

`PATHS` has every path that a test names and that a move might change; paths no move touches
(the README, `pyproject.toml`, the data of the tests) stay in the tests that read them. Each path is
relative to `REPO` and written with `/`. Two entries are there for another reason: `CLAUDE.md` and
`docs/ROADMAP.md` are not moved, but the developer page is, and the link it writes to each of them is
computed between two entries of the table (`link_to` in tests/gui/test_finish.py).

`REPO` is found by walking up from this file to the first folder that holds pyproject.toml, so the
file can lie at any depth. A test never takes a folder of the repository from its own `__file__`
(tests/repo/test_layout.py checks that); this is the one place that may.

`dead_links` reads the relative links `[text](target)` of a Markdown text. tests/repo/test_layout.py
runs it over every tracked document and tests/repo/test_readme.py over the README.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

PACKAGE_NAME = "outline_tracker"  # the name of the package, and of its folder at the top of the repository

PATHS: dict[str, str] = {
    # the package
    "command line": "outline_tracker/cli/main.py",
    "schema docs": "outline_tracker/core/schema_docs.py",
    "window package": "outline_tracker/gui/__init__.py",
    "window entry point": "outline_tracker/gui/app.py",
    "main window": "outline_tracker/gui/main_window.py",
    "panel frame": "outline_tracker/gui/widgets/panel_frame.py",
    "theme": "outline_tracker/gui/theme.py",
    "prompts": "outline_tracker/gui/view/prompts.py",
    "review table": "outline_tracker/gui/panels/review_table.py",
    "model factory": "outline_tracker/segmenter/load.py",
    "worker": "outline_tracker/gui/worker.py",
    "worker engine": "outline_tracker/gui/worker_engine.py",
    "worker jobs": "outline_tracker/gui/worker_jobs.py",
    "video panel": "outline_tracker/gui/panels/video_panel.py",
    "tracking panel": "outline_tracker/gui/panels/track_panel.py",
    "export panel": "outline_tracker/gui/panels/export_panel.py",
    # the tests and their helpers
    "layout helpers": "tests/support/layout_helpers.py",
    "test helpers": "tests/support/helpers.py",
    "tracker io test": "tests/core/test_tracker_io.py",
    "fileio test": "tests/core/test_fileio.py",
    "window helpers": "tests/support/gui_helpers.py",
    "session helpers": "tests/support/session_helpers.py",
    "tracking helpers": "tests/support/tracking_helpers.py",
    "theme test": "tests/gui/test_theme.py",
    "worker jobs test": "tests/gui/test_worker_jobs.py",
    "slow test helpers": "tests/support/pipeline_helpers.py",
    # documents
    "spec": "docs/design/SPEC-v0.1.md",
    "developer page": "docs/developer/README.md",
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


# ---------------------------------------------------------------------------------------------
# Links between documents

FENCES = ("```", "~~~")  # the two kinds of fence that open a block of code
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s[^)]*)?\)")  # `[text](target)`; the text may wrap, a title may follow
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")  # `https:`, `mailto:`
CODE_SPAN = re.compile(r"`(?:[^`\n]|\n(?![ \t]*\n))*`")  # `inline code`, also over a line end, never over a blank line


def outside_code(text: str) -> str:
    """`text` without its fenced blocks (``` and ~~~; a fence is closed by its own kind) and without its
    `inline code`. Every line end stays, so that a line number counted in the result is the line in `text`."""
    kept, fence = [], None
    for line in text.splitlines():
        marker = line.lstrip()[:3]
        if fence is None and marker in FENCES:
            fence = marker  # a block opens
            kept.append("")
        elif fence is not None:
            if marker == fence:
                fence = None  # and closes
            kept.append("")
        else:
            kept.append(line)
    return CODE_SPAN.sub(lambda span: "\n" * span.group().count("\n"), "\n".join(kept))


def dead_links(text: str, folder: Path) -> list[tuple[int, str]]:
    """`(line, target)` for every relative link `[text](target)` in the Markdown `text` whose target is not
    a file or folder below `folder`, the folder of the document that holds the link; `line` is the line
    on which the target is written. An anchor (`#...`) is cut off from the target. A link with a scheme
    (`https:`, `mailto:`) and a link that is only an anchor are skipped. Code is not read: neither fenced
    blocks nor `inline code`. The text of a link may wrap onto the next line, and a link may have a title."""
    plain = outside_code(text)
    found = []
    for link in LINK.finditer(plain):
        target = link.group(1)
        local = target.split("#")[0]
        if local and not SCHEME.match(target) and not (folder / local).exists():
            found.append((plain.count("\n", 0, link.start(1)) + 1, local))
    return found
