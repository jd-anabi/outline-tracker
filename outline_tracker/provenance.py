"""Which tool wrote a file, and on what machine: the version line and the facts of run.log (SPEC 8.9).

`tool_version()` is the line `outline-tracker 0.1.0 (commit abc1234)`, printed by `--version` and
written into run.log and outlines.npz. Where the commit comes from, in this order:
1. `direct_url.json` in the installed package's metadata, when the tool was installed from git
   (`uv tool install git+https://...`): an installed tool has no .git, and the installer wrote the
   commit there (`vcs_info.commit_id`).
2. `git rev-parse --short HEAD`, when the package's source folder is a tracked part of a git
   checkout (a developer's editable install). A package that merely lies inside some checkout, such
   as one in a virtual environment inside a student's repository, is not that: its files are not
   tracked there, and the commit of that repository is not the tool's.
3. Else `(commit unknown)`.
The line holds the version and a hexadecimal commit and nothing else: an editable install's
`direct_url.json` holds a `file://` path under a home folder, which is never printed.

Library versions come from the installed packages' metadata, so nothing here imports torch,
transformers or Qt; the device and the model's facts are in the session's run records.

No units or coordinates here: text only.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from outline_tracker import __version__

DISTRIBUTION = "outline-tracker"                 # the name the tool is installed under
PACKAGE_FOLDER = Path(__file__).resolve().parent  # the package's source folder
LIBRARIES = ("torch", "transformers", "numpy", "opencv-python-headless", "scikit-image")  # named in run.log
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_SHORT = 7  # hex digits of a commit taken from direct_url.json, as `git rev-parse --short` gives
_GIT_TIMEOUT_S = 5.0


def tool_version() -> str:
    """The tool's version line: `outline-tracker 0.1.0 (commit abc1234)`, or `(commit unknown)`
    when no commit can be found (see the module's text for where it is looked for). Plain ASCII
    without a path. No units or coordinates."""
    commit = _installed_commit() or _checkout_commit(PACKAGE_FOLDER)
    return f"{DISTRIBUTION} {__version__} (commit {commit or 'unknown'})"


def machine_text() -> str:
    """The machine for run.log: the operating system as Python names it, and the Python version,
    e.g. `macOS-15.6-arm64-arm-64bit; Python 3.12.3`. No user or computer name. No units."""
    return f"{platform.platform()}; Python {platform.python_version()}"


def library_versions(names: Iterable[str] = LIBRARIES) -> dict[str, str]:
    """The installed version of each named distribution (`torch`, `transformers`, ...), read from
    its metadata without importing it; "not installed" for one that is missing. No units."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _installed_commit() -> str | None:
    """The commit the tool was installed from, 7 hex digits, if its metadata names one."""
    try:
        text = metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
        commit = json.loads(text)["vcs_info"]["commit_id"] if text else None
    except (metadata.PackageNotFoundError, ValueError, KeyError, TypeError, OSError):
        return None
    if isinstance(commit, str) and _COMMIT.fullmatch(commit.lower()):
        return commit.lower()[:_SHORT]
    return None


def _checkout_commit(folder: Path) -> str | None:
    """The short commit of the git checkout that tracks `folder`'s `__init__.py`, if there is one
    and git can be run."""
    tracked = _git(folder, "ls-files", "--error-unmatch", "__init__.py")
    commit = _git(folder, "rev-parse", "--short", "HEAD") if tracked is not None else None
    return commit if commit and _COMMIT.fullmatch(commit) else None


def _git(folder: Path, *args: str) -> str | None:
    """What a git command prints when run for `folder`; None when git is missing, fails or hangs.
    Git's own variables (GIT_DIR and the like, set while a git hook runs) are not passed on: they
    would point git at another repository than the one `folder` is in."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    try:
        done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True,
                              timeout=_GIT_TIMEOUT_S, check=False, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
