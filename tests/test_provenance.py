"""The tool's version line and the machine facts of run.log (SPEC 8.9).

`tool_version()` is `outline-tracker 0.1.0 (commit abc1234)`. The commit comes from direct_url.json
in the installed package's metadata when the tool was installed from git (an installed tool has no
.git); from `git rev-parse --short HEAD` when the package's source folder is part of a git checkout
(a developer's `uv run`); else it is unknown. An editable install's direct_url.json holds a file://
path under somebody's home folder: that never reaches the line. The metadata here is faked, and the
git checkouts are made in the test's own temporary folder.
"""

import json
import os
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from outline_tracker import cli, provenance

REPO = Path(__file__).resolve().parents[1]
LINE = re.compile(r"outline-tracker 0\.2\.0\.dev0 \(commit ([0-9a-f]{7,40}|unknown)\)")
GIT_INSTALL = {"url": "https://github.com/jd-anabi/outline-tracker",
               "vcs_info": {"vcs": "git", "commit_id": "abc1234def5678900987654321abcdefabcdef12",
                            "requested_revision": "main"}}
EDITABLE = {"url": "file:///home/someone/outline-tracker", "dir_info": {"editable": True}}
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


class _Distribution:
    """Stands in for the installed distribution: only its direct_url.json is asked for."""

    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return self.direct_url


def _install(monkeypatch, direct_url, package_folder):
    """Make the tool look installed with this direct_url.json (a dict, text, or None for no file)
    and its source in `package_folder`."""
    text = json.dumps(direct_url) if isinstance(direct_url, dict) else direct_url
    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: _Distribution(text))
    monkeypatch.setattr(provenance, "PACKAGE_FOLDER", Path(package_folder))


def _git(folder, *args):
    """Run git for `folder`, whatever the user's own git settings are, and return what it prints."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}  # as inside a hook
    done = subprocess.run(["git", "-C", str(folder), "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True, env=env)
    return done.stdout.strip()


def _checkout(folder, tracked=True):
    """A git checkout with one commit; the package folder inside it is committed, or (tracked=False)
    ignored, as a virtual environment inside a student's repository is. Returns (package folder, HEAD)."""
    package = folder / ("outline_tracker" if tracked else ".venv/site-packages/outline_tracker")
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (folder / ".gitignore").write_text(".venv/\n")
    _git(folder, "init", "-q")
    _git(folder, "add", ".")
    _git(folder, "commit", "-q", "-m", "one")
    return package, _git(folder, "rev-parse", "HEAD")


def _no_git(*args, **kwargs):
    raise AssertionError("git must not be asked")


def test_a_git_install_takes_the_commit_from_direct_url_json(monkeypatch, tmp_path):
    _install(monkeypatch, GIT_INSTALL, tmp_path)
    monkeypatch.setattr(provenance.subprocess, "run", _no_git)
    assert provenance.tool_version() == "outline-tracker 0.2.0.dev0 (commit abc1234)"


@needs_git
def test_a_source_folder_in_a_git_checkout_takes_the_commit_from_git(monkeypatch, tmp_path):
    package, head = _checkout(tmp_path / "repo")
    _install(monkeypatch, EDITABLE, package)
    line = provenance.tool_version()
    commit = LINE.fullmatch(line).group(1)
    assert len(commit) >= 7 and head.startswith(commit)


@needs_git
def test_a_package_that_only_lies_inside_somebody_elses_checkout_has_no_commit(monkeypatch, tmp_path):
    package, _ = _checkout(tmp_path / "repo", tracked=False)
    _install(monkeypatch, None, package)
    assert provenance.tool_version() == "outline-tracker 0.2.0.dev0 (commit unknown)"


@pytest.mark.parametrize("direct_url", [None, EDITABLE, "{ not json", '["a list"]',
                                        {"url": "x", "vcs_info": {"commit_id": "file:///home/someone"}},
                                        {"url": "x", "vcs_info": {"commit_id": None}}],
                         ids=["no direct_url.json", "editable", "damaged", "not an object", "commit is not a hash",
                              "commit missing"])
def test_without_a_commit_anywhere_the_line_says_unknown(monkeypatch, tmp_path, direct_url):
    _install(monkeypatch, direct_url, tmp_path)  # a folder that is in no git checkout
    assert provenance.tool_version() == "outline-tracker 0.2.0.dev0 (commit unknown)"


def test_a_tool_that_is_not_installed_as_a_distribution_still_has_a_line(monkeypatch, tmp_path):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "distribution", missing)
    monkeypatch.setattr(provenance, "PACKAGE_FOLDER", tmp_path)
    assert provenance.tool_version() == "outline-tracker 0.2.0.dev0 (commit unknown)"


@needs_git
def test_git_variables_of_a_hook_do_not_point_at_another_repository(monkeypatch, tmp_path):
    package, head = _checkout(tmp_path / "repo")
    _, other_head = _checkout(tmp_path / "other", tracked=False)  # other files, so another commit
    assert other_head != head
    _install(monkeypatch, EDITABLE, package)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))
    assert head.startswith(LINE.fullmatch(provenance.tool_version()).group(1))


def test_a_computer_without_git_gives_unknown_not_an_error(monkeypatch, tmp_path):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    _install(monkeypatch, EDITABLE, tmp_path)
    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    assert provenance.tool_version() == "outline-tracker 0.2.0.dev0 (commit unknown)"


@pytest.mark.parametrize("direct_url", [GIT_INSTALL, EDITABLE, None], ids=["git", "editable", "plain"])
def test_the_line_never_holds_a_path(monkeypatch, tmp_path, direct_url):
    _install(monkeypatch, direct_url, tmp_path)
    line = provenance.tool_version()
    assert LINE.fullmatch(line), line
    assert "file:" not in line and "/" not in line and "\\" not in line and "someone" not in line


def test_the_version_option_prints_the_line(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, GIT_INSTALL, tmp_path)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out == "outline-tracker 0.2.0.dev0 (commit abc1234)\n"


@needs_git
@pytest.mark.skipif(not (REPO / ".git").exists(), reason="the tests do not run from a git checkout")
def test_in_this_checkout_the_line_names_the_checked_out_commit():
    head = _git(REPO, "rev-parse", "HEAD")
    commit = LINE.fullmatch(provenance.tool_version()).group(1)
    assert head.startswith(commit)


def test_machine_and_library_facts_need_no_heavy_import():
    import platform

    assert provenance.machine_text() == f"{platform.platform()}; Python {platform.python_version()}"
    versions = provenance.library_versions(["numpy", "torch", "no-such-package-xyz"])
    assert versions == {"numpy": np.__version__, "torch": metadata.version("torch"),
                        "no-such-package-xyz": "not installed"}
