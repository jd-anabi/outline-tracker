"""How the window is started from the command line (SPEC 11; task C0): `outline-tracker` with no
arguments, and `outline-tracker gui [VIDEO | SESSION.json]`.

No window opens here: `cli_gui.run` or `gui.app.main` is replaced by a recorder. A missing file is
one `ERROR: ...` line and exit code 1, in the wording of `check` and `convert`, before torch or Qt
is loaded (checked in a subprocess: this process has PySide6 loaded by pytest-qt).
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import helpers
from outline_tracker import cli, cli_gui, launch
from outline_tracker.gui import app
from outline_tracker.provenance import tool_version

REPO = Path(__file__).resolve().parents[2]
HEAVY = {"PySide6", "pyqtgraph", "torch", "torchvision", "transformers", "timm"}


@pytest.fixture
def no_window(monkeypatch):
    """Fail the test if the window's entry point is reached."""
    monkeypatch.setattr(app, "main", lambda argv: pytest.fail(f"the window was started with {argv}"))


# ---------------------------------------------------------------------------------------------
# outline-tracker (the launcher)


def test_the_installed_command_is_the_launcher():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"] == {"outline-tracker": "outline_tracker.launch:main"}


def test_no_arguments_at_all_runs_the_gui_command(monkeypatch):
    seen = []
    monkeypatch.setattr(cli_gui, "run", lambda args: seen.append(args) or 7)
    assert launch.main([]) == 7
    assert [(args.command, args.path) for args in seen] == [("gui", None)]
    monkeypatch.setattr(sys, "argv", ["outline-tracker"])  # as the installed command is started
    assert launch.main() == 7
    assert [(args.command, args.path) for args in seen] == [("gui", None)] * 2


def test_with_arguments_the_launcher_is_the_command_line(no_window, tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as stopped:
        launch.main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out == tool_version() + "\n"
    missing = tmp_path / "does_not_exist.MOV"
    assert launch.main(["check", str(missing)]) == 1
    assert capsys.readouterr().err == f"ERROR: No such file: {missing}\n"
    monkeypatch.setattr(sys, "argv", ["outline-tracker", "check", str(missing)])
    assert launch.main() == 1
    assert capsys.readouterr().err == f"ERROR: No such file: {missing}\n"


def test_the_command_line_alone_still_lists_the_commands_with_gui_among_them(no_window, capsys):
    assert cli.main([]) == 0
    assert any(line.split()[:1] == ["gui"] for line in capsys.readouterr().out.splitlines())


# ---------------------------------------------------------------------------------------------
# outline-tracker gui [VIDEO | SESSION.json]


def test_gui_help_names_what_it_opens(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["gui", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "outline-tracker gui" in shown and "VIDEO" in shown and "SESSION.json" in shown


def test_gui_starts_the_window_with_the_path_or_with_none(monkeypatch, tmp_path, capsys):
    started = []
    monkeypatch.setattr(app, "main", lambda argv: started.append(argv) or 3)
    clip = tmp_path / helpers.ODD_FOLDER / "dish_tracker.mp4"  # a space and non-ASCII characters
    clip.parent.mkdir()
    clip.write_bytes(b"")
    assert cli.main(["gui", str(clip)]) == 3  # the window's exit code is the command's
    assert cli.main(["gui"]) == 3
    assert started == [[str(clip)], []]
    shown = capsys.readouterr()
    assert (shown.out, shown.err) == ("", "")


def test_gui_with_a_missing_file_is_one_error_line_and_no_window(no_window, tmp_path, capsys):
    missing = tmp_path / "does_not_exist.mp4"
    assert cli.main(["gui", str(missing)]) == 1
    shown = capsys.readouterr()
    assert shown.err == f"ERROR: No such file: {missing}\n"
    assert shown.out == ""


def test_gui_with_a_folder_is_one_error_line_and_no_window(no_window, tmp_path, capsys):
    assert cli.main(["gui", str(tmp_path)]) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith("ERROR: ") and shown.err.count("\n") == 1 and str(tmp_path) in shown.err


def test_gui_takes_one_path_at_most(no_window, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["gui", "one_tracker.mp4", "two_tracker.mp4"])
    assert stopped.value.code == 2
    assert "two_tracker.mp4" in capsys.readouterr().err


def test_a_missing_file_ends_the_launcher_before_torch_or_qt_is_loaded(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    # -X importtime prints every module this process imports to stderr, one per line.
    done = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "outline_tracker.launch", "gui", str(missing)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert done.returncode == 1
    assert f"ERROR: No such file: {missing}" in done.stderr.splitlines()
    imported = {line.rsplit("|", 1)[-1].strip().split(".")[0] for line in done.stderr.splitlines()}
    assert "outline_tracker" in imported, done.stderr  # the listing is there and was parsed
    assert sorted(imported & HEAVY) == []
