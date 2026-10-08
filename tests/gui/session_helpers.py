"""What the tests of panels 1 and 2 and of session saving share (task C2).

Imported by name from tests/gui/test_session_panels.py and tests/gui/test_session_saving.py. Every
clip a test here works on is a copy in the test's own folder (the fixture `hd_clip` of
tests/gui/conftest.py), because the window writes a run folder next to the video. Frames are video
frame numbers counted from 0; fps_true is in frames per second.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

NAME = "Ada Lovelace"
FOLDER_OF_NAME = "dish_tracker_outline_Ada_Lovelace"  # SPEC 8.1: <video stem>_outline_<student>, blanks as _
LEFT = Qt.MouseButton.LeftButton


def body(window, number: int):
    """The widget that the module of panel `number` (1 to 9) put under the panel's hint line."""
    return window.panels[number - 1].body.itemAt(1).widget()


def type_into(qtbot, field, text: str, enter: bool = True) -> None:
    """Replace what a text field holds by `text`, key by key as a user types it, and press Enter
    (or leave the text uncommitted with `enter` false)."""
    field.selectAll()
    qtbot.keyClick(field, Qt.Key.Key_Delete)
    qtbot.keyClicks(field, text)
    if enter:
        qtbot.keyClick(field, Qt.Key.Key_Return)


def wait_for_check(qtbot, window):
    """Wait until the check of the open video has come back from its thread; returns panel 1's body."""
    panel = body(window, 1)
    qtbot.waitUntil(lambda: panel.check is not None, timeout=10_000)
    return panel


def settle(qtbot, window):
    """Wait until no check of a video is running any more (each holds its file open, and Windows
    moves or deletes no open file), and let what the checks sent arrive; returns panel 1's body."""
    panel = body(window, 1)
    qtbot.waitUntil(lambda: not any(thread.isRunning() for thread in panel.check_threads), timeout=10_000)
    QApplication.processEvents()
    return panel


def second_spelling(folder: Path, name: str) -> Path:
    """The folder `folder` under `name`, which differs from the folder's own name only in upper and
    lower case: one folder with two spellings, as a file system that ignores case has it (the
    default on macOS and Windows). Where the file system tells the two spellings apart, a link
    named `name` beside the folder stands in for the second one. Returns the second spelling."""
    assert name != folder.name and name.casefold() == folder.name.casefold()
    other = folder.with_name(name)
    if not other.exists():
        other.symlink_to(folder.name, target_is_directory=True)
    return other


def read_json(path) -> dict:
    """A JSON file's content."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path, rows) -> Path:
    """Write a course manifest at `path` (its folder is made): `rows` are (video file name,
    fps_true) pairs, fps_true as the text of the cell (frames per second)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["video_file,fps_true", *(f"{name},{fps}" for name, fps in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
