"""What the tests of panels 1 and 2 and of session saving share (task C2).

Imported by name from tests/gui/test_session_panels.py and tests/gui/test_session_saving.py, the
fixtures too (a fixture imported into a test module is that module's own). Every clip a test here
works on is a copy in the test's own folder, because the window writes a run folder next to the
video. Frames are video frame numbers counted from 0; fps_true is in frames per second.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

NAME = "Ada Lovelace"
FOLDER_OF_NAME = "dish_tracker_outline_Ada_Lovelace"  # SPEC 8.1: <video stem>_outline_<student>, blanks as _
LEFT = Qt.MouseButton.LeftButton


@pytest.fixture(autouse=True)
def own_settings(tmp_path):
    """Keep the application's settings out of the user's own: Qt's INI files of the user's scope
    go to a folder of this test. Returns that folder. Qt cannot be asked where they went before,
    so the scope is not put back: it stays on a test folder, which no later test may count on."""
    folder = tmp_path / "settings"
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
    return folder


@pytest.fixture(scope="session")
def hd_scene_clip(tmp_path_factory):
    """`dish_scene` at 1280 x 720 px, 6 frames, 240 frames per second: a clip that last week's check
    of a video has no warning for (720 px at the short side, a slow-motion frame rate)."""
    from outline_tracker import synthetic

    scene = synthetic.dish_scene(size=(1280, 720), n_frames=6)
    return synthetic.render(scene, tmp_path_factory.mktemp("hd") / "hd_tracker.mp4")


@pytest.fixture
def hd_clip(tmp_path, hd_scene_clip):
    """A copy of the 1280 x 720 clip in a folder of this test: the GroundTruth with its `path`."""
    folder = tmp_path / "hd video"
    folder.mkdir()
    path = Path(shutil.copyfile(hd_scene_clip.path, folder / hd_scene_clip.path.name))
    return replace(hd_scene_clip, path=path)


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
