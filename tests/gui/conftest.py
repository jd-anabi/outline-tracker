"""Fixtures for the tests of the window (tests/gui/): larger clips, a tracked run folder, a second
window, a settings folder of the test's own, and two guards (the application's look is put back; a
dialog that would block is an error). pytest finds a fixture by its name; nothing imports this file
(tests/test_repo_rules.py says why). The fixtures for the tests of every folder, `window` among them,
are in tests/conftest.py; plain helpers are in the helper modules beside this file.
"""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from helpers import new_window
from outline_tracker.segmenter.fake import ExactFake
from tracking_helpers import abc_session, run

WIDE = (1280, 720)  # a frame wide enough for the spec's stick of 926 px

# ---------------------------------------------------------------------------------------------
# Clips, and a run folder with results


@pytest.fixture(scope="module")
def wide_clip(tmp_path_factory):
    """`dish_scene` at 1280 x 720 px, 4 frames, H.264: the GroundTruth with its `path`. Rendered once
    for each test module that asks for it, not once for the run: tests open this file itself, not
    a copy of it, so it is shared no further than it was while each module had its own."""
    from outline_tracker import synthetic

    scene = synthetic.dish_scene(size=WIDE, n_frames=4)
    return synthetic.render(scene, tmp_path_factory.mktemp("wide") / "wide_tracker.mp4")


@pytest.fixture(scope="session")
def hd_scene_clip(tmp_path_factory):
    """`dish_scene` at 1280 x 720 px, 6 frames, 240 frames per second: a clip that last week's check
    of a video has no warning for (720 px at the short side, a slow-motion frame rate). Rendered
    once for the run: tests work on a copy of it (`hd_clip`), never on this file."""
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


@pytest.fixture(scope="session")
def dish_run(dish_clip, tmp_path_factory):
    """A run folder in which the core tracked A, B and C of the dish clip on every 2nd frame with
    `ExactFake`, with a scale. Made once for the run: tests open a copy of it (`opened_run`), never
    the folder itself."""
    folder = tmp_path_factory.mktemp("review") / "run"
    status, _ = run(dish_clip, abc_session(dish_clip, folder), folder, ExactFake(dish_clip))
    assert status == "complete"
    return folder


# ---------------------------------------------------------------------------------------------
# A second window; settings of the test's own


@pytest.fixture
def second_window(qtbot):
    """A second `MainWindow` for the same test, not shown yet: made and closed by the very function
    of the `window` fixture (`new_window`, tests/helpers.py), so it has the stand-in's factory, and
    after the test its worker is stopped and its video released. A saved session reopens in it."""
    yield from new_window(qtbot)


@pytest.fixture
def own_settings(tmp_path):
    """Keep the application's settings out of the user's own: Qt's INI files of the user's scope
    go to a folder of this test. Returns that folder. Qt cannot be asked where they went before,
    so the scope is not put back: it stays on a test folder, which no later test may count on.

    It acts on the tests that ask for it and on no other. The two modules whose every test needs it
    say so in their `pytestmark` (tests/gui/test_session_panels.py, tests/gui/test_session_saving.py).
    """
    folder = tmp_path / "settings"
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
    return folder


# ---------------------------------------------------------------------------------------------
# Guards


@pytest.fixture
def look(qapp):
    """Put the application's style, palette and style sheet back after the test."""
    from PySide6.QtGui import QPalette

    style, palette, sheet = qapp.style().name(), QPalette(qapp.palette()), qapp.styleSheet()
    yield
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if style:  # no name: a style sheet was in place before the test, and its wrapper cannot be asked
        qapp.setStyle(style)


@pytest.fixture
def never_blocking(monkeypatch):
    """Make `exec`, which would wait for the user and never return here, an error."""
    def blocked(*_):
        raise AssertionError("A dialog was run with exec(): it blocks the window. Use open().")

    for kind in (QDialog, QMessageBox, QFileDialog):
        monkeypatch.setattr(kind, "exec", blocked)
