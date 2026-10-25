"""Panels 1 and 2 (task C2): the clip, and fps_true with its source (SPEC 2 steps 1 to 4, 3.3, 3.4, 4.1, 8.1, 10.1).

The student's name, and the video's facts and warnings, are in tests/gui/test_session_panels.py.

Expected values come from the spec, the design note and the files: a new clip is frame 0 to the last frame
at step 2 (SPEC 8.10); the frame grid is start, start + step, ... (SPEC 3.4); t = frame / fps_true
(SPEC 3.3); the hints and messages are the design note's sentences. The synthetic clips are 320 x 240
px at 240 frames per second with 120 frames (the dish clip) and 1280 x 720 px with 6 frames (the clip
without a warning). Frames are video frame numbers counted from 0; fps_true is in frames per second.
"""

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from gui_helpers import record_dialogs, show
from outline_tracker.gui.panels import time_panel
from outline_tracker.gui.settings import settings
from session_helpers import LEFT, NAME, TIME_HINT, body, state, type_into, wait_for_check, write_manifest

pytestmark = pytest.mark.usefixtures("own_settings")  # every test here has a settings folder of its own

UNDER_100 = "fps_true is under 100 fps. This usually means a re-timed copy of the video. Check the value."


# ---------------------------------------------------------------------------------------------
# Panel 1: the clip


def test_a_new_videos_clip_is_shown_and_a_change_reaches_the_session_and_the_bottom_bar(window, qtbot,
                                                                                         clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    show(window, qtbot)
    panel, clip, bar = body(window, 1), window.controller.session.clip, window.navigation
    assert (panel.start_box.value(), panel.end_box.value(), panel.step_box.value()) == (0, 119, 2)
    assert all(box.isEnabled() for box in (panel.start_box, panel.end_box, panel.step_box))

    panel.start_box.setValue(10)
    panel.end_box.setValue(50)
    panel.step_box.setValue(5)
    assert (clip.start, clip.end, clip.step) == (10, 50, 5)
    # the grid of SPEC 3.4: 10, 15, ..., 50 are 9 frames
    assert bar.slider.maximum() == 8 and bar.frame == 10 and window.view.frame == 10
    bar.last()
    assert bar.frame == 50 and window.view.frame == 50
    assert panel.clip_message.isHidden()


def test_a_frame_typed_into_a_clip_box_counts_on_enter(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    panel = body(window, 1)
    type_into(qtbot, panel.end_box, "9", enter=False)
    assert window.controller.session.clip.end == 119  # not while it is being typed
    type_into(qtbot, panel.end_box, "90")
    assert window.controller.session.clip.end == 90


# The clip without a warning has 6 frames, 0 to 5; each case starts from the clip 1 to 4, step 2.
@pytest.mark.parametrize("box, value, marked, message, good, clip_then", [
    ("end_box", 0, "end_box", "The end frame (0) is before the start frame (1).", 4, (1, 4, 2)),
    ("start_box", 5, "end_box", "The end frame (4) is before the start frame (5).", 2, (2, 4, 2)),
    ("step_box", 0, "step_box", "The step must be 1 frame or more.", 3, (1, 4, 3)),
    ("end_box", 6, "end_box", "The end frame must be 5 or less. The video has 6 frames.", 4, (1, 4, 2)),
    ("start_box", 500, "start_box", "The start frame must be 5 or less. The video has 6 frames.", 2, (2, 4, 2)),
])
def test_a_clip_that_cannot_be_used_is_refused_where_it_is_typed(window, qtbot, hd_clip, box, value, marked, message,
                                                                 good, clip_then):
    window.open_path(hd_clip.path)
    panel, clip = wait_for_check(qtbot, window), window.controller.session.clip
    type_into(qtbot, panel.name_edit, NAME)
    panel.start_box.setValue(1)
    panel.end_box.setValue(4)
    assert window.panels[0].state == "done"
    changes = []
    window.controller.session_changed.connect(lambda: changes.append(True))

    getattr(panel, box).setValue(value)
    assert (clip.start, clip.end, clip.step) == (1, 4, 2) and changes == []  # the session keeps the last good clip
    assert getattr(panel, box).value() == value  # what was typed stays to be seen: it is not changed silently
    assert not panel.clip_message.isHidden() and panel.clip_message.text() == message
    assert panel.clip_message.property("kind") == "problem"
    assert getattr(panel, marked).property("check") == "error"
    assert window.panels[0].state == "attention"

    getattr(panel, box).setValue(good)  # a value that can be used
    assert panel.clip_message.isHidden() and getattr(panel, marked).property("check") in (None, "")
    assert (clip.start, clip.end, clip.step) == clip_then and window.panels[0].state == "done"
    assert (changes == []) == (clip_then == (1, 4, 2))  # the session says that it changed, when it did


def test_the_wheel_changes_no_clip_box_that_does_not_have_the_keyboard(window, qtbot, clip_in_odd_folder):
    # the dock scrolls with the wheel: a box that happens to be under the cursor must keep its value
    window.open_path(clip_in_odd_folder.path)
    show(window, qtbot)
    panel = body(window, 1)
    window.view.setFocus()
    for box, before in ((panel.start_box, 0), (panel.end_box, 119), (panel.step_box, 2)):
        at = QPoint(box.width() // 2, box.height() // 2)
        turn = QWheelEvent(QPointF(at), QPointF(box.mapToGlobal(at)), QPoint(0, 0), QPoint(0, 120),
                           Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                           False)
        QApplication.sendEvent(box, turn)
        assert box.value() == before and not box.hasFocus()
    clip = window.controller.session.clip
    assert (clip.start, clip.end, clip.step) == (0, 119, 2)


# ---------------------------------------------------------------------------------------------
# Panel 2: fps_true


def test_a_typed_fps_true_reaches_the_session_and_the_time_of_the_frame(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    show(window, qtbot)
    panel, time = body(window, 2), window.controller.session.time
    assert time.fps_true is None and window.navigation.time_label.text() == "t = –"
    assert state(window, 2) == ("todo", TIME_HINT) and panel.source_label.text() == ""

    type_into(qtbot, panel.fps_edit, "239.6")
    assert time.fps_true == 239.6 and time.source == "typed" and time.manifest_path is None
    assert state(window, 2) == ("done", "fps_true = 239.60 fps (typed in).")
    assert panel.source_label.text() == "typed in"
    assert panel.message.isHidden() and panel.fps_edit.property("check") in (None, "")
    window.navigation.slider.setValue(12)  # step 2: frame 24, and t = 24 / 239.6 s = 0.100 s (SPEC 3.3)
    assert window.navigation.frame == 24 and window.navigation.time_label.text() == "t = 0.100 s"


@pytest.mark.parametrize("typed, message", [
    ("0", "fps_true must be more than 0 fps."),
    ("-240", "fps_true must be more than 0 fps."),
    ("nan", "fps_true must be a number, for example 240."),
    ("inf", "fps_true must be a number, for example 240."),
    ("fast", "fps_true must be a number, for example 240."),
])
def test_an_fps_true_that_is_not_positive_and_finite_is_refused_where_it_is_typed(window, qtbot, clip_in_odd_folder,
                                                                                  typed, message):
    window.open_path(clip_in_odd_folder.path)
    panel, time = body(window, 2), window.controller.session.time
    type_into(qtbot, panel.fps_edit, typed)
    assert time.fps_true is None
    assert not panel.message.isHidden() and panel.message.text() == message
    assert panel.message.property("kind") == "problem" and panel.fps_edit.property("check") == "error"
    assert window.panels[1].state == "attention"
    assert panel.fps_edit.text() == typed  # what was typed stays to be seen

    type_into(qtbot, panel.fps_edit, "240")
    assert time.fps_true == 240.0 and panel.message.isHidden() and window.panels[1].state == "done"
    type_into(qtbot, panel.fps_edit, typed)  # refused again: the good value stays in the session
    assert time.fps_true == 240.0 and time.source == "typed" and window.panels[1].state == "attention"


def test_an_fps_true_under_100_is_taken_with_a_warning(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    panel, time = body(window, 2), window.controller.session.time
    type_into(qtbot, panel.fps_edit, "30")
    assert time.fps_true == 30.0  # warned about, not blocked (SPEC 3.3)
    assert state(window, 2) == ("attention", UNDER_100) and panel.fps_edit.property("check") == "warn"
    type_into(qtbot, panel.fps_edit, "100")  # the warning is for values below 100
    assert state(window, 2) == ("done", "fps_true = 100.00 fps (typed in).")
    assert panel.fps_edit.property("check") in (None, "")


def test_fps_true_comes_from_the_manifest_above_the_video(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    manifest = write_manifest(path.parent / "data" / "manifest.csv", [("other.MOV", "120"), ("dish.MOV", "239.6")])
    window.open_path(path)
    panel, time = body(window, 2), window.controller.session.time
    assert time.fps_true == 239.6 and time.source == "manifest"
    assert time.manifest_path is not None and manifest.samefile(time.manifest_path)
    assert state(window, 2) == ("done", "fps_true = 239.60 fps (from the manifest).")
    assert panel.source_label.text() == "from the manifest" and panel.fps_edit.text() == "239.6"

    type_into(qtbot, panel.fps_edit, "238")  # a typed value takes over
    assert (time.fps_true, time.source, time.manifest_path) == (238.0, "typed", None)
    assert panel.source_label.text() == "typed in"


def test_a_manifest_cell_that_is_no_frame_rate_is_no_source(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    write_manifest(path.parent / "data" / "manifest.csv", [("dish.MOV", "0")])
    window.open_path(path)
    assert window.controller.session.time.fps_true is None and state(window, 2) == ("todo", TIME_HINT)


def test_a_manifest_the_user_points_to_is_used_and_remembered(window, qtbot, monkeypatch, clip_in_odd_folder,
                                                               hd_clip, tmp_path, own_settings):
    asked = record_dialogs(monkeypatch)
    manifest = write_manifest(tmp_path / "course" / "my manifest.csv", [("dish.MOV", "239.6"), ("hd.mp4", "119.88")])
    window.open_path(clip_in_odd_folder.path)
    panel = body(window, 2)
    assert window.controller.session.time.fps_true is None  # no manifest above the video
    assert panel.manifest_button.text() == "Choose manifest"
    qtbot.mouseClick(panel.manifest_button, LEFT)
    (parent, title, filters, on_chosen), = asked.files
    assert parent is window and title == "Choose manifest" and "*.csv" in filters[0]

    on_chosen(manifest)
    time = window.controller.session.time
    assert (time.fps_true, time.source) == (239.6, "manifest") and manifest.samefile(time.manifest_path)
    assert state(window, 2) == ("done", "fps_true = 239.60 fps (from the manifest).")
    kept = settings()
    assert kept.format() == QSettings.Format.IniFormat and kept.scope() == QSettings.Scope.UserScope
    assert Path(kept.fileName()).is_relative_to(own_settings)  # in this test's folder, not the user's own
    assert manifest.samefile(kept.value(time_panel.MANIFEST_KEY))

    window.open_path(hd_clip.path)  # another video, with no manifest above it: the remembered one is asked
    time = window.controller.session.time
    assert (time.fps_true, time.source) == (119.88, "manifest") and manifest.samefile(time.manifest_path)


def test_a_manifest_without_the_video_changes_nothing_and_is_not_remembered(window, qtbot, monkeypatch,
                                                                            clip_in_odd_folder, tmp_path):
    asked = record_dialogs(monkeypatch)
    manifest = write_manifest(tmp_path / "course" / "manifest.csv", [("other.MOV", "239.6")])
    window.open_path(clip_in_odd_folder.path)
    panel = body(window, 2)
    type_into(qtbot, panel.fps_edit, "240")
    qtbot.mouseClick(panel.manifest_button, LEFT)
    asked.files[0][3](manifest)
    time = window.controller.session.time
    assert (time.fps_true, time.source) == (240.0, "typed")
    assert not panel.message.isHidden() and panel.message.property("kind") == "problem"
    assert panel.message.text() == ("manifest.csv has no fps_true for dish_tracker.mp4. Check the row of this "
                                    "video in the manifest, or type fps_true.")
    assert settings().value(time_panel.MANIFEST_KEY) is None
    assert asked.messages == []  # said in the panel, not in a dialog


def test_the_manifest_button_waits_for_a_video(window):
    assert not body(window, 2).manifest_button.isEnabled() and not body(window, 2).fps_edit.isEnabled()
