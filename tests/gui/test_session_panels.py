"""Panels 1 and 2 (task C2): the student's name, the video's facts and warnings, the clip, and
fps_true with its source (SPEC 2 steps 1 to 4, 3.3, 3.4, 4.1, 8.1, 10.1).

Expected values come from the spec, the design note and the files: the run folder is
`<video folder>/<video stem>_outline_<student>` (SPEC 8.1); a new clip is frame 0 to the last frame
at step 2 (SPEC 8.10); the frame grid is start, start + step, ... (SPEC 3.4); t = frame / fps_true
(SPEC 3.3); the hints and messages are the design note's sentences. The synthetic clips are 320 x 240
px at 240 frames per second with 120 frames (the dish clip) and 1280 x 720 px with 6 frames (the clip
without a warning). Frames are video frame numbers counted from 0; fps_true is in frames per second.
"""

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QKeySequence, QWheelEvent
from PySide6.QtWidgets import QApplication, QLabel

import helpers
from gui_helpers import record_dialogs, show
from outline_tracker import video
from outline_tracker.gui.panels import time_panel
from session_helpers import (FOLDER_OF_NAME, LEFT, NAME, body, hd_clip, hd_scene_clip,  # noqa: F401 (fixtures)
                             own_settings, settle, type_into, wait_for_check, write_manifest)

START_HINT = "Type your name. Then open your video (a _tracker.mp4 file)."
NAME_MISSING = "Type your name in panel 1 first. The run folder is named after you. Nothing is saved until then."
NAME_FIRST = "Type your name in panel 1 first."
TIME_HINT = "Type the true frame rate (fps_true), or measure it with the stopwatch."
UNDER_100 = "fps_true is under 100 fps. This usually means a re-timed copy of the video. Check the value."


def state(window, number: int) -> tuple[str, str]:
    """(state, hint) of panel `number`."""
    panel = window.panels[number - 1]
    return panel.state, panel.hint.text()


# ---------------------------------------------------------------------------------------------
# Panel 1: the name


def test_a_typed_name_reaches_the_session_and_names_the_run_folder(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    window.open_path(path)
    panel, controller = body(window, 1), window.controller
    assert controller.run_folder is None and panel.name_edit.text() == ""
    type_into(qtbot, panel.name_edit, NAME)
    assert controller.session.student == NAME
    assert controller.run_folder == path.parent / FOLDER_OF_NAME  # SPEC 8.1, beside the video
    assert panel.folder_label.full_text == FOLDER_OF_NAME  # the folder's name; its place is the tooltip
    assert helpers.ODD_FOLDER in panel.folder_label.toolTip()


def test_blanks_around_a_name_are_dropped(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    panel = body(window, 1)
    type_into(qtbot, panel.name_edit, "  Ada  ")
    assert window.controller.session.student == "Ada" and panel.name_edit.text() == "Ada"
    assert window.controller.run_folder.name == "dish_tracker_outline_Ada"


def test_a_name_typed_before_the_video_is_opened_is_the_new_sessions_name(window, qtbot, clip_in_odd_folder):
    # the workflow of SPEC 2: the name first, then the video
    panel = body(window, 1)
    type_into(qtbot, panel.name_edit, NAME)
    assert window.controller.session is None
    assert state(window, 1) == ("todo", "Open your video (a _tracker.mp4 file).")
    window.open_path(clip_in_odd_folder.path)
    assert window.controller.session.student == NAME
    assert window.controller.run_folder == clip_in_odd_folder.path.parent / FOLDER_OF_NAME
    assert panel.name_edit.text() == NAME


def test_without_a_name_the_panel_says_why_nothing_is_saved_and_track_and_export_are_refused(window, qtbot, hd_clip):
    panel, controller = body(window, 1), window.controller
    assert panel.name_message.isHidden()  # nothing is open: nothing to save
    for action in ("track", "export"):
        assert controller.refusal(action) == NAME_FIRST  # first of all the name
    window.open_path(hd_clip.path)
    assert not panel.name_message.isHidden() and panel.name_message.text() == NAME_MISSING
    assert panel.name_message.property("kind") == "problem"
    assert [controller.refusal(action) for action in ("track", "export")] == [NAME_FIRST, NAME_FIRST]
    assert state(window, 1) == ("todo", "Type your name. The run folder is named after you.")

    type_into(qtbot, panel.name_edit, NAME)
    assert panel.name_message.isHidden()
    assert [controller.refusal(action) for action in ("track", "export")] == [None, None]

    type_into(qtbot, panel.name_edit, "   ")  # blanks are no name
    assert controller.session.student == "" and controller.run_folder is None
    assert not panel.name_message.isHidden()
    assert controller.refusal("export") == NAME_FIRST
    with pytest.raises(ValueError, match="polish"):
        controller.refusal("polish")


def test_with_a_name_and_no_video_track_and_export_are_refused_for_the_video(window, qtbot):
    controller = window.controller
    assert controller.session is None
    type_into(qtbot, body(window, 1).name_edit, NAME)
    assert controller.refusal("track") == controller.refusal("export") == "Open your video in panel 1 first."


# ---------------------------------------------------------------------------------------------
# Panel 1: the video's facts, last week's check, the panel's state


def test_the_empty_window_keeps_the_start_hints_and_shows_no_facts(window):
    panel = body(window, 1)
    assert state(window, 1) == ("todo", START_HINT) and state(window, 2) == ("todo", TIME_HINT)
    assert panel.file_label.full_text == "" and panel.warnings_label.isHidden()
    assert not panel.start_box.isEnabled() and not panel.end_box.isEnabled() and not panel.step_box.isEnabled()
    assert panel.check is None
    # nothing of a video is shown before one is open: the name and the two buttons are all there is
    assert panel.video_part.isHidden() and not panel.name_edit.isHidden() and not panel.open_video_button.isHidden()
    assert all(part.parentWidget() is panel.video_part for part in (panel.file_label, panel.start_box,
                                                                    panel.folder_label, panel.save_message))


def test_the_panel_shows_what_the_file_says(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    window.open_path(path)
    panel = body(window, 1)
    assert not panel.video_part.isHidden()
    assert panel.file_label.full_text == "dish_tracker.mp4"
    assert helpers.ODD_FOLDER not in panel.file_label.full_text  # the name, never the folder
    assert panel.frame_size_label.text() == "320 × 240 px"
    assert panel.frames_label.text() == "120"
    assert panel.file_fps_label.text() == "240.00 fps"
    assert "file says" in panel.file_fps_word.text()  # labelled as the file's own statement (SPEC 3.3)
    size = path.stat().st_size
    assert 1_000 <= size < 1_000_000 and panel.size_label.text() == f"{size / 1000:.0f} kB"


@pytest.mark.parametrize("n_bytes, text", [(999, "999 bytes"), (601_379, "601 kB"), (12_300_000, "12.3 MB"),
                                           (2_500_000_000, "2.50 GB")])
def test_a_size_is_written_in_decimal_units(n_bytes, text):
    from outline_tracker.gui.panels.video_panel import size_text

    assert size_text(n_bytes) == text


def test_the_warnings_of_the_check_are_shown_and_the_panel_needs_attention(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    panel = wait_for_check(qtbot, window)
    # last week's check warns about a frame under 720 px at its short side; this clip is 320 x 240
    assert panel.check.warnings == video.check_video(clip_in_odd_folder.path).warnings
    assert len(panel.check.warnings) == 1 and "Low resolution (320x240)" in panel.check.warnings[0]
    assert not panel.warnings_label.isHidden() and panel.warnings_label.property("kind") == "warning"
    assert panel.warnings_label.text() == panel.check.warnings[0]
    assert state(window, 1) == ("attention", "This video has 1 warning. Read it below before you continue.")
    type_into(qtbot, panel.name_edit, NAME)  # a name does not take the warning away
    assert state(window, 1)[0] == "attention"


def test_two_warnings_are_counted_and_both_shown(window, qtbot, clip_in_odd_folder, monkeypatch):
    real = video.check_video

    def two_warnings(path):
        check = real(path)
        check.warnings.append("A second thing is wrong with this file.")
        return check

    monkeypatch.setattr(video, "check_video", two_warnings)
    window.open_path(clip_in_odd_folder.path)
    panel = wait_for_check(qtbot, window)
    assert state(window, 1) == ("attention", "This video has 2 warnings. Read them below before you continue.")
    assert panel.warnings_label.text().split("\n\n") == panel.check.warnings


def test_with_a_name_and_a_video_without_warnings_the_panel_is_done(window, qtbot, hd_clip):
    window.open_path(hd_clip.path)
    panel = wait_for_check(qtbot, window)
    assert panel.check.warnings == [] and panel.warnings_label.isHidden()
    assert state(window, 1) == ("todo", "Type your name. The run folder is named after you.")
    type_into(qtbot, panel.name_edit, NAME)
    # the design note's line: file, frame size, frames, clip (0 to the last frame, step 2: SPEC 8.10)
    assert state(window, 1) == ("done", "hd_tracker.mp4 · 1280 × 720 px · 6 frames · clip 0 to 5, step 2.")
    panel.step_box.setValue(1)
    assert state(window, 1) == ("done", "hd_tracker.mp4 · 1280 × 720 px · 6 frames · clip 0 to 5, step 1.")


def test_the_check_runs_off_the_gui_thread_and_is_over_when_the_window_has_closed(window, qtbot, clip_in_odd_folder,
                                                                                  monkeypatch):
    real, started, seen = video.check_video, threading.Event(), []

    def slow(path):  # a check that takes a while, as on a long video
        started.set()
        check = real(path)
        time.sleep(0.2)
        seen.append((threading.current_thread() is threading.main_thread(), path))
        return check

    monkeypatch.setattr(video, "check_video", slow)
    window.open_path(clip_in_odd_folder.path)
    assert started.wait(10)  # at work, in its own thread
    assert window.close()  # closing waits for the check: the file is not held open afterwards
    assert seen == [(False, clip_in_odd_folder.path)]
    clip_in_odd_folder.path.unlink()  # Windows refuses this while a file is open


def test_a_check_that_comes_back_for_another_video_is_dropped(window, qtbot, clip_in_odd_folder, hd_clip, monkeypatch):
    real, let_go = video.check_video, threading.Event()

    def held(path):  # the first video's check comes back only after the second video's
        if path == clip_in_odd_folder.path:
            let_go.wait(10)
        return real(path)

    monkeypatch.setattr(video, "check_video", held)
    window.open_path(clip_in_odd_folder.path)
    window.open_path(hd_clip.path)
    panel = wait_for_check(qtbot, window)
    assert panel.check.info.width == 1280 and panel.check.warnings == []
    let_go.set()
    settle(qtbot, window)  # the dish clip's check, with its warning, has arrived by now
    assert panel.check.info.width == 1280 and panel.warnings_label.isHidden()
    assert window.panels[0].state == "todo"  # no warning, and no name yet


def test_a_row_added_to_a_body_follows_what_is_there_with_the_notes_spacing(window, qtbot):
    # follows test_rows_added_to_the_body_follow_the_hint_with_the_notes_spacing (tests/gui/test_panel.py):
    # the design note's 8 px between rows and 12 px of padding, with panel 1's controls under the hint
    show(window, qtbot)
    panel = window.panels[0]
    controls = panel.body.itemAt(panel.body.count() - 1).widget()
    assert controls is body(window, 1)
    row = QLabel("a row of a later task")
    panel.body.addWidget(row)
    qtbot.waitUntil(row.isVisible)
    inside = panel.contentsRect()
    hint_at, controls_at, row_at = (part.mapTo(panel, QPoint(0, 0)) for part in (panel.hint, controls, row))
    assert controls_at.y() - (hint_at.y() + panel.hint.height()) == 8  # 8 px between rows
    assert row_at.y() - (controls_at.y() + controls.height()) == 8
    assert hint_at.x() - inside.left() == 12  # body padding: left, right, bottom
    assert inside.left() + inside.width() - (hint_at.x() + panel.hint.width()) == 12
    assert (controls_at.x(), controls.width()) == (hint_at.x(), panel.hint.width())  # as wide as the hint line
    assert inside.top() + inside.height() - (row_at.y() + row.height()) == 12
    assert hint_at.y() - inside.top() == 32  # right under the header row
    panel.set_expanded(False)
    assert not row.isVisible() and not controls.isVisible()


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
# Panel 1: its two buttons, and the File menu


def test_open_video_and_open_session_are_the_file_menus_actions(window, qtbot, monkeypatch, clip_in_odd_folder):
    asked = record_dialogs(monkeypatch)
    show(window, qtbot)
    panel = body(window, 1)
    assert (panel.open_video_button.text(), panel.open_session_button.text()) == ("Open video", "Open session")
    (menu,) = [action.menu() for action in window.menuBar().actions() if action.text() == "File"]
    entries = [action for action in menu.actions() if not action.isSeparator()]
    assert [action.text() for action in entries] == ["Open video", "Open session", "Save session", "Quit"]
    assert entries[1] is panel.open_session_action and entries[2] is panel.save_action
    assert panel.save_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Save)  # Ctrl+S, Cmd+S on a Mac

    qtbot.mouseClick(panel.open_video_button, LEFT)
    qtbot.mouseClick(panel.open_session_button, LEFT)
    panel.open_session_action.trigger()
    assert [(parent, title) for parent, title, _, _ in asked.files] == [
        (window, "Open video"), (window, "Open session"), (window, "Open session")]
    filters = asked.files[1][2]
    assert "*.json" in filters[0] and filters[-1] == "All files (*)"
    asked.files[0][3](clip_in_odd_folder.path)  # the file chosen for Open video
    assert window.controller.video_path == clip_in_odd_folder.path


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
    kept = time_panel.settings()
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
    assert time_panel.settings().value(time_panel.MANIFEST_KEY) is None
    assert asked.messages == []  # said in the panel, not in a dialog


def test_the_manifest_button_waits_for_a_video(window):
    assert not body(window, 2).manifest_button.isEnabled() and not body(window, 2).fps_edit.isEnabled()
