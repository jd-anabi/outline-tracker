"""Panel 9, Export (SPEC 8.1, 10.1, 10.2; task C6): what is in the panel, when Export all can be pressed, the run
folder, Open folder, and File > Export.

Export all itself, in the worker thread, with what it wrote, its warnings and a failure, is in
tests/gui/test_export_panel_export_all.py.

Expected values are the spec's and the design note's, not the panel's own output:
- SPEC 8.1 names the run folder `<video stem>_outline_<student>`.
The texts are the brief's and the design note's. Nothing is read from a pixel of text, and no test
waits with a delay: the export is parked on a gate (tests/support/prompt_helpers.py).

Frames are video frame numbers; sizes are bytes; times are s.
"""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QProgressBar, QPushButton

from export_helpers import calibrate, frames_of
from export_panel_helpers import (END, FRAMES, NAME_FIRST, RUN_FOLDER, export_panel, export_to_end, hint, is_off, is_on,
                                  state, tracked, watch_export)
from finish_helpers import menu_texts, record_every_dialog
from gui_helpers import show
from outline_tracker.core import schema
from outline_tracker.gui.panels import export_panel as export_module
from outline_tracker.gui.panels.export_panel import ExportPanel
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import Gate, gui_thread
from session_helpers import body
from track_helpers import NAME, Tracked, ready_to_track, run_to_end, track_panel
from tracking_helpers import center

START_HINT = "Export all writes the CSV files, the overlay video, the log and README.txt to the run folder."
VIDEO_FIRST = "Open your video in panel 1 first."
NO_RESULTS = "Nothing is tracked yet. Click Track in panel 7 (Track) first."
NO_FPS = "No frame rate is set. Type fps_true in panel 2 (Time). Export needs it to write times in s."
NO_SCALE = ("No scale is set. Place the stick in panel 3 (Calibration). Export needs the scale to write positions "
            "in mm.")  # the design note's sentence
TRACKING = "Tracking is running. Export all is ready when tracking has stopped."


# ---------------------------------------------------------------------------------------------
# What is in the panel, and when Export all can be pressed


def test_the_panels_controls_are_in_panel_9_under_its_hint(window):
    panel = export_panel(window)
    assert isinstance(panel, ExportPanel) and body(window, 9) is panel
    assert isinstance(panel.export_button, QPushButton) and panel.export_button.text() == "Export all"
    assert isinstance(panel.open_button, QPushButton) and panel.open_button.text() == "Open folder"
    assert isinstance(panel.folder_button, QPushButton) and panel.folder_button.text() == "Change folder"
    assert isinstance(panel.progress_bar, QProgressBar) and not panel.progress_bar.isTextVisible()
    assert not any(made.autoDefault() for made in (panel.export_button, panel.open_button, panel.folder_button))
    # an empty window: nothing to export, nowhere to look, and the hint the window started with
    assert is_off(window, NAME_FIRST, START_HINT) and state(window) == "todo"
    assert not panel.open_button.isEnabled() and not panel.folder_button.isEnabled()
    assert panel.progress_bar.isHidden() and panel.progress_label.isHidden()
    assert panel.message.isHidden() and panel.files_label.isHidden()
    assert panel.folder_label.full_text == "–" and not panel.exporting


def test_export_all_is_off_with_the_reason_until_there_is_something_to_export(window, qtbot, clip_in_odd_folder):
    clip, panel, controller = clip_in_odd_folder, export_panel(window), window.controller
    controller.set_student(NAME)
    panel.refresh()
    assert is_off(window, VIDEO_FIRST, START_HINT)
    window.segmenter_factory = lambda model, device: Tracked(ExactFake(clip))
    window.open_path(clip.path)
    show(window, qtbot)
    qtbot.waitUntil(lambda: worker_of(window).ready)
    assert is_off(window, NO_RESULTS) and state(window) == "todo"

    session = controller.session
    session.time.fps_true, session.clip.end = clip.scene.fps, END
    controller.touch()
    objects = body(window, 6)
    objects.add_object()
    assert objects.prompts.add_point(*center(clip, "A", 0), 1)
    tracking = track_panel(window)
    run_to_end(qtbot, tracking)
    assert tracking.jobs.status == "complete"
    # there are results now, and nothing to write them in mm with
    assert is_off(window, NO_SCALE) and state(window) == "attention"
    session.time.fps_true = None
    controller.touch()
    assert is_off(window, NO_FPS) and state(window) == "attention"  # panel 2 comes before panel 3
    calibrate(session)
    controller.touch()
    assert is_off(window, NO_FPS)
    session.time.fps_true = clip.scene.fps
    controller.touch()
    assert is_on(window) and hint(window) == START_HINT and state(window) == "todo"
    assert panel.export_button.property("kind") == "primary" and panel.open_button.property("kind") != "primary"
    assert not (controller.run_folder / schema.POSITIONS_CSV).exists()  # nothing was exported by looking


def test_export_all_is_off_while_tracking_runs(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:
        segmenter = Tracked(ExactFake(clip), gate=gate, park_at={3})
        tracking, _ = ready_to_track(window, qtbot, clip, segmenter, end=END)
        calibrate(window.controller.session)
        window.controller.touch()
        tracking.track()
        qtbot.waitUntil(gate.parked.is_set)
        assert tracking.jobs.running and is_off(window, TRACKING)
        gate.open()
        qtbot.waitUntil(lambda: not tracking.jobs.running)
    assert is_on(window)


# ---------------------------------------------------------------------------------------------
# The run folder: where it is, another one, Open folder


def test_the_run_folder_is_shown_and_change_folder_is_save_session_as(window, qtbot, clip_in_odd_folder, monkeypatch,
                                                                      tmp_path):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder = window.controller.run_folder
    assert panel.folder_label.full_text == RUN_FOLDER and panel.folder_label.toolTip() == str(run_folder)
    export_to_end(qtbot, panel)
    assert state(window) == "done" and not panel.files_label.isHidden()

    assert panel.folder_button.isEnabled()
    panel.folder_button.click()
    (parent, title, on_chosen), = asked.folders
    assert (parent, title) == (window, "Save session as")  # the File menu's item, and its function
    chosen = tmp_path / "second try"
    on_chosen(chosen)
    assert window.controller.run_folder == chosen
    assert panel.folder_label.full_text == "second try" and panel.folder_label.toolTip() == str(chosen)
    # the new folder has the session and the results, and no export yet
    assert {entry.name for entry in chosen.iterdir()} == {schema.SESSION_JSON, schema.RESULTS_NPZ}
    assert state(window) == "todo" and hint(window) == START_HINT and is_on(window)
    assert panel.message.isHidden() and panel.files_label.isHidden()
    export_to_end(qtbot, panel)
    assert (chosen / schema.POSITIONS_CSV).is_file() and (chosen / schema.OVERLAY_MP4).is_file()
    assert asked.messages == []


def test_change_folder_is_off_while_save_session_as_is(window, qtbot, clip_in_odd_folder):
    panel, item = export_panel(window), window.menus.save_as_action
    assert not item.isEnabled() and not panel.folder_button.isEnabled()
    window.open_path(clip_in_odd_folder.path)
    assert not item.isEnabled() and not panel.folder_button.isEnabled()  # a video, no name
    window.controller.set_student(NAME)
    assert item.isEnabled() and panel.folder_button.isEnabled()
    item.setEnabled(False)  # whoever switches the item off, for whatever reason
    assert not panel.folder_button.isEnabled()


def test_open_folder_shows_the_run_folder_once_there_is_one(window, qtbot, clip_in_odd_folder, monkeypatch):
    shown = []
    monkeypatch.setattr(export_module, "show_folder", lambda folder: shown.append(folder) or True)
    panel = export_panel(window)
    no_folder = "There is no run folder yet. Type your name and open your video in panel 1 first."
    assert not panel.open_button.isEnabled() and panel.open_button.toolTip() == no_folder
    panel.open_folder()  # the button's function, while the button is off
    assert shown == []
    window.controller.set_student(NAME)
    window.open_path(clip_in_odd_folder.path)
    run_folder = window.controller.run_folder
    assert not run_folder.exists() and not panel.open_button.isEnabled()  # named, and not on the disk yet
    window.controller.save_now()
    assert run_folder.is_dir() and panel.open_button.isEnabled()
    assert panel.open_button.toolTip() == "Show the run folder in the system's file browser"
    panel.open_button.click()
    assert shown == [run_folder]


def test_show_folder_hands_the_folder_to_the_systems_file_browser(monkeypatch, tmp_path):
    opened = []

    class Browser:
        @staticmethod
        def openUrl(url):
            opened.append(url)
            return True

    monkeypatch.setattr(export_module, "QDesktopServices", Browser)
    folder = tmp_path / "vidéo test ü" / RUN_FOLDER
    assert export_module.show_folder(folder) is True
    assert opened == [QUrl.fromLocalFile(str(folder))] and opened[0].isLocalFile()
    assert Path(opened[0].toLocalFile()) == folder  # the blank and the accents arrive as they are


def test_a_folder_that_the_system_cannot_show_is_said_in_the_status_bar(window, qtbot, clip_in_odd_folder,
                                                                        monkeypatch):
    monkeypatch.setattr(export_module, "show_folder", lambda folder: False)
    window.controller.set_student(NAME)
    window.open_path(clip_in_odd_folder.path)
    window.controller.save_now()
    export_panel(window).open_button.click()
    assert window.statusBar().currentMessage() == ("The run folder could not be opened. Point at its name in panel 9 "
                                                   "(Export) to see where it is.")


# ---------------------------------------------------------------------------------------------
# File > Export


def test_the_file_menu_has_export_between_save_session_as_and_quit(window):
    # both menus as SPEC 10.1 lists them
    assert menu_texts(window) == {
        "File": ["Open video", "Open session", "Save session", "Save session as", "Export", "Quit"],
        "Help": ["Quickstart", "About"]}
    made = window.menus
    file_menu, help_menu = [entry.menu() for entry in window.menuBar().actions()]
    assert [item for item in file_menu.actions() if not item.isSeparator()] == [
        made.open_action, made.open_session_action, made.save_action, made.save_as_action, made.export_action,
        made.quit_action]
    assert help_menu.actions() == [made.quickstart_action, made.about_action]
    assert (window.open_action, window.quit_action) == (made.open_action, made.quit_action)
    assert made.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)
    assert made.save_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Save)  # Ctrl+S, Cmd+S on a Mac
    assert made.quit_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Quit)
    assert made.export_action.shortcut().isEmpty()  # the spec gives Export no key
    assert not made.export_action.isEnabled()       # an empty window has nothing to export


def test_file_export_is_the_buttons_function(window, qtbot, clip_in_odd_folder, monkeypatch):
    panel = tracked(window, qtbot, clip_in_odd_folder)
    assert export_panel(window).export_action is window.menus.export_action
    watch = watch_export(monkeypatch)
    window.menus.export_action.trigger()
    assert panel.exporting
    qtbot.waitUntil(lambda: not panel.exporting)
    (folder, overlay, thread, _), = watch.calls
    assert folder == window.controller.run_folder and overlay is True and thread != gui_thread()
    assert panel.message.kind == "success" and state(window) == "done"
    assert frames_of(folder, schema.POSITIONS_CSV, "A") == FRAMES  # the 11 tracked frames are in the table
