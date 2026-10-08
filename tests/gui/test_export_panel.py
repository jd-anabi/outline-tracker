"""Panel 9, Export (SPEC 8.1, 10.1, 10.2; task C6): the run folder, Export all in the worker thread,
what it wrote, its warnings, a failure, Open folder, and File > Export.

Expected values are the spec's and the design note's, not the panel's own output:
- SPEC 8.1 lists the files of a run folder; for one track A and the model edgetam, Export all
  writes positions.csv, edgetam/A.csv, shapes.csv, radial.csv, outlines.npz, README.txt,
  overlay.mp4 and a block of run.log (session.json and results.npz are there before it);
- a file's size is read from the disk here and written as the command line writes it
  (1 kB = 1000 bytes);
- the time an export took is the difference of two readings of a clock the test gives the panel:
  102.5 s - 100 s = 2.5 s;
- a file that another program holds open is simulated where Windows reports it, at the rename.
The texts are the brief's and the design note's. Nothing is read from a pixel of text, and no test
waits with a delay: the export is parked on a gate (tests/gui/prompt_helpers.py).

Frames are video frame numbers; sizes are bytes; times are s.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QProgressBar, QPushButton

from export_helpers import calibrate, frames_of
from export_panel_helpers import (END, FRAMES, WRITTEN, Clock, export_panel, export_to_end, hint, is_off, is_on,
                                  listed, lock, state, tracked, watch_export)
from finish_helpers import menu_texts, record_every_dialog
from gui_helpers import show
from outline_tracker import schema
from outline_tracker.cli_export import size_text
from outline_tracker.gui.panels import export_panel as export_module
from outline_tracker.gui.panels.export_panel import ExportPanel
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Session
from prompt_helpers import Gate, gui_thread
from session_helpers import body, read_json
from track_helpers import NAME, Tracked, ready_to_track, run_log, run_to_end, track_panel
from tracking_helpers import center

START_HINT = "Export all writes the CSV files, the overlay video, the log and README.txt to the run folder."
NAME_FIRST = "Type your name in panel 1 first."
VIDEO_FIRST = "Open your video in panel 1 first."
NO_RESULTS = "Nothing is tracked yet. Click Track in panel 7 (Track) first."
NO_FPS = "No frame rate is set. Type fps_true in panel 2 (Time). Export needs it to write times in s."
NO_SCALE = ("No scale is set. Place the stick in panel 3 (Calibration). Export needs the scale to write positions "
            "in mm.")  # the design note's sentence
TRACKING = "Tracking is running. Export all is ready when tracking has stopped."
EXPORTING = "Export all is running. You can look at other frames meanwhile."
RUN_FOLDER = "dish_tracker_outline_Ada"  # SPEC 8.1: <video stem>_outline_<student>


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
# Export all


def test_export_all_runs_in_the_worker_thread_and_the_panel_says_what_it_writes(window, qtbot, clip_in_odd_folder,
                                                                              monkeypatch):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder = window.controller.run_folder
    assert run_folder == clip_in_odd_folder.path.parent / RUN_FOLDER and is_on(window)
    panel.clock = Clock(100.0, 102.5)
    with Gate() as gate:
        watch = watch_export(monkeypatch, gate=gate, says=["A_old.csv: removed (the session has no such track)."])
        window.controller.session.axes.angle_deg = 12.5  # a change that is not saved yet
        window.controller.touch()
        panel.export_button.click()
        qtbot.waitUntil(gate.parked.is_set)  # the GUI thread goes on while the export stands still
        (folder, overlay, thread, saved), = watch.calls
        assert folder == run_folder and overlay is True and thread != gui_thread()
        assert json.loads(saved)["axes"]["angle_deg"] == 12.5  # the session was saved before the worker read it
        assert panel.exporting and is_off(window, EXPORTING)
        assert not panel.progress_bar.isHidden() and not panel.progress_label.isHidden()
        assert (panel.progress_bar.minimum(), panel.progress_bar.maximum()) == (0, 0)  # busy: no invented progress
        qtbot.waitUntil(lambda: panel.progress_label.text() == "A_old.csv: removed (the session has no such track).")
        panel.export()                          # the button's function again,
        window.menus.export_action.trigger()    # and the menu's: one export at a time
        assert len(watch.calls) == 1 and panel.message.isHidden() and panel.files_label.isHidden()
        gate.open()
        qtbot.waitUntil(lambda: not panel.exporting)

    report, = watch.reports
    assert {path.relative_to(run_folder).as_posix() for path in report.files} == WRITTEN
    assert listed(panel) == {name: size_text((run_folder / name).stat().st_size) for name in WRITTEN}
    assert list(listed(panel)) == [path.relative_to(run_folder).as_posix() for path in report.files]
    assert panel.message.kind == "success" and panel.message.text() == "Export all wrote 8 files in 2.5 s."
    assert panel.seconds == pytest.approx(2.5) and panel.report is report
    assert panel.progress_bar.isHidden() and panel.progress_label.isHidden()
    assert asked.messages == [] and worker_of(window).traces == []  # no dialog for a success
    # the panel is done, and Open folder is the next step
    written = datetime.fromtimestamp((run_folder / schema.POSITIONS_CSV).stat().st_mtime)
    assert state(window) == "done" and hint(window) == f"Exported at {written:%H:%M}. Export all replaces these files."
    assert is_on(window) and panel.export_button.property("kind") != "primary"
    assert panel.open_button.property("kind") == "primary"


def test_the_exports_warnings_are_shown_as_warnings(window, qtbot, clip_in_odd_folder, monkeypatch):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    assert panel.message.kind == "success" and state(window) == "done"

    lock(monkeypatch, schema.SHAPES_CSV)  # open in Excel, say
    gone = "The video dish_tracker.mp4 is at none of the places the session knows."

    def not_found(self, run_folder):
        raise FileNotFoundError(gone)

    monkeypatch.setattr(Session, "locate_video", not_found)  # the disk with the video was unplugged
    panel.clock = Clock(10.0, 75.0)
    export_to_end(qtbot, panel)
    locked = ("shapes.csv is open in another program: the new data are in shapes.new.csv next to it. Close that "
              "program and export again.")
    skipped = f"overlay.mp4 was skipped: {gone} Every other file was written."
    assert panel.report.warnings == [locked, skipped]  # the export's own sentences (outline_tracker/export.py)
    assert panel.message.kind == "warning"
    assert panel.message.text() == f"Export all wrote 7 files in 1 min 5 s, with 2 warnings.\n\n{locked}\n\n{skipped}"
    names = (WRITTEN - {schema.SHAPES_CSV, schema.OVERLAY_MP4}) | {"shapes.new.csv"}
    assert listed(panel) == {name: size_text((run_folder / name).stat().st_size) for name in names}
    assert state(window) == "attention" and hint(window) == "The export has 2 warnings. Read them below."
    assert asked.messages == []  # a warning is never a dialog
    assert window.statusBar().currentMessage() == "Export all wrote 7 files in 1 min 5 s, with 2 warnings."


def stopped_by_a_setting(window, watch_with) -> str:
    window.controller.session.processing.radial_step_deg = 10  # radial.csv has a column every 5 degrees
    window.controller.touch()
    return "radial_step_deg = 10 cannot be exported"


def stopped_by_two_locked_files(window, watch_with) -> str:
    folder = window.controller.run_folder
    watch_with(error=PermissionError(
        "Could not write positions.csv: it is open in another program, and so is positions.new.csv. Close them "
        f"(folder: {folder}) and try again."))
    # the sentence names the folder, never its long path
    return ("Could not write positions.csv: it is open in another program, and so is positions.new.csv. Close them "
            f"(folder: {RUN_FOLDER}) and try again.")


def stopped_by_a_fault(window, watch_with) -> str:
    watch_with(error=KeyError("heads"))
    return "The export stopped on something it cannot use (KeyError: 'heads')."


@pytest.mark.parametrize("stop", [stopped_by_a_setting, stopped_by_two_locked_files, stopped_by_a_fault])
def test_a_failure_is_said_in_the_panel_and_in_a_dialog_and_its_trace_goes_to_run_log(window, qtbot, monkeypatch,
                                                                                    clip_in_odd_folder, stop):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder = window.controller.run_folder
    sentence = stop(window, lambda **how: watch_export(monkeypatch, **how))
    export_to_end(qtbot, panel)

    assert panel.message.kind == "problem" and panel.message.text().startswith("Export all stopped with an error. ")
    assert sentence in panel.message.text() and str(run_folder.parent) not in panel.message.text()
    assert "Traceback" not in panel.message.text()
    (parent, kind, text), = asked.messages
    heading, said, what_next = text.split("\n")
    assert (parent, kind, heading) == (window, "problem", "Export all stopped with an error")
    assert sentence in said and "run.log" in what_next and "Traceback" not in text
    assert state(window) == "attention" and hint(window) == "Export all stopped. See the message below."
    assert panel.report is None and panel.files_label.isHidden()
    assert not (run_folder / schema.POSITIONS_CSV).exists()
    log = run_log(run_folder)
    assert log.count("Traceback (most recent call last):") == 1 and f"Export all in the run folder {RUN_FOLDER}." in log
    assert len(worker_of(window).traces) == 1
    assert is_on(window)  # and it can be tried again


def test_without_a_name_export_says_what_is_missing_and_nothing_starts(window, qtbot, clip_in_odd_folder,
                                                                       monkeypatch):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    watch = watch_export(monkeypatch)
    window.controller.set_student("")
    assert is_off(window, NAME_FIRST)
    panel.export()
    assert (panel.message.kind, panel.message.text()) == ("problem", NAME_FIRST)
    assert watch.calls == [] and not panel.exporting and asked.messages == []
    window.controller.set_student(NAME)
    assert is_on(window) and panel.message.isHidden()


def test_a_session_that_cannot_be_saved_is_not_exported_from_its_older_file(window, qtbot, clip_in_odd_folder,
                                                                            monkeypatch):
    record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    watch = watch_export(monkeypatch)
    lock(monkeypatch, schema.SESSION_JSON)
    window.controller.session.calibration.stick["length_mm"] = 25.0  # the correction the export is for
    window.controller.touch()
    panel.export_button.click()
    told = ("session.json is open in another program. Close it there. Until then the session is saved as "
            "session.new.json in the run folder.")  # the controller's sentence
    assert (panel.message.kind, panel.message.text()) == ("problem", told)
    assert watch.calls == [] and not panel.exporting
    assert read_json(window.controller.run_folder / schema.SESSION_JSON)["calibration"]["stick"]["length_mm"] == 30.0


def test_closing_the_window_during_an_export_waits_for_it_and_ends_the_thread(window, qtbot, clip_in_odd_folder,
                                                                             monkeypatch):
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder, worker = window.controller.run_folder, worker_of(window)
    with Gate() as gate:
        watch = watch_export(monkeypatch, gate=gate)
        panel.export_button.click()
        qtbot.waitUntil(gate.parked.is_set)
        opener = threading.Timer(0.2, gate.open)  # the GUI thread waits in close(): another thread opens the gate
        opener.start()
        window.close()
        opener.join()
    assert not worker.is_running() and worker.traces == []
    assert not panel.exporting and len(watch.reports) == 1  # the export's end was taken over
    assert {entry.name for entry in run_folder.iterdir()} >= {schema.POSITIONS_CSV, schema.OVERLAY_MP4, schema.RUN_LOG}


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
