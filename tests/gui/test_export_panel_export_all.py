"""Panel 9, Export (SPEC 8.1, 10.1, 10.2; task C6): Export all in the worker thread, what it wrote, its warnings, a
failure, and closing the window during an export.

What is in the panel, when Export all can be pressed, the run folder, Open folder and File > Export are in
tests/gui/test_export_panel.py.

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
waits with a delay: the export is parked on a gate (tests/support/prompt_helpers.py).

Frames are video frame numbers; sizes are bytes; times are s.
"""

import json
from datetime import datetime

import pytest

from export_panel_helpers import (NAME_FIRST, RUN_FOLDER, WRITTEN, Clock, export_to_end, hint, is_off, is_on, listed,
                                  state, tracked, watch_export)
from finish_helpers import record_every_dialog
from outline_tracker.core import schema
from outline_tracker.export.pipeline import size_text
from outline_tracker.gui.worker import worker_of
from outline_tracker.core.session import Session
from prompt_helpers import Gate, gui_thread
from session_helpers import read_json
from track_helpers import NAME, run_log

EXPORTING = "Export all is running. You can look at other frames meanwhile."


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


def test_the_exports_warnings_are_shown_as_warnings(window, qtbot, clip_in_odd_folder, monkeypatch, lock_file):
    asked = record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    run_folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    assert panel.message.kind == "success" and state(window) == "done"

    lock_file(schema.SHAPES_CSV)  # open in Excel, say
    gone = "The video dish_tracker.mp4 is at none of the places the session knows."

    def not_found(self, run_folder):
        raise FileNotFoundError(gone)

    monkeypatch.setattr(Session, "locate_video", not_found)  # the disk with the video was unplugged
    panel.clock = Clock(10.0, 75.0)
    export_to_end(qtbot, panel)
    locked = ("shapes.csv is open in another program: the new data are in shapes.new.csv next to it. Close that "
              "program and export again.")
    skipped = f"overlay.mp4 was skipped: {gone} Every other file was written."
    assert panel.report.warnings == [locked, skipped]  # the export's own sentences (outline_tracker/export/pipeline.py)
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
                                                                            monkeypatch, lock_file):
    record_every_dialog(monkeypatch)
    panel = tracked(window, qtbot, clip_in_odd_folder)
    watch = watch_export(monkeypatch)
    lock_file(schema.SESSION_JSON)
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
        opener = gate.open_when(worker.stopping)  # the GUI thread waits in close(): another thread opens the gate
        window.close()
        opener.join()
    assert not worker.is_running() and worker.traces == []
    assert not panel.exporting and len(watch.reports) == 1  # the export's end was taken over
    assert {entry.name for entry in run_folder.iterdir()} >= {schema.POSITIONS_CSV, schema.OVERLAY_MP4, schema.RUN_LOG}
