"""The joins between the panels (task C9; SPEC 6.4, 8.1, 10.1, 10.2; decision X7).

Panels 5 to 9 were built side by side. These tests hold the places where two of them meet:

1. Export all and the flags table need no model: both run in the worker's one thread while no
   model is loaded, and the listing has no thread of its own any more. Tracking and outlines
   still wait for the model. A model is made in a thread that does nothing else and ends when the
   model is there; the worker's thread takes the model over and is the only one that uses it.
2. One lock: what a tracking run switches off is also off during an export, with the reason
   "An export is running."; Track is off during an export and Export all during a run. One
   function, `Jobs.writing()`, says whether and why.
3. Once a track has stored frames, the model of the session is fixed; the device is not.
4. Panel 9 needs attention when session.json or results.npz is newer than positions.csv.
5. The Quickstart marks the buttons of panels 8 and 9 in bold; docs/DEVELOPER.md names every
   module file, and no file that is not there.
6. gui/prompts.py and gui/panels/video_panel.py were split: what they offered is still there.

Expected values are the brief's sentences and SPEC 8.1's list of files, never the panels' own
output. A factory, a run or an export is parked on a gate (tests/gui/prompt_helpers.py), so
"during" is a state the test holds; where a file's time decides, the test sets it. Frames are
video frame numbers; (u, v) is in px of the video frame (SPEC 3.1).
"""

import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, Qt, QThread
from PySide6.QtWidgets import QAbstractButton

from export_helpers import calibrate
from export_panel_helpers import (END, FRAMES, P0_FILES, WRITTEN, export_panel, export_to_end, is_on, state, tracked,
                                  watch_export)
from export_panel_helpers import hint as export_hint
from export_panel_helpers import listed as files_listed
from finish_helpers import record_every_dialog
from gui_helpers import show
from helpers import click
from joins_helpers import (DAY, EXPORTING, RUNNING, WAIT_FOR_EXPORT, corrections, earlier_run, lockable, not_saying,
                           off, written_at)
from last_controls_helpers import (PACKAGE, REPO, choose, control_texts, let_run_end, marked_names, modules, parked,
                                   press_track, quickstart, record_confirm)
from outline_tracker import schema
from outline_tracker.gui.worker import worker_of
from outline_tracker.gui.worker_jobs import jobs_of
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import LEFT, NO_KEY, SAFETY_S, Gate, Watched, gui_thread, objects_panel, this_thread
from review_helpers import listed, listed_by, opened_run, review_panel
from session_helpers import body, read_json
from track_helpers import Tracked, ids, ready_to_track, results_of, run_to_end, track_panel, window_hint
from tracking_helpers import center

NO_INTERNET = "The model files could not be downloaded."
TRACK_NO_MODEL = f"The model could not be loaded. {NO_INTERNET}"
TRACK_LOADING = "The model is loading. Track is ready when the model is ready."
OUTLINE_NO_MODEL = f"The model could not be loaded, so no outline can be shown. {NO_INTERNET}"
MODEL_TIP = "The model that finds the outlines. EdgeTAM is the default."
DEVICE_TIP = "What the model runs on. auto takes the graphics processor if it works, else the processor (cpu)."
MODEL_FIXED = "The results were made with EdgeTAM. For another model, remove the objects or start a new session."
RESULTS_NEWER = "The results changed after the last export. Click Export all again."
SESSION_NEWER = "The session changed after the last export. Click Export all again."
STACK_BYTES = 64 * 1024 * 1024  # of a thread the model is made or used in (decision X7: 64 MiB)
DEVELOPER = REPO / "docs" / "DEVELOPER.md"
# What the two modules offered before they were split (as of the commit before task C9), private names too.
PROMPTS_OFFERED = [
    "BLACK", "BUSY_TEXT", "CASING", "DoubleClickWatch", "Drawn", "Found", "KINDS", "LINE_WIDTHS", "LOADING_TEXT",
    "MARK_SIZES", "MARK_SYMBOLS", "Mark", "NOTHING_FOUND", "NO_MODEL", "NO_OBJECT", "NO_OUTLINE", "PLUS_SIZE",
    "PointTool", "Prompts", "QColor", "QKeySequence", "QObject", "QShortcut", "Qt", "ResultsOnDisk", "Signal",
    "TOOL_TEXTS", "WAIT_TEXT", "WHITE", "_Shown", "colour_of", "crop_box", "dataclass", "decoder_tag", "found_in",
    "frame_hash", "frame_shown", "jobs_of", "marks_on", "np", "pg", "point_kind", "preview_input", "sys", "tracking"]
VIDEO_PANEL_OFFERED = [
    "CHECK_ERRORS", "CONTROL_HEIGHT", "CheckThread", "ElidedLabel", "HINT_DONE", "HINT_NAME", "HINT_VIDEO",
    "HINT_WARNINGS", "LABEL_WIDTH", "LARGEST_FRAME", "Message", "NAME_MISSING", "NOT_CHECKED", "NOT_WRITTEN",
    "NO_VALUE", "Path", "QEvent", "QGridLayout", "QHBoxLayout", "QLabel", "QLineEdit", "QObject", "QPushButton",
    "QSizePolicy", "QSpinBox", "QThread", "QVBoxLayout", "QWidget", "Qt", "RUNNING", "SESSION_FILTERS", "SPACING",
    "Signal", "VideoPanel", "_WheelGuard", "build", "clip_problem", "cv2", "dialogs", "field_label", "field_rows",
    "guard_wheel", "jobs_of", "size_text", "theme", "video"]


def exported_all(folder: Path, panel) -> bool:
    """Whether the run folder holds every file of SPEC 8.1 and the panel says what was written."""
    return ({entry.name for entry in folder.iterdir()} >= P0_FILES and set(files_listed(panel)) == WRITTEN
            and panel.message.kind == "success" and panel.message.text().startswith("Export all wrote 8 files in "))


def workers_thread(qtbot, worker) -> int:
    """The worker's one thread (X7), as `this_thread` numbers it: the thread in which the next
    task that is handed to the worker is called."""
    ran = []
    assert worker.run(lambda segmenter: ran.append(this_thread()), needs_model=False)
    qtbot.waitUntil(lambda: bool(ran))
    return ran[0]


# ---------------------------------------------------------------------------------------------
# 1. Export all and the flags need no model


def test_a_task_that_needs_no_model_is_taken_in_every_state_but_stopped(worker, qtbot):
    ran = []

    def task(segmenter):
        ran.append((segmenter, this_thread()))

    assert worker.state == "idle" and not worker.run(task)  # a task that needs the model waits for it, as before
    assert worker.run(task, needs_model=False) and worker.is_running()  # the thread starts for it
    qtbot.waitUntil(lambda: len(ran) == 1)
    worker.start(None, "edgetam", "cpu")  # a window without a model
    assert worker.state == "failed" and not worker.run(task)
    assert worker.run(task, needs_model=False)
    qtbot.waitUntil(lambda: len(ran) == 2)
    assert [segmenter for segmenter, _ in ran] == [None, None]  # there is no model to give it
    assert len({thread for _, thread in ran}) == 1 and gui_thread() not in {thread for _, thread in ran}
    worker.stop()
    assert not worker.run(task, needs_model=False) and not worker.run(task)
    assert not worker.is_running() and len(ran) == 2


def test_after_a_model_that_could_not_be_loaded_export_all_and_the_flags_run_in_the_workers_thread(
        window, qtbot, clip_in_odd_folder, monkeypatch):
    """A first start without the internet, so the model cannot be loaded: Export all and the flags
    table run all the same, and tracking and outlines still wait for the model."""
    record_every_dialog(monkeypatch)
    clip, loaded_in = clip_in_odd_folder, []

    def broken(model, device):
        loaded_in.append(this_thread())
        raise RuntimeError(NO_INTERNET)

    exports, listings = watch_export(monkeypatch), listed_by(monkeypatch)
    folder = earlier_run(window, qtbot, clip, broken)
    worker, panel, review, track = worker_of(window), export_panel(window), review_panel(window), track_panel(window)
    qtbot.waitUntil(lambda: worker.state == "failed")
    listed(qtbot, review)
    assert review.listing.problem == "" and review.listing.rows and review.listing.rows == listings.real(folder)
    assert is_on(window)  # Export all does not ask for the model
    export_to_end(qtbot, panel)
    assert exported_all(folder, panel) and state(window) == "done"
    assert export_hint(window).startswith("Exported at ")
    # one worker thread (X7): the one that takes the next task exported and listed
    assert len(loaded_in) == 1 and gui_thread() not in loaded_in
    in_worker = workers_thread(qtbot, worker)
    assert {thread for _, _, thread, _ in exports.calls} | set(listings.threads) == {in_worker} != {gui_thread()}
    # tracking and outlines still wait for the model
    objects = objects_panel(window)
    objects.add_object()
    assert objects.prompts.add_point(*center(clip, "B", 0), 1)
    assert not objects.prompts.busy and objects.prompts.outlines == {}
    assert objects.prompts.message == ("problem", OUTLINE_NO_MODEL)
    assert not track.track_button.isEnabled() and track.track_button.toolTip() == TRACK_NO_MODEL
    assert track.jobs.start() == TRACK_NO_MODEL and not track.jobs.running


def test_the_sentence_that_sent_the_student_to_the_command_line_is_gone(window, qtbot, clip_in_odd_folder,
                                                                        monkeypatch):
    from outline_tracker.gui.panels import export_panel as module

    record_every_dialog(monkeypatch)

    def broken(model, device):
        raise RuntimeError(NO_INTERNET)

    earlier_run(window, qtbot, clip_in_odd_folder, broken)
    qtbot.waitUntil(lambda: worker_of(window).state == "failed")
    panel = export_panel(window)
    for said in (panel.export_button.toolTip(), export_hint(window), panel.message.text()):
        assert "outline-tracker" not in said and "model" not in said
    texts = {name: value for name, value in vars(module).items() if isinstance(value, str) and name.isupper()}
    assert [name for name, value in texts.items() if "outline-tracker export" in value or "model" in value] == []


def test_while_a_load_never_ends_export_all_writes_every_file_in_the_workers_thread(window, qtbot,
                                                                                    clip_in_odd_folder, monkeypatch):
    """The brief's test for a model that is still loading. The factory stands at a gate: everything
    before `gate.open()` happens while the model is not there."""
    record_every_dialog(monkeypatch)
    clip, making = clip_in_odd_folder, []
    exports, listings = watch_export(monkeypatch), listed_by(monkeypatch)
    with Gate() as gate:
        model = Watched(ExactFake(clip))

        def never(key, device):
            making.append(this_thread())
            gate.park()
            return model

        folder = earlier_run(window, qtbot, clip, never)
        worker, panel, review, track = (worker_of(window), export_panel(window), review_panel(window),
                                        track_panel(window))
        qtbot.waitUntil(gate.parked.is_set)
        assert worker.state == "loading"
        # the flags of the earlier run are listed
        listed(qtbot, review)
        assert review.listing.problem == "" and review.listing.rows and review.listing.rows == listings.real(folder)
        # tracking and outlines wait for the model
        objects = objects_panel(window)
        objects.add_object()
        assert objects.prompts.add_point(*center(clip, "B", 0), 1)
        assert objects.prompts.busy and objects.prompts.outlines == {}
        assert not track.track_button.isEnabled() and track.track_button.toolTip() == TRACK_LOADING
        assert track.jobs.start() == TRACK_LOADING and not track.jobs.running
        # Export all does not: it is on, runs, and ends
        assert is_on(window)
        export_to_end(qtbot, panel)
        assert exported_all(folder, panel) and state(window) == "done"
        assert export_hint(window).startswith("Exported at ")
        # one worker thread (X7) did both; the model is being made in another, which does nothing else
        in_worker = workers_thread(qtbot, worker)
        assert {thread for _, _, thread, _ in exports.calls} | set(listings.threads) == {in_worker}
        assert len(making) == 1 and not {in_worker, gui_thread()} & set(making) and in_worker != gui_thread()
        # and the model is still not there: Track stays off with its reason, the outline is still awaited
        assert worker.state == "loading" and model.previews == [] and objects.prompts.busy
        assert not track.track_button.isEnabled() and track.track_button.toolTip() == TRACK_LOADING
        gate.open()
        qtbot.waitUntil(lambda: worker.ready and not objects.prompts.busy)
    assert "B" in objects.prompts.outlines and track.track_button.isEnabled()
    assert set(model.threads) == {in_worker}  # the model is used in the worker's one thread, never where it was made


def test_without_a_model_the_flags_are_listed_in_the_worker_thread_and_closing_ends_it(window, qtbot, dish_run,
                                                                                       tmp_path, monkeypatch):
    """The listing has no thread of its own (X7): without a model too, the worker's one thread
    lists, and closing the window ends that thread."""
    from outline_tracker.gui import review_table

    probe = []
    with Gate() as gate:
        calls = listed_by(monkeypatch, gate)
        review = opened_run(window, qtbot, dish_run, tmp_path / "run", model=False, wait=False)
        qtbot.waitUntil(gate.parked.is_set)
        assert review.worker.state == "failed" and review.worker.is_running()  # its thread started for the listing
        assert review.listing.pending and review.rows == []
        window.show_frame(60)  # the window does not wait for it
        assert window.view.frame == 60
        gate.open()
        listed(qtbot, review)
    assert review.rows and review.listing.problem == ""
    assert review.worker.run(lambda segmenter: probe.append(this_thread()), needs_model=False)  # the next task there
    qtbot.waitUntil(lambda: bool(probe))
    assert calls.threads == probe and gui_thread() not in probe  # one listing, in the thread that ran the probe
    window.close()
    assert not review.worker.is_running()
    review.listing.sync()  # and nothing starts a thread again
    assert not review.worker.is_running() and len(calls.threads) == 1
    assert not hasattr(review.listing, "is_running") and not hasattr(review_table, "QThread")


def test_a_model_is_made_in_a_thread_of_its_own_and_used_only_in_the_workers_thread(worker, qtbot):
    """X7 as it holds with a thread that only makes the model: the same stack where the model is
    made, the step before loading (torch's thread limit) in the thread that will use the model, and
    every call of the model in the worker's one thread."""
    order, makers, got, model = [], [], [], Watched()
    image = np.full((40, 60, 3), 255, np.uint8)  # an RGB image [row, column, 3]: light, with one dark block
    image[10:20, 10:20] = 0                      # whose center is (15, 15) px (SPEC 3.1)

    def factory(key, device):
        makers.append(QThread.currentThread())
        order.append(("factory", this_thread(), makers[0].stackSize()))
        return model

    worker.start(factory, "edgetam", "cpu", before_load=lambda: order.append(("before", this_thread(), None)))
    qtbot.waitUntil(lambda: worker.ready)
    in_worker = workers_thread(qtbot, worker)
    (_, before_in, _), (name, made_in, stack) = order  # the step before loading, then the factory, each once
    assert name == "factory" and before_in == in_worker != gui_thread()
    assert made_in not in (in_worker, gui_thread()) and stack == STACK_BYTES
    assert makers[0].isFinished() and worker.is_running()  # with the model ready, the worker's thread is the one left
    worker.request_preview(0, image, (0, 0), [click("A", 15.0, 15.0)])
    qtbot.waitUntil(lambda: not worker.busy)
    assert worker.run(lambda segmenter: got.append((segmenter, this_thread())))
    qtbot.waitUntil(lambda: bool(got))
    assert got == [(model, in_worker)] and model.threads == [in_worker]
    worker.stop()
    assert makers[0].isFinished() and not worker.is_running() and model.closed == 1


@pytest.mark.parametrize("first_fails", [False, True], ids=["the first is made", "the first fails"])
def test_a_model_asked_for_while_another_is_being_made_is_made_after_that_one_was_closed(worker, qtbot, first_fails):
    """One copy of the weights (X7), also with a thread that only makes the model: the second
    factory call begins when the first has returned and what it made is closed, by the worker's
    thread, which is where a model is taken over. Only the second one's answer counts; the trace
    of a first one that failed is kept for run.log."""
    events, states = [], []
    worker.state_changed.connect(lambda state, message: states.append(state))

    class Noting(ThresholdFake):
        def __init__(self, key):
            super().__init__()
            self.key = key

        def close(self):
            events.append(("closed", self.key))
            closed_in.append(this_thread())
            super().close()

    closed_in = []
    with Gate() as gate:
        def factory(key, device):
            events.append(("making", key))
            if key == "edgetam":
                gate.park()
                if first_fails:
                    raise OSError(NO_INTERNET)
            events.append(("made", key))
            return Noting(key)

        worker.start(factory, "edgetam", "cpu")
        qtbot.waitUntil(gate.parked.is_set)
        worker.start(factory, "sam2", "cpu")
        in_worker = workers_thread(qtbot, worker)  # the worker's thread has been told of the second load by now
        assert events == [("making", "edgetam")] and worker.state == "loading"  # not beside the first
        gate.open()
        qtbot.waitUntil(lambda: worker.ready)
    first = [] if first_fails else [("made", "edgetam"), ("closed", "edgetam")]
    assert events == [("making", "edgetam"), *first, ("making", "sam2"), ("made", "sam2")]
    assert closed_in == ([] if first_fails else [in_worker])
    assert states == ["loading", "loading", "ready"] and worker.wanted == ("sam2", "cpu")
    assert [trace.splitlines()[-1] for trace in worker.traces] == ([f"OSError: {NO_INTERNET}"] if first_fails else [])


@pytest.mark.parametrize("fails", [False, True], ids=["it is made", "it fails"])
def test_closing_while_a_model_is_being_made_waits_for_it_and_closes_what_was_made(worker, qtbot, fails):
    made, makers, inside = Watched(), [], threading.Event()
    with Gate() as gate:
        def factory(key, device):
            makers.append(QThread.currentThread())
            gate.park()
            if fails:
                raise OSError(NO_INTERNET)
            return made

        def hold(segmenter):  # keeps the worker's thread until `stop` asks it to end
            inside.set()
            worker.stopping.wait(SAFETY_S)

        worker.start(factory, "edgetam", "cpu")
        qtbot.waitUntil(gate.parked.is_set)
        assert worker.run(hold, needs_model=False)
        qtbot.waitUntil(inside.is_set)
        gate.open()
        worker.stop()  # the model may arrive before the worker's thread has ended, or after
    assert made.closed == (0 if fails else 1) and made.previews == []
    assert worker.state == "stopped" and not worker.ready and worker.device is None
    assert not worker.is_running() and makers[0].isFinished()  # no thread is left
    QCoreApplication.sendPostedEvents()  # what the worker's thread still reported before it ended
    # the trace of a load that failed is kept for run.log, once, whenever it arrived
    assert [trace.splitlines()[-1] for trace in worker.traces] == ([f"OSError: {NO_INTERNET}"] if fails else [])


# ---------------------------------------------------------------------------------------------
# 2. One lock


def a_tracked_object_and_a_new_one(window, qtbot, clip):
    """A is tracked on `FRAMES` with a scale; B is added after the run, fine with a typed window,
    selected, with the Positive tool chosen. Returns (ExportPanel, ObjectsPanel, ReviewPanel)."""
    panel = tracked(window, qtbot, clip)
    objects, review = objects_panel(window), review_panel(window)
    objects.add_object()
    objects.set_mode("fine")
    objects.set_fine_window(120)
    assert objects.prompts.selected == "B" and objects.prompts.tool_kind == "positive"
    return panel, objects, review


def press_export(qtbot, panel, gate) -> None:
    """Press Export all and wait until the export stands at `gate`, inside `export_all`."""
    panel.export_button.click()
    assert panel.exporting, panel.message.text() or panel.export_button.toolTip()
    qtbot.waitUntil(gate.parked.is_set)


def let_export_end(qtbot, panel, gate) -> None:
    gate.open()
    qtbot.waitUntil(lambda: not panel.exporting, timeout=30_000)


def test_the_one_function_gives_the_runs_reason_the_exports_reason_and_none_otherwise(window, qtbot,
                                                                                      clip_in_odd_folder,
                                                                                      monkeypatch):
    clip = clip_in_odd_folder
    with Gate() as run_gate:
        track, _ = ready_to_track(window, qtbot, clip, parked(clip, run_gate), end=END)
        calibrate(window.controller.session)
        window.controller.touch()
        jobs, panel = jobs_of(window), export_panel(window)
        assert jobs.writing() is None
        press_track(qtbot, track, run_gate)
        assert jobs.writing() == RUNNING
        # Export all is off during a run, with the run's reason
        assert not panel.export_button.isEnabled() and panel.export_button.toolTip().startswith(RUNNING)
        panel.export()
        assert not panel.exporting and jobs.writing() == RUNNING
        let_run_end(qtbot, track, run_gate)
    assert jobs.writing() is None and jobs.status == "complete"
    assert window.panels[6].state == "done"
    with Gate() as export_gate:
        watch_export(monkeypatch, gate=export_gate)
        press_export(qtbot, panel, export_gate)
        assert jobs.writing() == EXPORTING
        # Track is off during an export, with the export's reason, and the panel stays done
        assert not track.track_button.isEnabled() and track.track_button.toolTip() == EXPORTING
        assert window_hint(track) == EXPORTING and window.panels[6].state == "done"
        assert jobs.start() == EXPORTING and not jobs.running
        let_export_end(qtbot, panel, export_gate)
    assert jobs.writing() is None and panel.message.kind == "success"
    assert EXPORTING not in track.track_button.toolTip() and EXPORTING not in window_hint(track)


def test_during_an_export_every_part_a_run_switches_off_is_off_and_says_why(window, qtbot, clip_in_odd_folder,
                                                                            monkeypatch):
    panel, objects, review = a_tracked_object_and_a_new_one(window, qtbot, clip_in_odd_folder)
    parts, fixes, model_box = lockable(window), corrections(window), body(window, 7).model_box
    assert off(parts) == []  # B is selected: all on before the export
    objects.prompts.select("A")
    assert off(fixes) == []  # A is tracked: it can be fixed
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        everything = {**parts, **fixes, "model": model_box}
        assert sorted(off(everything)) == sorted(everything)
        assert not_saying(everything, EXPORTING) == []
        assert not panel.folder_button.isEnabled()  # Change folder is File > Save session as
        # what an export does not read stays usable, and so does looking at the flags
        assert window.menus.save_action.isEnabled() and window.menus.quit_action.isEnabled()
        assert body(window, 8).shape_box.isEnabled()
        let_export_end(qtbot, panel, gate)
    assert panel.message.kind == "success"
    assert off(fixes) == [] and not_saying(fixes, EXPORTING) == list(fixes)
    objects.prompts.select("B")
    assert off(parts) == [] and not_saying(parts, EXPORTING) == list(parts)
    assert all(part.toolTip() for part in parts.values())  # each has its own tip back
    assert panel.folder_button.isEnabled()


def test_during_an_export_a_click_on_the_picture_adds_no_point_and_the_tool_says_to_wait(window, qtbot,
                                                                                        clip_in_odd_folder,
                                                                                        monkeypatch):
    clip = clip_in_odd_folder
    panel, objects, _ = a_tracked_object_and_a_new_one(window, qtbot, clip)
    prompts, tool = objects.prompts, window.view.tool
    u, v = center(clip, "B", 0)
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        assert window.view.tool is tool  # the tool stays chosen, and says that it does nothing now
        assert window.tool_text.text() == WAIT_FOR_EXPORT and tool.cursor == Qt.CursorShape.ForbiddenCursor
        tool.click(u, v, LEFT, NO_KEY)  # what the view calls for a click on the picture
        assert not prompts.add_point(u, v, 1) and not prompts.add_point(u + 3, v, 0)
        assert not prompts.set_head(u, v)
        prompts.select("A")
        assert not prompts.undo()  # the Undo key
        a, b = window.controller.session.tracks
        assert b.prompts == [] and b.head_px is None
        assert [[tuple(point) for point in prompt.points_px] for prompt in a.prompts] == [[center(clip, "A", 0)]]
        let_export_end(qtbot, panel, gate)
    prompts.select("B")
    prompts.choose_tool("positive")
    assert window.tool_text.text() == "Positive: click on animal B."
    assert window.view.tool.cursor == Qt.CursorShape.CrossCursor
    window.view.tool.click(u, v, LEFT, NO_KEY)
    b = window.controller.session.tracks[1]
    assert [[tuple(point) for point in prompt.points_px] for prompt in b.prompts] == [[(u, v)]]
    qtbot.waitUntil(lambda: not prompts.busy)


def test_during_an_export_a_correction_is_refused_and_nothing_else_changes_its_input(window, qtbot,
                                                                                     clip_in_odd_folder,
                                                                                     monkeypatch):
    """A key, code, or the answer to a question asked before the export began, that reaches a
    function behind a button that is off."""
    clip, questions = clip_in_odd_folder, record_confirm(monkeypatch)
    loads = []
    panel, objects, review = a_tracked_object_and_a_new_one(window, qtbot, clip)
    worker_of(window).state_changed.connect(lambda state, message: loads.append(state))
    track = track_panel(window)
    objects.prompts.select("A")
    window.show_frame(0)
    objects.ask_remove()   # three questions asked before the export, answered during it
    review.retrack()       # A has its click on frame 0
    window.show_frame(10)
    review.end_track()
    assert [action for _, _, action, _ in questions.asked] == ["Remove", "Re-track from here", "End track here"]
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        for _, _, _, answer_yes in questions.asked:
            answer_yes()
        review.retrack()
        review.end_track()
        review.continue_as_new()
        assert len(questions.asked) == 3  # no correction asks now
        objects.add_object()
        objects.remove("A")
        objects.remove_button.click()
        objects.prompts.select("B")
        objects.set_mode("coarse")
        objects.set_fine_window(64)
        track.set_model("sam2")
        track.set_device("cpu")
        a, b = window.controller.session.tracks
        processing = window.controller.session.processing
        assert ids(window) == ["A", "B"] and a.ended_at is None and (b.mode, b.fine_window_px) == ("fine", 120)
        assert (processing.model, processing.device) == ("edgetam", "auto") and loads == []
        assert not review.jobs.running and review.jobs.start() == EXPORTING
        assert results_of(window).arrays("A").frames.tolist() == FRAMES
        let_export_end(qtbot, panel, gate)
    assert panel.message.kind == "success" and results_of(window).arrays("A").frames.tolist() == FRAMES


def test_no_other_module_of_the_window_names_panel_9_for_the_lock():
    """Every panel asks `Jobs.writing()`; none reads another panel's attribute."""
    gui = PACKAGE / "gui"
    others = [path for path in gui.rglob("*.py") if path.name != "export_panel.py"]
    naming = [path.relative_to(gui).as_posix() for path in others
              if re.search(r"ExportPanel|export_panel import|\.exporting\b", path.read_text(encoding="utf-8"))]
    assert naming == ["worker_jobs.py"]  # which keeps the lock: `Jobs.exporting`


# ---------------------------------------------------------------------------------------------
# 3. The model after results


def test_with_results_the_model_box_is_off_with_its_sentence_and_the_device_box_is_on(window, qtbot,
                                                                                    clip_in_odd_folder):
    clip, asked = clip_in_odd_folder, []

    def factory(model, device):
        asked.append((model, device))
        return Tracked(ExactFake(clip))

    window.segmenter_factory = factory
    track, objects = ready_to_track(window, qtbot, clip, end=10)
    model_box, device_box, processing = track.model_box, track.device_box, window.controller.session.processing
    assert model_box.isEnabled() and model_box.toolTip() == MODEL_TIP  # nothing is tracked yet
    assert device_box.isEnabled() and device_box.toolTip() == DEVICE_TIP
    run_to_end(qtbot, track)
    assert results_of(window).track_ids == ["A"]
    assert not model_box.isEnabled() and model_box.toolTip() == MODEL_FIXED
    assert device_box.isEnabled() and device_box.toolTip() == DEVICE_TIP
    track.set_model("sam2")  # the function behind the box
    assert processing.model == "edgetam" and asked == [("edgetam", "auto")]
    choose(device_box, "cpu")  # another device is no other model: the results stay comparable
    qtbot.waitUntil(lambda: track.worker.ready and len(asked) == 2)
    assert asked[-1] == ("edgetam", "cpu") and processing.device == "cpu"
    assert not model_box.isEnabled() and model_box.toolTip() == MODEL_FIXED
    objects.remove("A")  # the last object: no track has stored frames now
    assert ids(window) == [] and results_of(window).track_ids == []
    assert model_box.isEnabled() and model_box.toolTip() == MODEL_TIP
    choose(model_box, "sam2")
    qtbot.waitUntil(lambda: track.worker.ready and len(asked) == 3)
    assert asked[-1] == ("sam2", "cpu") and processing.model == "sam2"


def test_during_a_run_both_boxes_are_off_and_after_it_the_device_box_is_on_again(window, qtbot, clip_in_odd_folder):
    """After the run there are results, so the model box stays off, with its own sentence."""
    clip, asked = clip_in_odd_folder, []
    with Gate() as gate:
        segmenter = parked(clip, gate)

        def factory(model, device):
            asked.append((model, device))
            return segmenter

        window.segmenter_factory = factory
        panel, _ = ready_to_track(window, qtbot, clip, end=10)
        assert panel.model_box.isEnabled() and panel.device_box.isEnabled()
        press_track(qtbot, panel, gate)
        assert not panel.model_box.isEnabled() and not panel.device_box.isEnabled()
        assert panel.model_box.toolTip() == panel.device_box.toolTip() == RUNNING
        panel.set_model("sam2")  # the function behind the box
        panel.set_device("cpu")
        processing = window.controller.session.processing
        assert (processing.model, processing.device) == ("edgetam", "auto") and asked == [("edgetam", "auto")]
        let_run_end(qtbot, panel, gate)
    assert panel.device_box.isEnabled() and panel.device_box.toolTip() == DEVICE_TIP
    assert not panel.model_box.isEnabled() and panel.model_box.toolTip() == MODEL_FIXED
    assert panel.jobs.status == "complete" and asked == [("edgetam", "auto")]  # the run ended with its own model


# ---------------------------------------------------------------------------------------------
# 4. Exported files that are older than what they come from


def test_a_session_saved_after_the_export_needs_attention_and_a_new_export_is_done(window, qtbot,
                                                                                  clip_in_odd_folder):
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    assert state(window) == "done"  # a fresh export is never older: it saves the session first
    written_at(folder, results=(14, 0), session=(14, 1), positions=(14, 5))
    panel.refresh()
    assert state(window) == "done" and export_hint(window) == "Exported at 14:05. Export all replaces these files."
    window.controller.session.calibration.stick["length_mm"] = 25.0  # every position in mm is another one now
    window.controller.touch()
    window.controller.save_now()  # session.json is written now, long after 14:05 of that day
    assert state(window) == "attention" and export_hint(window) == SESSION_NEWER
    assert is_on(window) and panel.export_button.property("kind") == "primary"
    assert panel.open_button.property("kind") != "primary"
    export_to_end(qtbot, panel)
    assert state(window) == "done" and panel.message.kind == "success"
    assert export_hint(window).startswith("Exported at ") and panel.open_button.property("kind") == "primary"
    assert read_json(folder / schema.SESSION_JSON)["calibration"]["stick"]["length_mm"] == 25.0


def test_results_written_after_the_export_need_attention_and_a_new_export_is_done(window, qtbot, clip_in_odd_folder):
    """The panel's state comes from the times of three files: session.json counts as well as
    results.npz and positions.csv, so the test sets the time of each."""
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    assert state(window) == "todo" and panel.export_button.property("kind") == "primary"
    export_to_end(qtbot, panel)
    assert state(window) == "done"
    written_at(folder, results=(14, 0), session=(14, 1), positions=(14, 5))
    panel.refresh()
    assert state(window) == "done" and export_hint(window) == "Exported at 14:05. Export all replaces these files."
    written_at(folder, results=(14, 10))  # tracked again since
    panel.refresh()
    assert state(window) == "attention" and export_hint(window) == RESULTS_NEWER
    assert is_on(window) and panel.export_button.property("kind") == "primary"
    assert panel.message.isHidden() and panel.files_label.isHidden()  # what the older export wrote is not listed
    export_to_end(qtbot, panel)  # positions.csv is written now, long after 14:10 of that day
    assert state(window) == "done" and panel.message.kind == "success"


def test_an_export_on_the_disk_is_done_when_its_session_is_opened(window, second_window, qtbot,
                                                                  clip_in_odd_folder):
    """A session that is opened again: panel 9 is done when neither session.json nor results.npz
    is newer than positions.csv. A window that closes does not write a session that is on the
    disk as it is, and opening a session does not write it either."""
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    written_at(folder, results=(9, 30), session=(9, 30), positions=(9, 31))
    window.close()
    assert datetime.fromtimestamp((folder / schema.SESSION_JSON).stat().st_mtime) == datetime(*DAY, 9, 30)
    second_window.segmenter_factory = lambda model, device: ExactFake(clip_in_odd_folder)
    second_window.open_path(folder / schema.SESSION_JSON)
    show(second_window, qtbot)
    qtbot.waitUntil(lambda: worker_of(second_window).ready)
    again = export_panel(second_window)
    assert state(second_window) == "done" and is_on(second_window)
    assert export_hint(second_window) == "Exported at 09:31. Export all replaces these files."
    assert again.message.isHidden() and again.files_label.isHidden() and again.open_button.isEnabled()
    assert not second_window.controller.save_timer.isActive()  # nothing is due to be saved: nothing changed


# ---------------------------------------------------------------------------------------------
# 5. The two documents


def steps_of_the_quickstart() -> dict[int, str]:
    """The numbered steps of the README's Quickstart, by number (each is one line there)."""
    section = quickstart((REPO / "README.md").read_text(encoding="utf-8"))
    return {int(number): text for number, text in re.findall(r"^(\d+)\. +(.*)$", section, flags=re.MULTILINE)}


def test_the_quickstart_names_the_buttons_of_panels_8_and_9_in_bold_and_each_is_the_windows_own(window):
    steps, texts = steps_of_the_quickstart(), control_texts(window)
    for number, needed in ((8, "Re-track from here"), (9, "Export all")):
        buttons = {button.text() for button in body(window, number).findChildren(QAbstractButton)
                   if button.text()}  # a table has a corner button without a text
        marked = marked_names(steps[number])
        assert needed in buttons and needed in marked, (number, marked)
        assert [name for name in marked if name not in texts] == []
        unmarked = re.sub(r"\*\*[^*]+\*\*", "", steps[number])  # the step without what is in bold
        assert [text for text in buttons if text in unmarked] == [], number


def test_the_developers_page_names_every_module_file_in_a_table_and_every_module_it_names_exists():
    text = DEVELOPER.read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([\w/]+\.py)` \|", text, flags=re.MULTILINE)  # the first cell of a table's row
    assert {"gui/prompts.py", "gui/panels/video_panel.py", "gui/review_table.py"} <= set(modules())
    assert [module for module in modules() if module not in rows] == []
    assert [row for row in rows if rows.count(row) > 1] == []
    # none that does not exist: a path the page names is a file of the package, or of the repository
    named = set(re.findall(r"`([\w/]+\.py)`", text))
    assert [name for name in sorted(named) if not (PACKAGE / name).is_file() and not (REPO / name).is_file()] == []
    assert "tasks of their own" not in text  # panels 8 and 9 are in the table now


# ---------------------------------------------------------------------------------------------
# 6. The two files that were split


@pytest.mark.parametrize("name, offered", [("outline_tracker.gui.prompts", PROMPTS_OFFERED),
                                           ("outline_tracker.gui.panels.video_panel", VIDEO_PANEL_OFFERED)],
                         ids=["prompts.py", "video_panel.py"])
def test_every_name_a_split_module_offered_is_still_importable_from_it_and_it_is_under_400_lines(name, offered):
    import importlib

    module = importlib.import_module(name)
    assert [wanted for wanted in offered if not hasattr(module, wanted)] == []
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    assert len(lines) < 400, f"{Path(module.__file__).name} has {len(lines)} lines"


def test_the_parts_that_moved_are_the_same_objects_wherever_they_are_imported_from():
    from outline_tracker.gui import panel_parts, prompt_drawing, prompts
    from outline_tracker.gui.panels import video_panel

    for name in ("Message", "ElidedLabel", "guard_wheel", "field_label", "field_rows", "_WheelGuard"):
        assert getattr(video_panel, name) is getattr(panel_parts, name), name
    for name in ("Drawn", "colour_of", "BLACK", "CASING", "LINE_WIDTHS", "MARK_SIZES"):
        assert getattr(prompts, name) is getattr(prompt_drawing, name), name
