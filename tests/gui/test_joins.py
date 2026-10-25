"""The joins between the panels (1): Export all and the flags need no model, and how the model is made.

Task C9; SPEC 6.4, 8.1, 10.1, 10.2; decision X7. Panels 5 to 9 were built side by side. These tests hold the places
where two of them meet, numbered as that build step numbered them. This file holds the first:

1. Export all and the flags table need no model: both run in the worker's one thread while no
   model is loaded, and the listing has no thread of its own any more. Tracking and outlines
   still wait for the model. A model is made in a thread that does nothing else and ends when the
   model is there; the worker's thread takes the model over and is the only one that uses it.

The other joins are in tests/gui/test_joins_lock.py (2 and 3: the one lock, the model after results) and in
tests/gui/test_joins_exported_files.py (4 to 6: exported files older than what they come from, the Quickstart,
the two files that were split).

Expected values are the brief's sentences and SPEC 8.1's list of files, never the panels' own
output. A factory, a run or an export is parked on a gate (tests/support/prompt_helpers.py), so
"during" is a state the test holds; where a file's time decides, the test sets it. Frames are
video frame numbers; (u, v) is in px of the video frame (SPEC 3.1).
"""

import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QThread

from export_panel_helpers import P0_FILES, WRITTEN, export_panel, export_to_end, is_on, state, watch_export
from export_panel_helpers import hint as export_hint
from export_panel_helpers import listed as files_listed
from finish_helpers import record_every_dialog
from helpers import click
from joins_helpers import earlier_run
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import SAFETY_S, Gate, Watched, gui_thread, objects_panel, this_thread
from review_helpers import listed, listed_by, opened_run, review_panel
from track_helpers import track_panel
from tracking_helpers import center

NO_INTERNET = "The model files could not be downloaded."
TRACK_NO_MODEL = f"The model could not be loaded. {NO_INTERNET}"
TRACK_LOADING = "The model is loading. Track is ready when the model is ready."
OUTLINE_NO_MODEL = f"The model could not be loaded, so no outline can be shown. {NO_INTERNET}"
STACK_BYTES = 64 * 1024 * 1024  # of a thread the model is made or used in (decision X7: 64 MiB)


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
    from outline_tracker.gui.panels import review_table

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
