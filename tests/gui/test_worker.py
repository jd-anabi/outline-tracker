"""The worker thread (SPEC 5, 6.4, 10.2; decision X7; task C4): the model loads in it after the
window is shown, and it makes the outlines that are shown after a click.

What is checked, with a stand-in segmenter and no torch:
- nothing waits for the model: with the stand-in parked on a gate inside `preview`, the call that
  asked for the outline has returned, the window's event loop still runs, and of the requests
  made meanwhile only the newest one's outline is shown;
- what the worker reports arrives in the GUI thread;
- the model is made in the worker thread, after the window is shown, and until it reports ready
  nothing is `ready`; a model that cannot be loaded, or that fails on a frame, is reported in one
  plain line;
- the trace of such a failure is not shown, but it is written to the log when it happens and
  kept in `Worker.traces` (SPEC 10.2), also when nobody waits for the result any more;
- closing the window stops the thread.
A gate is a `threading.Event` (tests/gui/prompt_helpers.py): no test waits with a delay.

Expected positions come from the synthetic clips: the disks of `disk_scene` (centers within
0.25 px through `ThresholdFake`, the plan's measured condition), and for `ExactFake` the true mask
of the dish scene. The dish crop of that scene at 320 x 240 px is columns 49 to 271 and rows 8 to
231 (tests/tracking_helpers.py, `DISH_BOX`).
"""

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, QTimer

import helpers
import tracking_helpers
from gui_helpers import show
from outline_tracker.frame_source import FrameSource
from outline_tracker.gui.prompts import Prompts
from outline_tracker.gui.worker import Worker, before_load_for, worker_of
from outline_tracker.measure import mask_center
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import Gate, Watched, clicked_object, gui_thread, logged, objects_panel, panel_with, this_thread
from tracking_helpers import DISH_BOX, center, dish_circle

LOG = "outline_tracker.gui.worker"  # the logger the worker writes the trace of a failure to


class Receiver(QObject):
    """An object of the GUI thread that keeps what a worker reports, and in which thread it arrived."""

    def __init__(self, worker):
        super().__init__()
        self.states, self.previews, self.failures, self.threads = [], [], [], []
        worker.state_changed.connect(self.state)
        worker.preview_done.connect(self.preview)
        worker.preview_failed.connect(self.failed)

    def state(self, state, message):
        self.states.append((state, message))
        self.threads.append(this_thread())

    def preview(self, preview):
        self.previews.append(preview)
        self.threads.append(this_thread())

    def failed(self, serial, reason):
        self.failures.append((serial, reason))
        self.threads.append(this_thread())


# Overrides the shared `worker` of tests/gui/conftest.py: this one also keeps what it reports, in `worker.heard`.
@pytest.fixture
def worker(qtbot):
    """A worker of its own, not started; its thread is stopped after the test."""
    made = Worker()
    made.heard = Receiver(made)
    yield made
    made.stop()


@pytest.fixture
def frame_0(disk_clip):
    """Frame 0 of the disk clip, an RGB array [row, column, 3]."""
    with FrameSource(disk_clip.path) as source:
        return np.array(source.get(0))


@pytest.fixture
def slot_calls(monkeypatch):
    """Every call of the slot that takes the worker's outlines in the window (`Prompts.show_preview`)
    as (thread, preview). Ask for it before `window`, which connects the slot when it is made."""
    calls, original = [], Prompts.show_preview

    def show_preview(self, preview):
        calls.append((this_thread(), preview))
        return original(self, preview)

    monkeypatch.setattr(Prompts, "show_preview", show_preview)
    return calls


def loop_runs(qtbot) -> None:
    """Pass only when the GUI thread's event loop gets to an event posted now."""
    fired = []
    QTimer.singleShot(0, lambda: fired.append(True))
    qtbot.waitUntil(lambda: fired == [True])


# ---------------------------------------------------------------------------------------------
# A request returns at once, and the newest one wins


def test_a_request_returns_at_once_and_of_two_made_meanwhile_only_the_second_is_shown(worker, qtbot, disk_clip,
                                                                                    frame_0):
    with Gate() as gate:
        segmenter = Watched(gate=gate, park_at={0})
        worker.start(lambda model, device: segmenter, "edgetam", "cpu")
        qtbot.waitUntil(lambda: worker.ready)
        asked = [worker.request_preview(0, frame_0, (0, 0), [helpers.click("A", *center(disk_clip, "A", 0))])]
        qtbot.waitUntil(gate.parked.is_set)
        # the stand-in stands still inside preview(), and this test goes on: the call had returned
        assert worker.busy and worker.heard.previews == []
        loop_runs(qtbot)
        for name in "BC":
            asked.append(worker.request_preview(0, frame_0, (0, 0), [helpers.click(name, *center(disk_clip, name, 0))]))
        loop_runs(qtbot)
        assert worker.busy and worker.heard.previews == [] and len(segmenter.previews) == 1
        gate.open()
        qtbot.waitUntil(lambda: not worker.busy)
    assert asked == sorted(set(asked))  # every request has a number of its own, rising
    (shown,) = worker.heard.previews  # the first one's result is out of date, the second never ran
    assert (shown.serial, shown.frame, shown.offset) == (asked[2], 0, (0, 0))
    (result,) = shown.results
    assert result.obj_id == "C"
    u, v, area = mask_center(result.mask)
    cu, cv = center(disk_clip, "C", 0)
    assert area > 0 and abs(u + result.offset[0] - cu) < 0.25 and abs(v + result.offset[1] - cv) < 0.25
    assert [[name for name, _, _ in call] for call in segmenter.previews] == [["A"], ["C"]]
    assert worker.heard.failures == []
    # the model worked in the worker thread; what it found arrived in the GUI thread
    assert gui_thread() not in segmenter.threads and set(worker.heard.threads) == {gui_thread()}


def test_in_the_window_only_the_newest_outline_is_drawn_by_a_slot_in_the_gui_thread(slot_calls, window, qtbot,
                                                                                  disk_clip):
    with Gate() as gate:
        segmenter = Watched(gate=gate, park_at={0})
        panel, _ = clicked_object(window, qtbot, disk_clip, "A", segmenter=segmenter)  # the first request
        prompts, viewport = panel.prompts, window.view.viewport()
        qtbot.waitUntil(gate.parked.is_set)
        # the busy mark, while the model works
        assert prompts.busy and prompts.outlines == {} and not panel.busy_label.isHidden()
        assert viewport.cursor().shape() == Qt.CursorShape.BusyCursor
        assert window.tool_text.text() == "The outline is being updated."
        loop_runs(qtbot)  # the window is not blocked
        for name in "BC":  # two more requests while the first one is parked; the clicks are kept
            panel.add_object()
            assert prompts.add_point(*center(disk_clip, name, 0), 1)
        assert prompts.outlines == {} and slot_calls == []
        gate.open()
        qtbot.waitUntil(lambda: not prompts.busy)
    assert set(prompts.outlines) == {"A", "B", "C"}
    assert [[name for name, _, _ in call] for call in segmenter.previews] == [["A"], ["A", "B", "C"]]
    ((thread, preview),) = slot_calls  # one result reached the slot: the newest request's
    assert thread == gui_thread() and [result.obj_id for result in preview.results] == ["A", "B", "C"]
    assert gui_thread() not in segmenter.threads
    assert viewport.cursor().shape() == Qt.CursorShape.CrossCursor and panel.busy_label.isHidden()
    assert window.tool_text.text() == "Positive: click on animal C."


def test_points_and_outlines_show_only_on_their_frame(window, qtbot, disk_clip):
    segmenter = Watched()
    panel, _ = clicked_object(window, qtbot, disk_clip, segmenter=segmenter)
    prompts = panel.prompts
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    window.navigation.step(1)  # frame 2: the clicks and the outline belong to frame 0
    assert prompts.marks == [] and prompts.outlines == {}
    window.navigation.step(-1)
    assert [mark.kind for mark in prompts.marks] == ["positive"]
    assert "A" in prompts.outlines  # the same clicks: the outline is shown again, not asked for again
    assert len(segmenter.previews) == 1 and not prompts.busy


# ---------------------------------------------------------------------------------------------
# Loading the model


def test_the_model_loads_in_the_worker_after_the_window_is_shown_and_nothing_is_ready_before(window, qtbot,
                                                                                           disk_clip):
    made = []
    with Gate() as gate:
        def factory(model, device):
            made.append((model, device, this_thread()))
            gate.park()
            return ThresholdFake()

        window.segmenter_factory = factory
        worker, panel = worker_of(window), objects_panel(window)
        assert panel.worker is worker and worker_of(window) is worker  # one worker per window
        assert (worker.state, worker.ready, worker.is_running(), made) == ("idle", False, False, [])
        window.open_path(disk_clip.path)
        show(window, qtbot)
        qtbot.waitUntil(gate.parked.is_set)
        assert (worker.state, worker.ready) == ("loading", False)  # what a Track button waits for
        assert panel.model_label.text() == "Model loading"
        loop_runs(qtbot)
        # a click made now is kept, and its outline comes when the model is there
        panel.add_object()
        assert panel.prompts.add_point(*center(disk_clip, "A", 0), 1)
        assert panel.prompts.busy and panel.prompts.outlines == {}
        gate.open()
        qtbot.waitUntil(lambda: worker.ready)
        qtbot.waitUntil(lambda: "A" in panel.prompts.outlines)
    ((model, device, thread),) = made  # made once, with the session's model and device, off the GUI thread
    assert (model, device) == ("edgetam", "auto") and thread != gui_thread()
    assert worker.state == "ready" and panel.model_label.text() == "Model ready"


def test_what_the_worker_reports_about_the_model_arrives_in_the_gui_thread(worker, qtbot):
    worker.start(helpers.stand_in_segmenter, "edgetam", "cpu")
    qtbot.waitUntil(lambda: worker.ready)
    assert [state for state, _ in worker.heard.states] == ["loading", "ready"]
    assert set(worker.heard.threads) == {gui_thread()}
    assert worker.is_running()


def test_a_model_that_cannot_be_loaded_is_reported_in_one_plain_line(window, qtbot, disk_clip):
    def factory(model, device):
        raise OSError("The model files could not be downloaded.\nTraceback (most recent call last):\n  File ...")

    window.segmenter_factory = factory
    worker, panel = worker_of(window), objects_panel(window)
    window.open_path(disk_clip.path)
    show(window, qtbot)
    qtbot.waitUntil(lambda: worker.state == "failed")
    assert not worker.ready and worker.message == "The model files could not be downloaded."
    assert panel.model_label.text() == "Model not loaded"
    assert panel.model_label.toolTip() == "The model files could not be downloaded."
    # the clicks still work; the panel says why no outline appears
    panel.add_object()
    assert panel.prompts.add_point(*center(disk_clip, "A", 0), 1)
    assert not panel.prompts.busy and panel.prompts.outlines == {}
    assert panel.prompts.message == ("problem", "The model could not be loaded, so no outline can be shown. "
                                                "The model files could not be downloaded.")


def test_a_window_without_a_factory_has_no_model_and_starts_no_thread(worker, qtbot):
    worker.start(None, "edgetam", "cpu")
    assert (worker.state, worker.ready, worker.is_running()) == ("failed", False, False)
    assert worker.heard.states == [("failed", "This window was made without a model.")]


def test_the_real_models_factory_gets_the_thread_limit_and_a_stand_in_gets_none():
    from outline_tracker.from_tracker import load_segmenter
    from outline_tracker.segmenter import hf

    assert before_load_for(load_segmenter) is hf.reserve_ui_thread
    assert before_load_for(helpers.stand_in_segmenter) is None
    assert before_load_for(None) is None


def test_the_step_before_loading_runs_in_the_worker_just_before_the_factory(worker, qtbot):
    order = []

    def factory(model, device):
        order.append(("factory", this_thread()))
        return ThresholdFake()

    worker.start(factory, "edgetam", "cpu", before_load=lambda: order.append(("before", this_thread())))
    qtbot.waitUntil(lambda: worker.ready)
    assert [name for name, _ in order] == ["before", "factory"]
    assert gui_thread() not in [thread for _, thread in order]


# ---------------------------------------------------------------------------------------------
# A preview that fails, and the frame the model is shown


def test_a_model_that_fails_on_a_frame_is_reported_in_the_panel(window, qtbot, disk_clip):
    class Broken(ThresholdFake):
        asked = 0

        def preview(self, image, prompts):
            self.asked += 1
            raise RuntimeError("The model ran out of memory.\n  File \"somewhere.py\", line 1")

    segmenter = Broken()
    panel, (u, v) = clicked_object(window, qtbot, disk_clip, segmenter=segmenter)
    prompts = panel.prompts
    qtbot.waitUntil(lambda: not prompts.busy)
    assert prompts.message == ("problem", "The outline could not be made. The model ran out of memory.")
    assert prompts.outlines == {} and segmenter.asked == 1
    assert window.controller.session.tracks[0].prompts[0].points_px == [[u, v]]  # the click is kept
    # the same clicks are not tried again by themselves; the next click is
    window.controller.touch()
    assert not prompts.busy
    assert prompts.add_point(u + 30, v, 0)
    qtbot.waitUntil(lambda: segmenter.asked == 2 and not prompts.busy)


def test_the_model_is_shown_the_part_of_the_frame_that_tracking_shows_it(window, qtbot, dish_clip):
    # With a dish circle and "Crop to dish" on, tracking shows the model the dish square (SPEC 6.2):
    # the preview shows it the same part, and the outline comes back to the full frame's coordinates.
    segmenter = tracking_helpers.Watched(ExactFake(dish_clip))
    panel = panel_with(window, qtbot, dish_clip, segmenter)
    window.controller.session.circle = dish_circle(dish_clip.scene)
    window.controller.touch()
    panel.add_object()
    assert panel.prompts.add_point(*center(dish_clip, "A", 0), 1)
    qtbot.waitUntil(lambda: "A" in panel.prompts.outlines)
    c0, r0, width, height = DISH_BOX
    assert segmenter.views == [(0, (c0, r0), (width, height))]
    ((shape, (prompt,)),) = segmenter.previews
    assert shape == (height, width, 3)
    u, v = center(dish_clip, "A", 0)
    assert prompt.points_px == [(u - c0, v - r0)]  # the click, in px of the square
    # every point of the outline lies on the true outline: the true signed distance there is 0
    outline = panel.prompts.outlines["A"]
    distance = dish_clip.logits("A", 0)
    columns, rows = np.floor(outline[:, 0]).astype(int), np.floor(outline[:, 1]).astype(int)
    assert np.abs(distance[rows, columns]).max() < 1.0  # px, read at the nearest pixel center
    # and it encloses the true mask: the centroid of the area inside the polygon (shoelace formula) is the mask's
    x, y = outline[:, 0], outline[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    inside_u = ((x + np.roll(x, -1)) * cross).sum() / (3 * cross.sum())
    inside_v = ((y + np.roll(y, -1)) * cross).sum() / (3 * cross.sum())
    true_u, true_v, _ = mask_center(dish_clip.mask("A", 0))
    assert abs(inside_u - true_u) < 0.75 and abs(inside_v - true_v) < 0.75


# ---------------------------------------------------------------------------------------------
# The trace of a failure: not shown, but written to the log at once and kept (SPEC 10.2)


def test_a_model_that_cannot_be_loaded_leaves_its_trace_in_the_log_and_with_the_worker(worker, qtbot, caplog):
    def broken_factory(model, device):
        raise OSError("The model files could not be downloaded.")

    worker.start(broken_factory, "sam2", "mps")
    qtbot.waitUntil(lambda: worker.state == "failed")
    assert worker.heard.states[-1] == ("failed", "The model files could not be downloaded.")  # what is shown
    (trace,) = worker.traces
    first, *_, last = trace.splitlines()
    assert "sam2" in first and "mps" in first  # what was being loaded, and for which device
    assert "Traceback (most recent call last):" in trace and "in broken_factory" in trace  # where it failed
    assert last == "OSError: The model files could not be downloaded."
    assert logged(caplog, LOG) == [trace]  # the same text was written when it happened


def test_a_model_that_fails_on_frames_leaves_a_trace_each_time_and_the_newest_are_kept(worker, qtbot, frame_0,
                                                                                    monkeypatch, caplog):
    class Broken(ThresholdFake):
        def preview(self, image, prompts):
            raise RuntimeError(f"No memory for {prompts[0].obj_id}.")

    monkeypatch.setattr("outline_tracker.gui.worker.TRACES_KEPT", 2)
    worker.start(lambda model, device: Broken(), "edgetam", "cpu")
    qtbot.waitUntil(lambda: worker.ready)
    assert worker.traces == [] and logged(caplog, LOG) == []  # a model that loads leaves none
    for frame, name in ((0, "A"), (2, "B"), (4, "C")):
        worker.request_preview(frame, frame_0, (0, 0), [helpers.click(name, 10.5, 10.5)])
        qtbot.waitUntil(lambda: not worker.busy)
    assert [reason for _, reason in worker.heard.failures] == [f"No memory for {name}." for name in "ABC"]
    assert [trace.splitlines()[-1] for trace in worker.traces] == [f"RuntimeError: No memory for {name}."
                                                                   for name in "BC"]
    first = worker.traces[-1].splitlines()[0]
    assert "object C" in first and "frame 4" in first and "in preview" in worker.traces[-1]
    assert logged(caplog, LOG)[1:] == worker.traces and len(logged(caplog, LOG)) == 3  # the log has all three


def test_a_failure_nobody_waits_for_any_more_is_not_reported_but_its_trace_is_kept(worker, qtbot, frame_0):
    with Gate() as gate:
        class Broken(ThresholdFake):
            def preview(self, image, prompts):
                gate.park()
                raise RuntimeError("Too late.")

        worker.start(lambda model, device: Broken(), "edgetam", "cpu")
        qtbot.waitUntil(lambda: worker.ready)
        worker.request_preview(0, frame_0, (0, 0), [helpers.click("A", 10.5, 10.5)])
        qtbot.waitUntil(gate.parked.is_set)
        worker.cancel_preview()
        gate.open()
        qtbot.waitUntil(lambda: len(worker.traces) == 1)
    assert worker.heard.failures == [] and worker.traces[0].splitlines()[-1] == "RuntimeError: Too late."


# ---------------------------------------------------------------------------------------------
# Stopping


def test_closing_the_window_stops_the_worker_thread_and_closes_the_segmenter(window, qtbot, disk_clip):
    segmenter = Watched()
    panel, _ = clicked_object(window, qtbot, disk_clip, segmenter=segmenter)
    worker = panel.worker
    assert worker.is_running() and segmenter.closed == 0
    window.close()
    assert not worker.is_running() and not worker.ready and worker.state == "stopped"
    assert segmenter.closed == 1
    worker.stop()  # again: nothing more happens
    assert segmenter.closed == 1 and not worker.is_running()


def test_a_window_that_was_never_shown_starts_no_thread(window):
    worker = worker_of(window)
    assert (worker.state, worker.is_running()) == ("idle", False)
    window.close()
    assert (worker.state, worker.is_running()) == ("stopped", False)
