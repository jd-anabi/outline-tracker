"""The worker thread on its own (SPEC 5, 6.4, 10.2; decision X7; task C4): a `Worker` that loads the model and makes
the outlines, without a window.

What is checked, with a stand-in segmenter and no torch:
- nothing waits for the model: with the stand-in parked on a gate inside `preview`, the call that
  asked for the outline has returned, and of the requests made meanwhile only the newest one's
  outline is shown;
- what the worker reports arrives in the GUI thread;
- a worker without a factory has no model and starts no thread; the step before loading runs in the
  worker just before the factory;
- the trace of a failure is not shown, but it is written to the log when it happens and kept in
  `Worker.traces` (SPEC 10.2), also when nobody waits for the result any more.
The worker inside the window (the outlines after a click, the model loaded after the window is shown, closing the
window) is in tests/gui/test_worker_window.py. The `worker` fixture of this file overrides the one of
tests/gui/conftest.py.
A gate is a `threading.Event` (tests/support/prompt_helpers.py): no test waits with a delay.

Expected positions come from the synthetic clips: the disks of `disk_scene` (centers within
0.25 px through `ThresholdFake`, the plan's measured condition), and for `ExactFake` the true mask
of the dish scene. The dish crop of that scene at 320 x 240 px is columns 49 to 271 and rows 8 to
231 (tests/support/tracking_helpers.py, `DISH_BOX`).
"""

import numpy as np
import pytest
from PySide6.QtCore import QObject

import helpers
from outline_tracker.video.frame_source import FrameSource
from outline_tracker.gui.worker import Worker, before_load_for
from outline_tracker.measure.mask import mask_center
from outline_tracker.segmenter.fake import ThresholdFake
from prompt_helpers import Gate, Watched, gui_thread, logged, loop_runs, this_thread
from tracking_helpers import center

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


# ---------------------------------------------------------------------------------------------
# Loading the model


def test_what_the_worker_reports_about_the_model_arrives_in_the_gui_thread(worker, qtbot):
    worker.start(helpers.stand_in_segmenter, "edgetam", "cpu")
    qtbot.waitUntil(lambda: worker.ready)
    assert [state for state, _ in worker.heard.states] == ["loading", "ready"]
    assert set(worker.heard.threads) == {gui_thread()}
    assert worker.is_running()


def test_a_window_without_a_factory_has_no_model_and_starts_no_thread(worker, qtbot):
    worker.start(None, "edgetam", "cpu")
    assert (worker.state, worker.ready, worker.is_running()) == ("failed", False, False)
    assert worker.heard.states == [("failed", "This window was made without a model.")]


def test_the_real_models_factory_gets_the_thread_limit_and_a_stand_in_gets_none():
    from outline_tracker.segmenter.load import load_segmenter
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
