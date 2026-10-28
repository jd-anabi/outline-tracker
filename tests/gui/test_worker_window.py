"""The worker thread in the window (SPEC 5, 6.4, 10.2; decision X7; task C4): the model loads in the worker after
the window is shown, and it makes the outlines that are shown after a click.

What is checked, with a stand-in segmenter and no torch:
- the window's event loop still runs while the stand-in is parked inside `preview`, and of the
  requests made meanwhile only the newest one's outline is drawn, by a slot in the GUI thread;
- the model is made in the worker thread, after the window is shown, and until it reports ready
  nothing is `ready`; a model that cannot be loaded, or that fails on a frame, is reported in one
  plain line in the panel;
- the model is shown the part of the frame that tracking shows it;
- closing the window stops the thread, and a window that was never shown starts none.
The `Worker` on its own, with the traces of its failures, is in tests/gui/test_worker.py.
A gate is a `threading.Event` (tests/support/prompt_helpers.py): no test waits with a delay.

Expected positions come from the synthetic clips: the disks of `disk_scene` (centers within
0.25 px through `ThresholdFake`, the plan's measured condition), and for `ExactFake` the true mask
of the dish scene. The dish crop of that scene at 320 x 240 px is columns 49 to 271 and rows 8 to
231 (tests/support/tracking_helpers.py, `DISH_BOX`).
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt

import tracking_helpers
from gui_helpers import show
from outline_tracker.gui.view.prompts import Prompts
from outline_tracker.gui.worker import worker_of
from outline_tracker.measure.mask import mask_center
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import Gate, Watched, clicked_object, gui_thread, loop_runs, objects_panel, panel_with, this_thread
from tracking_helpers import DISH_BOX, center, dish_circle


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


# ---------------------------------------------------------------------------------------------
# A request returns at once, and the newest one wins


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
    # With a boundary and "Crop to boundary" on, tracking shows the model the boundary's square (SPEC 6.2):
    # the preview shows it the same part, and the outline comes back to the full frame's coordinates.
    segmenter = tracking_helpers.Watched(ExactFake(dish_clip))
    panel = panel_with(window, qtbot, dish_clip, segmenter)
    window.controller.session.boundary = dish_circle(dish_clip.scene)
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
