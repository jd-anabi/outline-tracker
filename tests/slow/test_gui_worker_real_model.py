"""The real EdgeTAM through the window's worker (SPEC 6.4, 10.2, 13.4; task C5), on `cpu` and on the
Apple GPU. Every test here is slow: it loads torch and the model.

The window is given the real model's factory (`from_tracker.load_segmenter`), so the worker takes
one processor thread for the window before it loads the model, as in the application. The clip is
last week's selftest clip (`synthetic.selftest_clip`): one dark ellipse on a 1080p frame whose
center is at (700.5 + 0.6 f, 500.5 + 0.2 f) px in frame f; a new session tracks every 2nd of its
40 frames, the 20 frames 0, 2, ..., 38. The object gets one positive click on its center on frame
0, in the window, as a student places it; the outline of that click comes from the model first.

The criterion is the selftest's: the job completes, every frame has a position, none 3 px or more
from the true center, and the worker recorded no error. Lines printed with the prefix `VALIDATION`
are the numbers for the report; show them with
`uv run pytest -m slow tests/slow/test_gui_worker_real_model.py -q -rP`. The s per frame are an
upper bound when anything else runs on the computer.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1); frames are video frame
numbers. torch is imported inside the test that asks for the Apple GPU, never at the top.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))  # the helpers of tests/gui, by name

from gui_helpers import show  # noqa: E402
from pipeline_helpers import expect_minutes  # noqa: E402
from prompt_helpers import gui_thread, objects_panel  # noqa: E402
from track_helpers import NAME, Heard, results_of, track_panel  # noqa: E402

from outline_tracker.from_tracker import load_segmenter  # noqa: E402
from outline_tracker.synthetic import selftest_clip  # noqa: E402

pytestmark = pytest.mark.slow

FRAMES = list(range(0, 40, 2))
LOADED_WITHIN_MS = 5 * 60 * 1000   # the first time the model is downloaded
TRACKED_WITHIN_MS = 8 * 60 * 1000  # 20 frames; a few s per frame on a slow processor


def _track_in_the_window(window, qtbot, folder, device: str) -> None:
    """Track the selftest clip in `window` with the real EdgeTAM on `device`, through the window's
    worker; assert the criterion and print the numbers. Errors are px, times s per tracked frame."""
    expect_minutes()
    clip = selftest_clip(folder)
    window.segmenter_factory = load_segmenter
    window.controller.set_student(NAME)
    window.open_path(clip["video"])
    session = window.controller.session
    session.processing.device = device  # before the window is shown: the worker loads for this device
    session.time.fps_true = 240.0
    window.controller.touch()
    show(window, qtbot)
    panel, objects = track_panel(window), objects_panel(window)
    worker = panel.worker
    qtbot.waitUntil(lambda: worker.state in ("ready", "failed"), timeout=LOADED_WITHIN_MS)
    assert worker.state == "ready", worker.message

    objects.add_object()
    assert objects.prompts.add_point(700.5, 500.5, 1)
    qtbot.waitUntil(lambda: not objects.prompts.busy, timeout=TRACKED_WITHIN_MS)
    assert set(objects.prompts.outlines) == {"A"}, objects.prompts.message
    assert panel.track_button.isEnabled(), window.panels[6].hint.text()
    estimate = window.panels[6].hint.text()

    heard = Heard(panel.jobs, window.controller)
    panel.track_button.click()
    assert panel.jobs.running
    qtbot.waitUntil(lambda: not panel.jobs.running, timeout=TRACKED_WITHIN_MS)

    arrays = results_of(window).arrays("A")
    error = np.hypot(arrays.u - (700.5 + 0.6 * arrays.frames), arrays.v - (500.5 + 0.2 * arrays.frames))
    (run,) = window.controller.session.runs
    print(f"VALIDATION gui worker, {device}: status {panel.jobs.status}; max error {np.nanmax(error):.3f} px, mean "
          f"{np.nanmean(error):.3f} px on {len(arrays)} frames; {run.seconds_per_frame:.2f} s per frame; device at "
          f"the end: {run.device}; before the run the panel said: {estimate}")

    assert (panel.jobs.status, panel.jobs.reason) == ("complete", "")
    assert worker.traces == []  # no error was recorded: not while loading, not for the outline, not in the job
    assert arrays.frames.tolist() == FRAMES and arrays.visible.all()
    assert np.isfinite(error).all() and error.max() < 3.0
    assert run.device == device and run.frames_done == 20  # no fall back to the processor
    assert window.controller.session.complete is True
    assert heard.progress[-1][:2] == (20, 20) and set(heard.threads) == {gui_thread()}
    assert panel.message.kind == "success"
    # the outline of frame 20 is on the picture, around the true center of that frame
    window.show_frame(20)
    center = panel.overlays.shown["A"].center
    assert np.hypot(center[0] - 712.5, center[1] - 504.5) < 3.0


def test_the_real_model_tracks_through_the_windows_worker_on_cpu(window, qtbot, tmp_path):
    _track_in_the_window(window, qtbot, tmp_path, "cpu")


def test_the_real_model_tracks_through_the_windows_worker_on_mps(window, qtbot, tmp_path):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    _track_in_the_window(window, qtbot, tmp_path, "mps")
