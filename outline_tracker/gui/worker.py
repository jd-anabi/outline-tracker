"""The worker thread (SPEC 5, 6.4, 10.2; decision X7): the model is loaded in it, and it makes
the outlines that are shown after a click, so that no long operation runs in the GUI thread.

Two objects. The engine (`_Engine`, in gui/worker_engine.py) is the one object that lives in the
one worker thread: it makes the segmenter and calls it. The `Worker` is its handle and lives in the
GUI thread: a panel calls its methods, which return at once, and listens to its signals. The engine's own signals are
connected only to bound methods of the `Worker`, so everything the engine reports is taken over by
the GUI thread's event loop, and every signal of the `Worker` is emitted in the GUI thread: a slot
connected to them runs there, however it was connected.

- Loading: `start(factory, model, device)` starts the thread (64 MiB of stack: a default Qt
  thread has 0.5 MiB on macOS, where the model has never run) and makes the segmenter in it with
  `factory(model, device)`. `state` goes from "idle" to "loading" and then to "ready", or to
  "failed" with one plain line in `message`. `start_when_shown(window)` does this for a window
  once it is shown, and stops the thread when the window closes. `load(model, device)` loads
  another model for that window in place of the one that is loaded (panel 7's two boxes, an
  opened session); `wanted` is what was asked for last, `device` what the loaded model runs on.
- Outlines: `request_preview` hands the engine a frame and the clicks on it. The newest request
  wins: a request that waits is replaced by a newer one, and the result of an older one is
  dropped. `busy` is true from a request until its result or its failure has arrived.
- Jobs: `run(task)` has `task(segmenter)` called in the thread, after what the thread is doing; an
  outline asked for meanwhile waits for it. gui/worker_jobs.py tracks with this.
- `stop()` ends the thread and closes the segmenter. It waits for the call the engine is in; a
  task that runs asks `stopping` and ends early.
- A failure (loading, or a frame) is shown as one plain line. Its trace is not shown: it is written
  to this module's log when it happens (`traced`) and kept in `Worker.traces` for run.log (SPEC 10.2).
- What comes back is put into the coordinates of the full frame by `found_in`: each mask
  measured as tracking measures it. What a request holds is `click_rules.preview_input`.

This module does not import torch. Before the real model is made, the engine calls
`segmenter.hf.reserve_ui_thread()` (which leaves one processor thread to the window): that is the
`before_load` step, which `before_load_for` gives for the real model's factory only.

Coordinates: an image is an RGB uint8 array [row, column, 3]; prompt points and a result's offset
are px in the pixel frame of the image given (SPEC 3.1: u to the right, v downward, pixel centers
at +0.5); `offset` = (column, row) says where that image's top-left pixel is in the full frame.
Frames are video frame numbers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Signal, Slot

# The engine's names are passed on: what this module offered before the engine had its own file.
from outline_tracker.gui.worker_engine import (REASON_LENGTH, Preview, _Engine, _Request, log,  # noqa: F401
                                               plain, traced)
from outline_tracker.measure import MODES, measure_mask
from outline_tracker.segmenter.base import ObjectPrompt
from outline_tracker.session import Processing
from outline_tracker.tracking_fine import fine_window

STACK_BYTES = 64 * 1024 * 1024  # of the worker thread (X7)
NO_FACTORY = "This window was made without a model."
TRACES_KEPT = 20  # `Worker.traces` holds this many, the newest; the log has every one


@dataclass(frozen=True)
class Found:
    """What a preview gave for one object. outline: (N, 2) points (u, v) along the mask's outline,
    px of the full frame. center: the mask's centroid (u, v), px of the full frame. auto_window:
    the fine window the rule chooses from this mask (SPEC 6.3), px; None when the session's
    `fine_window_factor` cannot be used."""

    outline: np.ndarray
    center: tuple[float, float]
    auto_window: int | None


def found_in(preview: Preview, session) -> tuple[dict[str, Found], list[str]]:
    """What `preview` holds for the tracks of `session`: (found by track id, the ids of the
    objects the model did not find). Each mask is put back into the full frame and measured as
    tracking measures it (`measure.measure_mask`), so the outline is the one a run would store.
    A result for an object the session no longer has is left out. Raises ValueError for a result
    that cannot be measured."""
    (c0, r0), box = preview.offset, (*preview.offset, *preview.size)
    tracks = {} if session is None else {track.id: track for track in session.tracks}
    found, missing = {}, []
    for result in preview.results:
        track = tracks.get(result.obj_id)
        if track is None:
            continue
        in_frame = replace(result, offset=(result.offset[0] + c0, result.offset[1] + r0))
        record = measure_mask(in_frame, preview.frame, box, track.mode if track.mode in MODES else "coarse")
        if not record.visible:
            missing.append(track.id)
            continue
        try:
            auto_window = fine_window(result, session.processing.fine_window_factor)
        except ValueError:
            auto_window = None
        found[track.id] = Found(np.asarray(record.outline_px, float), (record.u, record.v), auto_window)
    return found, missing


def before_load_for(factory) -> Callable[[], object] | None:
    """The step to take in the worker thread just before `factory` makes its segmenter: for the
    real model's factory (`from_tracker.load_segmenter`), `segmenter.hf.reserve_ui_thread`; for
    any other factory (a test's stand-in), None. Nothing is loaded by asking."""
    from outline_tracker.from_tracker import load_segmenter

    if factory is not load_segmenter:
        return None
    from outline_tracker.segmenter.hf import reserve_ui_thread  # loads torch only when it is called

    return reserve_ui_thread


class Worker(QObject):
    """The handle of the worker thread, for the GUI thread (see the module's text).

    `state`: "idle" (not started), "loading", "ready", "failed" or "stopped"; `message`: the plain
    reason while it is "failed". `ready`: the model can be used. `wanted`: (model, device) as the
    last load was asked for them, None before the first; `device`: what the loaded model runs on.
    `busy`: an outline was asked for and has not arrived. Signals, all emitted in the GUI thread:
    `state_changed(state, message)`, `preview_done(preview)` with a `Preview` (only for the newest
    request), `preview_failed(serial, reason)` and `busy_changed(busy)`. `traces`: for each failure
    of the model (loading it, or a frame, reported or not), what was being done and the error's
    full trace as one text; newest last, at most `TRACES_KEPT`. They are for run.log, never for
    the window (SPEC 10.2).
    """

    state_changed = Signal(str, str)
    preview_done = Signal(object)
    preview_failed = Signal(int, str)
    busy_changed = Signal(bool)
    _load = Signal(object, str, str, object, int)
    _wake = Signal()
    _run = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.state, self.message = "idle", ""
        self.wanted: tuple[str, str] | None = None
        self.traces: list[str] = []
        self.stopping = threading.Event()  # set by `stop()`: a task that runs ends after its current frame
        self._serial = 0
        self._loads = 0                   # the number of the newest load: an older one's answer is not taken
        self._awaited: int | None = None  # the request whose result is waited for
        self._window = None
        self._engine = _Engine()
        self._thread = QThread(self)
        self._thread.setStackSize(STACK_BYTES)
        self._engine.moveToThread(self._thread)
        self._load.connect(self._engine.load)
        self._wake.connect(self._engine.work)
        self._run.connect(self._engine.run)
        self._engine.state.connect(self._engine_state)
        self._engine.previewed.connect(self._previewed)
        self._engine.failed.connect(self._failed)
        self._starter = QTimer(self)  # fires once the event loop runs after the window was shown
        self._starter.setSingleShot(True)
        self._starter.setInterval(0)
        self._starter.timeout.connect(self._start_for_window)

    # ------------------------------------------------------------------ state

    @property
    def ready(self) -> bool:
        """Whether the model is loaded and can be used: what a Track button waits for."""
        return self.state == "ready"

    @property
    def device(self) -> str | None:
        """The device the loaded model runs on ("cpu", "mps" or "cuda"), as its segmenter says it
        now (a model may leave the Apple GPU while it works); None while no model is loaded, and
        for a segmenter that names none (a stand-in)."""
        device = getattr(self._engine.segmenter, "device", None)
        return None if device is None else str(device)

    @property
    def busy(self) -> bool:
        """Whether an outline was asked for whose result or failure has not arrived yet."""
        return self._awaited is not None

    def is_running(self) -> bool:
        """Whether the worker thread runs."""
        return self._thread.isRunning()

    def _set_state(self, state: str, message: str) -> None:
        self.state, self.message = state, message
        self.state_changed.emit(state, message)

    def _await(self, serial: int | None) -> None:
        was = self.busy
        self._awaited = serial
        if self.busy != was:
            self.busy_changed.emit(self.busy)

    def _keep(self, trace: str) -> None:
        if trace:
            self.traces.append(trace)
            del self.traces[:-TRACES_KEPT]

    def keep_trace(self, trace: str) -> None:
        """Keep the trace of a failure in a task (`run`) with the others in `traces`; "" keeps none."""
        self._keep(trace)

    def run(self, task) -> bool:
        """Have `task(segmenter)` called once in the worker thread, after what the thread is doing
        now; outlines asked for meanwhile wait until it has returned. `segmenter` is the loaded
        model. The task must raise nothing and report through signals of an object of the GUI
        thread. Returns whether it was handed over: only while the model is ready."""
        if self.ready:
            self._run.emit(task)
        return self.ready

    # ------------------------------------------------------------------ loading

    def start(self, factory, model: str, device: str, before_load=None) -> None:
        """Start the thread and load the model in it: `factory(model, device)` makes the segmenter
        (model: "edgetam", "sam2", ...; device: "auto", "cpu", "mps" or "cuda"), after
        `before_load()` if one is given (it is called once, at the first load). Returns at once;
        `state_changed` says how it went. A model that is loaded is closed first, and of two loads
        asked for in a row only the second one's answer counts. Without a factory the state is
        "failed" and no thread starts. After `stop` nothing starts."""
        if self.state == "stopped":
            return
        if factory is None:
            self._set_state("failed", NO_FACTORY)
            return
        if not self._thread.isRunning():
            self._thread.start()
        self.wanted, self._loads = (model, device), self._loads + 1
        self._set_state("loading", "")
        self._load.emit(factory, model, device, before_load, self._loads)

    def load(self, model: str, device: str) -> None:
        """Load `model` on `device` (as for `start`) with the factory of the window given to
        `start_when_shown`, in place of the model that is loaded. Nothing happens when that is
        what was asked for last and it did not fail, before the window's own start (which takes
        the session's model and device), and after `stop`."""
        same = (model, device) == self.wanted and self.state != "failed"  # what failed may be tried again
        if self._window is None or self.state in ("idle", "stopped") or same:
            return
        factory = self._window.segmenter_factory
        self.start(factory, model, device, before_load_for(factory))

    def start_when_shown(self, window) -> None:
        """Load the model of `window` once the window is shown: with its `segmenter_factory`, and
        the model and device of the open session (a new session's when none is open). The window's
        `closing` signal stops the thread."""
        self._window = window
        window.installEventFilter(self)
        window.closing.connect(self.stop)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Show and watched is self._window and self.state == "idle":
            self._starter.start()  # not inside show(): a file named at start is open by then
        return False

    def _start_for_window(self) -> None:
        if self.state != "idle":
            return
        session = self._window.controller.session
        processing = Processing() if session is None else session.processing
        factory = self._window.segmenter_factory
        self.start(factory, processing.model, processing.device, before_load_for(factory))

    @Slot(int, str, str, str)
    def _engine_state(self, number: int, state: str, message: str, trace: str) -> None:
        self._keep(trace)
        if self.state == "stopped" or number != self._loads:  # the answer of a load that a newer one replaced
            return
        self._set_state(state, message)
        if state == "failed" and self._awaited is not None:  # no model: no outline will come
            serial = self._awaited
            self._engine.ask(None)
            self._await(None)
            self.preview_failed.emit(serial, message)

    # ------------------------------------------------------------------ outlines

    def request_preview(self, frame: int, image: np.ndarray, offset: tuple[int, int],
                        prompts: list[ObjectPrompt]) -> int:
        """Ask for the outlines of the objects of `prompts` on one frame, and return at once with
        the request's number (rising from 1).

        frame: the video frame number. image: what the model is shown, an RGB uint8 array
        [row, column, 3]: the frame or a part of it; it must not be changed (the model gets a copy
        of an array that cannot be written to).
        offset: (column, row) of the image's top-left pixel in the full frame, px. prompts: one per
        object, points in px of `image`. The result comes with `preview_done`, a failure with
        `preview_failed`; a request made before the model is ready waits for it. A newer request
        replaces this one. While the state is "failed" or "stopped" nothing is asked.
        """
        self._serial += 1
        if self.state in ("failed", "stopped"):
            return self._serial
        self._engine.ask(_Request(self._serial, int(frame), image, (int(offset[0]), int(offset[1])), list(prompts)))
        self._await(self._serial)
        self._wake.emit()
        return self._serial

    def cancel_preview(self) -> None:
        """Forget the request that waits or runs: its result will not be reported."""
        self._engine.ask(None)
        self._await(None)

    @Slot(object)
    def _previewed(self, preview: Preview) -> None:
        if preview.serial != self._awaited:  # out of date, cancelled, or after stop()
            return
        self.preview_done.emit(preview)
        if preview.serial == self._awaited:  # unless a slot asked again meanwhile
            self._await(None)

    @Slot(int, str, str)
    def _failed(self, serial: int, reason: str, trace: str) -> None:
        self._keep(trace)  # also of a request that nobody waits for any more
        if serial != self._awaited:
            return
        self._await(None)
        self.preview_failed.emit(serial, reason)

    # ------------------------------------------------------------------ stopping

    def stop(self) -> None:
        """End the worker thread and close the segmenter; the state is "stopped" from then on.
        It waits until the call the thread is in has returned (loading the model, or one frame).
        Calling it again does nothing."""
        if self.state == "stopped":
            return
        self.stopping.set()
        self._starter.stop()
        self._engine.ask(None)
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        segmenter, self._engine.segmenter = self._engine.segmenter, None
        if segmenter is not None:
            segmenter.close()
        self._await(None)
        self._set_state("stopped", "")


def worker_of(window) -> Worker:
    """The worker of `window` (a `MainWindow`): made at the first call and kept as `window.worker`,
    so that every panel works with the same one. It loads the model once the window is shown and
    stops when the window closes (`Worker.start_when_shown`)."""
    worker = getattr(window, "worker", None)
    if worker is None:
        worker = window.worker = Worker(window)
        worker.start_when_shown(window)
    return worker
