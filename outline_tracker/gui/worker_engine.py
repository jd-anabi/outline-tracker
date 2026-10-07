"""The engine of the worker thread (SPEC 5, 6.4, 10.2; decision X7): the one object that lives in
the thread, has the segmenter made, and is the only one that calls it. Its handle for the GUI
thread is the `Worker` of gui/worker.py, whose module text says how the two work together; every
name here is also importable from there.

- `load` has the segmenter made with the factory it is given, in a thread that does nothing else
  (`_Making`, with the worker thread's stack) and ends when the factory has returned. The worker
  thread is free meanwhile for what needs no model (Export all, the flags table); it takes the
  segmenter over when it is made, and nothing else ever calls it. A model that was loaded before
  is closed first, and a second load waits until what the first one made is closed, so that there
  is one copy of the weights at any time. The step before loading (`before_load`, for the real
  model `segmenter.hf.reserve_ui_thread`) is taken once, here in the worker thread, which is the
  one that will use the model, however many models are loaded after one another.
- `work` makes the outlines of the request that waits; `run` calls one task with the segmenter.
- A failure is said in one plain line (`plain`); its trace goes to the log when it happens
  (`traced`) and is handed on for run.log, never shown (SPEC 10.2).

This module does not import torch. Coordinates: an image is an RGB uint8 array [row, column, 3];
prompt points and a result's offset are px in the pixel frame of the image given (SPEC 3.1: u to
the right, v downward, pixel centers at +0.5); `offset` = (column, row) says where that image's
top-left pixel is in the full frame. Frames are video frame numbers.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from outline_tracker.segmenter.base import MaskResult, ObjectPrompt

REASON_LENGTH = 200  # a reason is one line, cut to this many characters
# Of the worker thread and of a thread that makes a model (X7): a default Qt thread has 0.5 MiB on macOS.
STACK_BYTES = 64 * 1024 * 1024
# The worker's log, under the name it has always had: written to stderr, unless the program gives it another place.
log = logging.getLogger("outline_tracker.gui.worker")


@dataclass(frozen=True)
class Preview:
    """What the model outlined on one frame. serial: the number `request_preview` returned.
    frame: the video frame number. offset: (column, row) of the top-left pixel of the image the
    model was shown, in px of the full frame; size: that image's (width, height) in px. results:
    one `MaskResult` per prompt, in the order of the prompts, with offsets in px of that image."""

    serial: int
    frame: int
    offset: tuple[int, int]
    size: tuple[int, int]
    results: list[MaskResult]


@dataclass(frozen=True)
class _Request:
    serial: int
    frame: int
    image: np.ndarray
    offset: tuple[int, int]
    prompts: list[ObjectPrompt]


def plain(error: BaseException) -> str:
    """An error as one line for the user: the first line of its text, without a trace, cut to
    `REASON_LENGTH` characters; the error's kind when it has no text."""
    lines = str(error).strip().splitlines()
    return lines[0].strip()[:REASON_LENGTH] if lines else type(error).__name__


def traced(what: str) -> str:
    """For the handler of an error, in any thread: write `what` (one line that says what was being
    done) and the error's trace to the log now, and return the same text, for `Worker.traces`."""
    log.exception(what)
    return f"{what}\n{traceback.format_exc().rstrip()}"


class _Making(QThread):
    """A thread that makes one segmenter and ends: `factory(model, device)` is all it calls. What
    the factory returned is `made`; if it raised, `reason` is the plain line and `trace` the text
    for run.log (`traced`). The engine reads them when the thread has ended."""

    def __init__(self, factory, model: str, device: str, number: int):
        super().__init__()
        self.setStackSize(STACK_BYTES)
        self.job, self.number = (factory, model, device), number
        self.made, self.reason, self.trace = None, "", ""

    def run(self) -> None:
        factory, model, device = self.job
        try:
            self.made = factory(model, device)
        except Exception as error:  # whatever a model's loading raises: said in one line, never raised here
            self.reason = plain(error)
            self.trace = traced(f"The model {model} could not be loaded (device {device}).")


class _Engine(QObject):
    """The object in the worker thread. `load`, `work` and `run` are its slots and run there; `ask`
    may be called from any thread. `segmenter` is None until the one `load` asked for is made and
    taken over here, and while another one is being made."""

    # the number of the load; "ready" or "failed"; the plain reason of a failure; its trace (`traced`)
    state = Signal(int, str, str, str)
    previewed = Signal(object)     # a Preview
    failed = Signal(int, str, str)  # the serial of a request; the plain reason; the trace (`traced`)

    def __init__(self):
        super().__init__()
        self.segmenter = None
        self._lock = threading.Lock()
        self._waiting: _Request | None = None
        self._prepared = False  # `before_load` was called: it is called once in a thread's life
        self._making: _Making | None = None  # the thread that makes a segmenter now
        self._next: tuple | None = None      # the load asked for meanwhile: it begins when that thread has ended

    def ask(self, request: _Request | None) -> None:
        """Make `request` the one that waits, in place of any other; None: nothing waits."""
        with self._lock:
            self._waiting = request

    def _take(self) -> _Request | None:
        with self._lock:
            request, self._waiting = self._waiting, None
        return request

    @Slot(object, str, str, object, int)
    def load(self, factory, model: str, device: str, before_load, number: int) -> None:
        """Have the segmenter made: `factory(model, device)` in a thread of its own (`_Making`),
        after `before_load()` here the first time one is given. This returns at once, so the worker
        thread goes on with its tasks. The segmenter of a load before is closed first; while one
        is still being made, this load begins when that one has ended and is closed. `number`
        comes back with the answer, so that the `Worker` knows which load it is the answer of."""
        before, self.segmenter = self.segmenter, None
        try:
            if before is not None:
                before.close()
            if before_load is not None and not self._prepared:
                self._prepared = True
                before_load()
        except Exception as error:  # said in one line, never raised here
            self.state.emit(number, "failed", plain(error),
                            traced(f"The model {model} could not be loaded (device {device})."))
            return
        if self._making is not None:
            self._next = (factory, model, device, number)  # in place of a load that waited: the newest counts
        else:
            self._make(factory, model, device, number)

    def _make(self, factory, model: str, device: str, number: int) -> None:
        self._making = _Making(factory, model, device, number)
        self._making.finished.connect(self._made)  # emitted in that thread, so taken over by this one's loop
        self._making.start()

    @Slot()
    def _made(self) -> None:
        """The thread that made a segmenter has ended: take the segmenter over, here in the worker
        thread, and say how the load went. If another load was asked for meanwhile, what was made
        is closed and that load begins."""
        making, self._making = self._making, None
        if making is None:
            return
        making.wait()
        waiting, self._next = self._next, None
        if making.trace:
            self.state.emit(making.number, "failed", making.reason, making.trace)
        elif waiting is None:
            self.segmenter = making.made
            self.state.emit(making.number, "ready", "", "")
            self.work()
        elif making.made is not None:  # nobody will use it: one copy of the weights
            try:
                making.made.close()
            except Exception as error:  # kept for run.log; the load that waits begins all the same
                self.state.emit(making.number, "failed", plain(error),
                                traced(f"The model {making.job[1]} could not be closed."))
        if waiting is not None:
            self._make(*waiting)

    def end_making(self) -> str:
        """For `Worker.stop`, in the GUI thread, once the worker thread has ended: wait for the
        segmenter that is still being made, and close it. No load begins after this. Returns the
        trace of that load if it failed (`traced`), else ""."""
        making, self._making, self._next = self._making, None, None
        if making is None:
            return ""
        making.wait()
        if making.made is not None:
            making.made.close()
        return making.trace

    @Slot()
    def work(self) -> None:
        """Make the outlines of the request that waits, and of those that arrive meanwhile. Without
        a segmenter the request goes on waiting (`load` calls this when the model is there)."""
        while self.segmenter is not None:
            request = self._take()
            if request is None:
                return
            height, width = request.image.shape[:2]
            # a frame of the view's cache is read-only, and the model may write to what it is given
            image = request.image if request.image.flags.writeable else request.image.copy()
            try:
                set_view = getattr(self.segmenter, "set_view", None)  # ExactFake is told what each image shows
                if set_view is not None:
                    set_view(request.frame, request.offset, (width, height))
                results = self.segmenter.preview(image, request.prompts)
            except Exception as error:  # whatever a model raises on a frame
                ids = [prompt.obj_id for prompt in request.prompts]
                what = (f"The outline of {'object' if len(ids) == 1 else 'objects'} {', '.join(ids)} on frame "
                        f"{request.frame} could not be made ({width} x {height} px shown, from {request.offset}).")
                self.failed.emit(request.serial, plain(error), traced(what))
                continue
            with self._lock:
                newer = self._waiting is not None
            if not newer:  # else this result is out of date already
                self.previewed.emit(Preview(request.serial, request.frame, request.offset, (width, height), results))

    @Slot(object)
    def run(self, task) -> None:
        """Call `task(segmenter)` here, in the worker thread: one job, between two outlines. The
        task reports by itself and raises nothing; `segmenter` is None when no model was loaded."""
        task(self.segmenter)
