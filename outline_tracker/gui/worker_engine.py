"""The engine of the worker thread (SPEC 5, 6.4, 10.2; decision X7): the one object that lives in
the thread, makes the segmenter there and calls it. Its handle for the GUI thread is the `Worker`
of gui/worker.py, whose module text says how the two work together; every name here is also
importable from there.

- `load` makes the segmenter with the factory it is given. A model that was loaded before is
  closed first, so that there is one copy of the weights at any time. The step before loading
  (`before_load`, for the real model `segmenter.hf.reserve_ui_thread`) is taken once, however many
  models are loaded after one another.
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
from PySide6.QtCore import QObject, Signal, Slot

from outline_tracker.segmenter.base import MaskResult, ObjectPrompt

REASON_LENGTH = 200  # a reason is one line, cut to this many characters
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


class _Engine(QObject):
    """The object in the worker thread. `load`, `work` and `run` are its slots and run there; `ask`
    may be called from any thread. `segmenter` is None until `load` has made it, and while another
    one is being made."""

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
        """Make the segmenter: `factory(model, device)`, after `before_load()` the first time one
        is given. The segmenter of a load before is closed first. `number` comes back with the
        answer, so that the `Worker` knows which load it is the answer of."""
        before, self.segmenter = self.segmenter, None
        try:
            if before is not None:
                before.close()
            if before_load is not None and not self._prepared:
                self._prepared = True
                before_load()
            self.segmenter = factory(model, device)
        except Exception as error:  # whatever a model's loading raises: said in one line, never raised here
            self.state.emit(number, "failed", plain(error),
                            traced(f"The model {model} could not be loaded (device {device})."))
            return
        self.state.emit(number, "ready", "", "")
        self.work()

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
