"""What the tests of panel 7 share (task C5): a named session with clicked objects in a window, a
stand-in segmenter that keeps how tracking used it, and what a job reported.

Imported by name from tests/gui/test_worker_jobs.py, test_tracking_panel.py and test_overlays.py.
Nothing here imports torch. A test that tracks works on a copy of its clip in a folder of its own
(`own_copy`, or the fixture `clip_in_odd_folder`), because the window writes the run folder next
to the video. A stand-in is parked with the `Gate` of tests/gui/prompt_helpers.py; no helper waits
with a delay.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5). Frames are video frame numbers; "tracked frame n" counts the frames a job gave the model,
from 1.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject

from outline_tracker import schema
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ThresholdFake
from prompt_helpers import Gate, gui_thread, panel_with, this_thread
from tracking_helpers import center

NAME = "Ada"  # the student of these tests: the run folder is <video stem>_outline_Ada


def own_copy(clip, folder: Path):
    """A copy of `clip` (a GroundTruth with its `path`) in `folder`, which is made: the GroundTruth
    with the copy's path."""
    folder.mkdir(parents=True, exist_ok=True)
    return replace(clip, path=Path(shutil.copyfile(clip.path, folder / clip.path.name)))


class Tracked:
    """A stand-in segmenter that keeps how previews and jobs used it.

    `calls`: for every image it was given, (what, thread, writeable, shape): what is "preview",
    "start" or "step", thread the thread of the call (`this_thread`), writeable the image's
    `flags.writeable`, shape its shape. `closed`: how often `close` was called. The tracked frames
    (the calls of `start` and `step`, counted from 1) whose numbers are in `park_at` wait at `gate`
    first; the tracked frame `fail_at` raises `error` instead of answering. Everything else is the
    inner stand-in's (a `ThresholdFake` unless another is given).
    """

    def __init__(self, inner=None, gate: Gate | None = None, park_at=(), fail_at: int | None = None,
                 error: Exception | None = None):
        self.inner = ThresholdFake() if inner is None else inner
        self.gate, self.park_at, self.fail_at, self.error = gate, set(park_at), fail_at, error
        self.calls: list[tuple] = []
        self.closed = 0

    @property
    def tracked(self) -> int:
        """How many frames jobs have given the model so far."""
        return sum(what != "preview" for what, *_ in self.calls)

    def _note(self, what: str, image) -> None:
        self.calls.append((what, this_thread(), bool(image.flags.writeable), image.shape))
        if what == "preview":
            return
        if self.tracked in self.park_at:
            self.gate.park()
        if self.tracked == self.fail_at:
            raise self.error

    def preview(self, image, prompts):
        self._note("preview", image)
        return self.inner.preview(image, prompts)

    def start(self, image, prompts):
        self._note("start", image)
        return self.inner.start(image, prompts)

    def step(self, image):
        self._note("step", image)
        return self.inner.step(image)

    def close(self):
        self.closed += 1
        self.inner.close()

    def __getattr__(self, name):  # set_view, where the inner stand-in has it
        return getattr(self.inner, name)


class Saying(Tracked):
    """A `Tracked` that has something to say during a job, as the real model has when it changes
    its device: before it answers for the tracked frame n (counted from 1), it writes `says[n]`
    to `log`. `run_job` points `log` at the job's log while it tracks; a text that begins with two
    blanks is a note under the run's line there."""

    def __init__(self, says: dict[int, str], **how):
        super().__init__(**how)
        self.says = dict(says)
        self.log = lambda text: None  # outside a job nobody listens

    def _note(self, what: str, image) -> None:
        if what != "preview" and self.tracked + 1 in self.says:
            self.log(self.says[self.tracked + 1])
        super()._note(what, image)


class Heard(QObject):
    """An object of the GUI thread that keeps what a `Jobs` reports: `progress` holds (done, total,
    s per frame, s left), `finished` holds (status, reason), `order` the names of the signals as
    they came (with "session saved" for the controller's `saved`), `threads` the thread each one
    arrived in."""

    def __init__(self, jobs, controller):
        super().__init__()
        self.progress, self.finished, self.order, self.threads = [], [], [], []
        jobs.started.connect(self._started)
        jobs.progress.connect(self._progress)
        jobs.saved.connect(self._saved)
        jobs.finished.connect(self._finished)
        controller.saved.connect(self._session_saved)

    def _came(self, name: str) -> None:
        self.order.append(name)
        self.threads.append(this_thread())

    def _started(self):
        self._came("started")

    def _progress(self, done, total, s_per_frame, eta_s):
        self.progress.append((done, total, s_per_frame, eta_s))
        self._came("progress")

    def _saved(self):
        self._came("saved")

    def _finished(self, status, reason):
        self.finished.append((status, reason))
        self._came("finished")

    def _session_saved(self):
        self._came("session saved")


def track_panel(window):
    """The controls of panel 7 (Track) in the window: its `TrackPanel`."""
    from outline_tracker.gui.panels.track_panel import TrackPanel

    return window.panels[6].findChild(TrackPanel)


def ready_to_track(window, qtbot, clip, segmenter=None, ids=("A",), end: int | None = None):
    """A window in which Track can be pressed: the student is `NAME`, `clip` is open and shown, the
    model (`segmenter`, a `ThresholdFake` if None) is ready, fps_true is the scene's, the clip ends
    on frame `end` (its last frame if None), and each object of `ids` has one positive click on
    the center of the scene's object of that name on frame 0, with its outline shown. The window
    names its objects A, B, C in the order they are added, so `ids` is "A", "AB" or "ABC". Returns
    (the `TrackPanel`, the `ObjectsPanel`)."""
    assert "ABC".startswith("".join(ids))
    window.controller.set_student(NAME)
    objects = panel_with(window, qtbot, clip, segmenter)
    session = window.controller.session
    session.time.fps_true = clip.scene.fps
    if end is not None:
        session.clip.end = end
    window.controller.touch()
    for track_id in ids:
        objects.add_object()
        assert objects.prompts.add_point(*center(clip, track_id, 0), 1)
    qtbot.waitUntil(lambda: set(objects.prompts.outlines) == set(ids))
    return track_panel(window), objects


def run_to_end(qtbot, panel, timeout: int = 30_000) -> None:
    """Press Track and wait until the job has ended and reported it."""
    panel.track()
    assert panel.jobs.running, panel.message.text() or window_hint(panel)
    qtbot.waitUntil(lambda: not panel.jobs.running, timeout=timeout)


def window_hint(panel) -> str:
    """The hint line of panel 7."""
    return panel.window().panels[6].hint.text()


def results_of(window) -> ResultsStore:
    """results.npz of the window's run folder, read now."""
    return ResultsStore.load(window.controller.run_folder / schema.RESULTS_NPZ)


def session_on_disk(window) -> dict:
    """session.json of the window's run folder, read now."""
    return json.loads((window.controller.run_folder / schema.SESSION_JSON).read_text(encoding="utf-8"))


def run_log(folder) -> str:
    """The text of run.log in the run folder `folder`; "" while there is none."""
    path = Path(folder) / schema.RUN_LOG
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def off_the_gui_thread(segmenter: Tracked) -> bool:
    """Whether every call of the stand-in ran in another thread than the GUI's."""
    return gui_thread() not in {thread for _, thread, *_ in segmenter.calls}
