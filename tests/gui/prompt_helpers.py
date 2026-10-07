"""What the tests of panel 6 share (task C4): objects, clicks, the preview and the worker thread.

Imported by name from the test files beside it. Nothing here imports torch.

- `Gate` parks a call of a stand-in on a `threading.Event`: a test waits for `gate.parked`
  (`qtbot.waitUntil`), does what it has to do while the worker thread stands still, and opens the
  gate. No test waits with a delay. The gate opens by itself when its `with` block ends, so a test
  that fails inside it leaves no thread parked.
- `Watched` is a stand-in segmenter (a `ThresholdFake` unless another is given) that writes down
  every preview asked of it and in which thread, and parks the chosen calls on a gate.
- `panel_with` opens a clip in the window, shows the window and waits until the model is ready.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5); positions in a widget are Qt's device-independent px.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui_helpers import picture
from outline_tracker.segmenter.fake import ThresholdFake

LEFT, RIGHT, MIDDLE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
NO_KEY, ALT, SHIFT = (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.AltModifier,
                      Qt.KeyboardModifier.ShiftModifier)
QT_CONTROL, QT_META = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier
DISK_RADIUS = 12.0  # of the three disks of `synthetic.disk_scene`, px
SAFETY_S = 30       # a gate that nobody opens gives up after this long, so that no run hangs


class Gate:
    """Where a stand-in's call waits until the test lets it go on."""

    def __init__(self):
        self.parked = threading.Event()  # set when a call has arrived and waits
        self._open = threading.Event()

    def park(self) -> None:
        """Called by the stand-in, in the worker thread: wait here until the gate is open."""
        self.parked.set()
        if not self._open.wait(SAFETY_S):
            raise TimeoutError("The gate was never opened.")

    def open(self) -> None:
        self._open.set()

    def __enter__(self) -> Gate:
        return self

    def __exit__(self, *exc) -> None:
        self.open()


class Watched:
    """A stand-in segmenter that keeps what was asked of it.

    `previews`: for every call of `preview`, the prompts it was given as (obj_id, points, labels),
    points in px of the image given. `shapes`: the shape of the image of each call. `threads`: the
    thread each call ran in (`this_thread`). `closed`: how often `close` was called. The calls
    whose numbers (counted from 0) are in `park_at` wait at `gate` first. Everything else is the
    inner stand-in's.
    """

    def __init__(self, inner=None, gate: Gate | None = None, park_at=()):
        self.inner = ThresholdFake() if inner is None else inner
        self.gate, self.park_at = gate, set(park_at)
        self.previews: list[list[tuple]] = []
        self.shapes: list[tuple] = []
        self.threads: list[int] = []
        self.closed = 0

    def preview(self, image, prompts):
        call = len(self.previews)
        self.previews.append([(prompt.obj_id, list(prompt.points_px), list(prompt.labels)) for prompt in prompts])
        self.shapes.append(image.shape)
        self.threads.append(this_thread())
        if call in self.park_at:
            self.gate.park()
        return self.inner.preview(image, prompts)

    def close(self):
        self.closed += 1
        self.inner.close()

    def __getattr__(self, name):  # start, step, and set_view where the inner stand-in has it
        return getattr(self.inner, name)


def this_thread() -> int:
    """The thread this is called in, as Python numbers it (a Qt thread has such a number too)."""
    return threading.get_ident()


def gui_thread() -> int:
    """The thread the application and every widget live in: the one pytest runs the tests in."""
    return threading.main_thread().ident


def logged(caplog, name: str) -> list[str]:
    """What the logger `name` wrote during the test (pytest's `caplog`), in any thread, as a program
    that set up no logging gets it on stderr: the records from WARNING up, each as its message
    with, under it, the trace of the error it was logged with."""
    import logging

    return [logging.Formatter().format(record) for record in caplog.records
            if record.name == name and record.levelno >= logging.WARNING]


def objects_panel(window):
    """The controls of panel 6 (Objects) in the window: its `ObjectsPanel`."""
    from outline_tracker.gui.panels.objects_panel import ObjectsPanel

    return window.panels[5].findChild(ObjectsPanel)


def panel_with(window, qtbot, clip, segmenter=None):
    """Open `clip` in the window, show the window, wait until the model is ready, and return the
    `ObjectsPanel`. `segmenter`, if given, is what the window's factory makes in place of the
    plain `ThresholdFake`."""
    if segmenter is not None:
        window.segmenter_factory = lambda model, device: segmenter
    picture(window, qtbot, clip)
    panel = objects_panel(window)
    qtbot.waitUntil(lambda: panel.worker.ready)
    return panel


def clicked_object(window, qtbot, clip, track_id="A", frame=0, segmenter=None):
    """`panel_with`, then Add and one positive click on the center of the scene's object
    `track_id` in the frame shown (frame `frame`). Returns (panel, (u, v) of the click)."""
    from tracking_helpers import center

    panel = panel_with(window, qtbot, clip, segmenter)
    panel.add_object()
    u, v = center(clip, track_id, frame)
    assert panel.prompts.add_point(u, v, 1)
    return panel, (u, v)


def at(view, u: float, v: float) -> QPoint:
    """The point of the view's viewport that shows the video point (u, v), in whole px."""
    return view.mapFromScene(view.image_item.mapToScene(QPointF(u, v)))


def fit_of(view, size):
    """Where a frame of `size` = (width, height) px lies in the view when all of it shows, by
    geometry: (screen px per video px, left, top), the corner in the viewport's px."""
    (width, height), room = size, view.viewport().size()
    scale = min(room.width() / width, room.height() / height)
    return scale, (room.width() - width * scale) / 2, (room.height() - height * scale) / 2


def double_click(view, point: QPoint, button=LEFT) -> None:
    """A double click as Qt delivers it to a widget: press, release, a second press that arrives
    as a double-click event, and its release, all at `point` of the view's viewport."""
    viewport = view.viewport()
    QTest.mousePress(viewport, button, NO_KEY, point)
    QTest.mouseRelease(viewport, button, NO_KEY, point)
    QTest.mouseDClick(viewport, button, NO_KEY, point)
    QTest.mouseRelease(viewport, button, NO_KEY, point)
    QApplication.processEvents()


def tracked_folder(clip, folder, track_ids, step: int = 2):
    """A run folder `folder` with a results.npz in which the scene's disks `track_ids` are tracked on
    every `step`-th frame of the whole clip, as a job would leave it: each frame is decoded the way
    tracking decodes, a `ThresholdFake` finds each disk from a click on its true center, and the
    mask is measured on the whole frame. Returns `folder`."""
    from contextlib import closing

    from helpers import click
    from outline_tracker.measure import measure_mask
    from outline_tracker.results import ResultsStore
    from outline_tracker.schema import RESULTS_NPZ
    from outline_tracker.video import iter_rgb_frames
    from tracking_helpers import center

    width, height = clip.scene.size
    store, finder = ResultsStore(), ThresholdFake()
    with closing(iter_rgb_frames(clip.path, range(0, clip.scene.n_frames, step))) as frames:
        for frame, rgb in frames:
            prompts = [click(track_id, *center(clip, track_id, frame)) for track_id in track_ids]
            for result in finder.preview(rgb, prompts):
                store.put(result.obj_id, measure_mask(result, frame, (0, 0, width, height), "coarse"))
    folder.mkdir(parents=True, exist_ok=True)
    store.save(folder / RESULTS_NPZ)
    return folder
