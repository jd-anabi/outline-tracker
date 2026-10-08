"""The rules of a click on the video (SPEC 3.5, 5): which click is a negative point, which click
is the second of a double click, which picture takes a click, what marks the clicks of a frame
are, what the model is given for them (`preview_input`), and the results on disk that the edit
functions read. gui/prompts.py works with these; `PointTool` is what it gives the video view.

Negative points. A positive point says "this is the object", a negative one "this is not". A click
is negative when it is made with the right button, or with the left button while Alt (Option on a
Mac) or the physical Control key is held. On a Mac a two-finger click on the trackpad and a
Control-click both arrive as a right click already. Qt names the modifier keys by their role, not
by the key: on macOS the Command key is `ControlModifier` and the Control key is `MetaModifier`;
on Windows and Linux the Control key is `ControlModifier` and the Windows key is `MetaModifier`.
So the platform decides which of the two is the physical Control key, and a Command-click on a Mac
stays a positive point.

The frame of a click. The hash stored with a click is `video.frame_hash` of the frame as
`FrameSource.get` returns it: the frame of the sequential decode, bit for bit, which tracking
reads and the guard at a run's start compares. A picture in the view that is not that frame
(another source, no frame, a frame that cannot be read) takes no click (`frame_shown`).

Coordinates: (u, v) in px of the full video frame, Tracker's convention (u to the right, v
downward, pixel centers at +0.5). Frames are video frame numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt

from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_NPZ
from outline_tracker.segmenter.base import ObjectPrompt, PromptError, points_in_image
from outline_tracker.tracking_plan import view_box

NOT_THE_VIDEO = ("The picture shown is not a frame of the video as tracking reads it, so the click was not stored. "
                 "Go to another frame and back, then click again.")


def control_key(platform: str) -> Qt.KeyboardModifier:
    """The modifier Qt reports for the physical Control key on `platform` (a value of
    `sys.platform`): `MetaModifier` on macOS ("darwin"), `ControlModifier` everywhere else."""
    return Qt.KeyboardModifier.MetaModifier if platform == "darwin" else Qt.KeyboardModifier.ControlModifier


def is_negative_click(button, modifiers, platform: str) -> bool:
    """Whether a click with mouse button `button` and the held keys `modifiers` (Qt's values) is a
    negative point on `platform` (a value of `sys.platform`, such as "darwin" or "win32").

    True for the right button, whatever keys are held, and for the left button with Alt or with
    the physical Control key (`control_key`). False for every other click: a plain left click, a
    left click with Shift, with Command on a Mac or with the Windows key, and any other button.
    """
    if button == Qt.MouseButton.RightButton:
        return True
    if button != Qt.MouseButton.LeftButton:
        return False
    negative_keys = Qt.KeyboardModifier.AltModifier | control_key(platform)
    return bool(modifiers & negative_keys)


def point_kind(tool: str, button, modifiers, platform: str) -> str | None:
    """What a click places in the point tool `tool` ("positive", "negative" or "head"): "negative"
    for a negative click (`is_negative_click`) in every tool and for every click of the Negative
    tool, else "head" in the Head tool and "positive" in the Positive tool. None for a click with
    another button than the left or the right one: it places nothing."""
    if button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
        return None
    return "negative" if tool == "negative" or is_negative_click(button, modifiers, platform) else tool


class DoubleClickWatch(QObject):
    """Tells the second click of a double click. Qt delivers the second press of a double click as
    a double-click event, and the video view hands its release to the tool as one more click:
    installed as an event filter on the view's viewport, this object sees that event first.
    `take_second()` is true once for such a click. No quantities, so no units."""

    def __init__(self, viewport):
        super().__init__(viewport)
        self._second = False
        viewport.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self._second = True
        elif event.type() == QEvent.Type.MouseButtonPress:
            self._second = False
        return False

    def take_second(self) -> bool:
        """Whether the click that arrives now is the second of a double click; asked once per click."""
        second, self._second = self._second, False
        return second


def frame_shown(controller, view) -> tuple[int, np.ndarray]:
    """(frame number, frame) of the picture in `view` (the `VideoView`), as the video of
    `controller` (the `SessionController`) has it: an RGB uint8 array [row, column, 3], the frame
    tracking reads under that number. Raises ValueError, with a message for the user, when the
    picture is not such a frame: the view shows another source or no frame, or the frame cannot
    be read."""
    source = controller.source
    if source is None or view.source is not source or view.frame is None:
        raise ValueError(NOT_THE_VIDEO)
    try:
        return view.frame, source.get(view.frame)
    except (IndexError, ValueError) as error:
        raise ValueError(NOT_THE_VIDEO) from error


class ResultsOnDisk:
    """The results of the run folder as results.npz holds them now, for the functions that only
    read them. The file is read again whenever it was replaced or changed (its size, time and file
    number are compared); without a run folder or a file there are no records.

    A results file that cannot be opened at this moment leaves what was read before in place, and
    is read at the next call. During a run the worker thread replaces results.npz at every
    autosave, and the window reads it again after each one: on Windows the file can be refused to
    the reader (PermissionError) while the worker is replacing it once more. These functions are
    called in the GUI thread, so nothing waits and nothing is tried again here: the next autosave,
    or the end of the run, asks again."""

    def __init__(self, controller):
        self._controller = controller
        self._key, self._store = None, ResultsStore()

    def now(self) -> ResultsStore:
        """The store to read from; never change it. Raises ValueError for a file that is no
        results file of this version (`ResultsStore.load`). A file that is refused at this moment
        (PermissionError when it is looked at or opened) raises nothing: the store that was read
        from it before is returned, one without records if nothing was read from it yet (what was
        read in another run folder does not count), and the next call reads the file. Values in
        it are px and frame numbers."""
        folder = self._controller.run_folder
        path = None if folder is None else Path(folder) / RESULTS_NPZ
        try:
            if path is None or not path.is_file():
                self._key, self._store = None, ResultsStore()
                return self._store
            found = path.stat()
            key = (str(path), found.st_size, found.st_mtime_ns, found.st_ino)
            if key != self._key:
                self._store, self._key = ResultsStore.load(path), key
        except PermissionError:  # refused at this moment: the key stays, so the next call reads the file
            if self._key is not None and self._key[0] != str(path):  # what is held is another run folder's
                self._key, self._store = None, ResultsStore()
        return self._store


@dataclass(frozen=True)
class Mark:
    """One mark drawn on the frame shown: `kind` is "positive", "negative" or "head"; (u, v) in px
    of the full frame."""

    track_id: str
    kind: str
    u: float
    v: float


def marks_on(tracks, frame: int | None) -> list[Mark]:
    """The marks of video frame `frame`: for each track of `tracks` (the session's, in order) its
    points on that frame in the order they were clicked, then its head on the track's start frame."""
    marks = []
    for track in tracks:
        for prompt in track.prompts:
            if prompt.frame == frame:
                marks += [Mark(track.id, "positive" if label == 1 else "negative", float(u), float(v))
                          for (u, v), label in zip(prompt.points_px, prompt.labels)]
        if track.head_px is not None and track.start_frame == frame:
            marks.append(Mark(track.id, "head", float(track.head_px[0]), float(track.head_px[1])))
    return marks


def preview_input(session, frame: int, rgb: np.ndarray):
    """What the model is to outline on video frame `frame`, whose picture is `rgb` (the full frame,
    an RGB array [row, column, 3]): (signature, image, box, prompts), or None when no object has a
    positive point there that the model would be shown.

    box = (c0, r0, width, height), whole px of the full frame: the part tracking shows the model
    (`tracking_plan.view_box`: the dish square, or the whole frame). image: that part of `rgb`.
    prompts: one per object with clicks on the frame, in the session's order, points in px of
    `image`; points outside it are left out, as tracking leaves them out. signature: equal for
    equal clicks on the same frame and part, for telling whether an outline is still up to date.
    """
    if session is None:
        return None
    try:
        c0, r0, width, height = box = view_box(session)
    except ValueError:  # no frame size, or a dish circle that gives no crop: tracking would refuse too
        return None
    prompts = []
    for track in session.tracks:
        on_frame = [prompt for prompt in track.prompts if prompt.frame == frame]
        points = [(u - c0, v - r0) for prompt in on_frame for u, v in prompt.points_px]
        labels = [label for prompt in on_frame for label in prompt.labels]
        try:
            prompts.append(points_in_image(ObjectPrompt(track.id, points, labels), height, width))
        except PromptError:  # no positive point in what the model is shown: nothing to outline yet
            continue
    if not prompts:
        return None
    clicks = tuple((prompt.obj_id, tuple(prompt.points_px), tuple(prompt.labels)) for prompt in prompts)
    signature = (frame, box, clicks)
    return signature, np.ascontiguousarray(rgb[r0:r0 + height, c0:c0 + width]), box, prompts


class PointTool:
    """A tool of the video view (`VideoView.set_tool`): `kind` is "positive", "negative" or "head".
    `text` is the line shown above the picture, `cursor` the cursor over it; `prompts` (the
    `prompts.Prompts` it hands its clicks to) sets both."""

    def __init__(self, prompts, kind: str):
        self.kind, self._prompts = kind, prompts
        self.cursor, self.text = Qt.CursorShape.CrossCursor, ""

    def click(self, u: float, v: float, button, modifiers) -> None:
        """A click at (u, v), px of the video frame, with Qt's mouse button and held keys."""
        self._prompts.clicked(self.kind, u, v, button, modifiers)
