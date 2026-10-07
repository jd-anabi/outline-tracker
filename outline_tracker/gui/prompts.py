"""Clicks on objects, and what is drawn of them on the video (SPEC 3.1, 3.5, 5, 10.1, 10.2).

`Prompts` is the part of panel 6 that works on the video view: the three point tools (Positive,
Negative, Head), undo, the marks of the clicks on the frame shown, and the outline the model
gives for them (the preview), with the window of a fine object. The table and the buttons are in
gui/panels/objects_panel.py; the items that are drawn are made in gui/prompt_drawing.py.

- Every edit goes through `outline_tracker.tracking` (`add_prompt`, `undo_prompt`, `set_head`):
  the frame grid, the frame hash and the rule against a gap inside a track are decided there. A
  refusal is shown in that function's words (`message`) and changes nothing. After an edit the
  controller's `touch()` tells the rest of the window.
- The hash stored with a click is `video.frame_hash` of the frame tracking reads under that
  number; a picture in the view that is not that frame takes no click (gui/click_rules.py, which
  also says which click is negative and which is the second of a double click).
- After every change of the clicks on the frame shown, the worker is asked for the outlines of all
  objects clicked on that frame, on the part of the frame tracking would show the model
  (`tracking_plan.view_box`: the dish square, or the whole frame). The newest request wins.
- An outline on the picture is of the model that is chosen now. When another model begins to load
  (panel 7, an opened session), the outlines of the model before are taken away and the frame
  shown is asked of the new one; the worker keeps that request until the model is ready.
- While a tracking job or an export runs (`worker_jobs.Jobs.writing`) no point is placed, moved or
  taken back: that task works on the points as they were when it started. The tool's line says to wait.

Coordinates: (u, v) in px of the full video frame, Tracker's convention (u to the right, v
downward, the pixel in column c and row r with its center at (c + 0.5, r + 0.5)): clicks arrive
so from the view, and every item drawn is in these coordinates. A box is (c0, r0, width, height)
in whole px of the full frame. Frames are video frame numbers. Pen widths and mark sizes are
screen px. Nothing here imports torch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut  # noqa: F401 (QColor: offered from here before the split)

import pyqtgraph as pg

from outline_tracker import tracking
from outline_tracker.gui.click_rules import (DoubleClickWatch, Mark, PointTool, ResultsOnDisk, frame_shown, marks_on,
                                            point_kind, preview_input)
# The look of what is drawn has its own file; its names are passed on, as this module offered them before.
from outline_tracker.gui.prompt_drawing import (BLACK, CASING, LINE_WIDTHS, MARK_SIZES, MARK_SYMBOLS,  # noqa: F401
                                                PLUS_SIZE, WHITE, Drawn, colour_of, draw_outline, mark_spots)
from outline_tracker.gui.worker import Found, found_in
from outline_tracker.gui.worker_jobs import jobs_of
from outline_tracker.tracking_fine import crop_box  # noqa: F401 (offered from here before the split)
from outline_tracker.video import decoder_tag, frame_hash

KINDS = ("positive", "negative", "head")
TOOL_TEXTS = {"positive": "Positive: click on animal {id}.", "negative": "Negative: click on what is not animal {id}.",
              "head": "Head: click on the head of {id}."}
BUSY_TEXT = "The outline is being updated."
WAIT_TEXT = "Wait until tracking has stopped."  # a point tool's line while a job runs
WAIT_EXPORT_TEXT = "Wait until the export has ended."  # and while an export runs
LOADING_TEXT = "The model is loading. The outline appears when it is ready."
NO_OBJECT = "Click Add first. A point belongs to an object."
NOTHING_FOUND = "The model found nothing at the points of {objects}. Click on the animal itself."
NO_OUTLINE = "The outline could not be made. {reason}"
NO_MODEL = "The model could not be loaded, so no outline can be shown. {reason}"


@dataclass(frozen=True)
class _Shown:
    signature: tuple
    frame: int
    found: dict[str, Found]


class Prompts(QObject):
    """The point tools of one window, and what they draw (see the module's text).

    `selected`: the id of the object that gets the next point, or None. `tools`: the three
    `PointTool`s by kind; `tool_kind`: the kind of the one that is chosen, None for any other tool.
    `message`: (kind, text) for the panel to show, kind "warning" or "problem", ("", "") for none.
    `busy`: an outline is on its way. `marks`: the marks on the frame shown. `graphics`: the
    `Drawn` items of each outline shown, by id; `outlines` and `windows` read from them.
    `changed` is emitted whenever any of this, or the session, has changed.
    """

    changed = Signal()

    def __init__(self, window, worker):
        super().__init__(window)
        self._window, self._controller, self._view, self.worker = window, window.controller, window.view, worker
        self._jobs = jobs_of(window)
        self.results = ResultsOnDisk(window.controller)
        self.selected: str | None = None
        self.message, self._message_from = ("", ""), ""
        self.tools = {kind: PointTool(self, kind) for kind in KINDS}
        self.marks: list[Mark] = []
        self.graphics: dict[str, Drawn] = {}
        self._double_clicks = DoubleClickWatch(self._view.viewport())
        self._asked: tuple[int, tuple] | None = None  # the request on its way: its number and signature
        self._shown: _Shown | None = None             # the newest outlines, and the clicks they are for
        self._gave_up: tuple | None = None            # the clicks whose outline could not be made: not asked again

        self._points, self._pluses = pg.ScatterPlotItem(pxMode=True), pg.ScatterPlotItem(pxMode=True)
        self._points.setZValue(20)
        self._pluses.setZValue(21)
        self._view.add_item(self._points)
        self._view.add_item(self._pluses)

        undo = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), window)  # Ctrl+Z; Cmd+Z on a Mac
        undo.setContext(Qt.ShortcutContext.WindowShortcut)
        undo.activated.connect(self.undo)
        self._controller.video_opened.connect(self._video_opened)
        self._controller.session_changed.connect(self.refresh)
        self._view.frame_changed.connect(self._frame_requested)  # whoever put the frame on the screen
        self._view.tool_changed.connect(self._tool_changed)
        worker.preview_done.connect(self.show_preview)
        worker.preview_failed.connect(self._preview_failed)
        worker.busy_changed.connect(self._busy_changed)
        worker.state_changed.connect(self._model_state)
        self._jobs.writing_changed.connect(self._busy_changed)

    # ------------------------------------------------------------------ what is read

    @property
    def busy(self) -> bool:
        """Whether an outline was asked for and has not arrived yet."""
        return self.worker.busy

    @property
    def tool_kind(self) -> str | None:
        """ "positive", "negative" or "head" while that tool is chosen in the view; else None."""
        return next((kind for kind, tool in self.tools.items() if tool is self._view.tool), None)

    @property
    def outlines(self) -> dict[str, np.ndarray]:
        """The outline drawn for each object, by id: an array (N, 2) of points (u, v) along it, px
        of the full frame."""
        return {track_id: np.column_stack(drawn.line.getData())[:-1] for track_id, drawn in self.graphics.items()}

    @property
    def windows(self) -> dict[str, tuple[int, int, int, int]]:
        """The window drawn for each fine object that has an outline, by id: (c0, r0, W, W) in
        whole px of the full frame (SPEC 6.3)."""
        return {track_id: drawn.box for track_id, drawn in self.graphics.items() if drawn.box is not None}

    # ------------------------------------------------------------------ selection and tools

    def select(self, track_id: str | None) -> None:
        """Make the object `track_id` the one that gets the next point; None for no object, which
        also leaves a point tool."""
        self.selected = track_id
        self.refresh()

    def choose_tool(self, kind: str | None) -> None:
        """Choose the tool "positive", "negative" or "head", or None for Pan. Without a selected
        object there is nothing to click on, and the tool stays Pan."""
        self._view.set_tool(self.tools[kind] if kind is not None and self.selected is not None else None)

    def clicked(self, tool: str, u: float, v: float, button, modifiers) -> None:
        """A click of the point tool `tool` at (u, v), px of the video frame, with Qt's mouse button
        and held keys. The second click of a double click is dropped. What the click places is
        `click_rules.point_kind`: a negative point, a positive point, the head, or nothing."""
        if self._double_clicks.take_second():
            return
        kind = point_kind(tool, button, modifiers, sys.platform)
        if kind == "head":
            self.set_head(u, v)
        elif kind is not None:
            self.add_point(u, v, 1 if kind == "positive" else 0)

    # ------------------------------------------------------------------ edits

    def add_point(self, u: float, v: float, label: int) -> bool:
        """Add a point at (u, v), px of the full frame, to the selected object on the frame shown:
        `label` 1 for a positive point, 0 for a negative one. Returns whether it was added. A point
        outside the frame is not. A frame off the clip's grid is moved forward to the grid: the
        message says so, and the view goes to that frame."""
        session = self._controller.session
        inside = session is not None and 0 <= u < session.video.width and 0 <= v < session.video.height
        if not inside or self._jobs.writing():
            return False
        if self.selected is None:
            return self._refuse("warning", NO_OBJECT)
        try:
            frame, rgb = frame_shown(self._controller, self._view)
            edit = tracking.add_prompt(session, self.results.now(), self.selected, frame, (float(u), float(v)), label,
                                       frame_hash(rgb), decoder_tag())
        except ValueError as refused:
            return self._refuse("problem", str(refused))
        self._say("warning" if edit.note else "", edit.note, "edit")
        if edit.moved:
            self._window.show_frame(edit.frame)
        self._controller.touch()
        return True

    def set_head(self, u: float, v: float) -> bool:
        """Place the head of the selected object at (u, v), px of the full frame, on the frame
        shown. It is no point of the object and is never given to the model. Returns whether it
        was placed (the head belongs to the object's start frame)."""
        session = self._controller.session
        if session is None or self._view.frame is None or self._jobs.writing():
            return False
        if self.selected is None:
            return self._refuse("warning", NO_OBJECT)
        try:
            tracking.set_head(session, self.selected, self._view.frame, (float(u), float(v)))
        except ValueError as refused:
            return self._refuse("problem", str(refused))
        self._say("", "", "edit")
        self._controller.touch()
        return True

    def undo(self) -> bool:
        """Remove the newest point of the selected object. Returns whether there was one."""
        session = self._controller.session
        if session is None or self.selected is None or self._jobs.writing():
            return False
        try:
            removed = tracking.undo_prompt(session, self.results.now(), self.selected)
        except ValueError as refused:
            return self._refuse("problem", str(refused))
        if removed:
            self._say("", "", "edit")
            self._controller.touch()
        return removed

    def say(self, kind: str, text: str) -> None:
        """Show `text` in the panel: `kind` "warning" or "problem"; ("", "") takes the message away."""
        self._say(kind, text, "edit")
        self.changed.emit()

    def _say(self, kind: str, text: str, source: str) -> None:
        self.message, self._message_from = (kind, text) if text else ("", ""), source if text else ""

    def _refuse(self, kind: str, text: str) -> bool:
        self.say(kind, text)
        return False

    # ------------------------------------------------------------------ drawing

    def refresh(self) -> None:
        """Bring what is drawn in line with the session and the frame shown: the marks, the
        outlines, the tools' texts; ask for an outline when the clicks on this frame have none.
        Call it after showing another frame in a way the bottom bar does not report."""
        session = self._controller.session
        tracks = [] if session is None else session.tracks
        if self.selected is not None and all(track.id != self.selected for track in tracks):
            self.selected = None
        if self.selected is None and self.tool_kind is not None:
            self._view.set_tool(None)
        self._draw_marks(tracks)
        self._sync_preview()
        self._draw_outlines(tracks)
        self._update_tools()
        self.changed.emit()

    def _draw_marks(self, tracks) -> None:
        self.marks = marks_on(tracks, self._view.frame)
        points, pluses = mark_spots(self.marks, tracks)
        self._points.setData(points)
        self._pluses.setData(pluses)

    def _draw_outlines(self, tracks) -> None:
        for drawn in self.graphics.values():
            for item in drawn.items():
                self._view.remove_item(item)
        self.graphics = {}
        frame = self._view.frame
        if self._shown is None or self._shown.frame != frame:
            return
        for track in tracks:
            hit = self._shown.found.get(track.id)
            if hit is None or not any(prompt.frame == frame and prompt.points_px for prompt in track.prompts):
                continue
            drawn = draw_outline(track, hit, track.id == self.selected)
            for z, item in enumerate(drawn.items(), start=10):
                item.setZValue(z)
                self._view.add_item(item)
            self.graphics[track.id] = drawn

    def _update_tools(self) -> None:
        """Give the three tools their line and cursor, and make the view show them."""
        loading = self.worker.state in ("idle", "loading")
        for kind, tool in self.tools.items():
            text = (LOADING_TEXT if loading else BUSY_TEXT) if self.busy else TOOL_TEXTS[kind].format(id=self.selected)
            cursor = Qt.CursorShape.BusyCursor if self.busy else Qt.CursorShape.CrossCursor
            if self._jobs.writing():  # before anything else: a click does nothing now
                text = WAIT_TEXT if self._jobs.running else WAIT_EXPORT_TEXT
                cursor = Qt.CursorShape.ForbiddenCursor
            if (tool.text, tool.cursor) != (text, cursor):
                tool.text, tool.cursor = text, cursor
                if tool is self._view.tool:
                    self._view.set_tool(tool)  # the view takes the cursor, the window the line

    # ------------------------------------------------------------------ the preview

    def _wanted(self):
        """What the model is to outline on the frame shown (`click_rules.preview_input`), or None."""
        try:
            frame, rgb = frame_shown(self._controller, self._view)
        except ValueError:
            return None
        return preview_input(self._controller.session, frame, rgb)

    def _sync_preview(self) -> None:
        """Ask the worker for the outlines of the frame shown, unless they are there or on their way."""
        wanted, frame = self._wanted(), self._view.frame
        if wanted is None:  # nothing to outline here: what was made or asked for this frame is out of date
            if self._shown is not None and self._shown.frame == frame:
                self._shown = None
            if self._asked is not None and self._asked[1][0] == frame:
                self._asked = None
                self.worker.cancel_preview()
            return
        signature, image, box, prompts = wanted
        if (self._shown is not None and self._shown.signature == signature) or signature == self._gave_up or (
                self._asked is not None and self._asked[1] == signature):
            return
        if self.worker.state == "failed":
            self._say("problem", NO_MODEL.format(reason=self.worker.message), "preview")
            return
        self._asked = (self.worker.request_preview(frame, image, box[:2], prompts), signature)

    def show_preview(self, preview) -> None:
        """Take the outlines the worker made (a `worker.Preview`) and draw them. This is the slot of
        `Worker.preview_done`; it runs in the GUI thread."""
        if self._asked is None or preview.serial != self._asked[0]:
            return
        signature, self._asked = self._asked[1], None
        try:
            found, missing = found_in(preview, self._controller.session)
        except ValueError as error:  # a result that cannot be measured
            self._gave_up = signature
            self._say("problem", NO_OUTLINE.format(reason=str(error)), "preview")
            self.refresh()
            return
        self._shown = _Shown(signature, preview.frame, found)
        if missing:
            objects = ("object " if len(missing) == 1 else "objects ") + ", ".join(missing)
            self._say("warning", NOTHING_FOUND.format(objects=objects), "preview")
        elif self._message_from == "preview":
            self._say("", "", "")
        self.refresh()

    # ------------------------------------------------------------------ what the window and the worker report

    def _preview_failed(self, serial: int, reason: str) -> None:
        if self._asked is None or serial != self._asked[0]:
            return
        self._gave_up, self._asked = self._asked[1], None  # the same clicks are not tried again by themselves
        text = NO_MODEL if self.worker.state == "failed" else NO_OUTLINE
        self._say("problem", text.format(reason=reason), "preview")
        self.changed.emit()

    def _busy_changed(self, *_) -> None:
        """An outline is on its way or has arrived, or a tracking job or an export started or ended."""
        self._update_tools()
        self.changed.emit()

    def _model_state(self, state: str, message: str) -> None:
        """The model's state changed. A model that begins to load takes the place of the one before:
        that one's outlines, and the clicks it could not outline, are forgotten, and the frame
        shown is asked for again."""
        if state == "loading":
            self._shown = self._gave_up = None
            if self._message_from == "preview":
                self._say("", "", "")
            self.refresh()
        else:
            self._busy_changed()

    def _tool_changed(self) -> None:
        self.changed.emit()

    def _frame_requested(self, frame: int) -> None:
        self.refresh()

    def _video_opened(self) -> None:
        """Another video: back to Pan, and nothing of the video before it stays on the picture."""
        self._view.set_tool(None)
        self.selected, self._shown, self._gave_up = None, None, None
        if self._asked is not None:
            self._asked = None
            self.worker.cancel_preview()
        self._say("", "", "")
        self.refresh()
