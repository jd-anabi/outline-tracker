"""The video view (SPEC 3.1, 10.1, 10.2): one frame of the video, zoomed and moved by the user.

The frame is a pyqtgraph `ImageItem` in a `ViewBox`, with the array read row by row (`row-major`)
and the y axis pointing down. In the item's coordinates pixel (c, r) of the frame is the square
[c, c+1) x [r, r+1): a point of the picture is (u, v) of SPEC 3.1 as it is, and nothing here adds
or takes half a pixel. Graphics that other parts add (`add_item`) are in the same coordinates.

The wheel zooms at the cursor and a drag with the left button moves the picture, whatever tool is
chosen; a click (press and release in one place) goes to the tool. Frames come one at a time from
the source's `get(frame)`; the video is never read whole. Screen lengths are Qt's
device-independent px: "100%" is one video px on one of them.

pyqtgraph is imported after PySide6, so it uses that binding (outline_tracker/gui/app.py).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import QWidget

import pyqtgraph as pg

from outline_tracker.gui import theme

PAN_CURSOR = Qt.CursorShape.OpenHandCursor
DRAG_CURSOR = Qt.CursorShape.ClosedHandCursor


class _PictureBox(pg.ViewBox):
    """The ViewBox of the picture: its proportions are kept, v points down, and it has no menu.
    Only the left button moves it. pyqtgraph would zoom on a drag with the right button, and the
    right button is a point of its own here: a hand that slips while clicking must not zoom.
    `dragged(held)` says when a drag starts (true) and ends (false)."""

    dragged = Signal(bool)

    def __init__(self):
        super().__init__(lockAspect=True, invertY=True, enableMenu=False, defaultPadding=0.0)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis)
        self.dragged.emit(not ev.isFinish())


class VideoView(pg.GraphicsView):
    """Shows one frame of a source and reports clicks on it to the tool.

    `frame`: the number of the frame shown (video frame number, counted from 0), None while none
    is. `tool`: the tool that gets the clicks, None for Pan. `zoom`: screen px per video px.
    `zoom_changed(zoom)` and `tool_changed()` say when these change. `view_box` and `image_item`
    are pyqtgraph's parts. Esc returns to Pan, wherever the focus is in the view's window.
    """

    zoom_changed = Signal(float)
    tool_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, background=theme.LIGHT["canvas"])  # the same colour in both themes
        self.source = None
        self.frame: int | None = None
        self.tool = None
        self._zoom = 0.0
        self.view_box = _PictureBox()
        self.setCentralItem(self.view_box)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view_box.addItem(self.image_item)
        self.viewport().setCursor(PAN_CURSOR)

        self.view_box.sigRangeChanged.connect(self._view_changed)
        self.view_box.sigResized.connect(self._view_changed)
        self.view_box.dragged.connect(self._dragged)
        self.scene().sigMouseClicked.connect(self._clicked)
        escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape.setContext(Qt.ShortcutContext.WindowShortcut)
        escape.activated.connect(lambda: self.set_tool(None))

    # ------------------------------------------------------------------ frames

    def set_source(self, source) -> None:
        """Take the source of the frames: an object whose `get(frame)` returns frame `frame` as an
        RGB uint8 array [row, column, 3] (a `FrameSource`), or None. No frame is shown until
        `show_frame`; the first one shown is fitted into the view."""
        self.source = source
        self.frame = None
        self.image_item.clear()
        self.fit()

    def show_frame(self, frame: int) -> None:
        """Show video frame `frame` (counted from 0) of the source, with the gray levels it has.
        Raises `IndexError` when the video has no such frame; the frame shown then stays."""
        image = self.source.get(frame)
        self.image_item.setImage(image, autoLevels=False)
        self.frame = int(frame)

    # ------------------------------------------------------------------ zoom

    @property
    def zoom(self) -> float:
        """How large the picture is drawn: screen px (device-independent) per video px. 1.0 is
        "100%"; 0.0 while the view has no size yet."""
        return self._zoom

    def fit(self) -> None:
        """Show the whole frame, as large as the view allows, in its middle. The picture then
        follows the view's size until the user zooms or moves it."""
        self.view_box.enableAutoRange()

    def one_to_one(self) -> None:
        """Draw one video px on one screen px (100%), keeping the point in the middle of the view
        in the middle."""
        middle = self.view_box.viewRect().center()
        width, height = self.view_box.width(), self.view_box.height()
        left, top = round(middle.x() - width / 2), round(middle.y() - height / 2)  # px edges on px edges
        self.view_box.setRange(xRange=(left, left + width), yRange=(top, top + height), padding=0)

    def _view_changed(self, *_) -> None:
        width = self.view_box.viewRect().width()
        zoom = self.view_box.width() / width if width > 0 else 0.0
        # a picture drawn smaller than it is gets smoothed; enlarged, its pixels stay sharp squares
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, zoom < 1)
        if zoom != self._zoom:
            self._zoom = zoom
            self.zoom_changed.emit(zoom)

    # ------------------------------------------------------------------ tools and clicks

    def set_tool(self, tool) -> None:
        """Choose the tool that gets the clicks, or None for Pan. A tool is any object with a method
        `click(u, v, button, modifiers)` and an attribute `cursor` (a `Qt.CursorShape` or a
        `QCursor`), shown over the picture. A click inside the frame calls `click` with the point
        in px of the video frame (SPEC 3.1: u to the right, v down, pixel centers at +0.5), the
        mouse button and the keys held; a click outside the frame goes nowhere."""
        self.tool = tool
        self.viewport().setCursor(PAN_CURSOR if tool is None else tool.cursor)
        self.tool_changed.emit()

    def _dragged(self, held: bool) -> None:
        if self.tool is None:
            self.viewport().setCursor(DRAG_CURSOR if held else PAN_CURSOR)

    def _clicked(self, event) -> None:
        image = self.image_item.image
        if self.tool is None or image is None:
            return
        at = self.image_item.mapFromScene(event.scenePos())
        rows, columns = image.shape[:2]
        if 0 <= at.x() < columns and 0 <= at.y() < rows:
            event.accept()
            self.tool.click(at.x(), at.y(), event.button(), event.modifiers())

    # ------------------------------------------------------------------ graphics of other parts

    def add_item(self, item) -> None:
        """Draw a pyqtgraph (or Qt) graphics item over the picture. Its coordinates are px of the
        video frame (SPEC 3.1). It does not change what `fit` shows."""
        self.view_box.addItem(item, ignoreBounds=True)

    def remove_item(self, item) -> None:
        """Take an item added with `add_item` off the picture."""
        self.view_box.removeItem(item)
