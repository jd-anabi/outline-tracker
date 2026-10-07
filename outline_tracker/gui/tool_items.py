"""The graphics of the calibration tools on the picture (SPEC 10.1; the design note's video view):
stick, tape, circle points and fitted circle, axes arrows, and their labels.

Everything is white on a black casing, so that it shows on any video and is no track's colour:
lines 2 px wide on a casing of 4 px (black at alpha 180), the stick solid and the tape dashed, both
with crosshair ends of 15 px; circle points as hollow squares of 7 px and the fitted circle 1.5 px
wide; the axes as two arrows; labels as white text on black at alpha 180.

Coordinates and units: every point given here is (u, v) in px of the video frame (SPEC 3.1: u to
the right, v down, pixel centers at +0.5), as `VideoView.add_item` takes them. Line widths and the
sizes of ends, squares and text are screen px (Qt's device-independent px) at any zoom: pyqtgraph's
pens are cosmetic. Nothing here creates a Qt object when the module is imported.

`PointTool`, what the four tools of outline_tracker/gui/tools.py share, is here as well: the cross
cursor, which click places a point, and drawing the graphic again whenever the session changes.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QObject, Qt

import pyqtgraph as pg

from outline_tracker import geometry

WHITE = "#FFFFFF"
CASING = (0, 0, 0, 180)   # black, alpha 180: under every line, and behind every label
LINE_WIDTH = 2
CASING_WIDTH = 4
CIRCLE_WIDTH = 1.5
END_SIZE = 15             # the crosshair at an end of the stick or the tape
SQUARE_SIZE = 7           # a clicked point of the circle
HEAD_FRACTION = 0.25      # an arrow's head, as a part of the arrow's length
HEAD_ANGLE_DEG = 25.0     # between the arrow and each side of its head
CIRCLE_STEPS = 360        # corners of the polygon a circle is drawn as
LABEL_BEYOND = 1.3        # an axis label's distance from the origin, in arrow lengths
ZOOM_IN = " Scroll to zoom in for a precise click."


class Graphic:
    """The items one tool has on a view, added and removed together.

    `view` is the `VideoView` (its `add_item` and `remove_item`). `items` are the pyqtgraph items
    on the picture now, in the order they were added; `labels` are the texts of the labels among
    them. A tool clears the graphic and draws it again whenever what it shows has changed.
    """

    def __init__(self, view):
        self._view = view
        self.items: list = []
        self.labels: list[str] = []

    def clear(self) -> None:
        """Take every item of this graphic off the picture."""
        for item in self.items:
            self._view.remove_item(item)
        self.items, self.labels = [], []

    def _add(self, item):
        self._view.add_item(item)
        self.items.append(item)
        return item

    def _stroke(self, us, vs, width: float, dashed: bool = False, pairs: bool = False) -> None:
        """A casing and on it a white line of `width` screen px through the points (us, vs), px of
        the video frame; with `pairs`, one separate segment for every two points."""
        connect = "pairs" if pairs else "all"
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        for pen in (pg.mkPen(CASING, width=width + CASING_WIDTH - LINE_WIDTH),
                    pg.mkPen(WHITE, width=width, style=style)):
            self._add(pg.PlotCurveItem(x=list(us), y=list(vs), pen=pen, connect=connect, antialias=True))

    def _marks(self, points, symbol: str, size: int, width: float) -> None:
        """One mark of `size` screen px at every point (u, v) px of the video frame: its casing,
        and on it the mark in white, not filled."""
        us, vs = [point[0] for point in points], [point[1] for point in points]
        for pen in (pg.mkPen(CASING, width=width + CASING_WIDTH - LINE_WIDTH), pg.mkPen(WHITE, width=width)):
            self._add(pg.ScatterPlotItem(x=us, y=vs, symbol=symbol, size=size, pen=pen, brush=pg.mkBrush(None),
                                         pxMode=True, antialias=True))

    def line(self, p1, p2, dashed: bool = False) -> None:
        """The stick (solid) or the tape (`dashed`) from p1 to p2, px of the video frame."""
        self._stroke((p1[0], p2[0]), (p1[1], p2[1]), LINE_WIDTH, dashed)

    def ends(self, points) -> None:
        """A crosshair of 15 screen px at each of `points` (px of the video frame): the clicked
        ends of the stick or of the tape."""
        if points:
            self._marks(points, "crosshair", END_SIZE, LINE_WIDTH)

    def squares(self, points) -> None:
        """A hollow square of 7 screen px at each of `points` (px of the video frame): the clicked
        points of the circle."""
        if points:
            self._marks(points, "s", SQUARE_SIZE, CIRCLE_WIDTH)

    def circle(self, center, radius: float) -> None:
        """The fitted circle around `center` = (u, v) with `radius`, all in px of the video frame."""
        turns = [2 * math.pi * step / CIRCLE_STEPS for step in range(CIRCLE_STEPS + 1)]
        self._stroke([center[0] + radius * math.cos(turn) for turn in turns],
                     [center[1] + radius * math.sin(turn) for turn in turns], CIRCLE_WIDTH)

    def arrow(self, start, tip) -> None:
        """An arrow from `start` to `tip`, px of the video frame: the line, and a head of two short
        lines at the tip, a quarter of the arrow long."""
        du, dv = tip[0] - start[0], tip[1] - start[1]
        us, vs = [start[0], tip[0]], [start[1], tip[1]]
        for side in (-1, 1):
            turn = math.radians(180 + side * HEAD_ANGLE_DEG)  # back along the arrow, a little to one side
            cos, sin = math.cos(turn), math.sin(turn)
            us += [tip[0], tip[0] + HEAD_FRACTION * (du * cos - dv * sin)]
            vs += [tip[1], tip[1] + HEAD_FRACTION * (du * sin + dv * cos)]
        self._stroke(us, vs, LINE_WIDTH, pairs=True)

    def label(self, text: str, at, anchor=(0.5, 1.4)) -> None:
        """`text` in white on black at the point `at`, px of the video frame. `anchor` says which
        point of the text's box is put there, as fractions of its width and height from its top
        left corner: (0.5, 1.4) is the box centered above the point, clear of a line through it."""
        item = pg.TextItem(text, color=WHITE, fill=pg.mkBrush(CASING), anchor=anchor)
        item.setPos(at[0], at[1])
        self._add(item)
        self.labels.append(text)

    # ------------------------------------------------------------------ what each tool shows

    def pair(self, points, dashed: bool, text: str | None) -> None:
        """Draw the stick, or the tape (`dashed`), anew: a crosshair at each end placed so far
        (`points`, px of the video frame), the line once both are, and `text` above its middle."""
        self.clear()
        if len(points) == 2:
            (u1, v1), (u2, v2) = points
            self.line((u1, v1), (u2, v2), dashed)
            if text:
                self.label(text, ((u1 + u2) / 2, (v1 + v2) / 2))
        self.ends(points)

    def dish(self, points, center=None, radius: float = 0.0, text: str | None = None) -> None:
        """Draw the dish wall anew: a square at each clicked point and, with a `center` (u, v) and
        a `radius`, the fitted circle with `text` above its top; all in px of the video frame."""
        self.clear()
        if center is not None:
            self.circle(center, radius)
            if text:
                self.label(text, (center[0], center[1] - radius))
        self.squares(points)

    def axes(self, origin, angle_deg: float, length: float) -> None:
        """Draw the axes anew: two arrows of `length` px from `origin` = (u, v), px of the video
        frame, labelled "x" and "y". `angle_deg` is the direction of +x in degrees, counterclockwise
        on screen from the image's rightward direction; +y is a quarter turn further (SPEC 3.2), so
        for 0 x points right and y up. An origin or angle that is no number draws nothing."""
        self.clear()
        try:  # at 1 mm per px, `to_px` turns a length along an axis into a point of the frame
            frame = geometry.WorldFrame(1.0, math.radians(angle_deg), *origin)
        except (TypeError, ValueError):
            return
        for name, (x, y) in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
            tip = [float(c) for c in frame.to_px(length * x, length * y)]
            beyond = [float(c) for c in frame.to_px(LABEL_BEYOND * length * x, LABEL_BEYOND * length * y)]
            self.arrow((frame.u0, frame.v0), tip)
            self.label(name, beyond, anchor=(0.5, 0.5))


class PointTool(QObject):
    """What the four tools share: the cross cursor, the click rules, the graphic, and following the
    session. `controller` is the window's `SessionController`, `view` its `VideoView`.

    `graphic` holds what the tool has drawn (`tool_items.Graphic`). `refusal` is None, or the plain
    sentence that says why the last point or value was not taken.
    """

    cursor = Qt.CursorShape.CrossCursor
    precise = True  # a tool whose clicks set a length: its line says to zoom in while the picture is small

    def __init__(self, controller, view, parent: QObject | None = None):
        super().__init__(parent)
        self.controller, self.view = controller, view
        self.graphic = Graphic(view)
        self.refusal: str | None = None
        self._loaded = None    # the session object this tool has taken its points from
        self._double = False   # the click that comes next is the second one of a double click
        self._small = False    # the picture is drawn under 100%
        view.viewport().installEventFilter(self)
        view.zoom_changed.connect(self._zoom_changed)
        controller.video_opened.connect(self._video_opened)
        controller.session_changed.connect(self._session_changed)

    @property
    def session(self):
        """The session being worked on (`session.Session`), None before a video is open."""
        return self.controller.session

    @property
    def text(self) -> str:
        """What the line above the picture says while this tool is chosen."""
        return self._line() + (ZOOM_IN if self.precise and 0 < self.view.zoom < 1 else "")

    def click(self, u: float, v: float, button, modifiers) -> None:
        """A click of the view at (u, v), px of the video frame. The left button places a point
        (`add_tool_point`); the second click of a double click, and any other button, do nothing."""
        if self._double:
            self._double = False
        elif button == Qt.MouseButton.LeftButton:
            self.add_tool_point(u, v)

    def eventFilter(self, watched, event) -> bool:
        """Watch the view for a double click: Qt sends one as press, release, double click,
        release, and the view hands on a click for each release. Nothing is filtered out."""
        kind = event.type()
        if kind == QEvent.Type.MouseButtonDblClick:
            self._double = True
        elif kind == QEvent.Type.MouseButtonPress:
            self._double = False
        return False

    def _inside(self, u: float, v: float) -> bool:
        """Whether (u, v), px of the video frame, is a point of the open video's frame."""
        if self.session is None or not (math.isfinite(u) and math.isfinite(v)):
            return False
        return 0 <= u < self.session.video.width and 0 <= v < self.session.video.height

    def _say(self) -> None:
        """Show this tool's line again, if it is the chosen tool: the line has changed."""
        if self.view.tool is self:
            self.view.set_tool(self)

    def _zoom_changed(self, zoom: float) -> None:
        small = 0 < zoom < 1
        if small != self._small:  # the advice to zoom in comes or goes
            self._small = small
            if self.precise:
                self._say()

    def _video_opened(self) -> None:
        if self.view.tool is self:
            self.view.set_tool(None)
        self._session_changed()

    def _session_changed(self) -> None:
        if self.session is None:
            return
        if self.session is not self._loaded:
            self._loaded, self.refusal = self.session, None
            self._load()
        self._sync()
        self.draw()

    def _load(self) -> None:
        """Start from a session this tool has not seen yet."""

    def _sync(self) -> None:
        """Bring the tool's own points in line with the session, which may have been changed elsewhere."""

    def _line(self) -> str:
        raise NotImplementedError

    def add_tool_point(self, u: float, v: float) -> None:
        """Place a point at (u, v), px of the video frame, as a left click there does. A point
        outside the frame places nothing."""
        raise NotImplementedError

    def draw(self) -> None:
        """Draw the tool's graphics again from the session and the tool's own points."""
        raise NotImplementedError
