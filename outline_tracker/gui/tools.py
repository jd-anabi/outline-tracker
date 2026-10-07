"""The calibration tools of the video view (SPEC 4.2 to 4.5, 10.1, 10.2): Stick, Tape, Circle, Axes.

A tool is what `VideoView.set_tool` takes: `click(u, v, button, modifiers)` and `cursor`, and a
`text` for the line above the picture. A left click calls `add_tool_point(u, v)`; of a double click
the first click counts and the second is dropped. Each tool keeps one block of the session
(SPEC 8.10) and draws it on the picture:

- `StickTool`: `calibration.stick` = {"p1_px", "p2_px", "length_mm"}; the second end finishes it.
- `TapeTool`: `calibration.check` = {"p1_px", "p2_px", "true_mm"}; it needs the stick's scale.
- `CircleTool`: `circle` (points, and center, radius and RMS of `geometry.fit_circle`, `dish_mm`).
- `AxesTool`: `axes` (the origin, placed by one click, and the angle).

What the four share (the click rules, following the session) is `PointTool`, and how their graphics
look is `Graphic`, both in outline_tracker/gui/tool_items.py.

A block is in the session only while it is complete (two ends; a circle that was fitted), because
the core takes a block that is there as usable; points placed before that are the tool's own
(`points`). After every change the tool calls `controller.touch()`, and it draws its graphics again
from the session whenever a session is opened or changed. A tool that finishes returns the view to
Pan, and so does opening another video.

Coordinates and units (SPEC 3): points are (u, v) in px of the video frame (u to the right, v down,
pixel centers at +0.5); lengths typed by the user are mm; the axes angle is in degrees,
counterclockwise on screen from the image's rightward direction. Every number shown comes from
`geometry.stick_scale`, `geometry.tape_check` and `geometry.fit_circle`.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QObject

from outline_tracker import geometry
from outline_tracker.gui.tool_items import PointTool
from outline_tracker.session import Circle

STICK_MM, TAPE_MM = 30.0, 20.0  # the lengths a new session starts with (the course's ruler, SPEC 8.10)
AXIS_FRACTION = 0.15            # an axis arrow's length, as a part of the frame's shorter side
NEEDS_THREE = "The circle needs 3 or more points. Click more points on the dish wall."


def mm_per_px(session) -> float | None:
    """The session's scale in mm per px (`Session.world_frame`), or None while it has none."""
    try:
        return session.world_frame().k_mm_per_px
    except ValueError:
        return None


class _PairTool(PointTool):
    """A tool of two points and one length in mm: the stick, or the tape check. `points` are the
    ends placed so far, [[u, v], ...] in px of the video frame; `value_mm` is the typed length."""

    block = ""        # the key of `session.calibration` this tool keeps
    value_key = ""    # the key of the length in that block
    default_mm = 0.0  # the length of a new session, mm
    what = ""         # the length's name in a refusal
    dashed = False

    def __init__(self, controller, view, parent: QObject | None = None):
        super().__init__(controller, view, parent)
        self.points: list[list[float]] = []
        self.value_mm = self.default_mm
        self._stored = False  # the two ends are in the session

    def _block(self) -> dict | None:
        """The session's block, or None while it has none (or one that lacks a key, as a session
        file edited by hand may)."""
        block = getattr(self.session.calibration, self.block)
        return block if isinstance(block, dict) and {"p1_px", "p2_px", self.value_key} <= block.keys() else None

    def _load(self) -> None:
        self.points, self.value_mm, self._stored = [], self.default_mm, False

    def _sync(self) -> None:
        block = self._block()
        if block is not None:
            try:
                points = [[float(c) for c in block[key]] for key in ("p1_px", "p2_px")]
                self.points, self.value_mm = points, float(block[self.value_key])
            except (TypeError, ValueError):
                block = None  # no points and length that can be shown
        if block is None and self._stored:
            self.points = []  # cleared elsewhere
        self._stored = block is not None

    def _usable(self) -> bool:
        return True

    def _checked(self, p1, p2):
        """What geometry makes of the two ends; ValueError with a plain message if they are no pair."""
        raise NotImplementedError

    def add_tool_point(self, u: float, v: float) -> None:
        """Place an end at (u, v), px of the video frame. The second end completes the block in the
        session and returns the view to Pan; a third click starts again with a first end. A second
        end on the first one is refused in geometry's words (`refusal`)."""
        if not self._inside(u, v) or not self._usable():
            return
        point = [float(u), float(v)]
        if len(self.points) == 1:
            try:
                self._checked(self.points[0], point)
            except ValueError as refused:
                self.refusal = str(refused)
                self.controller.touch()
                return
            self.points = [self.points[0], point]
        else:
            self.points = [point]
        self.refusal = None
        self._store()
        if len(self.points) == 2 and self.view.tool is self:
            self.view.set_tool(None)
        else:
            self._say()

    def redo(self) -> None:
        """Remove the ends, from the session too, so that they can be clicked again."""
        if self.session is not None:
            self.points, self.refusal = [], None
            self._store()
            self._say()

    def set_value(self, mm: float) -> None:
        """Take the typed length in mm; with the two ends there it is in the session at once.
        Raises ValueError, and changes nothing, for a length that is not positive and finite."""
        mm = float(mm)
        if not (math.isfinite(mm) and mm > 0):
            raise ValueError(f"The {self.what} must be more than 0 mm.")
        self.value_mm = mm
        if self.session is not None:
            if self._block() is not None:
                self._block()[self.value_key] = mm
            self.controller.touch()

    def _store(self) -> None:
        block = None
        if len(self.points) == 2:
            block = {"p1_px": list(self.points[0]), "p2_px": list(self.points[1]), self.value_key: self.value_mm}
        setattr(self.session.calibration, self.block, block)
        self._stored = block is not None
        self.controller.touch()

    def _label(self) -> str | None:
        raise NotImplementedError

    def draw(self) -> None:
        self.graphic.pair(self.points, self.dashed, self._label() if len(self.points) == 2 else None)


class StickTool(_PairTool):
    """The calibration stick (SPEC 4.2): two clicked ends of a known length, `length_mm`."""

    block, value_key, default_mm, what = "stick", "length_mm", STICK_MM, "stick length"

    @property
    def length_mm(self) -> float:
        """The stick's true length as typed, mm."""
        return self.value_mm

    def scale(self) -> geometry.StickScale | None:
        """The scale the session's stick gives (`geometry.stick_scale`: mm per px, its relative
        uncertainty, the stick's length in px), or None while there is no usable stick."""
        block = self._block() if self.session is not None else None
        try:
            return None if block is None else self._checked(block["p1_px"], block["p2_px"], block["length_mm"])
        except (TypeError, ValueError):
            return None

    def _checked(self, p1, p2, length_mm=None):
        return geometry.stick_scale(p1, p2, self.value_mm if length_mm is None else length_mm,
                                    self.session.calibration.click_sigma_px)

    def _line(self) -> str:
        return "Stick: click the second end." if len(self.points) == 1 else "Stick: click the first end on the ruler."

    def _label(self) -> str:
        return f"Stick {self.value_mm:.2f} mm"


class TapeTool(_PairTool):
    """The tape-measure check (SPEC 4.3): two other ruler marks and their true distance, `true_mm`."""

    block, value_key, default_mm, what, dashed = "check", "true_mm", TAPE_MM, "true distance", True

    @property
    def true_mm(self) -> float:
        """The true distance of the two marks as typed, mm."""
        return self.value_mm

    def check(self) -> geometry.TapeCheck | None:
        """The session's check at the session's scale (`geometry.tape_check`: the measured distance
        in mm, its signed relative error, whether it passes), or None without marks or scale."""
        block = self._block() if self.session is not None else None
        try:
            return None if block is None else self._checked(block["p1_px"], block["p2_px"], block["true_mm"])
        except (TypeError, ValueError):
            return None

    def _usable(self) -> bool:
        return mm_per_px(self.session) is not None

    def _checked(self, p1, p2, true_mm=None):
        scale = mm_per_px(self.session)
        if scale is None:
            raise ValueError("The tape check needs the scale. Place the stick first.")
        return geometry.tape_check(p1, p2, self.value_mm if true_mm is None else true_mm, scale)

    def _sync(self) -> None:
        super()._sync()
        if self.view.tool is self and not self._usable():
            self.view.set_tool(None)  # the scale went while the tool was chosen

    def _line(self) -> str:
        return f"Tape: click the {'second' if len(self.points) == 1 else 'first'} ruler mark."

    def _label(self) -> str | None:
        check = self.check()
        return None if check is None else f"Tape {check.measured_mm:.2f} mm"


class AxesTool(PointTool):
    """The axes (SPEC 4.5): one click places the origin; the angle is typed.

    `placed` says whether the origin was put where it is, by a click or by `origin_to_center`.
    Until then it is where SPEC 4.5 puts it: at the circle's center if a circle is fitted, else at
    the frame's center (width / 2, height / 2), px of the video frame.
    """

    precise = False

    def __init__(self, controller, view, parent: QObject | None = None):
        super().__init__(controller, view, parent)
        self.placed = False

    @property
    def arrow_length_px(self) -> float:
        """How long an axis arrow is drawn, px of the video frame."""
        return AXIS_FRACTION * min(self.session.video.width, self.session.video.height)

    def _default(self) -> list[float]:
        circle, video = self.session.circle, self.session.video
        return [float(c) for c in circle.center_px] if circle is not None else [video.width / 2, video.height / 2]

    def _load(self) -> None:
        axes = self.session.axes
        try:
            unset = [float(c) for c in axes.origin_px] == [0.0, 0.0]  # as a new session has it: no origin of its own
        except (TypeError, ValueError):
            unset = True  # nothing that can be an origin (a session file edited by hand)
        if unset or len(axes.origin_px) != 2:
            axes.origin_px = self._default()
        self.placed = list(axes.origin_px) != self._default()

    def follow_circle(self) -> None:
        """The circle was fitted or cleared: an origin that was never placed goes to its new default."""
        if not self.placed:
            self.session.axes.origin_px = self._default()

    def add_tool_point(self, u: float, v: float) -> None:
        """Put the origin at (u, v), px of the video frame, and return the view to Pan."""
        if not self._inside(u, v):
            return
        self.session.axes.origin_px, self.placed = [float(u), float(v)], True
        self.controller.touch()
        if self.view.tool is self:
            self.view.set_tool(None)

    def origin_to_center(self) -> None:
        """Copy the fitted circle's center to the origin (px of the video frame). Without a circle
        nothing happens."""
        if self.session is not None and self.session.circle is not None:
            self.session.axes.origin_px, self.placed = [float(c) for c in self.session.circle.center_px], True
            self.controller.touch()

    def set_angle(self, degrees: float) -> None:
        """Take the direction of +x in degrees, counterclockwise on screen from the image's
        rightward direction. Raises ValueError, and changes nothing, for a value that is no number."""
        degrees = float(degrees)
        if not math.isfinite(degrees):
            raise ValueError("The axes angle must be a number of degrees.")
        if self.session is not None:
            self.session.axes.angle_deg = degrees
            self.controller.touch()

    def _line(self) -> str:
        return "Axes: click the new origin."

    def draw(self) -> None:
        self.graphic.axes(self.session.axes.origin_px, self.session.axes.angle_deg, self.arrow_length_px)


class CircleTool(PointTool):
    """The dish wall (SPEC 4.4): any number of clicked points, fitted from the third one on.

    `points` are the clicked points [[u, v], ...] in px of the video frame. `dish_mm` is the dish's
    known inner diameter in mm, or None. `axes` is the window's `AxesTool`: its origin follows the
    fit until it is placed. `refusal` says why nothing is fitted (too few points, points on a line).
    """

    def __init__(self, controller, view, axes: AxesTool, parent: QObject | None = None):
        super().__init__(controller, view, parent)
        self.axes = axes
        self.points: list[list[float]] = []
        self.dish_mm: float | None = None
        self._stored = False  # the points are in the session, as a fitted circle

    def _load(self) -> None:
        self.points, self._stored = [], False
        self.dish_mm = None if self.session.circle is not None else self._manifest_dish_mm()

    def _manifest_dish_mm(self) -> float | None:
        """The dish diameter in mm that the course's manifest states for the open video (SPEC 4.4)."""
        video = self.controller.video_path
        try:
            manifest = self.session.time.manifest_path or geometry.find_manifest(video)
            return None if manifest is None else geometry.dish_mm_from_manifest(video, manifest)
        except (OSError, TypeError):
            return None

    def _sync(self) -> None:
        circle = self.session.circle
        if circle is not None:
            self.points, self.dish_mm = [list(point) for point in circle.points_px], circle.dish_mm
        elif self._stored:
            self.points = []  # cleared elsewhere
        self._stored = circle is not None

    def add_tool_point(self, u: float, v: float) -> None:
        """Add a wall point at (u, v), px of the video frame, and fit the circle again."""
        if self._inside(u, v):
            self.points = [*self.points, [float(u), float(v)]]
            self._fit()

    def redo(self) -> None:
        """Remove the points and the circle, so that the wall can be clicked again."""
        if self.session is not None:
            self.points = []
            self._fit()

    def set_dish_mm(self, mm: float | None) -> None:
        """Take the dish's inner diameter in mm, or None for "not known"; with a circle there it is
        in the session at once. Raises ValueError, and changes nothing, for a value that is not
        positive and finite."""
        if mm is not None:
            mm = float(mm)
            if not (math.isfinite(mm) and mm > 0):
                raise ValueError("The dish diameter must be more than 0 mm.")
        self.dish_mm = mm
        if self.session is not None:
            if self.session.circle is not None:
                self.session.circle.dish_mm = mm
            self.controller.touch()

    def _fit(self) -> None:
        """Fit the points (`geometry.fit_circle`) and put the circle into the session; without a fit
        the session has no circle, and `refusal` says why."""
        session, fitted, self.refusal = self.session, None, None
        if 0 < len(self.points) < 3:
            self.refusal = NEEDS_THREE
        elif self.points:
            try:
                fitted = geometry.fit_circle(self.points)
            except ValueError as refused:
                self.refusal = str(refused)
        if fitted is None:
            session.circle = None
        else:
            session.circle = Circle(points_px=[list(point) for point in self.points],
                                    center_px=[float(c) for c in fitted.center_px],
                                    radius_px=float(fitted.radius_px), rms_px=float(fitted.rms_px),
                                    dish_mm=self.dish_mm,
                                    extra=session.circle.extra if session.circle is not None else {})
        self._stored = fitted is not None
        self.axes.follow_circle()
        self.controller.touch()
        self._say()

    def _line(self) -> str:
        return f"Circle: click points on the inner wall of the dish. {len(self.points)} placed; 6 or more is best."

    def draw(self) -> None:
        circle, scale = self.session.circle, mm_per_px(self.session)
        if circle is None:
            self.graphic.dish(self.points)
        else:
            text = f"R = {scale * circle.radius_px:.2f} mm" if scale else f"R = {circle.radius_px:.1f} px"
            self.graphic.dish(self.points, circle.center_px, circle.radius_px, text)
