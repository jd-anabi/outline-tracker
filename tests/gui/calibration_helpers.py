"""What the tests of the calibration tools and of panels 3 and 4 share (task C3).

Imported by name from the test files beside it. Points are (u, v) in px of the video frame
(SPEC 3.1: u to the right, v down, pixel centers at +0.5); places in the view are Qt's
device-independent px of its viewport. Where an image point lies on the screen is worked out by
geometry (`fit_of`: the whole frame, in the middle of the view), not asked of the view.
"""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from gui_helpers import StandInSource, picture, pixels, show
from outline_tracker.session import Session

LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
NO_KEY = Qt.KeyboardModifier.NoModifier


def body(window, number: int):
    """The widget that the module of panel `number` built: the row under the panel's hint line."""
    return window.panels[number - 1].body.itemAt(1).widget()


def opened(window, qtbot, clip, dark_frame: bool = False):
    """Open `clip` in the window and show it. Returns (calibration panel, dish panel), the widgets
    of panels 3 and 4. With `dark_frame` the view shows a dark gray stand-in frame of the clip's
    size, on which the white tool graphics can be told from the picture."""
    source = StandInSource(clip.scene.size) if dark_frame else None
    picture(window, qtbot, clip, source)
    QApplication.processEvents()
    return body(window, 3), body(window, 4)


def fit_of(view, size):
    """Where a frame of `size` = (width, height) px lies in the view when all of it shows, by
    geometry: (screen px per video px, left, top), the corner in the viewport's px."""
    (width, height), room = size, view.viewport().size()
    scale = min(room.width() / width, room.height() / height)
    return scale, (room.width() - width * scale) / 2, (room.height() - height * scale) / 2


def settle(view) -> None:
    """Let the window finish laying itself out: the line above the picture has another length for
    each tool and each step, and the room for the picture may follow it over a few rounds of events."""
    before = None
    for _ in range(20):
        QApplication.processEvents()
        now = view.viewport().size()
        if now == before:
            return
        before = now


def screen_point(view, size, u: float, v: float) -> QPoint:
    """The viewport px whose corner is nearest to the image point (u, v) of a fitted frame of
    `size`: a click there is within half a screen px of the point."""
    settle(view)
    scale, left, top = fit_of(view, size)
    return QPoint(round(left + u * scale), round(top + v * scale))


def outside_points(view, size) -> list[QPoint]:
    """Two viewport px beside a fitted frame of `size`, on the canvas: above and below it, or left
    and right of it, wherever the view has room (a frame fills the view in one direction only)."""
    settle(view)
    (scale, left, top), room = fit_of(view, size), view.viewport().size()
    if top >= 12:
        return [QPoint(room.width() // 2, round(top) - 6), QPoint(room.width() // 2, room.height() - 3)]
    assert left >= 12, "the frame fills the view: there is no canvas to click on"
    return [QPoint(round(left) - 6, room.height() // 2), QPoint(room.width() - 3, room.height() // 2)]


def click_at(qtbot, view, size, u: float, v: float, button=LEFT) -> None:
    """Click with the mouse on the image point (u, v) of a fitted frame of `size`."""
    qtbot.mouseClick(view.viewport(), button, NO_KEY, screen_point(view, size, u, v))
    QApplication.processEvents()


def lit(view, size, u: float, v: float, within: float = 2) -> bool:
    """Whether something light is drawn at the image point (u, v) of a fitted frame of `size`:
    a px at most `within` screen px left, right, above or below it whose red, green and blue are
    all 128 or more. The stand-in frame is gray 60 to 64 and the canvas is darker, so only a tool's
    white line is that light. (A hollow square of 7 px has its line 3.5 px from its middle.)"""
    settle(view)
    image = view.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    scale, left, top = fit_of(view, size)
    column, row = round((left + u * scale) * ratio), round((top + v * scale) * ratio)
    reach = math.ceil(within * ratio)
    near = pixels(image)[max(row - reach, 0):row + reach + 1, max(column - reach, 0):column + reach + 1]
    return bool((near.min(axis=2) >= 128).any())


def on_circle(center, radius: float, degrees) -> list[list[float]]:
    """Points on the circle around `center` = (u, v) with `radius`, at the given angles (degrees,
    clockwise on screen from the image's rightward direction); all in px of the video frame."""
    return [[center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a))]
            for a in degrees]


def place(tool, points) -> None:
    """Give the tool each of `points` as a click would."""
    for u, v in points:
        tool.add_tool_point(u, v)


def restart(window) -> Session:
    """What reopening a saved session does to the window: the session is written as session.json
    holds it, read again, and given to the controller as the session of the open video. Returns
    the new session object."""
    controller = window.controller
    again = Session.from_json(json.loads(json.dumps(controller.session.to_json())))
    controller.session = again
    controller.video_opened.emit()
    QApplication.processEvents()
    return again


def shown(window, qtbot):
    """Show the window with panels 3 and 4 open, and return it."""
    for number in (3, 4):
        window.panels[number - 1].set_expanded(True)
    show(window, qtbot)
    QApplication.processEvents()
    return window
