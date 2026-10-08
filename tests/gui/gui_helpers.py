"""What the tests of the video view, the navigation and the opening of a video share (task C1 on).

Imported by name from the test files beside it. Fixtures are not here: they are in tests/conftest.py and
tests/gui/conftest.py.

Where a part of the picture is on the screen is read from the drawn widget (`drawn`), never from
the view's own numbers: a stand-in frame has one block of pixels in a colour that nothing else
has, and the block is looked for in the grabbed image. Positions are Qt's device-independent px of
the widget; a grabbed image has `devicePixelRatio` device px for each of them.

`layout_findings` says what breaks the rule for the parts of a bar: nothing outside, nothing over
another part, no text cut. tests/gui/test_row_rule.py tries it on a made-up widget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStyle, QStyleOptionButton, QWidget

MARK = (255, 0, 255)   # the marked pixels of a stand-in frame (RGB): no other pixel has this colour
GREEN = (0, 255, 0)    # a second colour that no stand-in frame has


class StandInSource:
    """Frames made in memory, in place of a `FrameSource`: `get(k)` is frame k, an RGB uint8 array
    [row, column, 3] of `size` = (width, height) px, read-only; an `IndexError` outside 0 to
    `n_frames - 1`.

    Frame k is gray, 60 + k % 90 in even rows and 4 more in odd rows: so little contrast that a
    view which stretched the levels would show other values, and another gray for each of 90
    frames in a row. `marked` = (c0, r0, c1, r1) paints the pixels of columns c0 to c1 - 1 and rows
    r0 to r1 - 1 in `MARK`.
    """

    def __init__(self, size=(320, 240), n_frames=5, marked=None):
        self.size, self.n_frames, self.marked = size, n_frames, marked

    def get(self, k: int) -> np.ndarray:
        if not 0 <= k < self.n_frames:
            raise IndexError(f"The stand-in has no frame {k}.")
        width, height = self.size
        image = np.full((height, width, 3), 60 + k % 90, np.uint8)
        image[1::2] += 4
        if self.marked is not None:
            c0, r0, c1, r1 = self.marked
            image[r0:r1, c0:c1] = MARK
        image.flags.writeable = False
        return image


class ClickRecorder:
    """A tool for the video view: every click it is given is kept in `clicks` as
    (u, v, button, modifiers), u and v in px of the video frame (SPEC 3.1)."""

    cursor = Qt.CursorShape.CrossCursor
    text = "Recorder: click on the picture."

    def __init__(self):
        self.clicks = []

    def click(self, u, v, button, modifiers):
        self.clicks.append((u, v, button, modifiers))


class Dialogs:
    """What the window asked `dialogs.message` and `dialogs.open_file` for, in place of the dialogs:
    `messages` holds (parent, kind, text), `files` holds (parent, title, filters, on_chosen)."""

    def __init__(self):
        self.messages, self.files = [], []

    def message(self, parent, kind, text):
        self.messages.append((parent, kind, text))

    def open_file(self, parent, title, filters, on_chosen):
        self.files.append((parent, title, filters, on_chosen))


def record_dialogs(monkeypatch) -> Dialogs:
    """Replace the two functions of outline_tracker/gui/dialogs.py by a recorder, and return it."""
    from outline_tracker.gui import dialogs

    recorder = Dialogs()
    monkeypatch.setattr(dialogs, "message", recorder.message)
    monkeypatch.setattr(dialogs, "open_file", recorder.open_file)
    return recorder


def show(window, qtbot):
    """Show the window (offscreen), make it the active one (key shortcuts go to the active window),
    and return it."""
    with qtbot.waitExposed(window):
        window.show()
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window)
    return window


def shown(window, qtbot):
    """Show the window (offscreen) and wait until it is on the screen. Unlike `show`, it does not
    make the window the active one."""
    with qtbot.waitExposed(window):
        window.show()
    return window


def body(window, number: int):
    """The widget that the module of panel `number` (1 to 9) put under the panel's hint line."""
    return window.panels[number - 1].body.itemAt(1).widget()


def picture(window, qtbot, clip, source=None):
    """Open `clip` (a GroundTruth with its `path`) in the window, show the window, and return its
    video view. With `source`, the view then shows frame 0 of that stand-in, whole."""
    window.open_path(clip.path)
    show(window, qtbot)
    view = window.view
    if source is not None:
        view.set_source(source)
        view.show_frame(0)
        view.fit()
        QApplication.processEvents()
    return view


def visible_children(parent) -> list:
    """The widgets that are children of `parent` itself (not of its children) and show when it shows."""
    children = parent.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
    return [child for child in children if child.isVisibleTo(parent)]


def layout_findings(parent, longest=None) -> list[str]:
    """What breaks the rule for the parts of a bar among the visible children of `parent`: one line
    for each finding, or an empty list. The rule is that nothing is outside, nothing lies over
    another part, and no text is cut:

    - `outside`: a child that does not lie wholly inside the parent's contents rect;
    - `overlap`: two children whose rectangles share an area (a shared edge is none);
    - `cut`: a push button with less room for text, by its own style, than its text is wide; a
      label that is narrower than its text.

    longest: {child: the longest text it will show}, for a child that will show more than it does
    now; any other child is measured with the text it shows. A text is as wide as its advance in
    the child's own font, and it is taken for one line. A hidden child is left out.

    Device-independent px. A rectangle is written (left, top, width, height) in the parent's px.
    """
    def named(child) -> str:
        text = child.text() if hasattr(child, "text") else ""
        return f"{type(child).__name__} {text!r}" if text else type(child).__name__

    def placed(child) -> str:
        return f"{named(child)} at {child.geometry().getRect()}"

    def room(child) -> int:
        """The width the child has for its text."""
        if isinstance(child, QLabel):
            return child.contentsRect().width() - 2 * child.margin()
        option = QStyleOptionButton()
        child.initStyleOption(option)
        return child.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, child).width()

    children, inside = visible_children(parent), parent.contentsRect()
    found = [f"outside: {placed(child)} is not inside {inside.getRect()}"
             for child in children if not inside.contains(child.geometry())]
    found += [f"overlap: {placed(one)} and {placed(other)}"
              for index, one in enumerate(children) for other in children[index + 1:]
              if one.geometry().intersects(other.geometry())]
    for child in children:
        if isinstance(child, (QPushButton, QLabel)):
            text = (longest or {}).get(child, child.text())
            needed = child.fontMetrics().horizontalAdvance(text)
            if room(child) < needed:
                found.append(f"cut: {named(child)} has {room(child)} px for {text!r}, which is {needed} px wide")
    return found


def pixels(image: QImage) -> np.ndarray:
    """A QImage as an RGB uint8 array [row, column, 3], one entry per device px."""
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, :3 * width].reshape(height, width, 3).copy()


def shown_pixels(view) -> np.ndarray:
    """The picture the view's image item draws, as an RGB uint8 array [row, column, 3] in px of the
    video frame: what is on the screen at 100%, after everything the item does to the array."""
    return pixels(view.image_item.getPixmap().toImage())


def badge_pixels(window, badge) -> list[tuple[float, float, str]]:
    """Every device pixel that `window` draws in `badge`'s square, row by row: (x, y, colour). x and y
    are the pixel's centre in px from the badge's top left corner (x right, y down); the colour is
    `#RRGGBB` in capitals."""
    image = window.grab().toImage()
    ratio = image.devicePixelRatio()
    corner = badge.mapTo(window, QPoint(0, 0)) * ratio
    return [((column + 0.5) / ratio, (row + 0.5) / ratio,
             image.pixelColor(corner.x() + column, corner.y() + row).name().upper())
            for row in range(round(badge.height() * ratio)) for column in range(round(badge.width() * ratio))]


def inside_badge(x: float, y: float) -> bool:
    """True for a pixel with its centre at (x, y), px from the top left corner of a badge, that lies
    wholly inside the badge's outline ring. The badge is a circle 20 px across, so its centre is at
    (10, 10) and the 1 px ring's inner edge 9 px from it; half the diagonal of a pixel is 0.71 px."""
    return math.hypot(x - 10, y - 10) < 9 - 0.75


@dataclass(frozen=True)
class Area:
    """Where a colour is drawn in a widget: the box around it in the widget's device-independent px
    (`right` and `bottom` are the far edges), and whether every px of the box has the colour."""

    left: float
    right: float
    top: float
    bottom: float
    solid: bool

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> QPoint:
        """The middle of the box, as the whole px a simulated mouse event can aim at."""
        return QPoint(round((self.left + self.right) / 2 - 0.5), round((self.top + self.bottom) / 2 - 0.5))


def drawn(view, color=MARK) -> Area | None:
    """Where `color` (RGB) is drawn in the view's viewport right now, or None if nowhere."""
    QApplication.processEvents()
    image = view.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    rows, columns = np.nonzero((pixels(image) == color).all(axis=2))
    if not rows.size:
        return None
    box = (columns.min(), columns.max() + 1, rows.min(), rows.max() + 1)
    solid = rows.size == (box[1] - box[0]) * (box[3] - box[2])
    return Area(*(float(edge) / ratio for edge in box), solid)


def fit_of(view, size):
    """Where a frame of `size` = (width, height) px lies in the view when all of it shows, by
    geometry: (screen px per video px, left, top), the corner in the viewport's px."""
    (width, height), room = size, view.viewport().size()
    scale = min(room.width() / width, room.height() / height)
    return scale, (room.width() - width * scale) / 2, (room.height() - height * scale) / 2


def wheel(view, at: QPoint, notches: int = 1) -> None:
    """Turn the mouse wheel over the point `at` of the view's viewport (device-independent px):
    `notches` steps away from the user (zoom in), or towards the user when negative."""
    viewport = view.viewport()
    event = QWheelEvent(QPointF(at), QPointF(viewport.mapToGlobal(at)), QPoint(0, 0), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(viewport, event)
    QApplication.processEvents()


def drag(qtbot, view, start: QPoint, end: QPoint, button=Qt.MouseButton.LeftButton, release: bool = True) -> None:
    """Press `button` at `start` of the view's viewport, move to `end` in five steps, and release
    there (or keep the button down with `release` false). Device-independent px.

    The steps are 20 ms apart, as a hand's are: pyqtgraph passes on at most 100 mouse moves a
    second and drops a move that follows another one sooner."""
    viewport = view.viewport()
    qtbot.mousePress(viewport, button, Qt.KeyboardModifier.NoModifier, start)
    for step in range(1, 6):
        QTest.qWait(20)
        qtbot.mouseMove(viewport, start + (end - start) * step / 5)
        QApplication.processEvents()
    if release:
        qtbot.mouseRelease(viewport, button, Qt.KeyboardModifier.NoModifier, end)
        QApplication.processEvents()
