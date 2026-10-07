"""The bottom bar (SPEC 3.3, 3.4, 10.1, 10.2): moving through the clip on its frame grid.

Top to bottom: the slider, room for the strip of flags (a later task draws in it), and a row with
First, -10, -1, +1, +10 and Last, the frame box and the time t. Everything moves on the grid the
bar was given (`set_grid`): the slider has one position per grid frame, a button or a key moves a
number of grid steps, and a frame typed into the box goes to the nearest grid frame. The bar shows
no picture: it emits `frame_requested`, and whoever shows the frame answers.

Frames are video frame numbers, counted from 0. t = frame / fps_true in s (SPEC 3.3); fps_true is
in frames per second. Lengths are Qt's device-independent px.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget

HEIGHT = 76           # 8 + slider 20 + 2 + flag strip 6 + 4 + button row 28 + 8
PADDING = 8           # around the bar's content, and between the parts at the right of the row
SLIDER_HEIGHT = 20
FLAG_STRIP_HEIGHT = 6
BUTTON_HEIGHT = 28
BUTTON_GAP = 4
BUTTON_PADDING = 6    # left and right of a step button's text, inside its 1 px edge
FRAME_BOX_WIDTH = 96
LARGEST_FRAME = 9_999_999  # the box takes any number up to here; where it goes is decided by the grid
NOT_KNOWN = "t = –"   # while fps_true is not known
MINUS = "−"

# Text, tooltip, key, and grid steps (None: to the first or the last frame) of the six buttons.
STEPS = (
    ("First", "First frame of the clip (Home)", Qt.Key.Key_Home, None),
    (f"{MINUS}10", "Back 10 steps (Shift+←)", Qt.Modifier.SHIFT | Qt.Key.Key_Left, -10),
    (f"{MINUS}1", "Back 1 step (←)", Qt.Key.Key_Left, -1),
    ("+1", "Forward 1 step (→)", Qt.Key.Key_Right, 1),
    ("+10", "Forward 10 steps (Shift+→)", Qt.Modifier.SHIFT | Qt.Key.Key_Right, 10),
    ("Last", "Last frame of the clip (End)", Qt.Key.Key_End, None),
)


def numeric_font(font: QFont) -> QFont:
    """A copy of `font` one point larger and DemiBold, with digits that all have the same width
    (the OpenType feature `tnum`), so that a number that changes while it is watched does not make
    its neighbours jump. For the frame box and the time."""
    made = QFont(font)
    made.setPointSizeF(font.pointSizeF() + 1)
    made.setWeight(QFont.Weight.DemiBold)
    made.setFeature(QFont.Tag("tnum"), 1)
    return made


class NavigationBar(QFrame):
    """The bottom bar. It cannot be used until it has a grid (`set_grid`).

    `frame_requested(frame)` is emitted when the user asks for another frame with the slider, a
    button, a key or the frame box; the frame is always on the grid, and the bar shows it already.
    `frame` is the frame the bar is on (None without a grid). Parts: `slider`, `flag_strip`,
    `first_button`, `back_ten_button`, `back_button`, `forward_button`, `forward_ten_button`,
    `last_button`, `frame_box`, `time_label`.

    The keys work in the whole window the bar is in: left and right arrow for one step, with Shift
    for ten, Home and End. A field that is being typed in keeps them for its text.
    """

    frame_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("BottomBar")
        self.setFixedHeight(HEIGHT)
        self._grid: list[int] = []
        self._index = 0
        self._fps: float | None = None

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedHeight(SLIDER_HEIGHT)
        self.slider.setPageStep(10)
        self.slider.setToolTip("Move through the frames of the clip")
        self.slider.valueChanged.connect(self._go)
        self.flag_strip = QWidget()
        self.flag_strip.setFixedHeight(FLAG_STRIP_HEIGHT)

        row = QHBoxLayout()
        row.setSpacing(BUTTON_GAP)
        buttons = []
        for text, tip, key, steps in STEPS:
            button = QPushButton(text)
            button.setToolTip(tip)
            # as wide as its text needs, and no wider: the row must fit beside the dock in the smallest
            # window (960 px wide, 400 of them the dock's), also with a wide system font
            button.setFixedSize(button.fontMetrics().horizontalAdvance(text) + 2 * (BUTTON_PADDING + 1), BUTTON_HEIGHT)
            move = (lambda *_, steps=steps: self.step(steps)) if steps is not None else (
                self.first if key == Qt.Key.Key_Home else self.last)
            button.clicked.connect(move)
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(move)
            row.addWidget(button)
            buttons.append(button)
        (self.first_button, self.back_ten_button, self.back_button,
         self.forward_button, self.forward_ten_button, self.last_button) = buttons

        self.frame_box = QSpinBox()
        self.frame_box.setRange(0, LARGEST_FRAME)
        self.frame_box.setKeyboardTracking(False)  # a typed number counts when it is complete (Enter, or leaving)
        self.frame_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.frame_box.setFixedSize(FRAME_BOX_WIDTH, BUTTON_HEIGHT)
        self.frame_box.setToolTip("Go to a frame. A frame that is not on the frame grid goes to the nearest one.")
        self.frame_box.setFont(numeric_font(self.frame_box.font()))
        self.frame_box.valueChanged.connect(self._typed)
        frame_word = QLabel("Frame")
        frame_word.setBuddy(self.frame_box)
        self.time_label = QLabel(NOT_KNOWN)
        self.time_label.setToolTip("Time of this frame: frame / fps_true")
        self.time_label.setFont(numeric_font(self.time_label.font()))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # room for 999.999 s from the start, so that nothing moves when the digits change
        self.time_label.setMinimumWidth(self.time_label.fontMetrics().horizontalAdvance("t = 000.000 s"))
        row.addStretch(1)  # a later task shows the flags of the frame here
        row.addWidget(frame_word)
        row.addSpacing(PADDING - BUTTON_GAP)
        row.addWidget(self.frame_box)
        row.addSpacing(PADDING - BUTTON_GAP)
        row.addWidget(self.time_label)

        rows = QVBoxLayout(self)
        rows.setContentsMargins(PADDING, PADDING, PADDING, PADDING)
        rows.setSpacing(0)
        rows.addWidget(self.slider)
        rows.addSpacing(2)
        rows.addWidget(self.flag_strip)
        rows.addSpacing(4)
        rows.addLayout(row)
        self.setEnabled(False)

    @property
    def frame(self) -> int | None:
        """The video frame number the bar is on: always a frame of the grid. None without a grid."""
        return self._grid[self._index] if self._grid else None

    def set_grid(self, frames) -> None:
        """Take the frame grid of the clip: `frames` are its video frame numbers in rising order
        (`geometry.grid_frames`). The bar stays on its frame if the new grid has it, else it goes to
        the nearest grid frame (the first one for a bar that had no grid). Nothing is requested:
        the caller shows `frame`. An empty grid switches the bar off."""
        before = self.frame
        self._grid = [int(frame) for frame in frames]
        self._index = 0 if before is None or not self._grid else self._nearest(before)
        self.setEnabled(bool(self._grid))
        self._show()

    def set_frame(self, frame: int) -> None:
        """Put the bar on video frame `frame` (on the grid frame nearest to it) without requesting
        it: for the caller that has just shown this frame."""
        if self._grid:
            self._index = self._nearest(frame)
            self._show()

    def set_fps(self, fps_true: float | None) -> None:
        """Take the true frame rate in frames per second, for the time t = frame / fps_true; None
        while it is not known (the time is then a dash)."""
        self._fps = fps_true
        self._show()

    def step(self, steps: int) -> None:
        """Go `steps` grid steps forward (back when negative), stopping at the clip's ends."""
        self._go(self._index + steps)

    def first(self) -> None:
        """Go to the first frame of the grid."""
        self._go(0)

    def last(self) -> None:
        """Go to the last frame of the grid."""
        self._go(len(self._grid) - 1)

    def _nearest(self, frame: int) -> int:
        """Position in the grid of the frame nearest to video frame `frame`; of two that are equally
        near, the later one (decision X20: a frame typed off the grid means "from here on")."""
        return min(range(len(self._grid)), key=lambda index: (abs(self._grid[index] - frame), -index))

    def _typed(self, frame: int) -> None:
        if self._grid:
            self._go(self._nearest(frame))
            self._show()  # the box shows the grid frame, not what was typed

    def _go(self, index: int) -> None:
        """Move to the grid position `index` (brought inside the grid), and request its frame if
        that is another frame than before."""
        if not self._grid:
            return
        index = min(max(index, 0), len(self._grid) - 1)
        if index != self._index:
            self._index = index
            self._show()
            self.frame_requested.emit(self._grid[index])

    def _show(self) -> None:
        """Bring the slider, the frame box and the time in line with the grid and the frame, without
        any of them reporting a change."""
        frame = self.frame
        for part in (self.slider, self.frame_box):
            part.blockSignals(True)
        self.slider.setRange(0, max(len(self._grid) - 1, 0))
        self.slider.setValue(self._index)
        self.frame_box.setSingleStep(self._grid[1] - self._grid[0] if len(self._grid) > 1 else 1)
        self.frame_box.setValue(frame or 0)
        for part in (self.slider, self.frame_box):
            part.blockSignals(False)
        known = frame is not None and self._fps
        self.time_label.setText(f"t = {frame / self._fps:.3f} s" if known else NOT_KNOWN)
