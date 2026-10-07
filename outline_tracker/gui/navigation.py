"""The bottom bar (SPEC 3.3, 3.4, 10.1, 10.2): moving through the clip on its frame grid.

Top to bottom: the slider, room for the strip of flags (a later task draws in it), and a row with
First, -10, -1, Play, +1, +10 and Last, the frame box and the time t. Everything moves on the grid
the bar was given (`set_grid`): the slider has one position per grid frame, a button or a key moves
a number of grid steps, and a frame typed into the box goes to the nearest grid frame. The bar
shows no picture: it emits `frame_requested`, and whoever shows the frame answers.

Playing (Space, or the Play button) goes one grid step forward at each tick of a timer, 30 ticks a
second, from the frame the bar is on. It stops at the last grid frame, on Space or the button,
when a frame is asked for in any other way, when the grid is set again (the session changed), when
a text field of the bar's window or of a dialog over it gets the keyboard, and when `pause` is
called (the window does so when a tool is chosen, panel 2 when it opens the Stopwatch dialog).

Frames are video frame numbers, counted from 0. t = frame / fps_true in s (SPEC 3.3); fps_true is
in frames per second. Lengths are Qt's device-independent px.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSlider, QSpinBox, QTextEdit, QVBoxLayout, QWidget)

HEIGHT = 76           # 8 + slider 20 + 2 + flag strip 6 + 4 + button row 28 + 8
PADDING = 8           # around the bar's content, and between the parts at the right of the row
SLIDER_HEIGHT = 20
FLAG_STRIP_HEIGHT = 6
BUTTON_HEIGHT = 28
BUTTON_GAP = 4
BUTTON_PADDING = 6    # left and right of a button's text, inside its 1 px edge, where the row has room
LEAST_PADDING = 1     # what a button keeps of it in a row that is too narrow
PLAY_TICKS_PER_S = 30  # while playing: grid steps per s
PLAY_TEXTS = ("Play", "Pause")  # the Play button's text while paused, while playing
PLAY_TIP = "Play or pause (Space)"
TEXT_FIELDS = (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit)  # what is typed in: playing stops there
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


def tabular_font(font: QFont) -> QFont:
    """A copy of `font` with digits that all have the same width (the OpenType feature `tnum`), so
    that a number that changes while it is watched does not make its neighbours jump. For the
    status bar's read-outs."""
    made = QFont(font)
    made.setFeature(QFont.Tag("tnum"), 1)
    return made


def numeric_font(font: QFont) -> QFont:
    """A copy of `font` one point larger and DemiBold, with the digits of `tabular_font`. For the
    frame box and the time."""
    made = tabular_font(font)
    made.setPointSizeF(font.pointSizeF() + 1)
    made.setWeight(QFont.Weight.DemiBold)
    return made


class RowButton(QPushButton):
    """A button of the bar's row: `BUTTON_HEIGHT` high, and as wide as `widest` (the longest text it
    will show; its text when not given) with `BUTTON_PADDING` px at each side inside the 1 px edge.
    In a row that is too narrow for that it gives up padding, down to `LEAST_PADDING`, so that its
    text stays whole and no part of the row is pushed out of the bar. Device-independent px."""

    def __init__(self, text: str, widest: str | None = None):
        super().__init__(text)
        advance = self.fontMetrics().horizontalAdvance(text if widest is None else widest)
        self.full_width = advance + 2 * (BUTTON_PADDING + 1)
        self.setFixedHeight(BUTTON_HEIGHT)
        self.setMinimumWidth(advance + 2 * (LEAST_PADDING + 1))
        self.setMaximumWidth(self.full_width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # the width alone gives the room beside the text: the style sheet's padding would cut a text
        # in a button that gave some up
        self.setStyleSheet("padding: 0;")

    def sizeHint(self) -> QSize:
        return QSize(self.full_width, BUTTON_HEIGHT)


class NavigationBar(QFrame):
    """The bottom bar. It cannot be used until it has a grid (`set_grid`).

    `frame_requested(frame)` is emitted when the user asks for another frame with the slider, a
    button, a key or the frame box, and at each tick while playing; the frame is always on the
    grid, and the bar shows it already. `frame` is the frame the bar is on (None without a grid).
    `playing` says whether the clip is being played; `play_timer` gives the ticks. Parts: `slider`,
    `flag_strip`, `first_button`, `back_ten_button`, `back_button`, `play_button`,
    `forward_button`, `forward_ten_button`, `last_button`, `frame_box`, `time_label`.

    The keys work in the whole window the bar is in: left and right arrow for one step, with Shift
    for ten, Home and End, Space to play and pause. A field that is being typed in keeps them for
    its text.
    """

    frame_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("BottomBar")
        self.setFixedHeight(HEIGHT)
        self._grid: list[int] = []
        self._index = 0
        self._fps: float | None = None
        self._playing = False

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
            # as wide as its text needs, and no wider: the row must fit beside the dock in the smallest
            # window (960 px wide, 400 of them the dock's), also with a wide system font
            button = RowButton(text)
            button.setToolTip(tip)
            move = (lambda *_, steps=steps: self.step(steps)) if steps is not None else (
                self.first if key == Qt.Key.Key_Home else self.last)
            button.clicked.connect(move)
            self._window_key(key, move)
            buttons.append(button)
        (self.first_button, self.back_ten_button, self.back_button,
         self.forward_button, self.forward_ten_button, self.last_button) = buttons
        self.play_button = RowButton(PLAY_TEXTS[0], widest=max(PLAY_TEXTS, key=len))
        self.play_button.setToolTip(PLAY_TIP)
        self.play_button.clicked.connect(self.toggle_play)
        self._window_key(Qt.Key.Key_Space, self.toggle_play)
        buttons.insert(3, self.play_button)  # between the steps back and the steps forward
        self._row_buttons = buttons
        for button in buttons:
            row.addWidget(button)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(round(1000 / PLAY_TICKS_PER_S))
        self.play_timer.timeout.connect(self.tick)
        QApplication.instance().focusChanged.connect(self._focus_changed)

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

    def _window_key(self, key, action) -> None:
        """Let `key` call `action` wherever the focus is in the bar's window, while the bar is on."""
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(action)

    def minimumSizeHint(self) -> QSize:
        """The room the bar asks for: its row with every button at its full width. So the dock gives
        up width before a button gives up padding; the buttons do when the bar gets less than this."""
        least = self.layout().minimumSize()
        given_up = sum(button.full_width - button.minimumWidth() for button in self._row_buttons)
        return QSize(least.width() + given_up, least.height())

    @property
    def frame(self) -> int | None:
        """The video frame number the bar is on: always a frame of the grid. None without a grid."""
        return self._grid[self._index] if self._grid else None

    # ------------------------------------------------------------------ playing

    @property
    def playing(self) -> bool:
        """Whether the clip is being played: one grid step forward at each tick of `play_timer`."""
        return self._playing

    def play(self) -> None:
        """Play from the frame the bar is on, one grid step per tick, `PLAY_TICKS_PER_S` ticks per
        s. Nothing plays without a grid, or on the last frame of the grid."""
        if self._grid and self._index < len(self._grid) - 1:
            self._set_playing(True)

    def pause(self) -> None:
        """Stop playing; the bar stays on its frame. Nothing happens when nothing plays."""
        self._set_playing(False)

    def toggle_play(self) -> None:
        """Play, or pause when the clip is being played: the Play button, and Space."""
        if self._playing:
            self.pause()
        else:
            self.play()

    def tick(self) -> None:
        """One tick while playing: request the next frame of the grid. Playing stops when that is
        the last one, and when the frame could not be shown (whoever shows the frames has put the
        bar back with `set_frame`)."""
        if not self._playing:
            return
        wanted = self._index + 1
        self._move(wanted)
        if self._index != wanted or wanted >= len(self._grid) - 1:
            self.pause()

    def _set_playing(self, playing: bool) -> None:
        if playing == self._playing:
            return
        self._playing = playing
        if playing:
            self.play_timer.start()
        else:
            self.play_timer.stop()
        self.play_button.setText(PLAY_TEXTS[playing])

    def _focus_changed(self, _before, now) -> None:
        if self._playing and isinstance(now, TEXT_FIELDS) and self._is_over_my_window(now):
            self.pause()  # what is typed there must not be followed by a moving picture

    def _is_over_my_window(self, widget: QWidget) -> bool:
        """Whether `widget` is in the bar's window, or in a dialog over it: a window whose parent is
        in the bar's window, or in such a dialog. A field of any other window is not this bar's."""
        mine = self.window()
        while widget is not None:
            if widget.window() is mine:
                return True
            widget = widget.window().parentWidget()
        return False

    # ------------------------------------------------------------------ the grid and the frame

    def set_grid(self, frames) -> None:
        """Take the frame grid of the clip: `frames` are its video frame numbers in rising order
        (`geometry.grid_frames`). The bar stays on its frame if the new grid has it, else it goes to
        the nearest grid frame (the first one for a bar that had no grid). Nothing is requested:
        the caller shows `frame`. An empty grid switches the bar off. Playing stops."""
        self.pause()
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
        """The user asks for the grid position `index` (slider, button, key, frame box): playing
        stops, and the bar moves there."""
        self.pause()
        self._move(index)

    def _move(self, index: int) -> None:
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
