"""The Stopwatch dialog of panel 2 (SPEC 4.1): fps_true from a stopwatch that was filmed.

The user finds two frames that show the stopwatch and enters, for each, the frame's number and the
time the stopwatch shows on it. Then fps_true = (frame_b - frame_a) / (time_b - time_a), computed
by `geometry.fps_from_stopwatch`. Numbers that give no frame rate (equal readings, equal frames,
the later frame showing the earlier time) are refused in the dialog, in that function's words.

The dialog is window-modal and is shown with `open()`: the call returns at once, and the window
cannot be used until the dialog is closed. So a frame is looked for before the dialog is opened;
"Use frame shown" then takes its number from the view. The dialog changes nothing itself: on OK it
is accepted, and whoever opened it reads `fps` and `values()`.

Frames are video frame numbers counted from 0; readings are in s; fps_true is in frames per second.
Lengths of the layout are Qt's device-independent px.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QLayout, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from outline_tracker.geometry import fps_from_stopwatch
from outline_tracker.gui.navigation import LARGEST_FRAME, numeric_font
from outline_tracker.gui.panels.video_panel import SPACING, Message

MARGIN = 24               # around the dialog's content
BOX_SIZE = (104, 28)      # the least width and height of a number box
LONGEST_READING_S = 99_999.999  # a wide limit of the boxes only
KEYS = ("frame_a", "time_a_s", "frame_b", "time_b_s")  # of `values()`, and of the session's stopwatch block
HINT = ("Find two frames that show the stopwatch, several seconds apart. Enter each frame's number and the time "
        "the stopwatch shows on it.")
NO_RESULT = "fps_true = –"
RESULT = "fps_true = {fps:.2f} fps"
USE_SHOWN = "Use frame shown"
USE_SHOWN_TIP = "Take the number of the frame that the video view shows"


class StopwatchDialog(QDialog):
    """The dialog. `frame_shown` is the number of the frame the view shows (None when it shows
    none); `start` holds the numbers the boxes open with (`values()` of an earlier time, or the
    session's stopwatch block), None for zeros.

    Parts: `frame_a_box`, `time_a_box`, `use_a_button` for the first moment and `frame_b_box`,
    `time_b_box`, `use_b_button` for the second; `result_label`; `message` (why the numbers give no
    frame rate, shown once OK was asked for); `ok_button`, `cancel_button`. `fps` is fps_true in
    frames per second from what the boxes hold, None when they give none."""

    def __init__(self, parent: QWidget, frame_shown: int | None, start: dict | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Stopwatch")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._asked = False  # OK was asked for: from then on the dialog says why the numbers give no frame rate
        self.fps: float | None = None

        hint = QLabel(HINT)
        hint.setWordWrap(True)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(SPACING)
        boxes = []
        for row, moment in enumerate(("First", "Second")):
            frame_box, time_box, use_button = QSpinBox(), QDoubleSpinBox(), QPushButton(USE_SHOWN)
            frame_box.setRange(0, LARGEST_FRAME)
            frame_box.setToolTip(f"{moment} moment: the number of a frame that shows the stopwatch")
            time_box.setRange(0.0, LONGEST_READING_S)
            time_box.setDecimals(3)
            time_box.setSuffix(" s")
            time_box.setToolTip(f"{moment} moment: the time the stopwatch shows on that frame, in s")
            use_button.setToolTip(USE_SHOWN_TIP)
            use_button.setEnabled(frame_shown is not None)
            use_button.clicked.connect(lambda _=False, box=frame_box: box.setValue(frame_shown))
            for column, part in enumerate((QLabel(f"{moment} frame"), frame_box, use_button,
                                           QLabel("Stopwatch shows"), time_box)):
                grid.addWidget(part, row, column)
            for box in (frame_box, time_box):
                box.setMinimumSize(*BOX_SIZE)
                box.setAlignment(Qt.AlignmentFlag.AlignRight)
                box.valueChanged.connect(self._changed)
            boxes += [frame_box, time_box, use_button]
        (self.frame_a_box, self.time_a_box, self.use_a_button,
         self.frame_b_box, self.time_b_box, self.use_b_button) = boxes

        self.result_label = QLabel(NO_RESULT)
        self.result_label.setFont(numeric_font(self.result_label.font()))
        self.message = Message()
        self.ok_button, self.cancel_button = QPushButton("OK"), QPushButton("Cancel")
        self.ok_button.setToolTip("Use this frame rate as fps_true")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        buttons.addStretch(1)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)
        for button in (self.use_a_button, self.use_b_button, self.ok_button, self.cancel_button):
            button.setAutoDefault(False)  # Enter in a box enters its number, and closes nothing

        rows = QVBoxLayout(self)
        rows.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        rows.setSpacing(SPACING)
        rows.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)  # the dialog grows when the message comes
        rows.addWidget(hint)
        rows.addLayout(grid)
        rows.addWidget(self.result_label)
        rows.addWidget(self.message)
        rows.addLayout(buttons)
        for key, box in zip(KEYS, (self.frame_a_box, self.time_a_box, self.frame_b_box, self.time_b_box)):
            if start is not None and key in start:
                box.setValue(start[key])
        self._changed()

    def values(self) -> dict:
        """What the boxes hold: {"frame_a", "time_a_s", "frame_b", "time_b_s"}, the two video frame
        numbers and the two stopwatch readings in s."""
        return {"frame_a": self.frame_a_box.value(), "time_a_s": self.time_a_box.value(),
                "frame_b": self.frame_b_box.value(), "time_b_s": self.time_b_box.value()}

    def accept(self) -> None:
        """OK: close with the frame rate, or stay open and say why the numbers give none."""
        self._asked = True
        self._changed()
        if self.fps is not None:
            super().accept()

    def _changed(self) -> None:
        """Bring the result and the message in line with the boxes."""
        try:
            self.fps, why = fps_from_stopwatch(**self.values()), ""
        except ValueError as refused:
            self.fps, why = None, str(refused)
        self.result_label.setText(NO_RESULT if self.fps is None else RESULT.format(fps=self.fps))
        self.message.show_text("problem", why if self._asked else "")
