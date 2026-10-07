"""The read-outs at the right of the status bar (SPEC 3.1, 3.2, 4.6, 10.1): where the cursor is on
the picture, the gray value under it, and the device the model runs on.

From left to right: "Pixel 1024.5, 512.5 px", "Position 12.34, −5.67 mm", "Gray 128" and
"Device cpu"; the state of the model follows them, in the label that panel 6 adds. The last
message is the status bar's own, at the left (`showMessage`), and stays until the next one.

- Pixel: (u, v) in px of the video frame (SPEC 3.1: u to the right, v down, the middle of pixel
  (c, r) at (c + 0.5, r + 0.5)), as a click at that place would report it.
- Position: that point in mm in the session's axes, y up (`Session.world_frame`, SPEC 3.2); without
  a scale it says so.
- Gray: 0.299 R + 0.587 G + 0.114 B of the pixel under the cursor on the frame shown, on the 0 to
  255 scale (the gray of the brightness probes, SPEC 4.6).
While the cursor is not over the picture the three are empty. Each label has room for its longest
text and digits of one width, so nothing moves when a number changes.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLabel

from outline_tracker.gui.navigation import MINUS, tabular_font
from outline_tracker.probes import GRAY_WEIGHTS

NO_SCALE = "Position – (no scale)"
NO_DEVICE = "–"
ROOM = 4  # px that a label is wider than its longest text, so that no text touches the label's edge
# The longest text of each read-out: a frame up to 9999 px wide, 9999.99 mm from the origin.
LONGEST = {"pixel": "Pixel 0000.0, 0000.0 px", "position": f"Position {MINUS}0000.00, {MINUS}0000.00 mm",
           "gray": "Gray 000", "device": "Device cuda"}
TIPS = {"pixel": "Where the cursor is on the picture, in px of the video frame: across, down",
        "position": "Where the cursor is in your axes, in mm: x, y (y up)",
        "gray": "The gray value of the pixel under the cursor, from 0 (black) to 255 (white)",
        "device": "What the model runs on: cpu, mps (Apple GPU) or cuda (NVIDIA GPU)"}


def signed(value: float, decimals: int) -> str:
    """A number with `decimals` decimals and the minus sign "−" (U+2212), as the window writes it.
    A value that is written as zero has no sign."""
    text = f"{value:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0 else text.replace("-", MINUS)


def gray_at(image: np.ndarray, u: float, v: float) -> float:
    """The gray value of the pixel of `image` that holds the point (u, v): 0.299 R + 0.587 G +
    0.114 B on the 0 to 255 scale (SPEC 4.6). `image` is an RGB uint8 array [row, column, 3]; (u, v)
    is in px of it (SPEC 3.1), inside the image."""
    return float(np.dot(image[int(v), int(u), :3].astype(float), GRAY_WEIGHTS))


def device_of(worker) -> str | None:
    """The device the loaded model runs on ("cpu", "mps" or "cuda"), as its segmenter says it; None
    while no model is loaded, and for a segmenter that names none (a stand-in)."""
    segmenter = getattr(getattr(worker, "_engine", None), "segmenter", None)  # the worker offers no other way yet
    device = getattr(segmenter, "device", None)
    return None if device is None else str(device)


class Readouts(QObject):
    """The four read-outs of `window` (a `MainWindow`), added to its status bar as permanent
    widgets: `pixel_label`, `position_label`, `gray_label` and `device_label`. They follow the
    view's cursor and frame and the session by themselves; `follow(worker)` lets the device follow
    the model's worker."""

    def __init__(self, window):
        super().__init__(window)
        self._view, self._controller, self._worker = window.view, window.controller, None
        made = {}
        for name, longest in LONGEST.items():
            label = made[name] = QLabel()
            label.setFont(tabular_font(label.font()))
            label.setFixedWidth(label.fontMetrics().horizontalAdvance(longest) + ROOM)
            label.setToolTip(TIPS[name])
            window.statusBar().addPermanentWidget(label)
        self.pixel_label, self.position_label = made["pixel"], made["position"]
        self.gray_label, self.device_label = made["gray"], made["device"]
        self._view.cursor_moved.connect(self.show_cursor)
        self._view.frame_changed.connect(self.show_cursor)
        self._controller.session_changed.connect(self.show_cursor)
        self._controller.video_opened.connect(self.show_cursor)
        self.show_cursor()
        self.show_device()

    def follow(self, worker) -> None:
        """Show the device of the model that `worker` (the window's `Worker`) has loaded, now and
        whenever its state changes or an outline has come (the model may leave the Apple GPU then)."""
        self._worker = worker
        worker.state_changed.connect(self.show_device)
        worker.busy_changed.connect(self.show_device)
        self.show_device()

    def show_cursor(self, *_) -> None:
        """Bring the pixel, the position and the gray value in line with the cursor, the frame
        shown and the session's scale and axes."""
        at, image = self._view.cursor_px, self._view.image_item.image
        if at is None or image is None:
            for label in (self.pixel_label, self.position_label, self.gray_label):
                label.setText("")
            return
        u, v = at
        self.pixel_label.setText(f"Pixel {u:.1f}, {v:.1f} px")
        self.gray_label.setText(f"Gray {gray_at(image, u, v):.0f}")
        session, axes = self._controller.session, None
        if session is not None:
            try:
                axes = session.world_frame()
            except ValueError:  # no scale yet
                pass
        if axes is None:
            self.position_label.setText(NO_SCALE)
        else:
            x, y = axes.to_world(u, v)
            self.position_label.setText(f"Position {signed(float(x), 2)}, {signed(float(y), 2)} mm")

    def show_device(self, *_) -> None:
        """Bring the device in line with the model that is loaded."""
        self.device_label.setText(f"Device {device_of(self._worker) or NO_DEVICE}")
