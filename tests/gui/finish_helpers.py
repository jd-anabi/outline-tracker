"""What the tests of task C8a share: the status bar, the menus, playing, the Stopwatch dialog, About
and the run folder's rules.

Imported by name from the test files beside it. Nothing here imports torch.

- `record_every_dialog` replaces the three functions of outline_tracker/gui/dialogs.py by a
  recorder (`gui_helpers.record_dialogs` knows two of them).
- `named` opens a clip and types the student's name: the session then has its run folder.
- `hover` puts the mouse on the middle of a marked pixel. Where the pixel is on the screen is read
  from the drawn view (`gui_helpers.drawn`), never from the view's own numbers.
- `tick` is a tick of the play timer, emitted by the test: no test waits for time to pass.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers
at +0.5); positions in a widget are Qt's device-independent px; mm are in the session's axes, y up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from gui_helpers import MARK, StandInSource, drawn, picture
from session_helpers import NAME, body, type_into

MINUS = "−"                    # U+2212, the sign of a negative number in a read-out
GRAY_OF_MARK = 0.299 * MARK[0] + 0.587 * MARK[1] + 0.114 * MARK[2]  # SPEC 4.6: 105.315 for (255, 0, 255)
SMALL_FRAME = (8, 6)           # (width, height) in px: fitted into the view, one pixel covers about 70 screen px


class Asked:
    """What the window asked the dialogs for, in place of the dialogs: `messages` holds (parent,
    kind, text), `files` holds (parent, title, filters, on_chosen), `folders` holds (parent, title,
    on_chosen)."""

    def __init__(self):
        self.messages, self.files, self.folders = [], [], []

    def message(self, parent, kind, text):
        self.messages.append((parent, kind, text))

    def open_file(self, parent, title, filters, on_chosen):
        self.files.append((parent, title, filters, on_chosen))

    def choose_folder(self, parent, title, on_chosen):
        self.folders.append((parent, title, on_chosen))


def record_every_dialog(monkeypatch) -> Asked:
    """Replace `message`, `open_file` and `choose_folder` of outline_tracker/gui/dialogs.py by a
    recorder, and return it."""
    from outline_tracker.gui import dialogs

    recorder = Asked()
    for name in ("message", "open_file", "choose_folder"):
        monkeypatch.setattr(dialogs, name, getattr(recorder, name))
    return recorder


@pytest.fixture
def never_blocking(monkeypatch):
    """Make `exec`, which would wait for the user and never return here, an error."""
    def blocked(*_):
        raise AssertionError("A dialog was run with exec(): it blocks the window. Use open().")

    for kind in (QDialog, QMessageBox, QFileDialog):
        monkeypatch.setattr(kind, "exec", blocked)


def named(window, qtbot, clip, name: str = NAME) -> Path:
    """Open `clip` (a GroundTruth with its `path`, a copy in the test's own folder) in the window
    and type the student's name. Returns the clip's path."""
    window.open_path(clip.path)
    type_into(qtbot, body(window, 1).name_edit, name)
    return clip.path


def menu_texts(window) -> dict[str, list[str]]:
    """The menu bar as {menu's title: the texts of its items, without the separators}."""
    menus = [entry.menu() for entry in window.menuBar().actions() if entry.menu() is not None]
    return {menu.title(): [item.text() for item in menu.actions() if not item.isSeparator()] for menu in menus}


def small_picture(window, qtbot, clip, column: int, row: int):
    """Open `clip`, show the window, and let the view show a stand-in frame of `SMALL_FRAME` whose
    pixel (column, row) is the marked one. Returns the view."""
    source = StandInSource(SMALL_FRAME, marked=(column, row, column + 1, row + 1))
    return picture(window, qtbot, clip, source)


def move_mouse(view, place: QPoint) -> None:
    """Tell the view's viewport that the mouse moved to `place` (device-independent px of the
    viewport), no button held. The event is handed to the viewport itself, as `gui_helpers.wheel`
    hands over a turn of the wheel: a move simulated through the window system is not reported when
    the mouse was moved to that place last (in the window of the test before), and is misplaced on a
    screen with a scale factor."""
    viewport = view.viewport()
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(place), QPointF(viewport.mapToGlobal(place)),
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(viewport, event)
    QApplication.processEvents()


def hover(view, shift: tuple[int, int] = (0, 0)) -> QPoint:
    """Put the mouse on the middle of the marked pixel of the frame shown, or on the middle of the
    pixel `shift` = (columns, rows) away from it. Returns the place in the viewport's px."""
    seen = drawn(view)
    assert seen is not None and seen.solid and seen.width >= 40  # one video px on 40 screen px or more
    place = seen.center + QPoint(round(shift[0] * seen.width), round(shift[1] * seen.height))
    move_mouse(view, place)
    return place


def number(value: float, decimals: int) -> str:
    """A number as a read-out writes it: fixed decimals, and U+2212 for the minus sign."""
    return f"{value:.{decimals}f}".replace("-", MINUS)


def row_parts(bar) -> list:
    """The parts of the bottom bar's row, from left to right."""
    return [bar.first_button, bar.back_ten_button, bar.back_button, bar.play_button, bar.forward_button,
            bar.forward_ten_button, bar.last_button, bar.frame_box, bar.time_label]


def tick(bar, times: int = 1) -> None:
    """Let the bottom bar's play timer fire `times` times, without waiting for it."""
    for _ in range(times):
        bar.play_timer.timeout.emit()


def is_paused(bar) -> bool:
    """Whether every sign of the bottom bar says that nothing plays."""
    return not bar.playing and not bar.play_timer.isActive() and bar.play_button.text() == "Play"


def is_playing(bar) -> bool:
    """Whether every sign of the bottom bar says that the clip is being played."""
    return bar.playing and bar.play_timer.isActive() and bar.play_button.text() == "Pause"
