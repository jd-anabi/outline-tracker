"""The small parts that the panels of the dock share: `Message` (a line of text in a tinted box),
`ElidedLabel` (the name of a file or a folder, cut in the middle when it is too long), `guard_wheel`
(a box that the mouse wheel changes only while it has the keyboard), and the rows of a panel with
their column of labels (`field_label`, `field_rows`). They were in gui/panels/video_panel.py, from
where every name is still importable.

No quantities are computed here. Lengths of the layout are Qt's device-independent px.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QLineEdit, QSizePolicy, QSpinBox, QWidget

from outline_tracker.gui import theme

LABEL_WIDTH = 120     # the column of the field labels
SPACING = 8           # between two rows, and between two buttons
CONTROL_HEIGHT = 28   # a field, a box


class Message(QLabel):
    """A line of text in a tinted box, under the control it is about: `show_text(kind, text)` with
    kind `problem` or `warning`; an empty text hides it. It wraps, and is never wider than its room."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "msg")
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hide()

    def show_text(self, kind: str, text: str) -> None:
        self.setText(text)
        theme.set_property(self, "kind", kind)
        self.setVisible(bool(text))


class ElidedLabel(QLabel):
    """A label for the name of a file or a folder: `full_text` is the name, and what shows is cut
    in the middle when the label is too narrow for it. The tooltip holds the name, or what
    `set_full_text` was given as `tip` (the whole path)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str, tip: str | None = None) -> None:
        self.full_text = text
        self.setToolTip(text if tip is None else tip)
        self._cut()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._cut()

    def _cut(self) -> None:
        self.setText(self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideMiddle, self.width()))


class _WheelGuard(QObject):
    """Passes the mouse wheel on to what is behind a box while the box does not have the keyboard."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            event.ignore()  # the dock scrolls instead
            return True
        return False


def guard_wheel(box: QWidget) -> None:
    """Make `box` (a spin box, a combo box) react to the mouse wheel only while it has the keyboard:
    the dock scrolls with the wheel, and that must never change a value."""
    box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the wheel alone does not give it the keyboard
    box.installEventFilter(_WheelGuard(box))


def field_label(text: str) -> QLabel:
    """The label of a field, for the left column of a panel's rows."""
    label = QLabel(text)
    label.setFixedWidth(LABEL_WIDTH)
    return label


def field_rows(fields) -> QGridLayout:
    """The rows of a panel: each of `fields` is (label, widget), the label a text or a label made
    with `field_label`, in the left column, the widget filling the rest; or (None, widget) for a
    widget over both columns. A field that is typed in is the buddy of its label."""
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(SPACING)
    grid.setColumnStretch(1, 1)
    for row, (word, part) in enumerate(fields):
        if word is None:
            grid.addWidget(part, row, 0, 1, 2)
            continue
        label = field_label(word) if isinstance(word, str) else word
        grid.addWidget(label, row, 0)
        grid.addWidget(part, row, 1)
        if isinstance(part, (QLineEdit, QSpinBox)):
            label.setBuddy(part)
            part.setMinimumHeight(CONTROL_HEIGHT)
    return grid
