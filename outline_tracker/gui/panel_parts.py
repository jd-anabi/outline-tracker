"""The small parts that the panels of the dock share, so that no panel imports another panel.

Widgets: `Message` (a line of text in a tinted box), `IconMessage` (an icon and one or two sentences on
a tint), `ElidedLabel` (the name of a file or a folder, cut in the middle when it is too long), `NumberBox`
(a number with its unit) and `guard_wheel` (a box that the mouse wheel changes only while it has the
keyboard). Builders: `button`, `group_label`, `value_label`, `field_label`, and the layouts `row`, `column`
and `field_rows`; `follow_tool` ties tool buttons to the video view. Text: `number` and `point_text`. The
sizes that panels have in common (`SPACING`, `CONTROL_HEIGHT`, `LABEL_WIDTH` and the others) are defined
here once. `Message`, `ElidedLabel`, `guard_wheel`, `field_label` and `field_rows` were in
gui/panels/video_panel.py, from where they are still importable; the rest was in
gui/panels/calibration_panel.py.

Lengths of the layout are Qt's device-independent px. `point_text` takes a point in px of the video
frame (SPEC 3.1).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QSpinBox, QStyle, QVBoxLayout, QWidget)

from outline_tracker.gui import theme
from outline_tracker.gui.navigation import numeric_font

LABEL_WIDTH = 120     # the column of the field labels
BOX_WIDTH = 104       # the least width of a number box
SPACING = 8           # between two rows, and between two buttons
CONTROL_HEIGHT = 28   # a field, a box, a button
PRIMARY_HEIGHT = 32   # the button of the next step
GROUP_GAP = 16        # between two groups of one panel
WIDEST = 1e6          # a number box takes any value up to here: what can be used is decided where it is typed
DASH = "–"            # in place of a value there is none of
REDO_TIP = "Remove the points and click them again."
ICONS = {"success": "SP_DialogApplyButton", "warning": "SP_MessageBoxWarning", "problem": "SP_MessageBoxCritical"}


class Message(QLabel):
    """A line of text in a tinted box, under the control it is about: `show_text(kind, text)` with
    kind `problem` or `warning`; an empty text hides it. It wraps, and is never wider than its room.
    (`IconMessage` has an icon, and a third kind, `success`.)"""

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


class IconMessage(QFrame):
    """A message inside a panel: an icon and one or two sentences on a tint. `kind` is `success`,
    `warning` or `problem` (None while nothing is shown); the tint and the icon follow it, and the
    sentence always says in words what the colour says. (`Message` is a line of text without an icon.)"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "msg")
        self.kind: str | None = None
        self._icon, self._text = QLabel(), QLabel()
        self._text.setWordWrap(True)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(SPACING)
        row.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self._text, 1)
        self.hide()

    def text(self) -> str:
        """The sentence shown; empty while nothing is."""
        return self._text.text()

    def show_message(self, kind: str | None, text: str = "", status=None) -> None:
        """Show `text` as a message of `kind`; with kind None, show nothing. A text that was not
        shown before also goes to `status`, the window's status bar."""
        if kind and status is not None and text != self._text.text():
            status.showMessage(text)
        self.kind = kind
        self._text.setText(text if kind else "")
        if kind:
            icon = self.style().standardIcon(getattr(QStyle.StandardPixmap, ICONS[kind]))
            self._icon.setPixmap(icon.pixmap(16, 16))
            theme.set_property(self, "kind", kind)
        self.setVisible(kind is not None)


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


class NumberBox(QDoubleSpinBox):
    """A number with its unit as the suffix (" mm", "°"), right-aligned. A typed number counts when
    it is complete (Enter, or leaving the box). The wheel changes it only while the box has the
    keyboard focus: scrolling the dock must never change a value."""

    def __init__(self, suffix: str, decimals: int, lowest: float, tip: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setRange(lowest, WIDEST)
        self.setDecimals(decimals)
        self.setSuffix(suffix)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setMinimumSize(BOX_WIDTH, CONTROL_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(tip)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def show_value(self, value: float, refused: bool) -> None:
        """Show the session's `value`, without reporting a change. A box whose typed value was
        `refused` keeps what was typed, and is marked as wrong."""
        if not refused:
            self.blockSignals(True)
            self.setValue(value)
            self.blockSignals(False)
        if (self.property("check") or "") != ("error" if refused else ""):
            theme.set_property(self, "check", "error" if refused else "")


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


def button(text: str, tip: str, checkable: bool = False) -> QPushButton:
    """A button of a panel: no key presses it by accident. A tool's button is `checkable`."""
    made = QPushButton(text)
    made.setCheckable(checkable)
    made.setAutoDefault(False)
    made.setToolTip(tip)
    return made


def group_label(text: str) -> QLabel:
    """The DemiBold name of a group of rows in a panel."""
    made = QLabel(text)
    font = made.font()
    font.setWeight(QFont.Weight.DemiBold)
    made.setFont(font)
    return made


def value_label() -> QLabel:
    """A result (DemiBold, digits of one width, right-aligned), a dash until there is one."""
    made = QLabel(DASH)
    font = numeric_font(made.font())
    font.setPointSizeF(made.font().pointSizeF())  # the base size: only the frame box is larger
    made.setFont(font)
    made.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return made


def row(*parts) -> QHBoxLayout:
    """One row of a panel. A text becomes a field label in the label column, None becomes the
    free room, a widget is added as it is; the last widget of a labelled row takes the room."""
    made = QHBoxLayout()
    made.setSpacing(SPACING)
    for index, part in enumerate(parts):
        if part is None:
            made.addStretch(1)
        elif isinstance(part, str):
            label = QLabel(part)
            label.setFixedWidth(LABEL_WIDTH)
            made.addWidget(label)
        else:
            made.addWidget(part, 1 if isinstance(parts[0], str) and index == len(parts) - 1 else 0)
    return made


def column(widget: QWidget, *rows) -> None:
    """Lay `rows` out in `widget`, top to bottom: a layout, a widget, or a number of px of room."""
    made = QVBoxLayout(widget)
    made.setContentsMargins(0, 0, 0, 0)
    made.setSpacing(SPACING)
    for part in rows:
        if isinstance(part, int):
            made.addSpacing(part - SPACING)
        elif isinstance(part, QWidget):
            made.addWidget(part)
        else:
            made.addLayout(part)


def follow_tool(view, pairs) -> None:
    """Tie tool buttons to the view: `pairs` are (button, tool). A click on a button chooses its
    tool, or Pan if it was chosen; a button is checked exactly while its tool is the view's."""
    def show() -> None:
        for made, tool in pairs:
            made.setChecked(view.tool is tool)

    for made, tool in pairs:
        made.clicked.connect(lambda on, tool=tool: view.set_tool(tool if on else None))
    view.tool_changed.connect(show)


def number(value: float, decimals: int) -> str:
    """`value` with `decimals` decimals, a negative one with the minus sign "−" (U+2212), and
    "0.0" rather than "−0.0"."""
    text = f"{value:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0 else text.replace("-", "−")


def point_text(point) -> str:
    """A point (u, v) in px of the video frame, as "160.3, 119.8 px"."""
    return f"{number(point[0], 1)}, {number(point[1], 1)} px"
