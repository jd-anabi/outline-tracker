"""One numbered panel of the dock (SPEC 10.1): number, title, state, hint line, body.

The header row holds the number in a round badge, the title (a button: a click on it opens or
closes this panel, and no other) and the state in a word. The body is under it: first the hint
line, which says what to do next, then the rows a panel adds to `body`. A closed panel shows the
header row only.

A state is `todo`, `done` or `attention`. It is always a word beside the title ("Not started",
"Done", "Needs attention"), never a colour alone; the colours come from gui/theme.py, which selects
the parts by the dynamic properties `role` and `state` set here. Lengths are device-independent px.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from outline_tracker.gui import theme

STATE_WORDS = {"todo": "Not started", "done": "Done", "attention": "Needs attention"}

HEADER_HEIGHT = 32   # the header row
BADGE_SIZE = 20      # the round number badge
PADDING = 12         # around the body: left, right, bottom; and left and right of the header row
SPACING = 8          # between two rows of the body, and between the parts of the header row


class Panel(QFrame):
    """A panel of the dock: `number` (1 to 9) and `title` in the header, `hint` as the first row of the
    body, state `todo`, closed.

    `todo_word` is the state word while nothing is done ("Not started"; "Optional" for a panel that
    may be left out). Parts, for the panel's own task and for tests: `badge`, `header` (the title
    button), `state_label`, `header_row`, `hint`, and `body`, the layout that takes the panel's rows
    under the hint. `state` is the current state.
    """

    def __init__(self, number: int, title: str, hint: str = "", todo_word: str = STATE_WORDS["todo"],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.number, self.title, self.state = number, title, "todo"
        self._words = {**STATE_WORDS, "todo": todo_word}
        self.setProperty("role", "panel")

        self.badge = QLabel(str(number))
        self.badge.setProperty("role", "badge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(BADGE_SIZE, BADGE_SIZE)

        self.header = QToolButton()
        self.header.setProperty("role", "panelHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Ignored: the title takes the width that is left, and is cut when the dock is too narrow for
        # it. Otherwise a long title would widen the whole column beyond what the dock shows.
        self.header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        font = self.header.font()  # the system's font, one point larger and DemiBold
        font.setPointSizeF(font.pointSizeF() + 1)
        font.setWeight(QFont.Weight.DemiBold)
        self.header.setFont(font)

        self.state_label = QLabel()
        self.state_label.setProperty("role", "state")

        self.header_row = QWidget()
        self.header_row.setFixedHeight(HEADER_HEIGHT)
        row = QHBoxLayout(self.header_row)
        row.setContentsMargins(PADDING, 0, PADDING, 0)
        row.setSpacing(SPACING)
        row.addWidget(self.badge)
        row.addWidget(self.header, 1)
        row.addWidget(self.state_label)

        self.hint = QLabel(hint)
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)

        self._body_widget = QWidget()
        self.body = QVBoxLayout(self._body_widget)
        self.body.setContentsMargins(PADDING, 0, PADDING, PADDING)
        self.body.setSpacing(SPACING)
        self.body.addWidget(self.hint)

        whole = QVBoxLayout(self)
        whole.setContentsMargins(0, 0, 0, 0)
        whole.setSpacing(0)
        whole.addWidget(self.header_row)
        whole.addWidget(self._body_widget)

        self.header.toggled.connect(self.set_expanded)
        self.set_state("todo")
        self.set_expanded(False)

    def set_state(self, state: str) -> None:
        """Show the state `todo`, `done` or `attention`: the word beside the title, and the `state`
        property that the style sheet colours the badge, the word and the panel's edge by.
        Any other value is a `ValueError` and changes nothing."""
        if state not in STATE_WORDS:
            raise ValueError(f"A panel's state is one of {', '.join(STATE_WORDS)}, not {state!r}.")
        self.state = state
        self.state_label.setText(self._words[state])
        for part in (self, self.badge, self.state_label):
            theme.set_property(part, "state", state)

    def set_hint(self, text: str) -> None:
        """Replace the hint line: one or two short sentences that say what to do next, or why the
        panel's main button cannot be used."""
        self.hint.setText(text)

    def is_expanded(self) -> bool:
        """True while the body is open; False while the panel is its header row only."""
        return not self._body_widget.isHidden()

    def set_expanded(self, expanded: bool) -> None:
        """Open the body (`expanded` true) or close the panel to its header row. The arrow beside the
        title points down while the panel is open, to the right while it is closed."""
        self._body_widget.setVisible(expanded)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.header.setChecked(expanded)
