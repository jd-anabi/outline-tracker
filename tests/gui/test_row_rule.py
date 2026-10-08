"""The rule for the parts of a bar: nothing outside, nothing over another part, no text cut.

`layout_findings` (tests/gui/gui_helpers.py) looks at the visible children of a widget and says what
breaks the rule. Here it is tried on a made-up widget whose children are put at rectangles chosen
by hand.

Expected values come from geometry:
- the rectangles are set in the test, (left, top, width, height) in the parent's px (Qt's
  device-independent px, x to the right and y down);
- a text is as wide as the advance that the child's own font gives it;
- the made-up button has no border and 10 px of padding at each side by its style sheet, so its
  room for text is its width less 20 px.
"""

from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from gui_helpers import layout_findings

SIDE = 10  # px of padding at each side of the made-up button's text


def test_the_rule_names_what_is_outside_what_overlaps_and_what_is_cut(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(300, 60)
    button, label, rest, hidden = QPushButton("Play", parent), QLabel("Frame", parent), QWidget(parent), QWidget(parent)
    button.setStyleSheet(f"border: none; padding: 0 {SIDE}px;")
    hidden.hide()
    play, pause = (button.fontMetrics().horizontalAdvance(text) for text in ("Play", "Pause"))
    frame = label.fontMetrics().horizontalAdvance("Frame")
    assert 0 < play < pause and 0 < frame  # the font has letters, and the longer word is the wider one
    wide = pause + 2 * SIDE  # the button with room for the longer of its two texts
    inside = "(0, 0, 300, 60)"  # the parent's contents rect

    def arrange():
        """Button, label and the rest side by side, each touching the next, from the left edge to the
        right edge; button and label have exactly the room their texts need."""
        parent.setContentsMargins(0, 0, 0, 0)
        button.setGeometry(0, 10, wide, 28)
        label.setGeometry(wide, 10, frame, 28)
        rest.setGeometry(wide + frame, 0, 300 - wide - frame, 60)
        hidden.setGeometry(290, 20, 50, 80)  # past the right and the bottom edge, and over `rest`

    arrange()
    assert layout_findings(parent) == []
    assert layout_findings(parent, {button: "Pause"}) == []  # also when the button's longest text is asked for

    # outside: 1 px past the right edge; and at the edge of a parent that keeps 1 px free at its top
    rest.setGeometry(wide + frame, 0, 301 - wide - frame, 60)
    assert layout_findings(parent) == [
        f"outside: QWidget at ({wide + frame}, 0, {301 - wide - frame}, 60) is not inside {inside}"]
    arrange()
    parent.setContentsMargins(0, 1, 0, 0)
    assert layout_findings(parent) == [
        f"outside: QWidget at ({wide + frame}, 0, {300 - wide - frame}, 60) is not inside (0, 1, 300, 59)"]

    # overlap: the label begins in the button's last column of px
    arrange()
    label.move(wide - 1, 10)
    assert layout_findings(parent) == [
        f"overlap: QPushButton 'Play' at (0, 10, {wide}, 28) and QLabel 'Frame' at ({wide - 1}, 10, {frame}, 28)"]

    # cut, a button: 1 px less than its longest text needs, then 1 px less than the text it shows needs
    arrange()
    button.resize(wide - 1, 28)
    assert layout_findings(parent) == []  # "Play" is the shorter text
    assert layout_findings(parent, {button: "Pause"}) == [
        f"cut: QPushButton 'Play' has {pause - 1} px for 'Pause', which is {pause} px wide"]
    button.resize(play + 2 * SIDE, 28)
    assert layout_findings(parent) == []
    button.resize(play + 2 * SIDE - 1, 28)
    assert layout_findings(parent) == [f"cut: QPushButton 'Play' has {play - 1} px for 'Play', which is {play} px wide"]

    # cut, a label: 1 px narrower than its text
    arrange()
    label.resize(frame - 1, 28)
    assert layout_findings(parent) == [f"cut: QLabel 'Frame' has {frame - 1} px for 'Frame', which is {frame} px wide"]

    # a hidden child is left out: the same child is found once it shows
    arrange()
    assert layout_findings(parent) == []
    hidden.show()
    assert layout_findings(parent) == [
        f"outside: QWidget at (290, 20, 50, 80) is not inside {inside}",
        f"overlap: QWidget at ({wide + frame}, 0, {300 - wide - frame}, 60) and QWidget at (290, 20, 50, 80)"]
