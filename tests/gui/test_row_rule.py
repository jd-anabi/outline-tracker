"""The rule for the parts of a bar, and the bottom bar held to it in the smallest window (SPEC 10.1).

The rule: nothing outside, nothing over another part, no text cut. `layout_findings`
(tests/gui/gui_helpers.py) looks at the visible children of a widget and says what breaks it. It is
tried here on a made-up widget, and then the window's bottom bar is held to it at 960 x 600 px, in
the system's own font and in two larger ones. What may give way there is the dock, and after it the
padding of the row's buttons; no text may.

Expected values come from geometry and from the design note (docs/design/gui_design.md):
- for the made-up widget the rectangles are set in the test, (left, top, width, height) in the
  parent's px; its button has no border and 10 px of padding at each side by its style sheet, so
  its room for text is its width less 20 px;
- a text is as wide as the advance that the part's own font gives it;
- the smallest window is 960 px wide; the dock is 400 px wide where there is room and never under
  340; the bar's padding is 8 px; the frame box is 96 px wide;
- a button of the row is as wide as its longest text with 6 px at each side inside its 1 px edge;
  in a row that is too narrow it keeps 1 px of the 6; the Play button's longest text is "Pause";
- the time has room for "t = 000.000 s", and the frame box for a frame number of six digits.
Widths that were measured in one font on one computer are in no assertion.
"""

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontInfo, QFontMetrics
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStyle, QStyleOptionFrame, QWidget

from finish_helpers import row_parts
from gui_helpers import StandInSource, layout_findings, show, visible_children, with_larger_font
from outline_tracker.geometry import grid_frames

LEFT = Qt.MouseButton.LeftButton
SIDE = 10  # px of padding at each side of the made-up button's text

# The fonts in which the bottom bar is held to the rule: the application's font at so many times its
# point size (`larger_font`, tests/gui/conftest.py). Text is then wider by that much or somewhat less,
# and taller. A larger font takes its room from the dock first and then from the padding of the
# buttons; the two larger steps are there so that the rule is asked in both cases.
FONTS = [pytest.param(1, id="system_font"), pytest.param(1.15, id="15_percent_larger")]

# Every clause is asked in every font on every system; no case is skipped. Two cases of the largest
# step break the rule as the window is built today. Each is marked as failing, strictly: where a marked
# case passes after all, the run fails, so a mark is proved wherever the tests run. Each is an open
# question for the owner, and its mark goes with the answer.
# - The row. The largest step was chosen for the macOS system font, where it passes. In a font as wide
#   as the Linux test machine's (there the dock gives way in the system's own font already,
#   docs/PLAN.md) the row at 60 % larger needs more than a window of 960 px has, also with the least
#   padding. That was seen on a Mac with Verdana as the application's font, not yet on Linux. For
#   Windows nothing is known either way, so the case is asked there like any other.
# - The frame box. At 60 % larger six digits need more room than the 96 px box has: in the macOS
#   system font, and in every other family that was tried on a Mac.
ROW_FONTS = [*FONTS, pytest.param(1.6, id="60_percent_larger", marks=pytest.mark.xfail(
    sys.platform == "linux", strict=True, raises=AssertionError,
    reason="60 % larger in the wide font of the Linux test machine: the row needs more than 960 px has, also "
           "with the least padding. Open: is this step asked in such a font, or is the step chosen by the font?"))]
BOX_FONTS = [*FONTS, pytest.param(1.6, id="60_percent_larger", marks=pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="60 % larger: six digits need more room than the 96 px frame box has. Open: is a frame number of "
           "six digits asked at this size (then the box changes), or up to 15 % larger only?"))]


# ---------------------------------------------------------------------------------------------
# The rule, on a made-up widget


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

    # cut, a label: 1 px narrower than its text; a label's margin is at each side and is no room for text
    arrange()
    label.resize(frame - 1, 28)
    assert layout_findings(parent) == [f"cut: QLabel 'Frame' has {frame - 1} px for 'Frame', which is {frame} px wide"]
    label.setMargin(3)
    label.resize(frame + 2 * 3, 28)
    rest.setGeometry(wide + frame + 2 * 3, 0, 300 - wide - frame - 2 * 3, 60)  # out of the wider label's way
    assert layout_findings(parent) == []
    label.resize(frame + 2 * 3 - 1, 28)
    assert layout_findings(parent) == [f"cut: QLabel 'Frame' has {frame - 1} px for 'Frame', which is {frame} px wide"]
    label.setMargin(0)

    # a hidden child is left out: the same child is found once it shows
    arrange()
    assert layout_findings(parent) == []
    hidden.show()
    assert layout_findings(parent) == [
        f"outside: QWidget at (290, 20, 50, 80) is not inside {inside}",
        f"overlap: QWidget at ({wide + frame}, 0, {300 - wide - frame}, 60) and QWidget at (290, 20, 50, 80)"]


# ---------------------------------------------------------------------------------------------
# A larger font


def test_a_larger_font_is_the_applications_until_the_fixture_ends(qapp):
    own = qapp.font()
    first = QFontMetrics(own).horizontalAdvance("First")
    fixture = with_larger_font(qapp, 1.6)
    try:
        next(fixture)
        assert qapp.font().family() == own.family()
        assert qapp.font().pointSizeF() == pytest.approx(1.6 * own.pointSizeF())
        button = QPushButton("First")  # a widget that is made now has the font, and its text is wider
        assert button.font().pointSizeF() == pytest.approx(1.6 * own.pointSizeF())
        assert button.fontMetrics().horizontalAdvance("First") > first
        assert next(fixture, "over") == "over"
        assert qapp.font() == own
    finally:
        qapp.setFont(own)


# ---------------------------------------------------------------------------------------------
# The bottom bar in the smallest window


def smallest(window, qtbot, grid):
    """Give the window's bottom bar the frame grid `grid` (and the view a stand-in with 200 frames to
    show), make the window as small as it gets, and show it. Returns the bar."""
    window.view.set_source(StandInSource(n_frames=200))
    window.navigation.set_grid(grid)
    window.resize(window.minimumSize())
    show(window, qtbot)
    return window.navigation


def measured(window) -> str:
    """For the message of a failed assertion: the font, the widths, and where each part of the bar lies."""
    bar, asked = window.navigation, QApplication.font()
    used = QFontInfo(asked)
    lines = [f"font {used.family()!r} at {used.pixelSize()} px (asked for {asked.family()!r}, "
             f"{asked.pointSizeF():g} pt), style {QApplication.style().name()!r}, {sys.platform}",
             f"window {window.width()} px wide, dock {window.dock.width()}, bar {bar.width()}; "
             f"with all its padding the row asks for {bar.minimumSizeHint().width()}"]
    for part in visible_children(bar):
        text = part.text() if hasattr(part, "text") else ""
        wide = f", its text is {part.fontMetrics().horizontalAdvance(text)} px wide" if text else ""
        lines.append(f"{type(part).__name__} {text!r} at {part.geometry().getRect()}{wide}")
    return "\n".join(lines)


def hold_to_the_rule(window) -> None:
    """Assert the rule for the bottom bar of the shown window, whatever the Play button reads."""
    QApplication.processEvents()
    bar, dock, seen = window.navigation, window.dock, measured(window)
    assert window.width() == 960, seen  # no part made the smallest window wider

    parts = visible_children(bar)
    named = [bar.slider, bar.flag_strip, *row_parts(bar)]  # the seven buttons, the frame box and the time
    assert all(part in parts for part in named), seen
    assert [part.text() for part in parts if part not in named] == ["Frame"], seen  # and the word at the box
    assert layout_findings(bar, {bar.play_button: "Pause", bar.time_label: "t = 000.000 s"}) == [], seen

    assert 340 <= dock.width() <= 400, seen
    for button in row_parts(bar)[:7]:
        longest = "Pause" if button is bar.play_button else button.text()
        text = button.fontMetrics().horizontalAdvance(longest)
        if dock.width() > 340:  # the dock gives way first: until it is at its smallest, no button gives up padding
            assert button.width() == text + 2 * (6 + 1), f"{longest}\n{seen}"
        else:
            assert text + 2 * (1 + 1) <= button.width() <= text + 2 * (6 + 1), f"{longest}\n{seen}"
    assert bar.frame_box.width() == 96, seen
    time = bar.time_label.geometry()
    assert bar.first_button.x() == 8 and time.x() + time.width() == bar.width() - 8, seen  # the bar's padding


@pytest.mark.parametrize("larger_font", ROW_FONTS, indirect=True)
def test_in_the_smallest_window_no_part_of_the_bottom_bar_is_cut_or_lies_over_another(larger_font, window, qtbot):
    # `larger_font` comes before `window`: the window measures its texts while it is built
    bar = smallest(window, qtbot, grid_frames(3, 40, 4))
    assert bar.play_button.text() == "Play"
    hold_to_the_rule(window)
    qtbot.mouseClick(bar.play_button, LEFT)
    assert bar.play_button.text() == "Pause"
    hold_to_the_rule(window)


@pytest.mark.parametrize("larger_font", BOX_FONTS, indirect=True)
def test_in_the_smallest_window_the_frame_box_shows_a_frame_number_of_six_digits_whole(larger_font, window, qtbot):
    # `larger_font` comes before `window`, as above: the box takes its font while the window is built
    bar = smallest(window, qtbot, [0, 999_999])
    bar.set_frame(999_999)
    box = bar.frame_box
    assert (box.width(), box.text()) == (96, "999999")
    edit, option = box.lineEdit(), QStyleOptionFrame()
    edit.initStyleOption(option)
    room = edit.style().subElementRect(QStyle.SubElement.SE_LineEditContents, option, edit)
    room = room.marginsRemoved(edit.textMargins()).width()
    # a line edit keeps 2 px free at each side of its text and 1 px for the cursor; with less room than
    # that it moves the text to where the cursor is, and one end of the number is cut
    assert edit.fontMetrics().horizontalAdvance("999999") + 1 <= room - 2 * 2, f"room {room} px\n{measured(window)}"
