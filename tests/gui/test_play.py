"""Playing the clip (SPEC 10.1, 10.2; task C8a): Space or the Play button goes one grid step per
tick, and the bottom row with the button in it.

Expected values come from the spec and the design note:
- the row is First, -10, -1, Play/Pause, +1, +10, Last, 28 px high and 4 px apart, then the frame
  box (96 px) and the time (SPEC 10.1; Qt's device-independent px);
- playing shows the next frame of the clip's grid at each tick, 30 ticks per s, so the timer's
  interval is 1000 / 30 = 33 ms (rounded); it stops at the last grid frame;
- a button is as wide as its text with 6 px at each side inside its 1 px edge; in a row that is
  too narrow the buttons give up padding (down to 1 px) and every part stays in the bar.
No test waits for time to pass: a tick is the timer's own signal, emitted by the test.
"""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from finish_helpers import row_parts
from gui_helpers import ClickRecorder, StandInSource, picture, show
from outline_tracker.geometry import grid_frames
from outline_tracker.gui import navigation
from session_helpers import body

LEFT = Qt.MouseButton.LeftButton
SPACE = Qt.Key.Key_Space
EVERY_FOURTH = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]  # the grid of a clip from 3 to 40, step 4


@pytest.fixture
def bar(window, qtbot):
    """The window's bottom bar on the grid 3, 7, ..., 39, shown; `bar.asked` collects the frames it
    asks for. The view shows them from a stand-in with 200 frames."""
    made = window.navigation
    made.asked = []
    made.frame_requested.connect(made.asked.append)
    window.view.set_source(StandInSource(n_frames=200))
    made.set_grid(grid_frames(3, 40, 4))
    show(window, qtbot)
    return made


def tick(bar, times: int = 1) -> None:
    """Let the play timer fire `times` times, without waiting for it."""
    for _ in range(times):
        bar.play_timer.timeout.emit()


def is_paused(bar) -> bool:
    return not bar.playing and not bar.play_timer.isActive() and bar.play_button.text() == "Play"


def is_playing(bar) -> bool:
    return bar.playing and bar.play_timer.isActive() and bar.play_button.text() == "Pause"


# ---------------------------------------------------------------------------------------------
# The row


def test_the_row_has_play_between_the_steps_back_and_the_steps_forward(bar):
    # the bar as the design note lays it out (docs/design/gui_design.md): slider, flag strip, the row
    assert bar.height() == 76
    assert (bar.slider.geometry().top(), bar.slider.height()) == (8, 20)
    assert (bar.flag_strip.geometry().top(), bar.flag_strip.height()) == (8 + 20 + 2, 6)
    buttons = row_parts(bar)[:7]
    assert [button.text() for button in buttons] == ["First", "−10", "−1", "Play", "+1", "+10", "Last"]
    assert {button.height() for button in buttons} == {28}
    row_top = 8 + 20 + 2 + 6 + 4
    assert {button.mapTo(bar, QPoint(0, 0)).y() for button in buttons} == {row_top}
    assert row_top + 28 + 8 == 76
    lefts = [button.mapTo(bar, QPoint(0, 0)).x() for button in buttons]
    assert lefts[0] == 8  # the bar's padding
    gaps = [right - (left + button.width()) for button, left, right in zip(buttons, lefts, lefts[1:])]
    assert gaps == [4] * 6
    assert bar.frame_box.width() == 96
    box_left = bar.frame_box.mapTo(bar, QPoint(0, 0)).x()
    time_left = bar.time_label.mapTo(bar, QPoint(0, 0)).x()
    assert lefts[-1] + buttons[-1].width() < box_left < box_left + 96 <= time_left
    assert time_left + bar.time_label.width() == bar.width() - 8
    assert bar.slider.geometry().left() == 8 and bar.slider.geometry().right() + 1 == bar.width() - 8
    assert bar.play_button.toolTip() == "Play or pause (Space)"


def test_the_play_button_keeps_its_width_when_its_text_changes(bar, qtbot):
    button = bar.play_button
    metrics = button.fontMetrics()
    widest = max(metrics.horizontalAdvance("Play"), metrics.horizontalAdvance("Pause"))
    assert button.width() == widest + 2 * (6 + 1)  # 6 px beside the longer text, inside the 1 px edge
    before = [part.geometry() for part in row_parts(bar)]
    qtbot.mouseClick(button, LEFT)
    assert button.text() == "Pause"
    QApplication.processEvents()
    assert [part.geometry() for part in row_parts(bar)] == before  # nothing in the row moved


def test_in_the_smallest_window_the_row_shows_every_part(window, qtbot):
    window.resize(window.minimumSize())  # 960 x 600
    show(window, qtbot)
    bar = window.navigation
    row = row_parts(bar)
    lefts = [part.mapTo(bar, QPoint(0, 0)).x() for part in row]
    assert all(part.isVisible() for part in row)
    assert lefts[0] == 8 and lefts[-1] + row[-1].width() == bar.width() - 8
    for part, left, next_left in zip(row, lefts, lefts[1:]):
        assert left + part.width() <= next_left, part  # no part lies over the next one
    for button in row[:7]:  # no text is cut: at least 1 px beside it at each side, inside the 1 px edge
        widest = "Pause" if button is bar.play_button else button.text()
        assert button.width() >= button.fontMetrics().horizontalAdvance(widest) + 2 * (1 + 1), button.text()
    assert bar.frame_box.width() == 96
    assert bar.time_label.width() >= bar.time_label.fontMetrics().horizontalAdvance("t = 000.000 s")


def test_in_a_row_that_is_too_narrow_the_buttons_give_up_padding_and_every_part_stays(window, qtbot):
    window.resize(1440, 900)  # room for the row with all its padding
    show(window, qtbot)
    bar = window.navigation
    row = row_parts(bar)
    buttons = row[:7]
    full = [button.width() for button in buttons]
    for button, width in zip(buttons, full):
        widest = "Pause" if button is bar.play_button else button.text()
        assert width == button.fontMetrics().horizontalAdvance(widest) + 2 * (6 + 1), button.text()
    assert bar.minimumSizeHint().width() == bar.layout().minimumSize().width() + 7 * 2 * (6 - 1)
    needed = bar.minimumSizeHint().width()  # the row with all its padding, and the bar's 8 px at each side

    bar.resize(needed - 30, bar.height())  # 30 px short; the 7 buttons can give up 7 * 2 * 5 = 70 px
    lefts = [part.mapTo(bar, QPoint(0, 0)).x() for part in row]
    assert lefts[0] == 8 and lefts[-1] + row[-1].width() == bar.width() - 8
    for part, left, next_left in zip(row, lefts, lefts[1:]):
        assert left + part.width() <= next_left, part
    assert sum(full) - sum(button.width() for button in buttons) == 30  # the buttons gave what was short
    for button, width in zip(buttons, full):
        assert width - 2 * 5 <= button.width() <= width, button.text()  # never under 1 px of padding
    assert bar.frame_box.width() == 96  # the box and the time gave nothing
    assert bar.time_label.width() >= bar.time_label.fontMetrics().horizontalAdvance("t = 000.000 s")


# ---------------------------------------------------------------------------------------------
# Playing


def test_a_tick_comes_30_times_a_second(bar):
    assert navigation.PLAY_TICKS_PER_S == 30
    assert bar.play_timer.interval() == 33  # ms: 1000 / 30, rounded
    assert not bar.play_timer.isSingleShot()
    assert is_paused(bar)


def test_space_plays_one_grid_step_per_tick_and_space_pauses(window, bar, qtbot):
    bar.set_frame(7)
    qtbot.keyClick(window, SPACE)
    assert is_playing(bar)
    assert bar.asked == []  # playing starts from the frame shown: nothing is asked for until a tick
    tick(bar, 3)
    assert bar.asked == [11, 15, 19]
    assert (bar.frame, window.view.frame, bar.frame_box.value(), bar.slider.value()) == (19, 19, 19, 4)
    qtbot.keyClick(window, SPACE)
    assert is_paused(bar)
    tick(bar)  # a tick that was on its way when playing stopped moves nothing
    assert bar.asked == [11, 15, 19] and bar.frame == 19
    qtbot.keyClick(window, SPACE)  # and on from where it stopped
    tick(bar)
    assert is_playing(bar) and bar.asked == [11, 15, 19, 23]


def test_the_button_plays_and_pauses(window, bar, qtbot):
    qtbot.mouseClick(bar.play_button, LEFT)
    assert is_playing(bar)
    tick(bar, 2)
    assert bar.asked == [7, 11] and window.view.frame == 11
    qtbot.mouseClick(bar.play_button, LEFT)
    assert is_paused(bar) and bar.frame == 11


def test_playing_stops_at_the_last_grid_frame(window, bar, qtbot):
    bar.set_frame(31)
    qtbot.keyClick(window, SPACE)
    tick(bar)
    assert bar.frame == 35 and is_playing(bar)
    tick(bar)
    assert bar.frame == 39 and is_paused(bar)  # the last frame of the grid is shown, and playing is over
    assert bar.asked == [35, 39] and window.view.frame == 39
    qtbot.keyClick(window, SPACE)  # at the last frame there is nothing to play
    assert is_paused(bar) and bar.asked == [35, 39]


def test_every_frame_played_is_a_frame_of_the_grid(window, bar, qtbot):
    qtbot.keyClick(window, SPACE)
    tick(bar, 20)  # more ticks than the grid has frames
    assert bar.asked == EVERY_FOURTH[1:] and is_paused(bar)


def test_playing_stops_when_another_frame_is_asked_for(window, bar, qtbot):
    def typed():
        bar.frame_box.setFocus()
        bar.frame_box.selectAll()
        qtbot.keyClicks(bar.frame_box, "23")
        qtbot.keyClick(bar.frame_box, Qt.Key.Key_Return)

    ways = {"a step button": lambda: qtbot.mouseClick(bar.forward_button, LEFT),
            "the slider": lambda: bar.slider.setValue(6),
            "a key": lambda: qtbot.keyClick(window, Qt.Key.Key_Right),
            "Home": lambda: qtbot.keyClick(window, Qt.Key.Key_Home),
            "the frame box": typed}
    for name, ask in ways.items():
        window.view.setFocus()
        bar.set_frame(11)
        bar.play()
        assert is_playing(bar), name
        ask()
        assert is_paused(bar), name


def test_without_a_grid_nothing_plays(window, qtbot):
    show(window, qtbot)
    bar = window.navigation
    qtbot.keyClick(window, SPACE)
    bar.play()
    assert is_paused(bar) and not bar.play_button.isEnabled()


def test_playing_stops_when_a_tool_is_chosen(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip)
    bar = window.navigation
    for tool in (ClickRecorder(), None):  # a tool, and Esc back to Pan
        bar.play()
        assert is_playing(bar)
        view.set_tool(tool)
        assert is_paused(bar)


def test_playing_stops_when_a_value_of_the_session_changes(window, qtbot, disk_clip):
    picture(window, qtbot, disk_clip)
    bar = window.navigation
    bar.play()
    tick(bar)
    assert is_playing(bar) and bar.frame == 2  # a new session's clip has step 2
    window.controller.session.time.fps_true = 240.0  # what a field that was typed in does
    window.controller.touch()
    assert is_paused(bar) and bar.frame == 2


def test_playing_stops_when_a_text_field_gets_the_keyboard(window, qtbot, disk_clip):
    picture(window, qtbot, disk_clip)
    bar = window.navigation
    for field in (body(window, 1).name_edit, body(window, 2).fps_edit, bar.frame_box, body(window, 1).step_box):
        window.view.setFocus()
        bar.play()
        assert is_playing(bar)
        field.setFocus()
        QApplication.processEvents()
        assert is_paused(bar), field
    window.view.setFocus()
    bar.play()
    bar.slider.setFocus()  # no text field: playing goes on
    QApplication.processEvents()
    assert is_playing(bar)


def test_a_text_field_keeps_space_for_itself(window, qtbot, clip_in_odd_folder):
    picture(window, qtbot, clip_in_odd_folder)  # a copy: the typed name gives it a run folder beside it
    bar, name = window.navigation, body(window, 1).name_edit
    name.setFocus()
    QApplication.processEvents()
    qtbot.keyClicks(name, "Ada")
    qtbot.keyClick(name, SPACE)
    qtbot.keyClicks(name, "L")
    assert name.text() == "Ada L" and is_paused(bar)
    for part in (window.view, bar.slider, window.scroll, window.panels[0].header, bar.forward_button):
        part.setFocus()
        QApplication.processEvents()
        qtbot.keyClick(part, SPACE)
        assert is_playing(bar), part  # wherever else the focus is, Space plays
        qtbot.keyClick(part, SPACE)
        assert is_paused(bar), part


def test_playing_stops_at_a_frame_that_cannot_be_read(window, qtbot, disk_clip):
    # a video that ends before the frame count its file reports: frames 0 to 2 exist, the grid has more
    view = picture(window, qtbot, disk_clip, StandInSource(n_frames=3))
    bar = window.navigation
    qtbot.keyClick(window, SPACE)
    tick(bar)
    assert (view.frame, bar.frame) == (2, 2) and is_playing(bar)
    tick(bar)  # frame 4 does not exist
    assert (view.frame, bar.frame) == (2, 2) and is_paused(bar)
    assert "Frame 4" in window.statusBar().currentMessage()


def test_a_new_video_starts_paused(window, qtbot, disk_clip, dish_clip):
    picture(window, qtbot, disk_clip)
    bar = window.navigation
    bar.play()
    tick(bar)
    window.open_path(dish_clip.path)
    assert is_paused(bar) and bar.frame == 0
