"""The bottom bar and the keys that move through the clip (SPEC 3.3, 3.4, 10.1, 10.2; task C1).

Expected values come from the spec and the design note:
- the frame grid of a clip is start, start + step, ... up to its end (SPEC 3.4); the slider, the
  buttons and the keys move on it, and a typed frame goes to the nearest grid frame (a frame
  exactly between two goes to the later one, the direction of decision X20);
- a video that is opened shows its first frame (frame 0 of a new session's clip), also when another
  video was open before it and the bar was elsewhere;
- left and right arrow: 1 step; with Shift: 10 steps; Home and End: the first and the last frame;
- t = frame / fps_true in s with three decimals, or a dash while fps_true is not known (SPEC 3.3).
Where the bar's parts lie is held in tests/gui/test_play.py (the design note's px) and in
tests/gui/test_row_rule.py (the smallest window: nothing cut, nothing over another part).
"""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

import helpers
from gui_helpers import StandInSource, picture, show, shown_pixels
from outline_tracker import synthetic
from outline_tracker.frame_source import FrameSource
from outline_tracker.geometry import grid_frames
from outline_tracker.gui.navigation import NavigationBar

LEFT = Qt.MouseButton.LeftButton
SHIFT = Qt.KeyboardModifier.ShiftModifier
DASH = "t = –"  # en dash: not known
EVERY_FOURTH = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]  # the grid of a clip from 3 to 40, step 4


def typed(qtbot, bar, text: str) -> None:
    """Type `text` into the frame box in place of what it shows, and press Enter."""
    box = bar.frame_box
    box.setFocus()
    box.selectAll()
    qtbot.keyClicks(box, text)
    qtbot.keyClick(box, Qt.Key.Key_Return)


# ---------------------------------------------------------------------------------------------
# The bar alone


def test_the_window_offers_the_bar(window):
    assert isinstance(window.navigation, NavigationBar)
    assert window.bottom_bar is window.navigation


def test_before_a_video_is_open_the_bar_cannot_be_used(window, qtbot):
    show(window, qtbot)
    bar = window.navigation
    assert not bar.isEnabled()
    assert bar.frame is None
    assert bar.time_label.text() == DASH
    asked = []
    bar.frame_requested.connect(asked.append)
    for key in (Qt.Key.Key_Right, Qt.Key.Key_End):
        qtbot.keyClick(window, key)
    assert asked == []


def test_every_button_says_what_it_does_and_names_its_key(bar):
    tips = {button.text(): button.toolTip() for button in (
        bar.first_button, bar.back_ten_button, bar.back_button,
        bar.forward_button, bar.forward_ten_button, bar.last_button)}
    assert tips == {
        "First": "First frame of the clip (Home)",
        "−10": "Back 10 steps (Shift+←)",
        "−1": "Back 1 step (←)",
        "+1": "Forward 1 step (→)",
        "+10": "Forward 10 steps (Shift+→)",
        "Last": "Last frame of the clip (End)",
    }


def test_frame_and_time_are_one_point_larger_demibold_with_digits_of_one_width(bar, qapp):
    base = qapp.font().pointSizeF()
    for part in (bar.frame_box, bar.time_label):
        font = part.font()
        assert font.pointSizeF() == base + 1
        assert font.weight() == QFont.Weight.DemiBold
        assert font.featureValue(QFont.Tag("tnum")) == 1  # tabular digits: the text does not jump


def test_the_slider_only_lands_on_grid_frames(bar, qtbot):
    assert bar.frame == 3  # a new grid starts at its first frame
    slider = bar.slider
    assert (slider.minimum(), slider.maximum()) == (0, 9)  # one position for each of the 10 grid frames
    for position in range(slider.minimum(), slider.maximum() + 1):
        slider.setValue(position)
        assert bar.frame == EVERY_FOURTH[position]
    slider.setValue(500)  # past the end: the last grid frame
    assert bar.frame == 39
    assert bar.asked == EVERY_FOURTH[1:]  # each frame asked for once; frame 3 was already shown
    # a click anywhere on the slider lands on a grid frame too
    for x in (2, slider.width() // 3, slider.width() // 2 + 1, slider.width() - 3):
        qtbot.mouseClick(slider, LEFT, Qt.KeyboardModifier.NoModifier, QPoint(x, slider.height() // 2))
        assert bar.frame in EVERY_FOURTH and bar.frame_box.value() == bar.frame
    assert set(bar.asked) <= set(EVERY_FOURTH)


def test_the_buttons_move_one_step_ten_steps_and_to_the_ends(window, qtbot):
    bar = window.navigation
    window.view.set_source(StandInSource(n_frames=120))
    bar.set_grid(grid_frames(0, 119, 2))  # 0, 2, ..., 118: 60 frames
    show(window, qtbot)
    asked = []
    bar.frame_requested.connect(asked.append)
    presses = [(bar.forward_ten_button, 20), (bar.forward_button, 22), (bar.back_ten_button, 2),
               (bar.back_button, 0), (bar.last_button, 118), (bar.back_button, 116), (bar.first_button, 0)]
    for button, frame in presses:
        qtbot.mouseClick(button, LEFT)
        assert bar.frame == frame
        assert (bar.slider.value(), bar.frame_box.value()) == (frame // 2, frame)
    assert asked == [frame for _, frame in presses]


def test_a_step_past_an_end_stops_at_the_end_and_asks_for_nothing_twice(bar, qtbot):
    qtbot.mouseClick(bar.back_button, LEFT)  # already at the first frame
    qtbot.mouseClick(bar.first_button, LEFT)
    assert bar.frame == 3 and bar.asked == []
    qtbot.mouseClick(bar.forward_button, LEFT)
    qtbot.mouseClick(bar.forward_ten_button, LEFT)  # 1 + 10 steps is past the tenth frame
    assert bar.frame == 39
    qtbot.mouseClick(bar.forward_button, LEFT)
    qtbot.mouseClick(bar.last_button, LEFT)
    assert bar.asked == [7, 39]
    qtbot.mouseClick(bar.back_ten_button, LEFT)
    assert bar.frame == 3 and bar.asked == [7, 39, 3]


@pytest.mark.parametrize("text, frame", [
    ("7", 7), ("8", 7), ("10", 11), ("38", 39), ("0", 3), ("2", 3), ("40", 39), ("5000", 39),
    ("9", 11), ("5", 7),  # exactly between two grid frames: the later one
])
def test_a_typed_frame_goes_to_the_nearest_grid_frame(bar, qtbot, text, frame):
    bar.set_frame(19)
    typed(qtbot, bar, text)
    assert bar.frame == frame
    assert bar.frame_box.value() == frame  # the box shows where it went
    assert bar.slider.value() == EVERY_FOURTH.index(frame)
    assert bar.asked == [frame]


def test_set_frame_shows_a_frame_without_asking_for_it(bar):
    bar.set_frame(27)
    assert (bar.frame, bar.frame_box.value(), bar.slider.value()) == (27, 27, 6)
    bar.set_frame(28)  # off the grid: the nearest grid frame
    assert (bar.frame, bar.frame_box.value(), bar.slider.value()) == (27, 27, 6)
    assert bar.asked == []


def test_a_new_grid_keeps_the_frame_if_it_can(bar):
    bar.set_frame(27)
    bar.set_grid(grid_frames(3, 99, 8))  # 3, 11, 19, 27, ...: 27 is on it
    assert (bar.frame, bar.slider.value(), bar.slider.maximum()) == (27, 3, 12)
    bar.set_grid(grid_frames(40, 60, 5))  # 40, 45, ..., 60: the nearest to 27 is 40
    assert (bar.frame, bar.slider.value(), bar.slider.maximum()) == (40, 0, 4)
    assert bar.asked == []  # the bar does not ask: whoever set the grid shows `frame`


def test_an_empty_grid_switches_the_bar_off_and_the_next_grid_starts_at_its_first_frame(bar):
    bar.set_fps(2.0)
    bar.set_frame(27)
    assert bar.time_label.text() == "t = 13.500 s"  # 27 / 2
    bar.set_grid([])
    assert not bar.isEnabled() and bar.frame is None
    assert (bar.slider.value(), bar.slider.maximum(), bar.frame_box.value()) == (0, 0, 0)
    assert bar.time_label.text() == DASH  # no frame: no time
    bar.set_grid(grid_frames(3, 99, 8))  # 3, 11, 19, 27, ...: 27 is on it, but the bar was on no frame
    assert bar.isEnabled()
    assert (bar.frame, bar.frame_box.value(), bar.slider.value(), bar.slider.maximum()) == (3, 3, 0, 12)
    assert bar.time_label.text() == "t = 1.500 s"  # 3 / 2
    assert bar.asked == []


def test_t_is_the_frame_over_fps_true_in_seconds_or_a_dash(bar):
    assert bar.time_label.text() == DASH  # fps_true is not known yet
    bar.set_frame(7)
    bar.set_fps(240.0)
    assert bar.time_label.text() == "t = 0.029 s"  # 7 / 240 = 0.02917, shown with three decimals
    bar.set_frame(35)
    assert bar.time_label.text() == "t = 0.146 s"  # 35 / 240 = 0.14583
    bar.set_fps(2.0)
    assert bar.time_label.text() == "t = 17.500 s"  # 35 / 2
    bar.set_frame(7)
    assert bar.time_label.text() == "t = 3.500 s"
    bar.set_fps(None)
    assert bar.time_label.text() == DASH


# ---------------------------------------------------------------------------------------------
# In the window, with a video


def test_opening_a_video_puts_the_bar_on_the_clips_grid(window, qtbot, dish_clip):
    picture(window, qtbot, dish_clip)  # 120 frames; a new session's clip is 0 to 119, step 2
    bar = window.navigation
    assert bar.isEnabled()
    assert (bar.slider.minimum(), bar.slider.maximum()) == (0, 59)
    assert (bar.frame, window.view.frame) == (0, 0)
    assert bar.time_label.text() == DASH  # a new session has no fps_true


def test_a_second_video_opens_on_its_first_frame(window, qtbot, dish_clip, disk_clip, tmp_path):
    # Each video opened starts at its frame 0, wherever the bar was in the video before it: on a
    # frame the new clip has too, or past the new clip's end. The picture is frame 0 of the new file.
    short = synthetic.render(synthetic.closeup_scene(size=helpers.SMALL, n_frames=30), tmp_path / "short_tracker.mp4")
    view = picture(window, qtbot, dish_clip)
    bar, controller = window.navigation, window.controller
    controller.session.time.fps_true = 240.0
    controller.touch()
    bar.slider.setValue(20)
    assert (view.frame, bar.frame) == (40, 40)
    assert bar.time_label.text() == "t = 0.167 s"  # 40 / 240 = 0.1667

    window.open_path(disk_clip.path)  # as long as the first one: frame 40 is on its grid too
    assert controller.video_path == disk_clip.path
    assert (bar.slider.minimum(), bar.slider.maximum()) == (0, 59)
    assert (view.frame, bar.frame, bar.slider.value(), bar.frame_box.value()) == (0, 0, 0, 0)
    assert bar.time_label.text() == DASH  # the new session has no fps_true
    with FrameSource(disk_clip.path) as reference:
        assert not np.array_equal(reference.get(0), reference.get(40))  # the frames do differ
        assert np.array_equal(shown_pixels(view), reference.get(0))

    bar.slider.setValue(20)
    assert (view.frame, bar.frame) == (40, 40)
    window.open_path(short.path)  # frames 0 to 29: its grid is 0, 2, ..., 28, and frame 40 is past its end
    assert controller.video_path == short.path
    assert (bar.slider.minimum(), bar.slider.maximum()) == (0, 14)
    assert (view.frame, bar.frame, bar.slider.value(), bar.frame_box.value()) == (0, 0, 0, 0)
    with FrameSource(short.path) as reference:
        assert not np.array_equal(reference.get(0), reference.get(28))
        assert np.array_equal(shown_pixels(view), reference.get(0))

    qtbot.keyClick(window, Qt.Key.Key_End)
    assert (view.frame, bar.frame) == (28, 28)
    window.open_path(short.path)  # the same file once more: a new session, from its first frame
    assert (view.frame, bar.frame, bar.slider.value(), bar.frame_box.value()) == (0, 0, 0, 0)
    qtbot.keyClick(window, Qt.Key.Key_Right)  # and the bar moves on the new clip's grid from there
    assert (view.frame, bar.frame) == (2, 2)


def test_the_arrow_keys_move_one_and_ten_steps(window, qtbot, dish_clip):
    view = picture(window, qtbot, dish_clip)
    bar = window.navigation
    keys = [(Qt.Key.Key_Right, None, 2), (Qt.Key.Key_Right, SHIFT, 22), (Qt.Key.Key_Left, None, 20),
            (Qt.Key.Key_Right, None, 22), (Qt.Key.Key_Left, SHIFT, 2), (Qt.Key.Key_End, None, 118),
            (Qt.Key.Key_Right, None, 118), (Qt.Key.Key_Left, None, 116), (Qt.Key.Key_Home, None, 0),
            (Qt.Key.Key_Left, SHIFT, 0)]
    with FrameSource(dish_clip.path) as reference:
        for key, modifier, frame in keys:
            qtbot.keyClick(window, key, modifier or Qt.KeyboardModifier.NoModifier)
            assert (view.frame, bar.frame) == (frame, frame)
            assert np.array_equal(shown_pixels(view), reference.get(frame)), frame
        assert not np.array_equal(reference.get(0), reference.get(118))  # the frames do differ


def test_the_keys_work_wherever_the_focus_is_except_in_the_frame_box(window, qtbot, dish_clip):
    view = picture(window, qtbot, dish_clip)
    bar = window.navigation
    for part in (view, bar.slider, window.scroll, window.panels[0].header):
        part.setFocus()
        QApplication.processEvents()
        before = view.frame
        qtbot.keyClick(part, Qt.Key.Key_Right, SHIFT)
        assert view.frame == before + 20, part
    bar.frame_box.setFocus()  # there the arrows move the text cursor
    QApplication.processEvents()
    before = view.frame
    qtbot.keyClick(bar.frame_box, Qt.Key.Key_Left)
    qtbot.keyClick(bar.frame_box, Qt.Key.Key_Home)
    assert view.frame == before


def test_a_frame_asked_for_in_the_bar_is_the_frame_shown(window, qtbot, dish_clip):
    view = picture(window, qtbot, dish_clip)
    bar = window.navigation
    with FrameSource(dish_clip.path) as reference:
        bar.slider.setValue(17)
        assert view.frame == 34
        assert np.array_equal(shown_pixels(view), reference.get(34))
        typed(qtbot, bar, "101")  # between 100 and 102: the later one
        assert (view.frame, bar.frame) == (102, 102)
        assert np.array_equal(shown_pixels(view), reference.get(102))


def test_the_bar_follows_the_sessions_clip_and_fps_true(window, qtbot, dish_clip):
    view = picture(window, qtbot, dish_clip)
    bar, controller = window.navigation, window.controller
    bar.slider.setValue(30)  # frame 60
    controller.session.time.fps_true = 240.0
    controller.touch()
    assert bar.time_label.text() == "t = 0.250 s"  # 60 / 240
    assert view.frame == 60
    clip = controller.session.clip
    clip.start, clip.end, clip.step = 10, 50, 5  # 10, 15, ..., 50: frame 60 is past its end
    controller.touch()
    assert (bar.slider.minimum(), bar.slider.maximum()) == (0, 8)
    assert (bar.frame, view.frame) == (50, 50)
    assert bar.time_label.text() == "t = 0.208 s"  # 50 / 240 = 0.2083
    qtbot.keyClick(window, Qt.Key.Key_Home)
    assert (bar.frame, view.frame) == (10, 10)
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert (bar.frame, view.frame) == (15, 15)


def test_a_frame_that_cannot_be_read_is_reported_and_the_bar_stays(window, qtbot, dish_clip):
    # a video that ends before the frame count its file reports: frames 0 to 2 exist, the grid has more
    view = picture(window, qtbot, dish_clip, StandInSource(n_frames=3))
    bar = window.navigation
    qtbot.keyClick(window, Qt.Key.Key_Right)
    assert (view.frame, bar.frame) == (2, 2)
    qtbot.keyClick(window, Qt.Key.Key_Right)  # frame 4 does not exist
    assert (view.frame, bar.frame, bar.slider.value(), bar.frame_box.value()) == (2, 2, 1, 2)
    message = window.statusBar().currentMessage()
    assert "Frame 4" in message and "cannot be read" in message
