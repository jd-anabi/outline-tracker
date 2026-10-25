"""Panel 8, Review and fix (SPEC 9, 10.1; task C7): going to a flag, by a row, by Next and Previous, and by the keys.
The table and what the panel says are in tests/gui/test_review.py, the three corrections are in
tests/gui/test_review_fix.py.

Expected values:
- The dish scene's B and C are two ellipses on straight paths that pass each other in frame 60
  with a gap of 0.6 px. Their true gap on every frame comes from the two true ellipses
  (`export_helpers.ellipse_gap`); CONTACT is "within 3 px" here. Frames whose true gap is at most
  2.7 px must be listed for B and for C, frames whose gap is at least 3.3 px must not. A swims
  along the wall, far from both.
The texts are the design note's. Nothing is read from a pixel of text, and no test waits with a
delay. Frames are video frame numbers; times are s.
"""

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from review_helpers import GRID, current_row, frames_with, opened_run, row_of

LEFT = Qt.MouseButton.LeftButton


# ---------------------------------------------------------------------------------------------
# Going to a flag


def test_activating_a_row_moves_the_slider_there_and_selects_its_object(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    assert window.view.frame == 0 and review.prompts.selected is None
    review.go_to_row(row_of(review, "C", 62))
    assert window.view.frame == window.navigation.frame == 62
    assert window.navigation.slider.value() == GRID.index(62)  # the slider counts steps of the grid
    assert review.prompts.selected == "C"
    assert current_row(review) == row_of(review, "C", 62)
    # panel 6 shows the same selection
    objects = window.panels[5].body.itemAt(1).widget()
    assert objects.table.currentRow() == 2


def test_a_click_on_a_row_goes_to_its_frame(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    row = row_of(review, "B", 58)
    review.table.scrollTo(review.table.model().index(row, 0))
    place = review.table.visualRect(review.table.model().index(row, 1)).center()
    qtbot.mouseClick(review.table.viewport(), LEFT, Qt.KeyboardModifier.NoModifier, place)
    assert window.view.frame == window.navigation.frame == 58 and review.prompts.selected == "B"
    # a click on the same row again comes back to it from another frame
    window.navigation.step(3)
    assert window.view.frame == 64
    qtbot.mouseClick(review.table.viewport(), LEFT, Qt.KeyboardModifier.NoModifier, place)
    assert window.view.frame == window.navigation.frame == 58


def test_next_and_previous_go_through_the_selected_objects_flags_and_wrap(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    assert not review.next_button.isEnabled() and not review.previous_button.isEnabled()  # no object selected
    review.prompts.select("B")
    frames = frames_with(review.rows, "B", "CONTACT")
    assert {58, 60, 62} <= set(frames)  # by geometry: gaps of 0.6 to 0.7 px
    assert review.next_button.isEnabled() and review.previous_button.isEnabled()
    window.show_frame(60)
    review.next_flag()
    assert window.view.frame == window.navigation.frame == 62 and review.prompts.selected == "B"
    assert current_row(review) == row_of(review, "B", 62)
    review.previous_flag()
    review.previous_flag()
    assert window.view.frame == 58
    # from a frame between two flags, and from before the first one
    window.show_frame(0)
    review.next_flag()
    assert window.view.frame == frames[0]
    # around: before the first comes the last, after the last the first
    review.previous_flag()
    assert window.view.frame == frames[-1]
    review.next_flag()
    assert window.view.frame == frames[0]
    # every flag of B is reached once in a round, and no frame of another track's alone
    seen = []
    for _ in frames:
        seen.append(window.view.frame)
        review.next_flag()
    assert seen == frames and window.view.frame == frames[0]


def test_next_flag_goes_through_the_rows_that_are_listed(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    review.prompts.select("B")
    window.show_frame(10)
    review.next_flag()
    first = window.view.frame
    assert first == frames_with(review.rows, "B", "CONTACT")[0] > 12  # B and C are far apart until they meet
    review.shape_box.setChecked(True)  # now B has a row on every tracked frame (LOWRES, HEADGUESS)
    window.show_frame(10)
    review.next_flag()
    assert window.view.frame == 12
    review.previous_flag()
    review.previous_flag()
    assert window.view.frame == 8


def test_left_and_right_stay_the_bottom_bars_and_up_and_down_move_in_the_table(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    row = row_of(review, "B", 60)
    review.go_to_row(row)
    review.table.setFocus()
    qtbot.waitUntil(review.table.hasFocus)
    QTest.keyClick(review.table, Qt.Key.Key_Right)
    assert window.navigation.frame == window.view.frame == 62 and current_row(review) == row
    QTest.keyClick(review.table, Qt.Key.Key_Left)
    QTest.keyClick(review.table, Qt.Key.Key_Left)
    assert window.navigation.frame == 58 and current_row(review) == row
    QTest.keyClick(review.table, Qt.Key.Key_End)
    assert window.navigation.frame == GRID[-1] and current_row(review) == row
    QTest.keyClick(review.table, Qt.Key.Key_Home)
    assert window.navigation.frame == 0 and current_row(review) == row
    # down and up are the table's: the next row, and its frame
    QTest.keyClick(review.table, Qt.Key.Key_Down)
    assert current_row(review) == row + 1
    assert (review.prompts.selected, window.view.frame) == review.rows[row + 1][:2]
    QTest.keyClick(review.table, Qt.Key.Key_Up)
    QTest.keyClick(review.table, Qt.Key.Key_Up)
    assert current_row(review) == row - 1
    assert (review.prompts.selected, window.view.frame) == review.rows[row - 1][:2]
