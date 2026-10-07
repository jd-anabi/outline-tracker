"""The video view (SPEC 3.1, 10.1, 13.6; task C1): which frame is shown, where, and what a click reports.

Expected values come from the spec and from geometry:
- pixel (c, r) of a frame is the square [c, c+1) x [r, r+1), so a click in its middle is
  (u, v) = (c + 0.5, r + 0.5), with no further half pixel (SPEC 3.1, 10.1);
- "Fit" shows the whole frame in the middle of the view, v downward: a frame of W x H px in a view
  of w x h px is drawn at min(w / W, h / H) screen px per video px;
- "1:1" draws one video px on one screen px (Qt's device-independent px);
- the image shown for frame k is frame k of the file, bit for bit (a second `FrameSource`, and for
  the clip with gaps in its timestamps the sequential decode itself).
Where something is on the screen is read from the drawn view (tests/gui/gui_helpers.py), never
from the view's own numbers. The click test also runs on a 150% and on a 200% screen, each in a
process of its own (`QT_SCALE_FACTOR`).
"""

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

import helpers
from gui_helpers import GREEN, ClickRecorder, StandInSource, drag, drawn, picture, shown_pixels, wheel
from outline_tracker import video
from outline_tracker.frame_source import FrameSource
from outline_tracker.gui.video_view import VideoView

REPO = Path(__file__).resolve().parents[2]
SCREEN_FACTOR = "OUTLINE_TRACKER_TEST_SCREEN_FACTOR"  # set for the runs on a 150% and a 200% screen
LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
NO_KEY = Qt.KeyboardModifier.NoModifier
PAN_TEXT = "Pan: drag to move the picture. Scroll to zoom."


def fit_of(view, size):
    """Where a frame of `size` = (width, height) px lies in the view when all of it shows, by
    geometry: (screen px per video px, left, top), the corner in the viewport's px."""
    (width, height), room = size, view.viewport().size()
    scale = min(room.width() / width, room.height() / height)
    return scale, (room.width() - width * scale) / 2, (room.height() - height * scale) / 2


# ---------------------------------------------------------------------------------------------
# Which frame is shown


def test_the_window_shows_the_view_once_a_video_is_open(window, qtbot, disk_clip):
    assert isinstance(window.view, VideoView)
    assert window.view.frame is None and window.view.tool is None
    view = picture(window, qtbot, disk_clip)
    assert view.isVisible() and not window.open_button.isVisible()
    assert view.frame == 0  # the first frame of the clip
    assert window.video_area.isAncestorOf(view)


def test_the_image_shown_for_frame_k_equals_frame_source_get_k(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip)
    with FrameSource(disk_clip.path) as reference:  # a second reader of the same file
        assert not np.array_equal(reference.get(0), reference.get(64))  # the frames do differ
        for k in (64, 0, 119, 7, 118, 30):  # jumps back and forth, odd frames too
            view.show_frame(k)
            assert view.frame == k
            assert np.array_equal(view.image_item.image, reference.get(k)), k
            assert np.array_equal(shown_pixels(view), reference.get(k)), k


def test_on_a_clip_with_gaps_in_its_timestamps_the_image_is_still_frame_k(window, qtbot, gapped_clip):
    # the frames on both sides of each gap (before frames 30, 72 and 101), by counting from frame 0
    around = sorted({frame for gap in helpers.GAPS_BEFORE for frame in (gap - 1, gap)})
    assert around == [29, 30, 71, 72, 100, 101]
    counted = dict(video.iter_rgb_frames(gapped_clip.path, around))
    view = picture(window, qtbot, gapped_clip)
    for k in (101, 29, 72, 100, 30, 71):  # as jumps
        view.show_frame(k)
        assert view.frame == k
        assert np.array_equal(shown_pixels(view), counted[k]), k


def test_the_levels_of_a_frame_are_not_stretched(window, qtbot, disk_clip):
    source = StandInSource()  # frame 0 is gray 60 and 64 only: stretched levels would be 0 and 255
    view = picture(window, qtbot, disk_clip, source)
    assert set(np.unique(source.get(0))) == {60, 64}
    assert np.array_equal(shown_pixels(view), source.get(0))
    view.show_frame(3)
    assert np.array_equal(shown_pixels(view), source.get(3))


def test_a_frame_the_video_does_not_have_changes_nothing(window, qtbot, disk_clip):
    source = StandInSource(n_frames=3)
    view = picture(window, qtbot, disk_clip, source)
    view.show_frame(2)
    with pytest.raises(IndexError):
        view.show_frame(3)
    assert view.frame == 2
    assert np.array_equal(shown_pixels(view), source.get(2))


def test_a_new_source_starts_with_no_frame(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource())
    view.show_frame(4)
    view.set_source(StandInSource(size=(64, 48)))
    assert view.frame is None
    view.show_frame(1)
    assert view.frame == 1 and shown_pixels(view).shape == (48, 64, 3)


# ---------------------------------------------------------------------------------------------
# Where the picture is: Fit, 1:1, wheel, drag


def test_fit_shows_the_whole_frame_in_the_middle_with_v_downward(window, qtbot, disk_clip):
    # columns 40 to 79 and rows 20 to 39 are marked: left of the middle and above it
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(40, 20, 80, 40)))
    scale, left, top = fit_of(view, (320, 240))
    seen = drawn(view)
    assert seen.solid
    assert seen.left == pytest.approx(left + 40 * scale, abs=1)
    assert seen.right == pytest.approx(left + 80 * scale, abs=1)
    assert seen.top == pytest.approx(top + 20 * scale, abs=1)
    assert seen.bottom == pytest.approx(top + 40 * scale, abs=1)
    assert view.zoom == pytest.approx(scale, rel=1e-6)
    assert window.zoom_label.text() == f"{round(100 * scale)}%"


def test_the_picture_stays_fitted_when_the_view_changes_its_size(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(40, 20, 80, 40)))
    window.resize(window.width() + 180, window.height() + 40)
    QApplication.processEvents()
    scale, left, top = fit_of(view, (320, 240))
    seen = drawn(view)
    assert seen.left == pytest.approx(left + 40 * scale, abs=1)
    assert seen.bottom == pytest.approx(top + 40 * scale, abs=1)


@pytest.mark.xfail(strict=True, reason=(
    "Task C3: with a video open, the axes of SPEC 4.5 are drawn at the origin, which starts at the frame's "
    "center (160, 120). The y arrow goes up from there through the block this test marks (columns 150 to 169, "
    "rows 110 to 119), so the block is no longer one solid colour. The test that follows this one is "
    "test_one_to_one_draws_one_video_pixel_on_one_screen_pixel_beside_the_axes in tests/gui/test_tools.py: this "
    "test with the block at columns 190 to 209 and rows 140 to 149, clear of the axes. For J or the controller: "
    "delete this test."))
def test_one_to_one_draws_one_video_pixel_on_one_screen_pixel_and_fit_goes_back(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(150, 110, 170, 120)))
    assert [window.fit_button.text(), window.one_to_one_button.text()] == ["Fit", "1:1"]
    qtbot.mouseClick(window.one_to_one_button, LEFT)
    seen = drawn(view)
    assert seen.solid and (seen.width, seen.height) == (20, 10)  # the 20 x 10 px that are marked
    assert view.zoom == pytest.approx(1.0)
    assert window.zoom_label.text() == "100%"
    # the middle of the frame stays in the middle of the view: the marked block is around (160, 115)
    room = view.viewport().size()
    assert (seen.left + seen.right) / 2 == pytest.approx(room.width() / 2, abs=1)
    assert (seen.top + seen.bottom) / 2 == pytest.approx(room.height() / 2 - 5, abs=1)
    qtbot.mouseClick(window.fit_button, LEFT)
    scale, left, top = fit_of(view, (320, 240))
    seen = drawn(view)
    assert seen.left == pytest.approx(left + 150 * scale, abs=1)
    assert seen.width == pytest.approx(20 * scale, abs=1)


def test_the_wheel_zooms_at_the_cursor(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(60, 50, 70, 60)))
    before = drawn(view)
    wheel(view, before.center)
    closer = drawn(view)
    assert closer.width > before.width + 1 and closer.height > before.height + 1
    assert abs(closer.center.x() - before.center.x()) <= 1 and abs(closer.center.y() - before.center.y()) <= 1
    assert closer.width / closer.height == pytest.approx(1.0, abs=0.15)  # square pixels stay square
    wheel(view, closer.center, -1)
    again = drawn(view)
    assert again.width == pytest.approx(before.width, abs=1)


def test_pixels_are_smoothed_only_below_100_percent(window, qtbot, disk_clip):
    smooth = QPainter.RenderHint.SmoothPixmapTransform
    view = picture(window, qtbot, disk_clip, StandInSource((3200, 2400)))  # larger than any test window
    assert view.zoom < 1 and bool(view.renderHints() & smooth)
    view.one_to_one()
    assert not view.renderHints() & smooth
    view.fit()
    QApplication.processEvents()
    assert view.zoom < 1 and bool(view.renderHints() & smooth)


def test_a_drag_moves_the_picture_and_is_no_click(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(100, 80, 120, 100)))
    viewport = view.viewport()
    assert viewport.cursor().shape() == Qt.CursorShape.OpenHandCursor
    before = drawn(view)
    drag(qtbot, view, QPoint(60, 60), QPoint(100, 85), release=False)
    assert viewport.cursor().shape() == Qt.CursorShape.ClosedHandCursor  # while the picture is held
    qtbot.mouseRelease(viewport, LEFT, NO_KEY, QPoint(100, 85))
    assert viewport.cursor().shape() == Qt.CursorShape.OpenHandCursor
    after = drawn(view)
    assert after.left - before.left == pytest.approx(40, abs=1.5)
    assert after.top - before.top == pytest.approx(25, abs=1.5)
    assert after.width == pytest.approx(before.width, abs=1)  # moved, not zoomed

    tool = ClickRecorder()
    view.set_tool(tool)  # in a tool the picture can still be moved, and the cursor stays the tool's
    # (pressed at another place than the release above: a press at the same place is a double click)
    drag(qtbot, view, QPoint(240, 185), QPoint(200, 160), release=False)
    assert viewport.cursor().shape() == Qt.CursorShape.CrossCursor
    qtbot.mouseRelease(viewport, LEFT, NO_KEY, QPoint(200, 160))
    back = drawn(view)
    assert back.left == pytest.approx(before.left, abs=1.5) and back.top == pytest.approx(before.top, abs=1.5)
    assert tool.clicks == []


def test_a_drag_with_the_right_button_does_nothing(window, qtbot, disk_clip):
    # the right button is a negative point: a hand that slips must not zoom the picture
    tool = ClickRecorder()
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(100, 80, 120, 100)))
    view.set_tool(tool)
    before = drawn(view)
    drag(qtbot, view, QPoint(60, 60), QPoint(100, 85), button=RIGHT)
    assert drawn(view) == before
    assert tool.clicks == []


# ---------------------------------------------------------------------------------------------
# Clicks


@pytest.mark.parametrize("column, row", [(0, 0), (200, 100), (319, 239)])
def test_a_click_in_the_middle_of_a_pixel_reports_its_center(window, qtbot, disk_clip, column, row):
    factor = os.environ.get(SCREEN_FACTOR)
    if factor is not None:  # the run on a 150% or 200% screen: first make sure that it is one
        assert window.screen().devicePixelRatio() == float(factor)
    tool = ClickRecorder()
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(column, row, column + 1, row + 1)))
    room = view.viewport().size()
    seen = drawn(view)
    assert seen.width < 8  # the whole frame shows: the pixel is small
    # A pixel in a corner of the frame lies at the edge of the view, where zooming would push part of
    # it out of sight: first move the picture so that the pixel is well inside the view.
    drag(qtbot, view, seen.center, QPoint(room.width() // 2 + 9, room.height() // 2 + 7))
    view.set_tool(tool)
    for _ in range(40):  # zoom in on the marked pixel until it covers 8 x 8 screen px
        seen = drawn(view)
        if seen.width >= 8 and seen.height >= 8:
            break
        wheel(view, seen.center)
    seen = drawn(view)
    assert seen.solid and seen.width >= 8 and seen.height >= 8
    assert abs(seen.width - seen.height) <= 1  # all of the pixel shows: it is a square
    assert 0 < seen.left and seen.right < room.width() and 0 < seen.top and seen.bottom < room.height()

    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, seen.center)
    assert len(tool.clicks) == 1
    u, v, button, modifiers = tool.clicks[0]
    assert abs(u - (column + 0.5)) <= 0.2 and abs(v - (row + 0.5)) <= 0.2
    assert (math.floor(u), math.floor(v)) == (column, row)  # no extra half pixel
    assert (button, modifiers) == (LEFT, NO_KEY)

    # the right button with Shift held arrives as that (it is the negative point of the point tools)
    qtbot.mouseClick(view.viewport(), RIGHT, Qt.KeyboardModifier.ShiftModifier, seen.center)
    assert len(tool.clicks) == 2
    u, v, button, modifiers = tool.clicks[1]
    assert (math.floor(u), math.floor(v)) == (column, row)
    assert (button, modifiers) == (RIGHT, Qt.KeyboardModifier.ShiftModifier)


@pytest.mark.parametrize("factor", ["1.5", "2"])
def test_the_click_test_passes_on_a_screen_scaled_to(factor):
    # the three cases of the test above, in a fresh process whose screen has this device pixel ratio
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": factor, SCREEN_FACTOR: factor}
    command = [sys.executable, "-m", "pytest", "tests/gui/test_video_view.py", "-k", "reports_its_center",
               "-q", "--color=no", "-p", "no:cacheprovider"]
    done = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.strip().splitlines()
    assert re.match(r"3 passed, \d+ deselected", lines[-1] if lines else ""), done.stdout + done.stderr


def test_a_click_outside_the_image_is_ignored(window, qtbot, disk_clip):
    tool = ClickRecorder()
    # a frame four times as wide as high leaves room above and below it in any view of these tests
    view = picture(window, qtbot, disk_clip, StandInSource((400, 100), marked=(190, 40, 210, 60)))
    view.set_tool(tool)
    scale, left, top = fit_of(view, (400, 100))
    assert top >= 20  # there is room above the picture
    middle = view.viewport().width() // 2
    for outside in (QPoint(middle, 3), QPoint(middle, round(top) - 3),
                    QPoint(middle, view.viewport().height() - 3)):
        qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, outside)
    assert tool.clicks == []
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, QPoint(middle, round(top) + 3))  # just inside
    assert len(tool.clicks) == 1
    u, v = tool.clicks[0][:2]
    assert 0 <= u < 400 and 0 <= v < 100


def test_in_pan_a_click_does_nothing(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(100, 80, 120, 100)))
    before = drawn(view)
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, before.center)
    qtbot.mouseClick(view.viewport(), RIGHT, NO_KEY, before.center)  # and no menu opens
    assert drawn(view) == before
    assert QApplication.activePopupWidget() is None


# ---------------------------------------------------------------------------------------------
# Tools


def test_a_tool_brings_its_cursor_and_esc_returns_to_pan(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip)
    changes = []
    view.tool_changed.connect(lambda: changes.append(view.tool))
    assert view.tool is None
    assert view.viewport().cursor().shape() == Qt.CursorShape.OpenHandCursor
    assert window.tool_text.text() == PAN_TEXT
    tool = ClickRecorder()
    view.set_tool(tool)
    assert view.tool is tool
    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    assert window.tool_text.text() == "Recorder: click on the picture."  # the tool's own line
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert view.tool is None
    assert view.viewport().cursor().shape() == Qt.CursorShape.OpenHandCursor
    assert window.tool_text.text() == PAN_TEXT
    assert changes == [tool, None]


# ---------------------------------------------------------------------------------------------
# Graphics of other tasks, in image coordinates


def filled(left, top, width, height) -> QGraphicsRectItem:
    """A green rectangle without an edge, in px of the video frame."""
    item = QGraphicsRectItem(left, top, width, height)
    item.setPen(QPen(Qt.PenStyle.NoPen))
    item.setBrush(QBrush(QColor(*GREEN)))
    return item


def test_an_added_item_is_drawn_in_image_coordinates_and_can_be_removed(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(100, 80, 140, 100)))
    marked = drawn(view)
    item = filled(100, 80, 40, 20)  # exactly over the marked pixels
    view.add_item(item)
    covered = drawn(view, GREEN)
    assert drawn(view) is None  # nothing of the mark shows any more
    for edge in ("left", "right", "top", "bottom"):
        assert getattr(covered, edge) == pytest.approx(getattr(marked, edge), abs=1), edge
    wheel(view, marked.center, 3)  # it stays on its pixels at another zoom
    assert drawn(view) is None and drawn(view, GREEN).width > marked.width + 1
    view.fit()
    view.remove_item(item)
    assert drawn(view, GREEN) is None
    assert drawn(view) == marked


def test_an_added_item_does_not_change_what_fit_shows(window, qtbot, disk_clip):
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(100, 80, 140, 100)))
    marked = drawn(view)
    view.add_item(filled(900, 700, 50, 50))  # far outside the frame
    view.fit()
    assert drawn(view) == marked
