"""The status bar's read-outs (SPEC 3.1, 3.2, 4.6, 10.1; task C8a): where the cursor is on the
picture in px and in mm, the gray value under it, and the device of the model.

Expected values come from the spec and from geometry:
- pixel (c, r) of a frame is the square [c, c+1) x [r, r+1), so the cursor in its middle is at
  (u, v) = (c + 0.5, r + 0.5) (SPEC 3.1). The stand-in frame is 8 x 6 px, so one video px covers
  about 70 screen px and the middle of a pixel is met to a few hundredths of a px;
- (x, y) = k J (u - u0, v - v0) with y up (SPEC 3.2): with a stick of 4 px that is 2 mm long,
  k = 0.5 mm/px; with the origin at (2, 5) px and the axes not turned, the point (5.5, 3.5) px is
  at x = 0.5 * 3.5 = 1.75 mm and y = -0.5 * (3.5 - 5) = +0.75 mm (above the origin: y up); with
  +x turned 90 degrees counterclockwise on screen it is at x = 0.75 mm and y = -1.75 mm;
- gray = 0.299 R + 0.587 G + 0.114 B (SPEC 4.6): 105 for the marked pixel (255, 0, 255), and the
  stand-in frame 0 is gray 60 in even rows and 64 in odd rows.
No test reads a pixel of text: the texts are compared, and the widths with the labels' own fonts.
"""

import math

import pytest
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from finish_helpers import GRAY_OF_MARK, hover, move_mouse, number, small_picture
from gui_helpers import StandInSource, picture, show
from outline_tracker.gui import status_bar
from outline_tracker.segmenter.fake import ThresholdFake
from prompt_helpers import objects_panel

LONGEST = {"pixel_label": "Pixel 0000.0, 0000.0 px", "position_label": "Position −0000.00, −0000.00 mm",
           "gray_label": "Gray 000", "device_label": "Device cuda"}


def labels(window):
    readouts = window.readouts
    return readouts.pixel_label, readouts.position_label, readouts.gray_label, readouts.device_label


def texts(window) -> list[str]:
    """What the three read-outs of the cursor say: pixel, position, gray."""
    return [label.text() for label in labels(window)[:3]]


def with_scale(window, angle_deg: float = 0.0) -> None:
    """Give the session a stick of 4 px that is 2 mm long (0.5 mm per px), the origin at (2, 5) px
    and the axes turned by `angle_deg`."""
    session = window.controller.session
    session.calibration.stick = {"p1_px": [1.0, 1.0], "p2_px": [5.0, 1.0], "length_mm": 2.0}
    session.axes.origin_px, session.axes.angle_deg = [2.0, 5.0], angle_deg
    window.controller.touch()


# ---------------------------------------------------------------------------------------------
# The cursor on the picture


@pytest.mark.parametrize("column, row", [(0, 0), (5, 3), (7, 5)])
def test_the_cursor_in_the_middle_of_a_pixel_reads_its_center_and_its_gray_value(window, qtbot, disk_clip,
                                                                                 column, row):
    view = small_picture(window, qtbot, disk_clip, column, row)
    assert texts(window) == ["", "", ""]  # the mouse is not over the picture yet
    hover(view)
    u, v = view.cursor_px
    assert (math.floor(u), math.floor(v)) == (column, row)  # no extra half pixel
    assert abs(u - (column + 0.5)) <= 0.04 and abs(v - (row + 0.5)) <= 0.04
    assert window.readouts.pixel_label.text() == f"Pixel {column}.5, {row}.5 px"
    assert round(GRAY_OF_MARK) == 105
    assert window.readouts.gray_label.text() == "Gray 105"  # of the marked pixel (255, 0, 255)


def test_the_gray_value_is_the_one_of_the_pixel_under_the_cursor(window, qtbot, disk_clip):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    hover(view, shift=(-1, 0))  # the pixel at the left of the marked one: row 3 is odd, gray 64
    assert window.readouts.pixel_label.text() == "Pixel 4.5, 3.5 px"
    assert window.readouts.gray_label.text() == "Gray 64"
    hover(view, shift=(0, -1))  # the pixel above it: row 2 is even, gray 60
    assert window.readouts.pixel_label.text() == "Pixel 5.5, 2.5 px"
    assert window.readouts.gray_label.text() == "Gray 60"


def test_the_gray_value_follows_the_frame_under_a_cursor_that_stays(window, qtbot, disk_clip):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    hover(view, shift=(-1, -1))  # pixel (4, 2), an even row: gray 60 + the frame's number
    assert window.readouts.gray_label.text() == "Gray 60"
    view.show_frame(3)
    assert window.readouts.gray_label.text() == "Gray 63"
    assert window.readouts.pixel_label.text() == "Pixel 4.5, 2.5 px"


def test_without_a_scale_the_position_says_so(window, qtbot, disk_clip):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    hover(view)
    assert window.readouts.position_label.text() == "Position – (no scale)"


@pytest.mark.parametrize("angle_deg, x_mm, y_mm", [(0.0, 1.75, 0.75), (90.0, 0.75, -1.75)])
def test_with_a_stick_and_axes_the_position_is_the_world_frames_mm_with_y_up(window, qtbot, disk_clip,
                                                                             angle_deg, x_mm, y_mm):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    with_scale(window, angle_deg)
    hover(view)
    u, v = view.cursor_px
    x, y = window.controller.session.world_frame().to_world(u, v)
    # by hand for the pixel's middle (the module's text); the cursor is within 0.04 px of it, 0.02 mm
    assert abs(x - x_mm) <= 0.02 and abs(y - y_mm) <= 0.02
    assert window.readouts.position_label.text() == f"Position {number(x, 2)}, {number(y, 2)} mm"
    assert ("−" in window.readouts.position_label.text()) == (y_mm < 0)  # U+2212, never a hyphen
    assert "-" not in window.readouts.position_label.text()


def test_a_scale_that_is_set_while_the_cursor_stays_shows_at_once(window, qtbot, disk_clip):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    hover(view)
    with_scale(window)
    x, y = window.controller.session.world_frame().to_world(*view.cursor_px)
    assert window.readouts.position_label.text() == f"Position {number(x, 2)}, {number(y, 2)} mm"


@pytest.mark.parametrize("value, decimals, text", [(12.34, 2, "12.34"), (-5.67, 2, "−5.67"), (-0.006, 2, "−0.01"),
                                                   (-0.004, 2, "0.00"), (-0.0, 2, "0.00"), (0.0, 2, "0.00"),
                                                   (-1234.5, 1, "−1234.5")])
def test_a_number_is_written_with_the_minus_sign_and_zero_has_no_sign(value, decimals, text):
    assert status_bar.signed(value, decimals) == text  # U+2212; a point on an axis is at 0.00, not at −0.00


def test_outside_the_picture_the_read_outs_are_empty(window, qtbot, disk_clip):
    # a frame four times as wide as high leaves room above and below it in the view
    view = picture(window, qtbot, disk_clip, StandInSource((400, 100), marked=(190, 40, 210, 60)))
    with_scale(window)
    viewport = view.viewport()
    middle = QPoint(viewport.width() // 2, viewport.height() // 2)
    move_mouse(view, middle)
    assert all(texts(window)) and view.cursor_px is not None
    move_mouse(view, QPoint(viewport.width() // 2, 3))  # in the view, above the picture
    assert view.cursor_px is None and texts(window) == ["", "", ""]
    move_mouse(view, middle)
    assert all(texts(window))
    QApplication.sendEvent(viewport, QEvent(QEvent.Type.Leave))  # the mouse leaves the view
    assert view.cursor_px is None and texts(window) == ["", "", ""]


def test_before_a_video_is_open_the_read_outs_are_empty(window, qtbot):
    show(window, qtbot)
    assert window.view.cursor_px is None
    assert texts(window) == ["", "", ""]


# ---------------------------------------------------------------------------------------------
# Nothing jumps; the order; the last message


def test_the_read_outs_have_tabular_digits_and_room_for_their_longest_text(window, qtbot, disk_clip, qapp):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    qtbot.waitUntil(lambda: objects_panel(window).worker.ready)
    QApplication.processEvents()  # the model's label at the right has its last text, and its last width
    for name, longest in LONGEST.items():
        label = getattr(window.readouts, name)
        assert label.font().featureValue(QFont.Tag("tnum")) == 1, name  # digits of one width
        assert label.font().pointSizeF() == qapp.font().pointSizeF(), name  # the base size, as the message
        assert label.minimumWidth() >= label.fontMetrics().horizontalAdvance(longest), name
    empty = [label.geometry() for label in labels(window)]
    with_scale(window)
    hover(view)
    assert all(texts(window))
    assert [label.geometry() for label in labels(window)] == empty  # nothing moved when the texts came


def test_the_read_outs_are_at_the_right_in_the_notes_order_before_the_model_state(window, qtbot):
    window.resize(window.minimumSize())  # 960 px wide: the smallest window
    show(window, qtbot)
    bar = window.statusBar()
    row = [*labels(window), objects_panel(window).model_label]
    lefts = [part.mapTo(bar, QPoint(0, 0)).x() for part in row]
    assert all(part.isVisible() for part in row)
    for part, left, next_left in zip(row, lefts, lefts[1:]):
        assert left + part.width() <= next_left, part.text()  # pixel, position, gray, device, model
    assert lefts[0] > 0 and lefts[-1] + row[-1].width() <= bar.width()
    assert window.width() == 960  # the read-outs do not make the smallest window wider


def test_the_last_message_stays_while_the_cursor_moves(window, qtbot, disk_clip):
    view = small_picture(window, qtbot, disk_clip, 5, 3)
    window.statusBar().showMessage("The session is saved.")
    hover(view)
    hover(view, shift=(-1, 0))
    assert window.statusBar().currentMessage() == "The session is saved."


# ---------------------------------------------------------------------------------------------
# The device


class OnCpu(ThresholdFake):
    """A stand-in that says which device it runs on, as the real segmenter does."""

    device = "cpu"


def test_the_device_is_the_loaded_models(window, qtbot):
    window.segmenter_factory = lambda model, device: OnCpu()
    assert window.readouts.device_label.text() == "Device –"  # no model is loaded yet
    show(window, qtbot)
    qtbot.waitUntil(lambda: objects_panel(window).worker.ready)
    assert window.readouts.device_label.text() == "Device cpu"
    assert objects_panel(window).model_label.text() == "Model ready"  # the one place for the model's state


def test_a_model_that_names_no_device_leaves_the_dash(window, qtbot):
    show(window, qtbot)
    qtbot.waitUntil(lambda: objects_panel(window).worker.ready)
    assert window.readouts.device_label.text() == "Device –"
