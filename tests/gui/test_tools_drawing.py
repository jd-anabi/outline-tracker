"""The tools Stick, Tape, Circle and Axes (SPEC 4.2 to 4.5, 8.10, 10.1, 10.2; task C3): what is drawn on the
picture, also after another video or a session is opened.

What a click places, and where it lands in the session, is in tests/gui/test_tools.py.

Expected values come from the spec and from geometry:
- the circle points are computed on the dish wall of the synthetic clip, whose true center and
  radius are in its scene (`dish_clip`: center (160.3, 119.8), radius 108 px), so the fit must
  return that center and radius with no residual;
- the frame's center is (width / 2, height / 2); for angle 0 the x arrow points right and the y
  arrow up on the screen, for 90 degrees x points up and y left (SPEC 3.2: counterclockwise);
- labels follow the design note: "Stick 30.00 mm", "R = 16.20 mm", "x", "y".
What is drawn is read from the grabbed view at image points worked out by geometry, away from any
label: no test depends on a font or on a pixel of text.
"""

import pytest
from PySide6.QtWidgets import QApplication

from calibration_helpers import LEFT, SMALL, lit, on_circle, opened, place, restart, wall
from gui_helpers import StandInSource, drawn, fit_of, picture


# ---------------------------------------------------------------------------------------------
# What is drawn


def test_the_stick_and_the_tape_are_drawn_between_their_ends(window, qtbot, dish_clip):
    calibration, _ = opened(window, qtbot, dish_clip, dark_frame=True)
    view, stick, tape = window.view, calibration.stick_tool, calibration.tape_tool
    assert not lit(view, SMALL, 100.5, 20.5)
    stick.add_tool_point(50.5, 20.5)
    assert lit(view, SMALL, 50.5, 20.5)  # the first end is marked at once
    assert not lit(view, SMALL, 100.5, 20.5) and stick.graphic.labels == []
    stick.add_tool_point(250.5, 20.5)  # 200 px for 30 mm: 0.15 mm per px
    for u in (50.5, 100.5, 200.5, 250.5):  # the ends, and a quarter in from each (the label is at the middle)
        assert lit(view, SMALL, u, 20.5), u
    assert not lit(view, SMALL, 100.5, 60.5) and not lit(view, SMALL, 280.5, 20.5)
    assert stick.graphic.labels == ["Stick 30.00 mm"]

    place(tape, [(60.5, 100.5), (60.5, 220.5)])  # 120 px: 18.00 mm at 0.15 mm per px
    assert lit(view, SMALL, 60.5, 100.5) and lit(view, SMALL, 60.5, 220.5)
    assert sum(lit(view, SMALL, 60.5, v) for v in range(105, 216, 2)) >= 20  # a dashed line: most of it shows
    assert tape.graphic.labels == ["Tape 18.00 mm"]
    calibration.set_stick_length(60.0)  # the labels follow the session
    assert stick.graphic.labels == ["Stick 60.00 mm"] and tape.graphic.labels == ["Tape 36.00 mm"]


def test_the_circle_points_and_the_fitted_circle_are_drawn(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip, dark_frame=True)
    view, tool = window.view, dish.circle_tool
    cu, cv, radius = dish_clip.scene.dish
    points = wall(dish_clip)
    between = on_circle((cu, cv), radius, (30, 150, 210, 330))  # on the wall, where nobody clicked
    place(tool, points[:2])
    assert all(lit(view, SMALL, u, v, within=5) for u, v in points[:2])  # a hollow square around each click
    assert not any(lit(view, SMALL, u, v, within=1) for u, v in points[:2])  # hollow: nothing on the point itself
    assert not any(lit(view, SMALL, u, v) for u, v in between)  # no circle yet
    assert tool.graphic.labels == []
    place(tool, points[2:])
    assert all(lit(view, SMALL, u, v) for u, v in between)  # the fitted circle
    inside = on_circle((cu, cv), radius - 30, (30, 150, 210, 330))
    assert not any(lit(view, SMALL, u, v) for u, v in inside)
    assert tool.graphic.labels == ["R = 108.0 px"]  # no scale yet: px
    place(calibration.stick_tool, [(50.5, 20.5), (250.5, 20.5)])  # 0.15 mm per px
    assert tool.graphic.labels == ["R = 16.20 mm"]


@pytest.mark.parametrize("angle_deg, x_goes, y_goes", [(0.0, (1, 0), (0, -1)), (90.0, (0, -1), (-1, 0)),
                                                       (180.0, (-1, 0), (0, 1))])
def test_the_axes_are_two_arrows_at_the_origin_with_y_up_for_angle_0(window, qtbot, dish_clip, angle_deg, x_goes,
                                                                     y_goes):
    _, dish = opened(window, qtbot, dish_clip, dark_frame=True)
    view, axes = window.view, dish.axes_tool
    assert axes.graphic.labels == ["x", "y"]  # drawn from the moment the video is open
    axes.add_tool_point(150.5, 110.5)
    axes.set_angle(angle_deg)
    length = axes.arrow_length_px
    assert 20 <= length <= 100
    half = length / 2
    for du, dv in (x_goes, y_goes):  # half way along each arrow (du, dv: on the picture, v down)
        assert lit(view, SMALL, 150.5 + half * du, 110.5 + half * dv), (du, dv)
    others = {(1, 0), (-1, 0), (0, 1), (0, -1)} - {x_goes, y_goes}
    for du, dv in others:
        assert not lit(view, SMALL, 150.5 + half * du, 110.5 + half * dv), (du, dv)
    assert axes.graphic.labels == ["x", "y"]


# ---------------------------------------------------------------------------------------------
# Another video, and the same session again


def test_opening_another_video_returns_to_pan_and_clears_the_graphics(window, qtbot, dish_clip, disk_clip):
    calibration, dish = opened(window, qtbot, dish_clip)
    place(calibration.stick_tool, [(50.5, 20.5), (250.5, 20.5)])
    place(calibration.tape_tool, [(60.5, 200.5), (260.5, 200.5)])
    place(dish.circle_tool, wall(dish_clip))
    calibration.stick_tool.add_tool_point(10.5, 10.5)  # a pending end, too
    dish.axes_tool.add_tool_point(100.5, 50.5)
    window.view.set_tool(dish.circle_tool)

    window.open_path(disk_clip.path)
    QApplication.processEvents()
    session = window.controller.session
    assert window.view.tool is None
    assert session.calibration.stick is None and session.circle is None
    for tool in (calibration.stick_tool, calibration.tape_tool, dish.circle_tool):
        assert tool.points == [] and tool.graphic.items == [] and tool.graphic.labels == []
    assert session.axes.origin_px == [160.0, 120.0] and not dish.axes_tool.placed  # the new frame's center
    assert dish.axes_tool.graphic.labels == ["x", "y"]


def test_the_graphics_are_drawn_again_from_a_session_that_is_opened(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip)
    cu, cv, radius = dish_clip.scene.dish
    place(calibration.stick_tool, [(50.5, 20.5), (250.5, 20.5)])
    place(calibration.tape_tool, [(60.5, 200.5), (260.5, 200.5)])
    place(dish.circle_tool, wall(dish_clip))
    dish.axes_tool.add_tool_point(100.5, 50.5)
    dish.axes_tool.set_angle(90.0)
    before = window.controller.session

    again = restart(window)
    assert window.controller.session is again and again is not before
    assert again.to_json() == before.to_json()  # opening changed nothing: the origin is the session's own
    assert calibration.stick_tool.points == [[50.5, 20.5], [250.5, 20.5]]
    assert calibration.tape_tool.points == [[60.5, 200.5], [260.5, 200.5]]
    assert dish.circle_tool.points == wall(dish_clip)
    assert dish.axes_tool.placed
    assert calibration.stick_tool.graphic.labels == ["Stick 30.00 mm"]
    assert calibration.tape_tool.graphic.labels == ["Tape 30.00 mm"]
    assert dish.circle_tool.graphic.labels == ["R = 16.20 mm"]
    assert dish.axes_tool.graphic.labels == ["x", "y"]
    # a point placed now goes into the session that was opened, not into the one before it
    dish.circle_tool.add_tool_point(*on_circle((cu, cv), radius, (90,))[0])
    assert len(again.circle.points_px) == 7 and len(before.circle.points_px) == 6


def test_a_change_made_elsewhere_in_the_session_is_drawn(window, qtbot, dish_clip):
    # whoever changes the session and touches it sees the picture follow (an opened session, an undo)
    calibration, dish = opened(window, qtbot, dish_clip, dark_frame=True)
    view, session = window.view, window.controller.session
    session.calibration.stick = {"p1_px": [50.5, 20.5], "p2_px": [250.5, 20.5], "length_mm": 12.5}
    window.controller.touch()
    assert lit(view, SMALL, 100.5, 20.5)
    assert calibration.stick_tool.graphic.labels == ["Stick 12.50 mm"]
    session.calibration.stick = None
    window.controller.touch()
    assert not lit(view, SMALL, 100.5, 20.5) and calibration.stick_tool.graphic.labels == []


# ---------------------------------------------------------------------------------------------
# 1:1 and Fit, with the axes on the picture


def test_one_to_one_draws_one_video_pixel_on_one_screen_pixel_beside_the_axes(window, qtbot, disk_clip):
    # the block of pixels that is read here is right of and below the frame's center, where no
    # arrow and no label of the axes is
    view = picture(window, qtbot, disk_clip, StandInSource((320, 240), marked=(190, 140, 210, 150)))
    assert [window.fit_button.text(), window.one_to_one_button.text()] == ["Fit", "1:1"]
    qtbot.mouseClick(window.one_to_one_button, LEFT)
    seen = drawn(view)
    assert seen.solid and (seen.width, seen.height) == (20, 10)  # the 20 x 10 px that are marked
    assert view.zoom == pytest.approx(1.0)
    assert window.zoom_label.text() == "100%"
    # the middle of the frame stays in the middle of the view: the marked block is around (200, 145),
    # 40 px right of the frame's middle and 25 px below it
    room = view.viewport().size()
    assert (seen.left + seen.right) / 2 == pytest.approx(room.width() / 2 + 40, abs=1)
    assert (seen.top + seen.bottom) / 2 == pytest.approx(room.height() / 2 + 25, abs=1)
    qtbot.mouseClick(window.fit_button, LEFT)
    scale, left, top = fit_of(view, (320, 240))
    seen = drawn(view)
    assert seen.left == pytest.approx(left + 190 * scale, abs=1)
    assert seen.width == pytest.approx(20 * scale, abs=1)
