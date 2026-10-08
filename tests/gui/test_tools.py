"""The tools Stick, Tape, Circle and Axes (SPEC 4.2 to 4.5, 8.10, 10.1, 10.2; task C3): what a click
places, where it lands in the session, and what is drawn on the picture.

Expected values come from the spec and from geometry:
- a point is the clicked (u, v) in px of the video frame (SPEC 3.1), stored as SPEC 8.10 names it;
- the circle points are computed on the dish wall of the synthetic clip, whose true center and
  radius are in its scene (`dish_clip`: center (160.3, 119.8), radius 108 px), so the fit must
  return that center and radius with no residual;
- the frame's center is (width / 2, height / 2); for angle 0 the x arrow points right and the y
  arrow up on the screen, for 90 degrees x points up and y left (SPEC 3.2: counterclockwise);
- labels follow the design note: "Stick 30.00 mm", "R = 16.20 mm", "x", "y".
What is drawn is read from the grabbed view at image points worked out by geometry, away from any
label: no test depends on a font or on a pixel of text.
"""

import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from calibration_helpers import (LEFT, NO_KEY, RIGHT, click_at, fit_of, lit, on_circle, opened, outside_points,
                                 place, restart, screen_point)
from gui_helpers import ClickRecorder, StandInSource, drawn, picture
from outline_tracker import geometry

SMALL = (320, 240)
SIX = (0, 60, 120, 180, 240, 300)  # degrees: six points spread evenly around the wall
FIRST_END = "Stick: click the first end on the ruler."
SECOND_END = "Stick: click the second end."
ZOOM_IN = " Scroll to zoom in for a precise click."


def wall(clip, degrees=SIX):
    """Points on the true dish wall of a dish clip, px of the video frame."""
    cu, cv, radius = clip.scene.dish
    return on_circle((cu, cv), radius, degrees)


# ---------------------------------------------------------------------------------------------
# A tool is what the view takes


def test_each_tool_has_what_the_view_asks_of_a_tool(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip)
    for tool in (calibration.stick_tool, calibration.tape_tool, dish.circle_tool, dish.axes_tool):
        assert tool.cursor == Qt.CursorShape.CrossCursor
        assert callable(tool.click) and callable(tool.add_tool_point)
        window.view.set_tool(tool)
        assert window.view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
        assert window.tool_text.text() == tool.text != ""
    window.view.set_tool(None)


def test_the_tool_lines_are_the_design_notes(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip)
    view = window.view
    assert view.zoom > 1  # the small clip is drawn enlarged: no advice to zoom in
    view.set_tool(calibration.stick_tool)
    assert window.tool_text.text() == FIRST_END
    calibration.stick_tool.add_tool_point(50.5, 20.5)
    assert window.tool_text.text() == SECOND_END  # the line follows the tool while it is chosen
    calibration.stick_tool.add_tool_point(250.5, 20.5)
    view.set_tool(calibration.tape_tool)
    assert window.tool_text.text() == "Tape: click the first ruler mark."
    calibration.tape_tool.add_tool_point(60.5, 200.5)
    assert window.tool_text.text() == "Tape: click the second ruler mark."
    view.set_tool(dish.circle_tool)
    assert window.tool_text.text() == ("Circle: click points on the inner wall of the dish. 0 placed; "
                                       "6 or more is best.")
    place(dish.circle_tool, wall(dish_clip)[:2])
    assert "2 placed" in window.tool_text.text()
    view.set_tool(dish.axes_tool)
    assert window.tool_text.text() == "Axes: click the new origin."


def test_under_100_percent_the_precise_tools_say_to_zoom_in(window, qtbot, wide_clip):
    calibration, dish = opened(window, qtbot, wide_clip)
    view = window.view
    assert 0 < view.zoom < 1  # 1280 px do not fit beside the dock
    view.set_tool(calibration.stick_tool)
    assert window.tool_text.text() == FIRST_END + ZOOM_IN
    view.set_tool(dish.circle_tool)
    assert window.tool_text.text().endswith("6 or more is best." + ZOOM_IN)
    view.set_tool(dish.axes_tool)
    assert window.tool_text.text() == "Axes: click the new origin."  # the origin needs no such care
    view.set_tool(calibration.stick_tool)
    view.one_to_one()
    QApplication.processEvents()
    assert window.tool_text.text() == FIRST_END  # at 100% the advice goes


# ---------------------------------------------------------------------------------------------
# Clicks


def test_a_click_places_the_point_that_was_clicked(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    view, tool = window.view, dish.circle_tool
    view.set_tool(tool)
    scale = fit_of(view, SMALL)[0]
    click_at(qtbot, view, SMALL, 100.5, 80.5)
    click_at(qtbot, view, SMALL, 220.5, 30.5)
    assert len(tool.points) == 2
    for placed, aimed in zip(tool.points, [(100.5, 80.5), (220.5, 30.5)]):
        assert placed == pytest.approx(aimed, abs=1 / scale)  # within one screen px of the aim
    assert view.tool is tool  # a circle has no last point: the tool stays


def test_a_click_outside_the_image_places_nothing(window, qtbot, wide_clip):
    calibration, dish = opened(window, qtbot, wide_clip)
    view = window.view
    for tool in (calibration.stick_tool, dish.circle_tool, dish.axes_tool):
        view.set_tool(tool)
        for outside in outside_points(view, wide_clip.scene.size):  # on the canvas beside the picture
            qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, outside)
        assert view.tool is tool  # nothing was placed: the tool is still waiting for its click
        for u, v in ((-0.5, 100.0), (1280.0, 100.0), (100.0, -2.0), (100.0, 720.0), (math.nan, 5.0)):
            tool.add_tool_point(u, v)  # a point that is not on the frame, however it arrives
    session = window.controller.session
    assert calibration.stick_tool.points == [] and dish.circle_tool.points == []
    assert session.calibration.stick is None and session.circle is None
    assert session.axes.origin_px == [640.0, 360.0]  # still the frame's center
    assert not dish.axes_tool.placed


def test_only_the_left_button_places_a_point(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    tool = dish.circle_tool
    window.view.set_tool(tool)
    click_at(qtbot, window.view, SMALL, 100.5, 80.5, button=RIGHT)
    tool.click(120.5, 90.5, Qt.MouseButton.MiddleButton, NO_KEY)
    assert tool.points == []
    tool.click(120.5, 90.5, LEFT, NO_KEY)
    assert tool.points == [[120.5, 90.5]]


def test_of_a_double_click_the_first_click_counts_and_the_second_is_dropped(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    view, viewport = window.view, window.view.viewport()

    def double_click(u, v):
        at = screen_point(view, SMALL, u, v)
        qtbot.mouseClick(viewport, LEFT, NO_KEY, at)
        qtbot.mouseDClick(viewport, LEFT, NO_KEY, at)
        qtbot.mouseRelease(viewport, LEFT, NO_KEY, at)
        QApplication.processEvents()

    recorder = ClickRecorder()
    view.set_tool(recorder)
    double_click(60.5, 50.5)
    assert len(recorder.clicks) == 2  # the view does hand a double click on as two clicks

    tool = dish.circle_tool
    view.set_tool(tool)
    double_click(100.5, 80.5)
    assert len(tool.points) == 1
    assert tool.points[0] == pytest.approx([100.5, 80.5], abs=1.0)
    click_at(qtbot, view, SMALL, 200.5, 150.5)  # the next single click counts again
    assert len(tool.points) == 2
    double_click(150.5, 200.5)
    assert len(tool.points) == 3


# ---------------------------------------------------------------------------------------------
# Stick and tape


def test_the_two_stick_ends_land_in_the_session_and_the_tool_returns_to_pan(window, qtbot, dish_clip):
    calibration, _ = opened(window, qtbot, dish_clip)
    view, tool, session = window.view, calibration.stick_tool, window.controller.session
    changes = []
    window.controller.session_changed.connect(lambda: changes.append(session.calibration.stick))
    view.set_tool(tool)
    tool.add_tool_point(50.5, 20.5)
    assert tool.points == [[50.5, 20.5]]
    assert session.calibration.stick is None  # one end is no stick yet
    assert view.tool is tool
    tool.add_tool_point(250.5, 60.5)
    assert session.calibration.stick == {"p1_px": [50.5, 20.5], "p2_px": [250.5, 60.5], "length_mm": 30.0}
    assert view.tool is None  # finished: back to Pan
    assert changes and changes[-1] == session.calibration.stick  # the session was touched


def test_a_third_stick_click_starts_the_stick_again(window, qtbot, dish_clip):
    calibration, _ = opened(window, qtbot, dish_clip)
    tool, session = calibration.stick_tool, window.controller.session
    place(tool, [(50.5, 20.5), (250.5, 20.5)])
    window.view.set_tool(tool)
    tool.add_tool_point(80.5, 200.5)
    assert tool.points == [[80.5, 200.5]]
    assert session.calibration.stick is None  # the old stick is gone, the new one has one end
    assert window.view.tool is tool
    tool.add_tool_point(280.5, 200.5)
    assert session.calibration.stick["p1_px"] == [80.5, 200.5]
    assert session.calibration.stick["p2_px"] == [280.5, 200.5]


def test_a_second_end_on_the_first_one_is_refused_in_geometrys_words(window, qtbot, dish_clip):
    calibration, _ = opened(window, qtbot, dish_clip)
    tool = calibration.stick_tool
    with pytest.raises(ValueError) as same:
        geometry.stick_scale((50.5, 20.5), (50.5, 20.5), 30.0)
    place(tool, [(50.5, 20.5), (50.5, 20.5)])
    assert tool.points == [[50.5, 20.5]]  # the first end stays, the second is not taken
    assert tool.refusal == str(same.value)
    assert window.controller.session.calibration.stick is None
    tool.add_tool_point(250.5, 20.5)
    assert tool.refusal is None and window.controller.session.calibration.stick is not None


def test_the_tape_needs_the_stick(window, qtbot, dish_clip):
    calibration, _ = opened(window, qtbot, dish_clip)
    tape, session = calibration.tape_tool, window.controller.session
    place(tape, [(60.5, 200.5), (260.5, 200.5)])
    assert tape.points == [] and session.calibration.check is None  # no scale yet: nothing to check
    place(calibration.stick_tool, [(50.5, 20.5), (250.5, 20.5)])
    window.view.set_tool(tape)
    place(tape, [(60.5, 200.5), (260.5, 200.5)])
    assert session.calibration.check == {"p1_px": [60.5, 200.5], "p2_px": [260.5, 200.5], "true_mm": 20.0}
    assert window.view.tool is None
    # the stick is started again while the tape tool is chosen: without a scale the tool has no use
    window.view.set_tool(tape)
    calibration.stick_tool.add_tool_point(10.5, 10.5)
    assert window.view.tool is None


# ---------------------------------------------------------------------------------------------
# Circle and axes


def test_six_points_on_the_wall_give_the_dishs_center_and_radius(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    cu, cv, radius = dish_clip.scene.dish
    assert (cu, cv, radius) == (160.3, 119.8, 108.0)
    points = wall(dish_clip)
    place(dish.circle_tool, points)
    circle = window.controller.session.circle
    assert circle.points_px == points
    assert circle.center_px == pytest.approx([cu, cv], abs=1e-6)
    assert circle.radius_px == pytest.approx(radius, abs=1e-6)
    assert circle.rms_px == pytest.approx(0.0, abs=1e-6)
    assert all(isinstance(value, float) for value in (*circle.center_px, circle.radius_px, circle.rms_px))


def test_fewer_than_three_points_fit_nothing_and_say_how_many_are_needed(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    tool, session = dish.circle_tool, window.controller.session
    for count, point in enumerate(wall(dish_clip)[:2], start=1):
        tool.add_tool_point(*point)
        assert session.circle is None and len(tool.points) == count
        assert tool.refusal == "The circle needs 3 or more points. Click more points on the dish wall."
    tool.add_tool_point(*wall(dish_clip)[2])
    assert session.circle is not None and tool.refusal is None


def test_three_points_on_a_line_give_the_message_of_fit_circle(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    line = [[40.5, 50.5], [140.5, 100.5], [240.5, 150.5]]
    with pytest.raises(ValueError) as on_a_line:
        geometry.fit_circle(line)
    place(dish.circle_tool, line)
    assert window.controller.session.circle is None
    assert dish.circle_tool.refusal == str(on_a_line.value)
    dish.circle_tool.add_tool_point(60.5, 200.5)  # a point off the line: now a circle fits
    assert window.controller.session.circle is not None and dish.circle_tool.refusal is None


def test_the_origin_starts_at_the_frames_center_and_follows_the_circle_until_it_is_placed(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    session, axes, circle = window.controller.session, dish.axes_tool, dish.circle_tool
    cu, cv, radius = dish_clip.scene.dish
    assert session.axes.origin_px == [160.0, 120.0] and session.axes.angle_deg == 0.0
    assert not axes.placed
    place(circle, wall(dish_clip))
    assert session.axes.origin_px == pytest.approx([cu, cv], abs=1e-6)  # the circle's center
    circle.redo()
    assert session.axes.origin_px == [160.0, 120.0]  # no circle: the frame's center again

    window.view.set_tool(axes)
    axes.add_tool_point(100.5, 50.5)
    assert session.axes.origin_px == [100.5, 50.5] and axes.placed
    assert window.view.tool is None  # one click places the origin: back to Pan
    place(circle, wall(dish_clip))
    assert session.axes.origin_px == [100.5, 50.5]  # an origin placed by hand stays
    circle.redo()
    assert session.axes.origin_px == [100.5, 50.5]


def test_origin_to_center_copies_the_center_and_then_the_origin_stays(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    session, axes, circle = window.controller.session, dish.axes_tool, dish.circle_tool
    cu, cv, radius = dish_clip.scene.dish
    axes.origin_to_center()
    assert session.axes.origin_px == [160.0, 120.0] and not axes.placed  # no circle: nothing to copy
    axes.add_tool_point(100.5, 50.5)
    place(circle, wall(dish_clip))
    axes.origin_to_center()
    assert session.axes.origin_px == pytest.approx([cu, cv], abs=1e-6)
    assert session.axes.origin_px is not session.circle.center_px  # a copy, not the same list
    # the circle is clicked again somewhere else: the origin was put where it is, and stays
    circle.redo()
    place(circle, on_circle((150.0, 110.0), 90.0, SIX))
    assert session.circle.center_px == pytest.approx([150.0, 110.0], abs=1e-6)
    assert session.axes.origin_px == pytest.approx([cu, cv], abs=1e-6)


def test_the_angle_is_typed_in_degrees_and_must_be_a_number(window, qtbot, dish_clip):
    _, dish = opened(window, qtbot, dish_clip)
    session, axes = window.controller.session, dish.axes_tool
    axes.set_angle(30.0)
    assert session.axes.angle_deg == 30.0
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="angle"):
            axes.set_angle(bad)
    assert session.axes.angle_deg == 30.0


# ---------------------------------------------------------------------------------------------
# Redo


def test_redo_clears_the_points_of_a_tool(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip, dark_frame=True)
    session = window.controller.session
    place(calibration.stick_tool, [(50.5, 20.5), (250.5, 20.5)])
    place(calibration.tape_tool, [(60.5, 200.5), (260.5, 200.5)])
    place(dish.circle_tool, wall(dish_clip))
    assert lit(window.view, SMALL, 100.5, 20.5) and lit(window.view, SMALL, 110.5, 200.5)

    calibration.tape_tool.redo()
    assert session.calibration.check is None and calibration.tape_tool.points == []
    assert session.calibration.stick is not None  # only the tape
    assert not lit(window.view, SMALL, 110.5, 200.5) and lit(window.view, SMALL, 100.5, 20.5)
    calibration.stick_tool.redo()
    assert session.calibration.stick is None and calibration.stick_tool.points == []
    assert not lit(window.view, SMALL, 100.5, 20.5)
    dish.circle_tool.redo()
    assert session.circle is None and dish.circle_tool.points == []
    assert dish.circle_tool.refusal is None  # nothing is asked of a circle that was cleared
    for tool in (calibration.stick_tool, calibration.tape_tool, dish.circle_tool):
        assert tool.graphic.items == [] and tool.graphic.labels == []


def test_redo_of_one_pending_point_clears_it(window, qtbot, dish_clip):
    calibration, dish = opened(window, qtbot, dish_clip)
    calibration.stick_tool.add_tool_point(50.5, 20.5)
    dish.circle_tool.add_tool_point(60.5, 60.5)
    calibration.stick_tool.redo()
    dish.circle_tool.redo()
    assert calibration.stick_tool.points == [] and dish.circle_tool.points == []
    assert calibration.stick_tool.graphic.items == [] and dish.circle_tool.graphic.items == []


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
