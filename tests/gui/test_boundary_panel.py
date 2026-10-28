"""Panel 4, Boundary and axes (SPEC 4.4, 4.5, 8.10, 10.1; task C3; decision 8 of docs/ROADMAP.md): the
boundary's circle and its fit, its diameter, Origin to Center, the axes angle, "Crop to boundary for tracking".
A boundary is optional: with a video open the panel is done without one (decisions 8, 43), and the crop box
is off and disabled until there is one (decision 42).

Expected values come from the synthetic dish clip and from geometry, never from the program:
- the dish wall of `dish_clip` is the circle around (160.3, 119.8) px with radius 108 px (its
  scene), so six points computed on it fit that circle with an RMS of 0;
- a stick of 200 px for 30 mm is 0.15 mm per px, so R = 108 px is 16.20 mm and 2R is 32.40 mm;
- the frame is 320 x 240 px, so its center is (160.0, 120.0).
The wording is the design note's and decision 44's.
"""

import math
import shutil

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from calibration_helpers import SIX, body, on_circle, opened, place, restart, wall
from outline_tracker.core import geometry
from outline_tracker.gui.settings import MANIFEST_KEY, settings

START_HINT = "A boundary is optional: click Circle, then 6 or more points on its edge, spread around it."
OPTIONAL = "A boundary is optional: click Circle, then 6 or more points on its edge."
AT_FRAME_CENTER = f"{OPTIONAL} The origin is at 160.0, 120.0 px."
NEEDS_THREE = "The circle needs 3 or more points. Click more points on the boundary's edge."
STICK = [(50.5, 20.5), (250.5, 20.5)]  # 200 px: with 30 mm, 0.15 mm per px


# ---------------------------------------------------------------------------------------------
# Before a video, and what the panel holds


def test_without_a_video_the_panel_waits(window):
    panel = body(window, 4)
    assert not panel.isEnabled()
    assert window.panels[3].state == "todo" and window.panels[3].hint.text() == START_HINT


def test_the_controls_name_the_boundary_and_the_panel_is_done_at_once(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    assert panel.isEnabled()
    assert [panel.circle_button.text(), panel.circle_redo.text()] == ["Circle", "Redo"]
    assert [panel.origin_button.text(), panel.axes_button.text()] == ["Origin to Center", "Axes"]
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert "Boundary" in labels and "Diameter" in labels and not [text for text in labels if "Dish" in text]
    assert panel.circle_button.toolTip() == "Click points on the edge of the boundary"
    assert panel.diameter_box.toolTip() == "The boundary's diameter, if you know it: 2R should be close to it"
    assert panel.crop_box.text() == "Crop to boundary for tracking"
    assert panel.crop_box.toolTip() == ("Track only inside the square around the boundary: faster, and sharper on "
                                        "small objects")
    # no boundary: the box is off and disabled, and the setting keeps its default (decision 42)
    assert not panel.crop_box.isChecked() and not panel.crop_box.isEnabled()
    assert window.controller.session.processing.boundary_crop is True
    assert panel.circle_button.isCheckable() and panel.axes_button.isCheckable()
    assert not panel.origin_button.isEnabled()  # no circle yet: no center to go to
    assert panel.angle_box.suffix() == "°" and panel.angle_box.decimals() == 1 and panel.angle_box.value() == 0.0
    assert panel.diameter_box.suffix() == " mm" and panel.diameter_box.decimals() == 2
    assert panel.diameter_box.value() == 0.0 and panel.diameter_box.specialValueText() == "not known"
    for box in (panel.angle_box, panel.diameter_box):
        assert not box.keyboardTracking() and box.alignment() & Qt.AlignmentFlag.AlignRight
    for button in (panel.circle_button, panel.circle_redo, panel.origin_button, panel.axes_button):
        assert not button.autoDefault() and button.toolTip() != ""
    assert [panel.center_value.text(), panel.radius_value.text(), panel.rms_value.text(),
            panel.two_r_value.text()] == ["–"] * 4
    assert panel.origin_value.text() == "160.0, 120.0 px"  # the frame's center
    assert window.panels[3].state == "done" and window.panels[3].hint.text() == AT_FRAME_CENTER  # decision 8
    assert panel.message.isHidden()


# ---------------------------------------------------------------------------------------------
# The circle


def test_six_points_on_the_wall_show_the_center_the_radius_and_the_rms(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    assert panel.center_value.text() == "160.3, 119.8 px"
    assert panel.radius_value.text() == "108.0 px"  # no scale yet: px only
    assert panel.rms_value.text() == "0.0 px"
    assert panel.two_r_value.text() == "–"
    assert panel.origin_value.text() == "160.3, 119.8 px"  # the origin went to the center with the fit
    assert window.panels[3].state == "done"
    assert window.panels[3].hint.text() == "Boundary: R = 108.0 px, RMS 0.0 px. The origin is at the center."
    assert panel.message.isHidden() and panel.origin_button.isEnabled()


def test_with_a_scale_the_radius_is_in_mm_too_and_2r_is_shown(window, qtbot, dish_clip):
    calibration, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    place(calibration.stick_tool, STICK)  # the scale arrives after the circle: the panel follows
    assert panel.radius_value.text() == "16.20 mm (108.0 px)"
    assert panel.two_r_value.text() == "32.40 mm"
    assert window.panels[3].hint.text() == ("Boundary: R = 16.20 mm (108.0 px), RMS 0.0 px. "
                                            "The origin is at the center.")
    calibration.set_stick_length(60.0)
    assert panel.radius_value.text() == "32.40 mm (108.0 px)" and panel.two_r_value.text() == "64.80 mm"


def test_points_off_the_circle_show_their_rms(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    cu, cv, radius = dish_clip.scene.dish
    # alternately 2 px outside and 2 px inside the wall, evenly spread: by symmetry the fit is the
    # wall itself, and every point is 2 px from it
    points = [on_circle((cu, cv), radius + (2 if index % 2 else -2), (angle,))[0] for index, angle in enumerate(SIX)]
    place(panel.circle_tool, points)
    circle = window.controller.session.boundary
    assert circle.center_px == pytest.approx([cu, cv], abs=1e-6) and circle.radius_px == pytest.approx(radius)
    assert circle.rms_px == pytest.approx(2.0, abs=1e-6)
    assert panel.rms_value.text() == "2.0 px"


def test_fewer_than_three_points_need_attention_and_say_how_many_are_needed(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    for point in wall(dish_clip)[:2]:
        panel.circle_tool.add_tool_point(*point)
        assert window.panels[3].state == "attention" and window.panels[3].hint.text() == NEEDS_THREE
        assert (panel.message.kind, panel.message.text()) == ("warning", NEEDS_THREE)
        assert panel.center_value.text() == "–" and not panel.origin_button.isEnabled()
    panel.circle_tool.add_tool_point(*wall(dish_clip)[2])
    assert window.panels[3].state == "done" and panel.message.isHidden()


def test_three_points_on_a_line_show_the_message_of_fit_circle(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    line = [[40.5, 50.5], [140.5, 100.5], [240.5, 150.5]]
    with pytest.raises(ValueError) as on_a_line:
        geometry.fit_circle(line)
    place(panel.circle_tool, line)
    assert (panel.message.kind, panel.message.text()) == ("problem", str(on_a_line.value))
    assert window.statusBar().currentMessage() == str(on_a_line.value)  # the same text in the status line
    assert window.panels[3].state == "attention" and window.panels[3].hint.text() == str(on_a_line.value)
    assert panel.radius_value.text() == "–"


def test_redo_clears_the_boundary_and_the_panel_stays_done(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    qtbot.mouseClick(panel.circle_redo, Qt.MouseButton.LeftButton)
    assert window.controller.session.boundary is None and panel.circle_tool.points == []
    assert [panel.center_value.text(), panel.radius_value.text(), panel.rms_value.text()] == ["–"] * 3
    assert window.panels[3].state == "done" and window.panels[3].hint.text() == AT_FRAME_CENTER
    assert panel.message.isHidden() and not panel.origin_button.isEnabled()
    assert not panel.crop_box.isEnabled() and not panel.crop_box.isChecked()
    assert panel.origin_value.text() == "160.0, 120.0 px"


def test_the_circle_button_chooses_the_tool_and_stays_while_points_are_placed(window, qtbot, dish_clip):
    calibration, panel = opened(window, qtbot, dish_clip)
    qtbot.mouseClick(panel.circle_button, Qt.MouseButton.LeftButton)
    assert window.view.tool is panel.circle_tool and panel.circle_button.isChecked()
    place(panel.circle_tool, wall(dish_clip))
    assert window.view.tool is panel.circle_tool  # more points may follow
    qtbot.mouseClick(panel.circle_redo, Qt.MouseButton.LeftButton)
    assert window.view.tool is panel.circle_tool  # Redo is for clicking them again
    qtbot.mouseClick(panel.axes_button, Qt.MouseButton.LeftButton)
    assert window.view.tool is panel.axes_tool
    assert panel.axes_button.isChecked() and not panel.circle_button.isChecked()
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window.view.tool is None and not panel.axes_button.isChecked()


# ---------------------------------------------------------------------------------------------
# The diameter


def test_the_diameter_lands_in_the_session_with_or_before_the_boundary(window, qtbot, dish_clip):
    calibration, panel = opened(window, qtbot, dish_clip)
    panel.set_diameter_mm(35.0)  # typed before there is a circle
    assert window.controller.session.boundary is None and panel.diameter_box.value() == 35.0
    place(panel.circle_tool, wall(dish_clip))
    assert window.controller.session.boundary.diameter_mm == 35.0
    panel.diameter_box.setValue(32.0)  # typed into the box, with the circle there
    assert window.controller.session.boundary.diameter_mm == 32.0
    place(calibration.stick_tool, STICK)
    assert panel.two_r_value.text() == "32.40 mm"  # next to the 32.00 mm of the box
    panel.set_diameter_mm(0.0)  # "not known"
    assert window.controller.session.boundary.diameter_mm is None
    assert panel.diameter_box.text() == "not known"
    qtbot.mouseClick(panel.circle_redo, Qt.MouseButton.LeftButton)
    panel.set_diameter_mm(34.0)
    place(panel.circle_tool, wall(dish_clip))  # the diameter outlives Redo: it is the boundary's, not the clicks'
    assert window.controller.session.boundary.diameter_mm == 34.0


@pytest.mark.parametrize("bad", [-5.0, math.nan, math.inf])
def test_a_diameter_that_is_not_positive_and_finite_is_refused(window, qtbot, dish_clip, bad):
    _, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    panel.set_diameter_mm(35.0)
    panel.set_diameter_mm(bad)
    assert window.controller.session.boundary.diameter_mm == 35.0
    assert (panel.message.kind, panel.message.text()) == ("problem", "The diameter must be more than 0 mm.")
    assert panel.diameter_box.property("check") == "error" and window.panels[3].state == "attention"
    panel.set_diameter_mm(35.0)
    assert panel.message.isHidden() and window.panels[3].state == "done"


def test_the_dish_diameter_of_the_manifest_is_filled_in(window, qtbot, dish_clip, tmp_path):
    folder = tmp_path / "group B" / "videos"
    folder.mkdir(parents=True)
    video = folder / dish_clip.path.name
    shutil.copyfile(dish_clip.path, video)
    manifest = tmp_path / "group B" / "data" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("video_file,fps_true,dish_mm\ndish.MOV,240.0,34.5\nother.MOV,240.0,90\n", encoding="utf-8")
    assert geometry.dish_mm_from_manifest(video, manifest) == 34.5  # the row this video has

    window.open_path(video)
    panel = body(window, 4)
    assert panel.diameter_box.value() == 34.5
    place(panel.circle_tool, wall(dish_clip))
    assert window.controller.session.boundary.diameter_mm == 34.5


def test_the_dish_diameter_of_the_manifest_pointed_to_last_is_filled_in(window, qtbot, dish_clip, tmp_path,
                                                                       own_settings):
    """A manifest that is not data/manifest.csv above the video, which the time panel remembers. The
    settings scope stays on this test's folder afterwards, so the remembered manifest is taken out again."""
    video = tmp_path / "videos" / dish_clip.path.name
    video.parent.mkdir()
    shutil.copyfile(dish_clip.path, video)
    manifest = tmp_path / "course" / "my manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("video_file,fps_true,dish_mm\nother.MOV,240.0,90\ndish.MOV,240.0,34.5\n", encoding="utf-8")
    assert geometry.find_manifest(video) is None  # none above the video
    settings().setValue(MANIFEST_KEY, str(manifest))
    try:
        window.open_path(video)
        assert body(window, 4).diameter_box.value() == 34.5
    finally:
        settings().remove(MANIFEST_KEY)


# ---------------------------------------------------------------------------------------------
# The axes


def test_origin_to_center_copies_the_center_to_the_axes(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    session = window.controller.session
    panel.axes_tool.add_tool_point(100.5, 50.5)
    assert panel.origin_value.text() == "100.5, 50.5 px"
    place(panel.circle_tool, wall(dish_clip))
    assert session.axes.origin_px == [100.5, 50.5]  # placed by hand: the fit leaves it
    assert window.panels[3].hint.text() == "Boundary: R = 108.0 px, RMS 0.0 px. The origin is at 100.5, 50.5 px."
    qtbot.mouseClick(panel.origin_button, Qt.MouseButton.LeftButton)
    assert session.axes.origin_px == pytest.approx([160.3, 119.8], abs=1e-6)
    assert session.axes.origin_px == session.boundary.center_px
    assert panel.origin_value.text() == "160.3, 119.8 px"
    assert window.panels[3].hint.text().endswith("The origin is at the center.")


def test_the_axes_angle_is_typed_in_degrees(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    session = window.controller.session
    panel.set_angle(30.0)
    assert session.axes.angle_deg == 30.0 and panel.angle_box.value() == 30.0
    panel.angle_box.setValue(-12.5)  # typed into the box
    assert session.axes.angle_deg == -12.5
    panel.set_angle(math.nan)
    assert session.axes.angle_deg == -12.5
    assert panel.message.kind == "problem" and "angle" in panel.message.text()
    panel.set_angle(0.0)
    assert session.axes.angle_deg == 0.0 and panel.message.isHidden()


def test_crop_to_boundary_needs_a_boundary_and_is_on_until_the_user_unticks_it(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    session = window.controller.session
    assert not panel.crop_box.isEnabled() and not panel.crop_box.isChecked()  # no boundary yet (decision 42)
    place(panel.circle_tool, wall(dish_clip))
    assert panel.crop_box.isEnabled() and panel.crop_box.isChecked()  # the setting, on by default
    changes = []
    window.controller.session_changed.connect(lambda: changes.append(session.processing.boundary_crop))
    panel.crop_box.click()
    assert session.processing.boundary_crop is False and changes == [False]
    panel.set_crop(True)
    assert session.processing.boundary_crop is True and panel.crop_box.isChecked()
    for crop in (True, False):  # without a boundary the box is off on the screen only: the setting stays
        panel.set_crop(crop)
        panel.circle_tool.redo()
        assert not panel.crop_box.isEnabled() and not panel.crop_box.isChecked()
        assert session.processing.boundary_crop is crop
        place(panel.circle_tool, wall(dish_clip))
        assert panel.crop_box.isEnabled() and panel.crop_box.isChecked() is crop


def test_without_a_boundary_the_panel_is_done_wherever_the_origin_is(window, qtbot, dish_clip):
    _, panel = opened(window, qtbot, dish_clip)
    dock = window.panels[3]
    assert (dock.state, dock.hint.text()) == ("done", AT_FRAME_CENTER)  # the axes exist from the start
    panel.axes_tool.add_tool_point(100.5, 50.5)
    assert (dock.state, dock.hint.text()) == ("done", f"{OPTIONAL} The origin is at 100.5, 50.5 px.")
    for crop in (False, True):  # without a boundary there is no crop, whatever the setting
        panel.set_crop(crop)
        assert dock.state == "done" and window.controller.session.processing.boundary_crop is crop


# ---------------------------------------------------------------------------------------------
# The same session again, and another video


def test_a_session_that_is_opened_shows_its_circle_and_its_axes(window, qtbot, dish_clip):
    calibration, panel = opened(window, qtbot, dish_clip)
    place(calibration.stick_tool, STICK)
    place(panel.circle_tool, wall(dish_clip))
    panel.set_diameter_mm(33.0)
    panel.axes_tool.add_tool_point(100.5, 50.5)
    panel.set_angle(15.0)
    panel.set_crop(False)
    before = window.controller.session.to_json()
    again = restart(window)
    assert again.to_json() == before  # opening a session changes nothing in it
    assert panel.center_value.text() == "160.3, 119.8 px"
    assert panel.radius_value.text() == "16.20 mm (108.0 px)"
    assert panel.diameter_box.value() == 33.0 and panel.angle_box.value() == 15.0
    assert panel.origin_value.text() == "100.5, 50.5 px"
    assert not panel.crop_box.isChecked()
    assert window.panels[3].state == "done"
    # the origin was placed by hand before: a new fit still leaves it
    panel.circle_tool.redo()
    place(panel.circle_tool, on_circle((150.0, 110.0), 90.0, SIX))
    assert again.axes.origin_px == [100.5, 50.5]


@pytest.mark.parametrize("broken", [None, [5.0], ["a", "b"], [0, 0]])
def test_a_session_without_a_usable_origin_gets_the_default_one(window, qtbot, dish_clip, broken):
    # a session file may be edited by hand; [0, 0] is the origin of a session that never had one
    _, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    window.controller.session.axes.origin_px = broken
    again = restart(window)
    assert again.axes.origin_px == pytest.approx([160.3, 119.8], abs=1e-6)  # a circle exists: its center
    assert panel.origin_value.text() == "160.3, 119.8 px" and not panel.axes_tool.placed


def test_another_video_starts_with_a_new_boundary_panel(window, qtbot, dish_clip, disk_clip):
    _, panel = opened(window, qtbot, dish_clip)
    place(panel.circle_tool, wall(dish_clip))
    panel.set_diameter_mm(33.0)
    panel.set_angle(15.0)
    panel.set_crop(False)
    panel.set_diameter_mm(-1.0)  # a refusal is showing
    window.open_path(disk_clip.path)
    assert panel.center_value.text() == "–" and panel.origin_value.text() == "160.0, 120.0 px"
    assert panel.diameter_box.value() == 0.0 and panel.angle_box.value() == 0.0
    assert window.controller.session.processing.boundary_crop is True  # the new session's default
    assert not panel.crop_box.isChecked() and not panel.crop_box.isEnabled()  # and no boundary yet
    assert panel.diameter_box.property("check") in (None, "")
    assert window.panels[3].state == "done" and window.panels[3].hint.text() == AT_FRAME_CENTER
    assert panel.message.isHidden()
