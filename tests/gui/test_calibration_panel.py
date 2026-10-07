"""Panel 3, Calibration (SPEC 4.2, 4.3, 8.10, 10.1; task C3): the stick and its length, the scale,
the tape check.

Expected values are worked out here from the spec's formulas, never read from the program:
- a stick of 926 px for 30 mm: k = 30 / 926 = 0.032397 mm per px = 32.40 um per px, and
  sigma_k / k = sqrt(2) * 0.5 / 926 = 0.076%, shown as "32.40 µm/px ± 0.08%" (SPEC 4.2);
- with that scale, two marks 620 px apart measure 620 * 30 / 926 = 20.086 mm: +0.43% of 20 mm,
  inside the 1% rule; 630 px apart measure 20.410 mm: +2.05%, outside it (SPEC 4.3);
- 60 mm on the same stick: 60 / 926 = 64.79 um per px; 15 mm: 16.20 um per px.
The wording is the design note's. Colours are read from the drawn panel beside the text, never
from a pixel of text.
"""

import math

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from calibration_helpers import body, look, opened, place, restart, shown, wide_clip  # noqa: F401  (fixtures)
from gui_helpers import pixels
from outline_tracker.gui import theme

START_HINT = "Click Stick. Click the two ends of a known length on the ruler. Type the length."
STICK = [(100.5, 300.5), (1026.5, 300.5)]        # 926 px
WITHIN = [(200.5, 500.5), (820.5, 500.5)]        # 620 px
OUTSIDE = [(200.5, 500.5), (830.5, 500.5)]       # 630 px
NO_LENGTH = "The stick length must be more than 0 mm."
NO_DISTANCE = "The true distance must be more than 0 mm."


def test_the_spec_numbers_of_this_file():
    assert f"{1000 * 30 / 926:.2f}" == "32.40" and f"{100 * math.sqrt(2) * 0.5 / 926:.2f}" == "0.08"
    assert f"{620 * 30 / 926:.2f}" == "20.09" and f"{100 * (620 * 30 / 926 - 20) / 20:.1f}" == "0.4"
    assert f"{630 * 30 / 926:.2f}" == "20.41" and f"{100 * (630 * 30 / 926 - 20) / 20:.1f}" == "2.1"


# ---------------------------------------------------------------------------------------------
# Before a video, and what the panel holds


def test_without_a_video_the_panel_waits(window):
    panel = body(window, 3)
    assert not panel.isEnabled()  # nothing to click on yet
    assert window.panels[2].state == "todo" and window.panels[2].hint.text() == START_HINT
    assert panel.message.isHidden()


def test_the_controls_are_the_specs(window, qtbot, dish_clip):
    panel, _ = opened(window, qtbot, dish_clip)
    assert panel.isEnabled()
    assert [panel.stick_button.text(), panel.tape_button.text()] == ["Stick", "Tape"]
    assert [panel.stick_redo.text(), panel.tape_redo.text()] == ["Redo", "Redo"]
    assert panel.stick_redo.toolTip() == "Remove the points and click them again."
    assert panel.stick_button.isCheckable() and panel.tape_button.isCheckable()
    for box, value in ((panel.length_box, 30.0), (panel.true_box, 20.0)):
        assert box.suffix() == " mm" and box.decimals() == 2 and box.value() == value
        assert not box.keyboardTracking()  # a typed number counts when it is complete
        assert box.alignment() & Qt.AlignmentFlag.AlignRight
    for button in (panel.stick_button, panel.tape_button, panel.stick_redo, panel.tape_redo):
        assert not button.autoDefault() and button.toolTip() != ""  # Enter presses nothing
    assert panel.scale_value.text() == "–" and panel.tape_value.text() == "–"
    assert window.panels[2].state == "todo" and window.panels[2].hint.text() == START_HINT


# ---------------------------------------------------------------------------------------------
# The stick and the scale


def test_a_stick_of_926_px_for_30_mm_shows_32_40_um_per_px(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.set_stick_length(30.0)
    assert panel.scale_value.text() == "32.40 µm/px ± 0.08%"
    session = window.controller.session
    assert session.calibration.stick == {"p1_px": [100.5, 300.5], "p2_px": [1026.5, 300.5], "length_mm": 30.0}
    assert session.world_frame().k_mm_per_px == pytest.approx(30 / 926)
    assert window.panels[2].state == "done"
    assert window.panels[2].hint.text() == "Scale: 32.40 µm/px ± 0.08%."
    assert panel.message.isHidden()  # 926 px is long enough: nothing to warn about


def test_a_length_typed_with_the_stick_there_updates_the_scale_at_once(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.set_stick_length(60.0)
    assert panel.scale_value.text() == "64.79 µm/px ± 0.08%"
    assert window.controller.session.calibration.stick["length_mm"] == 60.0
    panel.length_box.setValue(15.0)  # typed into the box itself
    assert panel.scale_value.text() == "16.20 µm/px ± 0.08%"
    assert window.controller.session.calibration.stick["length_mm"] == 15.0
    assert window.panels[2].hint.text() == "Scale: 16.20 µm/px ± 0.08%."


def test_a_length_typed_before_the_stick_is_the_sticks_length(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    panel.set_stick_length(60.0)
    assert window.controller.session.calibration.stick is None and panel.scale_value.text() == "–"
    place(panel.stick_tool, STICK)
    assert window.controller.session.calibration.stick["length_mm"] == 60.0
    assert panel.scale_value.text() == "64.79 µm/px ± 0.08%" and panel.length_box.value() == 60.0


@pytest.mark.parametrize("bad", [0.0, -3.0, math.nan, math.inf])
def test_a_length_that_is_not_positive_and_finite_is_refused_where_it_is_typed(window, qtbot, wide_clip, bad):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.set_stick_length(bad)
    assert window.controller.session.calibration.stick["length_mm"] == 30.0  # the session keeps its length
    assert panel.scale_value.text() == "32.40 µm/px ± 0.08%"
    assert (panel.message.kind, panel.message.text()) == ("problem", NO_LENGTH)
    assert not panel.message.isHidden()
    assert panel.length_box.property("check") == "error"
    assert window.panels[2].state == "attention" and window.panels[2].hint.text() == NO_LENGTH
    panel.set_stick_length(45.0)  # a length that can be used takes the refusal away
    assert window.controller.session.calibration.stick["length_mm"] == 45.0
    assert panel.message.isHidden() and panel.length_box.property("check") in (None, "")
    assert window.panels[2].state == "done"


def test_a_zero_typed_into_the_box_stays_there_and_is_marked(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    panel.length_box.setValue(0.0)
    assert panel.length_box.value() == 0.0  # what was typed is not changed behind the user's back
    assert panel.length_box.property("check") == "error" and panel.message.text() == NO_LENGTH
    place(panel.stick_tool, STICK)
    assert window.controller.session.calibration.stick["length_mm"] == 30.0  # the last length that can be used
    panel.true_box.setValue(-1.0)
    assert panel.true_box.property("check") == "error"
    assert panel.message.text() == NO_LENGTH  # one message: the first field that is wrong


def test_a_stick_shorter_than_300_px_gets_a_warning(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, [(100.5, 300.5), (399.5, 300.5)])  # 299 px
    assert panel.scale_value.text() == f"{1000 * 30 / 299:.2f} µm/px ± {100 * math.sqrt(2) * 0.5 / 299:.2f}%"
    assert window.panels[2].state == "attention"
    assert panel.message.kind == "warning" and "300 px" in panel.message.text()
    assert "299.0 px" in panel.message.text()
    assert window.panels[2].hint.text() == panel.message.text()
    panel.stick_tool.redo()
    place(panel.stick_tool, [(100.5, 300.5), (400.5, 300.5)])  # 300 px: long enough
    assert window.panels[2].state == "done" and panel.message.isHidden()


def test_the_stick_button_chooses_the_tool_and_follows_it(window, qtbot, dish_clip):
    panel, dish = opened(window, qtbot, dish_clip)
    view = window.view
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)
    assert view.tool is panel.stick_tool and panel.stick_button.isChecked()
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert view.tool is None and not panel.stick_button.isChecked()
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)  # a second click lets go of the tool
    assert view.tool is None and not panel.stick_button.isChecked()
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)
    place(panel.stick_tool, [(50.5, 20.5), (250.5, 20.5)])
    assert view.tool is None and not panel.stick_button.isChecked()  # finished
    qtbot.mouseClick(panel.tape_button, Qt.MouseButton.LeftButton)
    assert view.tool is panel.tape_tool and panel.tape_button.isChecked()
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)  # one tool at a time, in the whole window
    assert view.tool is panel.stick_tool
    assert panel.stick_button.isChecked() and not panel.tape_button.isChecked()
    view.set_tool(dish.circle_tool)
    assert not panel.stick_button.isChecked() and dish.circle_button.isChecked()


def test_redo_of_the_stick_takes_the_scale_away(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    place(panel.tape_tool, WITHIN)
    qtbot.mouseClick(panel.stick_redo, Qt.MouseButton.LeftButton)
    session = window.controller.session
    assert session.calibration.stick is None
    assert session.calibration.check is not None  # the tape marks stay where they are
    assert panel.scale_value.text() == "–" and panel.tape_value.text() == "–"
    assert window.panels[2].state == "todo" and window.panels[2].hint.text() == START_HINT
    assert panel.message.isHidden()
    assert panel.length_box.value() == 30.0


def test_a_third_stick_click_takes_the_scale_away_until_the_second_end(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.stick_tool.add_tool_point(200.5, 100.5)
    assert panel.scale_value.text() == "–" and window.panels[2].state == "todo"
    panel.stick_tool.add_tool_point(200.5 + 926, 100.5)
    assert panel.scale_value.text() == "32.40 µm/px ± 0.08%" and window.panels[2].state == "done"


# ---------------------------------------------------------------------------------------------
# The tape check


def test_without_a_stick_the_tape_tool_is_disabled_and_the_panel_says_why(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    assert not panel.tape_button.isEnabled() and not panel.tape_button.isChecked()
    assert panel.tape_note.text() == "Place the stick first. The tape check needs the scale."
    assert not panel.tape_note.isHidden()
    place(panel.stick_tool, STICK)
    assert panel.tape_button.isEnabled() and panel.tape_note.isHidden()
    qtbot.mouseClick(panel.tape_button, Qt.MouseButton.LeftButton)
    assert window.view.tool is panel.tape_tool
    panel.stick_tool.redo()  # the scale goes while the tape tool is chosen
    assert window.view.tool is None
    assert not panel.tape_button.isEnabled() and not panel.tape_button.isChecked()
    assert not panel.tape_note.isHidden()


def test_a_tape_check_within_1_percent_passes_in_a_word_and_in_green(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    place(panel.tape_tool, WITHIN)
    panel.set_tape_true(20.0)
    session = window.controller.session
    assert session.calibration.check == {"p1_px": [200.5, 500.5], "p2_px": [820.5, 500.5], "true_mm": 20.0}
    assert panel.tape_value.text() == "20.09 mm, error 0.4%"
    assert panel.message.kind == "success"
    assert panel.message.text() == "Tape check: 20.09 mm measured, 20.00 mm true, error 0.4%. Passes (limit 1%)."
    assert window.panels[2].state == "done"
    assert window.panels[2].hint.text() == "Scale: 32.40 µm/px ± 0.08%. Tape check: 0.4%, passes (limit 1%)."
    assert window.statusBar().currentMessage() == panel.message.text()  # the same text in the status line


def test_a_tape_check_outside_1_percent_does_not_pass_in_a_word_and_in_red(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    place(panel.tape_tool, OUTSIDE)
    assert panel.tape_value.text() == "20.41 mm, error 2.1%"
    assert panel.message.kind == "problem"
    assert panel.message.text() == ("Tape check: 20.41 mm measured, 20.00 mm true, error 2.1%. "
                                    "Does not pass (limit 1%).")
    assert window.panels[2].state == "attention"
    assert window.panels[2].hint.text() == ("The tape check does not pass: 2.1% (limit 1%). "
                                            "Click Redo and place the stick again.")
    # the true distance is typed as what the marks do measure: now it passes, at once
    panel.set_tape_true(20.4)
    assert window.controller.session.calibration.check["true_mm"] == 20.4
    assert panel.message.kind == "success" and "Passes" in panel.message.text()
    assert window.panels[2].state == "done"
    qtbot.mouseClick(panel.tape_redo, Qt.MouseButton.LeftButton)
    assert window.controller.session.calibration.check is None
    assert panel.tape_value.text() == "–" and panel.message.isHidden()
    assert window.panels[2].hint.text() == "Scale: 32.40 µm/px ± 0.08%."


def test_a_measured_distance_below_the_true_one_has_a_minus_sign(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    place(panel.tape_tool, [(200.5, 500.5), (815.5, 500.5)])  # 615 px: 19.924 mm, 0.38% short
    assert f"{100 * (615 * 30 / 926 - 20) / 20:.1f}" == "-0.4"
    assert panel.tape_value.text() == "19.92 mm, error −0.4%"  # the minus sign, not a hyphen
    assert panel.message.kind == "success"


@pytest.mark.parametrize("bad", [0.0, -20.0, math.nan, math.inf])
def test_a_true_distance_that_is_not_positive_and_finite_is_refused(window, qtbot, wide_clip, bad):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    place(panel.tape_tool, WITHIN)
    panel.set_tape_true(bad)
    assert window.controller.session.calibration.check["true_mm"] == 20.0
    assert (panel.message.kind, panel.message.text()) == ("problem", NO_DISTANCE)
    assert panel.true_box.property("check") == "error"
    assert window.panels[2].state == "attention"
    panel.set_tape_true(20.0)
    assert panel.message.kind == "success" and panel.true_box.property("check") in (None, "")


def test_the_passing_and_the_failing_check_are_drawn_in_green_and_in_red(window, qtbot, qapp, wide_clip, look):
    theme.apply(qapp, False)
    panel, _ = opened(window, qtbot, wide_clip)
    shown(window, qtbot)

    def ground(widget, down: float = 3) -> str:
        """The colour `down` px under the widget's top edge, a third along: beside any text."""
        image = widget.grab().toImage()
        ratio = image.devicePixelRatio()
        red, green, blue = pixels(image)[round(down * ratio), round(widget.width() / 3 * ratio)]
        return f"#{red:02X}{green:02X}{blue:02X}"

    place(panel.stick_tool, STICK)
    place(panel.tape_tool, WITHIN)
    QApplication.processEvents()
    assert ground(panel.message) == theme.LIGHT["successBg"]
    panel.tape_tool.redo()
    place(panel.tape_tool, OUTSIDE)
    QApplication.processEvents()
    assert ground(panel.message) == theme.LIGHT["problemBg"]
    panel.set_stick_length(0.0)  # a length that cannot be used: its box is filled in the problem tint
    QApplication.processEvents()
    middle = panel.length_box.height() / 2  # clear of the shade under the box's top edge; the number is at the right
    assert ground(panel.length_box, middle) == theme.LIGHT["problemBg"]
    panel.set_stick_length(30.0)
    QApplication.processEvents()
    assert ground(panel.length_box, middle) == theme.LIGHT["field"]
    qtbot.mouseClick(panel.stick_button, Qt.MouseButton.LeftButton)  # the chosen tool's button is marked
    QApplication.processEvents()
    assert ground(panel.stick_button) == theme.LIGHT["accentSoft"]
    assert ground(panel.tape_button) == theme.LIGHT["panel"]


# ---------------------------------------------------------------------------------------------
# The wheel, and the same session again


def test_the_wheel_over_a_box_that_has_no_focus_changes_nothing(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    shown(window, qtbot)
    box = panel.length_box
    assert not box.hasFocus() and box.focusPolicy() == Qt.FocusPolicy.StrongFocus
    at = QPoint(box.width() // 3, box.height() // 2)
    turn = QWheelEvent(QPointF(at), QPointF(box.mapToGlobal(at)), QPoint(0, 0), QPoint(0, 120),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(box, turn)
    assert box.value() == 30.0  # scrolling the dock must never change a value
    box.setFocus()
    qtbot.waitUntil(box.hasFocus)
    QApplication.sendEvent(box, turn)
    assert box.value() == 31.0  # a box that is being typed in does take the wheel
    assert window.controller.session.calibration.stick is None and panel.stick_tool.length_mm == 31.0


def test_a_session_that_is_opened_shows_its_stick_and_its_check(window, qtbot, wide_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.set_stick_length(60.0)
    place(panel.tape_tool, WITHIN)
    panel.set_tape_true(40.0)
    before = window.controller.session.to_json()
    panel.length_box.setValue(0.0)  # a refusal is showing when the session is opened again
    again = restart(window)
    assert again.to_json() == before
    assert panel.length_box.value() == 60.0 and panel.true_box.value() == 40.0
    assert panel.length_box.property("check") in (None, "")
    assert panel.scale_value.text() == "64.79 µm/px ± 0.08%"
    assert panel.tape_value.text() == "40.17 mm, error 0.4%"
    assert window.panels[2].state == "done" and panel.message.kind == "success"


def test_a_scale_fitted_to_a_tracker_export_is_a_scale_too(window, qtbot, wide_clip):
    # a session made by from-tracker has no stick: its scale is the fit's (decision X18)
    panel, _ = opened(window, qtbot, wide_clip)
    session = window.controller.session
    session.calibration.tracker_fit = {"mm_per_px": 30 / 926, "rms_mm": 0.01, "n_points": 100}
    window.controller.touch()
    assert panel.scale_value.text() == "32.40 µm/px"  # no click precision to state
    assert window.panels[2].state == "done" and window.panels[2].hint.text() == "Scale: 32.40 µm/px."
    assert panel.tape_button.isEnabled() and panel.tape_note.isHidden()
    place(panel.tape_tool, WITHIN)
    assert panel.tape_value.text() == "20.09 mm, error 0.4%"


def test_a_stick_block_that_lacks_its_points_is_shown_as_no_stick(window, qtbot, wide_clip):
    # a session file may be edited by hand: what cannot be a stick must not stop the window
    panel, _ = opened(window, qtbot, wide_clip)
    session = window.controller.session
    for broken in ({"length_mm": 30.0}, {"p1_px": None, "p2_px": [1.0, 2.0], "length_mm": 30.0},
                   {"p1_px": [5.0, 5.0], "p2_px": [5.0, 5.0], "length_mm": 30.0}):
        session.calibration.stick = broken
        window.controller.touch()
        assert panel.scale_value.text() == "–" and window.panels[2].state == "todo"
        assert not panel.tape_button.isEnabled()


def test_another_video_starts_with_the_lengths_of_a_new_session(window, qtbot, wide_clip, dish_clip):
    panel, _ = opened(window, qtbot, wide_clip)
    place(panel.stick_tool, STICK)
    panel.set_stick_length(60.0)
    window.open_path(dish_clip.path)
    assert panel.length_box.value() == 30.0 and panel.true_box.value() == 20.0
    assert panel.scale_value.text() == "–"
    assert window.panels[2].state == "todo" and window.panels[2].hint.text() == START_HINT
