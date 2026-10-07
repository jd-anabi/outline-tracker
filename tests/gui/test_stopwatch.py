"""The Stopwatch dialog of panel 2 (SPEC 3.3, 4.1, 10.1; task C8a).

Expected values come from the spec: fps_true = (f_b - f_a) / (t_b - t_a), with two video frame
numbers and the two stopwatch readings in s; equal readings are an error. By hand:
- frames 12 and 492, readings 1.5 s and 3.5 s: 480 frames in 2 s, 240 frames per second;
- frames 100 and 40, readings 2.25 s and 2.0 s (the later moment first): -60 / -0.25 = 240;
- frames 0 and 90, readings 10.0 s and 13.0 s: 30 frames per second, under the 100 of SPEC 3.3.
The message for numbers that give no frame rate is `geometry.fps_from_stopwatch`'s own.

The dialog and playing (SPEC 10.1, the brief): playing stops when a field is used, and "Use frame
shown" fills a box from the view. The window cannot be used while the dialog is open, so playing
stops as the dialog opens. Frames while playing follow from the grid: a new session's clip has
step 2, so n ticks from frame f show frame f + 2 n. A tick is the timer's signal, emitted by the test.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit

from finish_helpers import is_paused, is_playing, named, never_blocking, tick  # noqa: F401 (fixture)
from gui_helpers import show
from outline_tracker import geometry
from outline_tracker.gui.stopwatch_dialog import StopwatchDialog
from session_helpers import body, hd_clip, hd_scene_clip, read_json, type_into  # noqa: F401 (fixtures)

LEFT = Qt.MouseButton.LeftButton


def opened(window, qtbot):
    """Click Stopwatch… in panel 2 and return the dialog that opened."""
    before = window.findChildren(StopwatchDialog)
    qtbot.mouseClick(body(window, 2).stopwatch_button, LEFT)
    (dialog,) = [found for found in window.findChildren(StopwatchDialog) if found not in before]
    return dialog


def fill(dialog, frame_a, time_a_s, frame_b, time_b_s) -> None:
    dialog.frame_a_box.setValue(frame_a)
    dialog.time_a_box.setValue(time_a_s)
    dialog.frame_b_box.setValue(frame_b)
    dialog.time_b_box.setValue(time_b_s)


def refusal_of(*numbers) -> str:
    """The message of `geometry.fps_from_stopwatch` for numbers that give no frame rate."""
    with pytest.raises(ValueError) as refused:
        geometry.fps_from_stopwatch(*numbers)
    return str(refused.value)


@pytest.fixture
def with_video(window, qtbot, clip_in_odd_folder):
    named(window, qtbot, clip_in_odd_folder)
    return show(window, qtbot)


# ---------------------------------------------------------------------------------------------


def test_the_stopwatch_button_waits_for_a_video(window, qtbot, clip_in_odd_folder):
    button = body(window, 2).stopwatch_button
    assert button.text() == "Stopwatch…" and not button.isEnabled()
    window.open_path(clip_in_odd_folder.path)
    assert button.isEnabled()


def test_the_dialog_is_window_modal_and_does_not_block(with_video, qtbot, never_blocking):  # noqa: F811
    window = with_video
    dialog = opened(window, qtbot)
    assert dialog.isVisible() and dialog.windowModality() == Qt.WindowModality.WindowModal
    assert dialog.parent() is window and dialog.windowTitle() == "Stopwatch"
    assert dialog.time_a_box.suffix() == dialog.time_b_box.suffix() == " s"
    assert dialog.result_label.text() == "fps_true = –" and dialog.message.isHidden()
    qtbot.mouseClick(dialog.cancel_button, LEFT)
    assert not dialog.isVisible()


@pytest.mark.parametrize("numbers, fps", [((12, 1.5, 492, 3.5), 240.0), ((100, 2.25, 40, 2.0), 240.0),
                                          ((0, 10.0, 90, 13.0), 30.0)])
def test_ok_puts_the_frame_rate_and_the_four_numbers_into_the_session(with_video, qtbot, numbers, fps):
    window = with_video
    time = window.controller.session.time
    dialog = opened(window, qtbot)
    fill(dialog, *numbers)
    assert dialog.result_label.text() == f"fps_true = {fps:.2f} fps"
    assert time.fps_true is None  # nothing reaches the session before OK
    qtbot.mouseClick(dialog.ok_button, LEFT)
    assert not dialog.isVisible()
    frame_a, time_a_s, frame_b, time_b_s = numbers
    assert (time.fps_true, time.source, time.manifest_path) == (fps, "stopwatch", None)
    assert time.stopwatch == {"frame_a": frame_a, "time_a_s": time_a_s, "frame_b": frame_b, "time_b_s": time_b_s}
    panel = body(window, 2)
    assert panel.source_label.text() == "from the stopwatch" and float(panel.fps_edit.text()) == fps
    assert window.panels[1].state == ("done" if fps >= 100 else "attention")  # under 100 fps: a warning, as typed
    assert window.navigation.time_label.text() == "t = 0.000 s"  # frame 0 of the clip, now with a time
    saved = read_json(window.controller.save_now())["time"]
    assert saved == {"fps_true": fps, "source": "stopwatch", "manifest_path": None, "stopwatch": time.stopwatch}


@pytest.mark.parametrize("numbers", [(12, 1.5, 492, 1.5), (12, 1.5, 12, 3.5), (12, 3.5, 492, 1.5)],
                         ids=["equal readings", "equal frames", "the later frame shows the earlier time"])
def test_numbers_that_give_no_frame_rate_are_refused_in_the_dialog(with_video, qtbot, numbers):
    window = with_video
    time = window.controller.session.time
    dialog = opened(window, qtbot)
    fill(dialog, *numbers)
    assert dialog.result_label.text() == "fps_true = –"
    qtbot.mouseClick(dialog.ok_button, LEFT)
    assert dialog.isVisible()  # the dialog stays, and says why in the function's own words
    assert not dialog.message.isHidden() and dialog.message.text() == refusal_of(*numbers)
    assert (time.fps_true, time.source, time.stopwatch) == (None, "typed", None)
    assert window.controller.session.time is time

    fill(dialog, 12, 1.5, 492, 3.5)  # put right: the message goes, and OK takes the value
    assert dialog.message.isHidden() and dialog.result_label.text() == "fps_true = 240.00 fps"
    qtbot.mouseClick(dialog.ok_button, LEFT)
    assert not dialog.isVisible() and time.fps_true == 240.0


def test_equal_readings_are_an_error_of_their_own(with_video):
    assert "readings are equal" in refusal_of(12, 1.5, 492, 1.5)  # SPEC 4.1


def test_cancel_changes_nothing(with_video, qtbot):
    window = with_video
    type_into(qtbot, body(window, 2).fps_edit, "239.6")
    dialog = opened(window, qtbot)
    fill(dialog, 12, 1.5, 492, 3.5)
    qtbot.mouseClick(dialog.cancel_button, LEFT)
    time = window.controller.session.time
    assert (time.fps_true, time.source, time.stopwatch) == (239.6, "typed", None)


def test_use_frame_shown_fills_a_frame_box_from_the_view(with_video, qtbot):
    window = with_video
    window.navigation.slider.setValue(20)  # a new session's clip has step 2: frame 40
    assert window.view.frame == 40
    dialog = opened(window, qtbot)
    assert (dialog.use_a_button.text(), dialog.use_b_button.text()) == ("Use frame shown", "Use frame shown")
    assert (dialog.frame_a_box.value(), dialog.frame_b_box.value()) == (0, 0)
    qtbot.mouseClick(dialog.use_a_button, LEFT)
    assert (dialog.frame_a_box.value(), dialog.frame_b_box.value()) == (40, 0)
    dialog.time_a_box.setValue(1.5)
    qtbot.mouseClick(dialog.cancel_button, LEFT)

    # the window cannot be used while the dialog is open: the second frame is looked for after it is
    # closed, and the dialog opens again with what was typed
    window.navigation.slider.setValue(50)
    again = opened(window, qtbot)
    assert (again.frame_a_box.value(), again.time_a_box.value()) == (40, 1.5)
    qtbot.mouseClick(again.use_b_button, LEFT)
    assert again.frame_b_box.value() == 100
    again.time_b_box.setValue(1.75)
    assert again.result_label.text() == "fps_true = 240.00 fps"  # 60 frames in 0.25 s
    qtbot.mouseClick(again.ok_button, LEFT)
    assert window.controller.session.time.stopwatch == {"frame_a": 40, "time_a_s": 1.5, "frame_b": 100,
                                                        "time_b_s": 1.75}


def test_the_dialog_opens_with_the_numbers_of_the_session(with_video, qtbot):
    window = with_video
    dialog = opened(window, qtbot)
    fill(dialog, 12, 1.5, 492, 3.5)
    qtbot.mouseClick(dialog.ok_button, LEFT)
    again = opened(window, qtbot)
    assert (again.frame_a_box.value(), again.time_a_box.value()) == (12, 1.5)
    assert (again.frame_b_box.value(), again.time_b_box.value()) == (492, 3.5)
    assert again.result_label.text() == "fps_true = 240.00 fps"
    qtbot.mouseClick(again.cancel_button, LEFT)


def test_what_was_typed_for_one_video_is_not_offered_for_the_next(with_video, qtbot, hd_clip):
    window = with_video
    dialog = opened(window, qtbot)
    fill(dialog, 12, 1.5, 492, 3.5)
    qtbot.mouseClick(dialog.cancel_button, LEFT)
    window.open_path(hd_clip.path)
    again = opened(window, qtbot)
    assert (again.frame_a_box.value(), again.time_a_box.value(), again.frame_b_box.value(),
            again.time_b_box.value()) == (0, 0.0, 0, 0.0)
    qtbot.mouseClick(again.cancel_button, LEFT)


def test_a_frame_rate_typed_afterwards_takes_the_stopwatchs_numbers_out_of_the_session(with_video, qtbot):
    window = with_video
    dialog = opened(window, qtbot)
    fill(dialog, 12, 1.5, 492, 3.5)
    qtbot.mouseClick(dialog.ok_button, LEFT)
    type_into(qtbot, body(window, 2).fps_edit, "239.6")
    time = window.controller.session.time
    assert (time.fps_true, time.source, time.stopwatch) == (239.6, "typed", None)


# ---------------------------------------------------------------------------------------------
# The dialog and playing


def test_opening_the_dialog_stops_playing(with_video, qtbot):
    # the window cannot be used while the dialog is open: nobody could pause there
    window = with_video
    bar = window.navigation
    window.view.setFocus()
    qtbot.keyClick(window, Qt.Key.Key_Space)
    tick(bar)
    assert is_playing(bar) and window.view.frame == 2
    # pressed without a turn of the event loop, so that no tick of the running timer itself comes in between
    body(window, 2).stopwatch_button.click()
    (dialog,) = window.findChildren(StopwatchDialog)
    assert is_paused(bar)  # at once, wherever the keyboard goes
    QApplication.processEvents()
    tick(bar, 5)  # a tick that was on its way moves nothing
    assert is_paused(bar) and (bar.frame, window.view.frame) == (2, 2)
    qtbot.mouseClick(dialog.use_a_button, LEFT)
    assert dialog.frame_a_box.value() == window.view.frame == 2
    qtbot.mouseClick(dialog.cancel_button, LEFT)
    assert is_paused(bar) and window.view.frame == 2  # closing the dialog starts nothing


def test_use_frame_shown_takes_the_frame_the_view_shows_at_the_click(with_video, qtbot):
    # not the frame it showed when the dialog opened: whatever moved the picture since then, the
    # number entered is the frame that is seen (it goes into fps_true)
    window = with_video
    bar = window.navigation
    dialog = opened(window, qtbot)
    assert window.view.frame == 0
    bar.slider.setValue(6)  # grid position 6: frame 12
    assert window.view.frame == 12
    qtbot.mouseClick(dialog.use_a_button, LEFT)
    assert (dialog.frame_a_box.value(), dialog.frame_b_box.value()) == (12, 0)
    bar.play()  # the user cannot, behind the dialog; whatever does must not leave a stale number
    tick(bar, 5)
    bar.pause()
    assert window.view.frame == 22
    qtbot.mouseClick(dialog.use_b_button, LEFT)
    assert (dialog.frame_a_box.value(), dialog.frame_b_box.value()) == (12, 22)
    qtbot.mouseClick(dialog.cancel_button, LEFT)


def test_use_frame_shown_changes_nothing_while_the_view_shows_no_frame(with_video, qtbot):
    window = with_video
    dialog = opened(window, qtbot)
    dialog.frame_a_box.setValue(40)
    window.view.set_source(None)  # no picture any more, behind the open dialog
    assert window.view.frame is None
    qtbot.mouseClick(dialog.use_a_button, LEFT)
    assert dialog.frame_a_box.value() == 40
    qtbot.mouseClick(dialog.cancel_button, LEFT)


def test_a_number_box_of_the_dialog_stops_playing_as_a_field_of_the_window_does(with_video, qtbot):
    window = with_video
    bar = window.navigation
    dialog = opened(window, qtbot)
    dialog.activateWindow()  # the keyboard goes to the active window
    qtbot.waitUntil(lambda: QApplication.activeWindow() is dialog)
    for box in (dialog.frame_a_box, dialog.time_a_box, dialog.frame_b_box, dialog.time_b_box):
        dialog.cancel_button.setFocus()
        QApplication.processEvents()
        bar.play()
        QApplication.processEvents()
        assert is_playing(bar), box  # a button of the dialog is no text field: playing goes on
        box.setFocus()
        QApplication.processEvents()
        assert is_paused(bar), box
    qtbot.mouseClick(dialog.cancel_button, LEFT)


def test_a_text_field_that_is_not_over_the_window_does_not_stop_playing(with_video, qtbot):
    window = with_video
    bar = window.navigation
    elsewhere = QLineEdit()  # never shown, and no part of the window or of a dialog over it
    qtbot.addWidget(elsewhere)
    window.view.setFocus()
    bar.play()
    QApplication.instance().focusChanged.emit(window.view, elsewhere)  # the keyboard went there
    assert is_playing(bar)
