"""Safe during a run, and Remove asks first (task C8b; SPEC 5, 6.4, 10.1).

While a tracking run is going nothing can change what the run works on: the objects and their
points (panel 6, and a click on the picture), the name, the clip and the open video (panel 1 and
the File menu). Each part that is off says why. fps_true, the calibration and the dish stay usable:
they change only the session, which the GUI thread saves. The run is parked on a gate inside the
stand-in, so "during a run" is a state the test holds, not a moment it has to catch.

Remove deletes tracked frames, which cannot be undone, so it asks first and says how many.

Frames are video frame numbers; (u, v) is in px of the video frame (SPEC 3.1).
"""

from PySide6.QtCore import Qt

from last_controls_helpers import RUNNING, let_run_end, parked, press_track, record_confirm
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import LEFT, NO_KEY, Gate
from session_helpers import body
from track_helpers import Tracked, ids, ready_to_track, results_of, run_to_end
from tracking_helpers import center

WAIT = "Wait until tracking has stopped."


def lockable(window) -> dict:
    """The parts that change what a run works on, by a name for the failure message: of panel 6
    Add, Remove, the three tools, the mode and the window (its box and Auto); of panel 1 the name,
    the clip and the two Open buttons."""
    one, six = body(window, 1), body(window, 6)
    return {"Add": six.add_button, "Remove": six.remove_button, "Positive": six.positive_button,
            "Negative": six.negative_button, "Head": six.head_button, "mode": six.mode_box,
            "window": six.window_box, "Auto": six.auto_button, "name": one.name_edit, "start": one.start_box,
            "end": one.end_box, "step": one.step_box, "Open video": one.open_video_button,
            "Open session": one.open_session_button}


def a_run_with_a_second_object(window, qtbot, clip, gate):
    """A is clicked and ready to track; B is added after it, fine with a typed window, selected,
    with the Positive tool chosen: every part of `lockable` is on. Returns (TrackPanel, ObjectsPanel)."""
    track, objects = ready_to_track(window, qtbot, clip, parked(clip, gate), end=10)
    objects.add_object()
    objects.set_mode("fine")
    objects.set_fine_window(120)
    assert objects.prompts.selected == "B" and objects.prompts.tool_kind == "positive"
    return track, objects


def test_during_a_run_every_part_that_changes_its_input_is_off_and_says_why(window, qtbot, clip_in_odd_folder):
    clip, menus = clip_in_odd_folder, window.menus
    with Gate() as gate:
        track, objects = a_run_with_a_second_object(window, qtbot, clip, gate)
        parts = lockable(window)
        assert [name for name, part in parts.items() if not part.isEnabled()] == []  # all on before the run
        assert menus.open_action.isEnabled() and menus.open_session_action.isEnabled()
        press_track(qtbot, track, gate)
        assert [name for name, part in parts.items() if part.isEnabled()] == []
        assert [name for name, part in parts.items() if RUNNING not in part.toolTip()] == []
        for item in (menus.open_action, menus.open_session_action):
            assert not item.isEnabled() and RUNNING in item.toolTip()
        # what tracking does not read stays usable: fps_true, the calibration, the dish, saving
        assert body(window, 2).fps_edit.isEnabled()
        assert body(window, 3).stick_button.isEnabled() and body(window, 3).length_box.isEnabled()
        assert body(window, 4).circle_button.isEnabled() and body(window, 4).crop_box.isEnabled()
        assert menus.save_action.isEnabled() and menus.quit_action.isEnabled()
        let_run_end(qtbot, track, gate)
    assert [name for name, part in parts.items() if not part.isEnabled()] == []
    assert [name for name, part in parts.items() if RUNNING in part.toolTip()] == []
    assert all(part.toolTip() for part in parts.values())  # each has its own tip back
    for item in (menus.open_action, menus.open_session_action):
        assert item.isEnabled() and RUNNING not in item.toolTip()


def test_during_a_run_a_click_on_the_picture_places_no_point_and_the_tool_says_to_wait(window, qtbot,
                                                                                    clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:
        track, objects = a_run_with_a_second_object(window, qtbot, clip, gate)
        prompts, tool = objects.prompts, window.view.tool
        press_track(qtbot, track, gate)
        u, v = center(clip, "B", 0)
        window.show_frame(0)
        assert window.view.tool is tool  # the tool stays chosen, and says that it does nothing now
        assert window.tool_text.text() == WAIT and tool.cursor == Qt.CursorShape.ForbiddenCursor
        tool.click(u, v, LEFT, NO_KEY)  # what the view calls for a click on the picture
        assert not prompts.add_point(u, v, 1) and not prompts.add_point(u + 3, v, 0)
        assert not prompts.set_head(u, v)
        prompts.select("A")
        assert not prompts.undo()  # the Undo key
        a, b = window.controller.session.tracks
        assert b.prompts == [] and b.head_px is None
        assert [([tuple(point) for point in prompt.points_px], list(prompt.labels)) for prompt in a.prompts] == [
            ([center(clip, "A", 0)], [1])]
        let_run_end(qtbot, track, gate)
    prompts.select("B")
    objects.prompts.choose_tool("positive")
    assert window.tool_text.text() == "Positive: click on animal B."
    assert window.view.tool.cursor == Qt.CursorShape.CrossCursor
    window.view.tool.click(u, v, LEFT, NO_KEY)
    b = window.controller.session.tracks[1]
    assert [[tuple(point) for point in prompt.points_px] for prompt in b.prompts] == [[(u, v)]]
    qtbot.waitUntil(lambda: not prompts.busy)


def test_during_a_run_the_panels_own_functions_change_nothing_either(window, qtbot, clip_in_odd_folder):
    """A key, or code, that reaches a function behind a button that is off."""
    clip = clip_in_odd_folder
    with Gate() as gate:
        track, objects = a_run_with_a_second_object(window, qtbot, clip, gate)
        press_track(qtbot, track, gate)
        objects.add_object()
        objects.set_mode("coarse")
        objects.set_fine_window(64)
        objects.remove_selected()
        objects.remove_button.click()
        b = window.controller.session.tracks[1]
        assert ids(window) == ["A", "B"] and (b.mode, b.fine_window_px) == ("fine", 120)
        let_run_end(qtbot, track, gate)
    assert track.jobs.status == "complete" and results_of(window).track_ids == ["A"]


# ---------------------------------------------------------------------------------------------
# Remove asks first


def test_remove_asks_before_tracked_frames_go_and_removes_only_on_yes(window, qtbot, clip_in_odd_folder, monkeypatch):
    clip, questions = clip_in_odd_folder, record_confirm(monkeypatch)
    track, objects = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB", end=10)
    run_to_end(qtbot, track)
    objects.prompts.select("A")
    objects.remove_button.click()
    (parent, text, action, on_confirmed), = questions.asked
    heading, _, rest = text.partition("\n")
    assert parent is window and action == "Remove"
    assert heading.endswith("?") and "A" in heading.rstrip("?").split()  # a question that names the object
    assert "6 tracked frames" in rest  # the clip 0 to 10 at step 2: frames 0, 2, 4, 6, 8 and 10
    # no answer, or Cancel: nothing is removed
    assert ids(window) == ["A", "B"] and results_of(window).track_ids == ["A", "B"]
    on_confirmed()
    assert ids(window) == ["B"] and results_of(window).track_ids == ["B"]
    assert objects.prompts.selected == "B"


def test_the_question_is_about_the_object_it_named_whatever_is_selected_at_the_answer(window, qtbot,
                                                                                    clip_in_odd_folder, monkeypatch):
    clip, questions = clip_in_odd_folder, record_confirm(monkeypatch)
    track, objects = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB", end=10)
    run_to_end(qtbot, track)
    objects.prompts.select("A")
    objects.remove_button.click()
    objects.prompts.select("B")  # the key 2, behind the dialog
    questions.asked[0][3]()
    assert ids(window) == ["B"] and results_of(window).track_ids == ["B"]


def test_an_object_without_results_is_removed_without_a_question(window, qtbot, clip_in_odd_folder, monkeypatch):
    clip, questions = clip_in_odd_folder, record_confirm(monkeypatch)
    track, objects = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB", end=10)
    objects.remove_button.click()  # B: clicked, not tracked yet
    assert ids(window) == ["A"] and questions.asked == []
    run_to_end(qtbot, track)
    objects.add_object()  # B again: added after the run, no points
    objects.remove_button.click()
    assert ids(window) == ["A"] and questions.asked == []
    assert results_of(window).track_ids == ["A"]
