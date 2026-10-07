"""Panel 6, Objects (SPEC 5, 10.1; task C4): the table, Add and Remove, mode and fine window, and
what the panel says about its state.

Expected values come from the spec and the design note: ids are A, B, ... with the colours of X14
(A yellow `#FFFF00`, B magenta `#FF00FF`); a new object is coarse with an automatic window and no
clicks; the status words are "no prompts", "ready", "tracked" and "ended" (SPEC 5); the panel is
done when at least one object has a click, and its hint line says the next step. Results that a
job would have left are made from the synthetic clip (`prompt_helpers.tracked_folder`). A disk of
`synthetic.disk_scene` has radius 12 px, so its fine window is clip(ceil(3 x 24), 96, 512) = 96 px
around its center (SPEC 6.3). Nothing is read from a pixel of text.
"""

from PySide6.QtCore import Qt

from gui_helpers import drawn
from outline_tracker import tracking
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_NPZ
from prompt_helpers import clicked_object, objects_panel, panel_with, tracked_folder
from tracking_helpers import center

ADD_HINT = "Click Add. Then click on one animal in the video. An outline appears."  # the window's, at start
CHECK = (" Check that each outline follows its animal. If the antennae matter, click on each antenna too. "
         "One click on the body leaves them out.")


def rows(panel) -> list[list[str]]:
    """The table's cells as texts, row by row."""
    table = panel.table
    return [[table.item(row, column).text() for column in range(table.columnCount())]
            for row in range(table.rowCount())]


def colours(panel) -> list[str]:
    """The colour swatch of each row, as `#RRGGBB`."""
    return [panel.table.item(row, 0).data(Qt.ItemDataRole.DecorationRole).name().upper()
            for row in range(panel.table.rowCount())]


def tool_buttons(panel):
    return panel.positive_button, panel.negative_button, panel.head_button


# ---------------------------------------------------------------------------------------------
# Add, Remove, and what can be used when


def test_the_panels_controls_are_in_panel_6_under_its_hint(window):
    panel = objects_panel(window)
    six = window.panels[5]
    assert (six.number, six.title) == (6, "Objects")
    assert six.body.itemAt(0).widget() is six.hint and six.body.itemAt(1).widget() is panel
    assert [panel.table.horizontalHeaderItem(column).text() for column in range(panel.table.columnCount())] == [
        "Object", "Mode", "Window", "Start", "Status"]
    assert [button.text() for button in (panel.add_button, panel.remove_button, *tool_buttons(panel))] == [
        "Add", "Remove", "Positive", "Negative", "Head"]
    assert all(button.isCheckable() for button in tool_buttons(panel))
    assert all(button.toolTip() for button in (panel.add_button, panel.remove_button, *tool_buttons(panel)))


def test_nothing_can_be_added_until_a_video_is_open(window, qtbot, disk_clip):
    panel = objects_panel(window)
    assert not panel.add_button.isEnabled() and not panel.remove_button.isEnabled()
    assert not any(button.isEnabled() for button in tool_buttons(panel))
    assert rows(panel) == [] and (window.panels[5].state, window.panels[5].hint.text()) == ("todo", ADD_HINT)
    panel_with(window, qtbot, disk_clip)
    assert panel.add_button.isEnabled()
    # still no object: nothing to remove, and no object a point could belong to
    assert not panel.remove_button.isEnabled() and not any(button.isEnabled() for button in tool_buttons(panel))
    assert not panel.mode_box.isEnabled() and not panel.window_box.isEnabled()


def test_add_makes_the_next_object_selects_it_and_chooses_positive(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    session, touched = window.controller.session, []
    window.controller.session_changed.connect(lambda: touched.append(True))
    panel.add_button.click()
    assert [(track.id, track.color, track.mode, track.fine_window_px, track.prompts) for track in session.tracks] == [
        ("A", "#FFFF00", "coarse", None, [])]
    assert touched == [True]  # the session's other parts were told, once
    assert panel.prompts.selected == "A" and panel.table.currentRow() == 0
    assert window.view.tool is panel.prompts.tools["positive"] and panel.positive_button.isChecked()
    assert panel.remove_button.isEnabled() and all(button.isEnabled() for button in tool_buttons(panel))
    panel.add_button.click()
    assert [(track.id, track.color) for track in session.tracks] == [("A", "#FFFF00"), ("B", "#FF00FF")]
    assert panel.prompts.selected == "B" and panel.table.currentRow() == 1
    assert touched == [True, True]


def test_remove_takes_the_selected_object_out_of_the_session_and_the_table(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip)
    panel.add_object()
    panel.add_object()
    touched = []
    window.controller.session_changed.connect(lambda: touched.append(True))
    panel.table.selectRow(1)  # a click on B's row
    assert panel.prompts.selected == "B"
    panel.remove_button.click()
    assert [track.id for track in window.controller.session.tracks] == ["A", "C"]
    assert [row[0] for row in rows(panel)] == ["A", "C"] and touched == [True]
    assert panel.prompts.selected == "C" and panel.table.currentRow() == 1  # the object that took B's row
    panel.remove_selected()
    panel.remove_selected()
    assert window.controller.session.tracks == [] and rows(panel) == []
    assert panel.prompts.selected is None and window.view.tool is None  # no object: no point tool
    assert not panel.remove_button.isEnabled() and not any(button.isEnabled() for button in tool_buttons(panel))
    panel.add_object()
    # the id is tracking's to choose (`next_track_id`): no run and no correction names A, so A is free again
    assert [(track.id, track.color) for track in window.controller.session.tracks] == [("A", "#FFFF00")]


def test_a_click_on_a_row_selects_that_object_for_the_next_point(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip, "A")
    panel.add_object()
    panel.table.selectRow(0)
    assert panel.prompts.selected == "A"
    u, v = center(disk_clip, "A", 0)
    assert panel.prompts.add_point(u + 30, v, 0)
    first, second = window.controller.session.tracks
    assert first.prompts[0].labels == [1, 0] and second.prompts == []


def test_removing_an_object_removes_its_points_and_its_preview_from_the_view(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip, "A")
    prompts = panel.prompts
    panel.add_object()
    assert prompts.add_point(*center(disk_clip, "B", 0), 1)
    qtbot.waitUntil(lambda: set(prompts.outlines) == {"A", "B"})
    prompts.select("A")
    panel.remove_selected()
    assert [track.id for track in window.controller.session.tracks] == ["B"]
    assert [mark.track_id for mark in prompts.marks] == ["B"]
    assert set(prompts.outlines) == {"B"} and set(prompts.graphics) == {"B"}
    assert drawn(window.view, (255, 255, 0)) is None  # nothing yellow is left on the picture
    qtbot.waitUntil(lambda: not prompts.busy)
    assert set(prompts.outlines) == {"B"}


# ---------------------------------------------------------------------------------------------
# The table


def test_the_table_shows_id_colour_mode_window_start_frame_and_status(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    window.navigation.step(3)  # frame 6
    panel.add_object()
    assert rows(panel) == [["A", "coarse", "–", "–", "no prompts"]] and colours(panel) == ["#FFFF00"]
    assert panel.prompts.add_point(*center(disk_clip, "A", 6), 1)
    assert rows(panel) == [["A", "coarse", "–", "6", "ready"]]
    panel.add_object()
    assert rows(panel) == [["A", "coarse", "–", "6", "ready"], ["B", "coarse", "–", "–", "no prompts"]]
    assert colours(panel) == ["#FFFF00", "#FF00FF"]
    assert all(not (panel.table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable) for column in range(5))


def test_the_table_follows_the_session(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip)
    session, controller = window.controller.session, window.controller
    tracking.add_object(session, "fine", 200)  # another part of the window changes the session
    controller.touch()
    assert rows(panel) == [["A", "coarse", "–", "0", "ready"], ["B", "fine", "200 px", "–", "no prompts"]]
    session.tracks[0].mode = "fine"
    controller.touch()
    assert rows(panel)[0] == ["A", "fine", "auto", "0", "ready"]
    session.tracks.clear()
    controller.touch()
    assert rows(panel) == [] and panel.prompts.selected is None


def test_an_object_with_results_is_tracked_and_removing_it_removes_its_results(window, qtbot, disk_clip, tmp_path):
    folder = tracked_folder(disk_clip, tmp_path / "run", ["A", "B"])
    panel = panel_with(window, qtbot, disk_clip)
    window.controller.run_folder = folder  # as the part that names the run folder will set it
    for name in "AB":
        panel.add_object()
        assert panel.prompts.add_point(*center(disk_clip, name, 0), 1)
    panel.add_object()
    assert panel.prompts.add_point(*center(disk_clip, "C", 0), 1)  # C has clicks and no results
    assert [row[4] for row in rows(panel)] == ["tracked", "tracked", "ready"]
    window.controller.session.tracks[1].ended_at = 40  # "End track here", a later panel's
    window.controller.touch()
    assert [row[4] for row in rows(panel)] == ["tracked", "ended", "ready"]
    # Remove reads results.npz as it is now and takes the object's records out of it
    panel.prompts.select("A")
    panel.remove_selected()
    assert [track.id for track in window.controller.session.tracks] == ["B", "C"]
    assert ResultsStore.load(folder / RESULTS_NPZ).track_ids == ["B"]
    assert [row[4] for row in rows(panel)] == ["ended", "ready"]


def test_a_removal_that_cannot_be_saved_is_refused_in_the_functions_words(window, qtbot, disk_clip, tmp_path):
    folder = tracked_folder(disk_clip, tmp_path / "run", ["A"])
    (folder / "results.new.npz").write_bytes(b"left by a job")  # results a job could not put in place
    panel, _ = clicked_object(window, qtbot, disk_clip)
    window.controller.run_folder = folder
    panel.remove_selected()
    assert [track.id for track in window.controller.session.tracks] == ["A"]  # nothing changed
    assert ResultsStore.load(folder / RESULTS_NPZ).track_ids == ["A"]
    kind, text = panel.prompts.message
    assert kind == "problem" and "results.new.npz" in text and "nothing was changed" in text
    assert panel.message_label.text() == text and not panel.message_label.isHidden()


# ---------------------------------------------------------------------------------------------
# Mode and fine window


def test_mode_and_fine_window_of_the_selected_object(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    panel.add_object()
    (track,) = window.controller.session.tracks
    touched = []
    window.controller.session_changed.connect(lambda: touched.append(True))
    assert panel.mode_box.isEnabled() and panel.mode_box.currentText() == "coarse"
    assert not panel.window_box.isEnabled() and not panel.auto_button.isEnabled()  # a window is for fine objects
    panel.mode_box.setCurrentText("fine")
    assert track.mode == "fine" and track.fine_window_px is None and touched == [True]
    assert rows(panel) == [["A", "fine", "auto", "–", "no prompts"]]
    assert panel.window_box.isEnabled() and panel.window_box.text() == "auto"
    panel.window_box.setValue(160)
    assert track.fine_window_px == 160 and rows(panel)[0][2] == "160 px" and panel.auto_button.isEnabled()
    panel.auto_button.click()  # the way back to automatic
    assert track.fine_window_px is None and rows(panel)[0][2] == "auto" and panel.window_box.text() == "auto"
    assert not panel.auto_button.isEnabled()
    panel.set_fine_window(120)
    panel.add_object()  # B is selected: the controls show B's values, and A keeps its own
    assert panel.mode_box.currentText() == "coarse" and not panel.window_box.isEnabled()
    assert (track.mode, track.fine_window_px) == ("fine", 120)
    panel.prompts.select("A")
    assert panel.mode_box.currentText() == "fine" and panel.window_box.value() == 120
    panel.set_mode("coarse")
    assert track.mode == "coarse" and rows(panel)[0][1:3] == ["coarse", "–"]


def test_the_mode_of_an_object_with_results_cannot_be_changed(window, qtbot, disk_clip, tmp_path):
    panel, _ = clicked_object(window, qtbot, disk_clip)
    window.controller.run_folder = tracked_folder(disk_clip, tmp_path / "run", ["A"])
    window.controller.touch()
    assert rows(panel)[0][4] == "tracked"
    panel.set_mode("fine")
    assert window.controller.session.tracks[0].mode == "coarse" and panel.mode_box.currentText() == "coarse"
    assert panel.prompts.message == ("problem", "Object A is tracked already, in coarse mode. To track the animal in "
                                                "fine mode, remove the object and add it again.")


def test_a_fine_object_shows_its_window_once_a_preview_gave_a_mask(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    prompts = panel.prompts
    panel.add_object()
    panel.set_mode("fine")
    assert window.controller.session.tracks[0].mode == "fine"
    assert prompts.windows == {}  # no mask yet: no window
    cu, cv = center(disk_clip, "A", 0)
    assert prompts.add_point(cu, cv, 1)
    qtbot.waitUntil(lambda: "A" in prompts.windows)
    # F = 24 px for a disk of radius 12 px, and 3 F = 72 px is under the smallest window: 96 px
    c0, r0, width, height = prompts.windows["A"]
    assert (width, height) == (96, 96)
    assert abs(c0 + 48 - cu) <= 1.0 and abs(r0 + 48 - cv) <= 1.0  # around the mask's center, on whole px
    panel.set_fine_window(120)
    c0, r0, width, height = prompts.windows["A"]
    assert (width, height) == (120, 120) and abs(c0 + 60 - cu) <= 1.0 and abs(r0 + 60 - cv) <= 1.0
    panel.set_fine_window(None)  # back to automatic
    assert prompts.windows["A"][2:] == (96, 96)
    panel.set_mode("coarse")
    assert prompts.windows == {} and "A" in prompts.outlines


# ---------------------------------------------------------------------------------------------
# State and hint


def test_the_panel_is_done_when_an_object_has_a_click_and_says_the_next_step(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    six, prompts = window.panels[5], panel.prompts
    assert (six.state, six.hint.text()) == ("todo", ADD_HINT)
    panel.add_object()
    assert (six.state, six.hint.text()) == (
        "todo", "Object A has no points. Click on the animal, or remove the object.")
    assert prompts.add_point(*center(disk_clip, "A", 0), 1)
    assert (six.state, six.hint.text()) == ("done", "1 object is ready to track." + CHECK)
    panel.add_object()  # one object has a click: done, and the hint names the object that has none
    assert (six.state, six.hint.text()) == (
        "done", "Object B has no points. Click on the animal, or remove the object.")
    assert prompts.add_point(*center(disk_clip, "B", 0), 1)
    assert (six.state, six.hint.text()) == ("done", "2 objects are ready to track." + CHECK)
    prompts.undo()
    prompts.select("A")
    prompts.undo()
    assert six.state == "todo"
    panel.remove_selected()
    panel.remove_selected()
    assert (six.state, six.hint.text()) == ("todo", ADD_HINT)
