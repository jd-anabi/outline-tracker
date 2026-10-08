"""Panel 8, Review and fix (SPEC 9, 10.1; task C7): the flags table, going to a flag, and what the
panel says. The three corrections are in tests/gui/test_review_fix.py.

Expected values:
- The dish scene's B and C are two ellipses on straight paths that pass each other in frame 60
  with a gap of 0.6 px. Their true gap on every frame comes from the two true ellipses
  (`export_helpers.ellipse_gap`); CONTACT is "within 3 px" here. Frames whose true gap is at most
  2.7 px must be listed for B and for C, frames whose gap is at least 3.3 px must not. A swims
  along the wall, far from both.
- The table is `tracking.flags_table(run_folder)`: the rows the panel lists are compared with that
  function's rows for the same folder, limited to the codes the panel is to list.
- t = frame / fps_true: frame 60 at 240 frames per second is 0.250 s.
- A body 14.5 px long is too small for shape numbers (LOWRES on every frame), and without a head
  click the head is a guess on every frame (HEADGUESS): SPEC 9.
- A single disk of 12 px radius that moves 0.5 px per frame inside the frame has no flag on its
  position: no neighbour, 3.9 mm/s, the same area on every frame.
The texts are the design note's. Nothing is read from a pixel of text, and no test waits with a
delay. Frames are video frame numbers; times are s.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QPushButton, QTableView

from gui_helpers import picture
from outline_tracker import schema, tracking
from prompt_helpers import Gate, gui_thread, this_thread
from review_helpers import (FPS, GRID, contact_frames, current_row, frames_with, give_scale, hint, listed, listed_by,
                            opened_run, review_panel, shown_rows, tracked_window, written_since)
from track_helpers import NAME, Tracked, own_copy, run_log

LISTED = ("LOST", "JUMP", "SIZE", "CONTACT", "EDGE", "MULTI")  # what the table lists by default (task C7)
MORE = ("LOWRES", "ORIENT", "HEADGUESS")                       # what "Show shape flags too" adds
NOT_STARTED = "Track first. The flags appear here."
NO_FLAGS = "No flags on positions. Play the video once and look at the outlines."
LISTING = "The flags are being listed."
LEFT = Qt.MouseButton.LeftButton


def of_the_function(folder, codes=LISTED) -> list[tuple]:
    """The rows of `tracking.flags_table(folder)` whose code is one of `codes`."""
    return [row for row in tracking.flags_table(folder) if row[3] in codes]


def row_of(review, track_id: str, frame: int, code: str = "CONTACT") -> int:
    """The table's row for that flag of that track on that frame."""
    (row,) = [row for row, (name, at, _, has) in enumerate(review.rows) if (name, at, has) == (track_id, frame, code)]
    return row


# ---------------------------------------------------------------------------------------------
# The panel's parts, and what it says before anything is tracked


def test_the_controls_are_in_panel_8_under_its_hint(window):
    review = review_panel(window)
    panel = window.panels[7]
    assert panel.body.itemAt(0).widget() is panel.hint and panel.body.itemAt(1).widget() is review
    assert isinstance(review.table, QTableView)
    model = review.table.model()
    assert [model.headerData(column, Qt.Orientation.Horizontal) for column in range(model.columnCount())] == [
        "Track", "Frame", "t", "Flag"]
    assert model.rowCount() == 0
    # the spec's labels, letter for letter
    buttons = {button.text(): button for button in review.findChildren(QPushButton)}
    assert set(buttons) == {"Previous flag", "Next flag", "Re-track from here", "End track here",
                            "Continue as new track"}
    assert buttons["Next flag"] is review.next_button and buttons["Previous flag"] is review.previous_button
    assert buttons["Re-track from here"] is review.retrack_button and buttons["End track here"] is review.end_button
    assert buttons["Continue as new track"] is review.continue_button
    assert review.end_button.property("kind") == "destructive"
    assert review.retrack_button.property("kind") != "primary"  # never primary: it replaces results
    assert not any(button.autoDefault() for button in buttons.values())  # no button reacts to Enter
    assert all(button.toolTip() for button in buttons.values())
    boxes = {box.text(): box for box in review.findChildren(QCheckBox)}
    assert set(boxes) == {"Show shape flags too", "Only the selected object"}
    assert boxes["Show shape flags too"] is review.shape_box and not review.shape_box.isChecked()
    assert boxes["Only the selected object"] is review.selected_box and not review.selected_box.isChecked()


def test_without_results_the_panel_is_not_started_and_every_button_is_off(window, qtbot, dish_clip, tmp_path):
    review = review_panel(window)
    buttons = review.findChildren(QPushButton)

    def nothing_yet():
        assert window.panels[7].state == "todo" and window.panels[7].state_label.text() == "Not started"
        assert hint(window) == NOT_STARTED
        assert review.rows == [] and not any(button.isEnabled() for button in buttons)
        assert all(button.toolTip() for button in buttons)  # a button that is off says why

    nothing_yet()  # no video
    window.controller.set_student(NAME)
    picture(window, qtbot, own_copy(dish_clip, tmp_path))
    qtbot.waitUntil(lambda: review.worker.ready)
    window.controller.save_now()
    listed(qtbot, review)
    nothing_yet()  # a video and a run folder, nothing tracked
    assert review.retrack_button.toolTip() == NOT_STARTED


# ---------------------------------------------------------------------------------------------
# The table


def test_the_table_lists_the_contact_rows(window, qtbot, dish_clip, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    surely, never = contact_frames(dish_clip, GRID)
    assert 60 in surely and len(surely) >= 5 and len(never) >= 40  # the scene's premise
    for track_id in "BC":
        flagged = frames_with(review.rows, track_id, "CONTACT")
        assert set(surely) <= set(flagged)
        assert not set(never) & set(flagged)
    assert frames_with(review.rows, "A", "CONTACT") == []
    assert frames_with(review.rows, "B", "CONTACT") == frames_with(review.rows, "C", "CONTACT")  # both tracks get it


def test_the_table_is_flags_table_for_the_position_flags_by_track_then_frame(window, qtbot, dish_run, tmp_path):
    folder = tmp_path / "run"
    review = opened_run(window, qtbot, dish_run, folder)
    assert review.rows == of_the_function(folder)
    assert review.rows and not any(code in MORE for *_, code in review.rows)
    keys = [(name, frame) for name, frame, _, _ in review.rows]
    assert keys == sorted(keys)  # by track, then frame
    # every cell: the track, the frame, t with its unit, the code
    assert shown_rows(review) == [(name, str(frame), f"{frame / FPS:.3f} s", code)
                                  for name, frame, _, code in review.rows]
    assert shown_rows(review)[row_of(review, "B", 60)] == ("B", "60", "0.250 s", "CONTACT")


def test_a_code_shows_with_its_meaning(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    model, row = review.table.model(), row_of(review, "B", 60)
    meaning = "identities may swap here: check"  # SPEC 9, "meaning for the student"
    for column in range(model.columnCount()):
        assert meaning in model.data(model.index(row, column), Qt.ItemDataRole.ToolTipRole)
    review.go_to_row(row)
    assert review.meaning_label.text() == f"CONTACT: {meaning}"


def test_show_shape_flags_too_adds_lowres_orient_and_headguess(window, qtbot, dish_run, tmp_path):
    folder = tmp_path / "run"
    review = opened_run(window, qtbot, dish_run, folder)
    position_rows = list(review.rows)
    review.shape_box.setChecked(True)
    assert review.rows == of_the_function(folder, LISTED + MORE) == tracking.flags_table(folder)
    for track_id in "ABC":  # on every tracked frame of a small object without a head click
        assert frames_with(review.rows, track_id, "LOWRES") == GRID
        assert frames_with(review.rows, track_id, "HEADGUESS") == GRID
    assert review.table.model().rowCount() == len(review.rows) > len(position_rows)
    review.shape_box.setChecked(False)
    assert review.rows == position_rows


def test_only_the_selected_object_limits_the_table_to_it(window, qtbot, dish_run, tmp_path):
    folder = tmp_path / "run"
    review = opened_run(window, qtbot, dish_run, folder)
    everything = of_the_function(folder)
    review.prompts.select("C")
    assert review.rows == everything  # the box is off
    review.selected_box.setChecked(True)
    assert review.rows == [row for row in everything if row[0] == "C"] and review.rows
    review.prompts.select("B")
    assert review.rows == [row for row in everything if row[0] == "B"] and review.rows
    review.prompts.select(None)
    assert review.rows == everything  # no object: nothing to limit it to


def test_the_rows_of_the_frame_shown_are_marked(window, qtbot, dish_run, tmp_path):
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    model = review.table.model()

    def marked():
        return [model.data(model.index(row, 0), Qt.ItemDataRole.BackgroundRole) is not None
                for row in range(model.rowCount())]

    window.show_frame(60)
    assert marked() == [frame == 60 for _, frame, _, _ in review.rows] and sum(marked()) >= 2  # B and C
    window.show_frame(0)
    assert not any(marked())


def test_the_flags_are_listed_in_the_worker_thread(window, qtbot, dish_run, tmp_path, monkeypatch):
    # Listing 10 tracks of 1,200 frames takes seconds: that is no work for the GUI thread (task C7).
    threads, real = [], tracking.flags_table

    def watched(folder):
        threads.append(this_thread())
        return real(folder)

    monkeypatch.setattr(tracking, "flags_table", watched)
    review = opened_run(window, qtbot, dish_run, tmp_path / "run")
    assert review.rows and threads and gui_thread() not in threads


def test_without_a_model_the_flags_are_listed_all_the_same(window, qtbot, dish_run, tmp_path):
    folder = tmp_path / "run"
    review = opened_run(window, qtbot, dish_run, folder, model=False)
    assert review.rows == of_the_function(folder) and 60 in frames_with(review.rows, "B", "CONTACT")


def test_without_a_model_the_flags_are_listed_outside_the_gui_thread_too(window, qtbot, dish_run, tmp_path,
                                                                         monkeypatch):
    # The worker takes a task only while its model is ready. The seconds a large run takes are no
    # work for the GUI thread all the same (task C7, ruling 2).
    calls = listed_by(monkeypatch)
    review = opened_run(window, qtbot, dish_run, tmp_path / "run", model=False)
    assert review.rows and calls.threads and gui_thread() not in calls.threads


def test_without_a_model_the_window_goes_on_while_the_flags_are_listed(window, qtbot, dish_run, tmp_path,
                                                                       monkeypatch):
    folder = tmp_path / "run"
    with Gate() as gate:
        calls = listed_by(monkeypatch, gate)
        review = opened_run(window, qtbot, dish_run, folder, model=False, wait=False)
        qtbot.waitUntil(lambda: bool(calls.threads))
        assert gui_thread() not in calls.threads
        qtbot.waitUntil(gate.parked.is_set)  # the listing has begun, and stands still
        assert review.listing.pending and review.rows == []
        assert hint(window).startswith(LISTING)
        window.show_frame(60)  # the window does not wait for it
        assert window.view.frame == window.navigation.frame == 60
        gate.open()
        listed(qtbot, review)
    assert review.rows == [row for row in calls.real(folder) if row[3] in LISTED]
    assert 60 in frames_with(review.rows, "B", "CONTACT") and not hint(window).startswith(LISTING)


# ---------------------------------------------------------------------------------------------
# A listing that goes wrong


@pytest.mark.parametrize("model", [True, False])  # in the worker thread; in the listing's own thread
def test_an_error_while_listing_is_one_plain_line_and_its_trace_goes_to_run_log(window, qtbot, dish_run, tmp_path,
                                                                                monkeypatch, model):
    folder = tmp_path / "run"
    calls = listed_by(monkeypatch, error=ZeroDivisionError("division by zero"))
    review = opened_run(window, qtbot, dish_run, folder, model=model)
    assert calls.threads and gui_thread() not in calls.threads
    # nothing was listed, which is not "no flags": the run is complete, and the panel is not done
    assert window.controller.session.complete and review.rows == []
    assert window.panels[7].state == "attention"
    said = hint(window)
    assert said == "The flags could not be listed: division by zero. The details are in run.log in the run folder."
    # the trace: kept by the worker and written to run.log, never shown (SPEC 10.2)
    (trace,) = review.worker.traces
    lines = trace.splitlines()
    assert lines[0] == "The flags of the run folder run could not be listed."
    assert lines[1] == "Traceback (most recent call last):" and lines[-1] == "ZeroDivisionError: division by zero"
    assert trace in run_log(folder)
    for visible in (said, review.message.text(), window.statusBar().currentMessage()):
        assert "Traceback" not in visible and "ZeroDivisionError" not in visible


def test_a_file_that_is_gone_while_the_flags_are_listed_is_a_problem_not_no_flags(window, qtbot, dish_run, tmp_path,
                                                                                  monkeypatch):
    # as when results.npz is taken away between the look at the folder and the reading of the file
    folder = tmp_path / "run"
    gone = FileNotFoundError(2, "No such file or directory", str(folder / schema.RESULTS_NPZ))
    listed_by(monkeypatch, error=gone)
    review = opened_run(window, qtbot, dish_run, folder)
    assert (folder / schema.SESSION_JSON).is_file() and window.controller.session.complete
    assert review.rows == [] and window.panels[7].state == "attention"
    said = hint(window)
    assert said == ("The flags could not be listed: results.npz could not be read. The details are in run.log in "
                    "the run folder.")
    assert str(folder) not in said  # a file's name, never its folder
    (trace,) = review.worker.traces
    assert trace.splitlines()[-1].startswith("FileNotFoundError: [Errno 2] No such file or directory")
    assert trace in run_log(folder)


def test_results_without_session_json_are_a_problem_until_the_session_is_saved_again(window, qtbot, dish_run,
                                                                                    tmp_path):
    folder = tmp_path / "run"
    review = opened_run(window, qtbot, dish_run, folder)
    rows, panel = list(review.rows), window.panels[7]
    assert rows and window.controller.session.complete
    (folder / schema.SESSION_JSON).unlink()
    with pytest.raises(FileNotFoundError):
        tracking.flags_table(folder)
    written_since(folder / schema.RESULTS_NPZ)  # the results are other ones: the flags are listed again
    review.listing.sync()
    listed(qtbot, review)
    assert review.rows == [] and panel.state == "attention"  # not done: the flags could not be listed
    assert hint(window) == ("session.json is not in the run folder, so the flags cannot be listed. Save the session "
                            "(File > Save session).")
    assert review.worker.traces == []  # the sentence says all there is to it
    # with the file back the flags are listed again, without anyone asking
    window.controller.touch()
    window.controller.save_now()
    assert (folder / schema.SESSION_JSON).is_file()
    listed(qtbot, review)
    assert review.rows == rows and hint(window).startswith(f"{len(rows)} flags on ")


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


# ---------------------------------------------------------------------------------------------
# What the panel says


def test_after_a_run_the_panel_needs_attention_with_the_number_of_flags(window, qtbot, clip_in_odd_folder):
    review, _, _ = tracked_window(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    rows = of_the_function(folder)
    surely, _ = contact_frames(clip_in_odd_folder, range(0, 81, 2))
    assert review.rows == rows and len(rows) >= 2 * len(surely) > 0
    tracks = {name for name, *_ in rows}
    assert {"B", "C"} <= tracks
    panel = window.panels[7]
    assert panel.state == "attention" and panel.state_label.text() == "Needs attention"
    assert hint(window).startswith(f"{len(rows)} flags on {len(tracks)} tracks. Click a row to go to its frame.")


def test_complete_tracking_without_a_position_flag_is_done(window, qtbot, disk_clip, tmp_path):
    clip = own_copy(disk_clip, tmp_path)
    review, _, _ = tracked_window(window, qtbot, clip, segmenter=Tracked(), ids="A", end=20)
    panel = window.panels[7]
    assert review.rows == []
    assert panel.state == "done" and panel.state_label.text() == "Done"
    assert hint(window).startswith(NO_FLAGS)
    # tracking that is not complete is not done, flags or not (as a cancelled run leaves the session)
    window.controller.session.complete = False
    window.controller.touch()
    assert panel.state == "attention" and "not complete" in hint(window)


def test_without_a_scale_or_fps_true_the_table_is_empty_and_the_hint_is_the_functions_sentence(window, qtbot,
                                                                                                clip_in_odd_folder):
    review, _, _ = tracked_window(window, qtbot, clip_in_odd_folder, scale=False)
    folder = window.controller.run_folder
    with pytest.raises(ValueError, match="no scale yet") as refused:
        tracking.flags_table(folder)
    assert review.rows == [] and review.table.model().rowCount() == 0
    assert hint(window) == str(refused.value)
    assert window.panels[7].state == "attention"
    # with the scale the flags are there, as soon as the changed session is on disk: nobody asks for them
    give_scale(window, save=False)
    qtbot.waitUntil(lambda: bool(review.rows), timeout=30_000)
    assert not review.listing.pending and 60 in frames_with(review.rows, "B", "CONTACT")
    assert hint(window).startswith(f"{len(review.rows)} flags on ")
    # and without fps_true they are not
    window.controller.session.time.fps_true = None
    window.controller.touch()
    window.controller.save_now()
    listed(qtbot, review)
    with pytest.raises(ValueError, match="fps_true is not known yet") as refused:
        tracking.flags_table(folder)
    assert review.rows == [] and hint(window) == str(refused.value)
