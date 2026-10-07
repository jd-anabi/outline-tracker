"""Panel 8, the three corrections (SPEC 6.6, 10.1; task C7): Re-track from here, End track here and
Continue as new track, from the window. The table and going through it are in test_review.py.

Expected values:
- The dish clip tracked to frame 80 on every 2nd frame has the frames 0, 2, ..., 80: 41 of them;
  frame 40 is row 20.
- `Renaming` is `ExactFake` behind other names. {"B": "C"} from frame 40: the track B follows the
  animal C from there (a swap); after the test takes the names away it answers every track with
  its own animal, as the model does once the right animal is clicked. {"A2": "A"}: the piece A2
  is the animal A.
- Positions after a correction are the ground-truth table's (`tracking_helpers.table_truth`);
  "identical" is what results.npz held before the correction.
- Two tracks on the same animal have the same outline: CONTACT on every frame. B and C apart:
  CONTACT only where the true ellipses are within 3 px (`contact_frames`), which frame 80 is not.
- The words of a refusal are the function's own: the test calls the function on a copy of the
  session and compares.
`dialogs.confirm` is replaced by a function that answers at once; no test waits with a delay (a
stand-in is parked on a gate). Frames are video frame numbers; positions are px (SPEC 3.1).
"""

import copy
import shutil

import numpy as np
import pytest

from export_helpers import table
from outline_tracker import schema, tracking
from outline_tracker.export import export_all
from outline_tracker.fileio import new_name
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import Gate
from review_helpers import (GRID, assert_tracks_identical, confirming, contact_frames, dish_run,  # noqa: F401
                            frames_with, hint, listed, object_cells, opened_run, results_file, track_again,
                            tracked_window)
from track_helpers import Tracked, results_of, run_to_end, session_on_disk
from tracking_helpers import Renaming, center, table_truth

FRAMES = list(range(0, 81, 2))  # the clip of these tests: 41 frames
K = 40                          # the frame of the corrections: row 20
RUNNING = "Tracking is running. The flags are listed again when it has stopped."


def swapped(window, qtbot, clip):
    """A window in which A, B and C were tracked to frame 80, B following the animal C from frame
    40 on; afterwards the stand-in follows the animal that is clicked. Returns (the `ReviewPanel`,
    the `ObjectsPanel`, the `TrackPanel`)."""
    stand_in = Renaming(clip, {"B": "C"}, from_frame=K)
    panels = tracked_window(window, qtbot, clip, Tracked(stand_in))
    stand_in.names = {}
    return panels


def click_on(window, objects, clip, track_id: str, frame: int, animal: str | None = None) -> None:
    """Go to `frame`, select `track_id` and click on the true center of `animal` (the track's own
    if None) there."""
    window.show_frame(frame)
    objects.prompts.select(track_id)
    assert objects.prompts.add_point(*center(clip, animal or track_id, frame), 1), objects.prompts.message


# ---------------------------------------------------------------------------------------------
# When the buttons can be used


def test_the_three_buttons_need_a_selected_object_that_has_results(window, qtbot, clip_in_odd_folder):
    review, objects, _ = tracked_window(window, qtbot, clip_in_odd_folder, ids="AB", end=20)
    three = (review.retrack_button, review.end_button, review.continue_button)
    objects.prompts.select(None)
    assert not any(button.isEnabled() for button in three)
    reason = review.retrack_button.toolTip()
    assert "object" in reason and hint(window).endswith(reason)  # the reason is the hint, and the tooltip
    assert review.end_button.toolTip() == review.continue_button.toolTip() == reason
    objects.prompts.select("B")
    assert all(button.isEnabled() for button in three)
    assert reason not in hint(window)
    # an object that was added after the run has nothing to fix yet
    objects.add_object()
    assert objects.prompts.selected == "C"
    assert not any(button.isEnabled() for button in three)
    assert "C" in review.end_button.toolTip() and hint(window).endswith(review.end_button.toolTip())


def test_during_a_run_the_three_buttons_are_off_and_the_hint_says_why(window, qtbot, clip_in_odd_folder,
                                                                       monkeypatch):
    clip = clip_in_odd_folder
    confirming(monkeypatch)
    listings, real = [], tracking.flags_table
    monkeypatch.setattr(tracking, "flags_table", lambda folder: listings.append(folder) or real(folder))
    with Gate() as gate:
        # the first run tracks 11 frames (0 to 20); the run of the correction parks on its 2nd frame
        review, objects, _ = tracked_window(window, qtbot, clip, Tracked(ExactFake(clip), gate, park_at={13}),
                                            ids="AB", end=20)
        three = (review.retrack_button, review.end_button, review.continue_button)
        click_on(window, objects, clip, "B", 10)
        qtbot.waitUntil(lambda: not objects.prompts.busy)
        assert all(button.isEnabled() for button in three)
        review.retrack()
        qtbot.waitUntil(gate.parked.is_set)
        assert review.jobs.running
        assert not any(button.isEnabled() for button in three)
        assert hint(window) == RUNNING and {button.toolTip() for button in three} == {RUNNING}
        assert len(listings) == 1  # the flags of the first run: none are listed between a correction and its run
        gate.open()
        track_again(qtbot, review)
    assert len(listings) == 2  # and they are listed again when that run has ended
    assert all(button.isEnabled() for button in three) and hint(window) != RUNNING
    assert results_of(window).arrays("B").frames.tolist() == list(range(0, 21, 2))


def test_without_a_model_only_re_track_is_off_and_a_track_can_still_be_ended(window, qtbot, dish_clip, dish_run,
                                                                             tmp_path, monkeypatch):
    asked = confirming(monkeypatch)
    review = opened_run(window, qtbot, dish_run, tmp_path / "run", model=False)
    review.prompts.select("B")
    window.show_frame(K)
    assert not review.retrack_button.isEnabled()
    assert "model" in review.retrack_button.toolTip() and hint(window).endswith(review.retrack_button.toolTip())
    assert review.end_button.isEnabled() and review.continue_button.isEnabled()
    review.retrack()  # the slot itself refuses too
    assert asked.asked == [] and window.controller.session.corrections == []
    review.end_track()
    assert results_of(window).arrays("B").frames.tolist() == list(range(0, K + 1, 2))
    listed(qtbot, review)
    # up to frame 40 B and C are far apart, and from there on only C is left: no CONTACT anywhere
    assert set(range(0, K + 1, 2)) <= set(contact_frames(dish_clip, GRID)[1])
    assert frames_with(review.rows, "B", "CONTACT") == frames_with(review.rows, "C", "CONTACT") == []


# ---------------------------------------------------------------------------------------------
# Re-track from here


def test_re_track_without_a_click_on_this_frame_chooses_the_positive_tool_and_says_to_click(window, qtbot,
                                                                                            clip_in_odd_folder,
                                                                                            monkeypatch):
    asked = confirming(monkeypatch)
    review, objects, _ = tracked_window(window, qtbot, clip_in_odd_folder, ids="AB", end=20)
    before = results_file(window)
    objects.prompts.select("B")
    objects.prompts.choose_tool(None)
    window.show_frame(10)
    review.retrack()
    assert asked.asked == [] and not review.jobs.running
    assert objects.prompts.tool_kind == "positive" and objects.positive_button.isChecked()
    said = hint(window)
    assert "Click on animal B" in said and "frame 10" in said and "Re-track from here again" in said
    assert window.controller.session.corrections == []
    assert_tracks_identical(results_file(window), before, "AB")
    # the sentence is about that object on that frame: it goes when another frame shows
    window.show_frame(12)
    assert hint(window) != said


def test_re_track_from_here_changes_only_frames_from_k_of_that_track(window, qtbot, clip_in_odd_folder,
                                                                     monkeypatch):
    clip = clip_in_odd_folder
    asked = confirming(monkeypatch)
    review, objects, track = swapped(window, qtbot, clip)
    before, arrays = results_file(window), results_of(window).arrays("B")
    wrong_u, wrong_v, _ = table_truth(clip, "C", FRAMES[20:])
    assert np.abs(arrays.u[20:] - wrong_u).max() <= 0.01  # the premise: B is on the animal C from row 20
    assert 80 in frames_with(review.rows, "B", "CONTACT")  # and so B and C have one outline to the end

    click_on(window, objects, clip, "B", K)
    review.retrack()
    assert asked.asked == [("Re-track B from frame 40?\nThe results of B from frame 40 to the end are replaced. "
                            "Other tracks do not change.", "Re-track from here")]
    # the correction was made and saved before the run that it started
    assert review.store.arrays("B").frames.tolist() == FRAMES[:20]  # `Edit.store`: B without its frames from 40
    assert session_on_disk(window)["corrections"][-1]["action"] == "retrack"
    track_again(qtbot, review)

    after, store = results_file(window), results_of(window)
    assert_tracks_identical(after, before, "AC")  # other tracks are untouched
    again = store.arrays("B")
    assert again.frames.tolist() == FRAMES
    for name in ("u", "v", "area_px", "outline_px"):  # B before frame 40 is what it was
        np.testing.assert_array_equal(getattr(again, name)[:20], getattr(arrays, name)[:20], err_msg=name)
    u, v, _ = table_truth(clip, "B", FRAMES[20:])  # and from frame 40 on it is the animal B
    assert np.abs(again.u[20:] - u).max() <= 0.01 and np.abs(again.v[20:] - v).max() <= 0.01
    assert np.abs(again.u[20:] - arrays.u[20:]).max() > 5  # which is not where it was

    session = window.controller.session
    (correction,) = session.corrections
    assert (correction.action, correction.tracks, correction.frame) == ("retrack", ["B"], K)
    assert session.complete is True and session_on_disk(window)["complete"] is True
    assert [(run["tracks"], run["start_frame"]) for run in session_on_disk(window)["runs"]] == [
        (["A", "B", "C"], 0), (["B"], K)]

    # the table, panel 6 and the picture show the new state
    surely, never = contact_frames(clip, FRAMES)
    flagged = frames_with(review.rows, "B", "CONTACT")
    assert set(surely) <= set(flagged) and not set(never) & set(flagged) and 80 in never
    assert review.rows == [row for row in tracking.flags_table(window.controller.run_folder)
                           if row[3] in ("LOST", "JUMP", "SIZE", "CONTACT", "EDGE", "MULTI")]
    assert object_cells(objects, "B")[-1] == "tracked"
    window.show_frame(80)
    assert np.allclose(track.overlays.shown["B"].center, center(clip, "B", 80), atol=0.3)


def test_cancel_in_the_question_changes_nothing(window, qtbot, clip_in_odd_folder, monkeypatch):
    clip = clip_in_odd_folder
    asked = confirming(monkeypatch, answer=False)
    review, objects, _ = tracked_window(window, qtbot, clip, ids="AB", end=20)
    click_on(window, objects, clip, "B", 10)
    before = results_file(window)
    review.retrack()
    review.end_track()
    assert [action for _, action in asked.asked] == ["Re-track from here", "End track here"]
    assert asked.asked[1][0] == ("End track B at frame 10?\nThe results of B after frame 10 are removed. "
                                 "Other tracks do not change.")
    assert not review.jobs.running and window.controller.session.corrections == []
    assert window.controller.session.tracks[1].ended_at is None
    assert_tracks_identical(results_file(window), before, "AB")


def test_a_refusal_is_shown_in_the_functions_own_words_and_nothing_changes(window, qtbot, clip_in_odd_folder,
                                                                           monkeypatch):
    clip = clip_in_odd_folder
    confirming(monkeypatch)
    review, objects, _ = tracked_window(window, qtbot, clip, ids="AB", end=20)
    click_on(window, objects, clip, "B", 10)
    folder = window.controller.run_folder
    # results that a job left aside because results.npz was locked: every correction refuses until that is settled
    shutil.copyfile(folder / schema.RESULTS_NPZ, new_name(folder / schema.RESULTS_NPZ))
    before, session = results_file(window), window.controller.session
    with pytest.raises(RuntimeError) as refused:
        tracking.end_track(copy.deepcopy(session), None, "B", 10, folder)
    review.end_track()
    assert review.message.kind == "problem" and review.message.text() == str(refused.value)
    review.retrack()
    assert review.message.kind == "problem" and review.message.text() == str(refused.value)
    assert not review.jobs.running and session.corrections == [] and session.tracks[1].ended_at is None
    assert [prompt.frame for prompt in session.tracks[1].prompts] == [0, 10]
    assert_tracks_identical(results_file(window), before, "AB")
    # once it is settled the correction is made, and the message goes
    new_name(folder / schema.RESULTS_NPZ).unlink()
    review.end_track()
    assert review.message.kind is None and session.tracks[1].ended_at == 10


# ---------------------------------------------------------------------------------------------
# End track here, Continue as new track


def test_end_track_here_then_continue_as_new_track_gives_a_and_a2_in_the_table_and_in_the_export(
        window, qtbot, clip_in_odd_folder, monkeypatch):
    clip = clip_in_odd_folder
    asked = confirming(monkeypatch)
    review, objects, track = tracked_window(window, qtbot, clip, Tracked(Renaming(clip, {"A2": "A"})))
    before = results_file(window)
    objects.prompts.select(None)
    assert review.piece_label.text() == ""  # no object is selected
    objects.prompts.select("A")
    assert review.piece_label.text() == "The new track is A2."

    # End track here: A's rows stop at frame 40
    window.show_frame(K)
    review.end_track()
    assert asked.asked == [("End track A at frame 40?\nThe results of A after frame 40 are removed. "
                            "Other tracks do not change.", "End track here")]
    session = window.controller.session
    assert results_of(window).arrays("A").frames.tolist() == FRAMES[:21]
    assert review.store.arrays("A").frames.tolist() == FRAMES[:21]  # `Edit.store`
    assert session.tracks[0].ended_at == K and session_on_disk(window)["tracks"][0]["ended_at"] == K
    assert [(done.action, done.tracks, done.frame) for done in session.corrections] == [("end", ["A"], K)]
    assert_tracks_identical(results_file(window), before, "BC")
    assert object_cells(objects, "A")[-1] == "ended"
    window.show_frame(K + 2)
    assert "A" not in track.overlays.shown  # the picture: no outline of A after its end

    # Continue as new track: the piece A2, selected, with the Positive tool
    review.continue_as_new()
    assert [one.id for one in session.tracks] == ["A", "B", "C", "A2"]
    piece = session.tracks[3]
    assert (piece.start_frame, piece.mode, piece.color, piece.prompts) == (K + 2, "coarse", session.tracks[0].color, [])
    assert objects.prompts.selected == "A2" and objects.prompts.tool_kind == "positive"
    said = hint(window)
    assert "Click on animal A2" in said and "Track" in said and "panel 7" in said
    assert len(asked.asked) == 1  # nothing is lost, so nothing is asked
    assert [(done.action, done.tracks, done.frame) for done in session.corrections][1:] == [
        ("new_piece", ["A2"], K + 2)]
    assert object_cells(objects, "A2")[0] == "A2"

    # a click on the animal, then Track
    assert objects.prompts.add_point(*center(clip, "A", K + 2), 1)
    qtbot.waitUntil(lambda: not objects.prompts.busy)
    run_to_end(qtbot, track)
    assert track.jobs.status == "complete"
    listed(qtbot, review)
    store = results_of(window)
    assert store.arrays("A").frames.tolist() == FRAMES[:21] and store.arrays("A2").frames.tolist() == FRAMES[21:]
    u, v, _ = table_truth(clip, "A", FRAMES[21:])
    assert np.abs(store.arrays("A2").u - u).max() <= 0.01 and np.abs(store.arrays("A2").v - v).max() <= 0.01
    assert_tracks_identical(results_file(window), before, "BC")

    # A and A2 in panel 6's table and in the flags table (every tracked frame has HEADGUESS: no head click)
    assert object_cells(objects, "A")[-1] == "ended" and object_cells(objects, "A2")[-1] == "tracked"
    review.shape_box.setChecked(True)
    assert frames_with(review.rows, "A", "HEADGUESS") == FRAMES[:21]
    assert frames_with(review.rows, "A2", "HEADGUESS") == FRAMES[21:]
    assert [name for name in dict.fromkeys(row[0] for row in review.rows)] == ["A", "A2", "B", "C"]
    # and in the export
    window.controller.save_now()
    export_all(window.controller.run_folder, log=lambda text: None)
    positions = table(window.controller.run_folder / schema.POSITIONS_CSV)
    assert positions[positions.track_id == "A"].frame.tolist() == FRAMES[:21]
    assert positions[positions.track_id == "A2"].frame.tolist() == FRAMES[21:]
    assert sorted(set(positions.track_id)) == ["A", "A2", "B", "C"]
    # the next piece of this animal would be A3, whichever of the two is selected
    objects.prompts.select("A")
    assert review.piece_label.text() == "The new track is A3."


def test_a_frame_off_the_grid_is_moved_onto_it_which_is_said_and_shown(window, qtbot, clip_in_odd_folder):
    review, objects, _ = tracked_window(window, qtbot, clip_in_odd_folder, ids="AB", end=20)
    objects.prompts.select("A")
    window.view.show_frame(11)  # a frame between two frames of the clip, as no button of the bar gives it
    review.continue_as_new()
    piece = window.controller.session.tracks[2]
    assert (piece.id, piece.start_frame) == ("A2", 12)
    assert review.message.kind == "warning"
    assert review.message.text() == ("Frame 11 is not a frame of the clip (frames 0 to 20, every 2): frame 12 is "
                                     "used instead.")
    assert window.view.frame == window.navigation.frame == 12
    assert objects.prompts.selected == "A2" and "Click on animal A2" in hint(window)
