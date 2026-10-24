"""The correction "Continue as new track" of SPEC 6.6, the ids, and the two prompt edits: `new_piece`, `next_track_id`,
`add_prompt` and `undo_prompt`, on the results store and on what a run makes of the new track.
"Re-track from here" (`retrack_from`) and "End track here" (`end_track`) are in tests/tracking/test_corrections.py.
The other edit functions are in tests/tracking/test_tracking_edit.py.

Expected values: the ground-truth table of `dish_clip` (120 frames; the sessions take every 2nd
frame, so the grid is 0, 2, ..., 118), frame counts worked out from that grid, and "identical":
what was in results.npz before the correction. Frames are video frame numbers; positions are px in
Tracker's convention (SPEC 3.1), derived x and y in mm (y up).
"""

import re
from itertools import product
from string import ascii_uppercase

import numpy as np
import pytest
from corrections_helpers import assert_recorded_now, assert_rows_identical, assert_tracks_identical
from results_helpers import read_npz
from tracking_helpers import (
    GRID,
    Renaming,
    center,
    clicks_at,
    make_session,
    run,
    table_truth,
)

from outline_tracker.measure.results import ResultsStore
from outline_tracker.core.session import Clip, Correction, Prompt, RunRecord, Session, Track
from outline_tracker.tracking.edit import (
    add_object,
    add_prompt,
    end_track,
    new_piece,
    next_track_id,
    pending_runs,
    undo_prompt,
)

TRACK_ID = re.compile(r"^[A-Z]+[0-9]*$")  # SPEC 3.4


# ---------------------------------------------------------------------------------------------
# Continue as new track


def test_end_track_and_new_piece_give_a_and_a2_tracked_from_m(tracked, dish_clip):
    folder, session, store = tracked
    before = read_npz(folder / "results.npz")
    end_track(session, store, "A", 58, folder)
    clicks = clicks_at(dish_clip, "A2", 70, on="A")
    done = new_piece(session, "A", 70, clicks)
    assert (done.track_ids, done.frame, done.moved, done.note) == (("A2",), 70, False, "")
    assert done.results_file is None  # a new piece has no results yet

    assert [a_track.id for a_track in session.tracks] == ["A", "B", "C", "A2"]
    parent, piece = session.tracks[0], session.tracks[-1]
    assert (piece.mode, piece.color, piece.start_frame, piece.ended_at) == ("coarse", parent.color, 70, None)
    assert piece.prompts == [clicks] and piece.prompts[0] is not clicks
    assert piece.head_px is None and piece.fine_window_px is None
    correction = session.corrections[-1]
    assert (correction.tracks, correction.action, correction.frame) == (["A2"], "new_piece", 70)
    assert correction.prompts == [clicks]
    assert_recorded_now(correction)
    assert session.complete is False  # A2 has clicks and no results yet
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A2",), 70)]

    # the stand-in answers for A2 with the ground truth of the animal A
    assert run(dish_clip, session, folder, Renaming(dish_clip, {"A2": "A"}))[0] == "complete"
    after = ResultsStore.load(folder / "results.npz")
    assert after.track_ids == ["A", "A2", "B", "C"]
    assert after.arrays("A").frames.tolist() == GRID[:30]   # frames 0 to 58
    assert after.arrays("A2").frames.tolist() == GRID[35:]  # frames 70 to 118
    u, v, area = table_truth(dish_clip, "A", GRID[35:])
    assert np.abs(after.arrays("A2").u - u).max() <= 0.01 and np.abs(after.arrays("A2").v - v).max() <= 0.01
    assert np.array_equal(after.arrays("A2").area_px, area)
    file_after = read_npz(folder / "results.npz")
    assert_tracks_identical(file_after, before, "BC")
    assert_rows_identical(file_after, before, "A", rows=30)
    assert session.complete is True


def test_new_piece_takes_the_next_free_number(tracked, dish_clip):
    _, session, _ = tracked
    assert new_piece(session, "A", 70, clicks_at(dish_clip, "A", 70)).track_ids == ("A2",)
    assert new_piece(session, "A", 90, clicks_at(dish_clip, "A", 90)).track_ids == ("A3",)
    # a piece of a piece is the next piece of the same animal
    assert new_piece(session, "A2", 100, clicks_at(dish_clip, "A", 100)).track_ids == ("A4",)
    assert new_piece(session, "C", 70, clicks_at(dish_clip, "C", 70)).track_ids == ("C2",)
    assert all(TRACK_ID.fullmatch(a_track.id) for a_track in session.tracks)


def test_new_piece_at_an_off_grid_frame_moves_forward_to_the_grid_and_says_so(tracked, dish_clip):
    _, session, _ = tracked
    done = new_piece(session, "A", 71, clicks_at(dish_clip, "A", 71))
    assert (done.track_ids, done.frame, done.moved) == (("A2",), 72, True)
    assert "71" in done.note and "72" in done.note
    piece = session.tracks[-1]
    assert (piece.start_frame, piece.prompts[0].frame, piece.prompts[0].frame_hash) == (72, 72, None)
    assert session.corrections[-1].frame == 72


def test_a_piece_of_a_fine_track_is_fine_and_starts_without_a_stored_window(tracked, dish_clip):
    _, session, _ = tracked
    session.tracks[0].mode, session.tracks[0].fine_window_px = "fine", 120
    new_piece(session, "A", 70, clicks_at(dish_clip, "A", 70))
    assert (session.tracks[-1].mode, session.tracks[-1].fine_window_px) == ("fine", None)
    assert session.tracks[0].fine_window_px == 120


def test_new_piece_without_clicks_waits_for_them(tracked):
    # the GUI's order: "Continue as new track", then the clicks (`add_prompt`, with a preview)
    _, session, store = tracked
    done = new_piece(session, "B", 70)
    assert done.track_ids == ("B2",) and session.tracks[-1].prompts == [] and session.tracks[-1].start_frame == 70
    assert session.corrections[-1].prompts == []
    assert pending_runs(session, store) == [] and session.complete is True  # nothing to track before a click


def test_names_of_another_kind_do_not_use_letters_and_have_no_pieces():
    # from-tracker keeps the names of the Tracker export, whatever they are
    session = Session(clip=Clip(start=0, end=20, step=2), tracks=[Track(id="mass A"), Track(id="b")])
    assert next_track_id(session) == "A"
    with pytest.raises(ValueError, match="capital-letter name"):
        new_piece(session, "mass A", 4)
    assert [a_track.id for a_track in session.tracks] == ["mass A", "b"] and session.corrections == []


@pytest.mark.parametrize("parent, prompts, message", [
    ("X", Prompt(points_px=[[1.5, 1.5]], labels=[1]), "no track X"),
    ("A", Prompt(points_px=[[1.5, 1.5]], labels=[0]), "positive click"),
], ids=["unknown parent", "only a negative click"])
def test_new_piece_refuses_what_it_cannot_use_and_changes_nothing(tracked, parent, prompts, message):
    _, session, _ = tracked
    before = session.to_json()
    with pytest.raises(ValueError, match=message):
        new_piece(session, parent, 70, prompts)
    assert session.to_json() == before


# ---------------------------------------------------------------------------------------------
# Track ids


def test_next_track_id_runs_from_a_to_z_then_aa_ab():
    expected = list(ascii_uppercase) + ["".join(pair) for pair in product(ascii_uppercase, repeat=2)]
    session = Session()
    made = []
    for _ in range(60):
        assert next_track_id(session) == expected[len(made)]
        made.append(add_object(session).id)
    assert made == expected[:60] and made[25:28] == ["Z", "AA", "AB"]
    assert all(TRACK_ID.fullmatch(track_id) for track_id in made)


def test_next_track_id_skips_the_letters_of_pieces_and_of_tracks_the_session_remembers():
    session = Session(tracks=[Track(id="A"), Track(id="B2")])  # B itself was removed, its piece is still there
    assert next_track_id(session) == "C"
    session.runs.append(RunRecord(tracks=["C"]))               # a removed track that was tracked once
    session.corrections.append(Correction(tracks=["D", "E2"], action="retrack"))
    assert next_track_id(session) == "F"  # the records of C, D and E must not read as those of a new object


# ---------------------------------------------------------------------------------------------
# Clicks: `add_prompt` and `undo_prompt`


def test_add_prompt_stores_the_frame_hash_and_undo_prompt_removes_the_last_click(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [], dish_crop=False)
    store = ResultsStore()
    name = add_object(session).id
    found, tag = clicks_at(dish_clip, "A", 6).frame_hash, clicks_at(dish_clip, "A", 6).decoder
    u, v = center(dish_clip, "A", 6)

    done = add_prompt(session, store, name, 6, (u, v), 1, found, tag)
    assert (done.track_ids, done.frame, done.moved, done.note, done.results_file) == (("A",), 6, False, "", None)
    add_prompt(session, store, name, 6, (u + 9.0, v), 0, found, tag)  # a negative click on the same frame
    a_track = session.tracks[0]
    assert a_track.prompts == [Prompt(frame=6, frame_hash=found, decoder=tag, points_px=[[u, v], [u + 9.0, v]],
                                      labels=[1, 0])]
    assert found.startswith("sha256:") and a_track.start_frame == 6
    assert session.complete is False  # an object with clicks waits to be tracked
    assert Session.from_json(session.to_json()).tracks == session.tracks

    assert undo_prompt(session, store, name) is True
    assert a_track.prompts == [Prompt(frame=6, frame_hash=found, decoder=tag, points_px=[[u, v]], labels=[1])]
    assert undo_prompt(session, store, name) is True
    assert a_track.prompts == [] and session.complete is True
    assert undo_prompt(session, store, name) is False  # nothing left to undo


def test_add_prompt_on_an_off_grid_frame_moves_forward_to_the_grid_and_says_so(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [], start=10, step=4, dish_crop=False)  # frames 10, 14, ..., 118
    store = ResultsStore()
    name = add_object(session).id
    done = add_prompt(session, store, name, 15, (20.5, 30.5), 1, "sha256:of-frame-15", "a-decoder")
    assert (done.frame, done.moved) == (18, True)
    assert "15" in done.note and "18" in done.note
    (prompt,) = session.tracks[0].prompts
    assert (prompt.frame, prompt.points_px, prompt.labels) == (18, [[20.5, 30.5]], [1])
    assert prompt.frame_hash is None and prompt.decoder is None  # the hash was that of frame 15
    assert session.tracks[0].start_frame == 18
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A",), 18)]
