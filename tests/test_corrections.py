"""The three corrections of SPEC 6.6, on the results store and on what `derive_track` makes of it:
"Re-track from here" (`retrack_from`), "End track here" (`end_track`) and "Continue as new track"
(`new_piece`); the ids they and `next_track_id` give; and the two prompt edits the brief names
(`add_prompt`, `undo_prompt`). The other edit functions are in tests/test_tracking_edit.py.

The call sequence of a re-track, tested here end to end with `ExactFake`:
`retrack_from(session, store, ids, k, prompts, run_folder)` (changes the session, removes the
records from frame k on and saves results.npz), then the caller saves the session, then
`run_job(Job(session, run_folder, video, make_segmenter), callbacks)` tracks what is pending.

Expected values: the ground-truth table of `dish_clip` (120 frames; the sessions take every 2nd
frame, so the grid is 0, 2, ..., 118), frame counts worked out from that grid, and "identical":
what was in results.npz before the correction. Frames are video frame numbers; positions are px in
Tracker's convention (SPEC 3.1), derived x and y in mm (y up).
"""

import re
import shutil
from datetime import datetime
from itertools import product
from string import ascii_uppercase

import numpy as np
import pytest
from helpers import ODD_FOLDER
from results_helpers import made_up, read_npz
from tracking_helpers import (
    Recorder,
    Renaming,
    Watched,
    abc_session,
    center,
    clicks_at,
    make_session,
    run,
    table_truth,
    track,
)

from outline_tracker.derive import derive_track
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_KEYS, npz_key
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Clip, Correction, Prompt, RunRecord, Session, Track, VideoRef
from outline_tracker.tracking_edit import (
    add_object,
    add_prompt,
    end_track,
    new_piece,
    next_track_id,
    pending_runs,
    remove_object,
    retrack_from,
    undo_prompt,
)
from outline_tracker.tracking_plan import plan_runs

GRID = list(range(0, 120, 2))  # `dish_clip` has 120 frames; the sessions here take every 2nd
K = 40                         # the frame of the re-track: row 20 of every track
TRACK_ID = re.compile(r"^[A-Z]+[0-9]*$")  # SPEC 3.4


def frames_of(folder, track_id):
    return ResultsStore.load(folder / "results.npz").arrays(track_id).frames.tolist()


def derived(session, store):
    """Every track of the store in world units, by id."""
    tracks = {a_track.id: a_track for a_track in session.tracks}
    return {name: derive_track(store.arrays(name), tracks[name], session.world_frame(), session.time.fps_true,
                               session.circle, session.processing) for name in store.track_ids}


def assert_rows_identical(after, before, track_id, rows):
    """The first `rows` rows of a track are the same in two readings of results.npz (`read_npz`):
    every array, and the bytes of the mask crops of those rows."""
    for key in RESULTS_KEYS:
        new, old = after[npz_key(track_id, key.name)], before[npz_key(track_id, key.name)]
        assert new.dtype == old.dtype, key.name
        if key.name == "mask_bits":
            shapes = before[npz_key(track_id, "mask_shape")][:rows].astype(np.int64)
            count = int(((shapes[:, 0] * shapes[:, 1] + 7) // 8).sum())
            new, old = new[:count], old[:count]
        else:
            new, old = new[:rows], old[:rows]
        np.testing.assert_array_equal(new, old, err_msg=f"{track_id}: {key.name}")


def assert_tracks_identical(after, before, track_ids):
    """Every array of the named tracks is the same in two readings of results.npz."""
    for track_id in track_ids:
        for key in RESULTS_KEYS:
            name = npz_key(track_id, key.name)
            assert after[name].dtype == before[name].dtype and after[name].shape == before[name].shape, name
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def assert_recorded_now(correction):
    """A correction's time is ISO 8601 with the offset from UTC, and it is now (within a minute)."""
    when = datetime.fromisoformat(correction.time)
    assert when.utcoffset() is not None
    assert abs((datetime.now().astimezone() - when).total_seconds()) < 60


@pytest.fixture(scope="module")
def tracked_folder(dish_clip, tmp_path_factory):
    """A run folder in which A, B and C of the dish clip are tracked on every grid frame."""
    folder = tmp_path_factory.mktemp("tracked")
    assert run(dish_clip, abc_session(dish_clip, folder), folder, ExactFake(dish_clip))[0] == "complete"
    return folder


@pytest.fixture
def tracked(tracked_folder, tmp_path):
    """This test's own copy of that run, in a folder named with a space and non-ASCII characters:
    (run folder, session, store)."""
    folder = tmp_path / ODD_FOLDER
    shutil.copytree(tracked_folder, folder)
    session = Session.load(folder / "session.json")
    assert session.complete is True
    return folder, session, ResultsStore.load(folder / "results.npz")


# ---------------------------------------------------------------------------------------------
# Re-track from here


def test_retrack_from_changes_only_frames_from_k_of_the_selected_track(dish_clip, tmp_path):
    # The first job: from frame 40 on the model follows the animal C under the name A (a swap).
    session = abc_session(dish_clip, tmp_path)
    assert run(dish_clip, session, tmp_path, Renaming(dish_clip, {"A": "C"}, from_frame=K))[0] == "complete"
    store = ResultsStore.load(tmp_path / "results.npz")
    wrong_u, wrong_v, _ = table_truth(dish_clip, "C", GRID[20:])
    assert np.abs(store.arrays("A").u[20:] - wrong_u).max() <= 0.01  # the test's premise: A is wrong from row 20
    file_before, derived_before = read_npz(tmp_path / "results.npz"), derived(session, store)

    done = retrack_from(session, store, ["A"], K, {"A": clicks_at(dish_clip, "A", K)}, tmp_path)
    assert (done.frame, done.moved, done.note, done.track_ids) == (K, False, "", ("A",))
    assert done.results_file == tmp_path / "results.npz"
    assert frames_of(tmp_path, "A") == GRID[:20]  # saved: the job reads results.npz from the disk
    assert session.complete is False
    assert [(plan.track_ids, plan.start_frame, plan.mode) for plan in pending_runs(session, store)] == [
        (("A",), K, "coarse")]

    session.save(tmp_path / "session.json")  # the caller's part
    segmenter = Watched(ExactFake(dish_clip))
    assert run(dish_clip, session, tmp_path, segmenter)[0] == "complete"
    ((_, (prompt,)),) = segmenter.starts  # one run, for A alone, started with the new click on frame 40
    assert (prompt.obj_id, prompt.points_px, prompt.labels) == ("A", [center(dish_clip, "A", K)], [1])
    assert segmenter.calls == ["start", *["step"] * 39, "close"]  # frames 40 to 118
    assert session.complete is True

    after = ResultsStore.load(tmp_path / "results.npz")
    file_after, derived_after = read_npz(tmp_path / "results.npz"), derived(session, after)
    # A from frame 40 on is A again: the centroids of the ground-truth masks
    u, v, area = table_truth(dish_clip, "A", GRID)
    arrays = after.arrays("A")
    assert arrays.frames.tolist() == GRID
    assert np.abs(arrays.u - u).max() <= 0.01 and np.abs(arrays.v - v).max() <= 0.01
    assert np.array_equal(arrays.area_px, area)
    # ... and nothing else changed: every array of B and C, and the rows of A before frame 40
    assert set(file_after) == set(file_before)
    assert_tracks_identical(file_after, file_before, "BC")
    assert_rows_identical(file_after, file_before, "A", rows=20)
    for name, rows in (("A", 20), ("B", 60), ("C", 60)):
        for column in ("u_px", "v_px", "x_mm", "y_mm"):
            new, old = getattr(derived_after[name], column), getattr(derived_before[name], column)
            assert len(new) == len(old) == 60
            np.testing.assert_array_equal(new[:rows], old[:rows], err_msg=f"{name}: {column}")


def test_retrack_from_records_the_correction_and_the_new_clicks(tracked, dish_clip):
    folder, session, store = tracked
    clicks = {"A": clicks_at(dish_clip, "A", K), "C": clicks_at(dish_clip, "C", K)}
    clicks["C"].points_px.append([3.5, 4.5])  # a negative click on something else
    clicks["C"].labels.append(0)
    retrack_from(session, store, ["A", "C"], K, clicks, folder)

    a, b, c = session.tracks
    assert a.prompts[-1] == clicks["A"] and c.prompts[-1] == clicks["C"]
    assert a.prompts[-1] is not clicks["A"]  # the session holds its own copy
    assert len(a.prompts) == 2 and len(b.prompts) == 1 and (a.start_frame, c.start_frame) == (0, 0)
    assert a.prompts[-1].frame_hash.startswith("sha256:") and a.prompts[-1].decoder
    (correction,) = session.corrections
    assert (correction.tracks, correction.action, correction.frame) == (["A", "C"], "retrack", K)
    assert correction.prompts == [clicks["A"], clicks["C"]]
    assert_recorded_now(correction)
    assert [frames_of(folder, name) for name in "ABC"] == [GRID[:20], GRID, GRID[:20]]
    assert Session.from_json(session.to_json()).corrections == session.corrections  # it fits session.json


def test_retrack_from_an_off_grid_frame_moves_forward_to_the_grid_and_says_so(tracked, dish_clip):
    folder, session, store = tracked
    typed = clicks_at(dish_clip, "B", 41)  # clicked on frame 41, which the clip (every 2nd frame) does not have
    done = retrack_from(session, store, ["B"], 41, {"B": typed}, folder)
    assert (done.frame, done.moved) == (42, True)
    assert "41" in done.note and "42" in done.note
    assert frames_of(folder, "B") == GRID[:21]  # frames 0 to 40 are kept: nothing before the typed frame is touched
    new = session.tracks[1].prompts[-1]
    assert (new.frame, new.points_px, new.labels) == (42, typed.points_px, [1])
    assert new.frame_hash is None and new.decoder is None  # the hash of frame 41 does not identify frame 42
    assert session.corrections[-1].frame == 42
    assert run(dish_clip, session, folder, ExactFake(dish_clip))[0] == "complete"
    assert frames_of(folder, "B") == GRID


def test_retrack_from_uses_the_clicks_already_placed_on_that_frame(tracked, dish_clip):
    # the GUI's order: go to frame 40, click (`add_prompt`, with a preview), then "Re-track from here"
    folder, session, store = tracked
    found, tag = clicks_at(dish_clip, "A", K).frame_hash, clicks_at(dish_clip, "A", K).decoder
    add_prompt(session, store, "A", K, center(dish_clip, "A", K), 1, found, tag)
    assert pending_runs(session, store) == [] and session.complete is True  # frame 40 is still tracked
    retrack_from(session, store, ["A"], K, {}, folder)
    assert session.corrections[-1].prompts == [session.tracks[0].prompts[-1]]
    assert [plan.start_frame for plan in pending_runs(session, store)] == [K]


def test_retrack_from_drops_the_clicks_of_later_frames(tracked, dish_clip):
    # A was corrected at frame 60 before; a re-track from frame 40 replaces those results, clicks included
    folder, session, store = tracked
    retrack_from(session, store, ["A"], 60, {"A": clicks_at(dish_clip, "A", 60)}, folder)
    retrack_from(session, store, ["A"], K, {"A": clicks_at(dish_clip, "A", K)}, folder)
    assert [prompt.frame for prompt in session.tracks[0].prompts] == [0, K]
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A",), K)]
    assert frames_of(folder, "A") == GRID[:20]


def test_retrack_from_a_frame_after_clicks_that_still_wait_is_refused(tracked, dish_clip):
    # A waits from frame 40: a run from frame 60 would leave frames 40 to 58 without results
    folder, session, store = tracked
    retrack_from(session, store, ["A"], K, {"A": clicks_at(dish_clip, "A", K)}, folder)
    before, results = session.to_json(), (folder / "results.npz").read_bytes()
    with pytest.raises(ValueError, match="clicks on frame 40 that are not tracked yet"):
        retrack_from(session, store, ["A"], 60, {"A": clicks_at(dish_clip, "A", 60)}, folder)
    assert session.to_json() == before and (folder / "results.npz").read_bytes() == results


def test_a_retrack_of_a_fine_track_keeps_its_window(closeup_clip, tmp_path):
    # B of the close-up clip: the plain 47 x 20 px body
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "B", mode="fine")], dish_crop=False)
    assert run(closeup_clip, session, tmp_path, ExactFake(closeup_clip))[0] == "complete"
    window = session.tracks[0].fine_window_px
    assert 139 <= window <= 144  # chosen on frame 0: three times the body's length, 3 x 47 = 141 px (SPEC 6.3)
    store = ResultsStore.load(tmp_path / "results.npz")
    retrack_from(session, store, ["B"], K, {"B": clicks_at(closeup_clip, "B", K)}, tmp_path)
    assert session.tracks[0].fine_window_px == window
    segmenter = Watched(ExactFake(closeup_clip))
    assert run(closeup_clip, session, tmp_path, segmenter)[0] == "complete"
    assert set(segmenter.shapes) == {(window, window, 3)}  # the stored window on every frame, not a new choice
    assert segmenter.calls[:2] == ["preview", "start"] and len(segmenter.shapes) == 40  # frames 40 to 118

    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("B")
    u, v, _ = table_truth(closeup_clip, "B", GRID)
    assert arrays.frames.tolist() == GRID and set(arrays.mode.tolist()) == {"fine"}
    assert np.abs(arrays.u - u).max() <= 0.01 and np.abs(arrays.v - v).max() <= 0.01
    assert session.tracks[0].fine_window_px == window and session.complete is True


@pytest.mark.parametrize("track_ids, prompts, message", [
    (["A", "X"], {}, "no track X"),
    ([], {}, "at least one track"),
    (["A"], {"B": Prompt(points_px=[[1.5, 1.5]], labels=[1])}, "given for B"),  # a track that is not re-tracked
    (["A"], {}, "positive click"),                                            # nothing to start the run with
    (["A"], {"A": Prompt(points_px=[[1.5, 1.5]], labels=[0])}, "positive click"),
    (["A"], {"A": Prompt(points_px=[[1.5, 1.5]], labels=[1, 0])}, "one label"),
], ids=["unknown track", "no track", "clicks for another track", "no clicks", "only a negative click",
        "labels do not fit"])
def test_retrack_from_refuses_what_it_cannot_use_and_changes_nothing(tracked, track_ids, prompts, message):
    folder, session, store = tracked
    before, results = session.to_json(), (folder / "results.npz").read_bytes()
    with pytest.raises(ValueError, match=message):
        retrack_from(session, store, track_ids, K, prompts, folder)
    assert session.to_json() == before and (folder / "results.npz").read_bytes() == results
    assert store.arrays("A").frames.tolist() == GRID


# ---------------------------------------------------------------------------------------------
# End track here


def test_end_track_leaves_the_tracks_frames_ending_at_k(tracked, dish_clip):
    folder, session, store = tracked
    before = read_npz(folder / "results.npz")
    done = end_track(session, store, "B", 50, folder)
    assert (done.frame, done.moved, done.note, done.track_ids) == (50, False, "", ("B",))
    assert done.results_file == folder / "results.npz"

    after = read_npz(folder / "results.npz")
    assert after[npz_key("B", "frames")].tolist() == GRID[:26]  # frames 0 to 50
    assert_rows_identical(after, before, "B", rows=26)
    assert_tracks_identical(after, before, "AC")
    assert [a_track.ended_at for a_track in session.tracks] == [None, 50, None]
    (correction,) = session.corrections
    assert (correction.tracks, correction.action, correction.frame, correction.prompts) == (["B"], "end", 50, [])
    assert_recorded_now(correction)
    assert session.complete is True  # B has every frame up to its end

    # a later job finds nothing to do, and B still ends at frame 50
    assert pending_runs(session, store) == []
    status, seen = run(dish_clip, session, folder, ExactFake(dish_clip))
    assert status == "complete" and any("Nothing to track" in line for line in seen.log)
    assert frames_of(folder, "B") == GRID[:26]


def test_end_track_between_two_grid_frames_keeps_the_frames_up_to_there(tracked):
    folder, session, store = tracked
    done = end_track(session, store, "B", 51, folder)
    assert (done.frame, done.moved) == (51, False)  # an end is not moved: the rows stop at the last frame up to it
    assert frames_of(folder, "B") == GRID[:26] and session.tracks[1].ended_at == 51
    assert session.complete is True


def test_end_track_drops_clicks_that_wait_beyond_the_end(tracked, dish_clip):
    folder, session, store = tracked
    retrack_from(session, store, ["B"], 60, {"B": clicks_at(dish_clip, "B", 60)}, folder)  # pending from frame 60
    end_track(session, store, "B", 50, folder)
    assert [prompt.frame for prompt in session.tracks[1].prompts] == [0]
    assert pending_runs(session, store) == [] and session.complete is True
    assert frames_of(folder, "B") == GRID[:26]


def test_end_track_makes_cancelled_results_whole_when_the_track_ends_where_it_stopped(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A")], dish_crop=False)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip), Recorder(cancel_after_frame=10))[0] == "cancelled"
    store = ResultsStore.load(tmp_path / "results.npz")
    assert session.complete is False
    done = end_track(session, store, "A", 10, tmp_path)
    assert done.results_file is None  # A has no record after frame 10: results.npz is not written again
    assert session.complete is True


def test_an_edit_keeps_results_complete_where_the_video_ends_before_the_clip(dish_clip, tmp_path):
    # the clip asks for frames 0 to 200 every 2; the video has frames 0 to 119, so tracking ends on frame 118
    tracks = [track(dish_clip, name) for name in "ABC"]
    session = make_session(dish_clip, tmp_path, tracks, end=200, dish_crop=False)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete" and session.complete is True
    store = ResultsStore.load(tmp_path / "results.npz")
    end_track(session, store, "B", 50, tmp_path)
    assert session.complete is True  # A and C have every frame the video has, B every frame up to its end
    remove_object(session, store, "C", tmp_path)
    assert session.complete is True and frames_of(tmp_path, "A") == GRID


@pytest.mark.parametrize("track_id, frame, message", [("X", 50, "no track X"), ("C", -2, "before")],
                         ids=["unknown track", "before its start"])
def test_end_track_refuses_what_it_cannot_do_and_changes_nothing(tracked, track_id, frame, message):
    folder, session, store = tracked
    before, results = session.to_json(), (folder / "results.npz").read_bytes()
    with pytest.raises(ValueError, match=message):
        end_track(session, store, track_id, frame, folder)
    assert session.to_json() == before and (folder / "results.npz").read_bytes() == results


def test_a_retrack_of_an_ended_track_stops_at_its_end(tracked, dish_clip):
    folder, session, store = tracked
    end_track(session, store, "B", 50, folder)
    retrack_from(session, store, ["B"], 30, {"B": clicks_at(dish_clip, "B", 30)}, folder)
    assert [(plan.track_ids, list(plan.frames)) for plan in pending_runs(session, store)] == [
        (("B",), list(range(30, 51, 2)))]
    assert run(dish_clip, session, folder, ExactFake(dish_clip))[0] == "complete"

    arrays = ResultsStore.load(folder / "results.npz").arrays("B")
    assert arrays.frames.tolist() == GRID[:26]  # never beyond frame 50
    u, v, _ = table_truth(dish_clip, "B", GRID[:26])
    assert np.abs(arrays.u - u).max() <= 0.01 and np.abs(arrays.v - v).max() <= 0.01
    assert session.tracks[1].ended_at == 50 and session.complete is True
    with pytest.raises(ValueError, match="ended at frame 50"):
        retrack_from(session, store, ["B"], 52, {"B": clicks_at(dish_clip, "B", 52)}, folder)


def test_a_run_that_stops_at_a_tracks_end_does_not_make_cancelled_results_complete(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A"), track(dish_clip, "B")], dish_crop=False)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip), Recorder(cancel_after_frame=10))[0] == "cancelled"
    store = ResultsStore.load(tmp_path / "results.npz")
    end_track(session, store, "B", 6, tmp_path)
    assert session.complete is False  # A stops at frame 10 of a clip that goes on to frame 118
    retrack_from(session, store, ["B"], 4, {"B": clicks_at(dish_clip, "B", 4)}, tmp_path)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"  # B's frames 4 and 6
    assert [frames_of(tmp_path, name) for name in "AB"] == [[0, 2, 4, 6, 8, 10], [0, 2, 4, 6]]
    assert session.complete is False  # the job ended on frame 6, which is not where the video ends


# ---------------------------------------------------------------------------------------------
# Planning the runs of tracks that were ended (hand-made results; a clip of frames 0 to 20 every 2)


def planned(*tracks, **frames_by_track):
    """`plan_runs` for the given tracks and a store with a made-up record on each named frame."""
    session = Session(video=VideoRef(width=64, height=48), clip=Clip(start=0, end=20, step=2), tracks=list(tracks))
    store = ResultsStore()
    for track_id, frames in frames_by_track.items():
        for frame in frames:
            store.put(track_id, made_up(frame, seed=frame))
    return [(plan.track_ids, list(plan.frames)) for plan in plan_runs(session, store)]


def with_clicks(track_id, *frames, **more):
    prompts = [Prompt(frame=frame, points_px=[[5.5, 5.5]], labels=[1]) for frame in frames]
    return Track(id=track_id, start_frame=frames[0], prompts=prompts, **more)


def test_a_run_for_an_ended_track_covers_the_frames_up_to_its_end_only():
    # A, ended at frame 10, and B wait from frame 6 on: they cannot share a run, since B goes on to frame 20
    plans = planned(with_clicks("A", 0, 6, ended_at=10), with_clicks("B", 0, 6), with_clicks("C", 0, 6, ended_at=10),
                    A=[0, 2, 4], B=[0, 2, 4], C=[0, 2, 4])
    assert plans == [(("A", "C"), [6, 8, 10]), (("B",), [6, 8, 10, 12, 14, 16, 18, 20])]
    # an end between two grid frames: the last frame up to it
    assert planned(with_clicks("A", 0, 6, ended_at=11), A=[0, 2, 4]) == [(("A",), [6, 8, 10])]
    assert planned(with_clicks("A", 10, ended_at=10)) == [(("A",), [10])]  # ended on its start frame


def test_clicks_after_the_end_of_an_ended_track_are_refused_not_tracked():
    with pytest.raises(ValueError, match="A was ended at frame 10 and has clicks on frame 12"):
        planned(with_clicks("A", 0, 12, ended_at=10), A=range(0, 11, 2))


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
