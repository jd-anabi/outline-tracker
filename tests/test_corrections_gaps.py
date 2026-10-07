"""What an edit must not leave behind (outline_tracker/tracking_edit.py; the corrections themselves
are in tests/test_corrections.py):

- a gap inside a track. SPEC 8.2 gives a track every grid frame from its first to its last, so
  `retrack_from` and `add_prompt` take no frame later than the first grid frame after the track's
  last record;
- results that a job wrote and an edit lost. With `store=None` an edit starts from results.npz as
  it is now, and the `Edit` carries the store that was saved;
- a correction in the session whose results are not in results.npz. When that file stays locked,
  `retrack_from`, `end_track` and `remove_object` raise RuntimeError and change nothing;
- results that a job had to put in results.new.npz, replaced or removed. While that file is in the
  run folder, the three raise RuntimeError and change nothing, that file least of all;
- clicks nobody knows of: the clicks of two re-tracks from the same frame add up.

Expected values: the ground truth of `dish_clip` (120 frames; the sessions take every 2nd, so the
grid is 0, 2, ..., 118) and frame counts worked out from that grid: a track with frames 0 to 6 can
go on from frame 8 at the latest. "Unchanged" is what session.json would hold (`to_json`) and the
bytes of results.npz before the call. Frames are video frame numbers; positions are px in
Tracker's convention (SPEC 3.1).
"""

import errno
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from helpers import ODD_FOLDER
from tracking_helpers import Recorder, Watched, abc_session, center, clicks_at, make_session, run, table_truth, track

from outline_tracker import fileio
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_KEYS, RESULTS_NPZ, SESSION_JSON
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Prompt, Session
from outline_tracker.tracking_edit import (
    add_object,
    add_prompt,
    end_track,
    pending_runs,
    remove_object,
    retrack_from,
    undo_prompt,
)

GRID = list(range(0, 120, 2))  # `dish_clip` has 120 frames; the sessions here take every 2nd
K = 40                         # the frame of a re-track: row 20 of every track
NEW_NPZ = "results.new.npz"    # where a save goes while results.npz stays locked (`fileio.new_name`)


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
    return folder, Session.load(folder / SESSION_JSON), ResultsStore.load(folder / RESULTS_NPZ)


def frames_on_disk(folder, name=RESULTS_NPZ):
    """The frames of every track of results.npz (or of the file `name`) of a run folder, by track id."""
    store = ResultsStore.load(folder / name)
    return {track_id: store.arrays(track_id).frames.tolist() for track_id in store.track_ids}


def files_of(folder):
    return sorted(path.name for path in folder.iterdir())


def assert_follows_the_animal(clip, folder, track_id, frames, name=RESULTS_NPZ):
    """results.npz (or the file `name`) has the track on exactly these frames, at the centroids of
    the clip's ground truth (within 0.01 px) and with its areas."""
    arrays = ResultsStore.load(folder / name).arrays(track_id)
    assert arrays.frames.tolist() == list(frames)
    u, v, area = table_truth(clip, track_id, frames)
    assert np.abs(arrays.u - u).max() <= 0.01 and np.abs(arrays.v - v).max() <= 0.01
    assert np.array_equal(arrays.area_px, area)


def assert_holds_the_file(store, path):
    """A store in memory holds what results.npz does: the same tracks, and every array of each."""
    on_disk = ResultsStore.load(path)
    assert store.track_ids == on_disk.track_ids
    for track_id in on_disk.track_ids:
        ours, theirs = store.arrays(track_id), on_disk.arrays(track_id)
        for key in RESULTS_KEYS:
            np.testing.assert_array_equal(getattr(ours, key.name), getattr(theirs, key.name),
                                          err_msg=f"{track_id}: {key.name}")


# ---------------------------------------------------------------------------------------------
# No gap inside a track


def retrack_a(session, store, clip, folder, frame):
    return retrack_from(session, store, ["A"], frame, {"A": clicks_at(clip, "A", frame)}, folder)


def click_a(session, store, clip, folder, frame):
    clicks = clicks_at(clip, "A", frame)
    return add_prompt(session, store, "A", frame, clicks.points_px[0], 1, clicks.frame_hash, clicks.decoder)


@pytest.fixture(params=["a cancelled job", "a re-track that was undone"])
def stops_at_6(request, dish_clip, tmp_path):
    """A run folder in which track A has results on frames 0, 2, 4 and 6 and nothing waits to be
    tracked: (run folder, session, store). The two ways there: a job that was cancelled after
    frame 6; "Re-track from here" on frame 8 of a whole track, and then undo of its click."""
    if request.param == "a cancelled job":
        folder = tmp_path
        session = make_session(dish_clip, folder, [track(dish_clip, "A")], dish_crop=False)
        assert run(dish_clip, session, folder, ExactFake(dish_clip), Recorder(cancel_after_frame=6))[0] == "cancelled"
        store = ResultsStore.load(folder / RESULTS_NPZ)
    else:
        folder, session, store = request.getfixturevalue("tracked")
        retrack_a(session, store, dish_clip, folder, 8)
        assert undo_prompt(session, store, "A") is True
    # the track is cut before frame 8, and nothing is pending
    assert store.arrays("A").frames.tolist() == [0, 2, 4, 6] and frames_on_disk(folder)["A"] == [0, 2, 4, 6]
    assert pending_runs(session, store) == [] and session.complete is False
    return folder, session, store


def assert_names_the_gap(message):
    """The refusal names the track, its last frame with results (6) and the latest frame that is
    allowed (8, the first grid frame after it)."""
    for part in ("Track A", "frame 6", "frame 8"):
        assert re.search(rf"\b{part}\b", message), message


@pytest.mark.parametrize("edit", [retrack_a, click_a], ids=["retrack_from", "add_prompt"])
@pytest.mark.parametrize("frame", [14, 10, 9, 118], ids=["frame 14", "frame 10", "frame 9, moved to 10", "the last"])
def test_an_edit_later_than_the_first_frame_after_the_last_record_is_refused(stops_at_6, dish_clip, edit, frame):
    # a run from frame 14 would leave frames 8, 10 and 12 without records
    folder, session, store = stops_at_6
    before, results = session.to_json(), (folder / RESULTS_NPZ).read_bytes()
    with pytest.raises(ValueError) as refused:
        edit(session, store, dish_clip, folder, frame)
    assert_names_the_gap(str(refused.value))
    assert session.to_json() == before and (folder / RESULTS_NPZ).read_bytes() == results
    assert store.arrays("A").frames.tolist() == [0, 2, 4, 6]


@pytest.mark.parametrize("frame, starts", [(8, 8), (7, 8), (4, 4)],
                         ids=["the first frame after the last record", "frame 7, moved to it", "an earlier frame"])
def test_after_a_refusal_a_retrack_from_an_allowed_frame_leaves_no_gap(stops_at_6, dish_clip, frame, starts):
    folder, session, store = stops_at_6
    with pytest.raises(ValueError) as refused:
        retrack_a(session, store, dish_clip, folder, 14)
    assert_names_the_gap(str(refused.value))

    done = retrack_a(session, store, dish_clip, folder, frame)  # the same call with an allowed frame
    assert (done.frame, done.moved) == (starts, frame != starts)
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A",), starts)]
    assert run(dish_clip, session, folder, ExactFake(dish_clip))[0] == "complete"
    assert_follows_the_animal(dish_clip, folder, "A", GRID)  # every grid frame from its first to its last (SPEC 8.2)
    assert session.complete is True


def test_after_a_refusal_a_click_on_the_first_frame_after_the_last_record_leaves_no_gap(stops_at_6, dish_clip):
    folder, session, store = stops_at_6
    with pytest.raises(ValueError) as refused:
        click_a(session, store, dish_clip, folder, 14)
    assert_names_the_gap(str(refused.value))

    assert click_a(session, store, dish_clip, folder, 8).frame == 8
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A",), 8)]
    assert run(dish_clip, session, folder, ExactFake(dish_clip))[0] == "complete"
    assert_follows_the_animal(dish_clip, folder, "A", GRID)
    assert session.complete is True


def test_a_track_without_results_starts_on_any_frame(tracked):
    # no record, no gap: the first click sets the start
    folder, session, store = tracked
    first, second = add_object(session).id, add_object(session).id
    add_prompt(session, store, first, 100, (30.5, 40.5), 1, None, None)
    retrack_from(session, store, [second], 60, {second: Prompt(points_px=[[30.5, 40.5]], labels=[1])}, folder)
    assert [(a_track.id, a_track.start_frame) for a_track in session.tracks[3:]] == [("D", 100), ("E", 60)]
    waiting = [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)]
    assert waiting == [(("E",), 60), (("D",), 100)]


# ---------------------------------------------------------------------------------------------
# The results an edit starts from, and the store it hands back


def retrack_c(session, store, clip, folder):
    return retrack_from(session, store, ["C"], K, {"C": clicks_at(clip, "C", K)}, folder)


def end_b(session, store, clip, folder):
    return end_track(session, store, "B", 50, folder)


def remove_b(session, store, clip, folder):
    return remove_object(session, store, "B", folder)


# each edit, and the frames it leaves in results.npz of a run in which A, B and C were whole
EDITS = {"retrack_from": (retrack_c, {"A": GRID, "B": GRID, "C": GRID[:20]}),
         "end_track": (end_b, {"A": GRID, "B": GRID[:26], "C": GRID}),
         "remove_object": (remove_b, {"A": GRID, "C": GRID})}


@pytest.mark.parametrize("name", EDITS)
def test_with_store_none_an_edit_keeps_the_records_a_job_wrote_since(tracked, dish_clip, name):
    # The stale store: loaded before a job, used after it. The job reads and writes results.npz itself.
    folder, session, stale = tracked
    retrack_a(session, stale, dish_clip, folder, K)
    assert run(dish_clip, session, folder, ExactFake(dish_clip))[0] == "complete"
    assert stale.arrays("A").frames.tolist() == GRID[:20]  # the premise: this store lacks what the job tracked

    edit, left = EDITS[name]
    done = edit(session, None, dish_clip, folder)  # None: results.npz of the run folder, as it is now
    assert frames_on_disk(folder) == left
    assert_follows_the_animal(dish_clip, folder, "A", GRID)  # frames 40 to 118 of A are the job's records
    if name != "remove_object":  # which returns the file it saved, not an `Edit`
        assert_holds_the_file(done.store, folder / RESULTS_NPZ)


@pytest.mark.parametrize("given", [True, False], ids=["a store given", "store=None"])
@pytest.mark.parametrize("name", ["retrack_from", "end_track"])
def test_the_edit_carries_the_store_that_was_saved(tracked, dish_clip, name, given):
    folder, session, store = tracked
    edit, left = EDITS[name]
    done = edit(session, store if given else None, dish_clip, folder)
    assert done.results_file == folder / RESULTS_NPZ and frames_on_disk(folder) == left
    assert_holds_the_file(done.store, folder / RESULTS_NPZ)
    assert (done.store is store) is given  # a store that was given is changed, and is the one handed back
    # the caller goes on with it; an edit that has nothing to save hands the store back all the same
    again = end_track(session, done.store, "A", 118, folder)
    assert again.results_file is None and again.store is done.store


def test_with_store_none_and_no_results_file_an_edit_starts_from_empty_results(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A"), track(dish_clip, "B")], dish_crop=False)
    done = end_track(session, None, "A", 50, tmp_path)
    assert done.results_file is None and done.store.track_ids == []
    assert remove_object(session, None, "B", tmp_path) is None
    assert [a_track.id for a_track in session.tracks] == ["A"] and session.tracks[0].ended_at == 50
    assert files_of(tmp_path) == []  # nothing to save, nothing written


# ---------------------------------------------------------------------------------------------
# A save that did not land


def lock(monkeypatch, *targets):
    """Make each of `targets` a file that another program holds open (Windows): every `os.replace`
    onto it raises PermissionError. The waits between the tries of `fileio` are skipped."""
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst) in targets:
            raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda wait_s: None)


@pytest.mark.parametrize("locked", [[RESULTS_NPZ], [RESULTS_NPZ, "results.new.npz"]],
                         ids=["results.npz locked", "results.new.npz locked too"])
@pytest.mark.parametrize("given", [True, False], ids=["a store given", "store=None"])
@pytest.mark.parametrize("name", EDITS)
def test_an_edit_whose_save_does_not_land_raises_and_changes_nothing(tracked, dish_clip, monkeypatch, name, given,
                                                                     locked):
    folder, session, store = tracked
    edit, left = EDITS[name]
    before, results, files = session.to_json(), (folder / RESULTS_NPZ).read_bytes(), files_of(folder)

    lock(monkeypatch, *(folder / file_name for file_name in locked))
    with pytest.raises(RuntimeError) as refused:
        edit(session, store if given else None, dish_clip, folder)
    message = str(refused.value)  # which file, close the program that holds it, try again
    assert RESULTS_NPZ in message and re.search(r"\b[Cc]lose\b", message) and "try again" in message
    assert files_of(folder) == files  # no results.new.npz, no temporary file
    assert session.to_json() == before and (folder / RESULTS_NPZ).read_bytes() == results
    assert {a_track: store.arrays(a_track).frames.tolist() for a_track in "ABC"} == dict.fromkeys("ABC", GRID)

    monkeypatch.undo()  # the other program has closed the file: the same call does its work
    edit(session, store if given else None, dish_clip, folder)
    assert frames_on_disk(folder) == left and files_of(folder) == files
    assert session.to_json() != before


@pytest.fixture
def left_aside(tracked, dish_clip, monkeypatch):
    """A run folder in which a job could not write results.npz: (run folder, session, store).
    Track A was re-tracked from frame 40 while another program held results.npz open, so the job
    left its results in results.new.npz (A, B and C on every grid frame), and results.npz still
    has A on frames 0 to 38 only, as `store` does. That program still holds results.npz."""
    folder, session, store = tracked
    retrack_a(session, store, dish_clip, folder, K)
    lock(monkeypatch, folder / RESULTS_NPZ)
    status, seen = run(dish_clip, session, folder, ExactFake(dish_clip))
    assert status == "complete" and any(NEW_NPZ in line for line in seen.log)
    assert frames_on_disk(folder) == {"A": GRID[:20], "B": GRID, "C": GRID}
    assert frames_on_disk(folder, NEW_NPZ) == dict.fromkeys("ABC", GRID)
    return folder, session, store


def assert_says_how_to_go_on(message):
    """The refusal names both files and what the user does: close the program that holds
    results.npz, rename results.new.npz, try again."""
    assert re.search(rf"\b{re.escape(NEW_NPZ)}\b", message) and re.search(rf"\b{re.escape(RESULTS_NPZ)}\b", message)
    assert re.search(r"\b[Cc]lose\b", message) and re.search(r"\brename\b", message) and "try again" in message


@pytest.mark.parametrize("held", [True, False], ids=["results.npz still locked", "results.npz free again"])
@pytest.mark.parametrize("given", [True, False], ids=["a store given", "store=None"])
@pytest.mark.parametrize("name", EDITS)
def test_an_edit_keeps_the_results_a_job_left_in_results_new_npz(left_aside, dish_clip, monkeypatch, name, given, held):
    # Each of these edits has records to remove. Its own save would go to results.new.npz while results.npz is
    # locked, and to results.npz, which lacks what the job tracked, when it is free: neither may happen.
    folder, session, store = left_aside
    if not held:
        monkeypatch.undo()
    edit, left = EDITS[name]
    before, files = session.to_json(), files_of(folder)
    results, aside = (folder / RESULTS_NPZ).read_bytes(), (folder / NEW_NPZ).read_bytes()

    with pytest.raises(RuntimeError) as refused:
        edit(session, store if given else None, dish_clip, folder)
    assert files_of(folder) == files and (folder / NEW_NPZ).read_bytes() == aside  # the job's only copy
    assert_follows_the_animal(dish_clip, folder, "A", GRID, NEW_NPZ)
    assert session.to_json() == before and (folder / RESULTS_NPZ).read_bytes() == results
    assert [store.arrays(a_track).frames.tolist() for a_track in "ABC"] == [GRID[:20], GRID, GRID]
    assert_says_how_to_go_on(str(refused.value))

    monkeypatch.undo()  # the user does what the message says: closes that program, renames the file
    os.replace(folder / NEW_NPZ, folder / RESULTS_NPZ)
    edit(session, ResultsStore.load(folder / RESULTS_NPZ) if given else None, dish_clip, folder)
    assert frames_on_disk(folder) == left and NEW_NPZ not in files_of(folder)
    assert_follows_the_animal(dish_clip, folder, "A", GRID)  # frames 40 to 118 of A are the job's records
    assert session.to_json() != before


@pytest.mark.parametrize("given", [True, False], ids=["a store given", "store=None"])
def test_an_edit_with_nothing_to_remove_is_refused_too_while_results_new_npz_is_there(left_aside, dish_clip,
                                                                                      monkeypatch, given):
    # results.npz has A up to frame 38, so "End track here" on frame 50 finds nothing to remove in it. The job's
    # records of A up to frame 118 are in results.new.npz: they would outlast the end that the session records.
    folder, session, store = left_aside
    before, files = session.to_json(), files_of(folder)
    results, aside = (folder / RESULTS_NPZ).read_bytes(), (folder / NEW_NPZ).read_bytes()

    with pytest.raises(RuntimeError) as refused:
        end_track(session, store if given else None, "A", 50, folder)
    assert files_of(folder) == files and (folder / NEW_NPZ).read_bytes() == aside
    assert session.to_json() == before and (folder / RESULTS_NPZ).read_bytes() == results
    assert_says_how_to_go_on(str(refused.value))

    monkeypatch.undo()  # the file is renamed: the same call ends the track in the job's results
    os.replace(folder / NEW_NPZ, folder / RESULTS_NPZ)
    done = end_track(session, None, "A", 50, folder)
    assert done.results_file == folder / RESULTS_NPZ
    assert frames_on_disk(folder) == {"A": GRID[:26], "B": GRID, "C": GRID} and session.tracks[0].ended_at == 50


# ---------------------------------------------------------------------------------------------
# Clicks on frame k add up


@pytest.mark.parametrize("undone", [0, 1], ids=["both clicks", "one after undo_prompt"])
def test_the_clicks_of_two_retracks_from_the_same_frame_add_up(tracked, dish_clip, undone):
    folder, session, store = tracked
    first = clicks_at(dish_clip, "A", K)
    second = Prompt(frame=K, frame_hash=first.frame_hash, decoder=first.decoder, points_px=[[30.5, 40.5]], labels=[0])
    retrack_from(session, store, ["A"], K, {"A": first}, folder)
    retrack_from(session, store, ["A"], K, {"A": second}, folder)  # a second look at the preview: one more click
    for _ in range(undone):
        assert undo_prompt(session, store, "A") is True  # takes the newest click away

    on_a = center(dish_clip, "A", K)
    points, labels = ([on_a, (30.5, 40.5)], [1, 0]) if not undone else ([on_a], [1])
    segmenter = Watched(ExactFake(dish_clip))
    assert run(dish_clip, session, folder, segmenter)[0] == "complete"
    ((_, (prompt,)),) = segmenter.starts  # what the model was given to start the run on frame 40
    assert (prompt.obj_id, prompt.points_px, prompt.labels) == ("A", points, labels)
    assert_follows_the_animal(dish_clip, folder, "A", GRID)
