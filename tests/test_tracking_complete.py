"""When a session says its results are complete (SPEC 6.4: "Partial outputs are valid and marked
"complete": false in session.json").

The flag describes the results, not the job that ran last: it is true only when every object with
clicks has a record on every frame it should have. `partial_tracks` names the objects that do not.

Expected values are worked out from the frames each test puts into the results: the clip's grid
(frames 0 to 20 every 2 for the hand-made stores, 0 to 118 every 2 for `dish_clip`, which has 120
frames), where a job was cancelled, and where the video ends. Frames are video frame numbers.
"""

from dataclasses import replace

import pytest
from results_helpers import made_up
from tracking_helpers import Recorder, center, make_session, run, track

from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Clip, Prompt, Session, Track
from outline_tracker.tracking import Job, plan_runs, run_job
from outline_tracker.tracking_plan import partial_tracks

GRID = list(range(0, 120, 2))  # `dish_clip` has 120 frames; the sessions here take every 2nd
UP_TO_10 = [0, 2, 4, 6, 8, 10]


def click_again(session, clip, track_id, frame):
    """Add a click on an object's center at a later frame, as "Re-track from here" does (SPEC 6.6):
    the object's next run starts on that frame."""
    (a_track,) = [a_track for a_track in session.tracks if a_track.id == track_id]
    a_track.prompts.append(Prompt(frame=frame, points_px=[list(center(clip, track_id, frame))], labels=[1]))


def frames_of(run_folder, track_id):
    return ResultsStore.load(run_folder / "results.npz").arrays(track_id).frames.tolist()


def complete(run_folder, session):
    """What session.json says; the caller's session object must say the same."""
    saved = Session.load(run_folder / "session.json").complete
    assert session.complete is saved
    return saved


# ---------------------------------------------------------------------------------------------
# Through jobs


def test_a_job_for_another_object_does_not_make_cancelled_results_complete(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A"), track(dish_clip, "B")])
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip), Recorder(cancel_after_frame=10))
    assert status == "cancelled" and complete(tmp_path, session) is False

    # a new object, tracked to the end in a job of its own: A and B still stop at frame 10
    session.tracks.append(track(dish_clip, "C"))
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert [frames_of(tmp_path, name) for name in "ABC"] == [UP_TO_10, UP_TO_10, GRID]
    assert complete(tmp_path, session) is False

    # B goes on from frame 12, the first it does not have: A still stops at frame 10
    click_again(session, dish_clip, "B", 12)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert [frames_of(tmp_path, name) for name in "ABC"] == [UP_TO_10, GRID, GRID]
    assert complete(tmp_path, session) is False

    # A goes on too: now every object has every frame
    click_again(session, dish_clip, "A", 12)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert [frames_of(tmp_path, name) for name in "ABC"] == [GRID, GRID, GRID]
    assert complete(tmp_path, session) is True


def test_a_job_for_one_object_leaves_the_results_partial_while_others_wait(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, name) for name in "ABC"])
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip), track_ids=["B"])[0] == "complete"
    assert ResultsStore.load(tmp_path / "results.npz").track_ids == ["B"]
    assert complete(tmp_path, session) is False  # A and C have clicks and no results

    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip), track_ids=["A"])[0] == "complete"
    assert complete(tmp_path, session) is False  # C still waits
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert complete(tmp_path, session) is True


def test_an_object_that_was_never_clicked_does_not_keep_the_results_partial(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B")]
    tracks[1].prompts = []  # added to the table, never clicked: there is nothing to track
    session = make_session(dish_clip, tmp_path, tracks)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert complete(tmp_path, session) is True


def test_frames_left_out_between_two_runs_keep_the_results_partial(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip), Recorder(cancel_after_frame=10))
    assert status == "cancelled"
    click_again(session, dish_clip, "B", 20)  # not the first frame B lacks: frames 12 to 18 stay untracked
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert frames_of(tmp_path, "B") == UP_TO_10 + list(range(20, 120, 2))
    assert complete(tmp_path, session) is False


def test_where_the_video_ends_before_the_clip_objects_are_complete_at_its_last_frame(dish_clip, tmp_path):
    # the clip asks for frames 0 to 200 every 2; the video has frames 0 to 119, so tracking ends on frame 118
    tracks = [track(dish_clip, "A"), track(dish_clip, "C", frame=60)]
    session = make_session(dish_clip, tmp_path, tracks, end=200)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip), track_ids=["A"])[0] == "complete"
    assert complete(tmp_path, session) is False  # C waits
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert frames_of(tmp_path, "A") == GRID and frames_of(tmp_path, "C") == list(range(60, 120, 2))
    assert complete(tmp_path, session) is True  # A, from the earlier job, also has all the frames there are


def test_a_job_that_was_cancelled_never_says_complete(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A")])
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert complete(tmp_path, session) is True

    # a second object, cancelled after frame 10 of its run: the results are partial although A's are whole
    session.tracks.append(track(dish_clip, "B"))
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip), Recorder(cancel_after_frame=10))
    assert status == "cancelled"
    assert [frames_of(tmp_path, name) for name in "AB"] == [GRID, UP_TO_10]
    assert complete(tmp_path, session) is False


def test_an_object_whose_run_was_cancelled_before_its_first_frame_still_waits(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A")])
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    results_before = (tmp_path / "results.npz").read_bytes()

    # B's run starts and is cancelled before its first frame: asked before the run (no), then before the frame (yes)
    session.tracks.append(track(dish_clip, "B"))
    answers = iter([False])
    callbacks = replace(Recorder().callbacks(), should_cancel=lambda: next(answers, True))
    status = run_job(Job(session, tmp_path, dish_clip.path, lambda: ExactFake(dish_clip)), callbacks)
    assert status == "cancelled"
    assert (tmp_path / "results.npz").read_bytes() == results_before
    assert (session.runs[-1].tracks, session.runs[-1].frames_done) == (["B"], 0)
    # A's results are whole and unchanged, but B has clicks and no results: a job would still find work
    assert [plan.track_ids for plan in plan_runs(session, ResultsStore.load(tmp_path / "results.npz"))] == [("B",)]
    assert complete(tmp_path, session) is False

    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"
    assert complete(tmp_path, session) is True


# ---------------------------------------------------------------------------------------------
# `partial_tracks` on hand-made results: a clip of frames 0 to 20 every 2


def clicked(track_id, frame=0, **more):
    return Track(id=track_id, start_frame=frame, prompts=[Prompt(frame=frame, points_px=[[5.5, 5.5]], labels=[1])],
                 **more)


def results(**frames_by_track):
    """A store with a made-up record on each of the given video frames of each track."""
    store = ResultsStore()
    for seed, (track_id, frames) in enumerate(frames_by_track.items()):
        for frame in frames:
            store.put(track_id, made_up(frame, seed=100 * seed + frame))
    return store


def session_of(*tracks, end=20):
    return Session(clip=Clip(start=0, end=end, step=2), tracks=list(tracks))


def test_no_track_is_partial_when_each_has_every_frame_from_its_first_to_the_clips_last():
    session = session_of(clicked("A"), clicked("B", frame=6))
    store = results(A=range(0, 21, 2), B=range(6, 21, 2))  # B starts later, which is not a gap
    assert partial_tracks(session, store) == []
    assert partial_tracks(session_of(clicked("A"), clicked("B", frame=6), end=21), store) == []  # end off the grid


def test_tracks_that_stop_early_have_a_gap_or_have_no_results_are_partial_in_session_order():
    session = session_of(clicked("D"), clicked("A"), clicked("C"), clicked("B"), Track(id="E"))
    store = results(A=range(0, 21, 2), B=range(0, 11, 2),             # B stops at frame 10
                    C=[0, 2, 4, 6, 8, 12, 14, 16, 18, 20])            # C lacks frame 10; D has no record at all
    assert partial_tracks(session, store) == ["D", "C", "B"]  # E was never clicked: nothing is missing


def test_clicks_that_are_not_tracked_yet_make_a_track_partial():
    again = clicked("A")
    again.prompts.append(Prompt(frame=12, points_px=[[7.5, 7.5]], labels=[1]))  # "Re-track from here" at frame 12
    assert partial_tracks(session_of(again), results(A=range(0, 11, 2))) == ["A"]  # frames 12 to 20 were cleared
    assert partial_tracks(session_of(again), results(A=range(0, 21, 2))) == []  # and have been tracked again
    # also when the track was ended at frame 10 before: the clicks on frame 12 are still not tracked
    # (`plan_runs` refuses clicks after a track's end, tests/test_corrections.py)
    again.ended_at = 10
    assert partial_tracks(session_of(again), results(A=range(0, 11, 2))) == ["A"]


@pytest.mark.parametrize("ended_at, frames, partial", [
    (10, range(0, 11, 2), False),  # "End track here" at frame 10: its records stop there
    (11, range(0, 11, 2), False),  # ended between two grid frames: frame 10 is the last one it can have
    (10, range(0, 9, 2), True),    # stops at frame 8, before its end
    (10, range(0, 21, 2), False),  # records beyond its end do not make it partial
], ids=["to its end", "end off the grid", "before its end", "beyond its end"])
def test_a_track_that_was_ended_is_whole_when_it_reaches_its_end(ended_at, frames, partial):
    session = session_of(clicked("A", ended_at=ended_at))
    assert partial_tracks(session, results(A=frames)) == (["A"] if partial else [])


def test_the_last_frame_the_video_has_can_be_given_when_the_clip_asks_for_more():
    session = session_of(clicked("A"), clicked("B"), clicked("C", ended_at=6), clicked("D", ended_at=18))
    # A to frame 14, B to 12, C to 6 where it was ended; D, ended at frame 18, to 14
    store = results(A=range(0, 15, 2), B=range(0, 13, 2), C=range(0, 7, 2), D=range(0, 15, 2))
    assert partial_tracks(session, store) == ["A", "B", "D"]  # the clip asks for frames up to 20
    assert partial_tracks(session, store, last_frame=14) == ["B"]  # the video ends with frame 14, before D's end
    assert partial_tracks(session, store, last_frame=15) == ["B"]  # or with frame 15, which is not on the grid
