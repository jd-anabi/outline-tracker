"""The coarse runner on synthetic clips (SPEC 6.1, 6.2): the dish crop and run planning, with one job of two starts.
What the runner makes of the frames, positions against the ground truth (SPEC 13.2), the prompts shifted into the
crop and the inputs of review focus 1, 2 and 4, is in tests/tracking/test_tracking_coarse_positions.py. What a job
does around the frames (cancel, autosave, progress, session changes) is in tests/tracking/test_tracking_job.py, the
frame hash guard in tests/tracking/test_tracking_guard.py.

Expected values come from geometry written out here (the dish square of SPEC 6.2), and from the ground truth of
outline_tracker/synthetic/clips.py. `ExactFake` returns that ground truth in whatever view the runner names, so a
position that is off shows a wrong crop corner or a wrong shift.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the
full frame. Frames are video frame numbers.
"""

import numpy as np
import pytest
from tracking_helpers import (
    DISH_BOX,
    FULL_SMALL,
    GRID,
    Watched,
    dish_circle,
    make_session,
    run,
    table_truth,
    track,
)

from outline_tracker.measure.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.core.session import Boundary, Session
from outline_tracker.tracking.job import RunPlan, dish_box, plan_runs
from outline_tracker.tracking.plan import plan_job


# ---------------------------------------------------------------------------------------------
# The dish crop (SPEC 6.2): [uc - R - m, uc + R + m] x [vc - R - m, vc + R + m], m = 0.03 R


@pytest.mark.parametrize("center_px, radius, expected", [
    # 160.3 -+ 111.24 = 49.06, 271.54; 119.8 -+ 111.24 = 8.56, 231.04
    ((160.3, 119.8), 108.0, (49, 8, 223, 224)),
    # 100 -+ 82.4 = 17.6, 182.4; 50 -+ 82.4 = -32.4 (clipped to 0), 132.4
    ((100.0, 50.0), 80.0, (17, 0, 166, 133)),
    # 300 -+ 51.5 = 248.5, 351.5 (clipped to 320); 200 -+ 51.5 = 148.5, 251.5 (clipped to 240)
    ((300.0, 200.0), 50.0, (248, 148, 72, 92)),
    # larger than the frame on every side
    ((160.0, 120.0), 500.0, (0, 0, 320, 240)),
], ids=["inside", "clipped at the top", "clipped right and bottom", "the whole frame"])
def test_dish_box_has_whole_pixel_corners_and_is_clipped_to_the_frame(center_px, radius, expected):
    box = dish_box(Boundary(center_px=list(center_px), radius_px=radius), (320, 240))
    assert box == expected
    assert all(type(value) is int for value in box)


@pytest.mark.parametrize("center_px, radius", [
    ((160.0, 120.0), 0.0), ((160.0, 120.0), -5.0), ((160.0, 120.0), float("nan")),
    ((float("inf"), 120.0), 50.0), ((-200.0, 100.0), 50.0), ((160.0, 400.0), 50.0),
], ids=["no radius", "negative radius", "radius not a number", "center not finite", "left of the frame",
        "below the frame"])
def test_dish_box_refuses_a_circle_that_gives_no_crop(center_px, radius):
    with pytest.raises(ValueError):
        dish_box(Boundary(center_px=list(center_px), radius_px=radius), (320, 240))


@pytest.mark.parametrize("has_circle, dish_crop, expected", [
    (True, True, DISH_BOX), (True, False, FULL_SMALL), (False, True, FULL_SMALL),
], ids=["circle and crop on", "crop switched off", "no circle"])
def test_the_model_sees_the_dish_square_only_with_a_circle_and_the_crop_on(dish_clip, tmp_path, has_circle,
                                                                             dish_crop, expected):
    circle = dish_circle(dish_clip.scene) if has_circle else None
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A")], circle=circle, dish_crop=dish_crop)
    (plan,) = plan_runs(session, ResultsStore())
    assert plan.input_box == expected


# ---------------------------------------------------------------------------------------------
# Runs (SPEC 6.1)


def test_coarse_objects_on_one_start_frame_share_a_run_and_a_later_start_makes_a_second(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "C", frame=60), track(dish_clip, "B")]
    session = make_session(dish_clip, tmp_path, tracks, circle=dish_circle(dish_clip.scene))
    assert plan_runs(session, ResultsStore()) == [
        RunPlan(track_ids=("A", "B"), start_frame=0, mode="coarse", frames=range(0, 120, 2), input_box=DISH_BOX),
        RunPlan(track_ids=("C",), start_frame=60, mode="coarse", frames=range(60, 120, 2), input_box=DISH_BOX),
    ]


def test_a_job_for_named_tracks_plans_the_runs_of_those_tracks_only(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "C", frame=60), track(dish_clip, "B")]
    session = make_session(dish_clip, tmp_path, tracks, circle=dish_circle(dish_clip.scene))
    from_0 = {"start_frame": 0, "mode": "coarse", "frames": range(0, 120, 2), "input_box": DISH_BOX}
    from_60 = RunPlan(track_ids=("C",), start_frame=60, mode="coarse", frames=range(60, 120, 2), input_box=DISH_BOX)
    assert plan_job(session, ResultsStore(), None) == [RunPlan(track_ids=("A", "B"), **from_0), from_60]  # all pending
    assert plan_job(session, ResultsStore(), ["B", "C"]) == [RunPlan(track_ids=("B",), **from_0), from_60]
    assert plan_job(session, ResultsStore(), []) == []
    assert [a_track.id for a_track in session.tracks] == ["A", "C", "B"]  # the session keeps its tracks
    with pytest.raises(ValueError, match=r"no track Z \(its tracks: A, C, B\)"):
        plan_job(session, ResultsStore(), ["A", "Z"])


def test_fine_objects_get_a_run_each_and_are_not_in_a_coarse_run(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B", mode="fine"), track(dish_clip, "C", mode="fine")]
    session = make_session(dish_clip, tmp_path, tracks)
    plans = plan_runs(session, ResultsStore())
    assert [(plan.track_ids, plan.start_frame, plan.mode) for plan in plans] == [
        (("A",), 0, "coarse"), (("B",), 0, "fine"), (("C",), 0, "fine")]
    assert all(plan.frames == range(0, 120, 2) for plan in plans)


def test_only_objects_with_prompts_and_without_results_are_planned(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B"), track(dish_clip, "C")]
    tracks[2].prompts = []  # an object that was added but never clicked on
    session = make_session(dish_clip, tmp_path, tracks)
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip), track_ids=["A"])
    assert status == "complete"
    (plan,) = plan_runs(session, ResultsStore.load(tmp_path / "results.npz"))
    assert plan.track_ids == ("B",)


@pytest.mark.parametrize("frame, said", [(7, "7"), (122, "122")], ids=["between grid frames", "after the clip"])
def test_a_start_frame_that_is_not_a_frame_of_the_clip_is_refused(dish_clip, tmp_path, frame, said):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A"), track(dish_clip, "B", frame=frame)])
    with pytest.raises(ValueError, match=rf"\bB\b.*\b{said}\b"):
        plan_runs(session, ResultsStore())


def test_a_mode_that_is_neither_coarse_nor_fine_is_refused(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A", mode="medium")])
    with pytest.raises(ValueError, match="medium"):
        plan_runs(session, ResultsStore())


def test_two_start_frames_are_two_sessions_of_the_model(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B"), track(dish_clip, "C", frame=60)]
    session = make_session(dish_clip, tmp_path, tracks, circle=dish_circle(dish_clip.scene))
    segmenter = Watched(ExactFake(dish_clip))
    status, _ = run(dish_clip, session, tmp_path, segmenter)
    assert status == "complete"

    # one model for the job; A and B in one session from frame 0, then C alone from frame 60
    assert segmenter.made == 1
    assert [[prompt.obj_id for prompt in prompts] for _, prompts in segmenter.starts] == [["A", "B"], ["C"]]
    assert segmenter.calls == ["start", *["step"] * 59, "close", "start", *["step"] * 29, "close"]
    later = list(range(60, 120, 2))
    assert [frame for frame, _, _ in segmenter.views] == GRID + later
    assert {(offset, size) for _, offset, size in segmenter.views} == {((49, 8), (223, 224))}

    store = ResultsStore.load(tmp_path / "results.npz")
    assert store.track_ids == ["A", "B", "C"]
    assert store.arrays("A").frames.tolist() == store.arrays("B").frames.tolist() == GRID
    c = store.arrays("C")
    assert c.frames.tolist() == later
    true_u, true_v, _ = table_truth(dish_clip, "C", later)
    np.testing.assert_allclose(c.u, true_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(c.v, true_v, rtol=0, atol=0.01)

    saved = Session.load(tmp_path / "session.json")
    assert [(r.tracks, r.start_frame, r.mode, r.frames_done) for r in saved.runs] == [
        (["A", "B"], 0, "coarse", 60), (["C"], 60, "coarse", 30)]
    assert saved.complete is True
