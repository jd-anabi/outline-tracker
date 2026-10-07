"""The coarse runner on synthetic clips (SPEC 6.1, 6.2, 13.2): the dish crop, run planning, positions
against the ground truth, and the inputs of review focus 1, 2 and 4. What a job does around the
frames (cancel, autosave, progress, session changes) is in tests/test_tracking_job.py, the frame
hash guard in tests/test_tracking_guard.py.

Expected values come from geometry written out here (the dish square of SPEC 6.2, where an ellipse
meets a crop's border), from the scenes' stated paths, and from the ground truth of
outline_tracker/synthetic.py. `ExactFake` returns that ground truth in whatever view the runner
names, so a position that is off shows a wrong crop corner or a wrong shift; `ThresholdFake` reads
the image itself, so it shows whether the right pixels were cut out.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the
full frame. Frames are video frame numbers.
"""

import numpy as np
import pytest
from tracking_helpers import (
    DISH_BOX,
    FULL_SMALL,
    Watched,
    box_truth,
    center,
    dish_circle,
    make_session,
    run,
    table_truth,
    track,
    track_at,
)

from outline_tracker import video
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from outline_tracker.session import Circle, Session
from outline_tracker.tracking import RunPlan, dish_box, plan_runs
from outline_tracker.tracking_plan import plan_job

GRID = list(range(0, 120, 2))  # the clips have 120 frames; the sessions here take every 2nd


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
    box = dish_box(Circle(center_px=list(center_px), radius_px=radius), (320, 240))
    assert box == expected
    assert all(type(value) is int for value in box)


@pytest.mark.parametrize("center_px, radius", [
    ((160.0, 120.0), 0.0), ((160.0, 120.0), -5.0), ((160.0, 120.0), float("nan")),
    ((float("inf"), 120.0), 50.0), ((-200.0, 100.0), 50.0), ((160.0, 400.0), 50.0),
], ids=["no radius", "negative radius", "radius not a number", "center not finite", "left of the frame",
        "below the frame"])
def test_dish_box_refuses_a_circle_that_gives_no_crop(center_px, radius):
    with pytest.raises(ValueError):
        dish_box(Circle(center_px=list(center_px), radius_px=radius), (320, 240))


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


# ---------------------------------------------------------------------------------------------
# Positions against the ground truth (SPEC 13.2)


@pytest.mark.parametrize("dish_crop, box", [(True, DISH_BOX), (False, FULL_SMALL)], ids=["crop on", "crop off"])
def test_exact_fake_positions_equal_the_true_centroids(dish_clip, tmp_path, dish_crop, box):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, name) for name in "ABC"],
                           circle=dish_circle(dish_clip.scene), dish_crop=dish_crop)
    segmenter = Watched(ExactFake(dish_clip))
    status, seen = run(dish_clip, session, tmp_path, segmenter)
    assert status == "complete" and seen.finished == ["complete"]
    # the model was shown the box, and nothing else
    assert {shape for shape, _ in segmenter.starts} == {(box[3], box[2], 3)}
    assert {(offset, size) for _, offset, size in segmenter.views} == {(box[:2], box[2:])}

    store = ResultsStore.load(tmp_path / "results.npz")
    assert store.track_ids == ["A", "B", "C"]
    for name in "ABC":
        arrays = store.arrays(name)
        assert arrays.frames.tolist() == GRID
        true_u, true_v, true_area = table_truth(dish_clip, name, GRID)
        _, _, _, at_border = box_truth(dish_clip, name, GRID, box)
        assert arrays.edge.tolist() == at_border.tolist()
        keep = ~arrays.edge  # SPEC 13.2 compares where the model saw the whole object
        assert keep.sum() >= 50
        np.testing.assert_allclose(arrays.u[keep], true_u[keep], rtol=0, atol=0.01)
        np.testing.assert_allclose(arrays.v[keep], true_v[keep], rtol=0, atol=0.01)
        assert arrays.area_px[keep].tolist() == true_area[keep].tolist()
        assert arrays.visible.all()
        assert set(arrays.mode.tolist()) == {"coarse"}
        np.testing.assert_allclose(arrays.cell_px, max(box[2:]) / 256)  # 0.875 px with the crop, 1.25 without
    # every record went to frame_result, in frame order, each frame's objects in the order of the session
    assert [(name, frame) for name, frame, _ in seen.results] == [(name, frame) for frame in GRID for name in "ABC"]
    name, frame, record = seen.results[-1]
    assert record.u == store.arrays(name).u[-1]


def test_an_object_cut_by_the_crop_is_at_the_edge_and_then_lost(dish_clip, tmp_path):
    # Circle (120, 120), R = 60: m = 1.8, the square is [58.2, 181.8]^2, so columns and rows 58 to 181.
    box = (58, 58, 124, 124)
    # B is an ellipse with semi-axes 7.25 x 3.09 px moving right along v = 120.17: its center is at
    # u = 160.55 + 0.643 (frame - 60). The last column of the box has its pixel centers at u = 181.5.
    circle = Circle(center_px=[120.0, 120.0], radius_px=60.0)
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")], circle=circle)
    assert plan_runs(session, ResultsStore())[0].input_box == box
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"

    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("B")
    assert arrays.frames.tolist() == GRID
    row = {frame: k for k, frame in enumerate(GRID)}
    # frame 76: center at 170.8, right end at 178.1: all of it is inside
    assert arrays.visible[row[76]] and not arrays.edge[row[76]]
    # frame 90: center at 179.8, so the box's last column cuts it
    assert arrays.visible[row[90]] and arrays.edge[row[90]]
    # frame 110: center at 192.7, left end at 185.4: nothing of it is inside
    assert not arrays.visible[row[110]] and not arrays.edge[row[110]]

    true_u, true_v, true_area, at_border = box_truth(dish_clip, "B", GRID, box)
    seen = true_area > 0
    assert arrays.visible.tolist() == seen.tolist()
    assert arrays.edge.tolist() == at_border.tolist()
    assert arrays.area_px.tolist() == true_area.tolist()
    np.testing.assert_allclose(arrays.u[seen], true_u[seen], rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v[seen], true_v[seen], rtol=0, atol=0.01)
    assert np.isnan(arrays.u[~seen]).all() and np.isnan(arrays.v[~seen]).all()
    # where it is whole, that is the true centroid of the full frame
    whole = seen & ~at_border
    table_u, table_v, _ = table_truth(dish_clip, "B", GRID)
    np.testing.assert_allclose(arrays.u[whole], table_u[whole], rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v[whole], table_v[whole], rtol=0, atol=0.01)


class Hashing(Watched):
    """Writes down the hash of every image it is given, and whether its memory is in one piece."""

    def __init__(self, inner):
        super().__init__(inner)
        self.hashes, self.contiguous = [], []

    def _see(self, image):
        self.hashes.append(video.frame_hash(image))
        self.contiguous.append(bool(image.flags.c_contiguous))

    def start(self, image, prompts):
        self._see(image)
        return super().start(image, prompts)

    def step(self, image):
        self._see(image)
        return super().step(image)


@pytest.mark.parametrize("dish_crop, box", [(True, DISH_BOX), (False, FULL_SMALL)], ids=["crop on", "crop off"])
def test_the_model_is_given_exactly_the_pixels_of_the_box(dish_clip, tmp_path, dish_crop, box):
    # With the crop off these are the decoded frames themselves, as last week's script gave them (SPEC 6.2).
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")], circle=dish_circle(dish_clip.scene),
                           dish_crop=dish_crop)
    segmenter = Hashing(ExactFake(dish_clip))
    status, _ = run(dish_clip, session, tmp_path, segmenter)
    assert status == "complete"
    c0, r0, width, height = box
    expected = [video.frame_hash(rgb[r0:r0 + height, c0:c0 + width])
                for _, rgb in video.iter_rgb_frames(dish_clip.path, GRID)]
    assert len(expected) == 60 and segmenter.hashes == expected
    assert all(segmenter.contiguous)


# The disks of `disk_scene` (radius 12 px): A at (60.3, 200.5) + frame (0.5, 0), B at (40.7, 110.1) +
# frame (0.37, 0.29). Over 120 frames A and B stay inside columns 28 to 132 and rows 98 to 213.
# Circle (80, 155), R = 70: m = 2.1, the square is [7.9, 152.1] x [82.9, 227.1], so it holds both.
DISK_CIRCLE, DISK_BOX = Circle(center_px=[80.0, 155.0], radius_px=70.0), (7, 82, 146, 146)


@pytest.mark.parametrize("names, circle, box", [("ABC", None, FULL_SMALL), ("AB", DISK_CIRCLE, DISK_BOX)],
                         ids=["full frame", "dish crop"])
def test_threshold_fake_follows_the_disks_within_a_quarter_pixel(disk_clip, tmp_path, names, circle, box):
    session = make_session(disk_clip, tmp_path, [track(disk_clip, name) for name in names], circle=circle)
    segmenter = Watched(ThresholdFake())
    status, _ = run(disk_clip, session, tmp_path, segmenter)
    assert status == "complete"
    assert {shape for shape, _ in segmenter.starts} == {(box[3], box[2], 3)}

    store = ResultsStore.load(tmp_path / "results.npz")
    assert store.track_ids == list(names)
    for name in names:
        arrays = store.arrays(name)
        assert arrays.frames.tolist() == GRID
        true = np.array([center(disk_clip, name, frame) for frame in GRID])  # the true centers, from the paths
        assert arrays.visible.all() and not arrays.edge.any()
        assert np.hypot(arrays.u - true[:, 0], arrays.v - true[:, 1]).max() < 0.25  # the template's tolerance
        assert (arrays.area_px > 300).all()  # pi 12^2 = 452 px
        np.testing.assert_allclose(arrays.cell_px, max(box[2:]) / 256)


# ---------------------------------------------------------------------------------------------
# Prompts and the crop (SPEC 6.2: prompts are shifted into the crop)


def test_a_point_outside_the_crop_is_dropped_and_the_others_are_shifted_into_it(dish_clip, tmp_path):
    (a_u, a_v), (b_u, b_v) = center(dish_clip, "A", 0), center(dish_clip, "B", 0)
    tracks = [
        # a positive click on A, a negative one in the frame's corner (outside the dish square, which
        # starts at column 49 and row 8), and a negative one next to A
        track_at("A", 0, [(a_u, a_v), (5.5, 5.5), (a_u + 20.0, a_v)], [1, 0, 0]),
        # a positive click left of the square, then the one on B
        track_at("B", 0, [(20.0, 100.0), (b_u, b_v)], [1, 1]),
    ]
    session = make_session(dish_clip, tmp_path, tracks, circle=dish_circle(dish_clip.scene))
    segmenter = Watched(ExactFake(dish_clip))
    status, _ = run(dish_clip, session, tmp_path, segmenter)
    assert status == "complete"

    ((shape, (for_a, for_b)),) = segmenter.starts
    assert shape == (224, 223, 3)
    assert (for_a.obj_id, for_a.labels) == ("A", [1, 0])
    np.testing.assert_allclose(for_a.points_px, [(a_u - 49, a_v - 8), (a_u + 20.0 - 49, a_v - 8)], rtol=0, atol=1e-9)
    assert (for_b.obj_id, for_b.labels) == ("B", [1])
    np.testing.assert_allclose(for_b.points_px, [(b_u - 49, b_v - 8)], rtol=0, atol=1e-9)
    # the session keeps the clicks as they were made, in px of the full frame
    assert session.tracks[0].prompts[0].points_px == [[a_u, a_v], [5.5, 5.5], [a_u + 20.0, a_v]]


@pytest.mark.parametrize("dish_crop, point", [(True, (20.0, 100.0)), (False, (330.0, 100.0))],
                         ids=["outside the dish crop", "outside the frame"])
def test_an_object_without_a_positive_point_inside_is_an_error_before_the_run(dish_clip, tmp_path, dish_crop,
                                                                               point):
    b_u, b_v = center(dish_clip, "B", 0)
    tracks = [track(dish_clip, "A"), track_at("B", 0, [point, (b_u, b_v)], [1, 0])]  # B's only positive is outside
    session = make_session(dish_clip, tmp_path, tracks, circle=dish_circle(dish_clip.scene), dish_crop=dish_crop)
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match="'B' has no positive point inside"):
        run(dish_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and segmenter.calls == []  # nothing was tracked, no model was loaded
    assert list(tmp_path.iterdir()) == []  # and nothing was written


# ---------------------------------------------------------------------------------------------
# Review focus 2, 4 and 1


def test_a_clip_that_ends_after_the_video_stops_at_the_last_frame_and_says_so(dish_clip, tmp_path):
    # the clip asks for frames 0 to 200 every 2 (101 frames); the video has frames 0 to 119
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")], end=200)
    assert len(plan_runs(session, ResultsStore())[0].frames) == 101
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete" and seen.finished == ["complete"]

    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("B")
    assert arrays.frames.tolist() == GRID
    true_u, true_v, _ = table_truth(dish_clip, "B", GRID)
    np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
    assert any("video" in line and "118" in line for line in seen.log)  # the last frame it could track
    # the progress ends at what there was to track
    done, total, _, eta_s = seen.progress[-1]
    assert (done, total, eta_s) == (60, 60, 0.0)
    saved = Session.load(tmp_path / "session.json")
    assert saved.runs[0].frames_done == 60 and saved.complete is True


def test_an_object_that_is_never_found_gives_lost_rows_and_the_run_completes(disk_clip, tmp_path):
    # D is a click on the light background: no disk comes within 80 px of (250.5, 200.5)
    tracks = [track(disk_clip, "A"), track_at("D", 0, [(250.5, 200.5)], [1])]
    session = make_session(disk_clip, tmp_path, tracks)
    status, seen = run(disk_clip, session, tmp_path, ThresholdFake())
    assert status == "complete" and seen.finished == ["complete"]

    store = ResultsStore.load(tmp_path / "results.npz")
    lost = store.arrays("D")
    assert lost.frames.tolist() == GRID
    assert not lost.visible.any() and not lost.edge.any()
    assert (lost.area_px == 0).all()
    assert np.isnan(lost.u).all() and np.isnan(lost.v).all() and np.isnan(lost.outline_px).all()
    found = store.arrays("A")
    true = np.array([center(disk_clip, "A", frame) for frame in GRID])
    assert found.visible.all()
    assert np.hypot(found.u - true[:, 0], found.v - true[:, 1]).max() < 0.25


def test_a_clip_in_a_folder_with_a_space_and_accents_tracks(clip_in_odd_folder):
    clip = clip_in_odd_folder
    run_folder = clip.path.parent / "dish_tracker_outline_zoé müller"
    session = make_session(clip, run_folder, [track(clip, "B")], circle=dish_circle(clip.scene))
    status, _ = run(clip, session, run_folder, ExactFake(clip))
    assert status == "complete"

    assert sorted(path.name for path in run_folder.iterdir()) == ["results.npz", "session.json"]
    arrays = ResultsStore.load(run_folder / "results.npz").arrays("B")
    assert arrays.frames.tolist() == GRID
    true_u, true_v, _ = table_truth(clip, "B", GRID)
    np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
    saved = Session.load(run_folder / "session.json")
    assert saved.complete is True
    assert saved.locate_video(run_folder) == clip.path
