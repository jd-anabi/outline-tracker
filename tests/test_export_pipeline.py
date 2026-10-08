"""Export through the code a user's click reaches (SPEC 6.3, 6.6, 8.2-8.6, 8.9, 13.2): fine tracks
made by `tracking.run_job`, and the three corrections made by `retrack_from`, `end_track` and
`new_piece`, each followed by the job it asks for and by `export.export_all`. The same checks with
results.npz made by hand are in tests/test_export_shapes.py and tests/test_export_corrections.py.

Every run uses `ExactFake`, which answers with the scene's ground truth in whatever part of the
frame the runner shows it, so expected values are the scenes' own: the shapes of `shapes_scene`
(an ellipse with semi-axes 40 and 15 px, a disk of radius 50 px), the paths of the scenes, the
true masks' centroids, and frame counts worked out from the grid. `Renaming` is `ExactFake` behind
another name: the piece A2 is the animal A; the track A follows the animal C from frame 40 on.

The sessions have the stick of tests/export_helpers.py: K = 0.1 mm per px, axis angle 0, so an
angle on screen is the same angle in the world. Coordinates: px in Tracker's convention (pixel
centers at +0.5), world mm with y up; frames are video frame numbers.

The rows of a track have no gap (SPEC 8.2): the last part pins that the real functions leave
none. What the export does with a results.npz that has one all the same is in
tests/test_export_pipeline_gaps.py.
"""

import json
import math

import numpy as np
import pytest
from export_helpers import (
    K,
    MODEL,
    calibrate,
    cells,
    frames_of,
    head_point,
    load,
    log_section,
    ray_ellipse,
    silent,
    table,
    true_pose,
    world,
)
from tracking_helpers import (
    DISH_BOX,
    Recorder,
    Renaming,
    box_truth,
    clicks_at,
    dish_circle,
    make_session,
    run,
    table_truth,
    track,
)

from outline_tracker import export
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.tracking import end_track, new_piece, retrack_from

GRID = list(range(0, 60, 2))        # shapes_scene: 60 frames, every 2nd
DISH_GRID = list(range(0, 120, 2))  # the dish and close-up clips: 120 frames
A_PX, B_PX, R_PX = 40.0, 15.0, 50.0
RADII = [f"r_{degrees:03d}" for degrees in range(0, 360, 5)]
POSITION_COLUMNS = ["u_px", "v_px", "x_mm", "y_mm", "area_mm2", "visible"]
RETRACK, END, PIECE = 40, 50, 70    # the frames of the corrections: rows 20, 25 and 35 of a whole track


def new_session(clip, run_folder, tracks):
    """A session for a synthetic clip with the given tracks, every 2nd frame, the scene's dish
    circle if it has one (with the dish crop on) and the stick of export_helpers."""
    scene = clip.scene
    for one in tracks:
        one.color = "#FFFF00"
    session = calibrate(make_session(clip, run_folder, tracks,
                                     circle=dish_circle(scene) if scene.dish is not None else None))
    session.processing.model = MODEL
    return session


def tracked(clip, run_folder, track_ids, segmenter=None, cancel_after_frame=None):
    """Track the named objects of a clip, coarse, into `run_folder` with one job; the session.
    `segmenter`: the stand-in, `ExactFake` if None. With `cancel_after_frame` the job is cancelled
    when that video frame has been tracked."""
    session = new_session(clip, run_folder, [track(clip, name) for name in track_ids])
    status, _ = run(clip, session, run_folder, segmenter or ExactFake(clip), Recorder(cancel_after_frame))
    assert status == ("complete" if cancel_after_frame is None else "cancelled")
    return session


def track_again(clip, session, run_folder, segmenter=None):
    """The caller's part after an edit: save the session, then the job that tracks what waits."""
    session.save(run_folder / "session.json")
    assert run(clip, session, run_folder, segmenter or ExactFake(clip))[0] == "complete"


def apart_deg(angle_a, angle_b, period):
    """The difference of two angles in degrees, both given in rad, modulo `period` degrees."""
    difference = np.degrees(np.asarray(angle_a) - np.asarray(angle_b))
    return np.abs((difference + period / 2) % period - period / 2)


# ---------------------------------------------------------------------------------------------
# Fine tracks made by the fine runner (SPEC 6.3, 8.4-8.6, 13.1 tolerances)


@pytest.fixture(scope="module")
def fine_run(shapes_clip, tmp_path_factory):
    """`shapes_scene` with the ellipse (A) and the disk (B) as fine objects, tracked by `run_job`
    and exported; the run folder. The windows are the runner's own choice."""
    folder = tmp_path_factory.mktemp("pipeline_fine") / "run"
    session = new_session(shapes_clip, folder, [track(shapes_clip, name, mode="fine") for name in "AB"])
    assert [one.fine_window_px for one in session.tracks] == [None, None]  # automatic
    assert run(shapes_clip, session, folder, ExactFake(shapes_clip))[0] == "complete"
    assert export.export_all(folder, log=silent).warnings == []
    return folder


def test_fine_ellipse_axes_and_heading_and_fine_disk_perimeter_and_solidity(fine_run, shapes_clip):
    shapes = table(fine_run / "shapes.csv")
    ellipse, disk = shapes[shapes.track_id == "A"], shapes[shapes.track_id == "B"]
    assert list(ellipse.frame) == GRID and list(disk.frame) == GRID
    assert set(table(fine_run / "positions.csv")["mode"]) == {"fine"}
    np.testing.assert_allclose(ellipse.major_mm / K, 2 * A_PX, rtol=0.02)
    np.testing.assert_allclose(ellipse.minor_mm / K, 2 * B_PX, rtol=0.02)
    true_axis = [true_pose(shapes_clip, "A", frame)[2] for frame in GRID]
    assert apart_deg(ellipse.theta_rad, true_axis, 180.0).max() < 1.0  # no head click: the axis, either way
    np.testing.assert_allclose(disk.perimeter_mm / K, 2 * math.pi * R_PX, rtol=0.005)
    assert disk.solidity.min() >= 0.99


def test_radial_profile_of_the_fine_disk_is_its_radius_at_all_72_angles(fine_run):
    radial = table(fine_run / "radial.csv")
    disk = radial[radial.track_id == "B"]
    assert list(disk.frame) == GRID and disk[RADII].shape == (30, 72)
    np.testing.assert_allclose(disk[RADII].to_numpy() / K, R_PX, rtol=0, atol=0.5)


def test_radial_profile_of_the_fine_ellipse_is_the_exact_ray_crossing(fine_run, shapes_clip):
    radial, shapes = table(fine_run / "radial.csv"), table(fine_run / "shapes.csv")
    radial, shapes = radial[radial.track_id == "A"], shapes[shapes.track_id == "A"]
    assert list(radial.frame) == list(shapes.frame) == GRID
    for (_, radii), (_, row) in zip(radial.iterrows(), shapes.iterrows(), strict=True):
        u, v, heading = true_pose(shapes_clip, "A", int(row.frame))
        # from the exported body center along the exported heading + phi, to the true ellipse (mm)
        expected = [ray_ellipse((row.core_x_mm, row.core_y_mm), row.theta_rad + math.radians(degrees), world(u, v),
                                K * A_PX, K * B_PX, heading) for degrees in range(0, 360, 5)]
        np.testing.assert_allclose(radii[RADII].to_numpy(float), expected, rtol=0.01)


def test_outlines_npz_holds_both_fine_tracks_with_128_points(fine_run):
    with np.load(fine_run / "outlines.npz") as outlines:
        assert sorted(outlines.files) == ["A__frames", "A__xieta_mm", "A__xy_mm", "B__frames", "B__xieta_mm",
                                          "B__xy_mm", "meta"]
        meta = json.loads(str(outlines["meta"]))
        assert meta["n_points"] == 128 and meta["tracks"] == ["A", "B"]
        for track_id in "AB":
            assert list(outlines[f"{track_id}__frames"]) == GRID
            assert outlines[f"{track_id}__xy_mm"].shape == (30, 128, 2)
            assert outlines[f"{track_id}__xieta_mm"].shape == (30, 128, 2)
            assert np.isfinite(outlines[f"{track_id}__xy_mm"]).all()


def test_fine_records_have_a_grid_cell_of_w_over_256_and_run_log_names_that_w(fine_run):
    session, store = load(fine_run)
    runs = log_section(fine_run, "runs:")
    assert len(runs) == 2
    # W = ceil(3 F) for the largest diameter F of the object's mask on the start frame (SPEC 6.3):
    # 3 x 80 px for the ellipse, 3 x 100 px for the disk; the range is that of the 47 px body in
    # tests/test_tracking_fine.py (139 to 144 around 141)
    for a_track, three_f in zip(session.tracks, (240, 300), strict=True):
        window = a_track.fine_window_px
        assert type(window) is int and three_f - 2 <= window <= three_f + 3
        arrays = store.arrays(a_track.id)
        assert arrays.frames.tolist() == GRID and set(arrays.mode.tolist()) == {"fine"}
        np.testing.assert_array_equal(arrays.cell_px, window / 256)  # a whole number over 256: exact
        (line,) = [line for line in runs if f"tracks {a_track.id};" in line]
        assert "fine" in line and f"W = {window} px" in line and "30 frames" in line


def test_closeup_heading_follows_the_head_click_through_the_fine_runner(closeup_clip, tmp_path):
    run_folder = tmp_path / "run"
    shrimp = track(closeup_clip, "A", mode="fine")
    shrimp.head_px = list(head_point(closeup_clip, "A", 0, 23.5))  # the front end of the 47 px body on frame 0
    session = new_session(closeup_clip, run_folder, [shrimp])
    assert run(closeup_clip, session, run_folder, ExactFake(closeup_clip))[0] == "complete"
    export.export_all(run_folder, log=silent)
    shapes = table(run_folder / "shapes.csv")
    assert list(shapes.frame) == DISH_GRID
    true_heading = [true_pose(closeup_clip, "A", frame)[2] for frame in DISH_GRID]
    assert apart_deg(shapes.theta_rad, true_heading, 360.0).max() < 5.0


# ---------------------------------------------------------------------------------------------
# The three corrections, each with the job it asks for (SPEC 6.6)


def test_after_retrack_from_only_the_rows_of_that_track_from_frame_k_on_change(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    # the first job: from frame 40 on the model follows the animal C under the name A (a swap)
    session = tracked(dish_clip, run_folder, "ABC", Renaming(dish_clip, {"A": "C"}, from_frame=RETRACK))
    export.export_all(run_folder, log=silent)
    _, before = cells(run_folder / "positions.csv")
    others = {name: (run_folder / MODEL / f"{name}.csv").read_bytes() for name in "BC"}
    assert len(log_section(run_folder, "runs:")) == 1
    assert "corrections: none" in (run_folder / "run.log").read_text(encoding="utf-8")

    retrack_from(session, None, ["A"], RETRACK, {"A": clicks_at(dish_clip, "A", RETRACK)}, run_folder)
    track_again(dish_clip, session, run_folder)
    assert export.export_all(run_folder, log=silent).warnings == []

    _, after = cells(run_folder / "positions.csv")
    assert [(row["track_id"], int(row["frame"])) for row in after] == [(name, frame) for name in "ABC"
                                                                       for frame in DISH_GRID]
    assert [(row["track_id"], row["frame"], row["t_s"]) for row in after] == [
        (row["track_id"], row["frame"], row["t_s"]) for row in before]
    # the premise, from the ground truth: on those frames the animals A and C are never at the same u
    late = DISH_GRID[RETRACK // 2:]
    assert np.abs(table_truth(dish_clip, "A", late)[0] - table_truth(dish_clip, "C", late)[0]).min() > 1.0
    for old, new in zip(before, after, strict=True):
        same = dict(zip(POSITION_COLUMNS, (old[column] == new[column] for column in POSITION_COLUMNS), strict=True))
        if new["track_id"] == "A" and int(new["frame"]) >= RETRACK:
            assert not same["u_px"] and not same["x_mm"], new
        else:
            assert all(same.values()), new
    # A is the animal A again, on every frame
    mine = table(run_folder / "positions.csv")
    mine = mine[mine.track_id == "A"]
    u, v, _, _ = box_truth(dish_clip, "A", DISH_GRID, DISH_BOX)
    np.testing.assert_allclose(mine.u_px, u, rtol=0, atol=0.01 + 0.0005)
    np.testing.assert_allclose(mine.v_px, v, rtol=0, atol=0.01 + 0.0005)
    np.testing.assert_allclose(mine.x_mm, world(u, v)[0], rtol=0, atol=K * 0.01 + 1e-6)
    # the Tracker-format files of the other tracks: the same bytes
    assert {name: (run_folder / MODEL / f"{name}.csv").read_bytes() for name in "BC"} == others
    (correction,) = session.corrections
    (line,) = log_section(run_folder, "corrections:")
    assert line.startswith(correction.time) and "retrack" in line and "track A" in line and "frame 40" in line
    runs = log_section(run_folder, "runs:")
    assert len(runs) == 2 and "tracks A;" in runs[1] and "start frame 40" in runs[1] and "40 frames" in runs[1]


def test_after_end_track_and_new_piece_the_export_has_a_a2_and_b(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    session = tracked(dish_clip, run_folder, "AB")
    export.export_all(run_folder, log=silent)

    end_track(session, None, "B", END, run_folder)
    piece = new_piece(session, "A", PIECE, clicks_at(dish_clip, "A2", PIECE, on="A")).track_ids[0]
    assert piece == "A2"
    track_again(dish_clip, session, run_folder, Renaming(dish_clip, {"A2": "A"}))  # the piece A2 is the animal A
    assert export.export_all(run_folder, log=silent).warnings == []

    later = DISH_GRID[PIECE // 2:]  # frames 70 to 118: 25 frames
    for name in ("positions.csv", "shapes.csv"):
        assert list(table(run_folder / name).track_id) == ["A"] * 60 + ["A2"] * 25 + ["B"] * 26
        assert frames_of(run_folder, name, "A") == DISH_GRID
        assert frames_of(run_folder, name, "A2") == later                       # its rows start at m
        assert frames_of(run_folder, name, "B") == DISH_GRID[:END // 2 + 1]     # its rows end at k
    # one .csv per track in the Tracker-format folder and nothing else
    assert sorted(path.name for path in (run_folder / MODEL).iterdir()) == ["A.csv", "A2.csv", "B.csv"]
    rows = {name: (run_folder / MODEL / f"{name}.csv").read_text().splitlines()[2:] for name in ("A", "A2", "B")}
    assert [len(rows[name]) for name in ("A", "A2", "B")] == [60, 25, 26]
    assert rows["A2"][0].split(",")[1] == "70" and rows["B"][-1].split(",")[1] == "50"
    # the piece's rows are the animal A from frame 70 on
    positions = table(run_folder / "positions.csv")
    mine = positions[positions.track_id == "A2"]
    u, v, _, _ = box_truth(dish_clip, "A", later, DISH_BOX)
    np.testing.assert_allclose(mine.u_px, u, rtol=0, atol=0.01 + 0.0005)
    np.testing.assert_allclose(mine.v_px, v, rtol=0, atol=0.01 + 0.0005)
    assert set(mine.visible) == {1}
    ended, made = log_section(run_folder, "corrections:")
    assert "end" in ended and "track B" in ended and "frame 50" in ended
    assert "new_piece" in made and "track A2" in made and "frame 70" in made


# ---------------------------------------------------------------------------------------------
# Rows with no gap (SPEC 8.2): every grid frame from a track's first to its last


def assert_every_frame_seen(run_folder, track_id, frames):
    """positions.csv and shapes.csv have the track on exactly these frames, visible on each."""
    for name in ("positions.csv", "shapes.csv"):
        assert frames_of(run_folder, name, track_id) == list(frames)
    positions = table(run_folder / "positions.csv")
    assert set(positions[positions.track_id == track_id].visible) == {1}


def test_an_edit_that_would_leave_a_gap_is_refused_and_the_export_has_none(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    session = tracked(dish_clip, run_folder, "A", cancel_after_frame=6)  # A has frames 0, 2, 4 and 6
    # a run from frame 14 would leave frames 8, 10 and 12 without records: frame 8 is the latest
    with pytest.raises(ValueError, match=r"\bframe 8\b"):
        retrack_from(session, None, ["A"], 14, {"A": clicks_at(dish_clip, "A", 14)}, run_folder)
    track_again(dish_clip, session, run_folder)  # nothing waits: the job tracks nothing
    assert export.export_all(run_folder, log=silent).warnings == []
    assert_every_frame_seen(run_folder, "A", [0, 2, 4, 6])


def test_a_job_that_goes_on_after_a_cancelled_run_leaves_no_gap(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    session = tracked(dish_clip, run_folder, "A", cancel_after_frame=6)
    # it goes on from the next grid frame, frame 8
    retrack_from(session, None, ["A"], 8, {"A": clicks_at(dish_clip, "A", 8)}, run_folder)
    track_again(dish_clip, session, run_folder)
    assert export.export_all(run_folder, log=silent).warnings == []
    assert_every_frame_seen(run_folder, "A", DISH_GRID)
    mine = table(run_folder / "positions.csv")
    u, v, _, _ = box_truth(dish_clip, "A", DISH_GRID, DISH_BOX)
    np.testing.assert_allclose(mine.u_px, u, rtol=0, atol=0.01 + 0.0005)
    np.testing.assert_allclose(mine.v_px, v, rtol=0, atol=0.01 + 0.0005)
