"""Export: the shape numbers through the whole pipeline, radial.csv and outlines.npz (SPEC 8.4-8.6,
13.1, 13.2).

`shapes_scene` holds an ellipse (A: semi-axes 40 and 15 px, turning) and a disk (B: radius 50 px).
Coarse tracks come from `tracking.run_job` with `ExactFake`, with the dish crop on and off; fine
tracks are stored by `export_helpers.add_fine` (the fine runner is tested where it is written).
Every expected value is the shape's own: 2a, 2b, the heading of the scene's path, 2 pi R, the exact
crossing of a ray with the true ellipse; the core's centroid is that of the true mask's core, worked
out in export_helpers. Scale K = 0.1 mm per px, axis angle 0, so an angle on screen is the same
angle in the world.

On `closeup_scene` only the heading is asserted, on `dish_scene` only CONTACT and LOWRES: a body
14 px long cannot meet 2 % and 1 degree.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
from export_helpers import (
    K,
    MODEL,
    RUN_FILES,
    add_fine,
    coarse_run,
    ellipse_gap,
    empty_run,
    head_point,
    load,
    ray_ellipse,
    table,
    true_core_center,
    true_pose,
    world,
)
from overlay_helpers import lost_record

from outline_tracker import export, schema

GRID = list(range(0, 60, 2))       # shapes_scene: 60 frames, every 2nd
DISH_GRID = list(range(0, 120, 2))  # the dish and close-up clips: 120 frames
A_PX, B_PX, R_PX = 40.0, 15.0, 50.0
RADII = [f"r_{degrees:03d}" for degrees in range(0, 360, 5)]


def silent(_line):
    """A log that keeps nothing."""


@pytest.fixture(scope="module", params=["coarse, dish crop", "coarse, whole frame", "fine"])
def shapes_run(request, shapes_clip, tmp_path_factory):
    """`shapes_scene` tracked and exported in one of three ways; the run folder."""
    folder = tmp_path_factory.mktemp("shapes") / "run"
    if request.param == "fine":
        empty_run(shapes_clip, folder)
        add_fine(shapes_clip, folder, "A", 240)  # W = 3 x the object's length (SPEC 6.3)
        add_fine(shapes_clip, folder, "B", 300)
    else:
        coarse_run(shapes_clip, folder, ["A", "B"], dish_crop=request.param == "coarse, dish crop")
    export.export_all(folder, log=silent)
    return folder


@pytest.fixture(scope="module")
def fine_run(shapes_clip, tmp_path_factory):
    """`shapes_scene` with both objects as fine tracks, exported; the run folder."""
    folder = tmp_path_factory.mktemp("fine") / "run"
    empty_run(shapes_clip, folder)
    add_fine(shapes_clip, folder, "A", 240)
    add_fine(shapes_clip, folder, "B", 300)
    export.export_all(folder, log=silent)
    return folder


def _apart_deg(angle_a, angle_b, period):
    """The difference of two angles in degrees, both given in rad, modulo `period` degrees."""
    difference = np.degrees(np.asarray(angle_a) - np.asarray(angle_b))
    return np.abs((difference + period / 2) % period - period / 2)


# ---------------------------------------------------------------------------------------------
# Descriptors (SPEC 13.1 tolerances through the pipeline)


def test_ellipse_axes_and_heading_and_disk_perimeter_and_solidity(shapes_run, shapes_clip):
    shapes = table(shapes_run / "shapes.csv")
    ellipse, disk = shapes[shapes.track_id == "A"], shapes[shapes.track_id == "B"]
    assert list(ellipse.frame) == GRID and list(disk.frame) == GRID
    np.testing.assert_allclose(ellipse.major_mm / K, 2 * A_PX, rtol=0.02)
    np.testing.assert_allclose(ellipse.minor_mm / K, 2 * B_PX, rtol=0.02)
    true_axis = [true_pose(shapes_clip, "A", frame)[2] for frame in GRID]
    assert _apart_deg(ellipse.theta_rad, true_axis, 180.0).max() < 1.0  # no head click: the axis, either way
    np.testing.assert_allclose(disk.perimeter_mm / K, 2 * math.pi * R_PX, rtol=0.005)
    assert disk.solidity.min() >= 0.99
    assert set(shapes.shape_ok) == {1} and not shapes["flags"].str.contains("LOWRES").any()


def test_core_centers_and_wall_distances_are_in_the_users_axes(shapes_run, shapes_clip):
    shapes = table(shapes_run / "shapes.csv")
    cu, cv, radius = shapes_clip.scene.dish
    for track_id in "AB":
        mine = shapes[shapes.track_id == track_id]
        # In each of the three runs the model was shown the whole object, and `ExactFake` answers with
        # the true mask. So the exported core is the core of the true mask, which export_helpers
        # works out from SPEC 7.3 with scipy; the cells have 6 decimals. (The shape's own center is
        # not the expected value, because a mask is whole pixels: on this clip the true mask's
        # centroid is up to 0.11 px off the center, and its core's, whose two ends lose whole
        # pixels to the opening, up to 0.22 px.)
        core_u, core_v = np.array([true_core_center(shapes_clip, track_id, frame) for frame in GRID]).T
        x, y = world(core_u, core_v)
        np.testing.assert_allclose(mine.core_x_mm, x, rtol=0, atol=1e-6)
        np.testing.assert_allclose(mine.core_y_mm, y, rtol=0, atol=1e-6)
        u, v, _ = np.array([true_pose(shapes_clip, track_id, frame) for frame in GRID]).T
        np.testing.assert_allclose(mine.wall_dist_centroid_mm, K * (radius - np.hypot(u - cu, v - cv)), rtol=0,
                                   atol=K * 0.1)


# ---------------------------------------------------------------------------------------------
# radial.csv and outlines.npz of fine tracks (SPEC 8.5, 8.6)


def test_a_run_with_a_fine_track_gets_every_file_and_shape_rows_for_that_track_only(shapes_clip, tmp_path):
    run_folder = tmp_path / "run"
    coarse_run(shapes_clip, run_folder, ["B"])
    add_fine(shapes_clip, run_folder, "A", 240)
    report = export.export_all(run_folder, log=silent)
    assert report.warnings == []
    assert {path.name for path in run_folder.iterdir()} == RUN_FILES
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "B.csv"}
    positions = table(run_folder / "positions.csv")
    assert list(positions.track_id) == ["A"] * 30 + ["B"] * 30
    assert list(positions["mode"]) == ["fine"] * 30 + ["coarse"] * 30
    radial = table(run_folder / "radial.csv")
    assert list(radial.columns) == ["track_id", "frame", "t_s", *RADII]
    assert list(radial.track_id) == ["A"] * 30 and list(radial.frame) == GRID
    with np.load(run_folder / "outlines.npz") as outlines:
        assert sorted(outlines.files) == ["A__frames", "A__xieta_mm", "A__xy_mm", "meta"]
        assert json.loads(str(outlines["meta"]))["tracks"] == ["A"]


def test_radial_profile_of_the_fine_disk_is_its_radius_at_all_72_angles(fine_run):
    radial = table(fine_run / "radial.csv")
    disk = radial[radial.track_id == "B"]
    assert list(disk.frame) == GRID
    np.testing.assert_allclose(disk[RADII].to_numpy() / K, R_PX, rtol=0, atol=0.5)
    np.testing.assert_allclose(disk.t_s, np.array(GRID) / 240.0, rtol=0, atol=5e-8)


def test_radial_profile_of_the_fine_ellipse_is_the_exact_ray_crossing(fine_run, shapes_clip):
    radial, shapes = table(fine_run / "radial.csv"), table(fine_run / "shapes.csv")
    radial, shapes = radial[radial.track_id == "A"], shapes[shapes.track_id == "A"]
    assert list(radial.frame) == list(shapes.frame) == GRID
    for (_, radii), (_, row) in zip(radial.iterrows(), shapes.iterrows(), strict=True):
        u, v, heading = true_pose(shapes_clip, "A", int(row.frame))
        true_center = world(u, v)
        # from the exported body center along the exported heading + phi, to the true ellipse (mm)
        expected = [ray_ellipse((row.core_x_mm, row.core_y_mm), row.theta_rad + math.radians(degrees),
                                true_center, K * A_PX, K * B_PX, heading) for degrees in range(0, 360, 5)]
        np.testing.assert_allclose(radii[RADII].to_numpy(float), expected, rtol=0.01)


def test_outlines_npz_holds_frames_and_128_point_outlines_in_both_frames(fine_run, shapes_clip):
    with np.load(fine_run / "outlines.npz") as outlines:
        assert sorted(outlines.files) == ["A__frames", "A__xieta_mm", "A__xy_mm", "B__frames", "B__xieta_mm",
                                          "B__xy_mm", "meta"]
        data = {key: outlines[key] for key in outlines.files}
    meta = json.loads(str(data["meta"]))
    assert meta["n_points"] == 128 and meta["tracks"] == ["A", "B"]
    assert meta["tool_version"].startswith("outline-tracker 0.1.0") and isinstance(meta["conventions"], str)
    for track_id in "AB":
        frames, xy, xieta = (data[schema.npz_key(track_id, name)] for name in ("frames", "xy_mm", "xieta_mm"))
        assert frames.dtype == np.int32 and list(frames) == GRID
        assert xy.dtype == np.float32 and xy.shape == (30, 128, 2)
        assert xieta.dtype == np.float32 and xieta.shape == (30, 128, 2)
        # counterclockwise in world coordinates (y up): a positive shoelace area
        x, y = xy[..., 0].astype(float), xy[..., 1].astype(float)
        assert (0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1) > 0).all()
        # the first point is the head point: on the xi axis of the body frame, in front of the center
        assert (xieta[:, 0, 0] > 0).all()
        np.testing.assert_allclose(xieta[:, 0, 1], 0.0, atol=K * 0.05)
    # the disk's outline lies on the true circle, in the user's axes
    u, v, _ = np.array([true_pose(shapes_clip, "B", frame) for frame in GRID]).T
    cx, cy = world(u, v)
    from_center = np.hypot(data["B__xy_mm"][..., 0] - cx[:, None], data["B__xy_mm"][..., 1] - cy[:, None])
    np.testing.assert_allclose(from_center / K, R_PX, rtol=0, atol=0.5)
    np.testing.assert_allclose(np.hypot(data["B__xieta_mm"][..., 0], data["B__xieta_mm"][..., 1]) / K, R_PX, atol=0.5)


def test_coarse_tracks_get_shape_rows_only_with_shape_files_for_coarse(shapes_clip, tmp_path):
    run_folder = tmp_path / "run"
    coarse_run(shapes_clip, run_folder, ["A", "B"])
    export.export_all(run_folder, log=silent)
    assert len(table(run_folder / "radial.csv")) == 0
    with np.load(run_folder / "outlines.npz") as outlines:
        assert outlines.files == ["meta"]

    session, _ = load(run_folder)
    session.processing.shape_files_for_coarse = True
    session.save(run_folder / "session.json")
    export.export_all(run_folder, log=silent)
    radial = table(run_folder / "radial.csv")
    assert list(radial.track_id) == ["A"] * 30 + ["B"] * 30 and list(radial.frame) == GRID * 2
    np.testing.assert_allclose(radial[radial.track_id == "B"][RADII].to_numpy() / K, R_PX, rtol=0, atol=0.5)
    with np.load(run_folder / "outlines.npz") as outlines:
        assert sorted(outlines.files) == ["A__frames", "A__xieta_mm", "A__xy_mm", "B__frames", "B__xieta_mm",
                                          "B__xy_mm", "meta"]
        assert outlines["A__xy_mm"].shape == (30, 128, 2)
        assert json.loads(str(outlines["meta"]))["tracks"] == ["A", "B"]


def test_radial_rows_of_a_lost_frame_are_empty(shapes_clip, tmp_path):
    run_folder = tmp_path / "run"
    empty_run(shapes_clip, run_folder)
    add_fine(shapes_clip, run_folder, "B", 300)
    _, store = load(run_folder)
    store.put("B", replace(lost_record(shapes_clip, 4), mode="fine"))
    store.save(run_folder / "results.npz")
    export.export_all(run_folder, log=silent)
    line = (run_folder / "radial.csv").read_text().split("\n")[3]  # header, frames 0 and 2, then frame 4
    assert line == "B,4,0.0166667" + "," * 72
    with np.load(run_folder / "outlines.npz") as outlines:
        assert np.isnan(outlines["B__xy_mm"][2]).all() and np.isnan(outlines["B__xieta_mm"][2]).all()
        assert np.isfinite(outlines["B__xy_mm"][1]).all()


# ---------------------------------------------------------------------------------------------
# closeup_scene: the heading with a head click; dish_scene: CONTACT and LOWRES


@pytest.mark.parametrize("mode", ["coarse", "fine"])
def test_closeup_heading_follows_the_head_click(closeup_clip, tmp_path, mode):
    run_folder = tmp_path / "run"
    head = head_point(closeup_clip, "A", 0, 23.5)  # the front end of the 47 px body on the start frame
    if mode == "coarse":
        coarse_run(closeup_clip, run_folder, ["A"], heads={"A": head})
    else:
        empty_run(closeup_clip, run_folder)
        add_fine(closeup_clip, run_folder, "A", 144, head_px=head)
    export.export_all(run_folder, log=silent)
    shapes = table(run_folder / "shapes.csv")
    true_heading = [true_pose(closeup_clip, "A", frame)[2] for frame in DISH_GRID]
    assert list(shapes.frame) == DISH_GRID
    assert _apart_deg(shapes.theta_rad, true_heading, 360.0).max() < 5.0
    assert not shapes["flags"].str.contains("HEADGUESS").any()


@pytest.mark.parametrize("dish_crop", [True, False], ids=["dish crop", "whole frame"])
def test_dish_scene_flags_contact_on_the_contact_frames_and_lowres_everywhere(dish_clip, tmp_path, dish_crop):
    run_folder = tmp_path / "run"
    coarse_run(dish_clip, run_folder, ["A", "B", "C"], dish_crop=dish_crop)
    export.export_all(run_folder, log=silent)
    positions, shapes = table(run_folder / "positions.csv"), table(run_folder / "shapes.csv")
    assert list(positions["flags"]) == list(shapes["flags"])
    contact = dish_clip.scene.contact
    assert contact.track_ids == ("B", "C") and contact.frame in DISH_GRID
    # the limit is 3 px here (2 grid cells are 1.75 or 2.5 px): frames well inside it must carry the
    # flag, frames well outside it must not; the true gap comes from the two true ellipses
    gaps = np.array([ellipse_gap(dish_clip, "B", "C", frame) for frame in DISH_GRID])
    assert gaps[DISH_GRID.index(contact.frame)] == pytest.approx(contact.gap_px, abs=0.05)
    for track_id in ("B", "C"):
        flagged = positions[positions.track_id == track_id]["flags"].str.contains("CONTACT").to_numpy()
        assert flagged[gaps <= 2.7].all() and (gaps <= 2.7).sum() >= 1
        assert not flagged[gaps >= 3.3].any() and (gaps >= 3.3).sum() >= 40
        mine = shapes[shapes.track_id == track_id]
        assert set(mine.shape_ok) == {0} and mine["flags"].str.contains("LOWRES").all()  # a body 14.5 px long
    assert not positions[positions.track_id == "A"]["flags"].str.contains("CONTACT").any()
