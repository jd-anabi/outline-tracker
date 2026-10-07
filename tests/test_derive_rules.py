"""Tests for outline_tracker.derive: the rules around the quantities of tests/test_derive.py:
lost rows, the heading reference (decision X17), the head guess, shapes without an extent or a
hull, the settings, a changed calibration, and speed.

Tracks are analytic shapes sent through measure_mask (tests/derive_helpers.py). Shapes and clicks
are in image px, angles of shapes in rad from +u toward +v; derived values are in mm and rad in
the world frame, y up, where an image angle a is the world angle -(a + alpha) (SPEC 3.2).
"""

import dataclasses
import time

import analytic_shapes as shapes
import derive_helpers as h
import numpy as np
import pytest
from derive_helpers import CENTER, FPS, TILTED, UPRIGHT, K

from outline_tracker import derive
from outline_tracker.measure import measure_mask
from outline_tracker.session import Circle

DISH = Circle(center_px=[900.2, 520.7], radius_px=400.0)
FLOATS = ("x_mm", "y_mm", "u_px", "v_px", "perimeter_mm", "major_mm", "minor_mm", "eccentricity", "theta_rad",
          "core_x_mm", "core_y_mm", "core_frac", "solidity", "circularity", "feret_max_mm", "largest_fraction",
          "px_along_major", "cells_along_major", "wall_dist_centroid_mm", "wall_dist_min_mm")


def tilted(frame, angle_deg, a=40.0, b=15.0):
    return h.ellipse_record(frame, CENTER, a, b, np.radians(angle_deg))


# --------------------------------------------------------------------------- lost rows


def test_a_lost_row_keeps_its_place_and_holds_no_numbers():
    records = [tilted(10, 20.0), h.lost_record(12), h.lost_record(14), tilted(16, 30.0)]
    track = h.derive(records, TILTED, head_px=h.along(CENTER, np.radians(20.0), 30.0), circle=DISH)
    assert track.frame.tolist() == [10, 12, 14, 16]
    assert track.t_s == pytest.approx(np.array([10, 12, 14, 16]) / FPS, rel=1e-15)
    assert track.visible.tolist() == [1, 0, 0, 1] and track.mode.tolist() == ["coarse"] * 4
    for name in FLOATS:
        column = getattr(track, name)
        assert np.isnan(column[1:3]).all() and np.isfinite(column[[0, 3]]).all(), name
    assert track.area_mm2[1:3].tolist() == [0.0, 0.0]                 # k^2 times a pixel count of 0
    assert track.n_components.tolist() == [1, 0, 0, 1] and track.shape_ok[1:3].tolist() == [0, 0]
    for name in ("outline_xy_mm", "outline_xieta_mm", "radial_mm", "heading"):
        block = getattr(track, name)
        assert np.isnan(block[1:3]).all() and np.isfinite(block[[0, 3]]).all(), name
    assert not track.orient.any() and not track.headguess.any()       # a lost row is LOST, nothing else
    # the heading goes on from the last frame before the gap: 10 degrees further in the image
    assert np.degrees(track.theta_rad[3] - track.theta_rad[0]) == pytest.approx(-10.0, abs=1.0)


def test_a_track_that_was_never_found_is_all_nan():
    track = h.derive([h.lost_record(0), h.lost_record(2)], TILTED, circle=DISH)
    assert track.visible.tolist() == [0, 0] and track.headguess.all() and not track.orient.any()
    assert np.isnan(track.theta_rad).all() and np.isnan(track.radial_mm).all() and np.isnan(track.x_mm).all()


# --------------------------------------------------------------------------- the heading reference (X17)


def test_one_axis_jump_flags_one_frame_and_becomes_the_reference():
    # 75 degrees between two frames is more than 60: that frame is flagged, the next ones are not.
    records = [tilted(i, 10.0) for i in range(5)] + [tilted(5 + i, 85.0) for i in range(5)]
    track = h.derive(records, TILTED, head_px=h.along(CENTER, np.radians(10.0), 30.0))
    assert np.flatnonzero(track.orient).tolist() == [5]
    assert np.degrees(track.theta_rad[5:] - track.theta_rad[4]) == pytest.approx(-75.0, abs=1.0)
    # 55 degrees is not a jump
    calm = h.derive([tilted(0, 10.0), tilted(1, 65.0)], TILTED, head_px=h.along(CENTER, np.radians(10.0), 30.0))
    assert not calm.orient.any()


def test_a_turn_while_the_object_is_lost_flags_the_first_frame_after_the_gap():
    records = ([tilted(i, 10.0) for i in range(3)] + [h.lost_record(3 + i) for i in range(4)]
               + [tilted(7 + i, 85.0) for i in range(3)])
    track = h.derive(records, TILTED, head_px=h.along(CENTER, np.radians(10.0), 30.0))
    assert np.flatnonzero(track.orient).tolist() == [7]
    assert np.degrees(track.theta_rad[7:] - track.theta_rad[2]) == pytest.approx(-75.0, abs=1.0)


def test_frames_with_the_core_fallback_are_flagged_and_still_carry_the_head():
    # A 72 x 12 px ellipse: the opening removes all of it, so every frame takes the fallback
    # (tests/test_measure_core.py). Flagged on every frame, and the head still goes round once.
    steps = np.radians(5.0) * np.arange(73)
    records = [h.ellipse_record(i, CENTER, 36.0, 6.0, 0.2 + steps[i]) for i in range(73)]
    assert all(one.core_fallback for one in records)
    track = h.derive(records, TILTED, head_px=h.along(CENTER, 0.2, 30.0))
    assert track.orient.all()
    assert np.degrees(np.abs(track.theta_rad - (h.world_angle(TILTED, 0.2) - steps))).max() < 2.0


def test_a_round_first_frame_takes_its_side_from_the_click_and_is_no_reference():
    # Frame 0 is nearly round (flagged); the click still says where the head is, and the first
    # frame with an axis takes the tip on that side.
    records = [tilted(0, 10.0, 20.0, 19.0)] + [tilted(1 + i, 10.0) for i in range(3)]
    for side in (1, -1):
        track = h.derive(records, TILTED, head_px=h.along(CENTER, np.radians(10.0), side * 30.0))
        assert track.orient.tolist() == [True, False, False, False]
        truth = h.world_angle(TILTED, np.radians(10.0) if side == 1 else np.radians(190.0))
        assert h.apart_deg(track.theta_rad[1:], truth).max() < 1.0
        assert h.apart_deg(track.theta_rad[0], truth) < 90.0


def test_the_head_click_belongs_to_the_start_frame():
    # The object backs away 10 px a frame. The click is 15 px ahead of where it is on the start
    # frame (frame 4), which is 25 px behind where it was on frame 0.
    a = np.radians(33.0)
    records = [h.ellipse_record(i, h.along(CENTER, a, -10.0 * i), 40.0, 15.0, a) for i in range(8)]
    track = h.derive(records, TILTED, head_px=h.along(CENTER, a, -25.0), start_frame=4)
    assert h.apart_deg(track.theta_rad, h.world_angle(TILTED, a)).max() < 1.0
    # lost on its start frame: the first frame on which it was found stands in
    records[4] = h.lost_record(4)
    late = h.derive(records, TILTED, head_px=h.along(CENTER, a, -25.0), start_frame=4)
    assert h.apart_deg(late.theta_rad[late.visible == 1], h.world_angle(TILTED, a + np.pi)).max() < 1.0


# --------------------------------------------------------------------------- the head guess


@pytest.mark.parametrize("angle_deg", [33.0, 100.0, -65.0])
@pytest.mark.parametrize("drift", ["none", "half a pixel along", "three pixels across"])
def test_with_too_little_motion_along_the_axis_the_head_is_the_end_with_x_not_negative(angle_deg, drift):
    a = np.radians(angle_deg)
    way, reach = {"none": (a, 0.0), "half a pixel along": (a + np.pi, 0.5),
                  "three pixels across": (a + np.pi / 2, 3.0)}[drift]
    records = [h.ellipse_record(i, h.along(CENTER, way, reach * i / 9), 40.0, 15.0, a) for i in range(10)]
    track = h.derive(records, TILTED)
    assert track.headguess.all()
    assert (track.heading[:, 0] > 0).all()
    assert h.axis_apart_deg(track.theta_rad, h.world_angle(TILTED, a)).max() < 1.0


def test_the_guess_uses_the_10th_visible_frame_or_the_last_one_of_a_short_track():
    a = np.radians(33.0)

    def moved(frame, travelled):
        return h.ellipse_record(frame, h.along(CENTER, a, travelled), 40.0, 15.0, a)

    # Rows 3 and 4 are lost, so the 10th visible frame is row 11. At row 9 the object is 2 px
    # behind where it started, at row 11 it is 3 px ahead, and later it is far behind.
    travelled = [0.0, -2.0, -2.0, None, None, -2.0, -2.0, -2.0, -2.0, -2.0, 3.0, 3.0, -20.0, -20.0]
    records = [h.lost_record(i) if s is None else moved(i, s) for i, s in enumerate(travelled)]
    track = h.derive(records, TILTED)
    assert h.apart_deg(track.theta_rad[track.visible == 1], h.world_angle(TILTED, a)).max() < 1.0
    # Four frames, 1 px backward each: the last visible frame stands in for the 10th.
    short = h.derive([moved(i, -1.0 * i) for i in range(4)], TILTED)
    assert h.apart_deg(short.theta_rad, h.world_angle(TILTED, a + np.pi)).max() < 1.0


# --------------------------------------------------------------------------- shapes without extent or hull


def with_outline(points):
    """A measured ellipse whose stored outline is replaced by `points` (256 x 2, image px)."""
    return dataclasses.replace(tilted(0, 20.0), outline_px=np.asarray(points, np.float32))


def test_an_outline_on_one_line_has_no_hull_and_raises_nothing():
    # 256 points back and forth on a segment of 64 px: a polygon of length 128 px without area.
    there = np.column_stack([CENTER[0] + np.arange(128) * 0.5, np.full(128, CENTER[1])])
    track = h.derive([with_outline(np.vstack([there, there[::-1] + (0.5, 0.0)]))], UPRIGHT)
    assert np.isnan(track.solidity[0]) and np.isnan(track.feret_max_mm[0])
    assert track.perimeter_mm[0] == pytest.approx(128.0 * K, rel=1e-5)
    assert track.circularity[0] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(track.outline_xy_mm[0]).all()


def test_an_outline_that_is_one_point_gives_nan_shape_numbers():
    track = h.derive([with_outline(np.tile(CENTER, (256, 1)))], UPRIGHT, circle=DISH)
    assert track.perimeter_mm[0] == 0.0
    assert np.isnan([track.solidity[0], track.circularity[0], track.feret_max_mm[0]]).all()
    assert np.isnan(track.radial_mm[0]).all()
    assert track.outline_xy_mm[0] == pytest.approx(np.tile(UPRIGHT.to_world(*CENTER), (128, 1)), abs=1e-4)
    assert np.isfinite(track.wall_dist_min_mm[0])


def test_a_body_center_outside_the_outline_starts_the_outline_at_the_point_farthest_ahead():
    # The stored outline is a 20 px square 100 px to the side of the body center, so the head ray
    # misses it. Of its corners, (+10, +10) lies farthest along the heading (cos 20 + sin 20).
    a = np.radians(20.0)
    middle = h.along(CENTER, a + np.pi / 2, 100.0)
    corners = np.array([[10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]]) + middle
    square = np.vstack([np.linspace(corners[i], corners[i + 1], 64, endpoint=False) for i in range(4)])
    track = h.derive([with_outline(square)], TILTED, head_px=h.along(CENTER, a, 30.0))
    assert track.outline_xy_mm[0, 0] == pytest.approx(TILTED.to_world(*corners[0]), abs=1e-3 * K)
    assert h.signed_area(track.outline_xy_mm[0]) == pytest.approx(400.0 * K ** 2, rel=1e-3)
    assert np.isnan(track.radial_mm[0, 0]) and np.isfinite(track.radial_mm[0]).any()


def test_a_mask_of_one_pixel_has_no_axis():
    # One pixel: no extent, so no axis and no eccentricity; flagged, and nothing raises.
    pixel = shapes.pixel_result(shapes.blocks((5, 5), (2, 3, 2, 3)), offset=(700, 400))
    track = h.derive([measure_mask(pixel, 0, h.FULL_HD, "coarse")], UPRIGHT)
    assert (track.major_mm[0], track.minor_mm[0], track.shape_ok[0]) == (0.0, 0.0, 0)
    assert np.isnan(track.eccentricity[0]) and track.orient[0]
    assert track.theta_rad[0] == 0.0                     # nothing to go by: +x
    assert track.area_mm2[0] == pytest.approx(K ** 2)


def test_shape_ratios_use_the_polygon_area_and_area_mm2_the_pixel_count():
    # One pixel with logits +1 on it and -1 around it: the level set is the square through the
    # midpoints to its four neighbors, of area 0.5 px^2 and side sqrt(0.5) px. It is its own
    # hull, so the solidity is 1 and the circularity pi / 4; with the pixel count they would be 2
    # and pi / 2.
    pixel = shapes.pixel_result(shapes.blocks((5, 5), (2, 3, 2, 3)), offset=(700, 400), logits=True)
    track = h.derive([measure_mask(pixel, 0, h.FULL_HD, "coarse")], TILTED)
    assert track.area_mm2[0] == pytest.approx(K ** 2, rel=1e-12)
    assert track.solidity[0] == pytest.approx(1.0, abs=1e-6)
    assert track.circularity[0] == pytest.approx(np.pi / 4, rel=2e-3)
    assert track.perimeter_mm[0] == pytest.approx(4 * np.sqrt(0.5) * K, rel=2e-3)
    assert track.feret_max_mm[0] == pytest.approx(1.0 * K, rel=2e-3)


def test_feret_max_is_the_largest_distance_between_two_outline_points():
    rectangle = np.array([[10.0, 5.0], [40.0, 5.0], [40.0, 45.0], [10.0, 45.0], [25.0, 20.0]])   # 30 x 40 px
    assert derive.feret_max(rectangle) == pytest.approx(50.0)
    stored = h.disk_record(0, CENTER, 23.5).outline_px                                # as results.npz holds it
    assert derive.feret_max(stored) == pytest.approx(47.0, rel=0.005)
    assert np.isnan(derive.feret_max(h.lost_record(0).outline_px))                    # an empty preview mask
    assert np.isnan(derive.feret_max(np.array([[0.0, 0.0], [3.0, 4.0]])))             # no hull


# --------------------------------------------------------------------------- settings and calibration


def test_outline_points_and_radial_step_are_settings():
    track = h.derive([tilted(0, 20.0)], TILTED, outline_points=64, radial_step_deg=10)
    assert track.outline_xy_mm.shape == track.outline_xieta_mm.shape == (1, 64, 2)
    assert track.radial_mm.shape == (1, 36)
    full = h.derive([tilted(0, 20.0)], TILTED)
    assert track.radial_mm[0] == pytest.approx(full.radial_mm[0, ::2], rel=1e-12)
    assert h.chords(track.outline_xy_mm[0]).sum() == pytest.approx(track.perimeter_mm[0], rel=0.005)


@pytest.mark.parametrize("problem, name", [
    ({"fps": 0.0}, "fps_true"), ({"fps": -240.0}, "fps_true"), ({"fps": float("nan")}, "fps_true"),
    ({"radial_step_deg": 7}, "radial_step_deg"), ({"radial_step_deg": 0}, "radial_step_deg"),
    ({"outline_points": 2}, "outline_points"),
])
def test_a_setting_that_makes_no_sense_is_refused_by_name(problem, name):
    with pytest.raises(ValueError, match=name):
        h.derive([tilted(0, 20.0)], TILTED, **problem)


def test_a_shorter_stick_scales_lengths_and_leaves_angles_and_ratios():
    # The stick is 29 mm, not 30: k shrinks by 29/30 and nothing is tracked again (SPEC 7).
    records = [tilted(i, 20.0 + 3.0 * i) for i in range(4)]
    click = h.along(CENTER, np.radians(20.0), 30.0)
    ratio = 29.0 / 30.0
    smaller = dataclasses.replace(TILTED, k_mm_per_px=K * ratio)
    old = h.derive(records, TILTED, head_px=click, circle=DISH)
    new = h.derive(records, smaller, head_px=click, circle=DISH)
    for name in ("x_mm", "y_mm", "core_x_mm", "core_y_mm", "perimeter_mm", "major_mm", "minor_mm", "feret_max_mm",
                 "wall_dist_centroid_mm", "wall_dist_min_mm", "radial_mm", "outline_xy_mm", "outline_xieta_mm"):
        assert getattr(new, name) == pytest.approx(getattr(old, name) * ratio, rel=1e-11, abs=1e-13), name
    assert new.area_mm2 == pytest.approx(old.area_mm2 * ratio ** 2, rel=1e-12)
    for name in ("theta_rad", "eccentricity", "solidity", "circularity", "px_along_major", "cells_along_major",
                 "u_px", "v_px", "t_s", "heading"):
        assert getattr(new, name) == pytest.approx(getattr(old, name), rel=1e-11, abs=1e-13), name


def test_1200_frames_take_well_under_a_second():
    # Eight different shapes, repeated: the cost per frame is what a fine track's export pays.
    kinds = [tilted(0, 20.0 * i) for i in range(4)] + [h.record(
        lambda u, v, a=a: shapes.body_with_rods(u, v, CENTER, a)[0], 0, CENTER, half=90) for a in (0.3, 1.1, 2.0, 4.0)]
    arrays = h.arrays_of([dataclasses.replace(kinds[i % 8], frame=i) for i in range(1200)])
    track = h.Track(id="A", head_px=list(CENTER))
    h.derive(kinds[:1])   # the first call loads scipy's hull code: not part of the cost per frame
    start = time.perf_counter()
    derived = derive.derive_track(arrays, track, TILTED, FPS, DISH, h.Processing())
    seconds = time.perf_counter() - start
    assert len(derived.frame) == 1200 and np.isfinite(derived.radial_mm).all()
    assert seconds < 3.0   # 0.35 s on a laptop; the limit leaves room for a slow test machine
