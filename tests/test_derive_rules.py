"""Tests for outline_tracker.derive: the rules around the quantities of tests/test_derive.py:
lost rows, the heading reference (decision X17), the head guess, shapes without an extent or a
hull, the largest piece of a mask with several (the size check), the settings, a changed
calibration, and speed.

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
from outline_tracker.derive_outline import hull_area_and_feret
from outline_tracker.measure import measure_mask
from outline_tracker.results import TrackArrays
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


def straight_mask(kind, length):
    """`length` pixels in one straight line, one pixel wide, with 2 px of background around them,
    indexed [row, column]: a row, a column, or a diagonal (down or up to the right on screen)."""
    line = {"row": np.ones((1, length), bool), "column": np.ones((length, 1), bool),
            "diagonal": np.eye(length, dtype=bool), "anti-diagonal": np.eye(length, dtype=bool)[::-1]}[kind]
    return np.pad(line, 2)


def turned(alpha_deg):
    """The calibration TILTED with its +x axis at `alpha_deg` degrees instead of 25."""
    return dataclasses.replace(TILTED, alpha_rad=np.radians(alpha_deg))


STRAIGHT = ["row", "column", "diagonal", "anti-diagonal"]
AXIS_ANGLES = [25.0, -30.0, 137.0, 90.0]   # degrees; 25 is TILTED
PLACES = [(700, 400), (37, 911), (1500, 600)]   # (column, row) of the mask array's corner, px


@pytest.mark.parametrize("alpha_deg", AXIS_ANGLES)
@pytest.mark.parametrize("kind", STRAIGHT)
def test_a_straight_one_pixel_wide_mask_without_logits_has_no_hull_at_any_axis_angle(kind, alpha_deg):
    # Without logits the outline is the chain of the pixel centers, there and back: every point on
    # one line, of length 2 (n - 1) steps of 1 px (row, column) or sqrt(2) px (diagonals). On one
    # line means no hull, whatever the calibration: turning the axes must not turn it into numbers.
    step = 1.0 if kind in ("row", "column") else np.sqrt(2.0)
    for length in (2, 12, 30):
        for place in PLACES:
            result = shapes.pixel_result(straight_mask(kind, length), offset=place)
            track = h.derive([measure_mask(result, 0, h.FULL_HD, "coarse")], turned(alpha_deg))
            assert np.isnan(track.solidity[0]) and np.isnan(track.feret_max_mm[0]), (length, place)
            assert track.perimeter_mm[0] == pytest.approx(2 * (length - 1) * step * K, rel=1e-6), (length, place)
            assert track.circularity[0] == pytest.approx(0.0, abs=1e-6), (length, place)


@pytest.mark.parametrize("alpha_deg", AXIS_ANGLES)
@pytest.mark.parametrize("kind", STRAIGHT)
def test_the_same_straight_masks_with_logits_have_a_hull(kind, alpha_deg):
    # Logits +1 on the n pixels and -1 around them: the level set runs through the midpoints to
    # the neighbors. For a row or column that is a hexagon n px from tip to tip and 1 px wide; for
    # a diagonal it is a rectangle (2n - 1) / sqrt(2) px long and 1 / sqrt(2) px wide. Both are
    # convex (solidity 1), and the largest distance is n px, or the rectangle's diagonal.
    # Each end of that distance lies within half a spacing of the 256 stored points, so it may
    # come out short by one spacing. These outlines are at most 2 sqrt(2) times as long as their
    # largest distance, so that is at most 2 sqrt(2) / 256 = 1.1% of it.
    for length in (2, 12, 30):
        feret_px = float(length) if kind in ("row", "column") else np.sqrt(((2 * length - 1) ** 2 + 1) / 2.0)
        for place in PLACES:
            result = shapes.pixel_result(straight_mask(kind, length), offset=place, logits=True)
            track = h.derive([measure_mask(result, 0, h.FULL_HD, "coarse")], turned(alpha_deg))
            assert track.solidity[0] == pytest.approx(1.0, abs=1e-3), (length, place)
            assert track.feret_max_mm[0] == pytest.approx(feret_px * K, rel=0.012), (length, place)
            assert track.feret_max_mm[0] <= feret_px * K * (1 + 1e-6), (length, place)


@pytest.mark.parametrize("alpha_deg", [0.0] + AXIS_ANGLES)
def test_a_long_strip_two_pixels_wide_keeps_its_hull_at_any_axis_angle(alpha_deg):
    # 400 x 2 pixels without logits: the outline is the rectangle through the centers of the
    # border pixels, 399 x 1 px. Thin, but not a line: convex, and its diagonal is the diameter
    # (short by at most one spacing of the stored points, 800 / 256 px, under 0.8%).
    strip = shapes.pixel_result(shapes.blocks((6, 404), (2, 4, 2, 402)), offset=(700, 400))
    track = h.derive([measure_mask(strip, 0, h.FULL_HD, "coarse")], turned(alpha_deg))
    assert track.solidity[0] == pytest.approx(1.0, abs=1e-6)
    assert track.feret_max_mm[0] == pytest.approx(np.hypot(399.0, 1.0) * K, rel=0.008)


def test_points_on_one_line_have_no_hull_wherever_the_line_lies_and_a_sliver_has_one():
    # 51 points on a segment of length 2, turned, moved about and given in another unit: always on
    # one line, so no hull. The same with the middle point 2e-11 off the line, a few times the
    # rounding that the world transform leaves on a short line: still no hull. With the middle
    # point 2e-5 off they are a triangle of base 2 and height 2e-5: a hull of area 2e-5 whose
    # largest distance is still the base. Areas scale with the unit squared, distances with the unit.
    position = np.linspace(-1.0, 1.0, 51)
    for angle_deg in (0.0, 17.0, 45.0, 90.0, 133.3):
        along_line = shapes.direction(np.radians(angle_deg))
        across = np.array([-along_line[1], along_line[0]])
        for middle in ((0.0, 0.0), (26.31, -17.47), (-311.2, 904.6)):
            for unit in (1.0, 1e-6, 1e6):
                case = (angle_deg, middle, unit)
                line = np.asarray(middle) + position[:, None] * along_line
                assert np.isnan(hull_area_and_feret(line * unit)).all(), case
                assert np.isnan(derive.feret_max(line * unit)), case
                wobbly, sliver = line.copy(), line.copy()
                wobbly[25] += 2e-11 * across
                sliver[25] += 2e-5 * across
                assert np.isnan(hull_area_and_feret(wobbly * unit)).all(), case
                area, feret = hull_area_and_feret(sliver * unit)
                assert area == pytest.approx(2e-5 * unit ** 2, rel=1e-6), case
                assert feret == pytest.approx(2.0 * unit, rel=1e-9), case


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


# --------------------------------------------------------------------------- the largest piece (size check)


@pytest.mark.parametrize("line_at, block_at", [((3, 15), (4, 5)), ((4, 5), (3, 15)), ((3, 5), (3, 15)),
                                               ((5, 15), (3, 5))])
def test_of_two_equal_pieces_the_size_check_takes_the_one_the_outline_goes_around(line_at, block_at):
    # Two pieces of 6 px each, their top-left pixels at (row, column) `line_at` and `block_at`: a
    # line of 1 x 6 px, L1 = 4 sqrt((6^2 - 1) / 12) = 6.83 px, and a block of 2 rows x 3 columns,
    # L1 = 4 sqrt((3^2 - 1) / 12) = 3.27 px. Neither is the larger one. The stored outline goes
    # around one of them, and the size check must be of that same piece. Without logits an outline
    # runs through the centers of the border pixels: it spans 5 px in u around the line and 2 px
    # around the block.
    mask = shapes.blocks((10, 30), (line_at[0], line_at[0] + 1, line_at[1], line_at[1] + 6),
                         (block_at[0], block_at[0] + 2, block_at[1], block_at[1] + 3))
    one = measure_mask(shapes.pixel_result(mask, offset=(700, 400)), 0, h.FULL_HD, "coarse")
    assert (one.area_px, one.n_components, one.second_fraction) == (12, 2, 1.0)
    span = float(np.ptp(one.outline_px[:, 0]))
    assert span == pytest.approx(5.0, abs=1e-3) or span == pytest.approx(2.0, abs=1e-3)
    longest = 6 if span > 4.0 else 3   # pixels along the major axis of the piece with the outline
    track = h.derive([one], UPRIGHT)
    assert track.px_along_major[0] == pytest.approx(4.0 * np.sqrt((longest ** 2 - 1) / 12.0), rel=1e-9)


def line_and_block(shape, line_at, block_at, *, logits, frame=0):
    """The record of the two pieces of the test above (a line of 1 x 6 px, a block of 2 rows x 3
    columns, top-left pixels at (row, column) `line_at` and `block_at`) in arrays of `shape` whose
    first row and column are those of the full frame, and L1 in px of the piece that the stored
    outline goes around. The outline says which piece that is by its width in u: without logits
    it runs through the centers of the border pixels (5 px around the line, 2 px around the
    block); with logits of +1 on the mask and -1 off it, along the pixel edges (6 px and 3 px,
    less up to 0.1 px where none of the 256 points falls on an end of the line)."""
    mask = shapes.blocks(shape, (line_at[0], line_at[0] + 1, line_at[1], line_at[1] + 6),
                         (block_at[0], block_at[0] + 2, block_at[1], block_at[1] + 3))
    one = measure_mask(shapes.pixel_result(mask, logits=logits), frame, h.FULL_HD, "coarse")
    assert (one.area_px, one.n_components, one.second_fraction) == (12, 2, 1.0)
    span = float(np.ptp(one.outline_px[:, 0]))
    wide, narrow = (6.0, 3.0) if logits else (5.0, 2.0)
    assert span == pytest.approx(wide, abs=0.1) or span == pytest.approx(narrow, abs=0.1)
    longest = 6 if span > 4.0 else 3   # pixels along the major axis of the piece with the outline
    return one, 4.0 * np.sqrt((longest ** 2 - 1) / 12.0)


@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("line_at, block_at", [((0, 7), (1, 1)), ((1, 0), (0, 8)), ((0, 0), (1, 8)),
                                               ((2, 1), (0, 9))])
def test_of_two_equal_pieces_on_the_first_row_of_the_models_image_the_size_check_follows_the_outline(
        line_at, block_at, logits):
    # The upper piece lies on row 0 of the arrays that measure_mask gets, as a mask on the first
    # row of the model's image does. measure_mask then had no row above the mask, and which of two
    # equal pieces it takes there is not what it takes when there is one (OpenCV labels in blocks
    # of 2 x 2 pixels). The stored crop does not hold that row, but the stored outline holds the
    # answer: the size check is of the piece the outline goes around, here too.
    one, expected = line_and_block((6, 20), line_at, block_at, logits=logits)
    assert one.mask_offset[1] == 0 and one.edge
    track = h.derive([one], UPRIGHT)
    assert track.px_along_major[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("logits", [False, True])
def test_wherever_two_equal_pieces_lie_in_the_models_image_the_size_check_follows_the_outline(logits):
    # Every place of the line and of the block in arrays of 4 rows x 12 columns in which they do
    # not touch (also not at a corner), one frame per place: with and without a free row above
    # the mask, a free column to its left, and so on. On every frame the size check is of the
    # piece with the outline.
    # Counted by hand: 4 * 7 places of the line times 3 * 10 of the block are 840. They touch
    # unless a row or a column is free between them: 10 of the 12 pairs of rows leave no free
    # row, and 3 + 2 + 1 + 0 + 1 + 2 + 3 = 12 of the 70 pairs of columns leave a free column, so
    # 10 * 58 = 580 touch and 260 are left.
    records, expected = [], []
    for line_row, line_col, block_row, block_col in np.ndindex(4, 7, 3, 10):
        free_row = block_row > line_row + 1 or line_row > block_row + 2
        free_column = block_col > line_col + 6 or line_col > block_col + 3
        if free_row or free_column:
            one, length = line_and_block((4, 12), (line_row, line_col), (block_row, block_col), logits=logits,
                                         frame=len(records))
            records.append(one)
            expected.append(length)
    assert len(records) == 260 and len(set(expected)) == 2   # the outline is around the line on some frames
    track = h.derive(records, UPRIGHT)
    assert track.px_along_major == pytest.approx(np.array(expected), rel=1e-9)


@pytest.mark.parametrize("logits", [False, True])
def test_an_equal_piece_inside_a_ring_does_not_take_the_size_check_from_the_ring(logits):
    # The 1 px thick border of a square of 13 x 13 px (13^2 - 11^2 = 48 px) and, inside it and not
    # touching it, a block of 6 rows x 8 columns (48 px). The outline of the ring runs around the
    # block too, but along the ring's pixels: the size check is the ring's.
    # Ring, about its center: each of its two full rows gives 2 * (1 + 4 + 9 + 16 + 25 + 36) = 182
    # and the 22 other pixels, 6 px from the center line, give 22 * 36 = 792; the variance is
    # (2 * 182 + 792) / 48 = 1156 / 48 px^2 along u and, by symmetry, along v. Block: (8^2 - 1) / 12
    # px^2 along its 8 columns.
    ring = shapes.blocks((13, 13), (0, 13, 0, 13)) & ~shapes.blocks((13, 13), (1, 12, 1, 12))
    mask = shapes.blocks((17, 17))
    mask[2:15, 2:15] = ring
    mask[5:11, 4:12] = True
    assert (ring.sum(), mask.sum()) == (48, 96)
    one = measure_mask(shapes.pixel_result(mask, offset=(700, 400), logits=logits), 0, h.FULL_HD, "coarse")
    assert (one.n_components, one.second_fraction) == (2, 1.0)
    span = float(np.ptp(one.outline_px[:, 0]))
    ring_wide, block_wide = (13.0, 8.0) if logits else (12.0, 7.0)   # along pixel edges, or through pixel centers
    assert span == pytest.approx(ring_wide, abs=0.1) or span == pytest.approx(block_wide, abs=0.1)
    expected = 4.0 * np.sqrt(1156 / 48) if span > 10.0 else 4.0 * np.sqrt((8 ** 2 - 1) / 12.0)
    track = h.derive([one], UPRIGHT)
    assert track.px_along_major[0] == pytest.approx(expected, rel=1e-9)


def test_only_a_mask_of_several_pieces_is_unpacked(monkeypatch):
    # Counted, not timed: a track whose masks all have one piece costs no unpacking of a stored
    # mask, lost rows included; of the three rows with the stray pixels only the second is looked at.
    unpacked = []
    stored_mask = TrackArrays.mask

    def counted(self, row):
        unpacked.append(int(row))
        return stored_mask(self, row)

    monkeypatch.setattr(TrackArrays, "mask", counted)
    h.derive([tilted(0, 20.0), tilted(1, 30.0), h.lost_record(2), tilted(3, 40.0)])
    assert unpacked == []
    h.derive(h.stray_pixel_records())
    assert unpacked == [1]


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
