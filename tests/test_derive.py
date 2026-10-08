"""Tests for outline_tracker.derive: the world quantities of SPEC 7 from the pixel records and a
calibration (SPEC 13.1: Moments, Head continuity, Outline geometry, Radial profile, Solidity and
wall distance, Resolution).

Tracks are analytic shapes sent through measure_mask (tests/derive_helpers.py). Expected values
come from the shapes and from SPEC 3.2: with the axis angle alpha, an image direction (cos a,
sin a), v down, is the world direction (cos(a + alpha), -sin(a + alpha)), so its world angle is
-(a + alpha): an angle that grows in (u, v) numbers shrinks in world coordinates (det J = -1).
Shapes and clicks are in image px; derived values are in mm and rad in the world frame, y up.
"""

import analytic_shapes as shapes
import derive_helpers as h
import numpy as np
import pytest
from derive_helpers import CENTER, FPS, TILTED, UPRIGHT, K

from outline_tracker import derive, schema
from outline_tracker.session import Circle

ALPHA = np.radians(25.0)   # the axis angle of TILTED


# --------------------------------------------------------------------------- positions, moments


def test_positions_and_areas_are_the_record_in_world_units():
    one = h.ellipse_record(120, CENTER, 40.0, 15.0, 0.3)
    track = h.derive([one], UPRIGHT)
    # alpha = 0: x = k (u - u0), y = -k (v - v0), y up
    assert track.x_mm[0] == pytest.approx(K * (one.u - 960.5), abs=1e-12)
    assert track.y_mm[0] == pytest.approx(-K * (one.v - 540.5), abs=1e-12)
    assert (track.u_px[0], track.v_px[0]) == (one.u, one.v)
    assert track.core_x_mm[0] == pytest.approx(K * (one.core_u - 960.5), abs=1e-12)
    assert track.core_y_mm[0] == pytest.approx(-K * (one.core_v - 540.5), abs=1e-12)
    assert track.area_mm2[0] == pytest.approx(K ** 2 * one.area_px, rel=1e-12)   # the pixel count, not the polygon
    assert track.t_s[0] == pytest.approx(120 / FPS, rel=1e-15) and track.frame[0] == 120
    assert (track.visible[0], track.mode[0], track.n_components[0]) == (1, "coarse", 1)
    assert (track.core_frac[0], track.largest_fraction[0]) == (one.core_frac, 1.0)


def test_every_csv_column_is_a_column_of_the_derived_track():
    track = h.derive([h.ellipse_record(0, CENTER, 40.0, 15.0, 0.3), h.lost_record(2)], head_px=CENTER)
    for columns in (schema.POSITIONS, schema.SHAPES):
        for row in range(2):
            values = {c.name: getattr(track, c.name)[row] for c in columns if c.name not in ("track_id", "flags")}
            line = schema.format_row(columns, {**values, "track_id": track.track_id, "flags": ""})
            assert line.startswith(f"A,{2 * row},")
    assert track.radial_mm.shape == (2, len(schema.RADIAL) - 3)
    assert track.outline_xy_mm.shape == track.outline_xieta_mm.shape == (2, schema.OUTLINE_POINTS, 2)


@pytest.mark.parametrize("angle_deg", [0, 30, 77.7, 123, -30])
def test_the_world_axis_angle_of_a_rotated_ellipse_is_right_with_alpha_25(angle_deg):
    a = np.radians(angle_deg)
    track = h.derive([h.ellipse_record(0, CENTER, 40.0, 15.0, a)], TILTED)
    assert h.axis_apart_deg(track.theta_rad[0], -(a + ALPHA)) < 1.0
    assert track.major_mm[0] == pytest.approx(80.0 * K, rel=0.02)
    assert track.minor_mm[0] == pytest.approx(30.0 * K, rel=0.02)
    assert track.eccentricity[0] == pytest.approx(np.sqrt(1 - (15 / 40) ** 2), abs=0.01)


# --------------------------------------------------------------------------- head continuity


@pytest.mark.parametrize("world", [UPRIGHT, TILTED], ids=["alpha0", "alpha25"])
@pytest.mark.parametrize("turn", [1, -1])
@pytest.mark.parametrize("clicked", [1, -1], ids=["head", "tail"])
def test_an_ellipse_rotating_through_2_pi_never_flips_its_head(world, turn, clicked):
    # 73 frames, 5 degrees apart in the image, drifting; the click on frame 0 names one tip the head.
    steps = np.radians(5.0) * np.arange(73)
    start = np.radians(12.0)
    centers = [(CENTER[0] + 0.31 * i, CENTER[1] - 0.17 * i) for i in range(73)]
    records = [h.ellipse_record(i, centers[i], 40.0, 15.0, start + turn * steps[i]) for i in range(73)]
    track = h.derive(records, world, head_px=h.along(centers[0], start, clicked * 30.0))
    first = h.world_angle(world, start if clicked == 1 else start + np.pi)   # in (-pi, pi]
    truth = first - turn * steps                                             # the image's turn, mirrored
    assert np.degrees(np.abs(track.theta_rad - truth)).max() < 2.0           # unwrapped: it leaves (-pi, pi]
    assert np.degrees(np.abs(np.diff(track.theta_rad))).max() < 7.0          # 5 degrees a frame, no flip
    assert np.allclose(track.heading, np.column_stack([np.cos(track.theta_rad), np.sin(track.theta_rad)]))
    assert not track.orient.any() and not track.headguess.any()


@pytest.mark.parametrize("angle_deg", [20, 137, 291])
def test_the_body_with_beating_rods_keeps_its_heading_within_5_degrees(angle_deg):
    # The 47 x 20 px body with two 3 px rods that swing 40 degrees to each side of "sideways",
    # two beats in 48 frames; the center is off the pixel grid so the thin rods rasterize whole.
    a = np.radians(angle_deg)
    rod_angles = np.radians(90.0 + 40.0 * np.sin(2 * np.pi * np.arange(48) / 24))
    records = [h.record(lambda u, v, rod=rod: shapes.body_with_rods(u, v, CENTER, a, rod_angle=rod)[0], i, CENTER,
                        half=90) for i, rod in enumerate(rod_angles)]
    track = h.derive(records, TILTED, head_px=h.along(CENTER, a, 15.0))
    assert h.apart_deg(track.theta_rad, h.world_angle(TILTED, a)).max() < 5.0
    assert np.degrees(np.abs(np.diff(track.theta_rad))).max() < 10.0
    assert not track.orient.any()


@pytest.mark.parametrize("forward", [1, -1])
def test_without_a_head_click_the_head_follows_the_motion_of_the_first_10_frames(forward):
    # 0.5 px a frame along the long axis for 10 frames (4.5 px from the 1st to the 10th), then
    # 3 px a frame the other way: only the first 10 tracked frames count.
    a = np.radians(33.0)
    travelled = [0.5 * i if i < 10 else 4.5 - 3.0 * (i - 9) for i in range(20)]
    records = [h.ellipse_record(i, h.along(CENTER, a, forward * s), 40.0, 15.0, a) for i, s in enumerate(travelled)]
    track = h.derive(records, TILTED)
    assert track.headguess.all()                       # every frame inherits the guess
    assert h.apart_deg(track.theta_rad, h.world_angle(TILTED, a if forward == 1 else a + np.pi)).max() < 1.0
    assert not h.derive(records, TILTED, head_px=h.along(CENTER, a, 30.0)).headguess.any()


def test_a_75_degree_turn_during_a_round_stretch_flags_the_stretch_and_one_frame():
    # 15 frames at 10 degrees, 30 frames nearly round (lambda2 / lambda1 = (19 / 20)^2 = 0.90 > 0.8)
    # during which the body turns to 85 degrees, then 15 frames at 85 degrees.
    turning = np.radians(10.0 + 75.0 * np.arange(1, 31) / 30)
    records = ([h.ellipse_record(i, CENTER, 40.0, 15.0, np.radians(10.0)) for i in range(15)]
               + [h.ellipse_record(15 + i, CENTER, 20.0, 19.0, turning[i]) for i in range(30)]
               + [h.ellipse_record(45 + i, CENTER, 40.0, 15.0, np.radians(85.0)) for i in range(15)])
    track = h.derive(records, TILTED, head_px=h.along(CENTER, np.radians(10.0), 30.0))
    assert not track.orient[:15].any()
    assert track.orient[15:45].all()                   # the axis is undefined: never a reference
    assert track.orient[45:].sum() <= 1 and not track.orient[46:].any()   # at most one frame after it
    # Which one: frame 45 is 75 degrees from the reference, still frame 14, so it is a jump; it
    # becomes the next reference, so nothing latches.
    assert np.flatnonzero(track.orient).tolist() == list(range(15, 46))
    assert h.apart_deg(track.theta_rad[:15], h.world_angle(TILTED, np.radians(10.0))).max() < 1.0
    assert np.degrees(track.theta_rad[45:] - track.theta_rad[14]) == pytest.approx(-75.0, abs=2.0)   # not +105


# --------------------------------------------------------------------------- outline


def test_the_outline_of_a_circle_has_128_points_counterclockwise_from_the_head_point():
    radius, click = 50.0, (CENTER[0] + 20.0, CENTER[1] - 30.0)
    track = h.derive([h.disk_record(0, CENTER, radius)], TILTED, head_px=click)
    xy, xieta = track.outline_xy_mm[0], track.outline_xieta_mm[0]
    assert xy.shape == (128, 2)
    assert h.signed_area(xy) == pytest.approx(np.pi * (radius * K) ** 2, rel=0.002)   # > 0: counterclockwise
    assert h.chords(xy).max() - h.chords(xy).min() < 0.01 * h.chords(xy).mean()
    center = np.array(TILTED.to_world(*CENTER))
    assert np.hypot(*(xy - center).T) == pytest.approx(radius * K, abs=0.05 * K)
    assert track.perimeter_mm[0] == pytest.approx(2 * np.pi * radius * K, rel=0.005)
    # The head point is where the ray from the core centroid along the heading leaves the outline:
    # in the body frame it lies on the xi axis, a radius ahead.
    core = np.array([track.core_x_mm[0], track.core_y_mm[0]])
    assert xieta[0] == pytest.approx((radius * K, 0.0), abs=0.1 * K)
    assert xy[0] - core == pytest.approx(radius * K * track.heading[0], abs=0.1 * K)
    assert (xy[0] - core) @ (np.array(TILTED.to_world(*click)) - core) > 0    # on the clicked side


@pytest.mark.parametrize("clicked", [1, -1], ids=["head", "tail"])
def test_the_outline_of_an_ellipse_starts_at_the_tip_on_the_clicked_side(clicked):
    a = np.radians(40.0)
    track = h.derive([h.ellipse_record(0, CENTER, 40.0, 15.0, a)], TILTED, head_px=h.along(CENTER, a, clicked * 25.0))
    tip = TILTED.to_world(*h.along(CENTER, a, clicked * 40.0))
    assert track.outline_xy_mm[0, 0] == pytest.approx(tip, abs=0.5 * K)
    assert h.signed_area(track.outline_xy_mm[0]) > 0
    # the body frame: xi along the long axis toward the head, eta across; the same sense of rotation
    xi, eta = track.outline_xieta_mm[0].T
    assert (xi / (40.0 * K)) ** 2 + (eta / (15.0 * K)) ** 2 == pytest.approx(1.0, abs=0.04)
    assert h.signed_area(track.outline_xieta_mm[0]) == pytest.approx(h.signed_area(track.outline_xy_mm[0]), rel=1e-9)
    assert (xi[0], eta[0]) == pytest.approx((40.0 * K, 0.0), abs=0.5 * K)
    assert xi.max() == pytest.approx(xi[0], abs=0.02 * K) and xi.min() == pytest.approx(-40.0 * K, abs=0.5 * K)


# --------------------------------------------------------------------------- radial profile


def test_the_radial_profile_of_a_circle_is_its_radius():
    track = h.derive([h.disk_record(0, CENTER, 50.0)], TILTED)
    assert track.radial_mm.shape == (1, 72)
    assert track.radial_mm[0] == pytest.approx(50.0 * K, abs=0.5 * K)


@pytest.mark.parametrize("angle_deg", [0, 30, 77.7, 123])
def test_the_radial_profile_of_an_ellipse_is_its_polar_equation(angle_deg):
    # The stored outline of the a = 40, b = 15 px ellipse, with the analytic center and heading.
    a = np.radians(angle_deg)
    one = h.ellipse_record(0, CENTER, 40.0, 15.0, a)
    polygon = np.column_stack(TILTED.to_world(one.outline_px[:, 0], one.outline_px[:, 1]))
    heading = h.world_vector(TILTED, shapes.direction(a)) / K
    radii = derive.radial_profile(polygon, TILTED.to_world(*CENTER), heading)
    phi = np.radians(np.arange(0, 360, 5))
    expected = K * 40.0 * 15.0 / np.sqrt((15.0 * np.cos(phi)) ** 2 + (40.0 * np.sin(phi)) ** 2)
    assert radii.shape == (72,)
    assert np.abs(radii / expected - 1.0).max() < 0.01


def test_the_radial_angle_runs_counterclockwise_from_the_heading():
    # A rectangle from (-1, -1) to (3, 2) seen from the origin, y up.
    rectangle = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 2.0], [-1.0, 2.0]])
    radii = derive.radial_profile(rectangle, (0.0, 0.0), (1.0, 0.0))
    at = {deg: radii[deg // 5] for deg in (0, 45, 90, 180, 270, 315)}
    assert at == pytest.approx({0: 3.0, 45: 2 * np.sqrt(2), 90: 2.0, 180: 1.0, 270: 1.0, 315: np.sqrt(2)})
    turned = derive.radial_profile(rectangle[::-1], (0.0, 0.0), (0.0, 1.0))   # heading along +y; clockwise polygon
    assert (turned[0], turned[18], turned[36], turned[54]) == pytest.approx((2.0, 1.0, 1.0, 3.0))


def test_the_radius_is_the_farthest_crossing_and_a_ray_that_misses_gives_nan():
    # An L with arms 1 wide: from (0.5, 3) toward (1, -1) the ray leaves the upright arm, crosses
    # the lying arm from (2.5, 1) to (3.5, 0) and leaves for good after 3 sqrt(2).
    letter = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [1.0, 1.0], [1.0, 4.0], [0.0, 4.0]])
    assert derive.radial_profile(letter, (0.5, 3.0), (1.0, -1.0))[0] == pytest.approx(3 * np.sqrt(2))
    # A square of side 2 seen from 5 away: only the rays within atan(1 / 4) = 14 degrees meet it.
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    radii = derive.radial_profile(square, (5.0, 0.0), (-1.0, 0.0))
    assert radii[[0, 1, 2, 70, 71]] == pytest.approx([6.0, 6 / np.cos(np.radians(5)), 1 / np.sin(np.radians(10)),
                                                      1 / np.sin(np.radians(10)), 6 / np.cos(np.radians(5))])
    assert np.isnan(radii[3:70]).all()


# --------------------------------------------------------------------------- solidity, circularity, Feret


def test_a_disk_is_solid_and_circular_and_its_feret_diameter_is_its_diameter():
    track = h.derive([h.disk_record(0, CENTER, 50.0)], TILTED)
    assert 0.99 <= track.solidity[0] <= 1.0 + 1e-9
    assert track.circularity[0] == pytest.approx(1.0, abs=0.005)
    assert track.feret_max_mm[0] == pytest.approx(100.0 * K, rel=0.005)


@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.37, -0.19)])
@pytest.mark.parametrize("angle_deg", [0, 17, 45, 73, 211])
def test_the_solidity_of_an_l_shape_is_its_area_over_its_hull(angle_deg, offset):
    # Arms 120 px long and 60 px wide: area 120^2 - 60^2 = 10800 px^2; the hull closes the notch
    # with one edge and adds half the missing square: 12600 px^2. Its longest diagonal joins the
    # two arm tips: 120 sqrt(2) px.
    a = np.radians(angle_deg)
    corner = (CENTER[0] + offset[0], CENTER[1] + offset[1])
    middle = h.along(h.along(corner, a, 60.0), a + np.pi / 2, 60.0)
    one = h.record(lambda u, v: shapes.l_shape(u, v, corner, 120.0, 60.0, a), 0, middle, half=100)
    track = h.derive([one], TILTED)
    assert track.solidity[0] == pytest.approx(10800 / 12600, rel=0.01)
    assert track.feret_max_mm[0] == pytest.approx(120.0 * np.sqrt(2) * K, rel=0.01)
    assert track.area_mm2[0] == pytest.approx(K ** 2 * one.area_px, rel=1e-12)   # the pixel count


# --------------------------------------------------------------------------- resolution


def test_shape_ok_needs_camera_pixels_and_grid_cells():
    def resolution(a, b, input_box, mode, **settings):
        one = h.ellipse_record(0, CENTER, a, b, np.radians(30.0), input_box=input_box, mode=mode)
        track = h.derive([one], TILTED, **settings)
        return track.px_along_major[0], track.cells_along_major[0], track.shape_ok[0]

    # A plain 14 x 6 px ellipse in a 96 px fine window: a cell is 96 / 256 px, so 37 cells but 14 px.
    px, cells, ok = resolution(7.0, 3.0, (652, 352, 96, 96), "fine")
    assert px == pytest.approx(14.0, abs=1.0) and cells == pytest.approx(px * 256 / 96, rel=1e-12)
    assert ok == 0
    assert resolution(7.0, 3.0, (652, 352, 96, 96), "fine", shape_ok_min=10)[2] == 1   # the limit is a setting
    # 80 x 30 px on the whole frame: 80 px, but a cell is 1920 / 256 = 7.5 px: 10.7 cells.
    px, cells, ok = resolution(40.0, 15.0, h.FULL_HD, "coarse")
    assert px == pytest.approx(80.0, rel=0.02) and cells == pytest.approx(px / 7.5, rel=1e-12)
    assert ok == 0
    # The same ellipse in a 240 px fine window: 80 px and 85 cells.
    assert resolution(40.0, 15.0, (580, 280, 240, 240), "fine")[2] == 1


def test_stray_pixels_do_not_lengthen_the_size_check():
    # A body of 6 rows x 15 columns of pixels, seen with grid cells of 1002 / 256 px; on the second
    # row of the track the mask also holds 2 pixels 240 columns away (h.stray_pixel_records). The
    # size check is of the largest piece, the body: n pixel centers in a row have the variance
    # (n^2 - 1) / 12, so L1 = 4 sqrt((15^2 - 1) / 12) = 17.28 px, which is 4.4 cells: too small on
    # every row. Over all 92 pixels the second row would read 146 px and 37 cells.
    track = h.derive(h.stray_pixel_records(), TILTED)
    body_px = 4.0 * np.sqrt((15 ** 2 - 1) / 12.0)
    assert track.n_components.tolist() == [1, 2, 1]
    assert track.px_along_major == pytest.approx([body_px] * 3, rel=1e-9)
    assert track.cells_along_major == pytest.approx([body_px / (1002 / 256)] * 3, rel=1e-9)
    assert track.shape_ok.tolist() == [0, 0, 0]
    # Everything else on the second row is still of the whole mask. Its 92 pixel centers, in px
    # from the body's top-left corner: the body's 90 and the two stray ones.
    u = np.concatenate([np.tile(np.arange(15) + 0.5, 6), [255.5, 256.5]])
    v = np.concatenate([np.repeat(np.arange(6) + 0.5, 15), [2.5, 2.5]])
    assert (len(u), u.sum(), v.sum()) == (92, 1187.0, 275.0)
    assert track.u_px[1] == pytest.approx(h.STRAY_AT[0] + 1187 / 92, abs=1e-9)
    assert track.v_px[1] == pytest.approx(h.STRAY_AT[1] + 275 / 92, abs=1e-9)
    du, dv = u - 1187 / 92, v - 275 / 92
    uu, uv, vv = np.mean(du * du), np.mean(du * dv), np.mean(dv * dv)
    lambda1 = (uu + vv) / 2 + np.hypot((uu - vv) / 2, uv)
    lambda2 = (uu + vv) / 2 - np.hypot((uu - vv) / 2, uv)
    assert 4.0 * np.sqrt(lambda1) == pytest.approx(145.96, abs=0.01)
    assert track.major_mm[1] / K == pytest.approx(4.0 * np.sqrt(lambda1), rel=1e-9)
    assert track.minor_mm[1] / K == pytest.approx(4.0 * np.sqrt(lambda2), rel=1e-9)
    assert track.eccentricity[1] == pytest.approx(np.sqrt(1.0 - lambda2 / lambda1), rel=1e-9)
    assert track.area_mm2[1] == pytest.approx(K ** 2 * 92, rel=1e-12)


def test_a_far_piece_of_two_pixels_does_not_lengthen_an_ellipse():
    # The a = 40, b = 15 px ellipse at 30 degrees, alone and with a piece of 2 px that lies 300 px to
    # its right. The second moment of a uniform ellipse along its major axis is a^2 / 4, so L1 = 2a
    # = 80 px, with the piece as without it (2% as for the moments above). Over all pixels the row
    # with the piece would read 87 px.
    a = np.radians(30.0)
    column, row = int(CENTER[0]) + 300, int(CENTER[1])   # the first of the piece's two pixels

    def with_piece(u, v):
        piece = shapes.box(u, v, (column + 1.0, row + 0.5), 1.0, 0.5, 0.0)   # two pixel centers lie inside
        return shapes.union(shapes.ellipse(u, v, CENTER, 40.0, 15.0, a), piece)

    records = [h.ellipse_record(0, CENTER, 40.0, 15.0, a), h.record(with_piece, 1, CENTER, half=320)]
    assert records[1].area_px == records[0].area_px + 2
    track = h.derive(records, TILTED)
    assert track.n_components.tolist() == [1, 2]
    assert track.px_along_major[1] == pytest.approx(80.0, rel=0.02)
    assert track.px_along_major[0] == track.major_mm[0] / K   # one piece: the full mask's value, to the bit


# --------------------------------------------------------------------------- dish wall


@pytest.mark.parametrize("distance", [0.0, 150.0, 395.0, 430.0])
def test_wall_distances_of_a_disk_at_a_known_distance_from_the_dish_center(distance):
    # Dish of radius 400 px; a disk of radius 12 px with its center `distance` px from the dish's.
    dish = Circle(center_px=[900.2, 520.7], radius_px=400.0)
    center = h.along(dish.center_px, np.radians(200.0), distance)
    one = h.disk_record(0, center, 12.0, half=30)
    track = h.derive([one], TILTED, circle=dish)
    assert track.wall_dist_centroid_mm[0] == pytest.approx((400.0 - distance) * K, abs=0.1 * K)
    assert track.wall_dist_min_mm[0] == pytest.approx((400.0 - distance - 12.0) * K, abs=0.1 * K)
    assert (track.wall_dist_centroid_mm[0] < 0) == (distance > 400.0)       # negative outside the circle
    assert (track.wall_dist_min_mm[0] < 0) == (distance > 388.0)
    without = h.derive([one], TILTED)
    assert np.isnan(without.wall_dist_centroid_mm[0]) and np.isnan(without.wall_dist_min_mm[0])
