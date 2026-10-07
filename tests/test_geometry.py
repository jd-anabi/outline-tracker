"""Tests of outline_tracker/geometry.py: the image <-> world transform and the calibration tools.

Expected values come from SPEC 3.2, 3.4 and 4.1-4.5 worked by hand, from the template's
`tracker_map`, from brute force over the definition, and from synthetic points whose truth is
known. Nothing is copied from the output of the code under test.

Coordinates: (u, v) are image pixels in Tracker's convention (origin at the top-left corner of the
frame, u to the right, v down, pixel centers at +0.5); (x, y) are mm in the user's axes, y up.
"""

import ast
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from test_tracker_io import MM_PER_PX, tracker_map  # the template's map, verbatim (0.05 mm per px)

from outline_tracker import geometry, tracker_io
from outline_tracker.geometry import WorldFrame
from outline_tracker.tracker_io import Calibration, fit_calibration

COS30, SIN30 = math.cos(math.radians(30.0)), 0.5


# --------------------------------------------------------------------------- the transform (SPEC 3.2)

@pytest.mark.parametrize("alpha_deg", [0.0, 90.0, -30.0])
def test_round_trip_px_world_px(alpha_deg):
    wf = WorldFrame(k_mm_per_px=0.0324, alpha_rad=math.radians(alpha_deg), u0=960.5, v0=540.5)
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, 1920, 50), rng.uniform(0, 1080, 50)
    u2, v2 = wf.to_px(*wf.to_world(u, v))
    np.testing.assert_allclose(u2, u, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v2, v, rtol=0, atol=1e-9)
    x, y = rng.uniform(-30, 30, 50), rng.uniform(-20, 20, 50)
    x2, y2 = wf.to_world(*wf.to_px(x, y))
    np.testing.assert_allclose(x2, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y2, y, rtol=0, atol=1e-9)
    assert wf.to_px(*wf.to_world(17.25, 203.5)) == pytest.approx((17.25, 203.5), abs=1e-9)


def test_alpha_zero_y_is_up():
    wf = WorldFrame(k_mm_per_px=0.05, alpha_rad=0.0, u0=100.0, v0=80.0)
    assert wf.to_world(100.0, 80.0) == pytest.approx((0.0, 0.0), abs=1e-12)
    # 10 px to the right of the origin and 20 px above it on screen
    assert wf.to_world(110.0, 60.0) == pytest.approx((0.5, 1.0), abs=1e-12)
    # a larger v is lower on screen: y gets smaller, x does not change
    x_hi, y_hi = wf.to_world(110.0, 60.0)
    x_lo, y_lo = wf.to_world(110.0, 61.0)
    assert y_lo < y_hi
    assert x_lo == pytest.approx(x_hi, abs=1e-12)
    assert wf.to_px(0.5, 1.0) == pytest.approx((110.0, 60.0), abs=1e-9)


def test_alpha_90_puts_x_up_and_y_left_on_screen():
    # alpha is the direction of +x, counterclockwise on screen from rightward: 90 deg is straight up
    wf = WorldFrame(k_mm_per_px=0.05, alpha_rad=math.radians(90.0), u0=100.0, v0=80.0)
    assert wf.to_world(100.0, 60.0) == pytest.approx((1.0, 0.0), abs=1e-12)  # 20 px above the origin
    assert wf.to_world(90.0, 80.0) == pytest.approx((0.0, 0.5), abs=1e-12)  # 10 px to its left
    assert wf.to_px(1.0, 0.0) == pytest.approx((100.0, 60.0), abs=1e-9)


def test_alpha_minus_30_puts_x_right_and_down_on_screen():
    # -30 deg is 30 deg clockwise on screen: along +x both u and v grow
    wf = WorldFrame(k_mm_per_px=0.05, alpha_rad=math.radians(-30.0), u0=100.0, v0=80.0)
    d = 40.0  # px, so 2 mm
    assert wf.to_world(100.0 + d * COS30, 80.0 + d * SIN30) == pytest.approx((2.0, 0.0), abs=1e-12)
    # +y is 90 deg counterclockwise on screen from +x: to the right and up
    assert wf.to_world(100.0 + d * SIN30, 80.0 - d * COS30) == pytest.approx((0.0, 2.0), abs=1e-12)
    assert wf.to_px(2.0, 0.0) == pytest.approx((100.0 + d * COS30, 80.0 + d * SIN30), abs=1e-9)


@pytest.mark.parametrize("alpha_deg", [0.0, 90.0, -30.0, 137.0])
def test_J_is_the_matrix_of_the_spec(alpha_deg):
    a = math.radians(alpha_deg)
    wf = WorldFrame(k_mm_per_px=0.05, alpha_rad=a, u0=3.0, v0=4.0)
    expected = np.array([[math.cos(a), -math.sin(a)], [-math.sin(a), -math.cos(a)]])
    assert wf.J.shape == (2, 2)
    np.testing.assert_allclose(wf.J, expected, rtol=0, atol=1e-15)
    assert np.linalg.det(wf.J) == pytest.approx(-1.0, abs=1e-12)
    np.testing.assert_allclose(wf.J @ wf.J, np.eye(2), rtol=0, atol=1e-12)  # a reflection
    p = np.array([17.25, 203.5])
    np.testing.assert_allclose(wf.to_world(*p), 0.05 * expected @ (p - np.array([3.0, 4.0])),
                               rtol=0, atol=1e-12)


def test_transform_keeps_array_shapes_and_nan():
    wf = WorldFrame(k_mm_per_px=0.05, alpha_rad=0.3, u0=100.0, v0=80.0)
    u = np.arange(12, dtype=float).reshape(3, 4)
    v = np.full((3, 4), 7.5)
    v[1, 2] = np.nan  # a lost frame
    x, y = wf.to_world(u, v)
    assert x.shape == y.shape == (3, 4)
    assert np.isnan(x[1, 2]) and np.isnan(y[1, 2])
    assert np.isfinite(np.delete(x.ravel(), 6)).all()
    u2, v2 = wf.to_px(x, y)
    assert u2.shape == v2.shape == (3, 4)
    np.testing.assert_allclose(u2[0], u[0], rtol=0, atol=1e-9)


@pytest.mark.parametrize("k", [0.0, -0.05, math.nan, math.inf])
def test_world_frame_needs_a_positive_scale(k):
    with pytest.raises(ValueError, match="scale"):
        WorldFrame(k_mm_per_px=k, alpha_rad=0.0, u0=0.0, v0=0.0)


def test_world_frame_needs_finite_origin_and_angle():
    with pytest.raises(ValueError):
        WorldFrame(k_mm_per_px=0.05, alpha_rad=math.nan, u0=0.0, v0=0.0)
    with pytest.raises(ValueError):
        WorldFrame(k_mm_per_px=0.05, alpha_rad=0.0, u0=math.inf, v0=0.0)


# --------------------------------------------------------------------------- covariances (SPEC 7.3)

def test_cov_to_world_by_hand_for_alpha_zero():
    # x = k (u - u0), y = -k (v - v0): variances scale by k^2 and the cross term changes sign
    wf = WorldFrame(k_mm_per_px=0.1, alpha_rad=0.0, u0=5.0, v0=6.0)
    np.testing.assert_allclose(wf.cov_to_world([4.0, 1.0, 9.0]), [0.04, -0.01, 0.09], rtol=0, atol=1e-15)


@pytest.mark.parametrize("alpha_deg", [0.0, 90.0, -30.0, 25.0])
def test_cov_to_world_is_the_covariance_of_the_mapped_points(alpha_deg):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(400, 2)) @ np.array([[9.0, 0.0], [4.0, 3.0]]) + np.array([300.5, 200.5])
    wf = WorldFrame(k_mm_per_px=0.0324, alpha_rad=math.radians(alpha_deg), u0=960.5, v0=540.5)
    cov_px = np.cov(pts.T, bias=True)  # px^2: [[uu, uv], [uv, vv]]
    x, y = wf.to_world(pts[:, 0], pts[:, 1])
    cov_mm = np.cov(np.vstack([x, y]), bias=True)  # mm^2: [[xx, xy], [xy, yy]]
    packed_px = np.array([cov_px[0, 0], cov_px[0, 1], cov_px[1, 1]])
    packed_mm = np.array([cov_mm[0, 0], cov_mm[0, 1], cov_mm[1, 1]])
    # the stored form of results.npz (cov_full, cov_core): [..., 3] = uu, uv, vv
    out = wf.cov_to_world(packed_px)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, packed_mm, rtol=1e-10, atol=1e-15)
    # the matrix form: [..., 2, 2]
    out = wf.cov_to_world(cov_px)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, cov_mm, rtol=1e-10, atol=1e-15)


def test_cov_to_world_per_frame_rows_keep_nan():
    wf = WorldFrame(k_mm_per_px=0.1, alpha_rad=0.0, u0=5.0, v0=6.0)
    rows = np.array([[4.0, 1.0, 9.0], [np.nan, np.nan, np.nan], [1.0, 0.0, 1.0]])
    out = wf.cov_to_world(rows)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], [0.04, -0.01, 0.09], rtol=0, atol=1e-15)
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2], [0.01, 0.0, 0.01], rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad", [[1.0, 2.0], np.zeros((4, 4)), 3.0])
def test_cov_to_world_refuses_other_shapes(bad):
    wf = WorldFrame(k_mm_per_px=0.1, alpha_rad=0.0, u0=5.0, v0=6.0)
    with pytest.raises(ValueError, match="covariance"):
        wf.cov_to_world(bad)


# --------------------------------------------------------------------------- Calibration <-> WorldFrame

def random_frames(n=100, seed=3):
    """n random (k mm/px, alpha rad, u0 px, v0 px): scales from 5 to 500 um/px, any angle, origins in
    and around a 4K frame."""
    rng = np.random.default_rng(seed)
    k = np.exp(rng.uniform(np.log(0.005), np.log(0.5), n))
    alpha = rng.uniform(-np.pi, np.pi, n)
    u0, v0 = rng.uniform(-500, 4500, n), rng.uniform(-500, 2500, n)
    return [tuple(map(float, row)) for row in zip(k, alpha, u0, v0)]


def spec_calibration(k, alpha, u0, v0):
    """The flip form of SPEC 3.2, typed from the specification: a = k cos(alpha), c = -k sin(alpha),
    tx = -(a u0 + c v0), ty = -(c u0 - a v0)."""
    a, c = k * math.cos(alpha), -k * math.sin(alpha)
    return Calibration(a=a, c=c, tx=-(a * u0 + c * v0), ty=-(c * u0 - a * v0), flip=True, rms_mm=0.0)


def frame_points(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 3840, 20), rng.uniform(0, 2160, 20)


def test_equals_calibration_flip_form_for_random_frames():
    frames = random_frames()
    assert len(frames) == 100
    for i, (k, alpha, u0, v0) in enumerate(frames):
        u, v = frame_points(i)
        x, y = WorldFrame(k, alpha, u0, v0).to_world(u, v)
        ex, ey = spec_calibration(k, alpha, u0, v0).to_mm(u, v)
        np.testing.assert_allclose(x, ex, rtol=0, atol=1e-9)
        np.testing.assert_allclose(y, ey, rtol=0, atol=1e-9)


def test_equals_trackers_map_for_random_frames():
    assert MM_PER_PX == 0.05
    frames = random_frames()
    assert len(frames) == 100
    for i, (k, alpha, u0, v0) in enumerate(frames):
        u, v = frame_points(i)
        x, y = WorldFrame(k, alpha, u0, v0).to_world(u, v)
        tx, ty = tracker_map(u, v, angle_deg=math.degrees(alpha), origin=(u0, v0))
        np.testing.assert_allclose(x, tx * (k / MM_PER_PX), rtol=0, atol=1e-9)
        np.testing.assert_allclose(y, ty * (k / MM_PER_PX), rtol=0, atol=1e-9)


def test_calibration_gives_scale_angle_and_origin():
    # SPEC 3.2: k = hypot(a, c), alpha = atan2(-c, a), (u0, v0) = to_px(0, 0)
    for k, alpha, u0, v0 in random_frames():
        wf = geometry.world_frame_from_calibration(spec_calibration(k, alpha, u0, v0))
        assert isinstance(wf, WorldFrame)
        assert wf.k_mm_per_px == pytest.approx(k, rel=1e-12)
        assert wf.alpha_rad == pytest.approx(alpha, abs=1e-12)
        assert (wf.u0, wf.v0) == pytest.approx((u0, v0), abs=1e-8)


def test_calibration_to_world_frame_and_back_is_the_identity():
    for k, alpha, u0, v0 in random_frames():
        cal = spec_calibration(k, alpha, u0, v0)
        back = geometry.calibration_from_world_frame(geometry.world_frame_from_calibration(cal))
        assert isinstance(back, Calibration)
        assert back.flip is True
        assert back.a == pytest.approx(cal.a, rel=1e-12, abs=1e-15)
        assert back.c == pytest.approx(cal.c, rel=1e-12, abs=1e-15)
        assert back.tx == pytest.approx(cal.tx, abs=1e-9)
        assert back.ty == pytest.approx(cal.ty, abs=1e-9)
        assert back.rms_mm == 0.0
        u, v = frame_points(7)
        np.testing.assert_allclose(back.to_mm(u, v), cal.to_mm(u, v), rtol=0, atol=1e-9)


def test_world_frame_to_calibration_uses_the_spec_formulas():
    k, alpha, u0, v0 = 0.0324, math.radians(12.0), 960.5, 540.5
    cal = geometry.calibration_from_world_frame(WorldFrame(k, alpha, u0, v0))
    expected = spec_calibration(k, alpha, u0, v0)
    assert cal.flip is True
    assert (cal.a, cal.c, cal.tx, cal.ty) == pytest.approx(
        (expected.a, expected.c, expected.tx, expected.ty), rel=1e-12, abs=1e-12)
    assert cal.mm_per_px == pytest.approx(k, rel=1e-12)


def test_calibration_from_world_frame_can_carry_the_fit_residual():
    cal = geometry.calibration_from_world_frame(WorldFrame(0.05, 0.0, 160.0, 120.0), rms_mm=0.002)
    assert cal.rms_mm == 0.002


def test_world_frame_from_a_fitted_calibration():
    # the from-tracker path: an export gives pixel and mm columns, fit_calibration gives the map
    rng = np.random.default_rng(4)
    px, py = rng.uniform(0, 1920, 30), rng.uniform(0, 1080, 30)
    k, angle_deg, origin = 0.0324, 12.0, (700.5, 400.5)
    x, y = (c * (k / MM_PER_PX) for c in tracker_map(px, py, angle_deg, origin))
    cal = fit_calibration(px, py, x, y)
    assert cal.flip
    wf = geometry.world_frame_from_calibration(cal)
    assert wf.k_mm_per_px == pytest.approx(k, rel=1e-9)
    assert math.degrees(wf.alpha_rad) == pytest.approx(angle_deg, abs=1e-7)
    assert (wf.u0, wf.v0) == pytest.approx(origin, abs=1e-6)
    np.testing.assert_allclose(wf.to_world(px, py), (x, y), rtol=0, atol=1e-9)


def mirrored_fit(px, py):
    """A fit to points whose y points down on screen, like image rows: x = k (px - 160),
    y = +k (py - 120). No flip, so not a map Tracker can produce."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    return fit_calibration(px, py, MM_PER_PX * (px - 160.0), MM_PER_PX * (py - 120.0))


@pytest.mark.parametrize("px, py", [
    ([10.5, 200.5, 90.5], [20.5, 30.5, 180.5]),  # three points, not on one line
    ([10.5, 200.5, 90.5, 300.5, 150.5, 40.5], [20.5, 30.5, 180.5, 220.5, 90.5, 130.5]),
])
def test_mirrored_fit_is_refused(px, py):
    cal = mirrored_fit(px, py)
    assert not cal.flip  # the premise: the data really are mirrored
    with pytest.raises(geometry.MirroredCalibrationError, match="mirrored"):
        geometry.world_frame_from_calibration(cal)


def test_mirrored_error_is_a_value_error():
    assert issubclass(geometry.MirroredCalibrationError, ValueError)


def test_calibration_with_zero_scale_is_refused():
    with pytest.raises(ValueError, match="scale"):
        geometry.world_frame_from_calibration(Calibration(a=0.0, c=0.0, tx=1.0, ty=2.0, flip=True, rms_mm=0.0))


# --------------------------------------------------------------------------- stick (SPEC 4.2)

def test_stick_926_px_for_30_mm():
    s = geometry.stick_scale((100.5, 400.5), (1026.5, 400.5), 30.0)
    assert s.length_px == pytest.approx(926.0, rel=1e-12)
    assert s.k_mm_per_px == pytest.approx(30.0 / 926.0, rel=1e-12)
    assert f"{1000 * s.k_mm_per_px:.2f}" == "32.40"  # um per px, as SPEC 4.2 displays it
    assert s.rel_uncertainty == pytest.approx(math.sqrt(2) * 0.5 / 926, rel=1e-12)
    assert f"{100 * s.rel_uncertainty:.2f}" == "0.08"  # percent
    assert s.too_short is False


def test_stick_at_an_angle_and_in_either_order():
    p1, p2 = (10.5, 20.5), (610.5, 820.5)  # 600 and 800 px apart: 1000 px
    s = geometry.stick_scale(p1, p2, 50.0, sigma_click_px=1.0)
    assert s.length_px == pytest.approx(1000.0, rel=1e-12)
    assert s.k_mm_per_px == pytest.approx(0.05, rel=1e-12)
    assert s.rel_uncertainty == pytest.approx(math.sqrt(2) / 1000, rel=1e-12)
    assert geometry.stick_scale(p2, p1, 50.0, sigma_click_px=1.0) == s


def test_stick_shorter_than_300_px_is_too_short():
    assert geometry.stick_scale((0.0, 0.0), (299.0, 0.0), 10.0).too_short is True
    assert geometry.stick_scale((0.0, 0.0), (299.9, 0.0), 10.0).too_short is True
    assert geometry.stick_scale((0.0, 0.0), (180.0, 240.0), 10.0).too_short is False  # exactly 300 px
    assert geometry.stick_scale((0.0, 0.0), (301.0, 0.0), 10.0).too_short is False


@pytest.mark.parametrize("length_mm", [0.0, -30.0, math.nan, math.inf])
def test_stick_length_must_be_positive(length_mm):
    with pytest.raises(ValueError, match="length"):
        geometry.stick_scale((0.0, 0.0), (926.0, 0.0), length_mm)


def test_stick_endpoints_must_differ():
    with pytest.raises(ValueError, match="same point"):
        geometry.stick_scale((412.5, 300.5), (412.5, 300.5), 30.0)


def test_stick_endpoints_must_be_numbers():
    with pytest.raises(ValueError):
        geometry.stick_scale((math.nan, 0.0), (926.0, 0.0), 30.0)
    with pytest.raises(ValueError):
        geometry.stick_scale((0.0, 0.0, 0.0), (926.0, 0.0), 30.0)


def test_click_sigma_must_not_be_negative():
    with pytest.raises(ValueError, match="sigma"):
        geometry.stick_scale((0.0, 0.0), (926.0, 0.0), 30.0, sigma_click_px=-0.5)


# --------------------------------------------------------------------------- tape check (SPEC 4.3)

def test_tape_20_mm_measured_as_20_1_passes():
    t = geometry.tape_check((100.0, 50.0), (502.0, 50.0), 20.0, 0.05)  # 402 px at 0.05 mm/px
    assert t.measured_mm == pytest.approx(20.1, rel=1e-12)
    assert t.rel_error == pytest.approx(0.005, rel=1e-9)  # +0.5 %
    assert t.ok is True


def test_tape_20_mm_measured_as_20_3_fails():
    t = geometry.tape_check((100.0, 50.0), (506.0, 50.0), 20.0, 0.05)  # 406 px
    assert t.measured_mm == pytest.approx(20.3, rel=1e-12)
    assert t.rel_error == pytest.approx(0.015, rel=1e-9)
    assert t.ok is False


@pytest.mark.parametrize("length_px, ok", [(403.96, True), (404.04, False), (396.04, True), (395.96, False)])
def test_tape_passes_within_one_percent_either_way(length_px, ok):
    # 400 px is 20 mm: +0.99 %, +1.01 %, -0.99 %, -1.01 %
    t = geometry.tape_check((0.0, 0.0), (0.0, length_px), 20.0, 0.05)
    assert t.rel_error == pytest.approx(length_px / 400.0 - 1.0, rel=1e-9)
    assert t.ok is ok


def test_tape_at_an_angle():
    t = geometry.tape_check((10.5, 20.5), (250.5, 340.5), 20.0, 0.05)  # 240 and 320 px apart: 400 px
    assert t.measured_mm == pytest.approx(20.0, rel=1e-12)
    assert t.rel_error == pytest.approx(0.0, abs=1e-12)
    assert t.ok is True


@pytest.mark.parametrize("true_mm, k", [(0.0, 0.05), (-20.0, 0.05), (20.0, 0.0), (20.0, -0.05), (math.nan, 0.05),
                                        (20.0, math.nan)])
def test_tape_needs_a_positive_distance_and_scale(true_mm, k):
    with pytest.raises(ValueError):
        geometry.tape_check((0.0, 0.0), (400.0, 0.0), true_mm, k)


def test_tape_points_must_differ():
    with pytest.raises(ValueError, match="same point"):
        geometry.tape_check((7.5, 7.5), (7.5, 7.5), 20.0, 0.05)


# --------------------------------------------------------------------------- stopwatch (SPEC 4.1)

def test_fps_from_stopwatch():
    assert geometry.fps_from_stopwatch(100, 1.00, 2500, 11.00) == pytest.approx(240.0, rel=1e-12)
    assert geometry.fps_from_stopwatch(2500, 11.00, 100, 1.00) == pytest.approx(240.0, rel=1e-12)
    assert geometry.fps_from_stopwatch(0, 0.0, 2396, 10.0) == pytest.approx(239.6, rel=1e-12)


def test_equal_stopwatch_readings_raise():
    with pytest.raises(ValueError, match="readings"):
        geometry.fps_from_stopwatch(100, 5.00, 2500, 5.00)


@pytest.mark.parametrize("args", [
    (100, 1.0, 100, 11.0),  # the same frame twice: 0 frames per second
    (2500, 1.0, 100, 11.0),  # the frame number falls while the stopwatch runs on
    (100, math.nan, 2500, 11.0),
    (100, 1.0, math.inf, 11.0),
])
def test_stopwatch_nonsense_raises(args):
    with pytest.raises(ValueError):
        geometry.fps_from_stopwatch(*args)


# --------------------------------------------------------------------------- circle fit (SPEC 4.4)

def circle_points(center, radius, angles_deg):
    a = np.radians(np.asarray(angles_deg, float))
    return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])


def test_three_points_give_the_circle_through_them():
    fit = geometry.fit_circle(circle_points((640.25, 355.75), 412.5, [10.0, 140.0, 255.0]))
    assert fit.center_px == pytest.approx((640.25, 355.75), abs=1e-6)
    assert fit.radius_px == pytest.approx(412.5, abs=1e-6)
    assert fit.rms_px == pytest.approx(0.0, abs=1e-6)
    assert isinstance(fit.center_px, tuple) and len(fit.center_px) == 2
    assert all(type(v) is float for v in (*fit.center_px, fit.radius_px, fit.rms_px))


def test_three_points_by_hand():
    # the right angle at (0, 0) subtends the diameter from (2, 0) to (0, 2)
    fit = geometry.fit_circle([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    assert fit.center_px == pytest.approx((1.0, 1.0), abs=1e-6)
    assert fit.radius_px == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_three_points_on_a_short_arc_are_still_exact():
    # 10 degrees of a dish wall: nearly on one line, but a circle all the same
    fit = geometry.fit_circle(circle_points((960.5, 540.5), 500.0, [80.0, 85.0, 90.0]))
    assert fit.center_px == pytest.approx((960.5, 540.5), abs=1e-6)
    assert fit.radius_px == pytest.approx(500.0, abs=1e-6)


def test_many_exact_points():
    fit = geometry.fit_circle(circle_points((960.5, 540.5), 500.0, np.arange(0, 360, 30)))
    assert fit.center_px == pytest.approx((960.5, 540.5), abs=1e-6)
    assert fit.radius_px == pytest.approx(500.0, abs=1e-6)
    assert fit.rms_px == pytest.approx(0.0, abs=1e-6)


def test_six_noisy_points_within_one_pixel():
    center, radius = np.array([960.5, 540.5]), 500.0
    exact = circle_points(center, radius, 20.0 + 60.0 * np.arange(6))  # spread evenly around
    pts = exact + np.random.default_rng(0).normal(0.0, 0.5, size=(6, 2))  # click noise, fixed seed
    fit = geometry.fit_circle(pts)
    assert math.hypot(fit.center_px[0] - center[0], fit.center_px[1] - center[1]) < 1.0
    assert abs(fit.radius_px - radius) < 1.0
    # the fit minimizes the radial residuals, so its RMS cannot exceed the RMS about the true circle
    rms_true = float(np.sqrt(np.mean((np.hypot(*(pts - center).T) - radius) ** 2)))
    assert 0.0 < fit.rms_px <= rms_true + 1e-9


def test_fit_is_the_geometric_least_squares_circle():
    # Noisy points on a third of the wall. At the minimum of sum (|p - c| - R)^2 the residuals sum to
    # zero (derivative in R) and so do the residuals times the unit vectors from the center
    # (derivative in c). The algebraic (Kasa) circle does not satisfy this on a noisy arc.
    exact = circle_points((960.5, 540.5), 500.0, np.linspace(200.0, 320.0, 8))
    pts = exact + np.random.default_rng(5).normal(0.0, 0.5, size=(8, 2))
    fit = geometry.fit_circle(pts)
    d = pts - np.array(fit.center_px)
    dist = np.hypot(d[:, 0], d[:, 1])
    res = dist - fit.radius_px
    assert abs(res.sum()) < 1e-6
    assert np.abs((res[:, None] * d / dist[:, None]).sum(axis=0)).max() < 1e-6
    assert fit.rms_px == pytest.approx(float(np.sqrt(np.mean(res ** 2))), rel=1e-9)
    assert abs(fit.radius_px - 500.0) < 10.0  # sanity on a short arc; the tolerance is not a spec value


def test_circle_needs_three_points():
    with pytest.raises(ValueError, match="3 points"):
        geometry.fit_circle([(0.0, 0.0), (2.0, 0.0)])
    with pytest.raises(ValueError, match="3 points"):
        geometry.fit_circle([])


@pytest.mark.parametrize("pts", [
    [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
    [(10.5, 35.25), (50.5, 55.25), (90.5, 75.25), (130.5, 95.25), (200.5, 130.25)],  # v = 0.5 u + 30
    [(100.5, 20.5), (100.5, 300.5), (100.5, 700.5)],  # a vertical line
    [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)],  # one point clicked three times
    [(5.0, 5.0), (5.0, 5.0), (9.0, 7.0)],  # two different points only
])
def test_collinear_circle_points_raise(pts):
    with pytest.raises(ValueError, match="one line"):
        geometry.fit_circle(pts)


def test_circle_points_must_be_pairs_of_numbers():
    with pytest.raises(ValueError):
        geometry.fit_circle([(0.0, 0.0), (2.0, 0.0), (math.nan, 2.0)])
    with pytest.raises(ValueError):
        geometry.fit_circle([(0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 2.0, 1.0)])


# --------------------------------------------------------------------------- the frame grid (SPEC 3.4, X20)

def test_grid_frames():
    assert list(geometry.grid_frames(96, 104, 2)) == [96, 98, 100, 102, 104]
    assert list(geometry.grid_frames(96, 105, 2)) == [96, 98, 100, 102, 104]  # the end is off the grid
    assert list(geometry.grid_frames(96, 110, 5)) == [96, 101, 106]
    assert list(geometry.grid_frames(7, 7, 3)) == [7]
    grid = geometry.grid_frames(0, 2399, 2)  # the clip of SPEC 8.10
    assert (grid[0], grid[-1], len(grid)) == (0, 2398, 1200)


@pytest.mark.parametrize("frame, expected", [
    (101, (102, True)),
    (103, (104, True)),
    (100, (100, False)),
    (90, (96, True)),
    (201, (200, True)),
    (96, (96, False)),
    (200, (200, False)),
    (199, (200, True)),
])
def test_snap_forward_step_2(frame, expected):
    assert geometry.snap_to_grid(frame, 96, 2, 200) == expected


@pytest.mark.parametrize("frame, expected", [
    (97, (101, True)),
    (101, (101, False)),
    (196, (196, False)),  # the last grid frame: 96 + 20 * 5
    (197, (196, True)),  # no grid frame at or after 197 inside the clip
    (200, (196, True)),
    (300, (196, True)),
])
def test_snap_forward_step_5(frame, expected):
    assert geometry.snap_to_grid(frame, 96, 5, 200) == expected


@pytest.mark.parametrize("step", [1, 2, 3, 5, 7, 200])
def test_snap_against_brute_force(step):
    start, end = 96, 200
    grid = [f for f in range(start, end + 1) if (f - start) % step == 0]  # SPEC 3.4, written out
    for frame in range(80, 215):
        later = [g for g in grid if g >= frame]
        want = later[0] if later else grid[-1]
        got, moved = geometry.snap_to_grid(frame, start, step, end)
        assert got == want, frame
        assert moved is (want != frame), frame
        assert type(got) is int


def test_snap_accepts_numpy_integers():
    assert geometry.snap_to_grid(np.int64(101), np.int64(96), np.int64(2), np.int64(200)) == (102, True)


@pytest.mark.parametrize("step", [0, -2])
def test_step_must_be_at_least_one(step):
    with pytest.raises(ValueError, match="step"):
        geometry.grid_frames(96, 200, step)
    with pytest.raises(ValueError, match="step"):
        geometry.snap_to_grid(100, 96, step, 200)


def test_end_must_not_be_before_start():
    with pytest.raises(ValueError, match="end"):
        geometry.grid_frames(200, 96, 2)
    with pytest.raises(ValueError, match="end"):
        geometry.snap_to_grid(100, 200, 2, 96)


def test_frames_must_be_whole_numbers():
    with pytest.raises(ValueError, match="whole number"):
        geometry.grid_frames(96, 200, 2.5)
    with pytest.raises(ValueError, match="whole number"):
        geometry.snap_to_grid(100.5, 96, 2, 200)


# --------------------------------------------------------------------------- manifest (SPEC 4.1, 4.4, X9)

HEADER = "video_file,group,fps_true,stick_mm,check_mm,dish_mm\n"
VIDEO_NAME = "groupB_2026-10-06_1325_main_tracker.mp4"
ROW = "groupB_2026-10-06_1325_main.MOV,B,{fps},30,20,{dish}\n"


def write_manifest(folder: Path, fps="239.6", dish="35", rows=None) -> Path:
    """Write <folder>/data/manifest.csv with one row for VIDEO_NAME's original (or the given rows)."""
    path = folder / "data" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + (ROW.format(fps=fps, dish=dish) if rows is None else rows), encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path):
    """Three places a manifest can be: the current folder, above the export, above the video."""
    cwd = tmp_path / "here"
    export = tmp_path / "group repo" / "ana" / "tracks" / "A.csv"
    video = tmp_path / "vidéos" / "day 1" / VIDEO_NAME
    for folder in (cwd, export.parent, video.parent):
        folder.mkdir(parents=True)
    return cwd, export, video


def test_fps_from_manifest_is_the_ported_function(tmp_path):
    assert geometry.fps_from_manifest is tracker_io._fps_from_manifest
    manifest = write_manifest(tmp_path, fps="238.0")
    assert geometry.fps_from_manifest(tmp_path / VIDEO_NAME, manifest) == pytest.approx(238.0)


def test_manifest_in_the_current_folder_comes_first(tree):
    cwd, export, video = tree
    here = write_manifest(cwd)
    write_manifest(export.parents[2])
    write_manifest(video.parents[1])
    found = geometry.find_manifest(video, export=export, cwd=cwd)
    assert found.is_absolute()
    assert found.samefile(here)


def test_then_upward_from_the_export(tree):
    cwd, export, video = tree
    above_export = write_manifest(export.parents[2])  # "group repo"/data/manifest.csv, two folders up
    write_manifest(video.parents[1])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_export)


def test_then_upward_from_the_video(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_video)
    assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)  # no export given


def test_upward_search_starts_in_the_files_own_folder(tree):
    cwd, export, video = tree
    beside_export = write_manifest(export.parent)
    write_manifest(export.parents[2])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(beside_export)


def test_no_manifest_gives_none(tree):
    cwd, export, video = tree
    assert geometry.find_manifest(video, export=export, cwd=cwd) is None


def test_current_folder_is_not_searched_upward(tree):
    cwd, export, video = tree
    write_manifest(cwd)
    deeper = cwd / "sub"  # last week's rule looks in the current folder itself, not above it
    deeper.mkdir()
    assert geometry.find_manifest(video, export=export, cwd=deeper) is None


def test_a_manifest_without_this_video_is_passed_over(tree):
    cwd, export, video = tree
    write_manifest(cwd, rows="another_clip.MOV,B,120.0,30,20,35\n")  # exists, but no row for the video
    above_export = write_manifest(export.parents[2])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_export)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX folder permissions")
def test_a_data_folder_that_cannot_be_opened_is_passed_over(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    locked = video.parent / "data"  # nearer to the video than the manifest, and closed to us
    locked.mkdir()
    locked.chmod(0)
    try:
        if os.access(locked, os.X_OK):
            pytest.skip("folder permissions are not enforced for this user")
        assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)
        assert geometry.dish_mm_from_manifest(video, locked / "manifest.csv") is None
    finally:
        locked.chmod(0o755)


def test_a_folder_named_like_the_manifest_is_passed_over(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    (video.parent / "data" / "manifest.csv").mkdir(parents=True)
    assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)


def test_default_current_folder_and_relative_paths(tree, monkeypatch):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    monkeypatch.chdir(video.parent)
    assert geometry.find_manifest(VIDEO_NAME).samefile(above_video)  # a bare file name
    here = write_manifest(video.parent)
    assert geometry.find_manifest(VIDEO_NAME).samefile(here)  # ./data/manifest.csv


def test_dish_mm_from_manifest(tmp_path):
    manifest = write_manifest(tmp_path, dish="34.6")
    assert geometry.dish_mm_from_manifest(tmp_path / VIDEO_NAME, manifest) == pytest.approx(34.6)
    # the original video's name works too (the ported rule drops "_tracker" and the extension)
    assert geometry.dish_mm_from_manifest(Path("groupB_2026-10-06_1325_main.MOV"), manifest) == pytest.approx(34.6)


@pytest.mark.parametrize("dish", ["", "about 35", "0", "-35", "nan", "inf"])
def test_dish_mm_that_is_not_a_diameter_gives_none(tmp_path, dish):
    manifest = write_manifest(tmp_path, dish=dish)
    assert geometry.dish_mm_from_manifest(tmp_path / VIDEO_NAME, manifest) is None


def test_dish_mm_without_file_column_or_row_gives_none(tmp_path):
    video = tmp_path / VIDEO_NAME
    assert geometry.dish_mm_from_manifest(video, tmp_path / "none.csv") is None
    manifest = write_manifest(tmp_path)
    assert geometry.dish_mm_from_manifest(tmp_path / "another_tracker.mp4", manifest) is None
    twice = write_manifest(tmp_path / "twice", rows=2 * ROW.format(fps="239.6", dish="35"))
    assert geometry.dish_mm_from_manifest(video, twice) is None
    no_dish = tmp_path / "no_dish.csv"
    no_dish.write_text("video_file,fps_true\ngroupB_2026-10-06_1325_main.MOV,239.6\n", encoding="utf-8")
    assert geometry.dish_mm_from_manifest(video, no_dish) is None
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert geometry.dish_mm_from_manifest(video, empty) is None


def test_manifest_rows_are_matched_like_last_week(tmp_path):
    # dish_mm and fps_true must come from the same row: wherever the ported reader finds the video,
    # so does dish_mm_from_manifest, and the other way round
    cases = {
        "plain": ROW.format(fps="239.6", dish="35"),
        "folder in video_file": "videos/" + ROW.format(fps="239.6", dish="35"),
        "row names the tracker copy": VIDEO_NAME + ",B,239.6,30,20,35\n",
        "other video": "another_clip.MOV,B,239.6,30,20,35\n",
        "two rows": 2 * ROW.format(fps="239.6", dish="35"),
        "no rows": "",
    }
    video = tmp_path / VIDEO_NAME
    found = {}
    for i, (name, rows) in enumerate(cases.items()):
        manifest = write_manifest(tmp_path / str(i), rows=rows)
        fps, dish = geometry.fps_from_manifest(video, manifest), geometry.dish_mm_from_manifest(video, manifest)
        assert (fps is None) == (dish is None), name
        found[name] = dish
    assert found["plain"] == pytest.approx(35.0) and found["folder in video_file"] == pytest.approx(35.0)
    assert found["other video"] is None and found["two rows"] is None and found["no rows"] is None


# --------------------------------------------------------------------------- module rules

def test_tracker_io_does_not_import_geometry():
    source = ast.parse(Path(tracker_io.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {node.module or ""} | {alias.name for alias in node.names}
    assert not [name for name in imported if "geometry" in name]
