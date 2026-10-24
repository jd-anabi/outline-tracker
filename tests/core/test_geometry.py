"""Tests of outline_tracker/core/geometry.py: the image <-> world transform and its tie to Tracker's calibration.

Expected values come from SPEC 3.2 and 7.3 worked by hand, from the template's `tracker_map`, from
brute force over the definition, and from synthetic points whose truth is known. Nothing is copied
from the output of the code under test.

Coordinates: (u, v) are image pixels in Tracker's convention (origin at the top-left corner of the
frame, u to the right, v down, pixel centers at +0.5); (x, y) are mm in the user's axes, y up.

The rest of the tests of the module are beside this file: the calibration tools (stick, tape check,
stopwatch, circle fit) in tests/core/test_geometry_calibration.py, and the frame grid and the
manifest in tests/core/test_geometry_grid_manifest.py.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from helpers import MM_PER_PX, tracker_map  # the template's map, verbatim (0.05 mm per px)

from outline_tracker.core import geometry, tracker_io
from outline_tracker.core.geometry import WorldFrame
from outline_tracker.core.tracker_io import Calibration, fit_calibration


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
