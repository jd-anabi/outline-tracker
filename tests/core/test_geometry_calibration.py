"""Tests of outline_tracker/core/geometry.py: the calibration tools, stick, tape check, stopwatch and circle fit.

Expected values come from SPEC 4.1-4.4 worked by hand, from the definition of the least-squares
circle, and from synthetic points whose truth is known. Nothing is copied from the output of the code
under test.

Coordinates: (u, v) are image pixels in Tracker's convention (origin at the top-left corner of the
frame, u to the right, v down, pixel centers at +0.5); lengths are in mm.

The rest of the tests of the module are in tests/core/test_geometry.py (the transform, its
covariances, its tie to Tracker's calibration) and tests/core/test_geometry_grid_manifest.py (the frame
grid and the manifest).
"""

import math

import numpy as np
import pytest

from outline_tracker.core import geometry


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
