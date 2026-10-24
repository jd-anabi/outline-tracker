"""`tracker_io.fit_calibration` on maps that the test itself made (docs/ROADMAP.md, W1 step 5).

Every expected value here is the map that the test made the points with, or a typed map and a typed
distance. None is taken from what the package returns.

- `fit_calibration`: 200 random maps of each form, made from a scale, an angle and a shift that
  the test drew (seeded numpy); rows without a value; the refusal below 1 px; the choice of the
  form under noise; points moved off a typed map by a typed distance to both sides, where the fit
  is still that map and `rms_mm` follows from the distance by arithmetic.

Coordinates: pixelx and pixely are px in Tracker's convention (pixelx to the right, pixely downward,
pixel centers at +0.5); x and y are mm in Tracker's axes, y up. A map is given by its scale in mm per
px, the angle of its x axis in radians and its shift in mm.

The tests of `read_tracker_export`, `make_plan` and `write_tracker_file` on known tracks and files
are in tests/core/test_tracker_io_known_files.py.
"""

import numpy as np
import pytest

from outline_tracker.core import tracker_io

N_RANDOM = 200
FRAME_PX = (1920.0, 1080.0)  # the points of a map are drawn inside a frame of this size, px
SAME_MM, SAME_PX = 0.5e-6, 0.5e-3  # half of the last decimal that a Tracker-format file has: 6 for mm, 3 for px


# ---------------------------------------------------------------------------------------------
# fit_calibration on known maps


def known_map(rng, mirrored=False):
    """A random pixel -> mm map, as (to_mm, scale, matrix).

    Tracker's map is a turn by `angle` after a flip of the vertical axis (its y points up, image rows
    go down), times `scale`, plus a shift; a mirrored map is the same without the flip:
        Tracker's:  x = a px + c py + tx,   y = c px - a py + ty
        mirrored:   x = a px - c py + tx,   y = c px + a py + ty
    with a = scale cos(angle) and c = scale sin(angle). The scale is between 0.003 and 1 mm per px,
    the angle is any, and the shift is up to 300 mm either way. `to_mm(px, py)` takes arrays;
    `matrix` is the 2 x 2 part of the map, mm per px.
    """
    scale = 10 ** rng.uniform(-2.5, 0.0)
    angle = rng.uniform(-np.pi, np.pi)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = scale * (turn if mirrored else turn @ np.diag([1.0, -1.0]))
    tx, ty = rng.uniform(-300, 300, 2)

    def to_mm(px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        return matrix[0, 0] * px + matrix[0, 1] * py + tx, matrix[1, 0] * px + matrix[1, 1] * py + ty

    return to_mm, scale, matrix


def assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng):
    """A fitted calibration is the map that made its points: the scale, the 2 x 2 part, no residual,
    and at five points in and around the frame `to_mm` gives the map's mm and `to_px` gives the px
    back from the map's own mm. "The same", and "no residual", is half of the last decimal that a
    Tracker-format file has, 0.5e-6 mm and 0.5e-3 px: rounding in the fit grows with the size of
    the numbers (here up to 2500 mm), and no file could show a difference below that."""
    assert cal.mm_per_px == pytest.approx(scale, rel=1e-6)
    assert cal.matrix() == pytest.approx(matrix, rel=0, abs=1e-6 * scale)
    assert cal.rms_mm < SAME_MM  # zero up to rounding
    probe_px, probe_py = rng.uniform(-100, 2000, 5), rng.uniform(-100, 1200, 5)
    probe_x, probe_y = to_mm(probe_px, probe_py)
    x, y = cal.to_mm(probe_px, probe_py)
    assert x == pytest.approx(probe_x, rel=0, abs=SAME_MM) and y == pytest.approx(probe_y, rel=0, abs=SAME_MM)
    px, py = cal.to_px(probe_x, probe_y)
    assert px == pytest.approx(probe_px, rel=0, abs=SAME_PX) and py == pytest.approx(probe_py, rel=0, abs=SAME_PX)


def test_fit_calibration_recovers_trackers_map_whatever_its_scale_angle_and_shift():
    rng = np.random.default_rng(2026)
    for case in range(N_RANDOM):
        to_mm, scale, matrix = known_map(rng)
        n = int(rng.integers(2, 41))  # two points are enough for Tracker's form
        px, py = rng.uniform(0, FRAME_PX[0], n), rng.uniform(0, FRAME_PX[1], n)
        x, y = to_mm(px, py)
        given = (px, py, x, y) if case % 2 else tuple(values.tolist() for values in (px, py, x, y))  # arrays, lists
        cal = tracker_io.fit_calibration(*given)
        assert cal.flip is True, case
        assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_fit_calibration_recovers_a_mirrored_map_from_points_that_are_not_on_one_line():
    rng = np.random.default_rng(2027)
    corners = np.array([(100.5, 100.5), (1800.5, 150.5), (900.5, 1000.5)])  # a triangle: never on one line
    for case in range(N_RANDOM):
        to_mm, scale, matrix = known_map(rng, mirrored=True)
        n_more = int(rng.integers(0, 38))
        px = np.concatenate([corners[:, 0], rng.uniform(0, FRAME_PX[0], n_more)])
        py = np.concatenate([corners[:, 1], rng.uniform(0, FRAME_PX[1], n_more)])
        x, y = to_mm(px, py)
        given = (px, py, x, y) if case % 2 else tuple(values.tolist() for values in (px, py, x, y))
        cal = tracker_io.fit_calibration(*given)
        assert cal.flip is False, case
        assert cal.matrix()[0, 0] == cal.matrix()[1, 1] and cal.matrix()[0, 1] == -cal.matrix()[1, 0]  # no flip
        assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_rows_with_a_missing_value_change_nothing():
    rng = np.random.default_rng(5)
    to_mm, scale, matrix = known_map(rng)
    px, py = rng.uniform(0, FRAME_PX[0], 12), rng.uniform(0, FRAME_PX[1], 12)
    x, y = to_mm(px, py)
    complete = tracker_io.fit_calibration(px, py, x, y)
    # Four more rows, each without one of its four values, and far off the map with the other three
    # (1e6 mm): a fit that used one of them would be another fit.
    nan = float("nan")
    extra = np.array([(nan, 500.5, 1e6, 1e6), (900.5, nan, 1e6, -1e6), (900.5, 500.5, nan, 1e6),
                      (900.5, 500.5, -1e6, nan)])
    at = [0, 5, 5, 12]  # where they stand among the complete rows: first, in the middle, last
    with_gaps = [np.insert(values, at, extra[:, k]) for k, values in enumerate((px, py, x, y))]
    cal = tracker_io.fit_calibration(*with_gaps)
    assert cal == complete
    assert cal.flip is True
    assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_points_in_one_place_or_under_1_px_apart_are_refused_and_1_px_is_enough():
    to_mm, scale, _ = known_map(np.random.default_rng(6))

    def fit(px, py):
        return tracker_io.fit_calibration(px, py, *to_mm(px, py))

    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.full(7, 640.5), np.full(7, 360.5))  # seven points in one place
    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.array([640.5, 641.0]), np.array([360.5, 360.5]))  # 0.5 px apart
    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.array([640.5, 640.5]), np.array([360.5, 361.25]))  # 0.75 px apart, along pixely
    cal = fit(np.array([640.5, 641.5]), np.array([360.5, 360.5]))  # exactly 1.0 px apart
    assert cal.flip is True and cal.mm_per_px == pytest.approx(scale, rel=1e-6)


def test_mirrored_data_keep_the_mirrored_form_under_small_noise_and_lose_it_under_noise_far_larger_than_the_map():
    # The unflipped form is kept only if it fits clearly better. Noise 1000 times smaller than the
    # map's extent leaves it far better; under noise 1000 times larger neither form fits, so neither
    # fits clearly better, and Tracker's form stands.
    rng = np.random.default_rng(11)
    to_mm, scale, _ = known_map(rng, mirrored=True)
    px, py = rng.uniform(0, FRAME_PX[0], 40), rng.uniform(0, FRAME_PX[1], 40)
    x, y = to_mm(px, py)
    extent_mm = scale * float(np.hypot(*FRAME_PX))  # the frame's diagonal, as the map sees it
    small, large = rng.normal(0, 1e-3 * extent_mm, (2, 40)), rng.normal(0, 1e3 * extent_mm, (2, 40))
    assert tracker_io.fit_calibration(px, py, x + small[0], y + small[1]).flip is False
    assert tracker_io.fit_calibration(px, py, x + large[0], y + large[1]).flip is True


# Tracker's map at 0.05 mm per px with its x axis turned by 12 degrees and a shift of (-31.5, 27.25)
# mm, and seven points in px that are not on one line. E_MM is how far each mm value is moved.
TYPED_SCALE, TYPED_ANGLE_DEG, TYPED_SHIFT = 0.05, 12.0, (-31.5, 27.25)
TYPED_POINTS = [(100.5, 80.5), (1800.5, 150.5), (900.5, 1000.5), (400.5, 600.5), (1500.5, 900.5),
                (250.5, 950.5), (1200.5, 300.5)]
PROBES = [(-100.0, -100.0), (2000.0, 1200.0), (960.0, 540.0), (0.5, 1079.5), (1919.5, 0.5)]  # px
E_MM = 0.02


@pytest.mark.parametrize("order", ["all +e, then all -e", "alternating"])
def test_points_moved_by_e_to_both_sides_along_x_give_the_typed_map_and_an_rms_of_e_over_root_2(order):
    # Each of the seven points is given twice: with its mm value on the typed map moved along x by
    # +e, and by -e. For a map that gives p where the typed map gives m,
    #     (p - (m + e))^2 + (p - (m - e))^2 = 2 (p - m)^2 + 2 e^2,
    # so the sum of squares over all rows is twice the sum over the unmoved values plus a constant.
    # The least-squares fit over all rows is therefore the map of the unmoved values: the typed
    # map. A fit over a part of the rows is another map: seven is odd, so in neither order is the
    # first half of the rows a set of whole pairs.
    a, c = TYPED_SCALE * np.cos(np.radians(TYPED_ANGLE_DEG)), TYPED_SCALE * np.sin(np.radians(TYPED_ANGLE_DEG))
    tx, ty = TYPED_SHIFT

    def typed_map(px, py):  # Tracker's form
        return a * px + c * py + tx, c * px - a * py + ty

    px, py = np.array(TYPED_POINTS).T
    x, y = typed_map(px, py)
    if order == "alternating":  # point 1 by +e, point 1 by -e, point 2 by +e, ...
        rows = px.repeat(2), py.repeat(2), x.repeat(2) + np.tile([E_MM, -E_MM], len(px)), y.repeat(2)
    else:
        rows = np.tile(px, 2), np.tile(py, 2), np.concatenate([x + E_MM, x - E_MM]), np.tile(y, 2)
    cal = tracker_io.fit_calibration(*rows)

    assert cal.flip is True
    assert cal.mm_per_px == pytest.approx(TYPED_SCALE, rel=1e-6)
    assert (cal.a, cal.c) == pytest.approx((a, c), rel=0, abs=1e-6 * TYPED_SCALE)  # the scale and the angle
    assert (cal.tx, cal.ty) == pytest.approx((tx, ty), rel=0, abs=SAME_MM)
    probe_px, probe_py = np.array(PROBES).T
    probe_x, probe_y = typed_map(probe_px, probe_py)
    fitted_x, fitted_y = cal.to_mm(probe_px, probe_py)
    assert fitted_x == pytest.approx(probe_x, rel=0, abs=SAME_MM)
    assert fitted_y == pytest.approx(probe_y, rel=0, abs=SAME_MM)
    # rms_mm is the root mean square over the x and the y residuals of all rows together (the
    # docstring of fit_calibration). At the typed map the 14 x residuals are +e or -e and the 14 y
    # residuals are 0: sqrt(14 e^2 / 28) = e / sqrt(2), about 0.01414 mm. The root mean square of
    # the distances would be e, and the mean of the residuals' sizes e / 2.
    assert cal.rms_mm == pytest.approx(E_MM / np.sqrt(2.0), rel=1e-9)
