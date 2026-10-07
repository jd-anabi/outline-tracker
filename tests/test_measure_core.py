"""Tests for outline_tracker.measure: the second moments and the core of a mask (SPEC 7.3; 13.1
Moments and Core).

Every shape is an implicit function rasterized at the pixel centers (tests/analytic_shapes.py),
and every expected value below comes from the shape's geometry or from counting pixels by hand.
A result given to measure_mask is in full-frame pixels; `input_box` = (c0, r0, width, height)
is the image the model saw, in full-frame pixels. Covariances are (mu_uu, mu_uv, mu_vv) in px^2.
"""

import analytic_shapes as shapes
import numpy as np
import pytest
from analytic_shapes import blocks

# These tests were written next to the ported test of tests/test_measure.py, which calls the
# module `segment` (the template's name), and were moved here unchanged: the name is kept.
from outline_tracker import measure as segment

FULL_HD = (0, 0, 1920, 1080)
CENTERS = [(0.5, 0.5), (0.0, 0.0), (0.37, -0.19)]  # sub-pixel positions of a shape's center


def measure(result, frame=0, input_box=FULL_HD, mode="coarse", **settings):
    return segment.measure_mask(result, frame, input_box, mode, **settings)


def covariance_of(mask):
    """Covariance (mu_uu, mu_uv, mu_vv) of a mask's pixel centers, in px^2. Written out here so
    that the premise of the core test does not rest on the code under test."""
    rows, cols = np.nonzero(mask)
    du, dv = cols - cols.mean(), rows - rows.mean()
    return np.array([(du * du).mean(), (du * dv).mean(), (dv * dv).mean()])


def axes_of(cov):
    """(L1, L2, angle): 4 sqrt(eigenvalue) for both eigenvalues, in px, and the angle of the major
    axis in the (u, v) plane from +u toward +v, in rad, for a covariance (mu_uu, mu_uv, mu_vv)."""
    values, vectors = np.linalg.eigh(np.array([[cov[0], cov[1]], [cov[1], cov[2]]]))
    major = vectors[:, 1]
    return 4 * np.sqrt(values[1]), 4 * np.sqrt(max(values[0], 0.0)), np.arctan2(major[1], major[0])


def axis_error_deg(angle, true_angle):
    """The angle between two axes in degrees, 0 to 90: an axis has no direction (it is defined mod 180)."""
    return abs((np.degrees(angle - true_angle) + 90.0) % 180.0 - 90.0)


# --------------------------------------------------------------------------- moments (13.1)


@pytest.mark.parametrize("offset", CENTERS)
@pytest.mark.parametrize("angle_deg", [0, 10, 30, 45, 60, 90, 123, -30, 77.7])
def test_moments_of_an_ellipse_give_its_axes_and_its_angle(angle_deg, offset):
    # A filled ellipse with semi-axes a, b has variances a^2 / 4 and b^2 / 4 along its axes,
    # so 4 sqrt(lambda) = 2a and 2b.
    center = (100.0 + offset[0], 100.0 + offset[1])
    u, v = shapes.pixel_centers(200, 200)
    d = shapes.ellipse(u, v, center, 40.0, 15.0, np.radians(angle_deg))
    record = measure(shapes.to_result(d))
    major, minor, angle = axes_of(record.cov_full)
    assert major == pytest.approx(80.0, rel=0.02)
    assert minor == pytest.approx(30.0, rel=0.02)
    assert axis_error_deg(angle, np.radians(angle_deg)) < 1.0
    assert (record.u, record.v) == pytest.approx(center, abs=0.25)
    assert record.area_px == int((d > 0).sum())


def test_the_covariance_is_stored_as_uu_uv_vv():
    # 3 rows x 9 columns: variance (9^2 - 1) / 12 along u and (3^2 - 1) / 12 along v, no tilt.
    wide = measure(shapes.pixel_result(blocks((20, 20), (4, 7, 5, 14))))
    assert wide.cov_full == pytest.approx([80 / 12, 0.0, 8 / 12])
    # Pixels on the diagonal toward +u and +v: u and v grow together, so mu_uv > 0; mirrored, < 0.
    diagonal = np.eye(6, dtype=bool)
    assert measure(shapes.pixel_result(diagonal)).cov_full == pytest.approx([35 / 12, 35 / 12, 35 / 12])
    assert measure(shapes.pixel_result(diagonal[:, ::-1])).cov_full == pytest.approx([35 / 12, -35 / 12, 35 / 12])


# --------------------------------------------------------------------------- core (7.3, 13.1)


# The premise of the next test has a margin of about 6%, and a 3 px rod is a coarse thing to
# rasterize: when its two sides run exactly through pixel centers (the body's center on whole
# pixels and the rods along the grid or its diagonal) it comes out up to a third thinner than
# drawn, and the premise is then false. These centers avoid that alignment.
ROD_CENTERS = [(0.5, 0.5), (0.37, -0.19), (0.13, 0.71)]


@pytest.mark.parametrize("offset", ROD_CENTERS)
@pytest.mark.parametrize("angle_deg", [0, 20, 45, 90, 137, 200, 291])
def test_the_core_follows_the_body_when_sideways_rods_mislead_the_full_mask(angle_deg, offset):
    # Body 47 x 20 px, two 3 px rods at 90 degrees reaching 30 px beyond the body's surface.
    angle = np.radians(angle_deg)
    center = (130.0 + offset[0], 130.0 + offset[1])
    u, v = shapes.pixel_centers(260, 260)
    d_all, d_body = shapes.body_with_rods(u, v, center, angle)
    mask, body = d_all > 0, d_body > 0

    # The premise: the rods carry the second moment across the body above the one along it.
    cov = covariance_of(mask)
    along, across = shapes.direction(angle), shapes.direction(angle + np.pi / 2)
    matrix = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
    assert across @ matrix @ across > along @ matrix @ along

    record = measure(shapes.to_result(d_all))
    assert axis_error_deg(axes_of(record.cov_full)[2], angle) > 45.0   # the full mask's axis is misled
    assert record.core_fallback is False
    assert record.core_r_px == max(1, int(np.floor(0.1 * axes_of(cov)[0] + 0.5)))
    assert axis_error_deg(axes_of(record.cov_core)[2], angle) <= 5.0
    assert np.hypot(record.core_u - center[0], record.core_v - center[1]) <= 0.5
    assert record.core_frac == pytest.approx(body.sum() / mask.sum(), rel=0.05)


@pytest.mark.parametrize("offset", CENTERS)
@pytest.mark.parametrize("angle_deg", [0, 15, 30, 45, 60, 90, 120])
def test_a_slender_ellipse_falls_back_to_the_full_mask(angle_deg, offset):
    # 72 x 12 px (aspect ratio 6). L1 is about 72 px, so the opening disk has radius 7: its pixel
    # centers span more than 12 px in every direction (14 along the grid, 12.7 on the diagonal),
    # the ellipse is 12 px across, so the disk fits nowhere and the opening removes everything.
    center = (100.0 + offset[0], 100.0 + offset[1])
    u, v = shapes.pixel_centers(200, 200)
    record = measure(shapes.to_result(shapes.ellipse(u, v, center, 36.0, 6.0, np.radians(angle_deg))))
    assert record.core_r_px == 7
    assert record.core_fallback is True
    assert record.core_frac < 0.5      # the fraction the opening left, not 1
    assert record.core_frac == 0.0
    assert np.array_equal(record.cov_core, record.cov_full)   # the core is the full mask
    assert (record.core_u, record.core_v) == (record.u, record.v)


def test_the_fallback_keeps_the_fraction_that_the_opening_left():
    # A 20 x 20 px square with a 2 px thick, 300 px long bar on its right side: 1000 px.
    # core_open_frac = 0.001 gives the smallest disk (radius 1: a pixel and its 4 neighbors).
    # The bar is too thin for it. Of the square it keeps the 18 x 18 inside and the two pixels
    # that touch the bar; putting the disk back on those gives the square without its 4 corner
    # pixels plus the bar's first 2 pixels: 398 px, less than half of 1000.
    mask = blocks((40, 340), (10, 30, 10, 30), (19, 21, 30, 330))
    assert mask.sum() == 1000
    record = measure(shapes.pixel_result(mask, offset=(50, 60)), core_open_frac=0.001)
    assert record.core_r_px == 1
    assert record.core_fallback is True
    assert record.core_frac == pytest.approx(398 / 1000, abs=1e-12)
    assert np.array_equal(record.cov_core, record.cov_full)
    assert (record.core_u, record.core_v) == (record.u, record.v)


@pytest.mark.parametrize("radius, lost_per_corner", [(1, 1), (2, 3), (3, 5)])
def test_the_core_is_an_opening_with_a_disk_of_pixels_within_the_radius(radius, lost_per_corner):
    # Opening a 20 x 20 px square with the disk x^2 + y^2 <= r^2 removes, in each corner, the pixels
    # (x, y), counted from the last pixel that the eroded square keeps, with x^2 + y^2 > r^2:
    # 1 of 1 for r = 1, 3 of 4 for r = 2 and 5 of 9 for r = 3. (A diamond would lose 6 for r = 3,
    # a square structuring element none.) The square's variance is (20^2 - 1) / 12 px^2.
    mask = blocks((44, 48), (10, 30, 12, 32))
    major = 4 * np.sqrt((20 ** 2 - 1) / 12)
    record = measure(shapes.pixel_result(mask, offset=(5, 9)), core_open_frac=radius / major)
    assert record.core_r_px == radius
    assert record.core_fallback is False
    assert record.core_frac == pytest.approx((400 - 4 * lost_per_corner) / 400, abs=1e-12)
    assert (record.core_u, record.core_v) == pytest.approx((5 + 22.0, 9 + 20.0), abs=1e-9)
    assert record.cov_core[1] == pytest.approx(0.0, abs=1e-9)
    assert record.cov_core[0] == pytest.approx(record.cov_core[2], abs=1e-9)
    assert record.cov_core[0] < record.cov_full[0]   # the corners are gone


def test_a_disk_far_larger_than_the_mask_is_a_fallback_not_a_huge_computation():
    # A disk of radius 12 px has L1 = 24 px; with core_open_frac = 50 the opening disk would be
    # about 1200 px in radius. It fits nowhere, so nothing is left and the core is the mask.
    u, v = shapes.pixel_centers(40, 40)
    record = measure(shapes.to_result(shapes.disk(u, v, (20.3, 19.8), 12.0)), core_open_frac=50.0)
    assert 1150 < record.core_r_px < 1250
    assert record.core_fallback is True and record.core_frac == 0.0
    assert np.array_equal(record.cov_core, record.cov_full)


@pytest.mark.parametrize("core_open_frac, radius", [(0.3125, 3), (0.5625, 5), (0.25, 2), (0.1, 1), (0.01, 1)])
def test_the_opening_radius_rounds_half_up_and_is_at_least_one(core_open_frac, radius):
    # A row of 7 px has variance (7^2 - 1) / 12 = 4 px^2 along it, so L1 = 4 sqrt(4) = 8 px exactly.
    # 0.3125 x 8 = 2.5 and 0.5625 x 8 = 4.5 are exact in binary: floor(x + 0.5) gives 3 and 5,
    # where Python's round() would give 2 and 4.
    record = measure(shapes.pixel_result(blocks((9, 15), (4, 5, 3, 10))), core_open_frac=core_open_frac)
    assert np.array_equal(record.cov_full, [4.0, 0.0, 0.0])   # exactly, or 2.5 would not be exactly 2.5
    assert record.core_r_px == radius
