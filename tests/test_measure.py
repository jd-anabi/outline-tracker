"""Tests for outline_tracker.measure: the ported mask_center (SPEC 3.1, 7.1, 13.1 Centroid).

The first test is the template's, unchanged but for its import line (tests/test_port_fidelity.py
checks that). The others check the same convention from geometry: a mask's center is the mean of
its pixel centers, and the pixel (column c, row r) has its center at (c + 0.5, r + 0.5), Tracker's
rule. Equality with last week's function on random masks is in tests/test_port_equivalence.py.
"""

import numpy as np
import pytest

# The template's test calls its module `segment`; the alias keeps that test's text unchanged.
from outline_tracker import measure as segment


def test_mask_center_uses_trackers_pixel_convention():
    m = np.zeros((40, 40), bool)
    m[20:22, 10:13] = True  # rows 20-21, columns 10-12
    cx, cy, area = segment.mask_center(m)
    assert (cx, cy, area) == (pytest.approx(11.5), pytest.approx(21.0), 6)
    assert np.isnan(segment.mask_center(np.zeros((5, 5), bool))[0])


@pytest.mark.parametrize(
    "first_col, last_col, first_row, last_row",
    [(0, 0, 0, 0), (3, 9, 5, 5), (10, 10, 2, 30), (0, 63, 0, 47)],
)
def test_a_filled_rectangle_is_centered_on_its_geometric_center(first_col, last_col, first_row, last_row):
    # The rectangle covers the plane from first_col to last_col + 1 and from first_row to last_row + 1.
    m = np.zeros((48, 64), bool)
    m[first_row:last_row + 1, first_col:last_col + 1] = True
    u, v, area = segment.mask_center(m)
    assert u == pytest.approx((first_col + last_col + 1) / 2)
    assert v == pytest.approx((first_row + last_row + 1) / 2)
    assert area == (last_col - first_col + 1) * (last_row - first_row + 1)


def test_rotating_the_image_by_180_degrees_maps_the_center_through_the_frame_center():
    # A point (u, v) of an image W wide and H high lands at (W - u, H - v) after a 180 degree turn.
    # That is true for the pixel-center convention (+0.5) and false for the pixel-index convention.
    rng = np.random.default_rng(3)
    h, w = 37, 53
    m = rng.random((h, w)) < 0.3
    u, v, area = segment.mask_center(m)
    u2, v2, area2 = segment.mask_center(m[::-1, ::-1])
    assert (u2, v2) == pytest.approx((w - u, h - v))
    assert area2 == area == int(m.sum())


def test_the_empty_mask_is_nan_with_area_zero_and_a_found_mask_gives_plain_numbers():
    u, v, area = segment.mask_center(np.zeros((6, 4), bool))
    assert np.isnan(u) and np.isnan(v) and area == 0 and type(area) is int
    u, v, area = segment.mask_center(np.ones((6, 4), bool))
    assert (u, v, area) == (2.0, 3.0, 24)
    assert type(u) is float and type(v) is float and type(area) is int
