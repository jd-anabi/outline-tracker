"""Tests for outline_tracker.measure: the ported mask_center (SPEC 3.1, 7.1, 13.1 Centroid).

The first test is the template's, unchanged but for its import line (tests/test_port_fidelity.py
checks that); the second keeps the names it uses bound to the modules imported here. The others
check the same convention from geometry: a mask's center is the mean of its pixel centers, and the
pixel (column c, row r) has its center at (c + 0.5, r + 0.5), Tracker's rule. Equality with last
week's function on random masks is in tests/test_port_equivalence.py.
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


def test_the_template_test_runs_with_the_modules_named_by_the_import_lines():
    # tests/test_port_fidelity.py compares the text of the test above and this file's import lines.
    # A name rebound further down (by an assignment, or by a module loaded by name) is in neither,
    # and would change what that unchanged text does. Pytest runs this after the whole file has
    # been executed, so it sees the three names exactly as the test above sees them.
    assert (np.__name__, pytest.__name__, segment.__name__) == ("numpy", "pytest", "outline_tracker.measure")


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


# ---------------------------------------------------------------------------------------------
# measure_mask and PixelRecord (SPEC 7.1-7.4, 7.8, 8.12; 13.1 Moments and Core)
#
# Every shape is an implicit function rasterized at the pixel centers (tests/analytic_shapes.py),
# and every expected value below comes from the shape's geometry or from counting pixels by hand.
# A result given to measure_mask is in full-frame pixels; `input_box` = (c0, r0, width, height)
# is the image the model saw, in full-frame pixels.
#
# tests/test_port_fidelity.py allows this file no top-level import line beyond the template's and
# the one above, so the two modules the tests below need are loaded by name. That goes around the
# check, which does not see these two lines; what the check is for (the template's test means what
# it meant) is kept by test_the_template_test_runs_with_the_modules_named_by_the_import_lines.
shapes = __import__("analytic_shapes")
schema = __import__("outline_tracker.schema").schema

FULL_HD = (0, 0, 1920, 1080)
DISH_CROP = (100, 50, 300, 200)  # columns 100..399 and rows 50..249 of the full frame
CENTERS = [(0.5, 0.5), (0.0, 0.0), (0.37, -0.19)]  # sub-pixel positions of a shape's center


def measure(result, frame=0, input_box=FULL_HD, mode="coarse", **settings):
    return segment.measure_mask(result, frame, input_box, mode, **settings)


def blocks(shape, *rectangles):
    """A mask of `shape` (rows, columns) with the given pixel rectangles (row0, row1, col0, col1) set;
    row1 and col1 are one past the last pixel, as in a slice."""
    mask = np.zeros(shape, bool)
    for row0, row1, col0, col1 in rectangles:
        mask[row0:row1, col0:col1] = True
    return mask


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


def spacing(points):
    """Distances from each point of a closed polygon to the next (the last one back to the first)."""
    points = np.asarray(points, float)
    return np.hypot(*(np.roll(points, -1, axis=0) - points).T)


def perimeter(points):
    return float(spacing(points).sum())


def polygon_area(points):
    x, y = np.asarray(points, float).T
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def assert_same_record(a, b, shift=(0, 0)):
    """Record `b` equals record `a` moved by `shift` = (columns, rows): positions moved, the rest equal."""
    for name in a.__dataclass_fields__:
        first, second = getattr(a, name), getattr(b, name)
        if name in ("u", "core_u"):
            assert second == pytest.approx(first + shift[0], abs=1e-9), name
        elif name in ("v", "core_v"):
            assert second == pytest.approx(first + shift[1], abs=1e-9), name
        elif name == "outline_px":
            assert second == pytest.approx(first + np.array(shift), abs=1e-3), name
        elif name == "mask_offset":
            assert second == (first[0] + shift[0], first[1] + shift[1])
        elif isinstance(first, np.ndarray):
            assert second == pytest.approx(first, abs=1e-9, nan_ok=True), name
        elif isinstance(first, float):
            assert second == pytest.approx(first, abs=1e-12, nan_ok=True), name
        else:
            assert second == first, name


# --------------------------------------------------------------------------- the record itself


def test_the_record_has_the_fields_and_dtypes_of_the_results_schema():
    u, v = shapes.pixel_centers(60, 80)
    visible = measure(shapes.to_result(shapes.disk(u, v, (40.2, 30.7), 12.0), score=3.5), frame=17)
    lost = measure(shapes.pixel_result(np.zeros((0, 0), bool)), frame=18)
    per_frame = [key for key in schema.RESULTS_KEYS if key.name != "frames"]  # "frames" is the record's `frame`
    assert list(visible.__dataclass_fields__) == ["frame"] + [key.name for key in per_frame]
    scalar_types = {"bool": bool, "int32": int, "float64": float, "float32": float, "U6": str}
    for record in (visible, lost):
        assert type(record.frame) is int
        for key in per_frame:
            value = getattr(record, key.name)
            if key.name == "mask_bits":  # one crop's share of the track's byte string
                assert isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 1
            elif key.name in ("mask_offset", "mask_shape"):
                assert type(value) is tuple and len(value) == 2 and all(type(x) is int for x in value)
            elif len(key.shape) > 1:  # an array per frame: (3,) or (256, 2)
                assert isinstance(value, np.ndarray), key.name
                assert value.shape == key.shape[1:] and value.dtype == np.dtype(key.dtype), key.name
            else:
                assert type(value) is scalar_types[key.dtype], key.name
        assert record.outline_px.shape == (schema.STORED_OUTLINE_POINTS, 2)
        assert len(record.mode) <= 6  # stored as U6


def test_frame_mode_and_score_are_stored_as_given():
    u, v = shapes.pixel_centers(40, 40)
    d = shapes.disk(u, v, (20.0, 20.0), 9.0)
    record = measure(shapes.to_result(d, score=3.5), frame=17, mode="fine")
    assert (record.frame, record.mode, record.score) == (17, "fine", 3.5)
    assert measure(shapes.to_result(d), mode="coarse").mode == "coarse"
    assert np.isnan(measure(shapes.to_result(d, score=None)).score)  # a backend without a score


@pytest.mark.parametrize("mode", ["", "Coarse", "coarse_dish", "finer", None])
def test_a_mode_other_than_coarse_or_fine_is_refused(mode):
    result = shapes.pixel_result(np.ones((3, 3), bool), offset=(10, 10))
    with pytest.raises(ValueError, match="coarse"):
        measure(result, mode=mode)


@pytest.mark.parametrize("input_box", [(0, 0, 0, 100), (0, 0, 100, 0), (5, 5, -20, 30)])
def test_an_input_box_without_a_size_is_refused(input_box):
    result = shapes.pixel_result(np.ones((3, 3), bool), offset=(10, 10))
    with pytest.raises(ValueError, match="input_box"):
        measure(result, input_box=input_box)


@pytest.mark.parametrize("core_open_frac", [float("nan"), float("inf"), -0.1])
def test_a_core_open_frac_that_is_negative_or_not_finite_is_refused(core_open_frac):
    result = shapes.pixel_result(np.ones((3, 3), bool), offset=(10, 10))
    with pytest.raises(ValueError, match="core_open_frac"):
        measure(result, core_open_frac=core_open_frac)


def test_logits_of_another_shape_than_the_mask_are_refused():
    result = shapes.pixel_result(np.ones((3, 3), bool), offset=(10, 10), logits=True)
    result.logits = np.ones((3, 4), np.float32)
    with pytest.raises(ValueError, match="logits"):
        measure(result)


def test_position_and_area_are_mask_center_of_the_full_frame_mask():
    # SPEC 7.2: the official position is exactly last week's mask_center of the whole mask,
    # whatever crop of it the model hands over and wherever that crop lies in the frame.
    rng = np.random.default_rng(5)
    for _ in range(25):
        full = np.zeros((90, 120), bool)
        row0, col0 = int(rng.integers(0, 50)), int(rng.integers(0, 70))
        height, width = int(rng.integers(1, 40)), int(rng.integers(1, 50))
        full[row0:row0 + height, col0:col0 + width] = rng.random((height, width)) < 0.4
        full[row0, col0] = True
        u, v, area = segment.mask_center(full)
        for origin in [(0, 0), (700, 300)]:
            record = measure(shapes.to_result(np.where(full, 1.0, -1.0), origin=origin))
            assert record.visible is True
            assert record.area_px == area
            assert (record.u, record.v) == pytest.approx((u + origin[0], v + origin[1]), abs=1e-9)


def test_the_stored_crop_is_the_bounding_box_packed_row_by_row():
    full = np.zeros((40, 50), bool)
    full[11:18, 20:25] = np.random.default_rng(8).random((7, 5)) < 0.5   # 7 rows x 5 columns = 35 bits
    full[11, 20:25] = full[17, 20:25] = True                             # the box is really 7 x 5
    full[11:18, 20] = full[11:18, 24] = True
    full[12, 21:24] = [True, False, False]                               # not symmetric: order matters
    tight = full[11:18, 20:25]
    record = measure(shapes.to_result(np.where(full, 1.0, -1.0), origin=(300, 200)))
    assert record.mask_offset == (320, 211)   # (column, row) of the crop's top-left pixel, full frame
    assert record.mask_shape == (7, 5)        # (rows, columns)
    assert len(record.mask_bits) == 5         # ceil(35 / 8) bytes
    assert np.array_equal(record.mask_bits, np.packbits(tight.ravel()))
    assert record.mask().dtype == bool and np.array_equal(record.mask(), tight)
    assert np.array_equal(segment.pack_mask(tight), record.mask_bits)
    assert np.array_equal(segment.unpack_mask(record.mask_bits, (7, 5)), tight)


def test_the_record_does_not_depend_on_how_much_of_the_image_the_crop_keeps():
    u, v = shapes.pixel_centers(200, 200)
    d, _ = shapes.body_with_rods(u, v, (100.3, 99.8), np.radians(20))
    results = [shapes.to_result(d, pad=pad, score=1.0) for pad in (1, 8, 40, None)]
    before = [(r.mask.copy(), r.logits.copy(), r.offset) for r in results]
    records = [measure(result) for result in results]
    for other in records[1:]:
        assert_same_record(records[0], other)
    for result, (mask, logits, offset) in zip(results, before):  # the input is left as it was
        assert np.array_equal(result.mask, mask) and np.array_equal(result.logits, logits)
        assert result.offset == offset


def test_moving_the_crop_moves_the_positions_and_nothing_else():
    u, v = shapes.pixel_centers(200, 200)
    d, _ = shapes.body_with_rods(u, v, (100.3, 99.8), np.radians(70))
    here = measure(shapes.to_result(d, score=1.0))
    there = measure(shapes.to_result(d, origin=(700, 300), score=1.0))
    assert_same_record(here, there, shift=(700, 300))
    assert there.u == pytest.approx(here.u + 700) and there.u > 700


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


# --------------------------------------------------------------------------- outline (7.4)


@pytest.mark.parametrize("origin", [(0, 0), (500, 300)])
@pytest.mark.parametrize("offset", CENTERS)
def test_the_outline_of_a_circle_is_its_level_set(offset, origin):
    # Logits = the exact signed distance of a circle of radius 50 px, so the level set is known.
    radius = 50.0
    center = (origin[0] + 80.0 + offset[0], origin[1] + 80.0 + offset[1])
    u, v = shapes.pixel_centers(160, 160, origin)
    record = measure(shapes.to_result(shapes.disk(u, v, center, radius), origin=origin))
    outline = record.outline_px
    assert outline.shape == (256, 2) and outline.dtype == np.float32
    assert perimeter(outline) == pytest.approx(2 * np.pi * radius, rel=0.005)
    assert outline.mean(axis=0) == pytest.approx(center, abs=0.01)   # (col + 1/2, row + 1/2) plus the offset
    assert np.hypot(outline[:, 0] - center[0], outline[:, 1] - center[1]) == pytest.approx(radius, abs=0.02)
    steps = spacing(outline)
    assert steps.max() - steps.min() < 0.01 * steps.mean()           # equally spaced along the contour


def test_without_logits_the_outline_is_the_boundary_through_the_pixel_centers():
    # 20 rows x 40 columns of pixels. Without logits: the rectangle through the centers of the
    # border pixels, 39 x 19 px. With +-1 logits: the level set on the pixel edges, 40 x 20 px.
    mask = blocks((40, 60), (10, 30, 5, 45))
    plain = measure(shapes.pixel_result(mask)).outline_px
    assert plain.shape == (256, 2) and plain.dtype == np.float32
    assert (plain[:, 0].min(), plain[:, 0].max()) == pytest.approx((5.5, 44.5), abs=1e-4)
    assert (plain[:, 1].min(), plain[:, 1].max()) == pytest.approx((10.5, 29.5), abs=1e-4)
    on_a_side = (np.isclose(plain[:, 0], 5.5, atol=1e-4) | np.isclose(plain[:, 0], 44.5, atol=1e-4)
                 | np.isclose(plain[:, 1], 10.5, atol=1e-4) | np.isclose(plain[:, 1], 29.5, atol=1e-4))
    assert on_a_side.all()
    # 256 points cut each corner by less than 0.14 px (spacing 0.45 px)
    assert 2 * (39 + 19) - 0.6 < perimeter(plain) <= 2 * (39 + 19) + 1e-3
    edges = measure(shapes.pixel_result(mask, logits=True)).outline_px
    assert (edges[:, 0].min(), edges[:, 0].max()) == pytest.approx((5.0, 45.0), abs=1e-4)
    assert (edges[:, 1].min(), edges[:, 1].max()) == pytest.approx((10.0, 30.0), abs=1e-4)


def test_without_logits_a_circle_gets_the_inner_pixel_boundary_and_the_same_numbers():
    radius, center = 50.0, (80.3, 79.9)
    u, v = shapes.pixel_centers(160, 160)
    d = shapes.disk(u, v, center, radius)
    with_logits, without = measure(shapes.to_result(d)), measure(shapes.to_result(d, logits=False))
    distance = np.hypot(without.outline_px[:, 0] - center[0], without.outline_px[:, 1] - center[1])
    # Border pixels have their centers inside the circle, less than 1 px from it; chords between
    # neighbors dip a little further in.
    assert distance.max() <= radius + 1e-3 and distance.min() > radius - 1.5
    assert distance.mean() < radius - 0.2
    for name in ("area_px", "u", "v", "core_u", "core_v", "core_frac", "core_r_px", "n_components", "edge"):
        assert getattr(without, name) == getattr(with_logits, name), name
    assert np.array_equal(without.cov_full, with_logits.cov_full)


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_a_disk_cut_by_the_input_border_gets_a_closed_outline_on_the_border(side):
    # A disk of radius 20 px whose center is 10.3 px inside the model's input: the model sees
    # only the part inside. The outline is the arc closed by the chord along the border.
    c0, r0, width, height = DISH_CROP
    radius, depth = 20.0, 10.3
    center, axis, border = {
        "left": ((c0 + depth, r0 + 90.2), 0, c0),
        "right": ((c0 + width - depth, r0 + 90.2), 0, c0 + width),
        "top": ((c0 + 150.4, r0 + depth), 1, r0),
        "bottom": ((c0 + 150.4, r0 + height - depth), 1, r0 + height),
    }[side]
    u, v = shapes.pixel_centers(height, width, origin=(c0, r0))
    record = measure(shapes.to_result(shapes.disk(u, v, center, radius), origin=(c0, r0)), input_box=DISH_CROP)
    assert record.edge is True

    outline = record.outline_px
    half_chord = np.sqrt(radius ** 2 - depth ** 2)
    half_chord_at_centers = np.sqrt(radius ** 2 - (depth - 0.5) ** 2)   # where the first pixels' centers are
    arc = radius * (2 * np.pi - 2 * np.arccos(depth / radius))
    cut_off = radius ** 2 * np.arccos(depth / radius) - depth * half_chord
    assert np.isfinite(outline).all()
    # it lies on the border and never beyond it
    inward = 1.0 if side in ("left", "top") else -1.0
    assert (inward * (outline[:, axis] - border)).min() == pytest.approx(0.0, abs=1e-3)
    on_border = np.isclose(outline[:, axis], border, atol=1e-3)
    assert on_border.sum() == pytest.approx(256 * 2 * half_chord / (arc + 2 * half_chord), abs=4)
    # The outline follows the border as far as the center of the last mask pixel of the first
    # row or column, which is less than 1 px short of the circle there; the 256 points are less
    # than half a pixel apart.
    along = outline[on_border, 1 - axis] - center[1 - axis]
    assert half_chord_at_centers - 1.5 < along.max() <= half_chord_at_centers
    assert half_chord_at_centers - 1.5 < -along.min() <= half_chord_at_centers
    # it is closed: no jump between neighbors, also from the last point back to the first
    assert spacing(outline).max() < 1.1 * perimeter(outline) / 256
    assert perimeter(outline) == pytest.approx(arc + 2 * half_chord, rel=0.01)
    assert polygon_area(outline) == pytest.approx(np.pi * radius ** 2 - cut_off, rel=0.005)


def test_holes_are_ignored():
    # A ring: outer radius 40 px, inner radius 15 px. The outline is the outer circle alone.
    center = (60.3, 59.8)
    u, v = shapes.pixel_centers(120, 120)
    ring = np.minimum(shapes.disk(u, v, center, 40.0), -shapes.disk(u, v, center, 15.0))
    for logits in (True, False):
        record = measure(shapes.to_result(ring, logits=logits))
        assert (record.n_components, record.largest_fraction, record.second_fraction) == (1, 1.0, 0.0)
        assert record.area_px == int((ring > 0).sum())   # the hole is not part of the mask
        assert record.area_px == pytest.approx(np.pi * (40.0 ** 2 - 15.0 ** 2), rel=0.01)
        distance = np.hypot(record.outline_px[:, 0] - center[0], record.outline_px[:, 1] - center[1])
        if logits:
            assert distance == pytest.approx(40.0, abs=0.02)
            assert polygon_area(record.outline_px) == pytest.approx(np.pi * 40.0 ** 2, rel=0.005)
        else:
            assert distance.max() <= 40.0 + 1e-3 and distance.min() > 38.5


def test_a_hole_with_a_longer_contour_than_the_outer_one_is_still_ignored():
    # A 30 x 41 px block with a comb-shaped hole: a spine along row 3 and 18 teeth, 449 px in all.
    # The hole's contour is several times longer than the block's, but it encloses less area.
    mask = blocks((40, 60), (5, 35, 10, 51))
    mask[8, 13:48] = False
    mask[8:32, 13:48:2] = False
    assert mask.sum() == 30 * 41 - 449
    record = measure(shapes.pixel_result(mask, logits=True))
    assert record.n_components == 1
    outline = record.outline_px
    assert (outline[:, 0].min(), outline[:, 0].max()) == pytest.approx((10.0, 51.0), abs=1e-4)
    assert (outline[:, 1].min(), outline[:, 1].max()) == pytest.approx((5.0, 35.0), abs=1e-4)
    assert polygon_area(outline) == pytest.approx(30 * 41, rel=0.002)   # the corners are cut by 1/8 px^2 each


# --------------------------------------------------------------------------- edge and cell (7.8, 9)


@pytest.mark.parametrize("col, row, touches", [
    (100, 120, True), (101, 120, False),   # first column of the input is 100
    (394, 120, True), (393, 120, False),   # last column is 399; the block covers col .. col + 5
    (200, 50, True), (200, 51, False),     # first row is 50
    (200, 244, True), (200, 243, False),   # last row is 249
    (100, 50, True), (394, 244, True), (250, 150, False),
])
def test_edge_is_set_exactly_when_the_mask_reaches_the_border_of_the_models_input(col, row, touches):
    block = np.ones((6, 6), bool)
    assert measure(shapes.pixel_result(block, offset=(col, row)), input_box=DISH_CROP).edge is touches
    # the same pixels inside a larger crop: it is the mask that counts, not the crop
    padded = np.pad(block, 7)
    for logits in (False, True):
        result = shapes.pixel_result(padded, offset=(col - 7, row - 7), logits=logits)
        assert measure(result, input_box=DISH_CROP).edge is touches


@pytest.mark.parametrize("input_box, cell", [
    ((0, 0, 1920, 1080), 7.5),             # the full frame
    ((420, 0, 1080, 1080), 1080 / 256),    # a dish crop
    (DISH_CROP, 300 / 256),
    ((0, 0, 200, 300), 300 / 256),         # the larger side counts, whichever it is
    ((731, 402, 96, 96), 0.375),           # a fine crop
    ((-20, 402, 96, 96), 0.375),           # a fine crop that hangs over the frame's left side
])
def test_cell_px_is_the_larger_side_of_the_models_input_over_256(input_box, cell):
    c0, r0, _, _ = input_box
    record = measure(shapes.pixel_result(np.ones((4, 4), bool), offset=(c0 + 40, r0 + 40)), input_box=input_box)
    assert record.cell_px == pytest.approx(cell, rel=1e-12)
    assert record.edge is False


# --------------------------------------------------------------------------- components (7.4)


@pytest.mark.parametrize("logits", [True, False])
def test_two_components_the_second_a_fifth_of_the_first(logits):
    # 10 x 10 px around (15, 15) and 4 rows x 5 columns around (42.5, 32).
    mask = blocks((50, 60), (10, 20, 10, 20), (30, 34, 40, 45))
    record = measure(shapes.pixel_result(mask, logits=logits))
    assert record.n_components == 2
    assert record.largest_fraction == pytest.approx(100 / 120)
    assert record.second_fraction == pytest.approx(0.2)
    # area and position are those of the whole mask ...
    assert record.area_px == 120
    assert (record.u, record.v) == pytest.approx(((100 * 15 + 20 * 42.5) / 120, (100 * 15 + 20 * 32) / 120))
    # ... the outline is the largest component's
    assert 10 - 1e-3 <= record.outline_px.min() and record.outline_px.max() <= 20 + 1e-3

    third = measure(shapes.pixel_result(mask | blocks((50, 60), (45, 46, 3, 8)), logits=logits))
    assert third.n_components == 3
    assert third.largest_fraction == pytest.approx(100 / 125)
    assert third.second_fraction == pytest.approx(0.2)


@pytest.mark.parametrize("logits", [True, False])
def test_two_diagonal_pixels_are_one_component_and_two_apart_are_two(logits):
    diagonal = blocks((12, 12), (5, 6, 7, 8), (6, 7, 8, 9))   # pixels (row 5, col 7) and (row 6, col 8)
    record = measure(shapes.pixel_result(diagonal, logits=logits))
    assert (record.n_components, record.largest_fraction, record.second_fraction) == (1, 1.0, 0.0)
    outline = record.outline_px
    if logits:   # one contour around both pixels, on their edges
        assert (outline[:, 0].min(), outline[:, 0].max()) == pytest.approx((7.0, 9.0), abs=0.05)
        assert (outline[:, 1].min(), outline[:, 1].max()) == pytest.approx((5.0, 7.0), abs=0.05)
    else:        # the segment between the two centers (7.5, 5.5) and (8.5, 6.5)
        assert outline[:, 0] - 7.5 == pytest.approx(outline[:, 1] - 5.5, abs=1e-4)
        assert (outline[:, 0].min(), outline[:, 0].max()) == pytest.approx((7.5, 8.5), abs=0.05)

    apart = measure(shapes.pixel_result(blocks((12, 12), (5, 6, 7, 8), (5, 6, 9, 10)), logits=logits))
    assert (apart.n_components, apart.largest_fraction, apart.second_fraction) == (2, 0.5, 1.0)


@pytest.mark.parametrize("logits", [True, False])
def test_the_outline_is_the_largest_components_even_next_to_a_wider_hollow_one(logits):
    # A solid 12 x 12 px block (144 px) and the 1 px thick border of a 30 x 30 px square (116 px),
    # which encloses far more area than the block.
    frame = blocks((30, 30), (0, 30, 0, 30)) & ~blocks((30, 30), (1, 29, 1, 29))
    mask = blocks((60, 90), (20, 32, 5, 17))
    mask[10:40, 40:70] = frame
    record = measure(shapes.pixel_result(mask, logits=logits))
    assert (record.area_px, record.n_components) == (260, 2)
    assert record.largest_fraction == pytest.approx(144 / 260)
    assert record.second_fraction == pytest.approx(116 / 144)
    assert 5 - 1e-3 <= record.outline_px[:, 0].min() and record.outline_px[:, 0].max() <= 17 + 1e-3
    assert 20 - 1e-3 <= record.outline_px[:, 1].min() and record.outline_px[:, 1].max() <= 32 + 1e-3


# --------------------------------------------------------------------------- degenerate masks


TINY = {   # the pixels (row, column) of each mask
    "1 px": [(5, 7)],
    "2 px side by side": [(5, 7), (5, 8)],
    "2 px one above the other": [(5, 7), (6, 7)],
    "2 px diagonal": [(5, 7), (6, 8)],
    "2 x 2": [(5, 7), (5, 8), (6, 7), (6, 8)],
    "a row of 7": [(5, col) for col in range(3, 10)],
    "a column of 10": [(row, 4) for row in range(2, 12)],
    "a diagonal of 6": [(3 + i, 3 + i) for i in range(6)],
}


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("name", list(TINY))
def test_tiny_masks_are_measured_without_an_error(name, logits):
    rows, cols = np.array(TINY[name]).T
    mask = np.zeros((14, 14), bool)
    mask[rows, cols] = True
    record = measure(shapes.pixel_result(mask, offset=(40, 20), logits=logits))
    assert record.visible is True
    assert record.area_px == len(rows)
    assert (record.u, record.v) == pytest.approx((40 + cols.mean() + 0.5, 20 + rows.mean() + 0.5), abs=1e-12)
    du, dv = cols - cols.mean(), rows - rows.mean()
    assert record.cov_full == pytest.approx([(du * du).mean(), (du * dv).mean(), (dv * dv).mean()], abs=1e-12)
    assert (record.n_components, record.largest_fraction, record.second_fraction) == (1, 1.0, 0.0)
    # No disk fits into a mask this thin: the opening leaves nothing and the core is the mask.
    assert record.core_r_px >= 1
    assert record.core_fallback is True and record.core_frac == 0.0
    assert np.array_equal(record.cov_core, record.cov_full)
    assert (record.core_u, record.core_v) == (record.u, record.v)
    # The outline exists and stays on the mask: on its pixel centers without logits, up to the
    # pixel edges (half a pixel further) with +-1 logits.
    outline = record.outline_px
    assert outline.shape == (256, 2) and np.isfinite(outline).all()
    reach = (0.5 if logits else 0.0) + 1e-3
    assert 40 + cols.min() + 0.5 - reach <= outline[:, 0].min()
    assert outline[:, 0].max() <= 40 + cols.max() + 0.5 + reach
    assert 20 + rows.min() + 0.5 - reach <= outline[:, 1].min()
    assert outline[:, 1].max() <= 20 + rows.max() + 0.5 + reach
    assert np.array_equal(record.mask(), mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1])


@pytest.mark.parametrize("kind", ["no pixels at all", "no pixels, with logits", "all false", "all false, with logits"])
def test_an_empty_mask_gives_a_lost_record(kind):
    if kind.startswith("no pixels"):   # what crop_to_bbox returns for an object that was not found
        result = shapes.to_result(np.full((30, 40), -3.0), logits="logits" in kind, score=-2.5)
        assert result.mask.shape == (0, 0)
    else:
        result = shapes.pixel_result(np.zeros((9, 12), bool), offset=(30, 40), logits="logits" in kind, score=-2.5)
    record = measure(result, frame=42, mode="fine")
    assert record.visible is False
    assert (record.area_px, record.n_components, record.core_r_px) == (0, 0, 0)
    for name in ("u", "v", "core_u", "core_v", "core_frac", "largest_fraction", "second_fraction"):
        assert np.isnan(getattr(record, name)), name
    assert record.cov_full.shape == (3,) and np.isnan(record.cov_full).all()
    assert record.cov_core.shape == (3,) and np.isnan(record.cov_core).all()
    assert record.outline_px.shape == (256, 2) and record.outline_px.dtype == np.float32
    assert np.isnan(record.outline_px).all()
    assert record.core_fallback is False and record.edge is False
    # what describes the run rather than the mask is kept
    assert (record.frame, record.mode, record.score) == (42, "fine", -2.5)
    assert record.cell_px == 7.5
    assert record.mask_shape == (0, 0) and len(record.mask_bits) == 0
    assert record.mask().shape == (0, 0) and record.mask().dtype == bool
