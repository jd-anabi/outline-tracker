"""Tests for outline_tracker.measure: the outline and the components of a mask (SPEC 7.4).

Every shape is an implicit function rasterized at the pixel centers (tests/analytic_shapes.py),
and every expected value below comes from the shape's geometry or from counting pixels by hand.
A result given to measure_mask is in full-frame pixels; `input_box` = (c0, r0, width, height)
is the image the model saw, in full-frame pixels. Outline points are (u, v) in px, pixel centers
at +0.5.
"""

import analytic_shapes as shapes
import numpy as np
import pytest
from analytic_shapes import blocks

# These tests were written next to the ported test of tests/test_measure.py, which calls the
# module `segment` (the template's name), and were moved here unchanged: the name is kept.
from outline_tracker import measure as segment

FULL_HD = (0, 0, 1920, 1080)
DISH_CROP = (100, 50, 300, 200)  # columns 100..399 and rows 50..249 of the full frame
CENTERS = [(0.5, 0.5), (0.0, 0.0), (0.37, -0.19)]  # sub-pixel positions of a shape's center


def measure(result, frame=0, input_box=FULL_HD, mode="coarse", **settings):
    return segment.measure_mask(result, frame, input_box, mode, **settings)


def spacing(points):
    """Distances from each point of a closed polygon to the next (the last one back to the first)."""
    points = np.asarray(points, float)
    return np.hypot(*(np.roll(points, -1, axis=0) - points).T)


def perimeter(points):
    return float(spacing(points).sum())


def polygon_area(points):
    x, y = np.asarray(points, float).T
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


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


# --------------------------------------------------------------------------- largest_piece


def around(row0, row1, col0, col1, *, on_edges):
    """An outline around the pixel rectangle (row0, row1, col0, col1) of `blocks`: 40 points (u, v)
    in px in the array's frame, 10 per side. With `on_edges` it runs along the rectangle's outer
    pixel edges, as the level set of logits of +1 and -1 does, else through the centers of its
    border pixels, as an outline without logits does."""
    inset = 0.0 if on_edges else 0.5
    left, right, top, bottom = col0 + inset, col1 - inset, row0 + inset, row1 - inset
    corners = np.array([(left, top), (right, top), (right, bottom), (left, bottom), (left, top)])
    steps = np.linspace(0.0, 1.0, 10, endpoint=False)[:, None]
    return np.concatenate([a + steps * (b - a) for a, b in zip(corners[:-1], corners[1:], strict=True)])


def test_largest_piece_is_the_piece_with_the_most_pixels_and_counts_every_piece():
    large, small, line = (10, 20, 10, 20), (30, 34, 40, 45), (45, 46, 3, 8)
    piece, sizes = segment.largest_piece(blocks((50, 60), large, small, line))
    assert piece.dtype == bool and np.array_equal(piece, blocks((50, 60), large))
    assert sorted(int(size) for size in sizes) == [5, 20, 100]


# Three blocks of 2 rows x 3 columns as (row0, row1, col0, col1). One free column lies between the
# first and the second, one free row between the second and the third: closer, and they would touch.
EQUAL_BLOCKS = [(1, 3, 1, 4), (2, 4, 5, 8), (5, 7, 2, 5)]


@pytest.mark.parametrize("on_edges", [True, False])
@pytest.mark.parametrize("taken", EQUAL_BLOCKS)
def test_of_equal_pieces_largest_piece_takes_the_one_that_the_outline_runs_along(taken, on_edges):
    piece, sizes = segment.largest_piece(blocks((8, 9), *EQUAL_BLOCKS), around(*taken, on_edges=on_edges))
    assert np.array_equal(piece, blocks((8, 9), taken))
    assert [int(size) for size in sizes] == [6, 6, 6]


@pytest.mark.parametrize("on_edges", [True, False])
def test_an_outline_does_not_make_a_smaller_piece_the_largest(on_edges):
    large, small = (1, 4, 1, 4), (1, 3, 6, 8)   # 9 px and 4 px
    mask = blocks((6, 10), large, small)
    piece, _ = segment.largest_piece(mask, around(*small, on_edges=on_edges))
    assert np.array_equal(piece, blocks((6, 10), large))


@pytest.mark.parametrize("outline", [np.full((256, 2), np.nan), np.full((256, 2), 500.0), np.zeros((0, 2)),
                                     np.array([[4.5, 0.5], [4.5, 5.5]])])
def test_an_outline_along_none_of_the_equal_pieces_leaves_the_piece_that_is_taken_without_one(outline):
    # No number (a lost frame), far away, no point, and two points in the free columns between
    # the pieces: nothing says which piece, so it is the one that measure_mask takes.
    mask = blocks((6, 9), (1, 3, 1, 3), (2, 4, 6, 8))
    piece, sizes = segment.largest_piece(mask, outline)
    without, _ = segment.largest_piece(mask)
    assert np.array_equal(piece, without) and piece.sum() == 4
    assert [int(size) for size in sizes] == [4, 4]
