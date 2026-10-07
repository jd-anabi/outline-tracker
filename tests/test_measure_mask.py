"""Tests for outline_tracker.measure: measure_mask and PixelRecord, the record itself (SPEC 7.1,
7.2, 7.8, 8.12, 9 EDGE).

What a record holds and how it is stored, the arguments measure_mask refuses, the border of the
model's input (`edge`, `cell_px`), and masks too small or too empty to measure. The second moments
and the core are in tests/test_measure_core.py, the outline and the components in
tests/test_measure_outline.py, the ported mask_center in tests/test_measure.py.

Every shape is an implicit function rasterized at the pixel centers (tests/analytic_shapes.py),
and every expected value below comes from the shape's geometry or from counting pixels by hand.
A result given to measure_mask is in full-frame pixels; `input_box` = (c0, r0, width, height)
is the image the model saw, in full-frame pixels.
"""

import analytic_shapes as shapes
import numpy as np
import pytest

from outline_tracker import schema

# These tests were written next to the ported test of tests/test_measure.py, which calls the
# module `segment` (the template's name), and were moved here unchanged: the name is kept.
from outline_tracker import measure as segment

FULL_HD = (0, 0, 1920, 1080)
DISH_CROP = (100, 50, 300, 200)  # columns 100..399 and rows 50..249 of the full frame


def measure(result, frame=0, input_box=FULL_HD, mode="coarse", **settings):
    return segment.measure_mask(result, frame, input_box, mode, **settings)


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
