"""`ThresholdFake` is the template's `DiskFinder` behind the Segmenter protocol (SPEC 12, 13.2).

Its rule, which was that stand-in's in last week's tests: the dark pixels are those whose red
value is below 128; an object's mask is every connected group of them whose center lies within
20 px of the object's last position; the last position then moves to the center of that mask.
The expected values are hand-made images whose boxes, centers and distances are written out here.
The 0.25 px test on the disk clip is in tests/test_fakes.py.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. A box is
(col0, row0, col1, row1): the pixels of the slice [row0:row1, col0:col1], whose center is
((col0 + col1) / 2, (row0 + row1) / 2).
"""

import numpy as np
import pytest
from helpers import assert_empty_result, click, mask_in_image

from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import ObjectPrompt
from outline_tracker.segmenter.fake import ThresholdFake

LIGHT, DARK = 220, 40  # gray levels of the hand-made images: background and objects
SHAPE = (60, 80)  # (rows, columns) of the hand-made images


def picture(*boxes):
    """A light RGB image of 80 x 60 px with a dark box for each (col0, row0, col1, row1)."""
    image = np.full((*SHAPE, 3), LIGHT, np.uint8)
    for col0, row0, col1, row1 in boxes:
        image[row0:row1, col0:col1] = DARK
    return image


def box_mask(*boxes):
    """The pixels of the boxes, a bool array of the hand-made images' shape."""
    return picture(*boxes)[:, :, 0] == DARK


def position(result):
    """Center of a result's mask in the image, px (Tracker's convention)."""
    u, v, _ = mask_center(result.mask)
    return u + result.offset[0], v + result.offset[1]


# ---------------------------------------------------------------------------------------------
# The rule, on hand-made images


def test_mask_is_the_dark_box_cropped_with_8_px_and_signed_distance_logits():
    box = (30, 20, 45, 30)  # columns 30..44, rows 20..29: center (37.5, 25.0)
    (result,) = ThresholdFake().start(picture(box), [click("A", 37.5, 25.0)])
    assert result.obj_id == "A" and result.score is None
    assert result.offset == (22, 12) and result.mask.shape == (26, 31)  # columns 22..52, rows 12..37
    assert result.mask.dtype == bool and np.array_equal(mask_in_image(result, SHAPE), box_mask(box))
    assert position(result) == (37.5, 25.0)
    logits = result.logits
    assert logits.dtype == np.float32 and logits.shape == result.mask.shape
    assert np.array_equal(logits > 0, result.mask)
    # Signed distance to the box's outline in px, positive inside, at pixel centers. Along the
    # rows and columns through the box it is the distance to the nearest edge (u = 30, 45; v = 20, 30).
    at = {(24, 37): 4.5, (20, 30): 0.5, (29, 44): 0.5, (24, 30): 0.5, (24, 29): -0.5, (24, 25): -4.5,
          (15, 37): -4.5, (30, 37): -0.5, (24, 45): -0.5, (24, 52): -7.5}
    for (row, col), distance in at.items():
        assert logits[row - 12, col - 22] == pytest.approx(distance, abs=1e-6), (row, col)


def test_only_the_red_channel_counts():
    image = picture()
    image[10:20, 10:20] = (DARK, LIGHT, LIGHT)  # dark in red only
    image[10:20, 24:34] = (LIGHT, DARK, DARK)   # dark in green and blue: not seen
    (result,) = ThresholdFake().start(image, [click("A", 22.0, 15.0)])  # between the two
    assert np.array_equal(mask_in_image(result, SHAPE), box_mask((10, 10, 20, 20)))


def test_a_pixel_is_dark_when_its_red_value_is_below_128():
    # the same box twice, green and blue light: with red at 127 it is dark, with red at 128 it is not
    box = (30, 20, 45, 30)  # center (37.5, 25.0)
    image = picture()
    image[20:30, 30:45, 0] = 127
    (found,) = ThresholdFake().start(image, [click("A", 37.5, 25.0)])
    assert np.array_equal(mask_in_image(found, SHAPE), box_mask(box))
    image[20:30, 30:45, 0] = 128
    (lost,) = ThresholdFake().start(image, [click("A", 37.5, 25.0)])
    assert_empty_result(lost)


def test_every_dark_group_within_20_px_belongs_to_the_object():
    near, also_near, far = (20, 20, 26, 26), (34, 20, 40, 26), (70, 20, 76, 26)  # 7.5, 6.5 and 42.5 px away
    (result,) = ThresholdFake().start(picture(near, also_near, far), [click("A", 30.5, 23.5)])
    assert np.array_equal(mask_in_image(result, SHAPE), box_mask(near, also_near))
    assert result.offset == (12, 12) and result.mask.shape == (22, 36)  # columns 12..47, rows 12..33
    assert np.array_equal(result.logits > 0, result.mask)


def test_within_20_px_is_counted_as_the_template_does():
    # DiskFinder compares a group's center, counted with pixel centers at whole numbers, with the
    # last position cut down to whole numbers, and wants strictly less than 20. A 3 x 3 box around
    # pixel (column 30, row 30) is then exactly 20 from a click at (10.99, 30.99): lost. One column
    # to the left it is 19 away: found. (Rounding the click, or using it as it is, would find both.)
    at_20, at_19 = (29, 29, 32, 32), (28, 29, 31, 32)
    (lost,) = ThresholdFake().start(picture(at_20), [click("A", 10.99, 30.99)])
    assert_empty_result(lost)
    (found,) = ThresholdFake().start(picture(at_19), [click("A", 10.99, 30.99)])
    assert np.array_equal(mask_in_image(found, SHAPE), box_mask(at_19))


def test_the_last_position_follows_the_mask_and_is_kept_while_the_object_is_lost():
    def box_at(u):  # a 6 x 6 box with its center at (u, 23.0)
        return (int(u) - 3, 20, int(u) + 3, 26)

    fake = ThresholdFake()
    (result,) = fake.start(picture(box_at(20)), [click("A", 20.0, 23.0)])
    assert position(result) == (20.0, 23.0)
    for u in (35, 50):  # 15 px per image: within reach of the last position only (50 is 30 px from the click)
        (result,) = fake.step(picture(box_at(u)))
        assert position(result) == (float(u), 23.0)
    (result,) = fake.step(picture())  # nothing dark
    assert_empty_result(result)
    (result,) = fake.step(picture(box_at(65)))  # 15 px from where it was last found, 45 px from the click
    assert position(result) == (65.0, 23.0)
    (result,) = fake.step(picture(box_at(20)))  # back at the click, 45 px from the last position: lost
    assert_empty_result(result)


BOXES = {"X": (10, 10, 16, 16), "Y": (40, 30, 46, 36), "Z": (64, 46, 70, 52)}  # centers (13, 13), (43, 33), (67, 49)


def test_the_first_positive_point_inside_the_image_is_the_start_position():
    image = picture(*BOXES.values())
    cases = [
        ([(13.0, 13.0), (43.0, 33.0), (67.0, 49.0)], [0, 1, 1], "Y"),   # a negative point is not a position
        ([(-3.0, 13.0), (67.0, 49.0), (43.0, 33.0)], [1, 1, 1], "Z"),   # a point outside the image is dropped
        ([(13.0, 13.0)], [1], "X"),
    ]
    for points, labels, expected in cases:
        (result,) = ThresholdFake().start(image, [ObjectPrompt("A", points, labels)])
        assert np.array_equal(mask_in_image(result, SHAPE), box_mask(BOXES[expected])), expected


def test_results_come_in_the_order_of_the_prompts():
    image = picture(*BOXES.values())
    fake = ThresholdFake()
    first = fake.start(image, [click("late", 67.0, 49.0), click("early", 13.0, 13.0), click("none", 30.0, 55.0)])
    again = fake.step(image)
    for results in (first, again):
        assert [result.obj_id for result in results] == ["late", "early", "none"]
        assert np.array_equal(mask_in_image(results[0], SHAPE), box_mask(BOXES["Z"]))
        assert np.array_equal(mask_in_image(results[1], SHAPE), box_mask(BOXES["X"]))
        assert_empty_result(results[2])  # 26 px from Y, the nearest box


def test_preview_gives_what_start_would_and_leaves_the_run_as_it_is():
    image = picture(*BOXES.values())
    fake = ThresholdFake()
    (before,) = fake.preview(image, [click("P", 43.0, 33.0)])  # no run yet
    assert before.obj_id == "P" and np.array_equal(mask_in_image(before, SHAPE), box_mask(BOXES["Y"]))
    fake.start(image, [click("A", 13.0, 13.0)])
    shown = fake.preview(image, [click("P", 43.0, 33.0), click("Q", 67.0, 49.0)])
    assert [result.obj_id for result in shown] == ["P", "Q"]
    assert np.array_equal(mask_in_image(shown[0], SHAPE), box_mask(BOXES["Y"]))
    assert np.array_equal(shown[0].logits, before.logits) and shown[0].offset == before.offset
    (result,) = fake.step(image)  # still A alone, still on X
    assert result.obj_id == "A" and np.array_equal(mask_in_image(result, SHAPE), box_mask(BOXES["X"]))


def test_an_image_that_is_dark_everywhere_has_finite_positive_logits():
    image = np.full((*SHAPE, 3), DARK, np.uint8)  # one group; its center (39.5, 29.5) is 0.7 px from the click
    (result,) = ThresholdFake().start(image, [click("A", 40.0, 30.0)])
    assert result.offset == (0, 0) and result.mask.shape == SHAPE and result.mask.all()
    assert np.isfinite(result.logits).all() and (result.logits > 0).all()
