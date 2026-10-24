"""The segmenter protocol and its records, and the points inside the image: `segmenter/base.py` (SPEC 12).

Nothing here loads torch: the tests call plain functions and read the records. Tests with the real
processor or the real model are in tests/slow/test_regression_reference.py and
tests/slow/test_real_model.py.

Coordinates: masks are arrays indexed [row, column]; prompt points and offsets are in px of the
image given to the segmenter, Tracker's convention (pixel centers at +0.5, SPEC 3.1).

The rest of the tests of the segmenter without a model are in tests/segmenter/test_hf_helpers.py
(`segmenter/hf.py`: what is kept from last week, loading, the thread), test_hf_helpers_prompts.py
(the prompts for the processor) and test_hf_helpers_fallback.py (the fall back from the Apple GPU).
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from outline_tracker.measure.mask import mask_center


# ---------------------------------------------------------------------------------------------
# base.py: the protocol and its two records, as SPEC 12 writes them


def _fields(cls) -> list[tuple[str, str]]:
    return [(f.name, f.type) for f in dataclasses.fields(cls)]


def test_prompt_and_result_have_the_fields_of_spec_12():
    from outline_tracker.segmenter.base import MaskResult, ObjectPrompt

    assert _fields(ObjectPrompt) == [
        ("obj_id", "str"),
        ("points_px", "list[tuple[float, float]]"),
        ("labels", "list[int]"),
        ("box_px", "tuple[float, float, float, float] | None"),
    ]
    assert ObjectPrompt("A", [(1.0, 2.0)], [1]).box_px is None
    assert _fields(MaskResult) == [
        ("obj_id", "str"),
        ("offset", "tuple[int, int]"),
        ("mask", "np.ndarray"),
        ("logits", "np.ndarray | None"),
        ("score", "float | None"),
    ]


def test_segmenter_protocol_has_start_step_preview_close():
    from outline_tracker.segmenter.base import Segmenter

    assert getattr(Segmenter, "_is_protocol", False)
    parameters = {name: list(inspect.signature(getattr(Segmenter, name)).parameters)
                  for name in ("start", "step", "preview", "close")}
    assert parameters == {
        "start": ["self", "image", "prompts"],
        "step": ["self", "image"],
        "preview": ["self", "image", "prompts"],  # X21
        "close": ["self"],
    }


def _frame_with_block(rows: slice, cols: slice, shape=(300, 400)):
    """A mask with one filled block, and logits that are positive exactly on the block."""
    mask = np.zeros(shape, bool)
    mask[rows, cols] = True
    rng = np.random.default_rng(1)
    logits = np.where(mask, 1.0, -1.0) * rng.uniform(0.5, 9.0, shape)
    return mask, logits.astype(np.float32)


def test_crop_is_the_bounding_box_plus_8_px():
    from outline_tracker.segmenter.base import MaskResult, crop_to_bbox

    mask, logits = _frame_with_block(slice(100, 110), slice(200, 220))  # rows 100-109, columns 200-219
    result = crop_to_bbox(mask, logits)
    assert isinstance(result, MaskResult)
    assert result.offset == (192, 92)  # (col0, row0) = (200 - 8, 100 - 8)
    assert all(type(v) is int for v in result.offset)
    assert result.mask.shape == (10 + 16, 20 + 16)
    assert result.mask.dtype == bool and result.logits.dtype == np.float32
    assert np.array_equal(result.mask, mask[92:118, 192:228])
    assert np.array_equal(result.logits, logits[92:118, 192:228])
    assert np.array_equal(result.logits > 0, result.mask)
    assert (result.obj_id, result.score) == ("", None)

    named = crop_to_bbox(mask, logits, pad=8, obj_id="B", score=3.5)
    assert (named.obj_id, named.score, named.offset) == ("B", 3.5, (192, 92))


def test_crop_pad_is_adjustable_and_clipped_at_the_image_border():
    from outline_tracker.segmenter.base import crop_to_bbox

    mask, logits = _frame_with_block(slice(0, 5), slice(395, 400))  # the top-right corner of 300 x 400
    result = crop_to_bbox(mask, logits)
    assert result.offset == (387, 0)
    assert result.mask.shape == (5 + 8, 5 + 8)
    assert np.array_equal(result.mask, mask[0:13, 387:400])

    tight = crop_to_bbox(mask, logits, pad=0)
    assert tight.offset == (395, 0) and tight.mask.shape == (5, 5) and tight.mask.all()


def test_crop_keeps_the_position_and_restores_the_full_mask():
    from outline_tracker.segmenter.base import crop_to_bbox

    rng = np.random.default_rng(2)
    mask = np.zeros((300, 400), bool)
    mask[120:160, 30:95] = rng.random((40, 65)) < 0.4  # a ragged object
    result = crop_to_bbox(mask, None)
    assert result.logits is None

    col0, row0 = result.offset
    restored = np.zeros_like(mask)
    restored[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    assert np.array_equal(restored, mask)

    # A position is mask_center of the crop plus its offset.
    u, v, area = mask_center(result.mask)
    u_full, v_full, area_full = mask_center(mask)
    assert (u + col0, v + row0) == pytest.approx((u_full, v_full), abs=1e-9)
    assert area == area_full == int(mask.sum())


def test_crop_does_not_keep_the_full_frame_alive():
    # SPEC 6.5: no full-frame float array survives the current frame, so a crop owns its data.
    from outline_tracker.segmenter.base import crop_to_bbox

    mask, logits = _frame_with_block(slice(100, 110), slice(200, 220))
    result = crop_to_bbox(mask, logits)
    assert result.mask.base is None and result.logits.base is None
    assert not np.shares_memory(result.logits, logits) and not np.shares_memory(result.mask, mask)


def test_an_empty_mask_gives_an_empty_result():
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros((300, 400), bool)
    result = crop_to_bbox(mask, np.full(mask.shape, -1024.0, np.float32), obj_id="A", score=-2.0)
    assert result.offset == (0, 0)
    assert result.mask.shape == (0, 0) and result.mask.dtype == bool
    assert result.logits.shape == (0, 0) and result.logits.dtype == np.float32
    assert (result.obj_id, result.score) == ("A", -2.0)
    u, v, area = mask_center(result.mask)
    assert np.isnan(u) and np.isnan(v) and area == 0
    assert crop_to_bbox(mask, None).logits is None


# ---------------------------------------------------------------------------------------------
# base.py: points outside the image are dropped, and an object needs a positive point


def test_points_outside_the_image_are_dropped():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    # A 1920 x 1080 image covers 0 <= u < 1920 and 0 <= v < 1080: pixel c is the square [c, c + 1) (SPEC 3.1).
    points = [(0.0, 0.0), (-0.01, 500.0), (1920.0, 500.0), (1919.99, 1079.99), (700.0, -0.01), (700.0, 1080.0),
              (700.5, 500.5), (-10.0, -10.0), (float("nan"), 500.0), (700.0, float("inf"))]
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    prompt = ObjectPrompt("A", list(points), list(labels), box_px=(1.0, 2.0, 3.0, 4.0))

    kept = base.points_in_image(prompt, height=1080, width=1920)

    assert kept.points_px == [(0.0, 0.0), (1919.99, 1079.99), (700.5, 500.5)]
    assert kept.labels == [1, 0, 1]  # every label stays with its point
    assert (kept.obj_id, kept.box_px) == ("A", (1.0, 2.0, 3.0, 4.0))
    assert (prompt.points_px, prompt.labels) == (points, labels)  # the caller's prompt is not changed

    # "The image" is the one given to the model: in a 96 px crop the same points are judged anew.
    small = ObjectPrompt("A", [(95.5, 0.5), (96.0, 0.5), (48.0, 95.99), (48.0, 96.0)], [1, 0, 0, 1])
    assert base.points_in_image(small, height=96, width=96).points_px == [(95.5, 0.5), (48.0, 95.99)]


def test_an_object_left_without_a_positive_point_is_refused():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    with pytest.raises(base.PromptError) as caught:  # the positive point is outside, the negative one inside
        base.points_in_image(ObjectPrompt("B2", [(2000.5, 500.5), (700.5, 500.5)], [1, 0]), height=1080, width=1920)
    message = str(caught.value)
    assert "'B2'" in message and "no positive point inside the image" in message
    assert "1920 x 1080 px" in message
    assert isinstance(caught.value, ValueError)
    with pytest.raises(base.PromptError, match="no positive point inside the image"):
        base.points_in_image(ObjectPrompt("A", [], []), height=1080, width=1920)
    with pytest.raises(base.PromptError, match="no positive point inside the image"):
        base.points_in_image(ObjectPrompt("A", [(700.5, 500.5)], [0]), height=1080, width=1920)


def test_points_and_labels_that_do_not_fit_are_refused():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    with pytest.raises(base.PromptError, match="'A' has 2 points and 1 labels"):
        base.points_in_image(ObjectPrompt("A", [(700.5, 500.5), (710.5, 500.5)], [1]), height=1080, width=1920)
    # To the model, 2 and 3 are the corners of a box and negative labels are padding.
    for label in (2, 3, -1, -10):
        with pytest.raises(base.PromptError, match="labels must be 1 .positive. or 0 .negative."):
            base.points_in_image(ObjectPrompt("A", [(700.5, 500.5), (710.5, 500.5)], [1, label]), height=1080,
                               width=1920)
