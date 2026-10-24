"""The prompts that go to the processor: `prompt_lists` and `add_prompts` of `segmenter/hf_prompts.py` (SPEC 6.2).

Nothing here loads torch: the processor is replaced by a recorder that keeps every call. Tests with
the real processor or the real model are in tests/slow/test_regression_reference.py and
tests/slow/test_real_model.py.

Coordinates: prompt points are in px of the image given to the segmenter, Tracker's convention
(pixel centers at +0.5, SPEC 3.1).

The rest of the tests of the segmenter without a model are in tests/segmenter/test_hf_helpers.py
(`segmenter/hf.py`: what is kept from last week, loading, the thread), test_hf_helpers_base.py
(the protocol, the records, the points inside the image) and test_hf_helpers_fallback.py (the fall
back from the Apple GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


# ---------------------------------------------------------------------------------------------
# hf_prompts.py: prompts, one add-prompt call per object


def test_prompt_lists_have_the_nesting_image_object_point():
    from outline_tracker.segmenter import hf_prompts
    from outline_tracker.segmenter.base import ObjectPrompt

    # numpy scalars must not reach the processor: np.float32 and np.int64 raise TypeError there.
    prompt = ObjectPrompt("A", [(np.float32(700.5), np.float64(500.5))], [np.int64(1)])
    points, labels = hf_prompts.prompt_lists(prompt)
    assert points == [[[[700.5, 500.5]]]]  # [image][object][point][x, y]
    assert labels == [[[1]]]  # [image][object][point]
    x, y = points[0][0][0]
    assert type(x) is float and type(y) is float
    assert type(labels[0][0][0]) is int


class RecordingProcessor:
    """Stands in for Sam2VideoProcessor: records every add-prompt call and, like the real one
    (`inference_session.obj_with_new_inputs = obj_ids`), overwrites the session's list each time."""

    def __init__(self):
        self.calls = []

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, input_points=None,
                                        input_labels=None, original_size=None):
        self.calls.append({"frame_idx": frame_idx, "obj_ids": obj_ids, "input_points": input_points,
                           "input_labels": input_labels, "original_size": original_size})
        inference_session.obj_with_new_inputs = obj_ids


def test_prompts_are_added_with_one_call_per_object():
    from outline_tracker.segmenter import hf_prompts
    from outline_tracker.segmenter.base import ObjectPrompt

    clicks = [(700.5, 500.5), (1000.5, 600.5), (1400.5, 350.5)]
    prompts = [ObjectPrompt(name, [click], [1]) for name, click in zip("ABC", clicks)]
    processor, session, size = RecordingProcessor(), SimpleNamespace(obj_with_new_inputs=[]), object()

    ids = hf_prompts.add_prompts(processor, session, prompts, size)

    assert ids == [1, 2, 3]  # integers 1..N, as last week
    assert len(processor.calls) == 3
    for k, (call, (x, y)) in enumerate(zip(processor.calls, clicks)):
        assert call["frame_idx"] == 0
        assert call["obj_ids"] == [k + 1]
        assert call["input_points"] == [[[[x, y]]]]
        assert call["input_labels"] == [[[1]]]
        assert call["original_size"] is size
    # Each call overwrote the list of objects with new inputs: afterwards it names all of them,
    # in a list of its own (the model removes entries from it while it runs).
    assert session.obj_with_new_inputs == [1, 2, 3]
    assert session.obj_with_new_inputs is not ids
    assert all(session.obj_with_new_inputs is not call["obj_ids"] for call in processor.calls)


# ---------------------------------------------------------------------------------------------
# Several points per object, negative points (hf_prompts.py)


def _objects_with_1_3_and_2_points():
    """Three objects in a 1920 x 1080 image; labels: 1 positive, 0 negative."""
    from outline_tracker.segmenter.base import ObjectPrompt

    return [
        ObjectPrompt("A", [(700.5, 500.5)], [1]),
        ObjectPrompt("B", [(1000.5, 600.5), (1030.5, 640.5), (1010.0, 590.25)], [1, 0, 1]),
        ObjectPrompt("C", [(1400.5, 350.5), (1380.5, 360.5)], [1, 0]),
    ]


def test_prompt_lists_for_objects_with_1_3_and_2_points():
    from outline_tracker.segmenter import hf_prompts

    lists = [hf_prompts.prompt_lists(prompt) for prompt in _objects_with_1_3_and_2_points()]
    assert lists == [
        ([[[[700.5, 500.5]]]], [[[1]]]),
        ([[[[1000.5, 600.5], [1030.5, 640.5], [1010.0, 590.25]]]], [[[1, 0, 1]]]),
        ([[[[1400.5, 350.5], [1380.5, 360.5]]]], [[[1, 0]]]),
    ]
    for points, labels in lists:  # [image][object]: one image and one object in every pair of lists
        assert len(points) == len(labels) == 1 and len(points[0]) == len(labels[0]) == 1
        assert all(type(value) is float for point in points[0][0] for value in point)
        assert all(type(label) is int for label in labels[0][0])


def test_objects_with_different_point_counts_are_never_put_into_one_call():
    from outline_tracker.segmenter import hf_prompts

    processor, session = RecordingProcessor(), SimpleNamespace(obj_with_new_inputs=[])
    ids = hf_prompts.add_prompts(processor, session, _objects_with_1_3_and_2_points(), (1080, 1920))

    assert ids == [1, 2, 3]
    assert [call["obj_ids"] for call in processor.calls] == [[1], [2], [3]]
    # One object per call, so the processor has nothing to pad (it pads unequal counts with -10).
    assert [len(call["input_points"][0]) for call in processor.calls] == [1, 1, 1]
    assert [len(call["input_points"][0][0]) for call in processor.calls] == [1, 3, 2]
    assert [call["input_labels"] for call in processor.calls] == [[[[1]]], [[[1, 0, 1]]], [[[1, 0]]]]
    assert session.obj_with_new_inputs == [1, 2, 3]
