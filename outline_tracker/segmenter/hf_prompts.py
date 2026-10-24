"""The prompts of a run for the Hugging Face processor: one object's points, and all objects put into a session.

`hf.py` calls `add_prompts` when a run starts or a fall back from the Apple GPU begins a new session. Prompt points
are px in the pixel frame of the image given to the model, Tracker's convention (pixel centers at +0.5, SPEC 3.1),
and go to the model unchanged, as last week. Nothing here imports torch or transformers.
"""

from __future__ import annotations

from outline_tracker.segmenter.base import ObjectPrompt


def prompt_lists(prompt: ObjectPrompt) -> tuple[list, list]:
    """One object's points and labels in the nesting the processor expects.

    Returns (input_points, input_labels) = ([[[[x, y], ...]]], [[[label, ...]]]): the levels are
    [image][object][point], with one image and one object. x and y are px in the pixel frame of
    the image given to the model (SPEC 3.1), unchanged; a label is 1 (positive) or 0 (negative).
    The values are Python floats and ints: the processor refuses numpy's float32 and int64 scalars.
    """
    points = [[float(x), float(y)] for x, y in prompt.points_px]
    labels = [int(label) for label in prompt.labels]
    return [[points]], [[labels]]


def add_prompts(processor, session, prompts: list[ObjectPrompt], original_size) -> list[int]:
    """Put the prompts of a run's first frame into a new session, with one call per object.

    Returns the integers 1..N by which the session knows the objects, in the order of `prompts`
    (as last week). Points are px in the pixel frame of the image; `original_size` is the image's
    (height, width) in px, as the processor returned it. Objects are never put into one call: the
    processor would pad those with fewer points. Each call replaces the session's list of objects
    with new inputs, so the list is set to all objects at the end, in a list of its own: the model
    removes entries from it while it runs.
    """
    ids = list(range(1, len(prompts) + 1))
    for obj, prompt in zip(ids, prompts):
        points, labels = prompt_lists(prompt)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=[obj], input_points=points, input_labels=labels,
            original_size=original_size)
    session.obj_with_new_inputs = list(ids)
    return ids
