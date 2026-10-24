"""The fall back from the Apple GPU in `step()` and `preview()`, and the model lock: `segmenter/hf.py` (SPEC 6.2).

Nothing here loads torch: `HFSegmenter._infer`, the one method that touches torch, is replaced by a
stand-in that returns the results of a small scene, with a known center for every object on every
frame. Tests with the real processor or the real model are in tests/slow/test_regression_reference.py
and tests/slow/test_real_model.py.

Coordinates: frames are RGB uint8 arrays indexed [row, column]; prompt points and centers are in px
of the frame, Tracker's convention (pixel centers at +0.5, SPEC 3.1).

The rest of the tests of the segmenter without a model are in tests/segmenter/test_hf_helpers.py
(`segmenter/hf.py`: what is kept from last week, loading, the thread), test_hf_helpers_base.py
(the protocol, the records, the points inside the image) and test_hf_helpers_prompts.py (the prompts
for the processor).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from outline_tracker.measure.mask import mask_center


# ---------------------------------------------------------------------------------------------
# hf.py: bookkeeping around the model call (fall back in step, preview, the lock), with a stand-in
# for `HFSegmenter._infer`, the one method that touches torch

FRAME_SHAPE = (120, 160, 3)  # rows, columns, channels of the stand-in frames


def _frame(t: int) -> np.ndarray:
    """Stand-in frame number t: an image whose every value is t."""
    return np.full(FRAME_SHAPE, t, np.uint8)


def _block(name: str, cols: tuple[int, int], rows: tuple[int, int]):
    """An object that fills columns cols[0]..cols[1] and rows rows[0]..rows[1], both inclusive.
    Its center is ((cols[0] + cols[1] + 1) / 2, (rows[0] + rows[1] + 1) / 2) px (SPEC 3.1)."""
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros(FRAME_SHAPE[:2], bool)
    mask[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1] = True
    return crop_to_bbox(mask, np.where(mask, 3.0, -3.0).astype(np.float32), obj_id=name, score=4.0)


def _not_found(name: str):
    """What the model returns for an object it calls absent: all logits -1024, so an empty mask."""
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros(FRAME_SHAPE[:2], bool)
    return crop_to_bbox(mask, np.full(mask.shape, -1024.0, np.float32), obj_id=name, score=-5.0)


def _scene(name: str, t: int):
    """The stand-in scene on frame t. Centers in px:
    A moves right, (25 + 10 t, 35); B is found on frame 0 only, at (105, 52); C is never found;
    D moves down, (65, 85 + 5 t)."""
    if name == "A":
        return _block("A", (20 + 10 * t, 29 + 10 * t), (30, 39))
    if name == "B" and t == 0:
        return _block("B", (100, 109), (50, 53))
    if name == "D":
        return _block("D", (60, 69), (80 + 5 * t, 89 + 5 * t))
    return _not_found(name)


def _scene_prompts(names: str = "ABCD"):
    from outline_tracker.segmenter.base import ObjectPrompt

    clicks = {"A": (25.0, 35.0), "B": (105.0, 52.0), "C": (140.0, 100.0), "D": (65.0, 85.0)}
    return [ObjectPrompt(name, [clicks[name]], [1]) for name in names]


def _center(result):
    """(u, v) of a result in px of the frame, or None if the object was not found."""
    u, v, area = mask_center(result.mask)
    return None if area == 0 else (u + result.offset[0], v + result.offset[1])


class InferStub:
    """Stands in for `HFSegmenter._infer`: returns the scene's results for the objects of the
    session, like the model one result per object in the order of the prompts, and raises `error`
    on frame `fail_on_frame` while the segmenter's device is `fail_device`. Records every call."""

    def __init__(self, segmenter, fail_on_frame=None, fail_device="mps", error=None):
        self.segmenter, self.fail_on_frame, self.fail_device = segmenter, fail_on_frame, fail_device
        self.error = error or NotImplementedError(
            "The operator 'aten::x' is not currently implemented for the MPS device.")
        self.calls, self.objects = [], []

    def __call__(self, rgb, prompts=None, keep=True):
        from outline_tracker.segmenter import hf

        t = int(rgb[0, 0, 0])
        self.calls.append(SimpleNamespace(device=self.segmenter.device, frame=t, prompts=prompts, keep=keep,
                                          locked=hf.MODEL_LOCK.locked()))
        if t == self.fail_on_frame and self.segmenter.device == self.fail_device:
            raise self.error
        objects = self.objects if prompts is None else [prompt.obj_id for prompt in prompts]
        if keep:
            self.objects = objects
        return [_scene(name, t) for name in objects]


def _stub_segmenter(device: str, **failure):
    """An `HFSegmenter` whose constructor did not run (no torch): a device, a model that records
    where it is moved, a log that collects lines, and the stand-in for `_infer`."""
    from outline_tracker.segmenter import hf

    segmenter = object.__new__(hf.HFSegmenter)
    segmenter.device = device
    segmenter.moves, segmenter.lines = [], []
    segmenter.model = SimpleNamespace(to=lambda where: segmenter.moves.append((where, hf.MODEL_LOCK.locked())))
    segmenter.log = segmenter.lines.append
    segmenter._infer = InferStub(segmenter, **failure)
    return segmenter


def test_step_falls_back_from_the_apple_gpu_and_reseeds_every_object(capsys):
    segmenter = _stub_segmenter("mps", fail_on_frame=2)  # the model call fails on the third frame

    first = segmenter.start(_frame(0), _scene_prompts())
    second = segmenter.step(_frame(1))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    third = segmenter.step(_frame(2))
    fourth = segmenter.step(_frame(3))

    # It moved to cpu once, opened a new session on the frame that failed, and went on from there.
    assert segmenter.device == "cpu"
    assert segmenter.moves == [("cpu", True)]  # with the model lock held
    calls = segmenter._infer.calls
    assert [(call.device, call.frame, call.prompts is not None) for call in calls] == [
        ("mps", 0, True), ("mps", 1, False), ("mps", 2, False), ("cpu", 2, True), ("cpu", 3, False)]
    assert all(call.keep and call.locked for call in calls)

    # The new session has one positive click per object, at the center where it was last found.
    seeds = calls[3].prompts
    assert [(seed.obj_id, seed.labels, len(seed.points_px)) for seed in seeds] == [
        ("A", [1], 1), ("B", [1], 1), ("D", [1], 1)]  # C was never found: no click
    assert seeds[0].points_px[0] == pytest.approx((35.0, 35.0))  # A on frame 1
    assert seeds[1].points_px[0] == pytest.approx((105.0, 52.0))  # B was not found on frame 1: frame 0
    assert seeds[2].points_px[0] == pytest.approx((65.0, 90.0))  # D on frame 1
    assert all(type(value) is float for seed in seeds for value in seed.points_px[0])

    # Every frame still gives one result per object, in the order of the prompts.
    for results in (first, second, third, fourth):
        assert [result.obj_id for result in results] == ["A", "B", "C", "D"]
    assert [_center(result) for result in first] == [(25.0, 35.0), (105.0, 52.0), None, (65.0, 85.0)]
    assert [_center(result) for result in second] == [(35.0, 35.0), None, None, (65.0, 90.0)]
    assert [_center(result) for result in third] == [(45.0, 35.0), None, None, (65.0, 95.0)]
    assert [_center(result) for result in fourth] == [(55.0, 35.0), None, None, (65.0, 100.0)]
    lost = third[2]  # C is not in the new session: an empty result without a score
    assert lost.mask.shape == (0, 0) and lost.mask.dtype == bool
    assert lost.logits.shape == (0, 0) and lost.logits.dtype == np.float32
    assert lost.offset == (0, 0) and lost.score is None

    # It is logged, once, through the segmenter's `log`; nothing is printed besides.
    (line,) = segmenter.lines
    assert "The Apple GPU failed (NotImplementedError" in line and "aten::x" in line
    assert "processor" in line and "A, B, D" in line and "C" in line.split("A, B, D")[1]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("error", [
    NotImplementedError("The operator 'aten::x' is not currently implemented for the MPS device."),
    RuntimeError("Expected all tensors to be on the same device"),
    TypeError("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64."),
], ids=lambda error: type(error).__name__)
def test_step_falls_back_on_a_runtime_error_or_a_type_error(error):
    segmenter = _stub_segmenter("mps", fail_on_frame=1, error=error)
    segmenter.start(_frame(0), _scene_prompts("AD"))
    results = segmenter.step(_frame(1))
    assert segmenter.device == "cpu" and segmenter.moves == [("cpu", True)]
    assert [_center(result) for result in results] == [(35.0, 35.0), (65.0, 90.0)]
    assert type(error).__name__ in segmenter.lines[0]


def test_step_does_not_hide_an_error_on_other_devices_or_of_another_kind():
    segmenter = _stub_segmenter("cpu", fail_on_frame=1, fail_device="cpu", error=RuntimeError("out of memory"))
    segmenter.start(_frame(0), _scene_prompts())
    with pytest.raises(RuntimeError, match="out of memory"):
        segmenter.step(_frame(1))
    assert segmenter.device == "cpu" and segmenter.moves == [] and segmenter.lines == []

    segmenter = _stub_segmenter("mps", fail_on_frame=1, error=KeyError("frame"))  # not what a failing GPU raises
    segmenter.start(_frame(0), _scene_prompts())
    with pytest.raises(KeyError):
        segmenter.step(_frame(1))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []


def test_step_after_a_fall_back_with_no_object_ever_found_calls_no_model():
    segmenter = _stub_segmenter("mps", fail_on_frame=1)
    segmenter.start(_frame(0), _scene_prompts("C"))
    for t in (1, 2):
        (result,) = segmenter.step(_frame(t))
        assert result.obj_id == "C" and result.mask.shape == (0, 0)
    assert segmenter.device == "cpu"
    # No click can be placed, so no new session is opened and the model is not called again.
    assert [(call.device, call.frame) for call in segmenter._infer.calls] == [("mps", 0), ("mps", 1)]
    assert len(segmenter.lines) == 1


def test_a_prompt_error_is_raised_before_the_model_is_called(capsys):
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    outside = ObjectPrompt("B", [(160.0, 52.0), (105.0, 52.0)], [1, 0])  # the frames are 160 px wide
    for call in ("start", "preview"):
        segmenter = _stub_segmenter("mps")
        with pytest.raises(base.PromptError, match="'B' has no positive point inside the image"):
            getattr(segmenter, call)(_frame(0), [*_scene_prompts("A"), outside])
        assert segmenter._infer.calls == []
        # A click in the wrong place is not a failure of the Apple GPU.
        assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    assert capsys.readouterr().out == ""


def test_start_and_preview_give_the_model_only_the_points_inside_the_image():
    from outline_tracker.segmenter.base import ObjectPrompt

    prompts = [ObjectPrompt("A", [(25.0, 35.0), (25.0, 120.0), (-3.0, 35.0)], [1, 1, 0]),  # the frames have 120 rows
               ObjectPrompt("D", [(65.0, 85.0), (70.0, 119.5)], [1, 0])]
    for call in ("start", "preview"):
        segmenter = _stub_segmenter("cpu")
        results = getattr(segmenter, call)(_frame(0), prompts)
        (made,) = segmenter._infer.calls
        assert [(p.obj_id, p.points_px, p.labels) for p in made.prompts] == [
            ("A", [(25.0, 35.0)], [1]), ("D", [(65.0, 85.0), (70.0, 119.5)], [1, 0])]
        assert made.locked and made.keep is (call == "start")
        assert [_center(result) for result in results] == [(25.0, 35.0), (65.0, 85.0)]


def test_preview_keeps_no_tracking_state():
    from outline_tracker.segmenter.base import ObjectPrompt

    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    segmenter.start(_frame(0), _scene_prompts("AD"))

    # A preview of another object, on another frame, in the middle of the run.
    (shown,) = segmenter.preview(_frame(0), [ObjectPrompt("B", [(105.0, 52.0)], [1])])
    assert shown.obj_id == "B" and _center(shown) == (105.0, 52.0)
    assert segmenter._infer.calls[-1].keep is False  # a session of its own, thrown away

    second = segmenter.step(_frame(1))
    third = segmenter.step(_frame(2))  # the fall back re-seeds the objects of the run, not the previewed one
    assert [seed.obj_id for seed in segmenter._infer.calls[-1].prompts] == ["A", "D"]
    assert [result.obj_id for result in second] == [result.obj_id for result in third] == ["A", "D"]
    assert [_center(result) for result in third] == [(45.0, 35.0), (65.0, 95.0)]


def test_preview_falls_back_from_the_apple_gpu_between_runs_only():
    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    (shown,) = segmenter.preview(_frame(2), _scene_prompts("A"))  # no run is active
    assert _center(shown) == (45.0, 35.0)
    assert segmenter.device == "cpu" and segmenter.moves == [("cpu", True)]
    assert [(call.device, call.keep) for call in segmenter._infer.calls] == [("mps", False), ("cpu", False)]
    assert len(segmenter.lines) == 1 and "The Apple GPU failed (NotImplementedError" in segmenter.lines[0]

    # During a run the session lives on the Apple GPU: the preview fails and the run is left alone.
    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    segmenter.session = object()  # what `_infer` leaves behind when a run starts
    segmenter.start(_frame(0), _scene_prompts("A"))
    with pytest.raises(NotImplementedError):
        segmenter.preview(_frame(2), _scene_prompts("A"))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    assert [_center(result) for result in segmenter.step(_frame(1))] == [(35.0, 35.0)]


def test_start_moves_the_model_with_the_lock_held():
    from outline_tracker.segmenter import hf

    held = []

    def forward(image, prompts=None):
        if segmenter.device == "mps":
            raise NotImplementedError("not implemented for the MPS device")
        return ["masks"]

    model = SimpleNamespace(to=lambda device: held.append((device, hf.MODEL_LOCK.locked())))
    segmenter = SimpleNamespace(device="mps", model=model, _forward=forward)
    assert hf.HFSegmenter.start(segmenter, "image", ["prompt"]) == ["masks"]
    assert held == [("cpu", True)]
    assert not hf.MODEL_LOCK.locked()
