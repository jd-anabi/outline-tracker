"""The backend's extensions with the real processor and a stand-in for the network (SPEC 5, 6.2): several points per
object, negative points, `preview`, the fall back in `step`. Every test here is slow: it needs torch, and none needs
the weights of the model (seconds).

The processor and the session are the real ones; the network is replaced by a stand-in that returns one disk per
object and can fail once, like an Apple GPU that lacks an operation. The tests with the real EdgeTAM (the negative
prompt of SPEC 13.4, `preview` against the first frame of a run, the fall back with the real model) are in
tests/slow/test_real_model.py.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full 1920 x 1080 frame; the model
works on a grid of 256 x 256 cells.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from layout_helpers import REPO
from pipeline_helpers import HD_H, HD_W, NAMES, _stored, position

from outline_tracker.segmenter.base import ObjectPrompt

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------------------------
# No weights: the real processor and session, a stand-in for the network


class StubModel:
    """Stands in for the network. Object number j of the session (0, 1, ...) is a disk of radius
    6 cells on the model's 256 x 256 grid, centered on cell (column 60 + 50 j + 2 n, row 100) at
    the stand-in's call number n, so its center in the image is ((60.5 + 50 j + 2 n) HD_W / 256,
    100.5 HD_H / 256) px. `absent` lists the (n, j) for which the object is reported absent, as the
    model does it: all logits -1024. On call number `fail_on_call` it raises what an Apple GPU
    raises for a missing operation, if the frame it was given lives on mps."""

    def __init__(self, fail_on_call=None, absent=()):
        self.fail_on_call, self.absent = fail_on_call, set(absent)
        self.calls, self.moved = [], []

    def eval(self):
        return self

    def to(self, device):
        self.moved.append(str(device))
        return self

    def __call__(self, inference_session, frame_idx, frame):
        import torch

        n = len(self.calls)
        self.calls.append(SimpleNamespace(session=inference_session, frame_idx=frame_idx, device=frame.device.type))
        if n == self.fail_on_call and frame.device.type == "mps":
            raise NotImplementedError("The operator 'aten::stand_in' is not currently implemented for the MPS device.")
        rows, cols = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
        masks, scores = [], []
        for j in range(len(inference_session.obj_ids)):
            found = (n, j) not in self.absent
            disk = 6.0 - torch.hypot(cols - (60.0 + 50.0 * j + 2.0 * n), rows - 100.0)
            masks.append(disk if found else torch.full((256, 256), -1024.0))
            scores.append(5.0 if found else -3.0)
        return SimpleNamespace(object_ids=list(inference_session.obj_ids), frame_idx=frame_idx,
                               pred_masks=torch.stack(masks)[:, None].to(frame.device),
                               object_score_logits=torch.tensor(scores, device=frame.device)[:, None])


def _disk_center(n: int, j: int) -> tuple[float, float]:
    """Center, in px of the image, of the stand-in's disk for object j at call n."""
    return (60.5 + 50.0 * j + 2.0 * n) * HD_W / 256, 100.5 * HD_H / 256


THREE = [  # objects with 1, 3 and 2 points, px in the 1920 x 1080 image
    ObjectPrompt("A", [(700.5, 500.5)], [1]),
    ObjectPrompt("B", [(1000.5, 600.5), (1030.5, 640.5), (1010.0, 590.25)], [1, 0, 1]),
    ObjectPrompt("C", [(1400.5, 350.5), (1380.5, 360.5)], [1, 0]),
]


def test_prompt_tensors_for_1_3_and_2_points_have_no_padding(processor):
    import torch

    from outline_tracker.segmenter import hf_prompts

    session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    ids = hf_prompts.add_prompts(processor, session, THREE, torch.tensor([HD_H, HD_W]))

    assert ids == [1, 2, 3] and session.obj_ids == ids and session.obj_with_new_inputs == ids
    for obj_idx, prompt in enumerate(THREE):
        coords, labels = _stored(session, obj_idx)
        count = len(prompt.points_px)
        assert tuple(coords.shape) == (1, 1, count, 2) and coords.dtype == torch.float32
        assert tuple(labels.shape) == (1, 1, count)
        assert bool((labels >= 0).all())  # neither the processor's padding (-10) nor "not a point" (-1)
        assert labels.flatten().tolist() == prompt.labels
        # The processor scales image px to the model's 1024 x 1024 input and does nothing else.
        expected = [value for x, y in prompt.points_px for value in (x * 1024 / HD_W, y * 1024 / HD_H)]
        assert coords.flatten().tolist() == pytest.approx(expected, abs=1e-3)


def test_start_stores_only_the_points_inside_the_image(processor):
    from outline_tracker.segmenter import base, hf

    # One point of A and one of B lie outside the 1920 x 1080 image; (-10, -10) is the value the
    # processor uses for padding and must never reach it.
    prompts = [
        ObjectPrompt("A", [(700.5, 500.5), (-10.0, -10.0)], [1, 0]),
        ObjectPrompt("B", [(1920.0, 600.5), (1000.5, 600.5), (1030.5, 640.5), (1010.0, 590.25)], [0, 1, 0, 1]),
        ObjectPrompt("C", [(1400.5, 350.5), (1380.5, 360.5)], [1, 0]),
    ]
    model = StubModel()
    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    results = segmenter.start(np.zeros((HD_H, HD_W, 3), np.uint8), prompts)

    assert [result.obj_id for result in results] == NAMES
    for obj_idx, kept in enumerate(THREE):  # what is left of each object: 1, 3 and 2 points
        coords, labels = _stored(segmenter.session, obj_idx)
        count = len(kept.points_px)
        assert tuple(coords.shape) == (1, 1, count, 2) and tuple(labels.shape) == (1, 1, count)
        assert bool((labels >= 0).all()) and labels.flatten().tolist() == kept.labels
        expected = [value for x, y in kept.points_px for value in (x * 1024 / HD_W, y * 1024 / HD_H)]
        assert coords.flatten().tolist() == pytest.approx(expected, abs=1e-3)
        assert bool((coords >= 0).all())

    with pytest.raises(base.PromptError, match="'B' has no positive point inside the image"):
        segmenter.start(np.zeros((HD_H, HD_W, 3), np.uint8), [THREE[0], ObjectPrompt("B", [(1920.0, 600.5)], [1])])
    assert len(model.calls) == 1  # the refused start did not reach the model


def test_preview_uses_a_session_of_its_own(processor):
    from outline_tracker.segmenter import hf

    model = StubModel()
    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    image = np.zeros((HD_H, HD_W, 3), np.uint8)
    segmenter.start(image, THREE)
    segmenter.step(image)
    session = segmenter.session

    shown = segmenter.preview(image, [ObjectPrompt("X", [(100.5, 200.5), (300.5, 200.5)], [1, 0])])

    assert [result.obj_id for result in shown] == ["X"]
    assert position(shown[0]) == pytest.approx(_disk_center(2, 0), abs=1.0)  # the stand-in's third call
    preview_call = model.calls[2]
    assert preview_call.session is not session and preview_call.frame_idx == 0
    assert preview_call.session.obj_ids == [1]
    coords, labels = _stored(preview_call.session, 0)
    assert tuple(coords.shape) == (1, 1, 2, 2) and labels.flatten().tolist() == [1, 0]
    # The run was not touched: same session, same objects, and its next frame is number 2.
    assert segmenter.session is session and session.obj_ids == [1, 2, 3]
    results = segmenter.step(image)
    assert model.calls[3].session is session and model.calls[3].frame_idx == 2
    assert [result.obj_id for result in results] == NAMES


def test_step_fallback_opens_a_new_session_on_cpu(processor):
    import torch

    from outline_tracker.segmenter import hf

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    # The third call fails on mps; B was not found on the second frame.
    model = StubModel(fail_on_call=2, absent={(1, 1)})
    segmenter = hf.HFSegmenter("edgetam", "mps", model=model, processor=processor)
    lines = []
    segmenter.log = lines.append
    image = np.zeros((HD_H, HD_W, 3), np.uint8)

    first = segmenter.start(image, THREE)
    second = segmenter.step(image)
    old_session = segmenter.session
    assert segmenter.device == "mps" and str(old_session.inference_device) == "mps"
    third = segmenter.step(image)  # fails on mps, runs again on cpu
    fourth = segmenter.step(image)

    assert segmenter.device == "cpu" and model.moved == ["mps", "cpu"]
    # The session counts its own frames: 0, 1, 2 on mps, then 0, 1 in the new session on cpu.
    assert [(call.device, call.frame_idx) for call in model.calls] == [
        ("mps", 0), ("mps", 1), ("mps", 2), ("cpu", 0), ("cpu", 1)]
    new_session = segmenter.session
    assert new_session is not old_session and str(new_session.inference_device) == "cpu"
    assert [call.session is new_session for call in model.calls] == [False, False, False, True, True]

    # Where each object was last found: A and C on the second frame, B on the first (geometry of
    # the stand-in's disks, within 1 px).
    last_seen = [position(second[0]), position(first[1]), position(second[2])]
    assert np.isnan(position(second[1])[0])
    assert last_seen[0] == pytest.approx(_disk_center(1, 0), abs=1.0)
    assert last_seen[1] == pytest.approx(_disk_center(0, 1), abs=1.0)
    assert last_seen[2] == pytest.approx(_disk_center(1, 2), abs=1.0)
    # The new session holds one positive point per object, exactly there, on cpu, with no padding.
    assert new_session.obj_ids == [1, 2, 3] and len(new_session.point_inputs_per_obj) == 3
    for obj_idx, (u, v) in enumerate(last_seen):
        coords, labels = _stored(new_session, obj_idx)
        assert tuple(coords.shape) == (1, 1, 1, 2) and labels.flatten().tolist() == [1]
        assert coords.flatten().tolist() == pytest.approx([u * 1024 / HD_W, v * 1024 / HD_H], abs=1e-3)
        assert coords.device.type == "cpu"

    for results in (third, fourth):
        assert [result.obj_id for result in results] == NAMES
        assert all(result.mask.any() for result in results)
    assert len(lines) == 1 and "The Apple GPU failed (NotImplementedError" in lines[0]


def test_reserve_ui_thread_with_the_real_torch():
    # In a process of its own: the thread count of this one must stay as it is for the regression
    # tests (SPEC 13.3 compares two runs with the same number of threads).
    code = (
        "import torch\n"
        "from outline_tracker.segmenter import hf\n"
        "before = torch.get_num_threads()\n"
        "first = hf.reserve_ui_thread()\n"
        "second = hf.reserve_ui_thread()\n"
        "print(before, first, second, torch.get_num_threads())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)
    assert done.returncode == 0, done.stderr
    before, first, second, after = (int(value) for value in done.stdout.split())
    assert first == second == after == max(1, before - 1)
