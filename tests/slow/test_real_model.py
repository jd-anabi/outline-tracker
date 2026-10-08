"""The backend's extensions with the real processor and the real EdgeTAM (SPEC 5, 6.2, 13.4):
several points per object, negative points, `preview`, the fall back in `step`. Every test here
is slow: it needs torch.

Three groups:
1. no weights (seconds): the real processor and session, with the network replaced by a stand-in
   that returns one disk per object and can fail once, like an Apple GPU that lacks an operation;
2. the real EdgeTAM on `cpu`: the negative prompt of SPEC 13.4, and `preview` against the first
   frame of a run;
3. the real EdgeTAM on `mps`: the model call is made to raise once on the third frame, and the
   run has to finish on `cpu`. A plain mps run does not fail by itself (docs/VALIDATION.md 1.2).

Scenes are made in memory, never written to disk: 1080p RGB frames in the selftest's style (a
bright background with a vignette, noise of sigma 3, dark ellipses), with exact ground truth.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_real_model.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame. An
ellipse is (u0, v0, a, b, angle): center, semi-axes in px, and the angle of the major axis in
degrees, from the u axis toward the v axis.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pipeline_helpers import H, W, position

from outline_tracker.segmenter.base import ObjectPrompt

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
DARK = 70.0  # gray value of an ellipse, as in the selftest
TINT = np.array([1.00, 0.90, 0.80])  # red, green, blue: the three channels differ


# ---------------------------------------------------------------------------------------------
# Scenes with exact ground truth


def _inside(u, v, ellipse):
    """True where the points (u, v), in px, lie inside the ellipse."""
    u0, v0, a, b, angle = ellipse
    cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    along = (u - u0) * cos + (v - v0) * sin
    across = -(u - u0) * sin + (v - v0) * cos
    return (along / a) ** 2 + (across / b) ** 2 <= 1.0


def _truth(ellipse) -> np.ndarray:
    """The full-frame mask of an ellipse: the pixels whose centers lie inside it."""
    rows, cols = np.mgrid[0:H, 0:W]
    return _inside(cols + 0.5, rows + 0.5, ellipse)


def _render(ellipses, seed: int) -> np.ndarray:
    """An RGB uint8 frame [row, column] with dark ellipses. Each pixel is darkened by the fraction
    of it that an ellipse covers (4 x 4 samples per pixel), so edges are smooth as in a real image."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:H, 0:W]
    base = 205.0 - 15.0 * ((cols + 0.5 - W / 2) ** 2 + (rows + 0.5 - H / 2) ** 2) / (0.49 * H) ** 2
    image = base[:, :, None] * TINT
    samples = (np.arange(4) + 0.5) / 4
    for ellipse in ellipses:
        u0, v0, a, b, _ = ellipse
        reach = int(np.ceil(max(a, b))) + 2
        col0, col1 = max(int(u0) - reach, 0), min(int(u0) + reach + 1, W)
        row0, row1 = max(int(v0) - reach, 0), min(int(v0) + reach + 1, H)
        u = (np.arange(col0, col1)[:, None] + samples).reshape(1, -1)
        v = (np.arange(row0, row1)[:, None] + samples).reshape(-1, 1)
        cover = _inside(u, v, ellipse).reshape(row1 - row0, 4, col1 - col0, 4).mean(axis=(1, 3))
        window = image[row0:row1, col0:col1]
        window += cover[:, :, None] * (DARK * TINT - window)
    return np.clip(image + rng.normal(0.0, 3.0, image.shape), 0, 255).astype(np.uint8)


def _full(result) -> np.ndarray:
    """The full-frame mask of a cropped result."""
    full = np.zeros((H, W), bool)
    col0, row0 = result.offset
    full[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    return full


# The moving scene: three small ellipses (semi-axes 8 and 3 px, the selftest's test shrimp), each
# moving along its long axis, at least 300 px apart.
STARTS = [(500.5, 300.5), (1000.5, 600.5), (1400.5, 350.5)]  # centers on frame 0, px
STEPS = [(0.6, 0.2), (-0.5, 0.3), (0.2, -0.6)]  # px per frame
NAMES = ["A", "B", "C"]


def _moving_ellipses(t: int):
    return [(u0 + du * t, v0 + dv * t, 8.0, 3.0, float(np.degrees(np.arctan2(dv, du))))
            for (u0, v0), (du, dv) in zip(STARTS, STEPS)]


def _moving_frame(t: int) -> np.ndarray:
    return _render(_moving_ellipses(t), seed=t)


def _moving_prompts() -> list[ObjectPrompt]:
    """Prompts on frame 0 with 1, 3 and 2 points. Positive points lie on the object's long axis,
    inside it (4 px from the center of an ellipse that is 8 px long on each side); negative points
    lie on the background, 40 px from the center across the long axis."""
    (a, b, c), prompts = STARTS, []
    prompts.append(ObjectPrompt("A", [a], [1]))
    along = np.array(STEPS[1]) / np.hypot(*STEPS[1])
    across = np.array([-along[1], along[0]])
    prompts.append(ObjectPrompt("B", [b, tuple(b + 40.0 * across), tuple(b + 4.0 * along)], [1, 0, 1]))
    along = np.array(STEPS[2]) / np.hypot(*STEPS[2])
    across = np.array([-along[1], along[0]])
    prompts.append(ObjectPrompt("C", [c, tuple(c - 40.0 * across)], [1, 0]))
    return prompts


def _same_results(a, b) -> bool:
    """Two lists of results are equal in every field, bit for bit."""
    return len(a) == len(b) and all(
        x.obj_id == y.obj_id and x.offset == y.offset and x.score == y.score
        and np.array_equal(x.mask, y.mask) and np.array_equal(x.logits, y.logits) for x, y in zip(a, b))


# ---------------------------------------------------------------------------------------------
# 1. No weights: the real processor and session, a stand-in for the network


class StubModel:
    """Stands in for the network. Object number j of the session (0, 1, ...) is a disk of radius
    6 cells on the model's 256 x 256 grid, centered on cell (column 60 + 50 j + 2 n, row 100) at
    the stand-in's call number n, so its center in the image is ((60.5 + 50 j + 2 n) W / 256,
    100.5 H / 256) px. `absent` lists the (n, j) for which the object is reported absent, as the
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
    return (60.5 + 50.0 * j + 2.0 * n) * W / 256, 100.5 * H / 256


def _stored(session, obj_idx: int):
    """(coordinates, labels) the session holds for an object on its first frame, as tensors."""
    stored = session.point_inputs_per_obj[obj_idx][0]  # [object index][frame index]
    return stored["point_coords"], stored["point_labels"]


THREE = [  # objects with 1, 3 and 2 points, px in the 1920 x 1080 image
    ObjectPrompt("A", [(700.5, 500.5)], [1]),
    ObjectPrompt("B", [(1000.5, 600.5), (1030.5, 640.5), (1010.0, 590.25)], [1, 0, 1]),
    ObjectPrompt("C", [(1400.5, 350.5), (1380.5, 360.5)], [1, 0]),
]


def test_prompt_tensors_for_1_3_and_2_points_have_no_padding(processor):
    import torch

    from outline_tracker.segmenter import hf

    session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    ids = hf.add_prompts(processor, session, THREE, torch.tensor([H, W]))

    assert ids == [1, 2, 3] and session.obj_ids == ids and session.obj_with_new_inputs == ids
    for obj_idx, prompt in enumerate(THREE):
        coords, labels = _stored(session, obj_idx)
        count = len(prompt.points_px)
        assert tuple(coords.shape) == (1, 1, count, 2) and coords.dtype == torch.float32
        assert tuple(labels.shape) == (1, 1, count)
        assert bool((labels >= 0).all())  # neither the processor's padding (-10) nor "not a point" (-1)
        assert labels.flatten().tolist() == prompt.labels
        # The processor scales image px to the model's 1024 x 1024 input and does nothing else.
        expected = [value for x, y in prompt.points_px for value in (x * 1024 / W, y * 1024 / H)]
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
    results = segmenter.start(np.zeros((H, W, 3), np.uint8), prompts)

    assert [result.obj_id for result in results] == NAMES
    for obj_idx, kept in enumerate(THREE):  # what is left of each object: 1, 3 and 2 points
        coords, labels = _stored(segmenter.session, obj_idx)
        count = len(kept.points_px)
        assert tuple(coords.shape) == (1, 1, count, 2) and tuple(labels.shape) == (1, 1, count)
        assert bool((labels >= 0).all()) and labels.flatten().tolist() == kept.labels
        expected = [value for x, y in kept.points_px for value in (x * 1024 / W, y * 1024 / H)]
        assert coords.flatten().tolist() == pytest.approx(expected, abs=1e-3)
        assert bool((coords >= 0).all())

    with pytest.raises(base.PromptError, match="'B' has no positive point inside the image"):
        segmenter.start(np.zeros((H, W, 3), np.uint8), [THREE[0], ObjectPrompt("B", [(1920.0, 600.5)], [1])])
    assert len(model.calls) == 1  # the refused start did not reach the model


def test_preview_uses_a_session_of_its_own(processor):
    from outline_tracker.segmenter import hf

    model = StubModel()
    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    image = np.zeros((H, W, 3), np.uint8)
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
    image = np.zeros((H, W, 3), np.uint8)

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
        assert coords.flatten().tolist() == pytest.approx([u * 1024 / W, v * 1024 / H], abs=1e-3)
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


# ---------------------------------------------------------------------------------------------
# 2. The real EdgeTAM on cpu


# Two equal ellipses with parallel axes touch exactly when the line between their centers, seen
# along their axes, ends on the ellipse with twice the semi-axes: (dx / 2a)^2 + (dy / 2b)^2 = 1.
# Here a = 40, b = 15 px (the ellipse of SPEC 13.1), both turned by 30 degrees, and the second
# center is at 45 degrees on that doubled ellipse, so they touch flank to flank, shifted lengthwise.
A_PX, B_PX, TURN_DEG, WHERE_DEG = 40.0, 15.0, 30.0, 45.0
CENTER_1 = (940.0, 520.0)


def _touching_ellipses():
    along, across = 2 * A_PX * np.cos(np.radians(WHERE_DEG)), 2 * B_PX * np.sin(np.radians(WHERE_DEG))
    cos, sin = np.cos(np.radians(TURN_DEG)), np.sin(np.radians(TURN_DEG))
    center_2 = (CENTER_1[0] + along * cos - across * sin, CENTER_1[1] + along * sin + across * cos)
    return (*CENTER_1, A_PX, B_PX, TURN_DEG), (*center_2, A_PX, B_PX, TURN_DEG)


@pytest.mark.weights
def test_negative_click_excludes_the_touching_neighbor(loaded):
    from outline_tracker.segmenter import hf

    model, processor = loaded
    clicked, neighbor = _touching_ellipses()
    truth, truth_neighbor = _truth(clicked), _truth(neighbor)
    # The scene is what it should be: two ellipses of pi a b = 1885 px that touch and do not overlap.
    for mask in (truth, truth_neighbor):
        assert mask.sum() == pytest.approx(np.pi * A_PX * B_PX, rel=0.01)
    assert not (truth & truth_neighbor).any()
    grown = cv2.dilate(truth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert (grown & truth_neighbor).any()  # within 2 px of each other
    image = _render([clicked, neighbor], seed=0)
    assert image[int(clicked[1]), int(clicked[0])].mean() < 90 < 150 < image[400, 800].mean()  # dark on bright

    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    here, there = clicked[:2], neighbor[:2]

    def overlaps(clicks):
        """Preview with the given (point, label) clicks: the fraction of the neighbor and of the clicked
        ellipse that the mask covers, the mask's area in px, and the seconds the preview took."""
        points, labels = zip(*clicks)
        t0 = time.perf_counter()
        (result,) = segmenter.preview(image, [ObjectPrompt("A", list(points), list(labels))])
        seconds = time.perf_counter() - t0
        mask = _full(result)
        return ((mask & truth_neighbor).sum() / truth_neighbor.sum(), (mask & truth).sum() / truth.sum(),
                int(mask.sum()), seconds)

    on_neighbor, on_clicked, area, seconds = overlaps([(here, 1), (there, 0)])
    alone = overlaps([(here, 1)])  # for comparison, not asserted
    both = overlaps([(here, 1), (there, 1)])  # for comparison, not asserted
    print(f"VALIDATION negative prompt (two touching ellipses, a = 40, b = 15 px, 1080p, cpu): positive on one, "
          f"negative on the other: mask of {area} px covers {100 * on_neighbor:.1f}% of the neighbor and "
          f"{100 * on_clicked:.1f}% of the clicked ellipse; preview took {seconds:.2f} s. For comparison, "
          f"positive click alone: {100 * alone[0]:.1f}% of the neighbor, {100 * alone[1]:.1f}% of the clicked one "
          f"({alone[2]} px); both clicks positive: {100 * both[0]:.1f}% and {100 * both[1]:.1f}% ({both[2]} px)")
    assert on_neighbor < 0.05  # SPEC 13.4
    assert on_clicked > 0.5  # and the clicked ellipse is still there

    # For the record, not asserted: the positive click moved from the center of its ellipse toward
    # the point where the two touch (the middle between the centers), the negative click as before.
    contact = (np.array(here) + np.array(there)) / 2
    for fraction in (0.5, 0.8, 0.95, 1.0):
        point = tuple(np.array(here) + fraction * (contact - np.array(here)))
        with_negative, without = overlaps([(point, 1), (there, 0)]), overlaps([(point, 1)])
        print(f"VALIDATION negative prompt, positive click at {fraction:.2f} of the way to the contact point: with the "
              f"negative click {100 * with_negative[0]:.1f}% of the neighbor, {100 * with_negative[1]:.1f}% of the "
              f"clicked ellipse ({with_negative[2]} px); positive click alone {100 * without[0]:.1f}%, "
              f"{100 * without[1]:.1f}% ({without[2]} px)")


@pytest.mark.weights
def test_preview_equals_the_first_frame_of_a_run(loaded):
    from outline_tracker.segmenter import hf

    model, processor = loaded
    frames = [_moving_frame(t) for t in range(3)]
    prompts = _moving_prompts()
    assert [len(prompt.points_px) for prompt in prompts] == [1, 3, 2]

    previewer = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    t0 = time.perf_counter()
    shown = previewer.preview(frames[0], prompts)
    seconds = time.perf_counter() - t0
    assert previewer.session is None  # nothing is kept

    plain = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    run = [plain.start(frames[0], prompts), plain.step(frames[1]), plain.step(frames[2])]

    # The same run with previews in between: on another frame, with other prompts.
    other = [ObjectPrompt("X", [STARTS[2], STARTS[0]], [1, 0])]
    mixed = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    interrupted = [mixed.start(frames[0], prompts)]
    mixed.preview(frames[2], other)
    interrupted.append(mixed.step(frames[1]))
    again = mixed.preview(frames[0], prompts)
    interrupted.append(mixed.step(frames[2]))

    errors = [float(np.hypot(*np.subtract(position(result), start))) for result, start in zip(shown, STARTS)]
    print(f"VALIDATION preview (3 objects with 1, 3 and 2 points, 1080p, cpu): {seconds:.2f} s; equal to the first "
          f"frame of a run: {_same_results(shown, run[0])}; a run with previews in between equals the plain run: "
          f"{all(_same_results(a, b) for a, b in zip(interrupted, run))}; distance of the previewed centers from "
          f"the true centers: {', '.join(f'{e:.2f}' for e in errors)} px")
    assert [result.obj_id for result in shown] == NAMES
    assert _same_results(shown, run[0])
    assert _same_results(again, run[0])
    assert all(_same_results(a, b) for a, b in zip(interrupted, run))
    assert np.isfinite(errors).all() and max(errors) < 3.0  # the prompts with several points found the ellipses


# ---------------------------------------------------------------------------------------------
# 3. The real EdgeTAM on the Apple GPU, with one injected failure


@pytest.mark.weights
def test_step_fallback_with_the_real_model_finishes_on_cpu(monkeypatch):
    import torch

    from outline_tracker.segmenter import hf

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    n_frames = 12
    segmenter = hf.HFSegmenter("edgetam", "mps")  # its own copy of the model
    lines = []
    segmenter.log = lines.append

    real_forward = segmenter.model.forward
    calls = []  # (device of the model, the session's frame number) of every call of the model

    def forward(*args, **kwargs):
        calls.append((next(segmenter.model.parameters()).device.type, kwargs["frame_idx"]))
        if len(calls) == 3 and segmenter.device == "mps":  # the third frame, once
            raise NotImplementedError("The operator 'aten::injected' is not currently implemented for the MPS device.")
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(segmenter.model, "forward", forward)

    prompts = [ObjectPrompt(name, [start], [1]) for name, start in zip(NAMES, STARTS)]
    positions, devices, seconds = [], [], []
    for t in range(n_frames):
        frame = _moving_frame(t)
        t0 = time.perf_counter()
        results = segmenter.start(frame, prompts) if t == 0 else segmenter.step(frame)
        seconds.append(time.perf_counter() - t0)
        devices.append(segmenter.device)
        assert [result.obj_id for result in results] == NAMES
        positions.append([position(result) for result in results])
        if t == 1:
            before = positions[1]  # where the objects were last found before the failure

    assert devices == ["mps", "mps"] + ["cpu"] * (n_frames - 2)
    assert next(segmenter.model.parameters()).device.type == "cpu"
    # Frames 0, 1, 2 on mps (the third raised), then the new session's frames 0, 1, ... on cpu.
    assert calls == [("mps", 0), ("mps", 1), ("mps", 2)] + [("cpu", k) for k in range(n_frames - 2)]
    assert len(lines) == 1 and "The Apple GPU failed (NotImplementedError" in lines[0]
    # The new session was started with one positive click per object where it was last found.
    for obj_idx, (u, v) in enumerate(before):
        coords, labels = _stored(segmenter.session, obj_idx)
        assert tuple(coords.shape) == (1, 1, 1, 2) and labels.flatten().tolist() == [1]
        assert coords.flatten().tolist() == pytest.approx([u * 1024 / W, v * 1024 / H], abs=1e-3)

    true = np.array([[ellipse[:2] for ellipse in _moving_ellipses(t)] for t in range(n_frames)])
    error = np.hypot(*(np.array(positions) - true).transpose(2, 0, 1))  # [frame, object], px
    print(f"VALIDATION step fallback (3 objects, {n_frames} frames, 1080p; the model call raised once on the third "
          f"frame on mps): finished on {segmenter.device}; max error against the true centers {np.nanmax(error):.3f} "
          f"px (frames 0-1 on mps: {np.nanmax(error[:2]):.3f}; the fallback frame: {np.nanmax(error[2]):.3f}; "
          f"after it: {np.nanmax(error[3:]):.3f}); lost: {int(np.isnan(error).sum())} of {error.size}; "
          f"{np.mean(seconds[3:]):.2f} s per frame on cpu after the fall back, {seconds[2]:.2f} s for the "
          f"fallback frame; log: {lines[0].strip()}")
    assert np.isfinite(error).all() and error.max() < 3.0
