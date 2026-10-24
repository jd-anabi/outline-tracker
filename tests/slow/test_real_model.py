"""The backend's extensions with the real EdgeTAM (SPEC 5, 6.2, 13.4): the negative prompt, `preview` against the
first frame of a run, and the fall back in `step`. Every test here is slow: it needs torch, and each loads the weights
of the model.

Two groups:
1. the real EdgeTAM on `cpu`: the negative prompt of SPEC 13.4, and `preview` against the first
   frame of a run;
2. the real EdgeTAM on `mps`: the model call is made to raise once on the third frame, and the
   run has to finish on `cpu`. A plain mps run does not fail by itself (docs/VALIDATION.md 1.2).

The tests with the real processor and a stand-in for the network, which need no weights, are in
tests/slow/test_real_model_without_weights.py.

Scenes are made in memory, never written to disk: 1080p RGB frames in the selftest's style (a
bright background with a vignette, noise of sigma 3, dark ellipses), with exact ground truth.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_real_model.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame. An
ellipse is (u0, v0, a, b, angle): center, semi-axes in px, and the angle of the major axis in
degrees, from the u axis toward the v axis.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from pipeline_helpers import HD_H, HD_W, NAMES, _stored, position

from outline_tracker.segmenter.base import ObjectPrompt

pytestmark = pytest.mark.slow

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
    rows, cols = np.mgrid[0:HD_H, 0:HD_W]
    return _inside(cols + 0.5, rows + 0.5, ellipse)


def _render(ellipses, seed: int) -> np.ndarray:
    """An RGB uint8 frame [row, column] with dark ellipses. Each pixel is darkened by the fraction
    of it that an ellipse covers (4 x 4 samples per pixel), so edges are smooth as in a real image."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:HD_H, 0:HD_W]
    base = 205.0 - 15.0 * ((cols + 0.5 - HD_W / 2) ** 2 + (rows + 0.5 - HD_H / 2) ** 2) / (0.49 * HD_H) ** 2
    image = base[:, :, None] * TINT
    samples = (np.arange(4) + 0.5) / 4
    for ellipse in ellipses:
        u0, v0, a, b, _ = ellipse
        reach = int(np.ceil(max(a, b))) + 2
        col0, col1 = max(int(u0) - reach, 0), min(int(u0) + reach + 1, HD_W)
        row0, row1 = max(int(v0) - reach, 0), min(int(v0) + reach + 1, HD_H)
        u = (np.arange(col0, col1)[:, None] + samples).reshape(1, -1)
        v = (np.arange(row0, row1)[:, None] + samples).reshape(-1, 1)
        cover = _inside(u, v, ellipse).reshape(row1 - row0, 4, col1 - col0, 4).mean(axis=(1, 3))
        window = image[row0:row1, col0:col1]
        window += cover[:, :, None] * (DARK * TINT - window)
    return np.clip(image + rng.normal(0.0, 3.0, image.shape), 0, 255).astype(np.uint8)


def _full(result) -> np.ndarray:
    """The full-frame mask of a cropped result."""
    full = np.zeros((HD_H, HD_W), bool)
    col0, row0 = result.offset
    full[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    return full


# The moving scene: three small ellipses (semi-axes 8 and 3 px, the selftest's test shrimp), each
# moving along its long axis, at least 300 px apart.
STARTS = [(500.5, 300.5), (1000.5, 600.5), (1400.5, 350.5)]  # centers on frame 0, px
STEPS = [(0.6, 0.2), (-0.5, 0.3), (0.2, -0.6)]  # px per frame


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
# 1. The real EdgeTAM on cpu


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
# 2. The real EdgeTAM on the Apple GPU, with one injected failure


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
        assert coords.flatten().tolist() == pytest.approx([u * 1024 / HD_W, v * 1024 / HD_H], abs=1e-3)

    true = np.array([[ellipse[:2] for ellipse in _moving_ellipses(t)] for t in range(n_frames)])
    error = np.hypot(*(np.array(positions) - true).transpose(2, 0, 1))  # [frame, object], px
    print(f"VALIDATION step fallback (3 objects, {n_frames} frames, 1080p; the model call raised once on the third "
          f"frame on mps): finished on {segmenter.device}; max error against the true centers {np.nanmax(error):.3f} "
          f"px (frames 0-1 on mps: {np.nanmax(error[:2]):.3f}; the fallback frame: {np.nanmax(error[2]):.3f}; "
          f"after it: {np.nanmax(error[3:]):.3f}); lost: {int(np.isnan(error).sum())} of {error.size}; "
          f"{np.mean(seconds[3:]):.2f} s per frame on cpu after the fall back, {seconds[2]:.2f} s for the "
          f"fallback frame; log: {lines[0].strip()}")
    assert np.isfinite(error).all() and error.max() < 3.0
