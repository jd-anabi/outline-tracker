"""The Hugging Face backend at the level of the segmenter: its prompts and masks, and the selftest
criterion of SPEC 13.4. Every test here is slow: it needs torch.

Three groups:
1. no weights (seconds): the real processor and session, with the network replaced by a stand-in
   that returns given logits: the clicks as the session holds them, scaled to the model's
   1024 x 1024 input; the mask as logits > 0 and the crop rule; and, for logits that draw a known
   disk, its center and area in the frame;
2. the real EdgeTAM on `cpu`: ONE loaded model, on the selftest clip (one object) and on a
   three-object clip made with the same recipe: what the segmenter reports, the clip's own
   properties, and the selftest criterion (max error < 3 px against the true positions). The first
   run downloads the model (56 MB) and converts it;
3. the selftest criterion on `mps`.

Until W1 step 5 the tests of groups 1 and 2 ran last week's script (`shrimp.segment`) beside the new
backend and compared the two; some test names still say "the reference". The script has left the
repository. What it confirmed is kept as the frozen table, tests/data/edgetam_cpu_positions.csv: the
positions that this backend gave in the run in which the script agreed within 0.01 px
(docs/VALIDATION.md, section 7). The comparison of a run with that table (SPEC 13.3) is asserted
once, in tests/slow/test_frozen_reference.py, for both clips; no test here reads the table.

Lines printed with the prefix `VALIDATION` are measured numbers; show them with
`uv run pytest -m slow tests/slow/test_regression_reference.py -q -rP`.

Coordinates: positions are in px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the
full frame: `mask_center` of a result's cropped mask plus its offset (col0, row0). Clips are 1080p.
"""

from __future__ import annotations

import hashlib
import math
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pipeline_helpers import _write_three_ellipse_clip

from outline_tracker import synthetic, tracker_io
from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import ObjectPrompt
from outline_tracker.video import iter_rgb_frames

pytestmark = pytest.mark.slow

W, H = 1920, 1080
CLICKS = [(700.5, 500.5), (1000.5, 600.5), (1400.5, 350.5)]  # px, one positive click per object
NAMES = ["A", "B", "C"]


def _prompts(clicks=CLICKS) -> list[ObjectPrompt]:
    return [ObjectPrompt(name, [click], [1]) for name, click in zip(NAMES, clicks)]


def _full(result, shape=(H, W)) -> np.ndarray:
    """The full-frame mask of a cropped result."""
    full = np.zeros(shape, bool)
    col0, row0 = result.offset
    full[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    return full


def _position(result) -> tuple[float, float, int]:
    """(u_px, v_px, area_px) of a result in the full frame; NaN, NaN, 0 if the object was not found."""
    u, v, area = mask_center(result.mask)
    return u + result.offset[0], v + result.offset[1], area


# ---------------------------------------------------------------------------------------------
# 1. No weights: the real processor and session, a stand-in for the network


class StubModel:
    """Stands in for the network: returns the given low-resolution logits and presence logits for
    every frame, records the frame index it was called with, and leaves the session untouched."""

    def __init__(self, pred_masks, scores):
        self.pred_masks, self.scores = pred_masks, scores
        self.frame_indices = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, inference_session, frame_idx, frame):
        self.frame_indices.append(frame_idx)
        return SimpleNamespace(object_ids=list(inference_session.obj_ids), pred_masks=self.pred_masks,
                               object_score_logits=self.scores, frame_idx=frame_idx)


@pytest.fixture(scope="module")
def processor():
    """The processor that `load_edgetam` builds: no weights and no download."""
    from transformers import Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor

    return Sam2VideoProcessor(image_processor=Sam2ImageProcessor(), video_processor=Sam2VideoVideoProcessor())


def _random_logits():
    import torch

    generator = torch.Generator().manual_seed(0)
    return torch.randn(3, 1, 256, 256, generator=generator) * 4


def _blob_logits():
    """Two compact objects on the model's 256 x 256 grid and one absent object (all -1024)."""
    import torch

    generator = torch.Generator().manual_seed(1)
    rows, cols = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
    disk = 6.0 - torch.hypot(cols - 100.0, rows - 120.0)
    ellipse = 1.0 - torch.sqrt(((cols - 250.0) / 9.0) ** 2 + ((rows - 3.0) / 4.0) ** 2)  # touches two borders
    absent = torch.full((256, 256), -1024.0)
    noise = 0.2 * torch.randn(2, 256, 256, generator=generator)
    return torch.stack([disk + noise[0], ellipse * 5 + noise[1], absent])[:, None]


def _differences(a, b, path="session") -> list[str]:
    """Where two session states differ, as readable lines (empty if they are equal)."""
    import torch

    if type(a) is not type(b):
        return [f"{path}: {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, torch.Tensor):
        same = a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
        return [] if same else [f"{path}: {a.flatten().tolist()[:6]} != {b.flatten().tolist()[:6]}"]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [f"{path}: keys {list(a)} != {list(b)}"]
        return [line for key in a for line in _differences(a[key], b[key], f"{path}[{key!r}]")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: {a!r} != {b!r}"]
        return [line for k, (x, y) in enumerate(zip(a, b)) for line in _differences(x, y, f"{path}[{k}]")]
    if type(a).__module__.startswith("transformers."):  # the session and its cache: every attribute
        return _differences(vars(a), vars(b), f"{path}.")
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _stored_point(session, obj_idx: int) -> tuple[list[float], list[int]]:
    """The prompt that a session holds for one object on its first frame: ([x, y], labels), with
    x and y on the model's 1024 x 1024 input, where the processor puts image px."""
    stored = session.point_inputs_per_obj[obj_idx][0]  # [object index][frame index]
    return stored["point_coords"].flatten().tolist(), stored["point_labels"].flatten().tolist()


def _leaves(value, path="session") -> list[str]:
    """The paths `_differences` walks, to show that a comparison of two sessions reaches the prompt
    tensors."""
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in _leaves(item, f"{path}[{key!r}]")]
    if type(value).__module__.startswith("transformers."):
        return _leaves(vars(value), f"{path}.")
    return [path]


def test_prompt_tensors_have_one_point_and_no_padding(processor):
    import torch

    from outline_tracker.segmenter import hf

    session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    ids = hf.add_prompts(processor, session, _prompts(), torch.tensor([H, W]))

    assert ids == [1, 2, 3]
    assert session.obj_with_new_inputs == ids
    assert session.obj_ids == ids
    for obj_idx, (x, y) in enumerate(CLICKS):
        stored = session.point_inputs_per_obj[obj_idx][0]  # [object index][frame index]
        coords, labels = stored["point_coords"], stored["point_labels"]
        assert tuple(coords.shape) == (1, 1, 1, 2) and coords.dtype == torch.float32
        assert tuple(labels.shape) == (1, 1, 1)
        assert bool((labels >= 0).all())  # neither the processor's padding (-10) nor "not a point" (-1)
        assert labels.flatten().tolist() == [1]
        # The processor scales image px to the model's 1024 x 1024 input and does nothing else.
        assert coords.flatten().tolist() == pytest.approx([x * 1024 / W, y * 1024 / H], abs=1e-3)


def test_session_equals_the_one_the_reference_builds(processor):
    # Until W1 step 5 this session was compared with the one that last week's script built from the same
    # clicks. What needs no script is asserted: what the session holds for each click.
    from outline_tracker.segmenter import hf

    image = np.zeros((H, W, 3), np.uint8)
    new = hf.HFSegmenter("edgetam", "cpu", model=StubModel(*_stub_outputs(_random_logits())), processor=processor)
    new.start(image, _prompts())  # one call per object

    leaves = _leaves(new.session)
    for obj_idx in range(3):
        assert f"session.['point_inputs_per_obj'][{obj_idx}][0]['point_coords']" in leaves
    assert "session.['obj_with_new_inputs']" in leaves and "session.['video_height']" in leaves
    assert new.session.obj_with_new_inputs == [1, 2, 3]

    # The session lists the three objects as having new inputs (asserted above) and holds one point with
    # the label 1 for each: its click, scaled from image px to the model's 1024 x 1024 input and nothing
    # else, as `test_prompt_tensors_have_one_point_and_no_padding` has it.
    assert new.session.obj_ids == [1, 2, 3]
    for obj_idx, (x, y) in enumerate(CLICKS):
        coords, labels = _stored_point(new.session, obj_idx)
        assert labels == [1]
        assert coords == pytest.approx([x * 1024 / W, y * 1024 / H], abs=1e-3)

    # A second session, with the second click moved 1 px to the right: that click is 1024 / 1920 further
    # right there, and nothing else of the session differs.
    moved = hf.HFSegmenter("edgetam", "cpu", model=StubModel(*_stub_outputs(_random_logits())), processor=processor)
    moved.start(image, _prompts([CLICKS[0], (CLICKS[1][0] + 1.0, CLICKS[1][1]), CLICKS[2]]))
    changed = _differences(new.session, moved.session)
    assert len(changed) == 1 and changed[0].startswith("session.['point_inputs_per_obj'][1][0]['point_coords']")
    (x, y), (moved_x, moved_y) = _stored_point(new.session, 1)[0], _stored_point(moved.session, 1)[0]
    assert moved_x - x == pytest.approx(1024 / W, abs=1e-3) and moved_y == y


def _stub_outputs(logits):
    import torch

    return logits, torch.tensor([5.0, 0.5, -3.0])  # presence logits, one per object


def _assert_cropped_to_its_mask(result) -> None:
    """The crop rule, from the result alone: the crop is the bounding box of the mask it holds with
    8 px added on every side, clipped to the frame; an object that was not found has shape (0, 0)."""
    mask = _full(result)
    if not mask.any():
        assert result.mask.shape == (0, 0)
        return
    rows, cols = np.nonzero(mask)
    col0, row0 = max(cols.min() - 8, 0), max(rows.min() - 8, 0)
    assert result.offset == (col0, row0)
    assert result.mask.shape == (min(rows.max() + 9, H) - row0, min(cols.max() + 9, W) - col0)


def _assert_the_disk_of_the_blob_logits(result) -> None:
    """The first object of `_blob_logits` is a known shape, so its mask has a known center and area.

    On the model's 256 x 256 grid the logits are 6 minus the distance, in cells, from the cell in
    column 100 and row 120: positive on the disk of radius 6 cells around that cell's center,
    (100.5, 120.5) cells from the grid's corner. A cell is 1920 / 256 = 7.5 px wide and
    1080 / 256 = 4.21875 px high in the frame. So the object is an ellipse with its center at
    (753.75, 508.359) px (Tracker's convention) and the semi-axes 45 and 25.3125 px; its area is
    pi * 45 * 25.3125 = 3578 px.

    What the mask may differ by, two parts that add:
    - the pixel grid: the mask's area differs from the shape's by at most half a pixel along its
      perimeter, and its center by at most 0.5 px. An ellipse's perimeter is at most
      2 pi sqrt((a^2 + b^2) / 2), here 229 px: 115 px of area;
    - the noise that `_blob_logits` adds, with a standard deviation of 0.2. The logits fall by 1
      per cell, so the noise moves the outline by that many cells, independently from cell to cell
      along the 2 pi 6 = 38 cells of the outline. To first order the area changes by the sum of
      those moves, with a standard deviation of 0.2 sqrt(2 pi 6) = 1.23 cells of area, 39 px; the
      center by their mean weighted with the outline's direction, 0.2 / sqrt(pi 6) = 0.046 cells
      per axis: 0.35 px along u, 0.19 px along v. Four standard deviations are allowed.
    Together: 270 px of area (7.6 %), 1.9 px along u and 1.3 px along v. A center that is half a
    cell off (3.75 px, 2.1 px) and a radius that is half a cell off (16 % of the area) are outside.
    """
    cell_w, cell_h, radius, noise = W / 256, H / 256, 6.0, 0.2
    a, b = radius * cell_w, radius * cell_h
    area_sd = noise * math.sqrt(2 * math.pi * radius) * cell_w * cell_h
    center_sd = noise / math.sqrt(math.pi * radius)  # cells
    perimeter = 2 * math.pi * math.sqrt((a * a + b * b) / 2)  # at most
    u_px, v_px, area_px = _position(result)
    assert area_px == pytest.approx(math.pi * a * b, abs=0.5 * perimeter + 4 * area_sd)
    assert u_px == pytest.approx(100.5 * cell_w, abs=0.5 + 4 * center_sd * cell_w)
    assert v_px == pytest.approx(120.5 * cell_h, abs=0.5 + 4 * center_sd * cell_h)


@pytest.mark.parametrize("make_logits", [_random_logits, _blob_logits], ids=["random", "blobs"])
def test_per_object_logits_give_the_reference_masks(processor, make_logits):
    # Until W1 step 5 the masks were compared with those that last week's script made of the same logits,
    # pixel for pixel. What needs no script is asserted: the rule that makes a mask of logits and crops it,
    # and, for the logits that draw a known disk, where that disk is.
    from outline_tracker.segmenter import hf

    image = np.zeros((H, W, 3), np.uint8)
    logits, scores = _stub_outputs(make_logits())
    model = StubModel(logits, scores)
    new = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)

    for step in range(3):
        results = new.start(image, _prompts()) if step == 0 else new.step(image)
        assert [r.obj_id for r in results] == NAMES
        assert [r.score for r in results] == pytest.approx([5.0, 0.5, -3.0])
        for result in results:
            assert result.mask.dtype == bool and result.logits.dtype == np.float32
            assert np.array_equal(result.logits > 0, result.mask)
            _assert_cropped_to_its_mask(result)  # the bounding box +/- 8 px, clipped to the frame
    if make_logits is _blob_logits:
        assert results[0].mask.size < 0.01 * H * W  # a real crop, not the whole frame
        assert [bool(result.mask.any()) for result in results] == [True, True, False]  # the third object is absent
        _assert_the_disk_of_the_blob_logits(results[0])
    # The session counts its own frames 0, 1, 2, ...: never the video's frame numbers.
    assert model.frame_indices == [0, 1, 2]


# ---------------------------------------------------------------------------------------------
# 2. The real EdgeTAM on cpu: one loaded model


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model("edgetam")


def _track(segmenter, video, frames, prompts) -> SimpleNamespace:
    """Drive a segmenter over the frames of a clip, read one after the other with
    `video.iter_rgb_frames`: `start` with the prompts on the first, `step` on every later one.

    Returns frames (video frame numbers), results (one list of MaskResult per frame) and seconds
    (per frame, decoding included).
    """
    done, results, seconds = [], [], []
    t0 = time.perf_counter()
    for i, (frame, rgb) in enumerate(iter_rgb_frames(video, frames)):
        results.append(segmenter.start(rgb, prompts) if i == 0 else segmenter.step(rgb))
        done.append(frame)
        seconds.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    return SimpleNamespace(frames=done, results=results, seconds=np.array(seconds))


def _timing(seconds: np.ndarray) -> str:
    return f"{seconds.mean():.2f} s per frame (mean of {len(seconds)}; median {np.median(seconds):.2f})"


@pytest.fixture(scope="module")
def selftest_runs(loaded, tmp_path_factory):
    """The selftest clip with its Tracker export (`synthetic.selftest_clip`), and the backend on it:
    on the frames and from the click that `tracker_io.make_plan` reads from the export."""
    from outline_tracker.segmenter import hf

    model, processor = loaded
    clip = synthetic.selftest_clip(tmp_path_factory.mktemp("selftest"))
    video, export = clip["video"], clip["export"]
    plan = tracker_io.make_plan(export, fps=240.0)
    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    run = _track(segmenter, video, plan.frames, [ObjectPrompt("selftest", [plan.points_px[0]], [1])])
    return SimpleNamespace(
        video=video, plan=plan, segmenter=segmenter, run=run,
        truth=pd.read_csv(export, skiprows=1),  # the click and the true positions
        positions=np.array([_position(results[0])[:2] for results in run.results]),
    )


def test_selftest_clip_within_3_px_of_truth_on_cpu(selftest_runs):
    s = selftest_runs
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    error = np.hypot(*(s.positions - true).T)
    print(f"VALIDATION selftest criterion, cpu: max error {np.nanmax(error):.3f} px; {_timing(s.run.seconds)}")
    assert np.isfinite(error).all() and error.max() < 3.0


def test_segmenter_reports_device_model_and_weights(selftest_runs):
    from outline_tracker.segmenter import edgetam_convert

    segmenter = selftest_runs.segmenter
    assert segmenter.device == "cpu"
    assert segmenter.model_id == "facebook/EdgeTAM"
    saved = edgetam_convert.edgetam_cache() / "model.safetensors"  # the converted copy the model is loaded from
    assert segmenter.weights_sha256 == hashlib.sha256(saved.read_bytes()).hexdigest()
    print(f"VALIDATION weights: {saved.stat().st_size} bytes, sha256 {segmenter.weights_sha256}")

    for results in selftest_runs.run.results:
        (result,) = results
        assert result.obj_id == "selftest"
        assert np.array_equal(result.logits > 0, result.mask)
        if result.mask.any():
            assert result.score > 0  # the presence logit: an object the model calls absent has no mask
    segmenter.close()
    assert segmenter.session is None


def test_three_objects_match_the_reference(loaded, tmp_path):
    # The reference is the frozen table, and the comparison with it is asserted once, in
    # tests/slow/test_frozen_reference.py::test_three_ellipses_equal_the_frozen_numbers. Here: the clip's
    # own properties, and the selftest criterion for this clip, asserted directly.
    from outline_tracker.segmenter import hf

    model, processor = loaded
    video = tmp_path / "three_tracker.mp4"
    truth = _write_three_ellipse_clip(video)
    frames = list(range(0, 40, 2))
    clicks = [(track[0][0] + 0.5, track[0][1] + 0.5) for track in truth]  # Tracker's convention (+0.5)
    separations = [np.hypot(a[f][0] - b[f][0], a[f][1] - b[f][1])
                   for a, b in [(truth[0], truth[1]), (truth[1], truth[2]), (truth[0], truth[2])] for f in range(40)]
    assert min(separations) >= 300.0
    _, first = next(iter_rgb_frames(video, [0]))
    red, green, blue = first.reshape(-1, 3).mean(axis=0)
    assert red > green + 10 and green > blue + 10  # the three channels differ, also after decoding

    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    run = _track(segmenter, video, frames, _prompts(clicks))

    assert run.frames == frames
    for results in run.results:
        assert [r.obj_id for r in results] == NAMES
    # [i][k] = (u_px, v_px) of NAMES[k] on frames[i]: where it was found, and its true center
    positions = np.array([[_position(result)[:2] for result in results] for results in run.results])
    true = np.array([[(track[frame][0] + 0.5, track[frame][1] + 0.5) for track in truth] for frame in frames])
    error = np.hypot(positions[..., 0] - true[..., 0], positions[..., 1] - true[..., 1])
    found = ~np.isnan(error)
    print(f"VALIDATION three-ellipse clip (3 objects, 20 frames, cpu): max error against the true positions "
          f"{error[found].max():.3f} px; lost: {int((~found).sum())} of {found.size}; {_timing(run.seconds)}")
    # the selftest criterion (SPEC 13.4) for this clip too: every found position under 3 px from the
    # true center, which the clip's recipe gives
    assert error[found].max() < 3.0


# ---------------------------------------------------------------------------------------------
# 3. The selftest criterion on the Apple GPU


def test_selftest_clip_within_3_px_of_truth_on_mps(selftest_runs):
    import torch

    from outline_tracker.segmenter import hf

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    s = selftest_runs
    segmenter = hf.HFSegmenter("edgetam", "mps")  # its own copy of the model: the shared one stays on cpu
    run = _track(segmenter, s.video, s.plan.frames, [ObjectPrompt("selftest", [s.plan.points_px[0]], [1])])
    positions = np.array([_position(results[0])[:2] for results in run.results])
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    error = np.hypot(*(positions - true).T)
    print(f"VALIDATION selftest criterion, mps: max error {np.nanmax(error):.3f} px; {_timing(run.seconds)}; "
          f"device at the end: {segmenter.device}; max distance from the cpu positions "
          f"{np.nanmax(np.hypot(*(positions - s.positions).T)):.3f} px")
    assert segmenter.device == "mps"  # it did not fall back to the processor
    assert np.isfinite(error).all() and error.max() < 3.0
