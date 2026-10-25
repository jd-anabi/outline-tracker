"""The Hugging Face backend at the level of the segmenter, with the real processor and a stand-in for the network: its
prompts and masks. Every test here is slow: it needs torch, and none needs the weights of the model (seconds).

The processor and the session are the real ones; the network is replaced by a stand-in that returns given logits.
The tests are about the clicks as the session holds them, scaled to the model's 1024 x 1024 input; the mask as
logits > 0 and the crop rule; for logits that draw a known disk, its center and area in the frame; and for random
logits, that each mask covers half of the frame and no two objects have the same.

Until W1 step 5 these tests ran last week's script (`shrimp.segment`) beside the new backend and compared the two;
some test names still say "the reference". The script has left the repository; what these tests assert needs no
script. The tests with the real EdgeTAM and the selftest criterion of SPEC 13.4 are in
tests/slow/test_regression_reference.py.

Coordinates: positions are in px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the
full frame: `mask_center` of a result's cropped mask plus its offset (col0, row0). Clips are 1080p.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pipeline_helpers import CLICKS, HD_H, HD_W, NAMES, _prompts, position

from outline_tracker.measure.mask import mask_center

pytestmark = pytest.mark.slow


def _full(result, shape=(HD_H, HD_W)) -> np.ndarray:
    """The full-frame mask of a cropped result."""
    full = np.zeros(shape, bool)
    col0, row0 = result.offset
    full[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    return full


# ---------------------------------------------------------------------------------------------
# No weights: the real processor and session, a stand-in for the network


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

    from outline_tracker.segmenter import hf_prompts

    session = processor.init_video_session(inference_device="cpu", dtype=torch.float32)
    ids = hf_prompts.add_prompts(processor, session, _prompts(), torch.tensor([HD_H, HD_W]))

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
        assert coords.flatten().tolist() == pytest.approx([x * 1024 / HD_W, y * 1024 / HD_H], abs=1e-3)


def test_session_equals_the_one_the_reference_builds(processor):
    # Until W1 step 5 this session was compared with the one that last week's script built from the same
    # clicks. What needs no script is asserted: what the session holds for each click.
    from outline_tracker.segmenter import hf

    image = np.zeros((HD_H, HD_W, 3), np.uint8)
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
        assert coords == pytest.approx([x * 1024 / HD_W, y * 1024 / HD_H], abs=1e-3)

    # A second session, with the second click moved 1 px to the right: that click is 1024 / 1920 further
    # right there, and nothing else of the session differs.
    moved = hf.HFSegmenter("edgetam", "cpu", model=StubModel(*_stub_outputs(_random_logits())), processor=processor)
    moved.start(image, _prompts([CLICKS[0], (CLICKS[1][0] + 1.0, CLICKS[1][1]), CLICKS[2]]))
    changed = _differences(new.session, moved.session)
    assert len(changed) == 1 and changed[0].startswith("session.['point_inputs_per_obj'][1][0]['point_coords']")
    (x, y), (moved_x, moved_y) = _stored_point(new.session, 1)[0], _stored_point(moved.session, 1)[0]
    assert moved_x - x == pytest.approx(1024 / HD_W, abs=1e-3) and moved_y == y


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
    assert result.mask.shape == (min(rows.max() + 9, HD_H) - row0, min(cols.max() + 9, HD_W) - col0)


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
    cell_w, cell_h, radius, noise = HD_W / 256, HD_H / 256, 6.0, 0.2
    a, b = radius * cell_w, radius * cell_h
    area_sd = noise * math.sqrt(2 * math.pi * radius) * cell_w * cell_h
    center_sd = noise / math.sqrt(math.pi * radius)  # cells
    perimeter = 2 * math.pi * math.sqrt((a * a + b * b) / 2)  # at most
    u_px, v_px = position(result)
    area_px = mask_center(result.mask)[2]  # the number of pixels of the mask
    assert area_px == pytest.approx(math.pi * a * b, abs=0.5 * perimeter + 4 * area_sd)
    assert u_px == pytest.approx(100.5 * cell_w, abs=0.5 + 4 * center_sd * cell_w)
    assert v_px == pytest.approx(120.5 * cell_h, abs=0.5 + 4 * center_sd * cell_h)


def _assert_half_of_the_frame_each_and_no_two_alike(results) -> None:
    """The masks of `_random_logits`, held to what those logits are: for each of the three objects a
    draw of its own, cell by cell, from a distribution that is symmetric about 0.

    So each mask covers close to half of the frame. And the masks of two objects agree (a pixel is in
    both or in neither) on close to half of the frame: no object got the mask of another, which would
    agree everywhere.

    How close: the pixels are not independent, so the bound comes from the cells and not from the
    number of pixels. The processor interpolates the 256 x 256 cells of the model's grid to the frame
    (bilinear): a pixel's logit is a weighted mean, with weights of one sign, of the up to four cells
    around it. Cut the frame along the lines through the cells' centers, into rectangles of
    1920 / 256 = 7.5 px by 1080 / 256 = 4.21875 px (half as wide or high at the frame's border). All
    pixels of a rectangle come from the same four cells (two or one at the border), and a rectangle
    holds at most 8 x 5 = 40 pixel centers.
    - The share of a rectangle's pixels that are in the mask lies between 0 and 1, and its mean is
      1/2: the draw with every sign changed is as likely and gives the complement. So its variance
      is at most 1/4, and so is its covariance with the share of another rectangle.
    - Two rectangles that share no cell are independent, and a rectangle shares a cell only with
      itself and its 8 neighbours.
    The covered share of the frame, N = 1920 x 1080 pixels, is the mean of the rectangles' shares
    weighted with their pixels, so its variance is at most 9 * 40 / (4 N) = 90 / N: a standard
    deviation of at most 0.0066. Four are allowed: 2.6 % of the frame. (Independent pixels would give
    0.5 / sqrt(N) = 0.00035.) The same bound holds for the share on which two masks agree: changing
    every sign of one of the two draws turns agreement into disagreement.
    """
    most_px = math.ceil(HD_W / 256) * math.ceil(HD_H / 256)  # pixel centers in one rectangle, at most
    limit = 4 * math.sqrt(9 * most_px / (4 * HD_W * HD_H))
    full = [_full(result) for result in results]
    for name, mask in zip(NAMES, full):
        assert mask.mean() == pytest.approx(0.5, abs=limit), name
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert (full[a] == full[b]).mean() == pytest.approx(0.5, abs=limit), (NAMES[a], NAMES[b])


@pytest.mark.parametrize("make_logits", [_random_logits, _blob_logits], ids=["random", "blobs"])
def test_per_object_logits_give_the_reference_masks(processor, make_logits):
    # Until W1 step 5 the masks were compared with those that last week's script made of the same logits,
    # pixel for pixel. What needs no script is asserted: the rule that makes a mask of logits and crops it;
    # for the logits that draw a known disk, where that disk is; and for the random logits, how much of
    # the frame each mask covers and that the three objects have masks of their own.
    from outline_tracker.segmenter import hf

    image = np.zeros((HD_H, HD_W, 3), np.uint8)
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
        assert results[0].mask.size < 0.01 * HD_H * HD_W  # a real crop, not the whole frame
        assert [bool(result.mask.any()) for result in results] == [True, True, False]  # the third object is absent
        _assert_the_disk_of_the_blob_logits(results[0])
    else:
        _assert_half_of_the_frame_each_and_no_two_alike(results)
    # The session counts its own frames 0, 1, 2, ...: never the video's frame numbers.
    assert model.frame_indices == [0, 1, 2]
