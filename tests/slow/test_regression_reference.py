"""The Hugging Face backend reproduces last week's script (SPEC 13.3), and the selftest criterion
of SPEC 13.4 at the level of the segmenter. Every test here is slow: it needs torch.

Three groups:
1. no weights (seconds): the real processor and session, with the network replaced by a stand-in
   that returns given logits. The new `HFSegmenter` and last week's `TransformersSegmenter`
   (tests/reference, package `shrimp`) run side by side through their own code;
2. the real EdgeTAM on `cpu`: ONE loaded model is given to both, in one process and with the same
   thread count, on the selftest clip (one object) and on a three-object clip made with the same
   recipe. The first run downloads the model (56 MB) and converts it;
3. the selftest criterion (max error < 3 px against the true positions) on `cpu` and on `mps`.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_regression_reference.py -q -rP`.

With `OUTLINE_TRACKER_FREEZE=1` the tests of group 2 also write the frozen numbers of
tests/slow/test_frozen_reference.py: the new backend's positions on both clips
(tests/data/edgetam_cpu_positions.csv) and the weights' hash (tests/data/edgetam_weights.txt). Each
test hands its numbers over after its own assertions have passed, so the files hold only what last
week's code and the true positions confirmed in that run (tests/frozen_helpers.py). Without the
switch nothing is written.

Coordinates: positions are in px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the
full frame: `mask_center` of a result's cropped mask plus its offset (col0, row0). Clips are 1080p.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from frozen_helpers import (COLUMNS, POSITIONS, REPO, WEIGHTS, freeze_asked, frozen_text, header_lines, position_rows,
                            write_frozen)

from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import ObjectPrompt

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


def _leaves(value, path="session") -> list[str]:
    """The paths `_differences` walks, to show that the comparison reaches the prompt tensors."""
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
    from shrimp import segment

    from outline_tracker.segmenter import hf

    image = np.zeros((H, W, 3), np.uint8)
    reference = segment.TransformersSegmenter("edgetam", "cpu", model=StubModel(_random_logits(), None),
                                              processor=processor)
    reference.start(image, CLICKS)  # last week's single call for all objects
    new = hf.HFSegmenter("edgetam", "cpu", model=StubModel(*_stub_outputs(_random_logits())), processor=processor)
    new.start(image, _prompts())  # one call per object

    assert _differences(reference.session, new.session) == []
    leaves = _leaves(new.session)
    for obj_idx in range(3):
        assert f"session.['point_inputs_per_obj'][{obj_idx}][0]['point_coords']" in leaves
    assert "session.['obj_with_new_inputs']" in leaves and "session.['video_height']" in leaves
    assert new.session.obj_with_new_inputs == [1, 2, 3]

    # The comparison can fail: one click moved by 1 px.
    moved = hf.HFSegmenter("edgetam", "cpu", model=StubModel(*_stub_outputs(_random_logits())), processor=processor)
    moved.start(image, _prompts([CLICKS[0], (CLICKS[1][0] + 1.0, CLICKS[1][1]), CLICKS[2]]))
    found = _differences(reference.session, moved.session)
    assert len(found) == 1 and found[0].startswith("session.['point_inputs_per_obj'][1][0]['point_coords']")


def _stub_outputs(logits):
    import torch

    return logits, torch.tensor([5.0, 0.5, -3.0])  # presence logits, one per object


@pytest.mark.parametrize("make_logits", [_random_logits, _blob_logits], ids=["random", "blobs"])
def test_per_object_logits_give_the_reference_masks(processor, make_logits):
    from shrimp import segment

    from outline_tracker.segmenter import hf

    image = np.zeros((H, W, 3), np.uint8)
    logits, scores = _stub_outputs(make_logits())
    old_model, new_model = StubModel(logits, None), StubModel(logits, scores)
    reference = segment.TransformersSegmenter("edgetam", "cpu", model=old_model, processor=processor)
    new = hf.HFSegmenter("edgetam", "cpu", model=new_model, processor=processor)

    for step in range(3):
        old_masks = reference.start(image, CLICKS) if step == 0 else reference.step(image)  # binarize=True, batched
        results = new.start(image, _prompts()) if step == 0 else new.step(image)
        assert [r.obj_id for r in results] == NAMES
        assert [r.score for r in results] == pytest.approx([5.0, 0.5, -3.0])
        for result, old_mask in zip(results, old_masks):
            assert np.array_equal(_full(result), old_mask)
            assert result.mask.dtype == bool and result.logits.dtype == np.float32
            assert np.array_equal(result.logits > 0, result.mask)
            if old_mask.any():  # cropped to the bounding box +/- 8 px, clipped to the frame
                rows, cols = np.nonzero(old_mask)
                col0, row0 = max(cols.min() - 8, 0), max(rows.min() - 8, 0)
                assert result.offset == (col0, row0)
                assert result.mask.shape == (min(rows.max() + 9, H) - row0, min(cols.max() + 9, W) - col0)
            else:
                assert result.mask.shape == (0, 0)
    if make_logits is _blob_logits:
        assert [bool(m.any()) for m in old_masks] == [True, True, False]  # the third object is absent
        assert results[0].mask.size < 0.01 * H * W  # a real crop, not the whole frame
    # The session counts its own frames 0, 1, 2, ...: never the video's frame numbers.
    assert old_model.frame_indices == new_model.frame_indices == [0, 1, 2]


# ---------------------------------------------------------------------------------------------
# 2. The real EdgeTAM on cpu: one loaded model for the reference and for the new backend


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model("edgetam")


def _track(segmenter, video, frames, prompts) -> SimpleNamespace:
    """Drive a segmenter over the frames of a clip, as last week's loop did.

    Returns frames (video frame numbers), results (one list of MaskResult per frame) and seconds
    (per frame, decoding included).
    """
    from shrimp import segment

    done, results, seconds = [], [], []
    t0 = time.perf_counter()
    for i, (frame, rgb) in enumerate(segment.iter_rgb_frames(video, frames)):
        results.append(segmenter.start(rgb, prompts) if i == 0 else segmenter.step(rgb))
        done.append(frame)
        seconds.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    return SimpleNamespace(frames=done, results=results, seconds=np.array(seconds))


def _timing(seconds: np.ndarray) -> str:
    return f"{seconds.mean():.2f} s per frame (mean of {len(seconds)}; median {np.median(seconds):.2f})"


FREEZE_COMMAND = ("OUTLINE_TRACKER_FREEZE=1 uv run pytest -m slow tests/slow/test_regression_reference.py -q -rP "
                  "-p no:cacheprovider")  # what writes the two frozen files
_handed_over = {}  # what the tests of a freeze run have handed over: "selftest", "three_ellipses", "weights"


def _hand_over_positions(clip, frames, names, positions, difference, error) -> None:
    """Take one clip's positions of the new backend for the frozen table, after the assertions of its
    test have passed. Does nothing unless this run was asked to freeze (tests/frozen_helpers.py).

    positions[i][k]: (u_px, v_px) of the object names[k] on the frame frames[i], px in Tracker's
    convention in the full frame. difference, error: each found row's distance in px from last week's
    position and from the true center. What these two checks do not confirm is refused: a row more
    than 0.01 px from last week's, or 3 px or more from the truth.
    """
    if not freeze_asked():
        return
    worst, farthest = float(np.nanmax(difference)), float(np.nanmax(error))
    assert worst <= 0.01 and farthest < 3.0, (
        f"Nothing was frozen: {clip} is {worst:.4f} px from last week's and {farthest:.3f} px from the truth.")
    _handed_over[clip] = SimpleNamespace(rows=position_rows(clip, frames, names, positions), worst=worst,
                                         farthest=farthest)
    _freeze_when_complete()


def _hand_over_weights(sha256: str, size: int) -> None:
    """Take the hash and the size in bytes of the weights file that the loaded model was read from,
    after the test has hashed the file again. Does nothing unless this run was asked to freeze."""
    if not freeze_asked():
        return
    _handed_over["weights"] = SimpleNamespace(sha256=sha256, size=size)
    _freeze_when_complete()


def _freeze_when_complete() -> None:
    """Write the two frozen files once both clips and the weights have been handed over in this run:
    one loaded model on cpu made all of it. A run of a part of this file writes nothing."""
    if set(_handed_over) != {"selftest", "three_ellipses", "weights"}:
        return
    import torch

    clips = {name: _handed_over[name] for name in ("selftest", "three_ellipses")}
    weights = _handed_over["weights"]
    rows = [row for clip in clips.values() for row in clip.rows]
    write_frozen(POSITIONS, frozen_text(header_lines(FREEZE_COMMAND, {
        "device": "cpu",
        "torch threads": str(torch.get_num_threads()),
        "weights sha256": weights.sha256,
        "agreement with the template": f"{max(clip.worst for clip in clips.values()):.4f} px at most over the "
                                       f"{len(rows)} rows (limit 0.01 px), asserted in the run that wrote this file",
        "largest distance from the true centers": "; ".join(f"{name} {clip.farthest:.3f} px"
                                                            for name, clip in clips.items()) + " (limit 3 px)",
    }), [COLUMNS, *rows]))
    write_frozen(WEIGHTS, frozen_text(header_lines(FREEZE_COMMAND, {
        "model": "facebook/EdgeTAM (edgetam), converted on first use; the tool loads the converted model.safetensors",
        "sha256, bytes": "of that model.safetensors; HFSegmenter.weights_sha256 reports this hash, and the run that "
                         "wrote this file took it again from the file's bytes",
        "original checkpoint, original revision": "Meta's edgetam.pt as the Hugging Face cache of this machine held "
                                                  "it; for information, never asserted",
    }), [f"sha256: {weights.sha256}", f"bytes: {weights.size}", *_original_checkpoint()]))
    print(f"FROZEN {len(rows)} rows in {POSITIONS.relative_to(REPO).as_posix()}; the weights' hash in "
          f"{WEIGHTS.relative_to(REPO).as_posix()}")


def _original_checkpoint() -> list[str]:
    """Meta's checkpoint as the Hugging Face cache of this computer holds it: a line with its name and
    its size in bytes, and a line with its revision. Nothing when the cache does not hold it."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache("facebook/EdgeTAM", "edgetam.pt")
    if not isinstance(found, str):
        return []
    return [f"original checkpoint: edgetam.pt of facebook/EdgeTAM, {Path(found).stat().st_size} bytes",
            f"original revision: {Path(found).parent.name}"]


@pytest.fixture(scope="module")
def selftest_runs(loaded, tmp_path_factory):
    """Last week's selftest with the reference segmenter, then the new backend on the same clip."""
    from shrimp import segment

    from outline_tracker.segmenter import hf

    model, processor = loaded
    folder = tmp_path_factory.mktemp("selftest")
    reference = segment.TransformersSegmenter("edgetam", "cpu", model=model, processor=processor)
    log = []
    summary = segment.selftest(model="edgetam", device="cpu", segmenter=reference, folder=folder, log=log.append)

    video, export = folder / "selftest_tracker.mp4", folder / "selftest.csv"
    plan = segment.make_plan(export, fps=240.0, video=video, manifest=folder / "none.csv")  # as track_video did
    segmenter = hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor)
    run = _track(segmenter, video, plan.frames, [ObjectPrompt("selftest", [plan.points_px[0]], [1])])
    return SimpleNamespace(
        video=video, plan=plan, segmenter=segmenter, run=run, summary=summary,
        truth=pd.read_csv(export, skiprows=1),  # the click and the true positions
        reference=pd.read_csv(folder / "edgetam" / "selftest.csv", skiprows=1),  # last week's positions
        positions=np.array([_position(results[0])[:2] for results in run.results]),
    )


def test_selftest_clip_positions_equal_the_reference_csv(selftest_runs):
    s = selftest_runs
    assert s.plan.frames == list(range(0, 40, 2))
    assert s.run.frames == s.plan.frames == s.reference["frame"].tolist()
    assert not s.reference["pixelx"].isna().iloc[0]  # the reference found the test shrimp at the click

    old = s.reference[["pixelx", "pixely"]].to_numpy(float)
    lost_old, lost_new = np.isnan(old[:, 0]), np.isnan(s.positions[:, 0])
    difference = np.hypot(*(s.positions - old).T)
    print(f"VALIDATION regression, selftest clip (1 object, 20 frames, cpu): max difference "
          f"{np.nanmax(difference):.4f} px; lost frames: reference {int(lost_old.sum())}, new {int(lost_new.sum())}")
    assert np.array_equal(lost_new, lost_old)
    assert np.nanmax(difference) <= 0.01
    # Last week's code agrees. Only now, and only when asked, the positions go to the frozen table.
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    _hand_over_positions("selftest", s.plan.frames, ["selftest"], s.positions[:, None, :], difference,
                         np.hypot(*(s.positions - true).T))


def test_selftest_clip_within_3_px_of_truth_on_cpu(selftest_runs):
    s = selftest_runs
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    error = np.hypot(*(s.positions - true).T)
    print(f"VALIDATION selftest criterion, cpu: max error {np.nanmax(error):.3f} px (new backend); "
          f"{_timing(s.run.seconds)}; reference on the same model: max error "
          f"{s.summary['max_error_px']:.3f} px, {s.summary['seconds_per_frame']:.2f} s per frame")
    assert s.summary["ok"]  # last week's code passes its own check here
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
    _hand_over_weights(segmenter.weights_sha256, saved.stat().st_size)  # frozen only when asked


def _write_three_ellipse_clip(path) -> list[list[tuple[float, float]]]:
    """The selftest's recipe (1080p, 40 frames, mp4v, noise sigma 3, dark ellipses with semi-axes
    8 and 3 px along their motion) with three ellipses on a background whose R, G and B differ.

    Returns truth[object][frame] = (x, y) in array coordinates (pixel centers at integers).
    """
    n = 40
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 205.0 - 15.0 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.49 * H) ** 2
    tint = np.array([0.80, 0.90, 1.00])  # blue, green, red
    starts = [(500.0, 300.0), (1000.0, 600.0), (1400.0, 350.0)]  # at least 300 px apart at all times
    steps = [(0.6, 0.2), (-0.5, 0.3), (0.2, -0.6)]  # px per frame
    truth = [[(x0 + dx * f, y0 + dy * f) for f in range(n)] for (x0, y0), (dx, dy) in zip(starts, steps)]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240, (W, H))
    for f in range(n):
        img = base[:, :, None] * tint
        for track, (dx, dy) in zip(truth, steps):
            x, y = track[f]
            angle = float(np.degrees(np.arctan2(dy, dx)))
            cv2.ellipse(img, (int(round(x * 16)), int(round(y * 16))), (8 * 16, 3 * 16), angle, 0, 360,
                        tuple(70.0 * tint), -1, cv2.LINE_AA, 4)
        img = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
        out.write(img)
    out.release()
    return truth


def test_three_objects_match_the_reference(loaded, tmp_path):
    from shrimp import segment

    from outline_tracker.segmenter import hf

    model, processor = loaded
    video = tmp_path / "three_tracker.mp4"
    truth = _write_three_ellipse_clip(video)
    frames = list(range(0, 40, 2))
    clicks = [(track[0][0] + 0.5, track[0][1] + 0.5) for track in truth]  # Tracker's convention (+0.5)
    separations = [np.hypot(a[f][0] - b[f][0], a[f][1] - b[f][1])
                   for a, b in [(truth[0], truth[1]), (truth[1], truth[2]), (truth[0], truth[2])] for f in range(40)]
    assert min(separations) >= 300.0
    _, first = next(segment.iter_rgb_frames(video, [0]))
    red, green, blue = first.reshape(-1, 3).mean(axis=0)
    assert red > green + 10 and green > blue + 10  # the three channels differ, also after decoding

    reference = segment.TransformersSegmenter("edgetam", "cpu", model=model, processor=processor)
    old = []  # per frame, the three full-frame masks, bit-packed
    for i, (_, rgb) in enumerate(segment.iter_rgb_frames(video, frames)):
        masks = reference.start(rgb, clicks) if i == 0 else reference.step(rgb)
        old.append([np.packbits(mask) for mask in masks[:3]])
    run = _track(hf.HFSegmenter("edgetam", "cpu", model=model, processor=processor), video, frames,
                 _prompts(clicks))

    assert run.frames == frames and len(old) == len(frames)
    difference = np.full((len(frames), 3), np.nan)
    error = np.full((len(frames), 3), np.nan)
    lost_old, lost_new = np.zeros((len(frames), 3), bool), np.zeros((len(frames), 3), bool)
    changed_pixels = 0
    for i, (packed, results) in enumerate(zip(old, run.results)):
        assert [r.obj_id for r in results] == NAMES
        for k, result in enumerate(results):
            old_mask = np.unpackbits(packed[k], count=H * W).reshape(H, W).astype(bool)
            old_u, old_v, _ = segment.mask_center(old_mask)
            u, v, _ = _position(result)
            lost_old[i, k], lost_new[i, k] = np.isnan(old_u), np.isnan(u)
            difference[i, k] = np.hypot(u - old_u, v - old_v)
            error[i, k] = np.hypot(u - (truth[k][frames[i]][0] + 0.5), v - (truth[k][frames[i]][1] + 0.5))
            changed_pixels += int(np.count_nonzero(_full(result) != old_mask))
    found = ~(lost_old | lost_new)
    print(f"VALIDATION regression, three-ellipse clip (3 objects, 20 frames, cpu): max difference "
          f"{difference[found].max():.4f} px; mask pixels that differ: {changed_pixels}; lost frames: reference "
          f"{int(lost_old.sum())}, new {int(lost_new.sum())} of {lost_old.size}; max error against the true "
          f"positions {np.nanmax(error):.3f} px; {_timing(run.seconds)}")
    assert not lost_old[0].any()  # the reference found every ellipse at its click
    assert np.array_equal(lost_new, lost_old)
    assert difference[found].max() <= 0.01
    # the selftest criterion (SPEC 13.4) for this clip too: every found position under 3 px from the
    # true center, which the clip's recipe gives
    assert error[found].max() < 3.0
    positions = np.array([[_position(result)[:2] for result in results] for results in run.results])
    _hand_over_positions("three_ellipses", frames, NAMES, positions, difference, error)


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
