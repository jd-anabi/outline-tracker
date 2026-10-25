"""The Hugging Face backend at the level of the segmenter, with the real EdgeTAM: what it reports and the selftest
criterion of SPEC 13.4. Every test here is slow: it needs torch, and each loads the weights of the model.

Two groups:
1. the real EdgeTAM on `cpu`: ONE loaded model, on the selftest clip (one object) and on a
   three-object clip made with the same recipe: what the segmenter reports, the clip's own
   properties, and the selftest criterion (max error < 3 px against the true positions). The first
   run downloads the model (56 MB) and converts it;
2. the selftest criterion on `mps`.

The tests with the real processor and a stand-in for the network, which need no weights, are in
tests/slow/test_regression_reference_without_weights.py.

Until W1 step 5 these tests, and those of that file, ran last week's script (`shrimp.segment`) beside the new
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
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pipeline_helpers import NAMES, _prompts, position, write_three_ellipse_clip

from outline_tracker.synthetic import clips as synthetic
from outline_tracker.core import tracker_io
from outline_tracker.segmenter.base import ObjectPrompt
from outline_tracker.video.reader import iter_rgb_frames

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------------------------
# 1. The real EdgeTAM on cpu: one loaded model


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
        positions=np.array([position(results[0]) for results in run.results]),
    )


@pytest.mark.weights
def test_selftest_clip_within_3_px_of_truth_on_cpu(selftest_runs):
    s = selftest_runs
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    error = np.hypot(*(s.positions - true).T)
    print(f"VALIDATION selftest criterion, cpu: max error {np.nanmax(error):.3f} px; {_timing(s.run.seconds)}")
    assert np.isfinite(error).all() and error.max() < 3.0


@pytest.mark.weights
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


@pytest.mark.weights
def test_three_objects_match_the_reference(loaded, tmp_path):
    # The reference is the frozen table, and the comparison with it is asserted once, in
    # tests/slow/test_frozen_reference.py::test_three_ellipses_equal_the_frozen_numbers. Here: the clip's
    # own properties, and the selftest criterion for this clip, asserted directly.
    from outline_tracker.segmenter import hf

    model, processor = loaded
    video = tmp_path / "three_tracker.mp4"
    truth = write_three_ellipse_clip(video)
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
    positions = np.array([[position(result) for result in results] for results in run.results])
    true = np.array([[(track[frame][0] + 0.5, track[frame][1] + 0.5) for track in truth] for frame in frames])
    error = np.hypot(positions[..., 0] - true[..., 0], positions[..., 1] - true[..., 1])
    found = ~np.isnan(error)
    print(f"VALIDATION three-ellipse clip (3 objects, 20 frames, cpu): max error against the true positions "
          f"{error[found].max():.3f} px; lost: {int((~found).sum())} of {found.size}; {_timing(run.seconds)}")
    # the selftest criterion (SPEC 13.4) for this clip too: every found position under 3 px from the
    # true center, which the clip's recipe gives
    assert error[found].max() < 3.0


# ---------------------------------------------------------------------------------------------
# 2. The selftest criterion on the Apple GPU


@pytest.mark.weights
def test_selftest_clip_within_3_px_of_truth_on_mps(selftest_runs):
    import torch

    from outline_tracker.segmenter import hf

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    s = selftest_runs
    segmenter = hf.HFSegmenter("edgetam", "mps")  # its own copy of the model: the shared one stays on cpu
    run = _track(segmenter, s.video, s.plan.frames, [ObjectPrompt("selftest", [s.plan.points_px[0]], [1])])
    positions = np.array([position(results[0]) for results in run.results])
    true = s.truth[["pixelx", "pixely"]].to_numpy(float)
    error = np.hypot(*(positions - true).T)
    print(f"VALIDATION selftest criterion, mps: max error {np.nanmax(error):.3f} px; {_timing(run.seconds)}; "
          f"device at the end: {segmenter.device}; max distance from the cpu positions "
          f"{np.nanmax(np.hypot(*(positions - s.positions).T)):.3f} px")
    assert segmenter.device == "mps"  # it did not fall back to the processor
    assert np.isfinite(error).all() and error.max() < 3.0
