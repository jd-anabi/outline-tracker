"""The whole pipeline reproduces last week's script with the real model (SPEC 13.3, 6.2).

`from_tracker` (a session from the Tracker export, `tracking.run_job`, `export.export_all`) against
last week's `shrimp.segment.track_video` with last week's `TransformersSegmenter` (the unmodified
copy in tests/reference). ONE loaded EdgeTAM serves both, on `cpu`, in one process and so with the
same number of torch threads; both get the same video, export, fps_true and options, coarse, the
whole frame (a Tracker export has no dish circle), no overlay. Two clips:
1. last week's selftest clip (`synthetic.selftest_clip`: 1080p, one dark ellipse) with its export of
   one track on frames 0, 2, ..., 38;
2. the three-ellipse clip of tests/slow/test_regression_reference.py with a `#multi` start file:
   three point masses marked once, on frame 0, tracked for 40 / 240 s at step 2.

What is compared: the Tracker-format files `<run>/edgetam/<id>.csv` with the files last week's
script wrote: same names, same frames, the same rows lost, and pixelx, pixely within 0.01 px. Both
sides write three decimals, so a difference is a multiple of 0.001 px. Every test here is slow.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_regression_pipeline.py -q -rP`.

Coordinates: pixelx, pixely are px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in
the full 1920 x 1080 frame; frames are video frame numbers; fps_true is 240 frames per second.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from from_tracker_helpers import write_start_file
from test_regression_reference import _write_three_ellipse_clip
from test_tracker_io import tracker_map

from outline_tracker import synthetic
from outline_tracker.from_tracker import from_tracker

pytestmark = pytest.mark.slow

MODEL = "edgetam"
FPS = 240.0  # fps_true of both clips, frames per second
# SPEC 8.1 without probes.csv (there are no probes) and overlay.mp4 (not asked for); MODEL is the
# folder of the Tracker-format files
RUN_FOLDER = {"session.json", "positions.csv", MODEL, "shapes.csv", "radial.csv", "outlines.npz", "results.npz",
              "run.log", "README.txt"}


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model(MODEL)


def _track_both_ways(loaded, video, export, folder, **options) -> SimpleNamespace:
    """Track one clip twice with the one loaded model on cpu: last week's `track_video` with last
    week's segmenter (files in `<folder>/reference/edgetam`), then `from_tracker` with the new
    backend (run folder `<folder>/run`). `options` (seconds in s, step in frames) go to both.

    Returns old (last week's files), new (the run's Tracker-format files), run (the run folder),
    old_s, new_s (s per tracked frame, as each side reports it) and threads (torch's thread count).
    """
    import torch
    from shrimp import segment

    from outline_tracker.segmenter import hf

    model, processor = loaded
    reference = segment.TransformersSegmenter(MODEL, "cpu", model=model, processor=processor)
    old = segment.track_video(video, export, model=MODEL, fps=FPS, out=folder / "reference" / MODEL, device="cpu",
                              segmenter=reference, overlay=False, manifest=folder / "none.csv",
                              log=lambda line: None, **options)
    segmenter = hf.HFSegmenter(MODEL, "cpu", model=model, processor=processor)
    new = from_tracker(video, export, fps=FPS, out=folder / "run", student="test", model=MODEL, device="cpu",
                       overlay=False, segmenter=segmenter, log=lambda line: None, **options)
    assert segmenter.device == reference.device == "cpu"
    return SimpleNamespace(old=old["files"], new=new.files, run=new.run_folder, old_s=old["seconds_per_frame"],
                           new_s=new.seconds_per_frame, threads=torch.get_num_threads())


def _compare(old_files, new_files, frames) -> SimpleNamespace:
    """Compare the new Tracker-format files with last week's, track by track, and assert what
    SPEC 13.3 asks: the same file names, the frames `frames` in both, the same rows lost (pixelx
    and pixely empty), and every other row's (pixelx, pixely) within 0.01 px of last week's.

    Returns worst (the largest distance in px), rows (rows compared), lost_old, lost_new, and
    same_bytes: how many of the new files equal last week's byte for byte (reported, not asserted:
    the mm and time columns are compared by tests/test_from_tracker_port.py).
    """
    assert [path.name for path in new_files] == [path.name for path in old_files]
    worst, rows, lost_old, lost_new = 0.0, 0, 0, 0
    same_bytes = sum(old.read_bytes() == new.read_bytes() for old, new in zip(old_files, new_files))
    for old_path, new_path in zip(old_files, new_files):
        old, new = pd.read_csv(old_path, skiprows=1), pd.read_csv(new_path, skiprows=1)
        assert new["frame"].tolist() == old["frame"].tolist() == frames
        gone_old = old[["pixelx", "pixely"]].isna().to_numpy()
        gone_new = new[["pixelx", "pixely"]].isna().to_numpy()
        assert not gone_old[0].any()  # last week's script found the object at its click
        assert np.array_equal(gone_new, gone_old)
        found = ~gone_old.any(axis=1)
        distance = np.hypot(*(new[["pixelx", "pixely"]].to_numpy(float) - old[["pixelx", "pixely"]].to_numpy(float)).T)
        assert distance[found].max() <= 0.01, f"{new_path.name}: {distance[found].max():.4f} px from last week's"
        worst, rows = max(worst, float(distance[found].max())), rows + int(found.sum())
        lost_old, lost_new = lost_old + int((~found).sum()), lost_new + int(gone_new.any(axis=1).sum())
    return SimpleNamespace(worst=worst, rows=rows, lost_old=lost_old, lost_new=lost_new, same_bytes=same_bytes)


def test_selftest_clip_through_the_pipeline_equals_last_weeks_files(loaded, tmp_path):
    clip = synthetic.selftest_clip(tmp_path / "clip")
    both = _track_both_ways(loaded, clip["video"], clip["export"], tmp_path)

    assert both.new == [tmp_path / "run" / MODEL / "selftest.csv"]
    numbers = _compare(both.old, both.new, frames=list(range(0, 40, 2)))
    # the full set of SPEC 8.1, each file with something in it, and nothing else
    assert {path.name for path in both.run.iterdir()} == RUN_FOLDER
    assert [path.name for path in (both.run / MODEL).iterdir()] == ["selftest.csv"]
    assert all(path.stat().st_size > 0 for path in both.run.iterdir() if path.is_file())
    print(f"VALIDATION pipeline regression, selftest clip (1 object, 20 frames, cpu, {both.threads} threads): max "
          f"difference {numbers.worst:.4f} px over {numbers.rows} rows; lost rows: reference {numbers.lost_old}, "
          f"new {numbers.lost_new}; files equal byte for byte: {numbers.same_bytes} of 1; s per frame: reference "
          f"{both.old_s:.2f}, pipeline {both.new_s:.2f}; run folder: "
          f"{', '.join(sorted(path.name for path in both.run.iterdir()))}")


def test_three_ellipses_from_a_multi_start_file_equal_last_weeks_files(loaded, tmp_path):
    video = tmp_path / "three_tracker.mp4"
    truth = _write_three_ellipse_clip(video)  # truth[object][frame] = (x, y), pixel centers at whole numbers
    marks = {name: (track[0][0] + 0.5, track[0][1] + 0.5) for name, track in zip("ABC", truth)}
    # Tracker's map with the origin in the middle of the frame, 0.05 mm per px, y up
    start = write_start_file(tmp_path / "ana" / "extra" / "start.csv", marks, frame=0,
                             to_mm=lambda px, py: tracker_map(px, py, 0.0, (960.0, 540.0)))
    assert start.read_text().startswith("#multi:")

    both = _track_both_ways(loaded, video, start, tmp_path, seconds=40 / FPS, step=2)

    assert [path.name for path in both.new] == ["A.csv", "B.csv", "C.csv"]
    assert sorted(path.name for path in (both.run / MODEL).iterdir()) == ["A.csv", "B.csv", "C.csv"]
    numbers = _compare(both.old, both.new, frames=list(range(0, 40, 2)))
    print(f"VALIDATION pipeline regression, three-ellipse clip with a #multi start file (3 objects, 20 frames, cpu, "
          f"{both.threads} threads): max difference {numbers.worst:.4f} px over {numbers.rows} rows; lost rows: "
          f"reference {numbers.lost_old}, new {numbers.lost_new}; files equal byte for byte: {numbers.same_bytes} of "
          f"3; s per frame: reference {both.old_s:.2f}, pipeline {both.new_s:.2f}")
