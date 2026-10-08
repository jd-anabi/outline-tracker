"""The whole pipeline with the real model gives the frozen positions (SPEC 13.3, 6.2).

`from_tracker` (a session from the Tracker export, `tracking.run_job`, `export.export_all`) with the
real EdgeTAM on `cpu`: coarse, the whole frame (a Tracker export has no dish circle), no overlay.
Two clips:
1. last week's selftest clip (`synthetic.selftest_clip`: 1080p, one dark ellipse) with its export of
   one track on frames 0, 2, ..., 38;
2. the three-ellipse clip of tests/slow/pipeline_helpers.py with a `#multi` start file:
   three point masses marked once, on frame 0, tracked for 40 / 240 s at step 2.

What is compared: the Tracker-format files `<run>/edgetam/<id>.csv` with the frozen table,
tests/data/edgetam_cpu_positions.csv: the names, the frames, the same rows lost, and pixelx, pixely
within 0.01 px on the machine that froze the table (tests/frozen_helpers.py says what is asserted on
another). A file has three decimals and the table four, so up to 0.0005 px per axis is rounding.
Beside that: what the run folder holds. Every test here is slow.

Until W1 step 5 these tests ran last week's script (`shrimp.segment.track_video`) on the same clips
and compared the files of the two; the test names still say so. The script has left the
repository, and the frozen table holds the positions of the run in which it agreed within 0.01 px
(docs/VALIDATION.md, section 7).

Lines printed with the prefix `VALIDATION` are measured numbers; show them with
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
from frozen_helpers import _compare_with_the_frozen_positions
from helpers import tracker_map
from pipeline_helpers import _write_three_ellipse_clip

from outline_tracker import synthetic
from outline_tracker.from_tracker import from_tracker

pytestmark = pytest.mark.slow

MODEL = "edgetam"
FPS = 240.0  # fps_true of both clips, frames per second
FRAMES = list(range(0, 40, 2))  # the tracked frames of both clips
# SPEC 8.1 without probes.csv (there are no probes) and overlay.mp4 (not asked for); MODEL is the
# folder of the Tracker-format files
RUN_FOLDER = {"session.json", "positions.csv", MODEL, "shapes.csv", "radial.csv", "outlines.npz", "results.npz",
              "run.log", "README.txt"}


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model(MODEL)


def _track_through_the_pipeline(loaded, video, export, folder, **options) -> SimpleNamespace:
    """Track one clip with the loaded model on cpu through `from_tracker`, into the run folder
    `<folder>/run`. `options` (seconds in s, step in frames) go to `from_tracker`.

    Returns files (the run's Tracker-format files), run (the run folder), seconds (s per tracked
    frame, as the run reports it), threads (torch's thread count) and weights_sha256 (the SHA-256 of
    the weights file that the model was loaded from).
    """
    import torch

    from outline_tracker.segmenter import hf

    model, processor = loaded
    segmenter = hf.HFSegmenter(MODEL, "cpu", model=model, processor=processor)
    new = from_tracker(video, export, fps=FPS, out=folder / "run", student="test", model=MODEL, device="cpu",
                       overlay=False, segmenter=segmenter, log=lambda line: None, **options)
    assert segmenter.device == "cpu"
    return SimpleNamespace(files=new.files, run=new.run_folder, seconds=new.seconds_per_frame,
                           threads=torch.get_num_threads(), weights_sha256=segmenter.weights_sha256)


def _compare(what, tracked, clip, names, true) -> None:
    """Compare the Tracker-format files of a run with the frozen positions of its clip, and assert what
    SPEC 13.3 asks: one file per name, the frames 0, 2, ..., 38 in each, the same rows lost (pixelx
    and pixely empty) and every other row's (pixelx, pixely) within 0.01 px of the frozen position,
    on the machine that froze the table (`_compare_with_the_frozen_positions`, which also prints what
    it measured).

    tracked: what `_track_through_the_pipeline` returned. clip: the clip's name in the frozen table.
    true: [i][k] = (u_px, v_px), the true center of names[k] on FRAMES[i], px in Tracker's convention
    in the full frame. The mm and time columns are compared by tests/test_from_tracker_port.py, with
    a stand-in model.
    """
    assert [path.name for path in tracked.files] == [f"{name}.csv" for name in names]
    tables = [pd.read_csv(path, skiprows=1) for path in tracked.files]
    assert [table["frame"].tolist() for table in tables] == [FRAMES] * len(names)
    # found[i][k] = (pixelx, pixely) of names[k] on FRAMES[i]; the empty cells of a lost row are read as NaN
    found = np.stack([table[["pixelx", "pixely"]].to_numpy(float) for table in tables], axis=1)
    _compare_with_the_frozen_positions(what, clip, FRAMES, names, found, true, tracked.weights_sha256)


def test_selftest_clip_through_the_pipeline_equals_last_weeks_files(loaded, tmp_path):
    # last week's files live on as the frozen table: its 20 rows of this clip
    clip = synthetic.selftest_clip(tmp_path / "clip")
    assert clip["frames"] == FRAMES
    tracked = _track_through_the_pipeline(loaded, clip["video"], clip["export"], tmp_path)

    assert tracked.files == [tmp_path / "run" / MODEL / "selftest.csv"]
    true = np.stack([clip["pixelx"], clip["pixely"]], axis=1)[:, None, :]  # [frame][the one object] = (u_px, v_px)
    _compare("selftest clip (1 object, 20 frames), pipeline on cpu", tracked, "selftest", ["selftest"], true)
    # the full set of SPEC 8.1, each file with something in it, and nothing else
    assert {path.name for path in tracked.run.iterdir()} == RUN_FOLDER
    assert [path.name for path in (tracked.run / MODEL).iterdir()] == ["selftest.csv"]
    assert all(path.stat().st_size > 0 for path in tracked.run.iterdir() if path.is_file())
    print(f"VALIDATION pipeline, selftest clip (1 object, 20 frames, cpu, {tracked.threads} threads): "
          f"{tracked.seconds:.2f} s per frame; run folder: "
          f"{', '.join(sorted(path.name for path in tracked.run.iterdir()))}")


def test_three_ellipses_from_a_multi_start_file_equal_last_weeks_files(loaded, tmp_path):
    # last week's files live on as the frozen table: its 60 rows of this clip
    video = tmp_path / "three_tracker.mp4"
    truth = _write_three_ellipse_clip(video)  # truth[object][frame] = (x, y), pixel centers at whole numbers
    marks = {name: (track[0][0] + 0.5, track[0][1] + 0.5) for name, track in zip("ABC", truth)}
    # Tracker's map with the origin in the middle of the frame, 0.05 mm per px, y up
    start = write_start_file(tmp_path / "ana" / "extra" / "start.csv", marks, frame=0,
                             to_mm=lambda px, py: tracker_map(px, py, 0.0, (960.0, 540.0)))
    assert start.read_text().startswith("#multi:")

    tracked = _track_through_the_pipeline(loaded, video, start, tmp_path, seconds=40 / FPS, step=2)

    assert [path.name for path in tracked.files] == ["A.csv", "B.csv", "C.csv"]
    assert sorted(path.name for path in (tracked.run / MODEL).iterdir()) == ["A.csv", "B.csv", "C.csv"]
    true = np.array([[(track[frame][0] + 0.5, track[frame][1] + 0.5) for track in truth] for frame in FRAMES])
    _compare("three-ellipse clip with a #multi start file (3 objects, 20 frames), pipeline on cpu", tracked,
             "three_ellipses", ["A", "B", "C"], true)
    print(f"VALIDATION pipeline, three-ellipse clip with a #multi start file (3 objects, 20 frames, cpu, "
          f"{tracked.threads} threads): {tracked.seconds:.2f} s per frame")
