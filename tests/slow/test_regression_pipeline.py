"""The whole pipeline with the real model: what a run leaves in its folder (SPEC 8.1, 6.2).

`from_tracker` (a session from the Tracker export, `tracking.run_job`, `export.export_all`) with the
real EdgeTAM on `cpu`: coarse, the whole frame (a Tracker export has no dish circle), no overlay.
Two clips:
1. last week's selftest clip (`synthetic.selftest_clip`: 1080p, one dark ellipse) with its export of
   one track on frames 0, 2, ..., 38;
2. the three-ellipse clip of tests/slow/pipeline_helpers.py with a `#multi` start file:
   three point masses marked once, on frame 0, tracked for 40 / 240 s at step 2.

What is asserted: the files that the run returns and the files that its folder holds, and for the
first clip that each of them has something in it. Every test here is slow.

Until W1 step 5 these tests ran last week's script (`shrimp.segment.track_video`) on the same clips
and compared the files of the two; the test names still say so. The script has left the
repository, and the frozen table, tests/data/edgetam_cpu_positions.csv, holds the positions of the
run in which it agreed within 0.01 px (docs/VALIDATION.md, section 7). The comparison of the run's
Tracker-format files with that table (SPEC 13.3: the names, the frames, the same rows lost, pixelx
and pixely) is asserted once, in tests/slow/test_frozen_reference.py, for both clips; no test here
reads the table.

Lines printed with the prefix `VALIDATION` are measured numbers; show them with
`uv run pytest -m slow tests/slow/test_regression_pipeline.py -q -rP`.

Units: the clips are 1920 x 1080 px; a click is in px in Tracker's convention (pixel centers at
+0.5, SPEC 3.1); frames are video frame numbers; fps_true is 240 frames per second.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from from_tracker_helpers import write_start_file
from helpers import tracker_map
from pipeline_helpers import write_three_ellipse_clip

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


# Overrides the shared `loaded` of tests/slow/conftest.py: this one loads the model that `MODEL` names.
@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model(MODEL)


def _track_through_the_pipeline(loaded, video, export, folder, **options) -> SimpleNamespace:
    """Track one clip with the loaded model on cpu through `from_tracker`, into the run folder
    `<folder>/run`. `options` (seconds in s, step in frames) go to `from_tracker`.

    Returns files (the run's Tracker-format files), run (the run folder), seconds (s per tracked
    frame, as the run reports it) and threads (torch's thread count).
    """
    import torch

    from outline_tracker.segmenter import hf

    model, processor = loaded
    segmenter = hf.HFSegmenter(MODEL, "cpu", model=model, processor=processor)
    new = from_tracker(video, export, fps=FPS, out=folder / "run", student="test", model=MODEL, device="cpu",
                       overlay=False, segmenter=segmenter, log=lambda line: None, **options)
    assert segmenter.device == "cpu"
    return SimpleNamespace(files=new.files, run=new.run_folder, seconds=new.seconds_per_frame,
                           threads=torch.get_num_threads())


def test_selftest_clip_through_the_pipeline_equals_last_weeks_files(loaded, tmp_path):
    # Last week's files live on as the frozen table, and the comparison of this run's file with it is
    # asserted once, in tests/slow/test_frozen_reference.py::
    # test_the_pipelines_tracker_files_equal_the_frozen_numbers[selftest]. Here: what the run folder holds.
    clip = synthetic.selftest_clip(tmp_path / "clip")
    assert clip["frames"] == FRAMES
    tracked = _track_through_the_pipeline(loaded, clip["video"], clip["export"], tmp_path)

    assert tracked.files == [tmp_path / "run" / MODEL / "selftest.csv"]
    # the full set of SPEC 8.1, each file with something in it, and nothing else
    assert {path.name for path in tracked.run.iterdir()} == RUN_FOLDER
    assert [path.name for path in (tracked.run / MODEL).iterdir()] == ["selftest.csv"]
    assert all(path.stat().st_size > 0 for path in tracked.run.iterdir() if path.is_file())
    print(f"VALIDATION pipeline, selftest clip (1 object, 20 frames, cpu, {tracked.threads} threads): "
          f"{tracked.seconds:.2f} s per frame; run folder: "
          f"{', '.join(sorted(path.name for path in tracked.run.iterdir()))}")


def test_three_ellipses_from_a_multi_start_file_equal_last_weeks_files(loaded, tmp_path):
    # Last week's files live on as the frozen table, and the comparison of this run's files with it is
    # asserted once, in tests/slow/test_frozen_reference.py::
    # test_the_pipelines_tracker_files_equal_the_frozen_numbers[three_ellipses]. Here: the start file is a
    # `#multi` one, and the run returns and leaves one Tracker-format file for each of its three marks.
    video = tmp_path / "three_tracker.mp4"
    truth = write_three_ellipse_clip(video)  # truth[object][frame] = (x, y), pixel centers at whole numbers
    marks = {name: (track[0][0] + 0.5, track[0][1] + 0.5) for name, track in zip("ABC", truth)}
    # Tracker's map with the origin in the middle of the frame, 0.05 mm per px, y up
    start = write_start_file(tmp_path / "ana" / "extra" / "start.csv", marks, frame=0,
                             to_mm=lambda px, py: tracker_map(px, py, 0.0, (960.0, 540.0)))
    assert start.read_text().startswith("#multi:")

    tracked = _track_through_the_pipeline(loaded, video, start, tmp_path, seconds=40 / FPS, step=2)

    assert [path.name for path in tracked.files] == ["A.csv", "B.csv", "C.csv"]
    assert sorted(path.name for path in (tracked.run / MODEL).iterdir()) == ["A.csv", "B.csv", "C.csv"]
    print(f"VALIDATION pipeline, three-ellipse clip with a #multi start file (3 objects, 20 frames, cpu, "
          f"{tracked.threads} threads): {tracked.seconds:.2f} s per frame")
