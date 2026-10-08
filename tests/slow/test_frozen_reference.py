"""The real EdgeTAM on the processor gives the frozen numbers (docs/ROADMAP.md, W1 step 4).

tests/data/edgetam_cpu_positions.csv holds where the model found each object of two short synthetic
clips on `cpu`, and tests/data/edgetam_weights.txt the hash of its weights. Both were written by a
run of tests/slow/test_regression_reference.py while last week's code was still in the repository:
in that run it agreed within 0.01 px, and every position was under 3 px from its true center; each
file's header says how. Last week's code left in W1 step 5, and with it what wrote the two files:
the command in their headers writes nothing now. Here the package alone runs and is compared with
them. This is the one place where a run is compared with the table; the two regression files beside
this one (test_regression_reference.py, test_regression_pipeline.py) run the same clips for what the
table does not hold.

The two clips, both 1920 x 1080 px and 40 frames, tracked on frames 0, 2, ..., 38 with one positive
click per object on frame 0:
1. `selftest`: `synthetic.selftest_clip`, one dark ellipse, with its Tracker export of one track;
2. `three_ellipses`: the three-ellipse clip of tests/slow/pipeline_helpers.py, with a
   `#multi` start file that marks A, B and C on frame 0.

Two levels:
- the segmenter: `HFSegmenter`, with the frames and clicks of `tracker_io.make_plan` and the images
  of `video.iter_rgb_frames`. A position is `mask_center` of the mask plus the crop's offset;
- the whole pipeline: `from_tracker`. A position is `pixelx`, `pixely` of a Tracker-format file,
  which has 3 decimals: up to 0.0005 px per axis from the table's 4 decimals.

The machine rule (`frozen_helpers.same_machine`): the limit of 0.01 px and the weights' hash are
asserted on the machine that froze the numbers. On another machine the position tests assert that
no row is lost and that every position is under 3 px from its true center, and print the distance
from the frozen numbers as measured; the weights test is skipped, and its reason names both hashes.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_frozen_reference.py -q -rPs` (`s` adds the reason of a skipped
test). Every test here is slow.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), u to the right, v
downward, in the full frame; frames are video frame numbers; fps_true is 240 frames per second.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from from_tracker_helpers import write_start_file
from frozen_helpers import compare_with_the_frozen_positions, machine_here, machine_of, read_weights, same_machine
from helpers import tracker_map
from pipeline_helpers import position, write_three_ellipse_clip

from outline_tracker import synthetic, tracker_io, video
from outline_tracker.from_tracker import from_tracker
from outline_tracker.segmenter.base import ObjectPrompt

pytestmark = pytest.mark.slow

MODEL = "edgetam"
FPS = 240.0  # fps_true of both clips, frames per second
FRAMES = list(range(0, 40, 2))  # the tracked frames of both clips


# Overrides the shared `loaded` of tests/slow/conftest.py: this one loads the model that `MODEL` names.
@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model(MODEL)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """The two clips, written once, by the table's names. Each has `video`, `export` (its Tracker
    export or start file), `options` (what `make_plan` and `from_tracker` get beside fps_true),
    `names` (its objects) and `true`: true[i][k] = (u_px, v_px), the true center of names[k] on
    FRAMES[i]."""
    folder = tmp_path_factory.mktemp("frozen_clips")
    one = synthetic.selftest_clip(folder / "selftest")
    assert one["frames"] == FRAMES
    selftest = SimpleNamespace(video=one["video"], export=one["export"], options={}, names=["selftest"],
                               true=np.stack([one["pixelx"], one["pixely"]], axis=1)[:, None, :])

    (folder / "three").mkdir()
    three_video = folder / "three" / "three_tracker.mp4"
    truth = write_three_ellipse_clip(three_video)  # truth[object][frame] = (x, y), pixel centers at whole numbers
    marks = {name: (track[0][0] + 0.5, track[0][1] + 0.5) for name, track in zip("ABC", truth)}
    # Tracker's map with the origin in the middle of the frame, 0.05 mm per px, y up
    start = write_start_file(folder / "three" / "start.csv", marks, frame=0,
                             to_mm=lambda px, py: tracker_map(px, py, 0.0, (960.0, 540.0)))
    true = np.array([[(track[frame][0] + 0.5, track[frame][1] + 0.5) for track in truth] for frame in FRAMES])
    three = SimpleNamespace(video=three_video, export=start, options={"seconds": 40 / FPS, "step": 2},
                            names=["A", "B", "C"], true=true)
    return {"selftest": selftest, "three_ellipses": three}


def _segmenter(loaded):
    """A new `HFSegmenter` on cpu around the module's loaded model."""
    from outline_tracker.segmenter import hf

    model, processor = loaded
    return hf.HFSegmenter(MODEL, "cpu", model=model, processor=processor)


def _segmenter_level(loaded, clips, name) -> None:
    """Track the clip `name` with the segmenter alone, on the frames and from the clicks that
    `make_plan` reads from the clip's export, and judge the positions against the frozen ones: the
    table's rows of this clip, by the machine rule with the hash of the weights that the segmenter
    reports (`frozen_helpers.compare_with_the_frozen_positions`)."""
    clip = clips[name]
    plan = tracker_io.make_plan(clip.export, fps=FPS, **clip.options)
    assert plan.frames == FRAMES and plan.names == clip.names
    prompts = [ObjectPrompt(track_id, [point], [1]) for track_id, point in zip(plan.names, plan.points_px)]
    segmenter = _segmenter(loaded)
    found = []  # found[i][k] = (u_px, v_px) of names[k] on FRAMES[i]
    for i, (frame, rgb) in enumerate(video.iter_rgb_frames(clip.video, plan.frames)):
        results = segmenter.start(rgb, prompts) if i == 0 else segmenter.step(rgb)
        assert frame == FRAMES[i] and [result.obj_id for result in results] == clip.names
        found.append([position(result) for result in results])
    segmenter.close()
    assert len(found) == len(FRAMES) and segmenter.device == "cpu"
    compare_with_the_frozen_positions(f"{name}, segmenter on cpu", name, FRAMES, clip.names, np.array(found),
                                      clip.true, segmenter.weights_sha256)


def test_selftest_clip_positions_equal_the_frozen_numbers(loaded, clips):
    _segmenter_level(loaded, clips, "selftest")


def test_three_ellipses_equal_the_frozen_numbers(loaded, clips):
    _segmenter_level(loaded, clips, "three_ellipses")


@pytest.mark.parametrize("name", ["selftest", "three_ellipses"])
def test_the_pipelines_tracker_files_equal_the_frozen_numbers(loaded, clips, name, tmp_path):
    clip = clips[name]
    segmenter = _segmenter(loaded)
    run = from_tracker(clip.video, clip.export, fps=FPS, out=tmp_path / "run", model=MODEL, device="cpu",
                       overlay=False, segmenter=segmenter, log=lambda line: None, **clip.options)
    assert segmenter.device == "cpu"
    assert run.files == [tmp_path / "run" / MODEL / f"{track_id}.csv" for track_id in clip.names]
    tables = [pd.read_csv(path, skiprows=1) for path in run.files]
    assert [table["frame"].tolist() for table in tables] == [FRAMES] * len(clip.names)
    # found[i][k] = (pixelx, pixely) of names[k] on FRAMES[i]; the empty cells of a lost row are read as NaN
    found = np.stack([table[["pixelx", "pixely"]].to_numpy(float) for table in tables], axis=1)
    compare_with_the_frozen_positions(f"{name}, pipeline (from_tracker) on cpu", name, FRAMES, clip.names, found,
                                      clip.true, segmenter.weights_sha256)


def test_the_weights_are_the_frozen_file(loaded):
    from outline_tracker.segmenter import hf

    segmenter = _segmenter(loaded)
    header, frozen_weights = read_weights()
    saved = hf.weights_file(MODEL)  # the file the model was loaded from
    assert saved is not None
    print(f"VALIDATION frozen numbers, weights: {saved.stat().st_size} bytes, sha256 {segmenter.weights_sha256}; "
          f"frozen: {frozen_weights['bytes']} bytes, sha256 {frozen_weights['sha256']}")
    # The rule is asked without the hash: on the machine that froze it, other weights are a failure, not
    # another machine.
    if not same_machine(header, machine_here()):
        made = machine_of(header)
        pytest.skip(f"The weights' hash is asserted on the machine that froze it ({made['chip']}, {made['platform']} "
                    f"{made['architecture']}, torch {made['torch']}). This is another one, and the converted file is "
                    f"written on each machine. SHA-256 here: {segmenter.weights_sha256}; frozen: "
                    f"{frozen_weights['sha256']}.")
    assert segmenter.weights_sha256 == frozen_weights["sha256"]
    assert saved.stat().st_size == int(frozen_weights["bytes"])
