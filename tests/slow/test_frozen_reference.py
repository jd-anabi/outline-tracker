"""The real EdgeTAM on the processor gives the frozen numbers (docs/ROADMAP.md, W1 step 4).

tests/data/edgetam_cpu_positions.csv holds where the model found each object of two short synthetic
clips on `cpu`, and tests/data/edgetam_weights.txt the hash of its weights. Both were written by the
run of tests/slow/test_regression_reference.py in which last week's code agreed within 0.01 px and
every position was under 3 px from its true center; each file's header says how. Here the package
alone runs again and is compared with them: nothing of last week's code is imported.

The two clips, both 1920 x 1080 px and 40 frames, tracked on frames 0, 2, ..., 38 with one positive
click per object on frame 0:
1. `selftest`: `synthetic.selftest_clip`, one dark ellipse, with its Tracker export of one track;
2. `three_ellipses`: the three-ellipse clip of tests/slow/test_regression_reference.py, with a
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
from frozen_helpers import (WEIGHTS, compare_with_frozen, machine_here, machine_of, read_frozen, read_positions,
                            same_machine)
from test_regression_reference import _write_three_ellipse_clip
from test_tracker_io import tracker_map

from outline_tracker import synthetic, tracker_io, video
from outline_tracker.from_tracker import from_tracker
from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import ObjectPrompt

pytestmark = pytest.mark.slow

MODEL = "edgetam"
FPS = 240.0  # fps_true of both clips, frames per second
FRAMES = list(range(0, 40, 2))  # the tracked frames of both clips


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for this module."""
    from outline_tracker.segmenter import hf

    return hf.load_model(MODEL)


@pytest.fixture(scope="module")
def frozen():
    """The frozen table: `header` (its `# key: value` lines) and `positions`, where
    positions[clip, frame, track_id] = (u_px, v_px)."""
    header, rows = read_positions()
    return SimpleNamespace(header=header, positions={(clip, frame, track_id): (u_px, v_px)
                                                     for clip, frame, track_id, u_px, v_px in rows})


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
    truth = _write_three_ellipse_clip(three_video)  # truth[object][frame] = (x, y), pixel centers at whole numbers
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


def _strict(frozen, segmenter) -> bool:
    """Whether the limit of 0.01 px is asserted in this run: on the machine that froze the numbers,
    with the weights that gave them (the machine rule, `frozen_helpers.same_machine`)."""
    return same_machine(frozen.header, machine_here(segmenter.weights_sha256))


def _frozen_positions(frozen, name, names) -> np.ndarray:
    """The frozen positions of the clip `name`: [i][k] = (u_px, v_px) of names[k] on FRAMES[i]."""
    return np.array([[frozen.positions[name, frame, track_id] for track_id in names] for frame in FRAMES])


def _position(result) -> tuple[float, float]:
    """(u_px, v_px) of a result in the full frame; NaN, NaN if the object was not found."""
    u, v, _ = mask_center(result.mask)
    return u + result.offset[0], v + result.offset[1]


def _segmenter_level(loaded, frozen, clips, name) -> None:
    """Track the clip `name` with the segmenter alone, on the frames and from the clicks that
    `make_plan` reads from the clip's export, and judge the positions against the frozen ones."""
    clip = clips[name]
    plan = tracker_io.make_plan(clip.export, fps=FPS, **clip.options)
    assert plan.frames == FRAMES and plan.names == clip.names
    prompts = [ObjectPrompt(track_id, [point], [1]) for track_id, point in zip(plan.names, plan.points_px)]
    segmenter = _segmenter(loaded)
    found = []  # found[i][k] = (u_px, v_px) of names[k] on FRAMES[i]
    for i, (frame, rgb) in enumerate(video.iter_rgb_frames(clip.video, plan.frames)):
        results = segmenter.start(rgb, prompts) if i == 0 else segmenter.step(rgb)
        assert frame == FRAMES[i] and [result.obj_id for result in results] == clip.names
        found.append([_position(result) for result in results])
    segmenter.close()
    assert len(found) == len(FRAMES) and segmenter.device == "cpu"
    compare_with_frozen(f"{name}, segmenter on cpu", np.array(found), _frozen_positions(frozen, name, clip.names),
                        clip.true, _strict(frozen, segmenter))


def test_selftest_clip_positions_equal_the_frozen_numbers(loaded, frozen, clips):
    _segmenter_level(loaded, frozen, clips, "selftest")


def test_three_ellipses_equal_the_frozen_numbers(loaded, frozen, clips):
    _segmenter_level(loaded, frozen, clips, "three_ellipses")


@pytest.mark.parametrize("name", ["selftest", "three_ellipses"])
def test_the_pipelines_tracker_files_equal_the_frozen_numbers(loaded, frozen, clips, name, tmp_path):
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
    compare_with_frozen(f"{name}, pipeline (from_tracker) on cpu", found, _frozen_positions(frozen, name, clip.names),
                        clip.true, _strict(frozen, segmenter))


def test_the_weights_are_the_frozen_file(loaded):
    from outline_tracker.segmenter import hf

    segmenter = _segmenter(loaded)
    header, rows = read_frozen(WEIGHTS)
    frozen_weights = dict(row.split(": ", 1) for row in rows)
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
