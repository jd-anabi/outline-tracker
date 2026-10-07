"""The stand-in segmenters (SPEC 12, 13.2): `ThresholdFake` on the disk clip, `ExactFake`, and the
rules both share. tests/test_fakes_threshold.py checks `ThresholdFake` against the template's
`DiskFinder` and on hand-made images.

Expected values come from geometry written out here (a disk's inequality, bounding boxes worked out
by hand), from the scenes' stated paths, and from the ground truth of outline_tracker/synthetic.py,
which tests/test_synthetic.py checks against exact inequalities.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. A view is the part
of the full frame a segmenter is shown: `offset` = (column, row) of its top-left pixel in the full
frame, `size` = (width, height). A result's own offset is in px of the view.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import SMALL, assert_empty_result, click, mask_in_image

from outline_tracker import synthetic, video
from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import ObjectPrompt, PromptError
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from outline_tracker.synthetic import GroundTruth, Scene, SceneObject
from outline_tracker.synthetic_shapes import Disk, Straight

REPO = Path(__file__).resolve().parents[1]
FRAME = (120, 90)  # (width, height) of the hand-made scene, px


def blank(size):
    """An image of `size` = (width, height) px. `ExactFake` looks only at its size."""
    return np.zeros((size[1], size[0], 3), np.uint8)


def disks():
    """Ground truth of a 120 x 90 px scene, 4 frames, with three disks of radius 12 px:
    A at (50.3, 40.7) + frame (0.5, 0.13); B at rest far outside the frame; C at rest at
    (4.2, 80.6), cut by the left and the bottom border of the frame."""
    objects = (SceneObject("A", Disk(12.0), Straight((50.3, 40.7), (0.5, 0.13))),
               SceneObject("B", Disk(12.0), Straight((-60.0, 30.0), (0.0, 0.0))),
               SceneObject("C", Disk(12.0), Straight((4.2, 80.6), (0.0, 0.0))))
    return GroundTruth(Scene(size=FRAME, fps=240.0, n_frames=4, objects=objects, mm_per_px=0.01))


def seen(truth, frame, offset, size, names, start=True):
    """What an `ExactFake` returns for the named objects when it is shown that view of that frame."""
    fake = ExactFake(truth)
    fake.set_view(frame, offset, size)
    prompts = [click(name) for name in names]
    return fake.start(blank(size), prompts) if start else fake.preview(blank(size), prompts)


# ---------------------------------------------------------------------------------------------
# ThresholdFake on the disk clip (the template's tolerance)


def test_threshold_fake_follows_the_disks_within_a_quarter_pixel(disk_clip):
    scene = disk_clip.scene
    assert all(isinstance(obj.shape, Disk) for obj in scene.objects)  # SPEC 13.2: tested on disks only
    width, height = scene.size
    fake, worst, frames = ThresholdFake(), 0.0, 0
    prompts = [click(obj.track_id, *obj.path.pose(0)[:2]) for obj in scene.objects]  # the true centers in frame 0
    for frame, rgb in video.iter_rgb_frames(disk_clip.path, range(scene.n_frames)):
        results = fake.start(rgb, prompts) if frame == 0 else fake.step(rgb)
        assert [result.obj_id for result in results] == ["A", "B", "C"]
        for obj, result in zip(scene.objects, results):
            u, v, area = mask_center(result.mask)
            true_u, true_v, _ = obj.path.pose(frame)
            assert area > 300  # pi 12^2 = 452 px
            worst = max(worst, float(np.hypot(u + result.offset[0] - true_u, v + result.offset[1] - true_v)))
            # cropped to the bounding box +/- 8 px (less only at the image's border), with logits
            rows, cols = np.nonzero(result.mask)
            (col0, row0), (n_rows, n_cols) = result.offset, result.mask.shape
            assert rows.min() == 8 or row0 == 0
            assert cols.min() == 8 or col0 == 0
            assert n_rows - 1 - rows.max() == 8 or row0 + n_rows == height
            assert n_cols - 1 - cols.max() == 8 or col0 + n_cols == width
            assert result.logits.dtype == np.float32 and result.logits.shape == result.mask.shape
            assert np.array_equal(result.logits > 0, result.mask)
            assert result.score is None
        frames += 1
    fake.close()
    assert frames == scene.n_frames == 120
    assert worst < 0.25


# ---------------------------------------------------------------------------------------------
# ExactFake: the ground-truth mask and logits, in the full frame and in a crop


@pytest.fixture(scope="module")
def closeup():
    """Ground truth of `closeup_scene` at 320 x 240 px: A, the shrimp with antennae, and B, the plain body."""
    return GroundTruth(synthetic.closeup_scene(size=SMALL, n_frames=120))


# A circles around (198.7, 119.8) at 48 px and reaches 45.5 px from its center; B stays left of
# column 89. So the second view cuts B, and the third holds all of A and nothing of B.
VIEWS = [((0, 0), SMALL), ((37, 21), (250, 200)), ((100, 20), (200, 200))]


@pytest.mark.parametrize("offset, size", VIEWS, ids=["full frame", "shifted crop", "crop around A"])
@pytest.mark.parametrize("frame", [0, 57, 119])
def test_exact_fake_returns_the_ground_truth_mask_and_logits_of_the_view(closeup, frame, offset, size):
    (col0, row0), (width, height) = offset, size
    results = seen(closeup, frame, offset, size, ["A", "B"])
    assert [result.obj_id for result in results] == ["A", "B"]
    visible = 0
    for result in results:
        true_mask = closeup.mask(result.obj_id, frame)[row0:row0 + height, col0:col0 + width]
        true_logits = closeup.logits(result.obj_id, frame)[row0:row0 + height, col0:col0 + width]
        assert true_mask.shape == (height, width)
        assert np.array_equal(mask_in_image(result, (height, width)), true_mask)  # exactly the ground truth
        assert result.score is None
        if not true_mask.any():
            assert_empty_result(result)
            continue
        visible += 1
        (c, r), (rows, cols) = result.offset, result.mask.shape
        assert result.mask.dtype == bool and result.logits.dtype == np.float32
        assert result.logits.shape == result.mask.shape
        assert np.array_equal(result.logits, true_logits[r:r + rows, c:c + cols])  # sliced, not drawn again
        assert np.array_equal(result.logits > 0, result.mask)
        assert result.mask.flags.owndata and result.logits.flags.owndata  # no full-frame array is kept alive
    assert visible >= 1


@pytest.mark.parametrize("offset, size", VIEWS[:2], ids=["full frame", "shifted crop"])
def test_exact_fake_positions_are_the_ground_truth_centroids(closeup, offset, size):
    table = closeup.table.set_index(["track_id", "frame"])
    for frame in (0, 30, 119):
        for result in seen(closeup, frame, offset, size, ["A"]):  # A lies inside both views
            u, v, area = mask_center(result.mask)
            row = table.loc[("A", frame)]
            assert area == row["area_px"]
            assert u + result.offset[0] + offset[0] == pytest.approx(row["u_px"], abs=1e-9)
            assert v + result.offset[1] + offset[1] == pytest.approx(row["v_px"], abs=1e-9)


# Disk A in frame 0: center (50.3, 40.7), radius 12. A pixel center (c + 0.5, r + 0.5) is inside if
# its distance from the center is below 12: on row 40 (0.2 px above the center) that is c = 38 to 61,
# on column 50 (0.2 px left of it) r = 29 to 52. So the bounding box is columns 38..61, rows 29..52,
# and with 8 px on every side the crop is columns 30..69, rows 21..60 (40 x 40 px).
@pytest.mark.parametrize("offset, size, crop_offset, crop_shape", [
    ((0, 0), FRAME, (30, 21), (40, 40)),
    ((25, 17), (80, 60), (5, 4), (40, 40)),     # the same crop, counted from the view's corner
    ((44, 33), (40, 40), (0, 0), (28, 26)),     # the view cuts the disk: columns 0..17, rows 0..19 of it
    ((-10, -6), (140, 110), (40, 27), (40, 40)),  # a view larger than the frame
], ids=["full frame", "shifted crop", "cut by the view", "larger than the frame"])
def test_exact_fake_crops_to_the_bounding_box_plus_8_px_with_the_offset_in_the_view(offset, size, crop_offset,
                                                                                  crop_shape):
    truth = disks()
    (result,) = seen(truth, 0, offset, size, ["A"])
    assert result.offset == crop_offset and all(isinstance(value, int) for value in result.offset)
    assert result.mask.shape == crop_shape and result.logits.shape == crop_shape
    u, v = np.meshgrid(np.arange(size[0]) + 0.5 + offset[0], np.arange(size[1]) + 0.5 + offset[1])
    in_frame = (u > 0) & (u < FRAME[0]) & (v > 0) & (v < FRAME[1])
    assert np.array_equal(mask_in_image(result, size[::-1]), (np.hypot(u - 50.3, v - 40.7) < 12.0) & in_frame)


def test_exact_fake_shows_each_frame_the_runner_names_and_keeps_the_order_of_the_prompts():
    truth = disks()
    fake = ExactFake(truth)
    fake.set_view(0, (0, 0), FRAME)
    first = fake.start(blank(FRAME), [click("C"), click("A")])
    fake.set_view(3, (0, 20), (100, 70))  # A has moved by (1.5, 0.39) px; another crop
    later = fake.step(blank((100, 70)))
    assert [result.obj_id for result in first] == [result.obj_id for result in later] == ["C", "A"]
    u, v = np.meshgrid(np.arange(100) + 0.5, np.arange(70) + 20.5)
    assert np.array_equal(mask_in_image(later[1], (70, 100)), np.hypot(u - 51.8, v - 41.09) < 12.0)
    assert np.array_equal(mask_in_image(first[1], FRAME[::-1]), truth.mask("A", 0))
    assert not np.array_equal(truth.mask("A", 0), truth.mask("A", 3))  # the two frames do differ
    assert later[0].mask.any()
    assert np.array_equal(mask_in_image(later[0], (70, 100)), truth.mask("C", 3)[20:90, 0:100])


@pytest.mark.parametrize("start", [True, False], ids=["start", "preview"])
def test_an_invisible_object_gives_an_empty_result(start):
    truth = disks()
    outside_view, outside_frame, both = (seen(truth, 0, offset, size, names, start) for offset, size, names in [
        ((80, 0), (40, 40), ["A"]),         # A's columns are 38..61: not in this view
        ((0, 0), FRAME, ["B"]),             # B is outside the frame
        ((0, 0), FRAME, ["B", "A", "B"]),   # the others are not disturbed
    ])
    assert_empty_result(outside_view[0])
    assert_empty_result(outside_frame[0])
    assert_empty_result(both[0])
    assert_empty_result(both[2])
    assert both[1].mask.shape == (40, 40) and [result.obj_id for result in both] == ["B", "A", "B"]
    assert mask_center(outside_view[0].mask)[2] == 0


def test_a_view_that_hangs_over_the_frame_shows_nothing_outside_it():
    # Fine mode pads its crop by repeating the frame's edge pixels (SPEC 6.3). C is cut by the left
    # and bottom borders; the view covers columns -20..43 and rows 60..123 of a 120 x 90 px frame.
    truth = disks()
    (result,) = seen(truth, 0, (-20, 60), (64, 64), ["C"])
    inside = np.zeros((64, 64), bool)
    inside[:30, 20:] = True  # rows 60..89 and columns 0..43 of the frame
    expected = np.zeros((64, 64), bool)
    expected[:30, 20:] = truth.mask("C", 0)[60:90, 0:44]
    assert expected.any() and np.array_equal(mask_in_image(result, (64, 64)), expected)
    # Logits: the truth inside the frame; outside it minus the size of the nearest frame pixel's
    # value, so nothing is "inside" there and the outline closes on the frame's border.
    edge = np.pad(truth.logits("C", 0)[60:90, 0:44], ((0, 34), (20, 0)), mode="edge")
    logits = np.where(inside, edge, -np.abs(edge))
    # C's center is (4.2, 80.6): its pixels are columns 0..15 and rows 69..89 of the frame, which are
    # columns 20..35 and rows 9..29 of the view. The crop's 8 px reach beyond the frame's borders.
    (c, r), (rows, cols) = result.offset, result.mask.shape
    assert (c, r) == (12, 1) and (rows, cols) == (37, 32)
    assert np.array_equal(result.logits, logits[r:r + rows, c:c + cols])
    assert np.array_equal(result.logits > 0, result.mask)
    row = truth.table.set_index(["track_id", "frame"]).loc[("C", 0)]
    u, v, area = mask_center(result.mask)
    assert area == row["area_px"]
    assert (u + c - 20, v + r + 60) == pytest.approx((row["u_px"], row["v_px"]), abs=1e-9)


def test_exact_fake_preview_leaves_the_run_as_it_is():
    truth = disks()
    fake = ExactFake(truth)
    fake.set_view(0, (0, 0), FRAME)
    fake.start(blank(FRAME), [click("A"), click("C")])
    fake.set_view(2, (0, 0), FRAME)
    (shown,) = fake.preview(blank(FRAME), [click("C")])
    assert shown.obj_id == "C" and np.array_equal(mask_in_image(shown, FRAME[::-1]), truth.mask("C", 2))
    fake.set_view(1, (0, 0), FRAME)
    results = fake.step(blank(FRAME))
    assert [result.obj_id for result in results] == ["A", "C"]
    assert np.array_equal(mask_in_image(results[0], FRAME[::-1]), truth.mask("A", 1))


# ---------------------------------------------------------------------------------------------
# Misuse is reported, not guessed around


def test_exact_fake_must_be_told_its_view_first():
    fake = ExactFake(disks())
    for call in (lambda: fake.start(blank(FRAME), [click("A")]), lambda: fake.preview(blank(FRAME), [click("A")])):
        with pytest.raises(RuntimeError, match="set_view"):
            call()


def test_exact_fake_refuses_an_image_that_is_not_the_view_it_was_told():
    fake = ExactFake(disks())
    fake.set_view(0, (10, 10), (80, 60))
    with pytest.raises(ValueError, match=r"80 x 60.*120 x 90"):
        fake.start(blank(FRAME), [click("A")])
    fake.start(blank((80, 60)), [click("A")])
    with pytest.raises(ValueError, match=r"80 x 60.*60 x 80"):
        fake.step(blank((60, 80)))


@pytest.mark.parametrize("frame, offset, size, complaint", [
    (0, (10.5, 0), (80, 60), "offset in px must be a whole number, not 10.5"),
    (0, (0, 0), (80.0, 60), "size in px must be a whole number, not 80.0"),
    (1.0, (0, 0), (80, 60), "frame number must be a whole number, not 1.0"),
    (0, (0, 0), (0, 60), "must be positive, not 0 x 60 px"),
    (0, (0, 0), (80, -1), "must be positive, not 80 x -1 px"),
])
def test_exact_fake_takes_only_whole_pixel_views_of_positive_size(frame, offset, size, complaint):
    fake = ExactFake(disks())
    with pytest.raises(ValueError, match=complaint):
        fake.set_view(frame, offset, size)
    with pytest.raises(RuntimeError, match="set_view"):  # a refused view is not a view
        fake.start(blank(FRAME), [click("A")])
    fake.set_view(np.int64(0), (np.int64(0), 0), FRAME)  # whole numbers from numpy are fine
    assert len(fake.start(blank(FRAME), [click("A")])) == 1


def test_exact_fake_knows_only_the_tracks_and_frames_of_its_scene():
    fake = ExactFake(disks())
    fake.set_view(0, (0, 0), FRAME)
    with pytest.raises(KeyError, match="No track 'Z'"):
        fake.start(blank(FRAME), [click("A"), click("Z")])
    fake.set_view(4, (0, 0), FRAME)  # the clip has frames 0 to 3
    with pytest.raises(IndexError, match="frame 4"):
        fake.start(blank(FRAME), [click("A")])


@pytest.mark.parametrize("make", [ThresholdFake, lambda: ExactFake(disks())], ids=["ThresholdFake", "ExactFake"])
def test_step_needs_a_run(make):
    fake = make()
    if isinstance(fake, ExactFake):
        fake.set_view(0, (0, 0), FRAME)
    with pytest.raises(RuntimeError, match="start"):
        fake.step(blank(FRAME))
    assert len(fake.start(blank(FRAME), [click("A")])) == 1
    assert len(fake.step(blank(FRAME))) == 1
    fake.close()
    with pytest.raises(RuntimeError, match="start"):
        fake.step(blank(FRAME))


@pytest.mark.parametrize("make", [ThresholdFake, lambda: ExactFake(disks())], ids=["ThresholdFake", "ExactFake"])
@pytest.mark.parametrize("method", ["start", "preview"])
def test_an_object_needs_a_positive_point_inside_the_image(make, method):
    # The rule of the protocol (segmenter/base.py), as with the real model.
    fake = make()
    if isinstance(fake, ExactFake):
        fake.set_view(0, (0, 0), FRAME)
    call = getattr(fake, method)
    for points, labels in [([(130.0, 40.0)], [1]), ([(50.0, 40.0)], [0]), ([(50.0, 40.0), (-1.0, 5.0)], [0, 1])]:
        with pytest.raises(PromptError, match="'A' has no positive point inside the image"):
            call(blank(FRAME), [ObjectPrompt("A", points, labels)])
    assert len(call(blank(FRAME), [ObjectPrompt("A", [(130.0, 40.0), (50.0, 40.0)], [1, 1])])) == 1
    assert call(blank(FRAME), []) == []


def test_importing_the_stand_ins_loads_neither_torch_nor_qt():
    # In a subprocess: pytest-qt has already imported PySide6 into this one.
    code = ("import json, sys\nimport outline_tracker.segmenter.fake\n"
            "print(json.dumps(sorted(m for m in ('torch', 'transformers', 'PySide6') if m in sys.modules)))")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []
