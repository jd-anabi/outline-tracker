"""Coarse mode with the real model on the dish-scale scene flags `LOWRES` (SPEC 13.4, 7.8, 9).

The clip is `synthetic.dish_scene()` as it stands: 1920 x 1080 px at 240 frames per second,
0.0324 mm per px, a dish of radius 486 px with three objects: A, a shrimp whose body is 14.5 px
long, and B and C, plain bodies of the same size. All three are tracked in coarse mode from one
positive click each, on frames 0, 2, ..., 58, with the scene's dish circle, so the model is shown
the dish square (SPEC 6.2), through `tracking.run_job` and `export.export_all`: one run on `cpu`,
shared by the three tests (one per object). On those frames B and C are more than 200 px apart
(they meet on frame 240).

Why `LOWRES` is the right answer, from the scene alone (SPEC 7.8): `shape_ok` needs at least 20
camera pixels AND 20 of the model's grid cells along the major axis. The bodies are 14.5 px long,
and a grid cell of the dish square (2 x 1.03 x 486 px, 1002 whole px wide) is 3.9 px, so a body
spans 3.7 cells. Asserted for each object: it is found at its click, and every frame on which it
has a mask has `shape_ok` = 0 and the flag `LOWRES`.

The test of B fails with the real model and is marked xfail (strict), with what was measured: on
one frame B's mask holds two stray pixels far from the body. Nothing was tuned;
docs/VALIDATION.md, section 4.3, has the numbers.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_coarse_lowres.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame;
frames are video frame numbers. Every test here is slow: it needs torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from pipeline_helpers import calibrated_session, track_and_export
from tracking_helpers import center, dish_circle, track

from outline_tracker import synthetic

pytestmark = pytest.mark.slow

FRAMES = list(range(0, 60, 2))  # the tracked video frames
SHAPE_OK_MIN = 20  # px and grid cells along the major axis that `shape_ok` asks for (SPEC 7.8)

STRAY_PIXELS = (
    "Measured on 2026-10-07 with EdgeTAM on cpu: B is LOWRES on 14 of the 15 frames on which it has a mask. The "
    "model lost B on frames 6 to 34. On frame 36 it came back with a mask of 99 px in two pieces: the body, and 2 "
    "px far away from it. The second moments of the whole mask then give a major axis of 134.8 px (34.4 cells), so "
    "by the rule of SPEC 7.8 shape_ok is 1 and the frame is not LOWRES (it is flagged ORIENT; MULTI needs a second "
    "piece of 10%). From frame 38 on the mask is the body again (15 px, LOWRES). A and C are LOWRES on all 30 "
    "frames. Nothing was tuned. For J (docs/VALIDATION.md 4.3): should px_along_major be taken from the largest "
    "piece of the mask?"
)


@pytest.fixture(scope="module")
def dish_run(tmp_path_factory):
    """The three objects of `dish_scene()` tracked coarse on FRAMES with the real EdgeTAM on cpu,
    and exported: `track_and_export`'s result (positions and shapes as tables)."""
    from outline_tracker.segmenter import hf

    scene = synthetic.dish_scene()
    assert scene.size == (1920, 1080) and scene.mm_per_px == 0.0324
    for obj in scene.objects:  # every body is shorter than 20 px: about 14.5
        assert 2 * obj.shape.a_px == pytest.approx(14.5, abs=0.1) and 2 * obj.shape.a_px < SHAPE_OK_MIN
    folder = tmp_path_factory.mktemp("dish")
    dish = synthetic.render(scene, folder / "dish_tracker.mp4")
    apart = min(np.hypot(*np.subtract(center(dish, "B", frame), center(dish, "C", frame))) for frame in FRAMES)
    assert apart > 200.0

    session = calibrated_session(dish, folder / "run", [track(dish, name) for name in "ABC"], step=2, end=FRAMES[-1],
                                 circle=dish_circle(scene))
    done = track_and_export(dish, session, folder / "run", hf.HFSegmenter("edgetam", "cpu"))
    assert done.status == "complete", "\n".join(done.log)

    notes = []
    for name in "ABC":
        shapes = done.shapes[done.shapes.track_id == name]
        seen = (done.positions[done.positions.track_id == name].visible == 1).to_numpy()
        visible = shapes[seen]
        flags = sorted({code for cell in shapes["flags"] for code in cell.split(";") if code})
        notes.append(f"{name}: found on {int(seen.sum())} of {len(seen)} frames, px_along_major "
                     f"{visible.px_along_major.min():.1f} to {visible.px_along_major.max():.1f}, cells_along_major "
                     f"{visible.cells_along_major.min():.1f} to {visible.cells_along_major.max():.1f}, LOWRES on "
                     f"{int(shapes['flags'].str.contains('LOWRES').sum())} frames (all flags: {', '.join(flags)})")
    print(f"VALIDATION coarse mode at dish scale (3 objects, {len(FRAMES)} frames, dish square, cpu): "
          f"{'; '.join(notes)}; {done.seconds_per_frame:.2f} s per frame")
    return done


@pytest.mark.parametrize("name", ["A", pytest.param("B", marks=pytest.mark.xfail(strict=True, reason=STRAY_PIXELS)),
                                  "C"])
def test_every_frame_with_a_mask_is_flagged_lowres(dish_run, name):
    shapes = dish_run.shapes[dish_run.shapes.track_id == name]
    positions = dish_run.positions[dish_run.positions.track_id == name]
    assert shapes.frame.tolist() == positions.frame.tolist() == FRAMES and set(positions["mode"]) == {"coarse"}
    seen = (positions.visible == 1).to_numpy()
    assert seen[0]  # the model found the object at its click
    # the model was shown the dish square, 1002 px wide (floor and ceil of 960.3 -+ 500.58): cells of 3.91 px
    cell = (shapes.px_along_major[seen] / shapes.cells_along_major[seen]).to_numpy()
    assert np.allclose(cell, 1002 / 256, atol=0.02)
    assert (shapes.shape_ok[seen] == 0).all()
    assert shapes["flags"][seen].str.contains("LOWRES").all()
