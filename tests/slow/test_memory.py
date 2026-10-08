"""Memory of a run with the real model (SPEC 6.5): ten coarse objects at 1080p stay under 3 GB, and
memory is flat in the number of frames.

The clip is `memory_child.ten_ellipse_scene()`: 1920 x 1080 px, 100 frames at 240 frames per
second, ten dark ellipses (semi-axes 8 and 3 px) more than 250 px apart. A child process
(tests/slow/memory_child.py, started with this Python) tracks all ten in one coarse run on the
whole frame, every frame, with EdgeTAM on `cpu`, through `tracking.run_job`, and samples its own
memory in the job's progress callback. It runs alone in its process, so pytest, the rendering of
the clip and the other tests are not in the numbers.

What is asserted, with GB = 10^9 bytes and MB = 10^6 bytes (the stricter reading):
- peak resident memory of the child (`ru_maxrss`) under 3 GB;
- resident memory at tracked frame 100 less than 50 MB above that at frame 40, each taken as the
  median of the ten readings that end with the one right after that frame (frames 31 to 40 and 91
  to 100). From frame 21 on the model's memory of earlier frames is full: the pruning keeps 20
  frames. A single reading cannot be used: with no leak, about every fifth reading is 57 to 214 MB
  below the readings next to it, for one frame (docs/VALIDATION.md 4.4; at most three of any ten
  readings in a row there). A median of ten is not moved by up to four such readings, and a
  steady leak shows in it in full (the middles of the two groups are 60 frames apart), so the
  test sees a leak of 50 / 60 = 0.83 MB per frame or more.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_memory.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1); frames are video frame
numbers. The test is slow (the child needs torch) and runs on macOS and Linux only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from memory_child import NAMES, ten_ellipse_scene
from pipeline_helpers import expect_minutes

from outline_tracker import synthetic

pytestmark = pytest.mark.slow

SLOW = Path(__file__).resolve().parent
GB, MB = 1e9, 1e6  # bytes
WINDOW = 10  # readings in each median


def level_at(resident, frame: int) -> float:
    """Resident memory at tracked frame `frame`, in bytes: the median of the WINDOW readings that
    end with the one right after that frame. `resident[k]` is the reading right after tracked
    frame k + 1, in bytes."""
    return float(np.median(np.asarray(resident, float)[frame - WINDOW:frame]))


def growth_40_to_100(resident) -> float:
    """Growth of resident memory from tracked frame 40 to tracked frame 100, in bytes: the
    difference of the two levels of `level_at`."""
    return level_at(resident, 100) - level_at(resident, 40)


def test_growth_is_not_moved_by_single_readings_and_sees_a_slow_leak():
    """`growth_40_to_100` on series whose answer is known by arithmetic. It needs no model and
    takes milliseconds; it carries this file's slow mark because it guards the test below."""
    frames = np.arange(1, 101)  # the reading at index k is the one after frame k + 1
    flat = np.full(100, 1400 * MB)

    # No leak, one reading 108 MB low (the dip one recorded run had after frame 80), on frame 40 or 100.
    for frame in (40, 100):
        dip = flat.copy()
        dip[frame - 1] -= 108 * MB
        assert growth_40_to_100(dip) == 0.0
    # No leak, four of the ten readings before each frame low: still nothing.
    dips = flat.copy()
    dips[[31, 34, 36, 39, 90, 93, 96, 99]] -= 108 * MB
    assert growth_40_to_100(dips) == 0.0

    # A leak of 1.7 MB per frame: the two medians sit at frames 35.5 and 95.5, 60 frames apart.
    leak = flat + 1.7 * MB * frames
    assert growth_40_to_100(leak) == pytest.approx(60 * 1.7 * MB)
    # The same leak with the 108 MB dip on frame 100, where the two single readings differ by
    # 60 x 1.7 - 108 = -6 MB: the later median moves to frame 94.5, one frame of leak less.
    hidden = leak.copy()
    hidden[99] -= 108 * MB
    assert hidden[99] - hidden[39] == pytest.approx(-6 * MB)
    assert growth_40_to_100(hidden) == pytest.approx(59 * 1.7 * MB)
    assert growth_40_to_100(hidden) > 50 * MB


@pytest.mark.skipif(sys.platform == "win32", reason="the child reads its memory with ps and resource: not on Windows")
@pytest.mark.weights
def test_ten_coarse_objects_at_1080p_stay_under_3_gb_and_flat(tmp_path):
    expect_minutes()
    scene = ten_ellipse_scene()
    assert scene.size == (1920, 1080) and scene.n_frames == 100 and len(scene.objects) == 10
    clip = synthetic.render(scene, tmp_path / "ten_tracker.mp4")
    centers = np.array([[obj.path.pose(frame)[:2] for obj in scene.objects] for frame in (0, 99)])  # [frame, object]
    apart = np.hypot(*(centers[:, :, None, :] - centers[:, None, :, :]).transpose(3, 0, 1, 2))
    assert apart[apart > 0].min() > 250.0  # well apart, at the start and at the end (the paths are straight)

    samples = tmp_path / "samples.json"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(SLOW.parent), str(SLOW)] + [path for path in [os.environ.get("PYTHONPATH")] if path])}
    child = subprocess.run([sys.executable, str(SLOW / "memory_child.py"), str(clip.path), str(tmp_path / "run"),
                            str(samples)], env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=570)
    assert child.returncode == 0, child.stderr[-4000:]
    data = json.loads(samples.read_text(encoding="utf-8"))
    assert data["status"] == "complete", "\n".join(data["log"])
    assert data["frames"] == 100 and len(data["resident_bytes"]) == 100

    resident = np.array(data["resident_bytes"], float)  # resident[k]: right after tracked frame k + 1
    growth = growth_40_to_100(resident)
    single = resident[99] - resident[39]  # printed, not asserted: one low reading moves it by its full depth
    peak_growth = data["peak_after"][99] - data["peak_after"][39]
    at = ", ".join(f"{frame}: {resident[frame - 1] / MB:.0f}" for frame in (1, 20, 40, 60, 80, 100))
    low, high = resident[39:].min() / MB, resident[39:].max() / MB
    print(f"VALIDATION memory, 10 coarse objects ({', '.join(NAMES[:2])}, ...), 1080p, 100 frames, cpu, in a process "
          f"of its own: peak resident memory {data['peak_bytes'] / GB:.2f} GB (limit 3); median of the {WINDOW} "
          f"readings ending after frame 40: {level_at(resident, 40) / MB:.0f} MB, after frame 100: "
          f"{level_at(resident, 100) / MB:.0f} MB, growth {growth / MB:+.0f} MB (limit 50); single readings in MB "
          f"after frame {at}; between the single readings after frames 40 and 100: {single / MB:+.0f} MB, between "
          f"the peaks: {peak_growth / MB:+.0f} MB; all readings from frame 40 on: {low:.0f} to {high:.0f} MB; lost "
          f"{data['lost']} of {10 * data['frames']} object-frames; {data['seconds_per_frame']:.2f} s per frame")
    print("VALIDATION memory, every reading in MB, frames 1 to 100: " + " ".join(f"{one / MB:.0f}" for one in resident))

    assert data["peak_bytes"] < 3 * GB
    assert growth < 50 * MB
