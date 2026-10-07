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
- resident memory right after tracked frame 100 less than 50 MB above that after frame 40. By
  frame 40 the model's memory of earlier frames is full: the pruning keeps 20 frames.

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


@pytest.mark.skipif(sys.platform == "win32", reason="the child reads its memory with ps and resource: not on Windows")
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
    growth = resident[99] - resident[39]
    peak_growth = data["peak_after"][99] - data["peak_after"][39]
    at = ", ".join(f"{frame}: {resident[frame - 1] / MB:.0f}" for frame in (1, 20, 40, 60, 80, 100))
    low, high = resident[39:].min() / MB, resident[39:].max() / MB
    print(f"VALIDATION memory, 10 coarse objects ({', '.join(NAMES[:2])}, ...), 1080p, 100 frames, cpu, in a process "
          f"of its own: peak resident memory {data['peak_bytes'] / GB:.2f} GB (limit 3); resident memory in MB after "
          f"frame {at}; from frame 40 to frame 100 it grew by {growth / MB:+.0f} MB (limit 50), the peak by "
          f"{peak_growth / MB:+.0f} MB; all samples from frame 40 on: {low:.0f} to {high:.0f} MB; lost {data['lost']} "
          f"of {10 * data['frames']} object-frames; {data['seconds_per_frame']:.2f} s per frame")

    assert data["peak_bytes"] < 3 * GB
    assert growth < 50 * MB
