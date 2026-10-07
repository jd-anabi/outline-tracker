"""The child process of tests/slow/test_memory.py: tracks ten coarse objects of a 1080p clip with
the real EdgeTAM on `cpu` and writes down its own memory after every tracked frame (SPEC 6.5).

Run as `python memory_child.py VIDEO RUN_FOLDER SAMPLES.json` with tests/ and tests/slow/ on
PYTHONPATH. VIDEO is `ten_ellipse_scene()` written by `synthetic.render`. The process holds what a
run of the tool holds (the model, the job, the results); the clip was rendered by the parent.

SAMPLES.json: status, frames (tracked), lost (object-frames without a mask), seconds_per_frame (s),
resident_bytes (a list: resident memory of this process right after tracked frame 1, 2, ..., read
with `ps -o rss=`), peak_bytes (the process's largest resident memory, `resource.getrusage`, read
at the end) and peak_after (the same reading after each tracked frame). macOS and Linux only.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame;
speeds in px per frame; frames are video frame numbers.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from outline_tracker.synthetic import GroundTruth, Scene, SceneObject
from outline_tracker.synthetic_shapes import Ellipse, Straight

NAMES = "ABCDEFGHIJ"


def ten_ellipse_scene(n_frames: int = 100) -> Scene:
    """Ten small dark ellipses on a plain 1920 x 1080 px background, 240 frames per second.

    Each has the semi-axes 8 and 3 px of the selftest's test shrimp. On frame 0 they stand in two
    rows of five, 350 px apart along u and 480 px along v; each moves 0.5 px per frame along its
    long axis, in a direction of its own (36 degrees apart), so after 100 frames any two are still
    more than 250 px apart.
    """
    objects = []
    for k, name in enumerate(NAMES):
        start = (260.3 + 350.0 * (k % 5), 300.4 + 480.0 * (k // 5))
        angle = 2.0 * math.pi * k / len(NAMES)
        step = (0.5 * math.cos(angle), 0.5 * math.sin(angle))
        objects.append(SceneObject(name, Ellipse(8.0, 3.0), Straight(start, step)))
    return Scene((1920, 1080), 240.0, n_frames, tuple(objects), mm_per_px=0.0324)


def resident_bytes() -> int:
    """Resident memory of this process now, in bytes (`ps` reports KiB on macOS and on Linux)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) * 1024


def peak_bytes() -> int:
    """The largest resident memory this process has had, in bytes (`ru_maxrss`: bytes on macOS,
    KiB on Linux)."""
    import resource  # not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def main(video: str, run_folder: str, samples_path: str) -> None:
    """Track the ten objects, coarse, the whole frame, every frame, and write SAMPLES.json."""
    from pipeline_helpers import calibrated_session, track_job
    from tracking_helpers import track

    from outline_tracker.segmenter import hf

    clip = GroundTruth(ten_ellipse_scene(), Path(video))
    session = calibrated_session(clip, Path(run_folder), [track(clip, name) for name in NAMES], step=1)
    resident, peak_after = [], []

    def sample(done: int) -> None:
        resident.append(resident_bytes())
        peak_after.append(peak_bytes())

    done = track_job(clip, session, Path(run_folder), hf.HFSegmenter("edgetam", "cpu"), on_frame=sample)
    Path(samples_path).write_text(json.dumps({
        "status": done.status, "frames": done.frames, "lost": done.lost, "seconds_per_frame": done.seconds_per_frame,
        "resident_bytes": resident, "peak_after": peak_after, "peak_bytes": peak_bytes(), "log": done.log,
    }), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:4])
