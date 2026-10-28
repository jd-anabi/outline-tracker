"""Session files of version 1, the format of 0.1.0, fitted to a clip on disk (W3 step 1; rule 5 of
docs/ROADMAP.md section 2).

Each file is an example of docs/design/formats.md in version 1, example 1 (made in the window) or example 2
(made by from-tracker), with the video block of the clip and its points moved into the clip's frame by
literal values. It is written as 0.1.0 wrote it (`json.dumps(..., indent=2, ensure_ascii=False)` and a line
end, in UTF-8) and gets the time `OLD_NS`, so that a test sees whether a file was written, copied or left.
"""

import json
import os
from pathlib import Path

from session_json_helpers import formats_example

from outline_tracker.core.fileio import sha256_first_64mib
from outline_tracker.video.reader import probe

OLD_NS = 1_700_000_000 * 10**9  # the time of an old file: a whole second, which every file system keeps


def video_block(clip_path, run_folder, relpath: str | None = None) -> dict:
    """The video block of version 1 (and 2) for the clip at `clip_path`, seen from `run_folder`: the facts of
    the file, its path relative to the folder with forward slashes, or `relpath` when given (a video that
    moved)."""
    clip_path, info = Path(clip_path), probe(clip_path)
    if relpath is None:
        relpath = Path(os.path.relpath(clip_path, run_folder)).as_posix()
    return {"relpath": relpath, "abspath": str(clip_path), "size": clip_path.stat().st_size,
            "sha256_first_64MiB": sha256_first_64mib(clip_path), "width": info.width, "height": info.height,
            "n_frames": info.n_frames, "fps_container": info.fps_container}


def window_session(clip_path, run_folder, relpath: str | None = None) -> dict:
    """Example 1 of formats.md (a session made in the window) in version 1, for a clip of 320 x 240 pixels and
    120 frames: every point inside the frame, the clip 0 to 118 in steps of 2, no runs and no corrections."""
    data = formats_example(1, 1)
    data["video"] = video_block(clip_path, run_folder, relpath)
    data["clip"] = {"start": 0, "end": 118, "step": 2}
    data["calibration"]["stick"].update(p1_px=[40.5, 200.5], p2_px=[160.5, 200.5])
    data["calibration"]["check"].update(p1_px=[40.5, 220.5], p2_px=[120.5, 220.5])
    data["axes"]["origin_px"] = [160.5, 120.5]
    data["circle"].update(points_px=[[60.5, 120.5], [160.5, 20.5], [260.5, 120.5]], center_px=[160.5, 120.5],
                          radius_px=100.0)
    data["probes"][0]["rect_px"] = [10, 10, 30, 30]
    track = data["tracks"][0]
    track.update(start_frame=0, head_px=[170.5, 110.5], ended_at=None)
    track["prompts"][0].update(frame=0, points_px=[[160.5, 120.5]])
    data["runs"], data["corrections"] = [], []
    return data


def tracker_session(clip_path, run_folder, student: str = "ana") -> dict:
    """Example 2 of formats.md (a session made by from-tracker) in version 1, for a clip of 320 x 240 pixels
    and at least 70 frames, of the student `student`: every point inside the frame, the clip 0 to 68 in steps
    of 2, one run and no corrections."""
    data = formats_example(2, 1)
    data["student"] = student
    data["video"] = video_block(clip_path, run_folder)
    data["clip"] = {"start": 0, "end": 68, "step": 2}
    data["axes"]["origin_px"] = [160.5, 120.5]
    for track, point in zip(data["tracks"], ([80.5, 100.5], [200.5, 170.5])):
        track["prompts"][0]["points_px"] = [point]
    data["runs"][0]["frames_done"] = 35
    return data


def write_v1(path, data: dict) -> Path:
    """Write `data` to `path` (its folder is made) as 0.1.0 wrote a session.json, with the time `OLD_NS`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.utime(path, ns=(OLD_NS, OLD_NS))
    return path


def snapshot(folder) -> dict:
    """Every file below `folder`, by its path relative to it: (its bytes, its time in ns)."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(folder.rglob("*")) if path.is_file()}
