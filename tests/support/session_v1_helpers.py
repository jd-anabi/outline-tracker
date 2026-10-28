"""Session files of version 1, the format of 0.1.0, fitted to a clip on disk (W3 step 1; rule 5 of
docs/ROADMAP.md section 2).

Each file is an example of docs/design/formats.md in version 1, example 1 (made in the window) or example 2
(made by from-tracker), with the video block of the clip and its points moved into the clip's frame by
literal values. It is written as 0.1.0 wrote it (`json.dumps(..., indent=2, ensure_ascii=False)` and a line
end, in UTF-8) and gets the time `OLD_NS`, so that a test sees whether a file was written, copied or left.

`old_run` makes a whole run folder of version 1 for the synthetic dish clip: such a session.json, and the
results.npz of its tracks as the tool tracks them with `ExactFake` (tests/support/export_helpers.py does the
same for the export tests), tracked in another folder so that nothing the tool writes there is kept but the
results.

Units and coordinates (SPEC 3): px in Tracker's convention (pixel centers at +0.5), mm, frames of the video.
"""

import copy
import json
import os
import shutil
from pathlib import Path

from session_json_helpers import formats_example
from tracking_helpers import center, frame_identity, run

from outline_tracker.core import schema
from outline_tracker.core.fileio import sha256_first_64mib
from outline_tracker.core.session import Session
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.video.reader import probe

OLD_NS = 1_700_000_000 * 10**9  # the time of an old file: a whole second, which every file system keeps
RUN = "dish_tracker_outline_ana"  # the run folder of "ana" for dish_tracker.mp4 (SPEC 8.1)
K = 0.1  # mm per px of both kinds of old run: a stick of 300 px for 30 mm, or a Tracker fit of 0.1
ORIGIN = (100.5, 60.25)  # the origin of the axes of both kinds, px
ANGLE_DEG = {"window": 0.0, "from-tracker": 90.0}  # the axis angle of each kind (formats.md, examples 1 and 2)
IDS = {"window": ("A", "B", "C"), "from-tracker": ("A", "B")}  # the tracks of each kind
COLORS = ("#FFFF00", "#FF00FF", "#00FFFF")


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


def old_run(clip, kind: str, scratch) -> Path:
    """A run folder of version 1 beside the dish clip (`clip`, a GroundTruth whose video is a copy in the test's
    own folder): `<clip folder>/RUN` with session.json made "window" (example 1: a stick, the dish circle and
    its crop, the axes at 0 degrees, the tracks A, B and C) or "from-tracker" (example 2: a scale fitted to a
    Tracker export, no circle, the axes at 90 degrees, the tracks A and B), each with one click on the center
    of its object on frame 0, the clip 0 to 118 in steps of 2, fps_true 240 (the scene's), one run, no
    corrections; and results.npz, tracked in the folder `scratch`. Both files get the time `OLD_NS`. Returns
    the run folder."""
    run_folder = clip.path.parent / RUN
    if kind == "window":
        data = window_session(clip.path, run_folder)
        data["calibration"]["stick"] = {"p1_px": [10.0, 20.0], "p2_px": [310.0, 20.0], "length_mm": 30.0}
        cu, cv, radius = clip.scene.dish
        data["circle"].update(points_px=[[cu - radius, cv], [cu, cv - radius], [cu + radius, cv]],
                              center_px=[cu, cv], radius_px=radius)
        data["time"]["fps_true"] = 240.0
    else:
        data = tracker_session(clip.path, run_folder)
        data["calibration"]["tracker_fit"]["mm_per_px"] = K
        data["clip"]["end"] = 118
    data["axes"] = {"origin_px": list(ORIGIN), "angle_deg": ANGLE_DEG[kind]}
    frame_hash, decoder = frame_identity(clip, 0)
    data["tracks"] = [{"id": track_id, "color": color, "mode": "coarse", "fine_window_px": None, "start_frame": 0,
                       "head_px": None, "ended_at": None,
                       "prompts": [{"frame": 0, "frame_hash": frame_hash, "decoder": decoder,
                                    "points_px": [[float(value) for value in center(clip, track_id, 0)]],
                                    "labels": [1]}]}
                      for track_id, color in zip(IDS[kind], COLORS)]
    example_run = formats_example(1 if kind == "window" else 2, 1)["runs"][0]
    data["runs"] = [dict(example_run, tracks=list(IDS[kind]), start_frame=0, mode="coarse", frames_done=60)]
    data["corrections"] = []

    tracking = copy.deepcopy(data)
    tracking["runs"] = []
    status, _ = run(clip, Session.from_json(tracking), scratch, ExactFake(clip))
    assert status == "complete"
    results = run_folder / schema.RESULTS_NPZ
    run_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(scratch) / schema.RESULTS_NPZ, results)
    os.utime(results, ns=(OLD_NS, OLD_NS))
    write_v1(run_folder / schema.SESSION_JSON, data)
    return run_folder
