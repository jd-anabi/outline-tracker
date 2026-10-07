"""A session for a synthetic clip, tracked with the real model and exported: shared by the slow
pipeline tests (tests/slow/test_fine_mode.py, test_coarse_lowres.py, test_memory.py and its child
process memory_child.py).

The session is made as the fast tracking tests make one (tests/tracking_helpers.py): clicks from
the scene's own paths, fps_true = the scene's frame rate. It gets a calibration stick at the
scene's own scale, so the mm columns of the exports are the scene's mm.

Units and coordinates (SPEC 3): px in Tracker's convention, u to the right, v downward, pixel
(column c, row r) with its center at (c + 0.5, r + 0.5); mm in the session's axes, y up, origin at
the middle of the frame; frames are video frame numbers; times in s. Nothing here imports torch.
"""

from __future__ import annotations

import faulthandler
import json
from types import SimpleNamespace

from export_helpers import table
from tracking_helpers import make_session

from outline_tracker import schema
from outline_tracker.export import export_all
from outline_tracker.tracking import Callbacks, Job, run_job

STICK_PX = 1000.0  # length of the calibration stick of the sessions made here, px


def expect_minutes() -> None:
    """Call at the start of a test or fixture that runs for minutes. pytest is set to write the
    stacks of all threads into the output of a test that takes more than 120 s
    (`faulthandler_timeout`, pyproject.toml), which helps with a test that hangs; this cancels it
    for the test that is running, whose long run is expected."""
    faulthandler.cancel_dump_traceback_later()


def calibrated_session(clip, run_folder, tracks, *, step, end=None, circle=None):
    """A session for the synthetic clip `clip` (a GroundTruth with its `path`): video frames 0 to
    `end` (the clip's last frame if None) every `step`, the given tracks, fps_true = the scene's
    frame rate, a horizontal stick of 1000 px whose length in mm is 1000 x the scene's mm per px,
    and the origin of the axes at the middle of the frame, px. `circle`: the dish wall
    (`session.Circle`, px) or None; with a circle, coarse tracking shows the model the dish square.
    """
    scene = clip.scene
    for one in tracks:
        one.color = "#FFFF00"
    session = make_session(clip, run_folder, tracks, step=step, end=end, circle=circle)
    session.calibration.stick = {"p1_px": [100.0, 40.0], "p2_px": [100.0 + STICK_PX, 40.0],
                                 "length_mm": STICK_PX * scene.mm_per_px}
    session.axes.origin_px = [scene.size[0] / 2, scene.size[1] / 2]
    return session


def track_job(clip, session, run_folder, segmenter, on_frame=None) -> SimpleNamespace:
    """Track the session's pending objects with `tracking.run_job` and the given segmenter.

    `on_frame(done)`, if given, is called after every tracked frame with the number of frames
    tracked so far over the whole job. Returns status ("complete", "cancelled" or "failed"), log
    (the job's lines), seconds_per_frame (s per tracked frame over the job: decoding, the model,
    measuring and saving), frames (tracked frames) and lost (object-frames without a mask).
    """
    run_folder.mkdir(parents=True, exist_ok=True)
    log, progress, lost = [], [], []

    def report(done, total, s_per_frame, eta_s):
        progress.append((done, s_per_frame))
        if on_frame is not None:
            on_frame(done)

    callbacks = Callbacks(progress=report, log=log.append, finished=lambda status: None, should_cancel=lambda: False,
                          frame_result=lambda track_id, frame, record: lost.append(1) if not record.visible else None)
    status = run_job(Job(session, run_folder, clip.path, lambda: segmenter), callbacks)
    done, s_per_frame = progress[-1] if progress else (0, 0.0)
    return SimpleNamespace(status=status, log=log, seconds_per_frame=s_per_frame, frames=done, lost=len(lost))


def track_and_export(clip, session, run_folder, segmenter) -> SimpleNamespace:
    """`track_job`, then `export.export_all` without the overlay. Returns what `track_job` returns
    and, read back from the run folder, positions and shapes: positions.csv and shapes.csv as
    tables (their columns carry the units: mm, px, s, rad), and session: session.json as written
    by the job, parsed."""
    done = track_job(clip, session, run_folder, segmenter)
    export_all(run_folder, log=done.log.append)
    done.positions = table(run_folder / schema.POSITIONS_CSV)
    done.shapes = table(run_folder / schema.SHAPES_CSV)
    done.session = json.loads((run_folder / schema.SESSION_JSON).read_text(encoding="utf-8"))
    return done
