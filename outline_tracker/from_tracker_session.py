"""What `from_tracker` settles before anything is tracked: fps_true, the student's name, the run
folder, and the session that a Tracker export stands for (SPEC 11; decisions X9, X14, X18).

A Tracker export gives last week's `Plan` (`tracker_io.make_plan`): the objects' names, one pixel
position each on the first marked frame, the frames to track, fps_true and Tracker's calibration.
Here that becomes a `Session`: the clip is the plan's frames, the axes and the scale are the fitted
calibration in the form of SPEC 3.2, and every object is a track with one positive click.

Units and coordinates (SPEC 3): px in Tracker's image coordinates (origin at the top-left corner, u
to the right, v downward, pixel centers at +0.5); mm in the user's axes with y up; fps_true in frames
per second; frames are video frame numbers. No Qt, no torch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from outline_tracker import schema
from outline_tracker.geometry import WorldFrame, find_manifest, fps_from_manifest
from outline_tracker.run_folder import default_run_folder, folder_has_foreign_tables
from outline_tracker.session import (Axes, CalibrationSettings, Clip, Processing, Prompt, Session, TimeSettings,
                                     Track, VideoRef)
from outline_tracker.tracker_io import Plan, read_tracker_export
from outline_tracker.video import VideoInfo

# Last week's overlay colors (shrimp.segment.COLORS, there as blue, green, red) as RGB hex, in the
# same order: the first track is yellow, and the eleventh has the first color again (X14).
TRACK_COLORS = ["#FFFF00", "#FF00FF", "#00FF00", "#0080FF", "#FF8000", "#00FFFF", "#FF0080", "#FF0000", "#0000FF",
                "#80FF80"]
ELSEWHERE = "Choose another folder with --out DIR, or another name with --student NAME."


@dataclass(frozen=True)
class FpsChoice:
    """Where fps_true comes from (decision X9).

    fps: fps_true in frames per second, or None when neither `--fps` nor a manifest gives it: then
    last week's `make_plan` takes it from the export's t column, or raises its "Unknown fps_true".
    source: what `time.source` of the session says: "typed", "manifest" or "tracker-export".
    manifest: the manifest that gave it. passed_over: a manifest that lists the video but has no
    positive number in its fps_true cell, so that it was not used.
    """

    fps: float | None
    source: str
    manifest: Path | None = None
    passed_over: Path | None = None


def choose_fps(typed, video: Path, export: Path, cwd=None) -> FpsChoice:
    """fps_true for a run, in the order of decision X9: `typed` (the `--fps` option), else the
    `data/manifest.csv` that lists the video, looked for in the current folder, then upward from the
    export's folder, then upward from the video's (`geometry.find_manifest`); else nothing, which
    leaves it to the export's t column.

    typed: frames per second, or None. video, export: paths. cwd: the current folder (None: the
    process's own). A manifest cell that is empty, zero, negative or not a number is no source.
    Raises ValueError for a `typed` that is not a positive, finite number.
    """
    if typed is not None:
        if not (math.isfinite(typed) and typed > 0):
            raise ValueError("--fps must be a positive number of frames per second (fps_true, for example "
                             f"--fps 239.6), not {typed:g}.")
        return FpsChoice(float(typed), "typed")
    manifest = find_manifest(video, export, cwd)
    if manifest is not None:
        fps = fps_from_manifest(video, manifest)
        if fps is not None and math.isfinite(fps) and fps > 0:  # an empty cell reads as nan, a 0 as 0.0
            return FpsChoice(float(fps), "manifest", manifest)
    return FpsChoice(None, "tracker-export", passed_over=manifest)


def choose_run_folder(video: Path, export: Path, out, student: str | None) -> tuple[Path, str, bool]:
    """(run folder, student name, whether the name was taken from the export's folder) (SPEC 8.1).

    The name is `student` when given, else the name of the export's home folder: the folder the
    export is in, or the one above it when that folder is called `extra` (last week's per-student
    folder, where the model's files went). The run folder is `out` when given, else
    `<video folder>/<video stem>_outline_<name>/` (`run_folder.default_run_folder`, which makes the
    name safe for a folder). Nothing is created. Raises ValueError when a name is needed for the
    folder and there is none. No units or coordinates.
    """
    inferred = student is None
    if inferred:
        folder = Path(os.path.abspath(export)).parent
        student = (folder.parent if folder.name == "extra" else folder).name
    if out is not None:
        return Path(out), student, inferred
    if inferred and not student.strip():
        raise ValueError(f"The folder of {export.name} has no name to take as yours: give your name with "
                         "--student NAME, or the run folder with --out DIR.")
    return default_run_folder(video, student), student, inferred


def check_run_folder(run: Path) -> bool:
    """Make sure a run may write into `run`; returns True when the folder holds an earlier
    from-tracker run, which the new run then replaces (as last week's script wrote over its files).

    Raises ValueError, and nothing is touched, for a folder that directly holds other .csv or .txt
    files (a student's folder of Tracker files; their notebooks read every such file as a track),
    and for one that holds a session this command must not replace: one that was made in the app
    (it has no scale fitted to a Tracker export), one with corrections, or one that cannot be read.
    No units or coordinates.
    """
    if folder_has_foreign_tables(run):
        raise ValueError(f"{run} holds other .csv or .txt files: this looks like a folder of Tracker files, and "
                         "the run's files are not written into one. Choose another folder with --out DIR.")
    session_path = run / schema.SESSION_JSON
    if not session_path.is_file():
        return (run / schema.RESULTS_NPZ).is_file()
    try:
        earlier = Session.load(session_path)
    except ValueError as err:
        raise ValueError(f"{run} holds a session that cannot be read, so it is not replaced: {err} "
                         f"{ELSEWHERE}") from None
    if earlier.calibration.tracker_fit is None or earlier.corrections:
        raise ValueError(f"{run} holds a run that was made or corrected in the app: from-tracker does not "
                         f"replace it. {ELSEWHERE}")
    return True


def count_fit_points(export: Path) -> int:
    """How many rows of a Tracker export the calibration was fitted to: those with pixelx, pixely
    (px) and x, y (mm) all present, over every point mass of the file (`tracker_io.fit_calibration`
    uses exactly these)."""
    rows = pd.concat(read_tracker_export(export).values(), ignore_index=True)
    return int(np.isfinite(rows[["pixelx", "pixely", "x", "y"]].to_numpy(float)).all(axis=1).sum())


def build_session(plan: Plan, world: WorldFrame, info: VideoInfo, video: Path, run: Path, *, last_frame: int,
                  fps: FpsChoice, student: str, fine_ids: list[str], model: str, device: str, n_fit_points: int,
                  start_hash: str, decoder: str) -> Session:
    """The session a Tracker export stands for, ready for `tracking.run_job`.

    plan: last week's plan (names, first pixel positions in image px, start frame, step, fps_true
    in frames per second, the fitted calibration). world: that calibration as scale (mm per px),
    origin (image px) and axis angle (rad), SPEC 3.2. info: the video's size in px, frame count and
    the frame rate the file states. video, run: the video and the run folder, whose paths the
    session stores. last_frame: the last video frame number to track, on the plan's grid. fps: where
    plan.fps came from. fine_ids: the names tracked in fine mode. model, device: as given to the
    command; `model` names the Tracker-format folder. n_fit_points: rows the calibration was fitted
    to. start_hash, decoder: `video.frame_hash` of the start frame as tracking decodes it, and
    `video.decoder_tag()`, so that the run-start guard has something to compare (SPEC 3.5).

    There is no stick: the scale is `calibration.tracker_fit` (X18). Each name is a track with one
    positive click at its first pixel position, in last week's colors. `complete` is False until a
    run has tracked every frame.
    """
    tracks = [
        Track(id=name, color=TRACK_COLORS[k % len(TRACK_COLORS)], mode="fine" if name in fine_ids else "coarse",
              start_frame=plan.start,
              prompts=[Prompt(frame=plan.start, frame_hash=start_hash, decoder=decoder,
                              points_px=[[float(u), float(v)]], labels=[1])])
        for k, (name, (u, v)) in enumerate(zip(plan.names, plan.points_px))
    ]
    return Session(
        student=student, complete=False,
        video=VideoRef.from_file(video, run, width=info.width, height=info.height, n_frames=info.n_frames,
                                 fps_container=info.fps_container),
        clip=Clip(start=plan.start, end=int(last_frame), step=plan.step),
        time=TimeSettings(fps_true=plan.fps, source=fps.source,
                          manifest_path=None if fps.manifest is None else str(fps.manifest)),
        calibration=CalibrationSettings(stick=None, check=None, tracker_fit={
            "mm_per_px": world.k_mm_per_px, "rms_mm": float(plan.calibration.rms_mm), "n_points": n_fit_points}),
        axes=Axes(origin_px=[world.u0, world.v0], angle_deg=math.degrees(world.alpha_rad)),
        processing=Processing(model=model, device=device),
        tracks=tracks,
    )
