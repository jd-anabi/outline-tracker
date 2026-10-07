"""from-tracker: last week's workflow with this week's outputs (SPEC 11). The fallback if the app slips.

Calibration and start points come from a file exported from Tracker (one track, or several point
masses marked on one frame), exactly as last week: `tracker_io.make_plan` decides the frames, the
start positions, fps_true and the pixel -> mm map. `from_tracker` turns that into a session
(outline_tracker/from_tracker_session.py), tracks it with `tracking.run_job` on the whole frame (a
Tracker export has no dish circle, so there is no dish crop), and writes every file of SPEC 8 with
`export.export_all`. Objects named with `fine_ids` get a fine run each, in a window chosen from
their mask on the first frame (SPEC 6.3).

What is last week's and stays: the errors of `make_plan`, the console lines, and `_flags`, the three
checks behind the `CHECK:` lines (lost, jump, size change), moved here unchanged
(tests/test_port_fidelity.py compares it with the reference). The nine flags of SPEC 9 are in the
`flags` columns of the CSV files and in run.log; they are not CHECK lines (decision X22).

Units and coordinates (SPEC 3): px in Tracker's image coordinates (pixel centers at +0.5); mm in
the user's axes with y up; s; fps_true in frames per second; frames are video frame numbers and
t_s = frame / fps_true. No Qt. torch is loaded only by `load_segmenter`, when a run needs the model.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outline_tracker import schema
from outline_tracker.cli_probe import SLOW_MOTION_FPS, UNKNOWN_FPS
from outline_tracker.export import ExportReport, derive_all, export_all
from outline_tracker.export_tables import tracker_file_names
from outline_tracker.from_tracker_session import (FpsChoice, build_session, check_run_folder, choose_fps,
                                                  choose_run_folder, count_fit_points)
from outline_tracker.geometry import world_frame_from_calibration
from outline_tracker.results import ResultsStore
from outline_tracker.session import Session
from outline_tracker.tracker_io import Plan, make_plan
from outline_tracker.tracking import CANCELLED, FAILED, Callbacks, Job, run_job
from outline_tracker.video import decoder_tag, frame_hash, iter_rgb_frames, probe

MAX_SPEED_MM_S = 100.0  # last week's limit: a step faster than this, in mm per s, is a jump to something else


# Last week's three checks on one track, unchanged: name, then arrays with one entry per tracked
# frame: video frame numbers, t in s, x and y in mm (NaN where the object was lost), mask area in px.
# Returns the messages that the console shows as `CHECK:` lines.
def _flags(name, frames, t, x, y, area):
    msgs = []
    lost = ~np.isfinite(x)
    if lost.any():
        msgs.append(f"{name}: lost in {int(lost.sum())} of {len(x)} frames (first at t = {t[np.argmax(lost)]:.3f} s)")
    ok = np.nonzero(~lost)[0]
    if len(ok) >= 2:
        sp = np.hypot(np.diff(x[ok]), np.diff(y[ok])) / np.diff(t[ok])
        for j in np.nonzero(sp > MAX_SPEED_MM_S)[0][:5]:
            msgs.append(f"{name}: jumps {sp[j] * np.diff(t[ok])[j]:.2f} mm at t = {t[ok][j + 1]:.3f} s "
                        f"(frame {frames[ok][j + 1]}): check the video there")
        a = area[ok].astype(float)
        med = np.median(a)
        big = np.nonzero((a > 2 * med) | (a < 0.5 * med))[0]
        if len(big):
            msgs.append(f"{name}: outline size changes by more than 2x in {len(big)} frames (first at "
                        f"t = {t[ok][big[0]]:.3f} s): two shrimp touching, or a lost outline")
    return msgs


@dataclass(frozen=True)
class FromTrackerResult:
    """What a from-tracker run left behind.

    run_folder: the folder with every file of the run (SPEC 8.1). files: the Tracker-format files
    `<run folder>/<model>/<id>.csv`, one per tracked object, in the order of the export's names (x
    and y in mm in Tracker's axes, pixelx and pixely in px, t in s). overlay: overlay.mp4, or None
    when it was not asked for or could not be written. flags: last week's CHECK messages (`_flags`)
    for those objects, in the same order; nothing else (X22). plan: last week's plan, as
    `tracker_io.make_plan` made it from the export (frame numbers, fps_true in frames per second).
    seconds_per_frame: s per tracked frame over the whole job; 0.0 if no frame was tracked.
    """

    run_folder: Path
    files: list[Path]
    overlay: Path | None
    flags: list[str]
    plan: Plan
    seconds_per_frame: float


def load_segmenter(model: str, device: str):
    """The real model behind the segmenter protocol: `segmenter.hf.HFSegmenter(model, device)`.

    model: "edgetam", "sam2" or "sam2-small". device: "auto" (the fastest that works), "cpu", "mps"
    or "cuda". The first call downloads the model. Images and prompts are in px of the image given
    (segmenter/base.py). This is the one place of the module that loads torch.
    """
    from outline_tracker.segmenter.hf import HFSegmenter  # here, never at the top: it loads torch

    return HFSegmenter(model, device)


def from_tracker(video, export, *, fine_ids: Iterable[str] = (), step: int | None = None,
                 seconds: float | None = None, fps: float | None = None, out=None, student: str | None = None,
                 model: str = "edgetam", device: str = "auto", overlay: bool = True, segmenter=None,
                 log: Callable[[str], object] = print) -> FromTrackerResult:
    """Track the objects marked in a Tracker export and write the files of SPEC 8.

    video: the video that was open in Tracker (the `..._tracker.mp4` copy), so that frame numbers
    and pixels match. export: a file exported from Tracker with the columns frame, x, y (mm, in
    Tracker's axes, y up), pixelx, pixely (px, pixel centers at +0.5): one track, or several point
    masses all marked on the same frame.
    fine_ids: names, as they are in the export, of the objects tracked in fine mode; the others
    are coarse. step: track every step-th frame (default: the Tracker track's step, or 2).
    seconds: how long to track, in s of real time (default: the length of the Tracker track, or
    10 s). fps: fps_true in frames per second; without it, the manifest, then the export's t column
    (`from_tracker_session.choose_fps`, decision X9).
    out: the run folder (default: `<video folder>/<video stem>_outline_<student>/`). student: the
    name for that folder and for the session (default: the name of the export's home folder).
    model: "edgetam" or "sam2"; it also names the folder of the Tracker-format files. device:
    "auto", "cpu", "mps" or "cuda". overlay: also write overlay.mp4. segmenter: a segmenter to use
    instead of loading the model (the tests' stand-in). log: called with every console line.

    The run folder then holds session.json, results.npz, positions.csv, `<model>/<id>.csv`,
    shapes.csv, radial.csv, outlines.npz, README.txt, run.log and, if asked for, overlay.mp4. An
    earlier from-tracker run in that folder is replaced, as last week. After Ctrl+C the frames
    tracked so far are exported, and the session says that the results are not complete.

    Raises, before anything is written: FileNotFoundError (no such video or export) and ValueError
    with a message for the user: what `make_plan` refuses (its messages are last week's), a mirrored
    calibration, a name in `fine_ids` that the export does not have, an fps that is not positive,
    an export that starts beyond the video's last frame or marks a point outside the picture, a run
    folder that holds Tracker files or a run made in the app. OSError when the video or the model
    cannot be read or loaded. RuntimeError when tracking failed on a frame: the frames before it
    were exported first, and the message says where.
    """
    video, export = Path(video), Path(export)
    if not video.is_file():
        raise FileNotFoundError(f"No such video: {video}")
    if not export.is_file():
        raise FileNotFoundError(f"No such Tracker export: {export}")
    fps_from = choose_fps(fps, video, export)
    # video=None: the manifest was looked for above (X9), so last week's function is left with the
    # fps given here or, without one, the export's t column and its own "Unknown fps_true"
    plan = make_plan(export, seconds, step, fps_from.fps, video=None)
    if not (math.isfinite(plan.fps) and plan.fps > 0):
        raise ValueError(UNKNOWN_FPS)
    try:
        world = world_frame_from_calibration(plan.calibration)
    except ValueError as err:  # mirrored (y down on screen), or without a usable scale
        raise type(err)(f"{export.name}: {err}") from None
    fine = [fine_ids] if isinstance(fine_ids, str) else list(fine_ids)
    unknown = [name for name in fine if name not in plan.names]
    if unknown:
        raise ValueError(f"--fine names {', '.join(unknown)}, but {export.name} holds {', '.join(plan.names)}. Give "
                         "the names as they are in the export, separated by commas.")
    file_names = tracker_file_names(plan.names)  # two names that would share a file are refused now, not after tracking

    info = probe(video)
    frames = plan.frames
    shortened = bool(info.n_frames) and frames[-1] >= info.n_frames
    if shortened:
        frames = [frame for frame in frames if frame < info.n_frames]
        if not frames:
            raise ValueError(_starts_beyond(export, plan.start, video, f"has {info.n_frames} frames"))
    outside = [name for name, (u, v) in zip(plan.names, plan.points_px)
               if not (0 <= u < info.width and 0 <= v < info.height)]
    if outside:
        raise ValueError(f"{export.name} marks {', '.join(outside)} outside the picture of {video.name} "
                         f"({info.width} x {info.height} px). Is this the video that was open in Tracker?")
    run, student, inferred = choose_run_folder(video, export, out, student)
    replaces = check_run_folder(run)

    if shortened:
        log(f"  The video has {info.n_frames} frames: tracking only up to frame {frames[-1]}.")
    log(f"{model}: {len(plan.names)} shrimp ({', '.join(plan.names)}), frames {frames[0]}-{frames[-1]} every "
        f"{plan.step} ({len(frames)} frames, {len(frames) * plan.step / plan.fps:.1f} s at fps_true = {plan.fps:g}); "
        f"scale {1000 * plan.calibration.mm_per_px:.2f} um per pixel")
    for line in _setup_lines(plan, fps_from, export, student, inferred, run, replaces):
        log(line)

    with closing(iter_rgb_frames(video, [plan.start])) as decoded:
        first = next(decoded, None)
    if first is None:  # the file's frame count is missing or wrong, and the video ends before the start
        raise ValueError(_starts_beyond(export, plan.start, video, "ends before it"))
    session = build_session(plan, world, info, video, run, last_frame=frames[-1], fps=fps_from, student=student,
                            fine_ids=fine, model=model, device=device, n_fit_points=count_fit_points(export),
                            start_hash=frame_hash(first[1]), decoder=decoder_tag())

    if segmenter is None:
        log("  loading the model (the first time this downloads it) ...")
        segmenter = load_segmenter(model, device)
        log(f"  running on: {segmenter.device}")
    if replaces:  # only now: a model that cannot be loaded leaves the earlier run as it was
        for stale in (schema.RESULTS_NPZ, schema.OVERLAY_MP4):
            (run / stale).unlink(missing_ok=True)
    console = _Console(log)
    job = Job(session, run, video, lambda: segmenter)
    status = run_job(job, Callbacks(progress=console.progress, frame_result=lambda *result: None,
                                    log=lambda text: log(textwrap.indent(text, "  ")),
                                    finished=lambda status: None, should_cancel=lambda: False))
    if status == CANCELLED:
        log(f"  stopped at frame {console.done}; saving what was tracked so far")

    files, flags, overlay_path = [], [], None
    if (run / schema.RESULTS_NPZ).is_file():  # at least one frame was tracked
        report = export_all(run, overlay=overlay, log=lambda text: log(f"  {text}"))
        files, flags = _tracks_and_checks(job.session, run, plan.names, file_names, report)
        overlay_path = next((path for path in report.files if path.parent == run and path.suffix == ".mp4"), None)
        for message in flags:
            log("  CHECK: " + message)
        log(f"  saved {', '.join(str(path) for path in files)}" + (f" and {overlay_path}" if overlay_path else ""))
        others = [path.name for path in report.files if path.parent == run and path != overlay_path]
        log(f"  also in {run}: {', '.join([*others, schema.SESSION_JSON, schema.RESULTS_NPZ])}")
    if status == FAILED:
        kept = f"The frames tracked before it are saved in {run}." if files else f"Nothing was tracked ({run})."
        raise RuntimeError(f"Tracking stopped with an error (see the lines above). {kept}")
    return FromTrackerResult(run, files, overlay_path, flags, plan, console.s_per_frame)


def _starts_beyond(export: Path, start: int, video: Path, what: str) -> str:
    """The message for an export whose first marked frame the video does not have."""
    return (f"{export.name} starts on frame {start}, but {video.name} {what}. Is this the video that was open in "
            "Tracker (the ..._tracker.mp4 copy)?")


def _setup_lines(plan: Plan, fps_from: FpsChoice, export: Path, student: str, inferred: bool, run: Path,
                 replaces: bool) -> list[str]:
    """The console lines under the plan line: where fps_true came from (frames per second), the
    student's name and the run folder, and last week's warnings. Plain ASCII apart from names."""
    lines = []
    if fps_from.source == "typed":
        lines.append(f"  fps_true = {plan.fps:g} (given with --fps)")
    elif fps_from.source == "manifest":
        lines.append(f"  fps_true = {plan.fps:g} (from the manifest {fps_from.manifest})")
    else:
        if fps_from.passed_over is not None:
            lines.append(f"  NOTE: {fps_from.passed_over} lists this video without a usable fps_true.")
        lines.append(f"  fps_true = {plan.fps:g} (from the t column of {export.name}: that is right only if "
                     "Tracker's frame rate was set to fps_true. To be sure, give --fps or fill in data/manifest.csv)")
    if plan.fps < SLOW_MOTION_FPS:
        lines.append(f"  WARNING: fps_true = {plan.fps:g} is below {SLOW_MOTION_FPS:g} frames per second. For a "
                     "slow-motion video that usually means a re-timed copy: check the file with "
                     "`outline-tracker check`.")
    if plan.calibration.rms_mm > 1e-3:
        lines.append(f"  WARNING: the pixel and mm columns of {export.name} do not fit one calibration "
                     f"(rms {1000 * plan.calibration.rms_mm:.1f} um). Were they exported from the same .trk?")
    if inferred:
        lines.append(f"  student name: {student or 'none'} (from the folder of {export.name}; for another: "
                     "--student NAME)")
    else:
        lines.append(f"  student name: {student}")
    lines.append(f"  run folder: {run}")
    if replaces:
        lines.append("  This folder holds an earlier from-tracker run: this run replaces its results.")
    return lines


class _Console:
    """Last week's progress lines, from the job's progress reports. `done`: frames tracked so far,
    over all runs of the job; `s_per_frame`: s per tracked frame so far."""

    def __init__(self, log: Callable[[str], object]):
        self.log, self.done, self.s_per_frame = log, 0, 0.0

    def progress(self, done: int, total: int, s_per_frame: float, eta_s: float) -> None:
        self.done, self.s_per_frame = done, s_per_frame
        if done in (1, 5) or done % 50 == 0 or done == total:
            self.log(f"  frame {done}/{total}: {s_per_frame:.2f} s per frame, about {eta_s / 60:.0f} min left")


def _tracks_and_checks(session: Session, run: Path, names: list[str], file_names: dict[str, str],
                       report: ExportReport) -> tuple[list[Path], list[str]]:
    """(the Tracker-format files, last week's CHECK messages) of the exported tracks, both in the
    order of `names`. The checks are `_flags` on what the export wrote for each track: t in s, x
    and y in mm in the user's axes (NaN where lost), and the mask area in px. A track without a
    tracked frame (the run was stopped before its turn) has neither a file nor a check."""
    data = derive_all(session, ResultsStore.load(run / schema.RESULTS_NPZ))
    in_model_folder = {path.name: path for path in report.files if path.parent != run}
    files, flags = [], []
    for name in names:
        if name not in data.derived:
            continue
        track = data.derived[name]
        flags += _flags(name, track.frame, track.t_s, track.x_mm, track.y_mm, data.arrays[name].area_px)
        # `<id>.csv`, or `<id>.csv.new` when another program held `<id>.csv` open (the export said so)
        files.append(in_model_folder.get(file_names[name]) or in_model_folder[f"{file_names[name]}.new"])
    return files, flags
