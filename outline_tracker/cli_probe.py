"""The `probe` command (SPEC 4.6, 8.7, 11): brightness probes only, with no model.

Two forms. `probe VIDEO --rect NAME:u0,v0,u1,v1 ...` takes everything from the command line.
`probe SESSION.json` takes the rectangles, the frames, fps_true and the video from a session and writes
into the session's run folder (the folder that holds the session file); the session is only read.
Either way the result is probes.csv (one row per frame and rectangle) plus a short entry appended to
run.log in the same folder.

Units and coordinates: a rectangle is two opposite corners (u0, v0, u1, v1) in image px in Tracker's
convention (SPEC 3.1): u to the right, v down, from the top-left corner of the frame, pixel centers at
+0.5. Frames are the video's own frame numbers, counted from 0; the first and the last frame asked for
are both measured, and every frame between them. fps_true is the real frame rate in frames per s
(t_s = frame / fps_true); the frame rate written in the file is never used.

`cli.py` holds the command's parser and turns the errors raised here into one `ERROR: ...` line.
pandas and OpenCV are imported only when the command runs. No Qt, no torch.
"""

from __future__ import annotations

import argparse
import math
import operator
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from outline_tracker import __version__

if TYPE_CHECKING:  # for the annotations only: video loads OpenCV
    from outline_tracker.video import VideoInfo

USAGE = """%(prog)s VIDEO --rect NAME:u0,v0,u1,v1 [--rect ...] [--start F] [--end F]
                             [--fps F] (--student NAME | --out DIR)
       %(prog)s SESSION.json"""

EXAMPLE = """\
Usage:
    outline-tracker probe "path/to/your video_tracker.mp4" --rect LED1:100,100,140,140 --fps 239.6 --student ana
    outline-tracker probe "path/to/the run folder/session.json"

A rectangle is NAME:u0,v0,u1,v1: a name, a colon, then two opposite corners in image pixels (u to the
right, v down, from the top-left corner of the frame, as Tracker shows pixel positions). The result is
probes.csv with the columns frame, t_s, probe, r, g, b, gray (means on the 0 to 255 scale), one row per
frame and rectangle. With a session file, the rectangles, the frames, fps_true and the folder come from
the session.
"""

# the wording of last week's script (tracker_io.make_plan), which from-tracker also gives
UNKNOWN_FPS = "Unknown fps_true: give it with --fps (e.g. --fps 239.6), or fill in data/manifest.csv."
SLOW_MOTION_FPS = 100.0  # an fps_true below this gets a warning (SPEC 3.3)


@dataclass(frozen=True)
class _Job:
    """One probe pass: what to read, what to measure, where to write."""

    video: Path
    info: VideoInfo  # what the file says about itself, for run.log
    boxes: dict[str, Sequence[float]]  # name -> (u0, v0, u1, v1) in image px, in the order to write
    start: int  # first and last frame to measure: video frame numbers, both included
    end: int
    fps_true: float  # frames per s
    fps_source: str  # where fps_true came from, for the console and run.log
    out: Path  # the folder for probes.csv and run.log
    warnings: list[str] = field(default_factory=list)


def parse_rect(text: str) -> tuple[str, tuple[float, ...]]:
    """One `--rect NAME:u0,v0,u1,v1` as (name, (u0, v0, u1, v1)).

    The four numbers are two opposite corners of the rectangle in image px (u to the right, v down, pixel
    centers at +0.5; SPEC 3.1), in either order. The name is everything before the last colon. Raises
    argparse.ArgumentTypeError (a usage error, exit code 2) for anything else: no name, no colon, or not
    four finite numbers.
    """
    name, colon, numbers = text.rpartition(":")
    name = name.strip()
    try:
        corners = tuple(float(part) for part in numbers.split(","))
    except ValueError:
        corners = ()
    if not (colon and name and len(corners) == 4 and all(math.isfinite(corner) for corner in corners)):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not NAME:u0,v0,u1,v1 (a name, a colon, then two opposite corners in image px, "
            "for example LED1:100,100,140,140)")
    return name, corners


def run(args: argparse.Namespace) -> int:
    """Run `probe` for parsed arguments: measure, write probes.csv, append to run.log. Returns 0.

    `args` has `source` (a video, or a session file: a name ending in .json), `rect` (a list of
    `parse_rect` results, or None), `start` and `end` (video frame numbers counted from 0, both included;
    None for the first and the last frame of the video), `fps` (fps_true in frames per s, or None: then
    from data/manifest.csv, decision X9), `out` (a folder) and `student` (a name: the folder is then the
    default run folder of SPEC 8.1). Rectangles are in image px (u to the right, v down, pixel centers at
    +0.5); t_s = frame / fps_true in s. With a session, everything comes from the session and probes.csv
    goes next to it.

    Prints what it does. A video that ends before the last frame asked for gives a `NOTE:` line and the
    rows up to its last frame. Raises ValueError or OSError with a message for the user, before anything
    is written, when something is missing or wrong (no rectangle, unknown fps_true, no output folder, a
    folder of Tracker files, a rectangle outside the frame, no frame to read, a video that is not there).
    """
    from outline_tracker import fileio, probes, schema  # here: they load pandas and OpenCV

    job = _session_job(args) if Path(args.source).suffix.lower() == ".json" else _video_job(args)
    n_boxes = f"{len(job.boxes)} rectangle{'' if len(job.boxes) == 1 else 's'}"
    print(f"probe: {n_boxes} ({', '.join(job.boxes)}), frames {job.start} to {job.end} of {job.video.name}; "
          f"fps_true = {job.fps_true:g} ({job.fps_source})")
    for warning in job.warnings:
        print(f"WARNING: {warning}")
    notes: list[str] = []
    table = probes.measure_probes(job.video, job.boxes, job.start, job.end, job.fps_true, log=notes.append)
    if table.empty:
        raise ValueError(" ".join(notes))  # measure_probes says why: the video ends before the first frame
    for note in notes:
        print(f"NOTE: {note}")

    text = schema.csv_text(schema.PROBES, table.to_dict("records"))
    written = fileio.atomic_write(job.out / schema.PROBES_CSV,
                                  lambda tmp: tmp.write_bytes(text.encode(schema.CSV_ENCODING)))
    n_frames = len(table) // len(job.boxes)
    print(f"wrote {written}: {len(table)} rows ({n_frames} frame{'' if n_frames == 1 else 's'} x {n_boxes})")
    first, last = int(table["frame"].iloc[0]), int(table["frame"].iloc[-1])
    entry = _log_entry(job, first, last, n_frames, f"{written.name} ({len(table)} rows)", notes)
    log = _append_run_log(job.out, entry)
    for path, name in ((written, schema.PROBES_CSV), (log, schema.RUN_LOG)):
        if path.name != name:  # atomic_write's way out when the target stays locked
            print(f"NOTE: {name} is open in another program, so this went to {path.name}. Close {name}, "
                  f"then run the command again or rename the file.")
    return 0


# --------------------------------------------------------------------------- the two forms

def _video_job(args: argparse.Namespace) -> _Job:
    """The pass for `probe VIDEO --rect ...`."""
    from outline_tracker import video

    if not args.rect:
        raise ValueError("No rectangle to measure: give at least one with --rect NAME:u0,v0,u1,v1 (two opposite "
                         "corners in image px), for example --rect LED1:100,100,140,140.")
    boxes = _boxes(args.rect)
    path = Path(args.source)
    info = video.probe(path)
    out = _output_folder(args, path)
    fps_true, source = _fps_true(args.fps, path)
    start = 0 if args.start is None else args.start
    end = args.end
    if end is None:
        end = info.n_frames - 1
        if end < 0:
            raise ValueError("The file does not say how many frames it has: give the last frame to measure "
                             "with --end F.")
        if start > end:
            raise ValueError(f"The video has {info.n_frames} frames (0 to {end}), so there is no frame {start} "
                             "to start at.")
    return _Job(path, info, boxes, start, end, fps_true, source, out, _fps_warnings(fps_true))


def _session_job(args: argparse.Namespace) -> _Job:
    """The pass for `probe SESSION.json`."""
    from outline_tracker import video
    from outline_tracker.session import Session

    path = Path(args.source)
    given = [option for option, value in (("--rect", args.rect or None), ("--start", args.start), ("--end", args.end),
                                          ("--fps", args.fps), ("--out", args.out), ("--student", args.student))
             if value is not None]
    if given:
        raise ValueError(f"{' and '.join(given)} cannot be used with a session file: the rectangles, the frames, "
                         f"fps_true and the folder all come from {path.name}.")
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    session = Session.load(path)
    if not session.probes:
        raise ValueError(f"The session {path.name} has no probe rectangle, so there is nothing to measure.")
    boxes = _boxes((box.name, box.rect_px) for box in session.probes)
    fps_true = session.time.fps_true
    if fps_true is None:
        raise ValueError(f"The session {path.name} has no fps_true yet: set it in the session first.")
    clip = session.clip
    try:
        start, end = operator.index(clip.start), operator.index(clip.end)
    except TypeError:
        raise ValueError(f"The clip of the session {path.name} must start and end at whole frame numbers, not "
                         f"{clip.start!r} and {clip.end!r}.") from None
    found = session.locate_video(path.parent)
    return _Job(found, video.probe(found), boxes, start, end, float(fps_true),
                f"the session, {session.time.source}", path.parent, _fps_warnings(fps_true))


def _boxes(named: Iterable[tuple[str, Sequence[float]]]) -> dict[str, Sequence[float]]:
    """name -> rectangle, in the order given. Two rectangles with one name are an error."""
    boxes: dict[str, Sequence[float]] = {}
    for name, box in named:
        if name in boxes:
            raise ValueError(f"Two rectangles are named {name}: give each probe its own name.")
        boxes[name] = box
    return boxes


def _output_folder(args: argparse.Namespace, video_path: Path) -> Path:
    """`--out`, else the default run folder for `--student` (SPEC 8.1); never a folder of Tracker files."""
    from outline_tracker import run_folder

    if args.out is not None:
        out = Path(args.out)
    elif args.student is not None:
        out = run_folder.default_run_folder(video_path, args.student)
    else:
        raise ValueError("Say where probes.csv goes: add --student NAME (it is then written to "
                         f"{run_folder.default_run_folder(video_path, 'NAME')}, next to the video) or --out DIR "
                         "(a folder you choose).")
    if run_folder.folder_has_foreign_tables(out):
        raise ValueError(f"{out} holds other .csv or .txt files: it looks like a folder of Tracker files, and "
                         "probes.csv is not written into one. Choose another folder with --out DIR.")
    return out


def _fps_true(typed: float | None, video_path: Path) -> tuple[float, str]:
    """(fps_true in frames per s, where it came from): `--fps`, else the manifest (decision X9)."""
    from outline_tracker import geometry

    if typed is not None:
        if not (math.isfinite(typed) and typed > 0):
            raise ValueError("--fps must be a positive number of frames per second (fps_true, for example "
                             f"--fps 239.6), not {typed:g}.")
        return typed, "--fps"
    manifest = geometry.find_manifest(video_path, cwd=Path.cwd())
    fps = None if manifest is None else geometry.fps_from_manifest(video_path, manifest)
    if fps is None or not (math.isfinite(fps) and fps > 0):  # an empty cell reads as nan, a 0 as 0.0
        raise ValueError(UNKNOWN_FPS)
    return fps, f"manifest {manifest}"


def _fps_warnings(fps_true: float) -> list[str]:
    """The warning of SPEC 3.3 for an fps_true (frames per s) below 100, or nothing. It never blocks."""
    if fps_true >= SLOW_MOTION_FPS:
        return []
    return [f"fps_true = {fps_true:g} is below {SLOW_MOTION_FPS:g} frames per second. For a slow-motion video that "
            "usually means a re-timed copy: check the file with `outline-tracker check`."]


# --------------------------------------------------------------------------- run.log (SPEC 8.9)

def _log_entry(job: _Job, first: int, last: int, n_frames: int, wrote: str, notes: list[str]) -> list[str]:
    """The lines of one run.log entry: what was measured (frames `first` to `last`), and what was written."""
    info = job.info
    boxes = "; ".join(f"{name} " + ",".join(f"{float(corner):.10g}" for corner in box)
                      for name, box in job.boxes.items())
    when = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    return [
        f"==== probe, {when}, outline-tracker {__version__} ====",
        f"video: {os.path.abspath(job.video)} ({info.width} x {info.height} px; the file says {info.n_frames} "
        f"frames, {info.fps_container:.2f} fps)",
        f"fps_true: {job.fps_true!r} (source: {job.fps_source})",
        f"probes: {boxes} (name, then u0,v0,u1,v1 in image px)",
        f"frames: {first} to {last}, every frame ({n_frames} frame{'' if n_frames == 1 else 's'})",
        *(f"warning: {warning}" for warning in job.warnings),
        *(f"note: {note}" for note in notes),
        f"wrote: {wrote}",
    ]


def _append_run_log(folder: Path, lines: list[str]) -> Path:
    """Add one entry to `<folder>/run.log` (UTF-8, LF line ends), after a blank line when the log has
    entries already. What is there is kept as it is; the file is replaced in one step, as every output
    file is (`fileio.atomic_write`). Returns the path written."""
    from outline_tracker import fileio, schema

    path = folder / schema.RUN_LOG
    earlier = path.read_bytes() if path.is_file() else b""
    if earlier and not earlier.endswith(b"\n"):
        earlier += b"\n"
    entry = ("\n".join(lines) + "\n").encode("utf-8")
    return fileio.atomic_write(path, lambda tmp: tmp.write_bytes(earlier + (b"\n" if earlier else b"") + entry))
