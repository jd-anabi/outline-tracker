"""Export: every output file of a run folder except the overlay's frames, from session.json and
results.npz alone (SPEC 8.1-8.6, 8.9, 8.11).

`export_all` reads the two files, derives every world quantity with the session's current
calibration (`derive.derive_track`), computes the flags (`qc.compute_flags`) and writes
positions.csv, the Tracker-format folder `<model>/<id>.csv`, shapes.csv, radial.csv, outlines.npz,
README.txt, on request overlay.mp4, and a block of run.log. No video is decoded (but for the
overlay) and no model is loaded, so a new stick length, fps_true or head click costs one export.

What is exported:
- The tracks that the session lists and results.npz holds, in plain text order of their ids. A
  track that only results.npz holds is left out, with a warning.
- Per track the frames that results.npz holds, so a run that was stopped exports what it has.
- radial.csv and outlines.npz are always written. They hold the fine tracks (a track with records
  the fine runner made), or every track with the setting `shape_files_for_coarse`; with no such
  track, the header alone and `meta` alone (decision X12).

Every file is written through `fileio.atomic_write`; a file that another program holds open gets
its new data next to it (`<stem>.new<suffix>`) and a warning, and the next export that can write
the file removes that neighbor, which is then the older of the two. The rows, arrays and the
Tracker-format folder are in `export_tables`, the log's text in `export_log`.

Units and coordinates (SPEC 3): mm in the user's axes with y up, s, rad; px in Tracker's image
coordinates (pixel centers at +0.5); frames are video frame numbers. No Qt, no torch.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from outline_tracker import export_log, export_tables, provenance, schema, schema_docs
from outline_tracker.derive import DerivedTrack, derive_track
from outline_tracker.fileio import atomic_write, new_name
from outline_tracker.overlay import write_overlay
from outline_tracker.qc import compute_flags, summary_lines
from outline_tracker.results import ResultsStore, TrackArrays
from outline_tracker.session import Session

__all__ = ["ExportData", "ExportReport", "derive_all", "export_all"]


@dataclass(frozen=True)
class ExportReport:
    """What an export did. files: the paths written, in the order of writing; a path ending in
    `.new` (or `.new.<suffix>`) holds the data of a file that was locked. warnings: what the user
    should be told, one text each (they were also given to `log`)."""

    files: list[Path]
    warnings: list[str]


@dataclass(frozen=True)
class ExportData:
    """Everything an export writes, before it is text. All mappings are by track id, in the order
    of the files (plain text order).

    arrays: the tracks' records from results.npz, image px. derived: the same tracks in world units
    (mm in the user's axes with y up, s, rad; see `derive.DerivedTrack`). flags: each track's
    `flags` cells, one per row. shape_tracks: the ids that get rows in radial.csv and keys in
    outlines.npz. unlisted: ids that results.npz holds but the session does not list; they are
    not exported.
    """

    arrays: dict[str, TrackArrays]
    derived: dict[str, DerivedTrack]
    flags: dict[str, list[str]]
    shape_tracks: list[str]
    unlisted: list[str]


def derive_all(session: Session, store: ResultsStore) -> ExportData:
    """Every exported track in world units, with its flags: what `export_all` writes, in memory.

    session: its calibration, axes, fps_true (frames per s), circle and settings are used as they
    are now. store: the pixel-space records (image px, video frame numbers). Nothing is read from
    disk and no mask is measured again.

    Raises ValueError, with a message for the user, when the session has no fps_true or no scale,
    when `radial_step_deg` is not 5 (radial.csv has 72 fixed columns), and what `derive_track` and
    `compute_flags` raise for settings they cannot use.
    """
    step = session.processing.radial_step_deg
    if step != schema.RADIAL_STEP_DEG:
        raise ValueError(f"radial_step_deg = {step!r} cannot be exported: radial.csv has {schema.RADIAL_ANGLES} "
                         f"fixed columns, one every {schema.RADIAL_STEP_DEG} degrees. Set it back to "
                         f"{schema.RADIAL_STEP_DEG}.")
    if session.time.fps_true is None:
        raise ValueError("This session has no fps_true yet: take it from the manifest or the stopwatch clip, or "
                         "type it in, then export.")
    world_frame = session.world_frame()
    tracks = {track.id: track for track in session.tracks}
    ids = [track_id for track_id in store.track_ids if track_id in tracks]  # the store gives text order
    arrays = {track_id: store.arrays(track_id) for track_id in ids}
    derived = {track_id: derive_track(arrays[track_id], tracks[track_id], world_frame, session.time.fps_true,
                                      session.circle, session.processing) for track_id in ids}
    flags = compute_flags(derived, arrays, world_frame, session.processing)
    every = bool(session.processing.shape_files_for_coarse)
    shape_tracks = [track_id for track_id in ids if every or bool(np.any(arrays[track_id].mode == "fine"))]
    return ExportData(arrays, derived, flags, shape_tracks,
                      unlisted=[track_id for track_id in store.track_ids if track_id not in tracks])


def export_all(run_folder, overlay: bool = False, log: Callable[[str], object] = print) -> ExportReport:
    """Write every output file of a run folder from its session.json and results.npz.

    run_folder: the folder that holds the two files; the outputs go there too. overlay: also write
    overlay.mp4, for which the session's video is looked up (`Session.locate_video`); if it cannot
    be found or read, every other file is still written and a warning says that the overlay was
    skipped. Without `overlay` an overlay.mp4 that is there is left as it is. log: called with one
    line of text for each file removed from the Tracker-format folder and each warning.

    The files are in the units of their columns (SPEC 8): mm in the user's axes with y up, s, rad,
    px in image coordinates. Returns an `ExportReport`. Raises, before anything is written:
    FileNotFoundError when session.json or results.npz is missing; ValueError for what
    `derive_all` refuses, a model name that gives no folder name, two track ids that give one file
    name, and what `Session.load` and `ResultsStore.load` raise. PermissionError when a locked
    file's `.new` neighbor is locked as well.
    """
    run_folder = Path(run_folder)
    session = Session.load(run_folder / schema.SESSION_JSON)
    results = run_folder / schema.RESULTS_NPZ
    if not results.is_file():
        raise FileNotFoundError(f"There is no {schema.RESULTS_NPZ} in {run_folder}: nothing was tracked there "
                                "yet. Track first, then export.")
    data = derive_all(session, ResultsStore.load(results))
    model_folder = run_folder / _model_folder_name(session.processing.model)
    if model_folder.exists() and not model_folder.is_dir():
        raise ValueError(f"The Tracker-format files go into a folder named after the model, {model_folder.name}, "
                         f"but {model_folder} is a file. Move it away, or give the session another model name.")
    names = export_tables.tracker_file_names(list(data.derived))

    files: list[Path] = []
    warnings: list[str] = []

    def warn(text: str) -> None:
        warnings.append(text)
        log(f"WARNING: {text}")

    def write(name: str, write_fn: Callable[[Path], object]) -> None:
        target = run_folder / name
        written = atomic_write(target, write_fn)
        files.append(written)
        if written != target:
            warn(f"{name} is open in another program: the new data are in {written.name} next to it. Close that "
                 "program and export again.")
        elif new_name(target).is_file():  # what an export left while `name` was locked is older than `name` now
            with contextlib.suppress(OSError):
                new_name(target).unlink()
                log(f"{new_name(target).name}: removed (left by an export while {name} was locked).")

    def write_text(name: str, text: str) -> None:
        write(name, lambda tmp: tmp.write_bytes(text.encode(schema.CSV_ENCODING)))

    for track_id in data.unlisted:
        warn(f"{schema.RESULTS_NPZ} holds a track {track_id!r} that the session does not list: it is not exported.")
    write_text(schema.POSITIONS_CSV, schema.csv_text(
        schema.POSITIONS, export_tables.table_rows(schema.POSITIONS, data.derived, data.flags)))
    tracker_files, tracker_warnings = export_tables.write_tracker_folder(model_folder, data.derived, names, log)
    files += tracker_files
    for text in tracker_warnings:
        warn(text)
    write_text(schema.SHAPES_CSV, schema.csv_text(
        schema.SHAPES, export_tables.table_rows(schema.SHAPES, data.derived, data.flags)))
    write_text(schema.RADIAL_CSV, schema.csv_text(
        schema.RADIAL, export_tables.radial_rows(data.derived, data.shape_tracks)))
    outlines = export_tables.outline_arrays(data.derived, data.shape_tracks, session.processing.outline_points,
                                            provenance.tool_version())
    write(schema.OUTLINES_NPZ, lambda tmp: export_tables.save_outlines(tmp, outlines))
    write_text(schema.README_TXT, schema_docs.readme_text())

    outputs = [f"{path.relative_to(run_folder).as_posix()}: {path.stat().st_size:,} bytes" for path in files]
    if overlay:
        outputs.append(_write_overlay(run_folder, session, files, warn, log))
    else:
        outputs.append(f"{schema.OVERLAY_MP4}: not made by this export (it was not asked for)")
    outputs += [f"warning: {text}" for text in warnings]
    outputs.append(f"{schema.RUN_LOG}: this file")
    block = export_log.export_block(session, data.arrays, summary_lines(data.flags, data.derived), outputs,
                                    datetime.now().astimezone())
    log_path = run_folder / schema.RUN_LOG
    written = export_log.append_block(log_path, block)
    files.append(written)
    if written != log_path:
        warn(f"{schema.RUN_LOG} is open in another program: the log with this export is in {written.name} next to "
             "it. Close that program and export again.")
    return ExportReport(files, warnings)


def _model_folder_name(model) -> str:
    """The name of the Tracker-format folder: the session's model name, made safe for a file name.
    Raises ValueError for a name that gives no folder of its own."""
    name = export_tables.safe_name(model.strip()) if isinstance(model, str) else ""
    if not name.strip("._"):
        raise ValueError(f"The session's model name is {model!r}: it names the folder of the Tracker-format files "
                         "(for example edgetam or sam2) and must hold a letter or a digit.")
    return name


def _write_overlay(run_folder: Path, session: Session, files: list[Path], warn: Callable[[str], None],
                   log: Callable[[str], object]) -> str:
    """Write overlay.mp4 if the session's video can be found and read; returns the line about it
    for run.log's list of outputs. The written path is added to `files`."""
    target = run_folder / schema.OVERLAY_MP4
    try:
        video = session.locate_video(run_folder)
        written, count = write_overlay(run_folder, video, target, log=log)
    except (OSError, ValueError) as err:  # the video is gone or is another file, or it cannot be read
        warn(f"{schema.OVERLAY_MP4} was skipped: {err} Every other file was written.")
        return f"{schema.OVERLAY_MP4}: skipped ({err})"
    files.append(written)
    if written != target:
        warn(f"{schema.OVERLAY_MP4} is open in another program: the new video is in {written.name} next to it. "
             "Close that program and export again.")
    return f"{written.relative_to(run_folder).as_posix()}: {written.stat().st_size:,} bytes, {count} frames"
