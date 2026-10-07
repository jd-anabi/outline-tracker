"""The tables of an export: the rows of positions.csv, shapes.csv and radial.csv, the arrays of
outlines.npz and the Tracker-format folder `<model>/<id>.csv` (SPEC 8.2-8.6).

Everything here is made from derived tracks (`derive.DerivedTrack`) and the flags cells of `qc`;
the column lists, number formats and file texts are `schema`'s. `export.export_all` decides what is
written where.

Rules of the rows:
- One row per frame of a track as it is given, tracks in the order given (plain text order of
  their ids), frames ascending.
- On a lost frame every measured number is an empty cell, the area too (an empty mask measures
  nothing); `frame` and `t_s` stay, the counts `visible`, `n_components` and `shape_ok` are 0.
- positions.csv and shapes.csv have every frame of the clip's grid from a track's first to its
  last (SPEC 8.2). A track whose records leave one out is given a lost row there first
  (`fill_gaps`); the Tracker-format files, radial.csv and outlines.npz hold the frames with a
  record only.

The Tracker-format folder holds nothing but `<id>.csv`: students' own loaders read every .csv and
.txt file in it as a track. So a file is written under `<id>.csv.tmp` by last week's writer
(`tracker_io.write_tracker_file`, its bytes unchanged) and renamed; if `<id>.csv` stays locked by
another program the new data stay next to it as `<id>.csv.new` (`fileio.replace_with_retry`: the
waits and the rule of every other output file, under these two names); what an earlier export left
under those two names is removed, and so are the files of tracks that no longer exist.

Units and coordinates: as the columns say (SPEC 3): mm in the user's axes with y up, s, rad
counterclockwise from +x, px in Tracker's image coordinates (pixel centers at +0.5). Frames are
video frame numbers. No Qt, no torch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

from outline_tracker import fileio, schema, tracker_io
from outline_tracker.derive import DerivedTrack

CONVENTIONS = (
    "xy_mm: mm in the user's axes, y up. xieta_mm: mm in the body frame, origin at the core centroid, xi toward "
    "the head, eta 90 degrees counterclockwise from xi. Every outline has n_points points equally spaced along "
    "it, counterclockwise in the user's axes, starting at the head point; rows of frames where the track is not "
    "visible are NaN. frames: video frame numbers, t_s = frame / fps_true."
)
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")  # last week's rule for a track's file name
_TMP, _NEW = ".tmp", ".new"                # added to `<id>.csv`: neither name ends in .csv or .txt


def safe_name(text: str) -> str:
    """A track id or model name as a file name, by last week's rule: every run of characters other
    than letters, digits, `_`, `.` and `-` becomes one `_`. Text only, no units."""
    return _UNSAFE.sub("_", text)


def tracker_file_names(track_ids: Sequence[str]) -> dict[str, str]:
    """The file name `<safe id>.csv` of every track in the Tracker-format folder, by track id.

    Raises ValueError when two ids give one file name (also when they differ only in case, which
    Windows and macOS do not tell apart): one of the tracks would overwrite the other. No units.
    """
    names: dict[str, str] = {}
    taken: dict[str, str] = {}
    for track_id in track_ids:
        name = f"{safe_name(track_id)}.csv"
        other = taken.setdefault(name.casefold(), track_id)
        if other != track_id:
            raise ValueError(f"The tracks {other!r} and {track_id!r} would both be written to {name}: file names "
                             "keep only letters, digits, '_', '.' and '-'. Rename one of the tracks.")
        names[track_id] = name
    return names


def fill_gaps(derived: DerivedTrack, cells: Sequence[str], grid_start: int, grid_step: int,
              fps_true: float) -> tuple[DerivedTrack, list[str]]:
    """A track's rows without a gap (SPEC 8.2): the track and its `flags` cells, with a lost row
    added for every frame of the clip's grid that lies between the track's first and last frame
    and is not among its frames. Tracking leaves no such frame; a results.npz made by hand or by
    an older version may.

    derived: the track in world units (mm in the user's axes with y up, s, rad; px in image
    coordinates), one row per stored frame. cells: its `flags` cells, one per row. grid_start,
    grid_step: the clip's start frame and step, video frame numbers: the grid is every frame that
    is a whole number of steps from the start. fps_true: frames per s, for the new rows' `t_s`.

    An added row is what a lost frame is: `visible`, the counts and `shape_ok` 0, every measured
    number NaN, `t_s` = frame / fps_true, the flags LOST and, as on every row of a track without a
    head click, HEADGUESS; its `mode` is that of the row before it. A row whose frame is not on
    the grid stays. Returns (track, cells); the ones given when no frame is missing.
    """
    frames = derived.frame
    if len(frames) < 2:
        return derived, list(cells)
    first, last = int(frames[0]), int(frames[-1])
    on_grid = np.arange(first + (grid_start - first) % grid_step, last, grid_step)
    missing = np.setdiff1d(on_grid, frames)
    if not len(missing):
        return derived, list(cells)
    at = np.searchsorted(frames, missing)  # the row each missing frame comes before; never row 0
    filled = {}
    for field in dataclasses.fields(derived):
        if field.name == "track_id":
            continue
        column = getattr(derived, field.name)
        if field.name == "frame":
            new = missing
        elif field.name == "t_s":
            new = missing / float(fps_true)
        elif field.name in ("mode", "headguess"):  # one per track: as on the row before
            new = column[at - 1]
        elif column.dtype.kind == "f":
            new = np.full((len(missing), *column.shape[1:]), np.nan)
        else:  # `visible`, the counts, `shape_ok`, `orient`
            new = np.zeros((len(missing), *column.shape[1:]), column.dtype)
        filled[field.name] = np.insert(column, at, new, axis=0)
    lost = schema.join_flags(["LOST", "HEADGUESS"] if derived.headguess[0] else ["LOST"])
    return dataclasses.replace(derived, **filled), np.insert(np.array(cells, dtype=object), at, lost).tolist()


def table_rows(columns: Sequence[schema.Column], derived_by_track: Mapping[str, DerivedTrack],
               flags_by_track: Mapping[str, Sequence[str]]) -> Iterator[list[object]]:
    """The rows of positions.csv or shapes.csv (`columns` = `schema.POSITIONS` or `schema.SHAPES`),
    each a list in column order for `schema.csv_text`.

    derived_by_track: the tracks in the order of the file, in world units (mm, y up; s; rad; px in
    image coordinates), without a gap (`fill_gaps`). flags_by_track: each track's `flags` cells,
    one per row (`qc.compute_flags`, `fill_gaps`). On a lost row every float except `t_s` is NaN,
    which the file shows as an empty cell.
    """
    for track_id, derived in derived_by_track.items():
        lost = derived.visible == 0
        values = {}
        for column in columns:
            if column.name in ("track_id", "flags"):
                continue
            value = getattr(derived, column.name)
            if column.dtype == "float" and column.name != "t_s":
                value = np.where(lost, np.nan, value)
            values[column.name] = value.tolist()
        cells = flags_by_track[track_id]
        for row in range(len(derived.frame)):
            yield [track_id if column.name == "track_id" else cells[row] if column.name == "flags"
                   else values[column.name][row] for column in columns]


def radial_rows(derived_by_track: Mapping[str, DerivedTrack], track_ids: Sequence[str]) -> Iterator[list[object]]:
    """The rows of radial.csv for the named tracks: track id, frame, t_s (s), then the 72 radii in
    mm at 0, 5, ..., 355 degrees counterclockwise from the head direction (`schema.RADIAL`); NaN,
    an empty cell, on a frame where the track is not visible or a ray misses the outline."""
    for track_id in track_ids:
        derived = derived_by_track[track_id]
        radii = np.where((derived.visible == 0)[:, None], np.nan, derived.radial_mm).tolist()
        for frame, t_s, row in zip(derived.frame.tolist(), derived.t_s.tolist(), radii):
            yield [track_id, frame, t_s, *row]


def outline_arrays(derived_by_track: Mapping[str, DerivedTrack], track_ids: Sequence[str], n_points: int,
                   tool_version: str) -> dict[str, np.ndarray]:
    """The arrays of outlines.npz (SPEC 8.6) by key: for each named track `<id>__frames` (int32
    [n], video frame numbers), `<id>__xy_mm` and `<id>__xieta_mm` (float32 [n, N, 2]: the outline
    in world mm, y up, and in the body frame, mm), and `meta`, a JSON text with the fields of
    `schema.OUTLINES_META_FIELDS`. With no track the file holds `meta` alone."""
    arrays: dict[str, np.ndarray] = {}
    for track_id in track_ids:
        derived = derived_by_track[track_id]
        arrays[schema.npz_key(track_id, "frames")] = derived.frame.astype(np.int32)
        arrays[schema.npz_key(track_id, "xy_mm")] = derived.outline_xy_mm.astype(np.float32)
        arrays[schema.npz_key(track_id, "xieta_mm")] = derived.outline_xieta_mm.astype(np.float32)
    meta = {"n_points": int(n_points), "tracks": list(track_ids), "conventions": CONVENTIONS,
            "tool_version": tool_version}
    arrays[schema.OUTLINES_META_KEY] = np.array(json.dumps(meta))
    return arrays


def save_outlines(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write the arrays of `outline_arrays` to `path` as a compressed npz, without pickled objects."""
    with open(path, "wb") as f:  # an open file: given a path, numpy adds ".npz" to its name
        np.savez_compressed(f, **arrays)


def write_tracker_folder(folder: Path, derived_by_track: Mapping[str, DerivedTrack], names: Mapping[str, str],
                         log: Callable[[str], object]) -> tuple[list[Path], list[str]]:
    """Write `<folder>/<id>.csv` for every track in last week's Tracker layout (SPEC 8.3) and clear
    the folder of what is not a current track's file.

    folder: the Tracker-format folder, named after the model; created if needed. derived_by_track:
    the tracks as stored, one row per tracked frame (not filled by `fill_gaps`), with t_s in s,
    x_mm and y_mm in mm in the user's axes (y up), u_px and v_px in image px; a lost frame keeps
    its row with those four empty, as in positions.csv. names: each track's file name
    (`tracker_file_names`). log: told, one line each, which files were removed.

    Returns (the files written, warnings). A file is `<id>.csv`, or `<id>.csv.new` when `<id>.csv`
    stayed locked by another program, which a warning then says. Raises PermissionError when the
    `.new` file is locked as well.
    """
    folder.mkdir(parents=True, exist_ok=True)
    warnings = []
    current = set(names.values())
    for entry in sorted(folder.iterdir()):
        lower = entry.name.lower()
        if entry.name.startswith(".") or not entry.is_file():
            continue
        if lower.endswith((".csv" + _TMP, ".csv" + _NEW)):
            reason = "left by an earlier export"
        elif lower.endswith(".csv") and entry.name not in current:
            reason = "there is no such track any more"
        else:
            continue
        try:
            entry.unlink()
            log(f"{folder.name}/{entry.name}: removed ({reason}).")
        except OSError as err:
            warnings.append(f"{folder.name}/{entry.name} could not be removed ({err}); {reason}. Students' loaders "
                            "read every .csv file of that folder as a track: remove it by hand.")
    written = []
    for track_id, derived in derived_by_track.items():
        target = folder / names[track_id]
        tmp = target.with_name(target.name + _TMP)
        try:
            tracker_io.write_tracker_file(tmp, track_id, derived.frame, derived.t_s, derived.x_mm, derived.y_mm,
                                          derived.u_px, derived.v_px)
            path = fileio.replace_with_retry(tmp, target, target.with_name(target.name + _NEW))
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        written.append(path)
        if path != target:
            warnings.append(f"{folder.name}/{target.name} is open in another program: the new data are in "
                            f"{path.name} next to it. Close that program and export again.")
    return written, warnings

