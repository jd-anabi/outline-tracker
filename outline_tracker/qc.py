"""Quality flags per track and frame (SPEC 9), and their summary for run.log (SPEC 8.9).

`compute_flags` gives every row of every track its `flags` cell: the codes that hold on that
frame, in the order of SPEC 9, joined by ';' (`schema.join_flags`); an empty text when nothing is
wrong. The same text goes to positions.csv and shapes.csv. `summary_lines` counts the cells per
track and code for the log. The console CHECK lines of last week are not made here (decision X22).

The nine codes and where each comes from:
- LOST: the mask is empty (`visible` of the records). A lost row carries LOST and nothing else,
  except HEADGUESS: it has no position, area, outline or heading that another code could judge.
- JUMP: last week's rule (`_flags` of shrimp.segment). The speed between two consecutive visible
  frames of the track, the distance between their centroids in mm over the real time between
  them in s (whatever the frame step, and across lost frames), is strictly above `jump_mm_s`.
  The later of the two frames gets the flag.
- SIZE: last week's rule. The mask's area in px is strictly above 2 times or strictly below 0.5
  times the median area over the track's visible frames.
- CONTACT: the outline is within max(3 px, 2 grid cells) of another track's outline on the same
  video frame (`qc_contact`; decision X6). Both tracks get it.
- EDGE: the mask touches the border of the image the model saw (`edge` of the records).
- MULTI: the mask has more than one piece and the second-largest has at least 10% of the pixels
  of the largest (`n_components`, `second_fraction` of the records).
- LOWRES: `shape_ok` of the derived track is 0 (SPEC 7.8).
- ORIENT, HEADGUESS: as the derived track says (SPEC 7.3; `derive_heading`). HEADGUESS is on
  every row of a track without a head click, lost rows included.

JUMP, SIZE and HEADGUESS are decided per track id: a later piece of an animal (A2) is a track of
its own, with its own median and no step from where A ended.

Units and frames: positions and times are read from the derived tracks, in mm in the world frame
(y up) and in s; areas, grid cells and outlines from the records, in image px (u to the right,
v down, pixel centers at +0.5). Frames are video frame numbers.

The comparison of two tracks' outlines is in `qc_contact`; its `polygon_distance` (the least
distance between two outlines, in px for stored ones) is offered here too.

No Qt, no torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from outline_tracker.derive import DerivedTrack
from outline_tracker.geometry import WorldFrame
from outline_tracker.qc_contact import contact_marks, polygon_distance
from outline_tracker.results import TrackArrays
from outline_tracker.schema import FLAG_SEPARATOR, FLAGS, join_flags
from outline_tracker.session import Processing

__all__ = ["compute_flags", "polygon_distance", "summary_lines"]

SIZE_LOW, SIZE_HIGH = 0.5, 2.0   # SIZE: area outside this range times the track's median (SPEC 9, as last week)
MULTI_MIN = 0.1                  # MULTI: second piece at least this fraction of the largest (SPEC 9)


def compute_flags(derived_by_track: Mapping[str, DerivedTrack], arrays_by_track: Mapping[str, TrackArrays],
                  world_frame: WorldFrame, processing: Processing) -> dict[str, list[str]]:
    """The `flags` cell of every row of every track (SPEC 9; the module's text gives the rules).

    derived_by_track: every track in world units (`derive.derive_track`): `x_mm`, `y_mm` in mm
    (y up) and `t_s` in s give the speed; `shape_ok`, `orient`, `headguess` are taken as they are.
    arrays_by_track: the same tracks' records from the results store, in image px: `visible`,
    `area_px`, `edge`, `n_components`, `second_fraction`, and for CONTACT `frames`, `cell_px` and
    `outline_px`. world_frame: the calibration the derived tracks were made with. Nothing is read
    from it: speeds use the derived mm, CONTACT is measured in px. processing: `jump_mm_s`, the
    speed limit of JUMP in mm per s.

    Returns {track id: [text, ...]}, the tracks in the order of `derived_by_track`, one text per
    row in frame order: the codes of that frame in SPEC 9 order joined by ';', "" for none.
    Raises ValueError for a `jump_mm_s` that is not a positive number, for a track that is in
    only one of the two mappings, and for a derived track and records of different frames.
    """
    limit = float(processing.jump_mm_s)
    if not (math.isfinite(limit) and limit > 0):
        raise ValueError(f"jump_mm_s = {processing.jump_mm_s!r}: the speed limit of the JUMP flag must be a "
                         f"positive number of mm per s.")
    _check_same_tracks(derived_by_track, arrays_by_track)
    contact = contact_marks(arrays_by_track)
    flags = {}
    for track_id, derived in derived_by_track.items():
        arrays = arrays_by_track[track_id]
        seen = arrays.visible.astype(bool)
        marks = {
            "LOST": ~seen,
            "JUMP": _jumps(derived, seen, limit),
            "SIZE": _size_changes(arrays.area_px, seen),
            "CONTACT": contact[track_id],
            "EDGE": seen & arrays.edge,
            "MULTI": seen & (arrays.n_components > 1) & (arrays.second_fraction >= MULTI_MIN),
            "LOWRES": seen & (derived.shape_ok == 0),
            "ORIENT": seen & derived.orient,
            "HEADGUESS": derived.headguess.astype(bool),
        }
        flags[track_id] = [join_flags(code for code in FLAGS if marks[code][row]) for row in range(len(seen))]
    return flags


def summary_lines(flags_by_track: Mapping[str, Sequence[str]],
                  derived_by_track: Mapping[str, DerivedTrack]) -> list[str]:
    """The QC summary of run.log (SPEC 8.9), in the wording of last week's CHECK lines.

    flags_by_track: what `compute_flags` returned. derived_by_track: the same tracks; their
    `frame` (video frame numbers) and `t_s` (frame / fps_true, s) name the first occurrence.

    Returns plain ASCII lines without line ends (given ASCII track ids), the tracks in the order
    of `flags_by_track`, and for each track one line per code that occurs, in SPEC 9 order:
    "A: JUMP in 3 of 600 frames (first at frame 120, t = 0.500 s)". A track without any flag
    gives "A: no flags in 600 frames". Raises ValueError for a track without a derived track and
    for a number of cells other than the track's number of frames.
    """
    lines = []
    for track_id, cells in flags_by_track.items():
        if track_id not in derived_by_track:
            raise ValueError(f"The QC summary has flags for track {track_id}, but no derived track of that name.")
        derived = derived_by_track[track_id]
        count = len(derived.frame)
        if len(cells) != count:
            raise ValueError(f"Track {track_id} has {count} frames, but {len(cells)} flags cells were given.")
        held = [cell.split(FLAG_SEPARATOR) for cell in cells]
        before = len(lines)
        for code in FLAGS:
            rows = [row for row, codes in enumerate(held) if code in codes]
            if rows:
                lines.append(f"{track_id}: {code} in {len(rows)} of {count} frames (first at frame "
                             f"{int(derived.frame[rows[0]])}, t = {float(derived.t_s[rows[0]]):.3f} s)")
        if len(lines) == before:
            lines.append(f"{track_id}: no flags in {count} frames")
    return lines


def _check_same_tracks(derived_by_track, arrays_by_track) -> None:
    """Raise ValueError unless both mappings hold the same tracks on the same video frames."""
    alone = sorted(set(derived_by_track) ^ set(arrays_by_track))
    if alone:
        raise ValueError(f"Flags need the derived track and the records of every track; only one of the two was "
                         f"given for: {', '.join(alone)}.")
    for track_id, derived in derived_by_track.items():
        if not np.array_equal(derived.frame, arrays_by_track[track_id].frames):
            raise ValueError(f"Track {track_id}: the derived track and the records are not of the same frames.")


def _jumps(derived: DerivedTrack, seen: np.ndarray, limit: float) -> np.ndarray:
    """[n] booleans: the later frame of every pair of consecutive visible frames between which
    the centroid moved faster than `limit` mm per s (world mm over the real time difference)."""
    jump = np.zeros(len(seen), bool)
    ok = np.flatnonzero(seen)
    if len(ok) >= 2:
        speed = np.hypot(np.diff(derived.x_mm[ok]), np.diff(derived.y_mm[ok])) / np.diff(derived.t_s[ok])
        jump[ok[1:][speed > limit]] = True
    return jump


def _size_changes(area_px: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """[n] booleans: the visible frames whose area (px) is strictly outside SIZE_LOW to SIZE_HIGH
    times the median area of the track's visible frames."""
    odd = np.zeros(len(seen), bool)
    ok = np.flatnonzero(seen)
    if len(ok) >= 2:
        area = area_px[ok].astype(float)
        median = np.median(area)
        odd[ok[(area > SIZE_HIGH * median) | (area < SIZE_LOW * median)]] = True
    return odd
