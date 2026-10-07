"""Edits of a session and its results without Qt (SPEC 5, 6.6): what the GUI's panels call.

- The object table: `add_object`, `remove_object`; ids and colors are in tracking_ids.py
  (`next_track_id`).
- Clicks: `add_prompt`, `undo_prompt`, and the head click, `set_head`.
- What "Track" would start: `pending_runs`.
- The three corrections: `retrack_from` ("Re-track from here"), `end_track` ("End track here") and
  `new_piece` ("Continue as new track") are written in outline_tracker/tracking_corrections.py, which
  builds on this module: `Edit`, and the helpers that find a track, move a frame onto the grid and
  read and save the results, are here. Each correction adds an entry to `session.corrections`
  (SPEC 8.10).
- The flags table of a run folder: `flags_table` (outline_tracker/tracking_flags.py).

Every name of `__all__` is importable from here, the three corrections included, and from
outline_tracker.tracking too.

Who writes what. These functions change the `Session` they are given and never save it: saving
session.json is the caller's part (the GUI saves on its own thread). `store` is the `ResultsStore`
of the run folder as results.npz holds it now. A function that removes records (`retrack_from`,
`end_track`, `remove_object`) saves the results to results.npz at once, because a tracking job
reads them from that file and writes them itself. To these three the GUI passes None for `store`:
they then read results.npz as it is now, so that nothing a job tracked since is lost (a store
loaded before that job would replace it), and `retrack_from` and `end_track` hand back the store
that was saved (`Edit.store`), the one to go on with. When results.npz stays locked by another
program, they raise RuntimeError and change nothing: no correction is recorded without its
results. They do the same while the run folder holds a results.new.npz, where a job leaves its
results when it cannot write results.npz: that file is never replaced or removed here, the user
renames or removes it first. `session.complete` follows the results
(`tracking_plan.partial_tracks`).

The sequence of a correction, "Re-track from here" for one, is in tracking_corrections.py. No edit
makes a gap inside a track (SPEC 8.2): a track with results is clicked on, and re-tracked from, the
first grid frame after its last record at the latest (`tracking_plan.check_no_gap`).

Units and coordinates (SPEC 3.1, 3.4): frames are video frame numbers; a start, re-track or
new-piece frame that is not on the clip's grid is moved forward to the next grid frame (X20), and
the returned `Edit` says so. Points are (u, v) in px of the full frame, Tracker's convention: u to
the right, v downward, the pixel in column c and row r with its center at (c + 0.5, r + 0.5).
No Qt, no torch.
"""

from __future__ import annotations

import contextlib
import copy
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from outline_tracker.fileio import new_name
from outline_tracker.geometry import grid_frames, snap_to_grid
from outline_tracker.measure import MODES
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_NPZ
from outline_tracker.session import Prompt, Session, Track
from outline_tracker.tracking_flags import flags_table
from outline_tracker.tracking_ids import TRACK_COLORS, next_track_id, track_color
from outline_tracker.tracking_plan import RunPlan, check_no_gap, partial_tracks, pending_from, plan_runs

if TYPE_CHECKING:  # for tools that read this file: the names that `__getattr__` below gives at run time
    from outline_tracker.tracking_corrections import end_track, new_piece, retrack_from

__all__ = ["TRACK_COLORS", "Edit", "add_object", "add_prompt", "end_track", "flags_table", "new_piece",
           "next_track_id", "pending_runs", "remove_object", "retrack_from", "set_head", "undo_prompt"]


@dataclass(frozen=True)
class Edit:
    """What an edit did, for the caller to show.

    track_ids: the tracks it changed; for `new_piece`, the id of the piece it made. frame: the
    video frame number it was made on, a frame of the clip's grid except for `end_track`. moved:
    the frame asked for was not on the grid, and the edit was made on `frame` instead (X20). note:
    one sentence for the user that says so, "" when the frame was not moved. results_file: the
    file the results were saved to, results.npz of the run folder (an edit whose save does not
    land there raises); None when the edit did not change the results. store: from `retrack_from`
    and `end_track`, the results as results.npz holds them after the edit, for the caller to go on
    with: the store that was given, or the one read from the file; None from the other edits.
    """

    track_ids: tuple[str, ...]
    frame: int
    moved: bool = False
    note: str = ""
    results_file: Path | None = None
    store: ResultsStore | None = None


# --------------------------------------------------------------------------- the results of the run folder

def _refuse_results_left_aside(run_folder) -> None:
    """Raise RuntimeError when the run folder holds a results.new.npz: results that were saved
    there because results.npz was locked (`fileio.atomic_write`), by a job as a rule. Then
    results.npz may lack what that job tracked, and a save of an edit could replace that file,
    so the user settles it first. Nothing is read or written here."""
    target = Path(run_folder) / RESULTS_NPZ
    aside = new_name(target)
    if aside.exists():
        raise RuntimeError(
            f"{aside.name} in {target.parent} holds results that were saved while {target.name} was open in another "
            f"program, so the change was not made, and nothing was changed. Close that program. If {aside.name} is "
            f"the newer of the two files, it holds results that are not in {target.name}: rename it to {target.name}, "
            "replacing that file. If it is the older one, remove it. Then try again.")


def _results(store: ResultsStore | None, run_folder) -> ResultsStore:
    """The results an edit starts from: `store`, or for None what results.npz of the run folder
    holds now (no records when there is no such file yet). Raises RuntimeError while the folder
    holds a results.new.npz (`_refuse_results_left_aside`), also for an edit that saves nothing:
    the records it is about may be in that file."""
    _refuse_results_left_aside(run_folder)
    if store is not None:
        return store
    path = Path(run_folder) / RESULTS_NPZ
    return ResultsStore.load(path) if path.is_file() else ResultsStore()


def _save(results: ResultsStore, given: bool, run_folder, remove: Callable[[ResultsStore], None]) -> Path:
    """Take records out of the results and save them to results.npz of the run folder, which is
    returned. `remove` takes them out of the store it is called with. The file itself must take
    them, since a job reads it: when it stays locked by another program, the results.new.npz
    written instead is removed and RuntimeError is raised. A results.new.npz that is there already
    is not this save's to replace or remove: RuntimeError before anything is written. `given`:
    `results` is the caller's store; it is changed only when the save has landed (a copy is saved
    first)."""
    target = Path(run_folder) / RESULTS_NPZ
    _refuse_results_left_aside(run_folder)
    work = copy.deepcopy(results) if given else results
    remove(work)
    try:
        written = work.save(target)
    except PermissionError as err:  # results.new.npz is locked as well, or the folder takes no file
        raise RuntimeError(f"The change could not be saved, and nothing was changed. {err}") from err
    if written != target:
        with contextlib.suppress(OSError):
            written.unlink()
        raise RuntimeError(f"{target.name} in {target.parent} is open in another program, so the change could not "
                           "be saved, and nothing was changed. Close that program, then try again.")
    if given:
        remove(results)
    return target


# --------------------------------------------------------------------------- the object table

def _track(session: Session, track_id: str) -> Track:
    for track in session.tracks:
        if track.id == track_id:
            return track
    known = ", ".join(track.id for track in session.tracks) or "none"
    raise ValueError(f"The session has no track {track_id} (its tracks: {known}).")


def add_object(session: Session, mode: str = "coarse", fine_window_px: int | None = None) -> Track:
    """Add an object to the session's table and return its track: the next id (`next_track_id`),
    that id's color (`track_color`), no clicks yet. `mode` is "coarse" or "fine"; `fine_window_px`
    is the side of a fine object's window in px, None for automatic (SPEC 6.3). Raises ValueError
    for another mode."""
    if mode not in MODES:
        raise ValueError(f"The mode of an object must be 'coarse' or 'fine', not {mode!r}.")
    track_id = next_track_id(session)
    track = Track(id=track_id, color=track_color(track_id), mode=mode, fine_window_px=fine_window_px)
    session.tracks.append(track)
    return track


def remove_object(session: Session, store: ResultsStore | None, track_id: str, run_folder) -> Path | None:
    """Remove an object: its track from the session and its records from the results, so that they
    cannot be taken for another object's. `store`: the results, changed here too, or None for
    results.npz of `run_folder` as it is now (the module's text); load that file afterwards for
    the store to go on with. Returns the file the results were saved to (results.npz), None when
    the object had no records. `session.complete` follows what is left. Raises ValueError for an
    unknown track, and RuntimeError, changing nothing, when results.npz stays locked or the run
    folder holds a results.new.npz (the module's text). No units."""
    track = _track(session, track_id)
    results = _results(store, run_folder)
    reach = _reach(session, results)
    results_file = None
    if track_id in results.track_ids:
        first = int(results.arrays(track_id).frames[0])
        results_file = _save(results, store is not None, run_folder, lambda kept: kept.replace_from(track_id, first))
    session.tracks = [other for other in session.tracks if other is not track]
    session.complete = not partial_tracks(session, results, reach)
    return results_file


def _reach(session: Session, store: ResultsStore) -> int | None:
    """The last video frame tracking can reach, as far as an edit can know it. A video may end
    before its clip does, which only the job that tracked it saw: while the session says its
    results are complete, they reach as far as tracking can, so it is the last frame any track
    has. Else None, the clip's end: after an edit of partial results of such a video, `complete`
    stays False until a job has run. For `partial_tracks`; read before the edit changes anything."""
    if not session.complete or not store.track_ids:
        return None
    return max(int(store.arrays(track_id).frames[-1]) for track_id in store.track_ids)


# --------------------------------------------------------------------------- clicks

def _on_grid(session: Session, frame) -> tuple[int, bool, str]:
    """A frame moved onto the clip's grid (`geometry.snap_to_grid`, forward): (grid frame, whether
    it was moved, the note for the user or "")."""
    clip = session.clip
    snapped, moved = snap_to_grid(frame, clip.start, clip.step, clip.end)
    if not moved:
        return snapped, False, ""
    grid = grid_frames(clip.start, clip.end, clip.step)
    return snapped, True, (f"Frame {frame} is not a frame of the clip (frames {grid[0]} to {grid[-1]}, every "
                           f"{grid.step}): frame {snapped} is used instead.")


def _start_on(track: Track, frame: int) -> None:
    """Make `frame` the track's start frame. A head click belonged to the old start frame (SPEC
    7.3) and goes with it."""
    if frame != track.start_frame:
        track.start_frame, track.head_px = frame, None


def add_prompt(session: Session, store: ResultsStore, track_id: str, frame, point_px, label: int,
               frame_hash: str | None, decoder: str | None) -> Edit:
    """Add one click to a track (SPEC 5).

    frame: the video frame the click was placed on; off the clip's grid it is moved forward to
    the grid (the returned `Edit` says so). point_px: (u, v) in px of the full frame. label: 1
    for a positive click, 0 for a negative one. frame_hash, decoder: `video.frame_hash` of that
    decoded frame and `video.decoder_tag()`, from the caller, which shows the frame (SPEC 3.5);
    both are stored with the click, unless the frame was moved: then they are of another frame.

    Clicks on one frame share a `Prompt`. The first click of a track sets its start frame. A
    click on a frame the track has results for waits for "Re-track from here" (`retrack_from`);
    any other click makes the track pending from that frame, and `session.complete` False.

    Raises ValueError, changing nothing, for an unknown track, a point that is not two finite
    numbers, a label other than 1 or 0, a frame after the end of a track that was ended, a second
    frame of clicks while the track has clicks that are not tracked yet (its run starts on the
    frame of those, `plan_runs`), and a frame later than the first grid frame after the track's
    last record: a run from there would leave a gap (`tracking_plan.check_no_gap`).
    """
    track = _track(session, track_id)
    point = _point(point_px)
    if label not in (0, 1) or isinstance(label, bool):
        raise ValueError(f"A click's label must be 1 (positive) or 0 (negative), not {label!r}.")
    frame, moved, note = _on_grid(session, frame)
    if track.ended_at is not None and frame > track.ended_at:
        raise ValueError(f"Track {track_id} was ended at frame {track.ended_at}: it takes no click on frame {frame}. "
                         "Continue the animal as a new track from there.")
    waits = pending_from(track, store)
    if waits is not None and waits != frame and not _tracked(store, track_id, frame):
        raise ValueError(f"Track {track_id} has clicks on frame {waits} that are not tracked yet, and its next run "
                         f"starts there. Click on that frame, undo those clicks, or track first.")
    check_no_gap(session, track, store, frame)
    if moved:
        frame_hash = decoder = None
    if not track.prompts:
        _start_on(track, frame)
    last = track.prompts[-1] if track.prompts else None
    if last is None or (last.frame, last.frame_hash, last.decoder) != (frame, frame_hash, decoder):
        last = Prompt(frame=frame, frame_hash=frame_hash, decoder=decoder)
        track.prompts.append(last)
    last.points_px.append(point)
    last.labels.append(int(label))
    if pending_from(track, store) is not None:
        session.complete = False
    return Edit((track_id,), frame, moved, note)


def _point(point_px) -> list[float]:
    """A click's point as the session stores it: [u, v], px of the full frame. Raises ValueError
    for anything but two finite numbers."""
    try:
        u, v = point_px
    except (TypeError, ValueError):
        u = v = None
    if not all(isinstance(value, numbers.Real) and math.isfinite(value) for value in (u, v)):
        raise ValueError(f"A click's point must be two numbers, (u, v) in px of the full frame, not {point_px!r}.")
    return [float(u), float(v)]


def _tracked(store: ResultsStore, track_id: str, frame: int) -> bool:
    return track_id in store.track_ids and frame in store.arrays(track_id).frames


def undo_prompt(session: Session, store: ResultsStore, track_id: str) -> bool:
    """Remove the newest click of a track (SPEC 5: undo). Returns False when the track has no
    click. The start frame stays; `session.complete` follows what is left to track. The results
    are not touched: when the clicks of "Re-track from here" on frame k are undone, the track has
    no results from k on and nothing waits; its next click is on k or earlier (`check_no_gap`).
    Raises ValueError for an unknown track. No units."""
    track = _track(session, track_id)
    if not track.prompts:
        return False
    reach = _reach(session, store)
    last = track.prompts[-1]
    last.points_px.pop()
    last.labels.pop()
    if not last.points_px:
        track.prompts.pop()
    session.complete = not partial_tracks(session, store, reach)
    return True


def set_head(session: Session, track_id: str, frame: int, point_px) -> None:
    """Store the head click of a track: `point_px` = (u, v) in px of the full frame, placed on
    video frame `frame`; None removes it. The heading takes its side from this click on the
    track's start frame (SPEC 7.3), so it must be placed there. Raises ValueError for an unknown
    track, a track without clicks (it has no start frame yet) and another frame."""
    track = _track(session, track_id)
    if point_px is None:
        track.head_px = None
        return
    if not track.prompts:
        raise ValueError(f"Track {track_id} has no clicks yet. Click the object first: the head click belongs to "
                         "the frame it starts on.")
    if frame != track.start_frame:
        raise ValueError(f"The head click of track {track_id} belongs to its start frame (frame "
                         f"{track.start_frame}), not to frame {frame}: go to that frame and click the head there.")
    track.head_px = [float(point_px[0]), float(point_px[1])]


def pending_runs(session: Session, store: ResultsStore) -> list[RunPlan]:
    """The runs "Track" would start now (`tracking_plan.plan_runs`): for each, its tracks, start
    frame, mode and video frames. Empty when nothing waits. Raises ValueError like `plan_runs`."""
    return plan_runs(session, store)


# --------------------------------------------------------------------------- the corrections (SPEC 6.6)

_CORRECTIONS = ("end_track", "new_piece", "retrack_from")


def __getattr__(name: str):
    """The three corrections, which are written in tracking_corrections.py. That module imports from this
    one, so this one loads them when they are asked for (PEP 562): the two modules form no import cycle,
    whichever is imported first, and `from outline_tracker.tracking_edit import retrack_from` works."""
    if name in _CORRECTIONS:
        from outline_tracker import tracking_corrections
        return getattr(tracking_corrections, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
