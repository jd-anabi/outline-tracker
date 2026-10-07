"""The three corrections of SPEC 6.6, without Qt: `retrack_from` ("Re-track from here"), `end_track`
("End track here") and `new_piece` ("Continue as new track"). Each adds an entry to `session.corrections`
(SPEC 8.10).

The rest of the edit API is in outline_tracker/tracking_edit.py, which this module imports from: the object
table (`add_object`, `remove_object`), the clicks, `pending_runs`, the `Edit` that every edit returns, and the
helpers that find a track, move a frame onto the grid and read and save the results. The three functions are
importable from there and from outline_tracker.tracking too. "The module's text" in the docstrings below means
this text.

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

"Re-track from here", the whole sequence:

    edit = retrack_from(session, None, ["A"], k, {"A": clicks}, run_folder)
    session.save(run_folder / "session.json")                               # the caller's part
    run_job(Job(session, run_folder, video_path, make_segmenter), callbacks)

The job tracks what is pending: the tracks named, from frame k on. `new_piece` and a new object
are tracked by the same last two lines. No edit makes a gap inside a track (SPEC 8.2): a track
with results is clicked on, and re-tracked from, the first grid frame after its last record at the
latest (`tracking_plan.check_no_gap`).

Units and coordinates (SPEC 3.1, 3.4): frames are video frame numbers; a start, re-track or
new-piece frame that is not on the clip's grid is moved forward to the next grid frame (X20), and
the returned `Edit` says so. Points are (u, v) in px of the full frame, Tracker's convention: u to
the right, v downward, the pixel in column c and row r with its center at (c + 0.5, r + 0.5).
No Qt, no torch.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping, Sequence
from datetime import datetime

from outline_tracker.results import ResultsStore
from outline_tracker.session import Correction, Prompt, Session, Track
from outline_tracker.tracking_edit import Edit, _on_grid, _reach, _results, _save, _start_on, _track
from outline_tracker.tracking_ids import next_piece_id
from outline_tracker.tracking_plan import check_no_gap, partial_tracks, pending_from

__all__ = ["end_track", "new_piece", "retrack_from"]


# --------------------------------------------------------------------------- corrections (SPEC 6.6)

def _placed(track_id: str, clicks: Prompt, frame: int, moved: bool) -> Prompt:
    """The session's own copy of clicks given for a track, on grid frame `frame`. When the frame
    was moved the copy has no frame hash: the hash given is that of another frame."""
    if len(clicks.points_px) != len(clicks.labels):
        raise ValueError(f"The clicks for track {track_id} have {len(clicks.points_px)} points and "
                         f"{len(clicks.labels)} labels: there must be one label (1 or 0) per point.")
    placed = copy.deepcopy(clicks)
    placed.frame = frame
    placed.points_px = [[float(u), float(v)] for u, v in clicks.points_px]
    if moved:
        placed.frame_hash = placed.decoder = None
    return placed


def _record(session: Session, track_ids: Sequence[str], action: str, frame: int, clicks: Sequence[Prompt] = ()) -> None:
    """Add a correction to the session (SPEC 8.10), with the local time now as ISO 8601 with the
    offset from UTC, and copies of the clicks."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    session.corrections.append(Correction(time=now, tracks=list(track_ids), action=action, frame=frame,
                                          prompts=copy.deepcopy(list(clicks))))


def retrack_from(session: Session, store: ResultsStore | None, track_ids: Sequence[str], frame_k,
                 prompts: Mapping[str, Prompt] | None, run_folder) -> Edit:
    """"Re-track from here" (SPEC 6.6), up to the job that tracks (the module's text has the
    sequence): the named tracks get new clicks on frame k and lose their results from k on.

    store: the results, changed here too, or None for results.npz of `run_folder` as it is now
    (the module's text); `Edit.store` is the store that was saved.
    frame_k: a video frame number, moved forward onto the clip's grid if needed (`Edit.moved`).
    prompts: track id -> the new clicks for that track, one `Prompt` whose points are (u, v) in px
    of the full frame and whose `frame_hash` and `decoder` are those of the frame clicked on (they
    are dropped when the frame was moved); its `frame` is set to k. A track that already has
    clicks on frame k needs none here. Every track needs a positive click on k.

    Clicks on frame k add up: those a track has there already, from `add_prompt` or from an
    earlier "Re-track from here" on the same k, stay, and the run starts with all of them. The
    earlier ones are removed with `undo_prompt` (the newest first), before the new ones are given.

    For each track: its clicks on later frames are removed, since the results they led to are
    replaced; a k before its start frame, and the first clicks of a track that had none, make k
    its start frame; its records at frames >= k are removed (`ResultsStore.replace_from`). The
    results are saved to results.npz of `run_folder`, the correction is recorded with the tracks'
    clicks on k, and `session.complete` is False until a job has tracked them. Other tracks are
    untouched. A track that was ended stays ended: the job tracks it from k to its end.

    Raises ValueError, changing nothing, for no track, an unknown track, clicks for a track that
    is not named, clicks whose points and labels do not match, a track without a positive click
    on k, a k after the end of a track that was ended, a k after the frame of clicks that still
    wait to be tracked, and a k later than the first grid frame after a track's last record: the
    frames between would have no results (`tracking_plan.check_no_gap`). Raises RuntimeError,
    changing nothing, when results.npz stays locked by another program, or the run folder holds a
    results.new.npz (the module's text).
    """
    names = list(dict.fromkeys(track_ids))
    if not names:
        raise ValueError("\"Re-track from here\" needs at least one track.")
    tracks = [_track(session, name) for name in names]
    given = dict(prompts or {})
    stray = [name for name in given if name not in names]
    if stray:
        raise ValueError(f"Clicks were given for {', '.join(stray)}, but the tracks to re-track are "
                         f"{', '.join(names)}.")
    frame, moved, note = _on_grid(session, frame_k)
    new = {name: _placed(name, clicks, frame, moved) for name, clicks in given.items()}
    results = _results(store, run_folder)
    for track in tracks:
        if track.ended_at is not None and frame > track.ended_at:
            raise ValueError(f"Track {track.id} was ended at frame {track.ended_at}: it cannot be tracked again from "
                             f"frame {frame}. Continue the animal as a new track from there.")
        waits = pending_from(track, results)
        if waits is not None and waits < frame:
            raise ValueError(f"Track {track.id} has clicks on frame {waits} that are not tracked yet: a run from "
                             f"frame {frame} would leave the frames between untracked. Track first, or re-track "
                             f"from frame {waits}.")
        check_no_gap(session, track, results, frame)
        on_k = [prompt for prompt in [*track.prompts, *([new[track.id]] if track.id in new else [])]
                if prompt.frame == frame]
        if not any(label == 1 for prompt in on_k for label in prompt.labels):
            raise ValueError(f"Track {track.id} has no positive click on frame {frame}: click the object there.")

    def remove(kept: ResultsStore) -> None:
        for name in names:
            kept.replace_from(name, frame)

    results_file = _save(results, store is not None, run_folder, remove)  # first: it may refuse
    for track in tracks:
        _start_on(track, min(track.start_frame, frame) if track.prompts else frame)
        track.prompts = [prompt for prompt in track.prompts if prompt.frame <= frame]
        if track.id in new:
            track.prompts.append(new[track.id])
    _record(session, names, "retrack", frame,
            [prompt for track in tracks for prompt in track.prompts if prompt.frame == frame])
    session.complete = False
    return Edit(tuple(names), frame, moved, note, results_file, results)


def end_track(session: Session, store: ResultsStore | None, track_id: str, frame_k: int, run_folder) -> Edit:
    """"End track here" (SPEC 6.6): the track's records after video frame `frame_k` are removed,
    and it is marked as ended there (`ended_at`), so that no later job tracks it beyond.

    store: the results, changed here too, or None for results.npz of `run_folder` as it is now
    (the module's text); `Edit.store` is that store after the edit. `frame_k` is used as it is;
    between two grid frames, the rows stop at the last grid frame up to it. Clicks of the track on
    later frames are removed. When records were removed, the results are saved to results.npz of
    `run_folder` (`Edit.results_file`). The correction is recorded and `session.complete` follows
    the results: a track that stops at its end is whole. Raises ValueError, changing nothing, for
    an unknown track and a frame before the track's start frame, and RuntimeError, changing
    nothing, when results.npz stays locked by another program, or the run folder holds a
    results.new.npz (the module's text).
    """
    track = _track(session, track_id)
    frame = operator.index(frame_k)
    if frame < track.start_frame:
        raise ValueError(f"Track {track_id} starts on frame {track.start_frame}: it cannot end before that, on frame "
                         f"{frame}. Remove the object instead.")
    results = _results(store, run_folder)
    reach = _reach(session, results)
    results_file = None
    if track_id in results.track_ids and results.arrays(track_id).frames[-1] > frame:
        results_file = _save(results, store is not None, run_folder, lambda kept: kept.truncate_after(track_id, frame))
    track.prompts = [prompt for prompt in track.prompts if prompt.frame <= frame]
    track.ended_at = frame
    _record(session, [track_id], "end", frame)
    session.complete = not partial_tracks(session, results, reach)
    return Edit((track_id,), frame, results_file=results_file, store=results)


def new_piece(session: Session, parent_id: str, frame_m, prompts: Prompt | None = None) -> Edit:
    """"Continue as new track" (SPEC 6.6): add the next free piece of the animal of `parent_id`
    (A2, then A3, ...; also when `parent_id` is itself a piece), starting on frame m. The piece's
    id is `Edit.track_ids[0]`.

    frame_m: a video frame number, moved forward onto the clip's grid if needed (`Edit.moved`).
    prompts: the piece's clicks on that frame, one `Prompt` (points (u, v) in px of the full frame,
    the frame's hash and decoder tag, dropped when the frame was moved); None when the clicks
    follow with `add_prompt`. The piece has the parent's mode and color and nothing else of it: a
    fine piece gets its window from its own start frame (SPEC 6.3), and has no head click. The
    correction is recorded; with clicks, `session.complete` is False until a job has tracked the
    piece from m. The results are not touched: ending the parent is `end_track`.

    Raises ValueError, changing nothing, for an unknown parent, a parent whose id is not capital
    letters with an optional number, clicks whose points and labels do not match, and clicks
    without a positive one.
    """
    parent = _track(session, parent_id)
    piece_id = next_piece_id(session, parent.id)
    frame, moved, note = _on_grid(session, frame_m)
    clicks = [] if prompts is None else [_placed(piece_id, prompts, frame, moved)]
    if clicks and 1 not in clicks[0].labels:
        raise ValueError(f"The clicks for the new track {piece_id} hold no positive click: click on the animal itself.")
    session.tracks.append(Track(id=piece_id, color=parent.color, mode=parent.mode, start_frame=frame, prompts=clicks))
    _record(session, [piece_id], "new_piece", frame, clicks)
    if clicks:
        session.complete = False
    return Edit((piece_id,), frame, moved, note)
