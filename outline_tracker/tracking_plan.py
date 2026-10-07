"""Planning the runs of a tracking job (SPEC 6.1, 6.2): which objects are tracked together, on which
frames, what part of the frame the model is shown, and the clicks as the model gets them. Also
what is left when jobs have run: the tracks whose results are partial (`partial_tracks`, SPEC 6.4),
and the frame up to which a track's next run may start (`check_no_gap`, SPEC 8.2).

A run is a set of tracks plus a start frame: coarse objects with the same start frame share one
streaming session of the model, a fine object has a session of its own. A run covers the frames of
the clip's grid from its start frame to the clip's end, or to the frame where its tracks were ended
("End track here", SPEC 6.6): a track is never tracked beyond its end.

Units and coordinates (SPEC 3.1): px in Tracker's convention, the origin at the top-left corner of
the frame, u to the right, v downward, the pixel in column c and row r with its center at
(c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the full frame: the pixels of
the array slice [r0:r0 + height, c0:c0 + width]. Frames are video frame numbers.

No Qt, no torch.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, replace

from outline_tracker.geometry import grid_frames
from outline_tracker.measure import MODES
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import ObjectPrompt, points_in_image
from outline_tracker.session import Circle, Session, Track

DISH_MARGIN = 0.03  # the dish crop reaches m = 0.03 R beyond the fitted wall on every side (SPEC 6.2)


@dataclass(frozen=True)
class RunPlan:
    """One run of a job (SPEC 6.1).

    track_ids: the objects of the run, in the order of the session; one for a fine run.
    start_frame: the video frame number the run starts on, the frame of the objects' clicks.
    mode: "coarse" or "fine". frames: the video frame numbers of the run, on the clip's grid from
    `start_frame` to the clip's end, or to the last grid frame up to `ended_at` of its tracks, which
    is then the same for all of them. input_box: (c0, r0, width, height) in whole px of the full
    frame, the part of the frame the model is shown in a coarse run: the dish square or the whole
    frame. For a fine run it is the part on which the object is first looked for.
    """

    track_ids: tuple[str, ...]
    start_frame: int
    mode: str
    frames: range
    input_box: tuple[int, int, int, int]


def _finite(*values) -> bool:
    return all(isinstance(value, numbers.Real) and math.isfinite(value) for value in values)


def dish_box(circle: Circle, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The dish crop of SPEC 6.2: the square [uc - R - m, uc + R + m] x [vc - R - m, vc + R + m] with
    m = 0.03 R, widened to whole pixels (floor and ceil) and clipped to the frame.

    `circle` is the fitted dish wall (center (uc, vc) and radius R in px of the full frame, Tracker's
    convention); `frame_size` = (width, height) of the frame in px. Returns (c0, r0, width, height)
    in whole px of the full frame: the pixels [r0:r0 + height, c0:c0 + width]. Raises ValueError
    for a circle without a positive, finite radius or a finite center, and for one whose square
    lies outside the frame.
    """
    (cu, cv), radius = circle.center_px, circle.radius_px
    width, height = frame_size
    if not _finite(cu, cv, radius) or radius <= 0:
        raise ValueError(f"The dish circle (center {circle.center_px}, radius {radius!r} px) cannot give a crop: "
                         "fit the circle again, or switch off \"Crop to dish\".")
    half = radius * (1.0 + DISH_MARGIN)
    c0, c1 = max(math.floor(cu - half), 0), min(math.ceil(cu + half), int(width))
    r0, r1 = max(math.floor(cv - half), 0), min(math.ceil(cv + half), int(height))
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"The dish circle (center {circle.center_px}, radius {radius:g} px) lies outside the "
                         f"{width} x {height} px frame: fit the circle again, or switch off \"Crop to dish\".")
    return c0, r0, c1 - c0, r1 - r0


def view_box(session: Session) -> tuple[int, int, int, int]:
    """What coarse tracking shows the model (SPEC 6.2): the dish square (`dish_box`) when the session
    has a circle and `processing.dish_crop` is on, else the whole frame. Returns
    (c0, r0, width, height) in whole px of the full frame. Raises ValueError when the session does
    not hold the frame size of its video, or the circle gives no crop."""
    width, height = session.video.width, session.video.height
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise ValueError(f"The session does not know the frame size of its video ({width!r} x {height!r} px): "
                         "open the video again.")
    if session.circle is not None and session.processing.dish_crop:
        return dish_box(session.circle, (width, height))
    return 0, 0, width, height


def pending_from(track: Track, store: ResultsStore) -> int | None:
    """The video frame from which a track still has to be tracked: the frame of its newest clicks,
    unless `store`, the results so far, already holds that frame. None for a track that has
    nothing to track."""
    if not track.prompts:
        return None
    start = max(prompt.frame for prompt in track.prompts)
    if track.id in store.track_ids and start in store.arrays(track.id).frames:
        return None
    return start


def _last_grid_frame(grid: range, limit: int) -> int:
    """The last frame of the clip's grid at or before video frame `limit`; a frame before the grid's
    first one when `limit` is."""
    return grid.start + (min(limit, grid[-1]) - grid.start) // grid.step * grid.step


def check_no_gap(session: Session, track: Track, store: ResultsStore, frame: int) -> None:
    """Refuse a run of `track` from video frame `frame` that would leave a gap inside the track.

    A track has every frame of the clip's grid from its first to its last (SPEC 8.2), so its next
    run starts no later than the first grid frame after its last record in `store`, the results so
    far. `frame` is a frame of the clip's grid. A track without records is not checked: its first
    clicks set its start. Raises ValueError, which names the track, its last frame with results
    and the latest frame that is allowed, and as `geometry.grid_frames` does.
    """
    if track.id not in store.track_ids:
        return
    clip = session.clip
    grid = grid_frames(clip.start, clip.end, clip.step)
    last = int(store.arrays(track.id).frames[-1])
    latest = max(grid[0], _last_grid_frame(grid, last) + grid.step)
    if frame > latest:
        raise ValueError(f"Track {track.id} has results up to frame {last} only: tracking it from frame {frame} on "
                         f"would leave the frames between without results. Go to frame {latest} or an earlier one.")


def plan_runs(session: Session, store: ResultsStore) -> list[RunPlan]:
    """The runs that "Track" starts (SPEC 6.1): one per group of pending coarse objects with the same
    start frame and the same last frame, and one per pending fine object.

    An object is pending when it has clicks and `store` (the results so far) does not hold the
    frame of its newest clicks; its run starts on that frame, a video frame number. For an object
    that was never tracked this is its start frame. A run ends on the clip's last grid frame; the
    run of a track that was ended, on the last grid frame up to its `ended_at`. Runs are in the
    order of their start frames, a coarse run before the fine runs of the same frame. Each run's
    `input_box` is `view_box(session)`, in px of the full frame. Raises ValueError for a clip
    without frames (`geometry.grid_frames`), a mode other than "coarse" or "fine", clicks on a
    frame that is not on the clip's grid, and clicks after the end of a track that was ended.
    """
    clip = session.clip
    grid = grid_frames(clip.start, clip.end, clip.step)
    box = view_box(session)
    coarse: dict[tuple[int, int], list[str]] = {}  # (start frame, last frame) -> the tracks
    fine: list[tuple[int, int, str]] = []
    for track in session.tracks:
        if track.mode not in MODES:
            raise ValueError(f"Track {track.id}: the mode must be 'coarse' or 'fine', not {track.mode!r}.")
        start = pending_from(track, store)
        if start is None:
            continue
        if start not in grid:
            raise ValueError(f"Track {track.id} was clicked on frame {start}, which is not a frame of the clip "
                             f"(frames {grid[0]} to {grid[-1]}, every {grid.step}). Go to a frame of the clip and "
                             "click the object there.")
        last = grid[-1] if track.ended_at is None else _last_grid_frame(grid, track.ended_at)
        if start > last:
            raise ValueError(f"Track {track.id} was ended at frame {track.ended_at} and has clicks on frame {start}, "
                             "after its end. Undo those clicks, or continue the animal as a new track from there.")
        if track.mode == "coarse":
            coarse.setdefault((int(start), last), []).append(track.id)
        else:
            fine.append((int(start), last, track.id))

    def frames(start: int, last: int) -> range:
        return grid[grid.index(start):grid.index(last) + 1]

    plans = [RunPlan(tuple(ids), start, "coarse", frames(start, last), box) for (start, last), ids in coarse.items()]
    plans += [RunPlan((track_id,), start, "fine", frames(start, last), box) for start, last, track_id in fine]
    return sorted(plans, key=lambda plan: (plan.start_frame, plan.mode != "coarse"))


def plan_job(session: Session, store: ResultsStore, track_ids: Sequence[str] | None) -> list[RunPlan]:
    """The runs of a job: those of every pending track (`plan_runs`) when `track_ids` is None, else
    those of the named tracks only. Raises ValueError for a name the session has no track for, and
    as `plan_runs` does. Frames are video frame numbers, boxes px of the full frame."""
    if track_ids is None:
        return plan_runs(session, store)
    known = [track.id for track in session.tracks]
    unknown = [track_id for track_id in track_ids if track_id not in known]
    if unknown:
        raise ValueError(f"The session has no track {', '.join(unknown)} (its tracks: {', '.join(known) or 'none'}).")
    return plan_runs(replace(session, tracks=[track for track in session.tracks if track.id in track_ids]), store)


def partial_tracks(session: Session, store: ResultsStore, last_frame: int | None = None) -> list[str]:
    """The tracks whose results are partial (SPEC 6.4), in the order of the session. The session's
    `complete` is true only when there is none.

    A track is partial when it has clicks but no record in `store` (the results so far); when it
    has clicks that are not tracked yet (`pending_from`: `plan_runs` has a run for it, or refuses
    its clicks, those after the end of a track that was ended); when its records stop before its
    last frame; or when a frame of the clip's grid is missing between its first and its last
    record. Its last frame is the last frame of the clip's grid, or the last grid
    frame up to `track.ended_at` for a track that was ended. `last_frame` is the last video frame
    number tracking can reach when the video ends before the clip does; None for the clip's end.
    A track without clicks and without records is not partial. Raises ValueError for a clip
    without frames (`geometry.grid_frames`).
    """
    clip = session.clip
    grid = grid_frames(clip.start, clip.end, clip.step)
    reach = grid[-1] if last_frame is None else min(grid[-1], int(last_frame))
    partial = []
    for track in session.tracks:
        if track.id not in store.track_ids:
            if track.prompts:  # clicked on, never tracked
                partial.append(track.id)
            continue
        frames = store.arrays(track.id).frames
        first, last = int(frames[0]), int(frames[-1])
        end = _last_grid_frame(grid, reach if track.ended_at is None else min(reach, track.ended_at))
        gap = (last - first) // grid.step + 1 > len(frames)
        if pending_from(track, store) is not None or last < end or gap:
            partial.append(track.id)
    return partial


def run_prompts(session: Session, plan: RunPlan) -> list[ObjectPrompt]:
    """The clicks that start a run, as the model gets them: one prompt per object of the plan, in
    its order, holding the object's clicks on the run's start frame.

    The session stores clicks in px of the full frame; here they are shifted into `plan.input_box`
    (px in the pixel frame of the image the model is shown, SPEC 6.2), and the clicks that fall
    outside that image are dropped. Raises `PromptError`, a ValueError, for an object that has no
    positive click left inside it.
    """
    c0, r0, width, height = plan.input_box
    tracks = {track.id: track for track in session.tracks}
    prompts = []
    for track_id in plan.track_ids:
        points, labels = [], []
        for prompt in tracks[track_id].prompts:
            if prompt.frame == plan.start_frame:
                points += [(u - c0, v - r0) for u, v in prompt.points_px]
                labels += list(prompt.labels)
        prompts.append(points_in_image(ObjectPrompt(track_id, points, labels), height, width))
    return prompts
