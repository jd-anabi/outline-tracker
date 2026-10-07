"""Tracking jobs: from a session with prompts to the pixel-space records of results.npz (SPEC 6).

`run_job` is the one entry point, for the command line and for the GUI's worker thread. It plans
the runs (outline_tracker/tracking_plan.py), checks the start frames (tracking_guard.py), and
tracks each run: every frame of the run is decoded, cut to the part the model sees, given to the
segmenter, measured (`measure.measure_mask`) and discarded. A coarse run shows the model one fixed
part of the frame, for all its objects; a fine run shows it a window that follows its one object
(outline_tracker/tracking_fine.py).

Who writes what: `run_job` works on a copy of the job's session and writes only results.npz.
Every change it makes to the session (a run's record, whether the results are complete, a frame
hash made anew on this computer, a fine window it chose) is handed to `Callbacks.save_session` as
a `SessionChanges`, so that session.json has one writer.

Units and coordinates (SPEC 3.1): records are in px in the full frame, Tracker's convention, the
pixel in column c and row r with its center at (c + 0.5, r + 0.5); a box is
(c0, r0, width, height) in whole px of the full frame. Frames are video frame numbers; times are
in s. No Qt, and no torch: the segmenter comes from the job's factory.
"""

from __future__ import annotations

import copy
import math
import numbers
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from outline_tracker.measure import PixelRecord, measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_NPZ, SESSION_JSON
from outline_tracker.segmenter.base import ObjectPrompt, Segmenter
from outline_tracker.session import RunRecord, Session
from outline_tracker.tracking_fine import FineStart, check_fine_settings, fine_starts, fine_window, fine_window_px
from outline_tracker.tracking_guard import FrameHashMismatch, FrameHashUpdate, check_start_frames
from outline_tracker.tracking_plan import RunPlan, dish_box, partial_tracks, plan_runs, run_prompts
from outline_tracker.video import iter_rgb_frames

__all__ = ["AUTOSAVE_EVERY", "Callbacks", "FrameHashMismatch", "FrameHashUpdate", "Job", "RunPlan", "SessionChanges",
           "dish_box", "fine_window", "fine_window_px", "partial_tracks", "plan_runs", "run_job"]

AUTOSAVE_EVERY = 200  # results and session are saved after every this many tracked frames (SPEC 6.4)
COMPLETE, CANCELLED, FAILED = "complete", "cancelled", "failed"  # what `run_job` returns


@dataclass
class Job:
    """What to track. session: the session with its clip, settings and prompts; `run_job` reads it
    once, at the start. run_folder: the folder of results.npz and session.json. video_path: the
    session's video. make_segmenter: called once, without arguments, when the checks have passed;
    returns the segmenter (segmenter/base.py) for every run of the job. track_ids: the tracks to
    track, None for every track that is pending (`plan_runs`)."""

    session: Session
    run_folder: Path
    video_path: Path
    make_segmenter: Callable[[], Segmenter]
    track_ids: Sequence[str] | None = None


@dataclass(frozen=True)
class SessionChanges:
    """What a job changed in the session, to be applied by the session's one writer.

    complete: False while results are partial (SPEC 6.4): while a job runs, and after it when a
    track's results stop early, have a gap or are still to come (`partial_tracks`). run: the record
    of the run the change is about, as it is now (a copy: the receiver's own), and run_index: its
    place in `session.runs`. frame_hashes: the hashes of start frames made anew on this computer
    for clicks that came from another one (X8); they go with the first change of a job.
    fine_windows: (track id, window in px) for each fine window the job chose from the object's
    mask on its start frame (SPEC 6.3); it goes with the change that starts the track's run, and
    becomes the track's `fine_window_px`, so that a later run of the track shows the model the
    object at the same scale. The 96 px of an object that was not found there are not stored.
    """

    complete: bool
    run_index: int
    run: RunRecord
    frame_hashes: tuple[FrameHashUpdate, ...] = ()
    fine_windows: tuple[tuple[str, int], ...] = ()

    def apply(self, session: Session) -> None:
        """Make the changes in `session`: the run's record replaces the one at `run_index`, or is
        added when the session has no such run yet, each new frame hash is stored in the prompts
        it is for (`FrameHashUpdate.apply`), and each chosen fine window (px) in its track, if
        the session still has it."""
        session.complete = self.complete
        record = copy.deepcopy(self.run)
        if self.run_index < len(session.runs):
            session.runs[self.run_index] = record
        else:
            session.runs.append(record)
        for new_hash in self.frame_hashes:
            new_hash.apply(session)
        for track_id, window in self.fine_windows:
            for track in session.tracks:
                if track.id == track_id:
                    track.fine_window_px = window


@dataclass
class Callbacks:
    """How a job reports (SPEC 6.4). All are called on the thread that runs `run_job`.

    progress(done, total, s_per_frame, eta_s): after every tracked frame. done, total: tracked
    frames and frames to track, over all runs of the job; s_per_frame: s per tracked frame so far;
    eta_s: s left at that rate.
    frame_result(track_id, frame, record): one object's `PixelRecord` on video frame `frame`, in px
    in the full frame.
    log(msg): text for the log, one line; only the traceback of a failed run is several lines.
    finished(status): once, when `run_job` returns "complete", "cancelled" or "failed".
    should_cancel(): asked before every run and every frame; True stops the job there.
    save_session(changes): a `SessionChanges` to apply and save. None, the default, applies it to
    the job's own session object and saves session.json in the run folder; the GUI passes a function
    that hands the changes to its own thread.
    """

    progress: Callable[[int, int, float, float], None]
    frame_result: Callable[[str, int, PixelRecord], None]
    log: Callable[[str], None]
    finished: Callable[[str], None]
    should_cancel: Callable[[], bool]
    save_session: Callable[[SessionChanges], None] | None = None


def run_job(job: Job, callbacks: Callbacks) -> str:
    """Track the pending objects of `job.session` and write their records to results.npz.

    Before anything is tracked: the settings are checked, results.npz of the run folder is read if
    there is one, the runs are planned for the job's tracks (`plan_runs`), the prompts are shifted
    into what the model sees (`run_prompts`) and the start frames are checked (`check_start_frames`).
    Then `make_segmenter` is called, and each fine object is looked for on its start frame, which
    gives its window, the window's first place and its clicks in that window
    (`tracking_fine.fine_starts`; that reads the video up to the last such frame once more).
    What fails there is raised, nothing is written and `finished` is not called: ValueError
    (`core_open_frac` not finite or negative, `fps_true` not a positive number, a fine window or
    `fine_window_factor` that cannot be used, an unknown track, a prompt that cannot be used, a
    fine object whose window holds none of its positive clicks, a start frame the video does not
    have, a results.npz of another version), `FrameHashMismatch`, and whatever `make_segmenter`
    raises.

    Then each run is tracked, frame by frame: a coarse run on the part of the frame that
    `plan.input_box` names, a fine run on a window that follows its object (SPEC 6.3). A fine
    window the job chose from the object's mask goes to the session when the object's run starts
    (`SessionChanges`); the 96 px of an object the model did not find are for that run only.
    Records are in px in the full frame, frames are video frame numbers. results.npz and, through
    `save_session`, the session are saved when a run starts (the session only), after every 200
    tracked frames, and when a run ends. Returns "complete"; "cancelled" when `should_cancel` said
    so or the user pressed Ctrl+C; "failed" when the model or the video raised an error during a
    run, which is logged. In all three cases what was tracked is saved. A video that ends before
    the clip does is tracked to its last frame, which is logged, and is "complete".

    The session's `complete` describes the results, not this job: it is True only after "complete",
    and then only if no track of the session is left partial (`partial_tracks`): one that an
    earlier job left before its end, one with a gap, one that has clicks and still waits.

    Clicks whose frame hash another computer's decoder made cannot be compared here; this
    computer's hash of their start frame goes to the session with the first change (X8), and is
    compared from then on.
    """
    session = copy.deepcopy(job.session)
    _check_settings(session)
    results_path = Path(job.run_folder) / RESULTS_NPZ
    store = ResultsStore.load(results_path) if results_path.is_file() else ResultsStore()
    plans = _plan(session, store, job.track_ids)
    if not plans:
        callbacks.log("Nothing to track: no object has clicks that are not tracked yet.")
        callbacks.finished(COMPLETE)
        return COMPLETE
    check_fine_settings(session, plans)
    prompts = [run_prompts(session, plan) for plan in plans]
    new_hashes = check_start_frames(job.video_path, session, plans, callbacks.log)

    segmenter = job.make_segmenter()
    own_log = getattr(segmenter, "log", None)
    if own_log is not None:  # so that a switch from the Apple GPU to the processor reaches the log
        segmenter.log = callbacks.log
    frame_size = (session.video.width, session.video.height)
    status = COMPLETE
    try:
        fine = fine_starts(job.video_path, session, plans, segmenter)  # before any run: a preview needs the model idle
        runner = _Runner(job, session, store, callbacks, segmenter, total=sum(len(plan.frames) for plan in plans),
                         new_hashes=new_hashes)
        for number, (plan, run_prompt) in enumerate(zip(plans, prompts), start=1):
            if callbacks.should_cancel():
                status = CANCELLED
                break
            if plan.mode == "fine":
                start = fine[plan.track_ids[0]]
                status = runner.track(plan, start.crop, start.prompts, number, len(plans), start)
            else:
                status = runner.track(plan, _FixedView(plan.input_box, frame_size), run_prompt, number, len(plans))
            if status != COMPLETE:
                break
    finally:
        if own_log is not None:
            segmenter.log = own_log
    callbacks.finished(status)
    return status


def _check_settings(session: Session) -> None:
    """Refuse the settings that would only fail later, on the first frame or at export."""
    def number(value) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)

    frac = session.processing.core_open_frac
    if not number(frac) or frac < 0:
        raise ValueError(f"core_open_frac must be a number that is 0 or more (a fraction of the body length; 0.1 "
                         f"unless changed), not {frac!r}.")
    fps = session.time.fps_true
    if not number(fps) or fps <= 0:
        raise ValueError(f"fps_true must be a positive number of frames per second, not {fps!r}. Take it from the "
                         "manifest or the stopwatch clip, or type it in, before tracking.")


def _plan(session: Session, store: ResultsStore, track_ids: Sequence[str] | None) -> list[RunPlan]:
    """The runs of a job: those of every pending track, or of the named tracks only."""
    if track_ids is None:
        return plan_runs(session, store)
    known = [track.id for track in session.tracks]
    unknown = [track_id for track_id in track_ids if track_id not in known]
    if unknown:
        raise ValueError(f"The session has no track {', '.join(unknown)} (its tracks: {', '.join(known) or 'none'}).")
    return plan_runs(replace(session, tracks=[track for track in session.tracks if track.id in track_ids]), store)


def _now() -> str:
    """The time now, local, as ISO 8601 with the offset from UTC, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class _FixedView:
    """What the model is shown during a coarse run: always the same `box` = (c0, r0, width, height),
    whole px of the full frame (the dish square or the whole frame, SPEC 6.2). `frame_size` =
    (width, height) of the video's frames, px. A fine run has `tracking_fine.FollowCrop` instead."""

    mode = "coarse"

    def __init__(self, box: tuple[int, int, int, int], frame_size: tuple[int, int]):
        self.box = box
        self.whole = box == (0, 0, *frame_size)

    def describe(self) -> str:
        c0, r0, width, height = self.box
        seen = "the whole frame" if self.whole else f"columns {c0} to {c0 + width - 1}, rows {r0} to {r0 + height - 1}"
        return f"{seen} ({width} x {height} px)"

    def cut(self, rgb: np.ndarray) -> np.ndarray:
        # without a crop the model gets the decoded frame itself, as last week (SPEC 6.2)
        c0, r0, width, height = self.box
        return rgb if self.whole else np.ascontiguousarray(rgb[r0:r0 + height, c0:c0 + width])

    def measure(self, result, frame: int, core_open_frac: float) -> PixelRecord:
        in_frame = replace(result, offset=(result.offset[0] + self.box[0], result.offset[1] + self.box[1]))
        return measure_mask(in_frame, frame, self.box, self.mode, core_open_frac)


class _Runner:
    """One job while it runs: its session copy, results, segmenter and frame counts."""

    def __init__(self, job: Job, session: Session, store: ResultsStore, callbacks: Callbacks, segmenter, total: int,
                 new_hashes: Sequence[FrameHashUpdate] = ()):
        self.session, self.store, self.callbacks, self.segmenter = session, store, callbacks, segmenter
        self.video_path = Path(job.video_path)
        self.results_path = Path(job.run_folder) / RESULTS_NPZ
        self.save_session = callbacks.save_session or _save_to_run_folder(job, callbacks.log)
        self.total, self.done = total, 0  # frames to track and frames tracked, over all runs of the job
        self.began = time.perf_counter()
        self.new_hashes = tuple(new_hashes)  # frame hashes made anew (X8), until they are handed over
        self.video_end: int | None = None  # the last frame a run got from a video that ended before the run did

    def track(self, plan: RunPlan, view, prompts: list[ObjectPrompt], number: int, count: int,
              fine: FineStart | None = None) -> str:
        """Track one run (SPEC 6.2, 6.3): every frame of `plan.frames` is cut to what `view` shows
        the model (`view.cut`; `view.box`, px of the full frame, says which part), given to the
        segmenter, and each object's result is shifted back into the full frame and measured
        (`view.measure`). `view` is a `_FixedView` for a coarse run and the `FollowCrop` of `fine`,
        how the run starts, for a fine one. `prompts` are in px of the first image. Returns
        "complete", "cancelled" or "failed"."""
        session, callbacks, segmenter, log = self.session, self.callbacks, self.segmenter, self.callbacks.log
        set_view = getattr(segmenter, "set_view", None)  # ExactFake is told what each image shows
        frames = plan.frames
        log(f"Run {number} of {count}: {', '.join(plan.track_ids)} ({plan.mode}), frames {frames[0]} to {frames[-1]} "
            f"every {frames.step} ({len(frames)} frames). The model sees {view.describe()}.")
        windows = ()
        if fine is not None:
            if not fine.found:
                log(f"  The model found nothing at the clicks of {plan.track_ids[0]} on frame {plan.start_frame}: "
                    "the window starts around the first click.")
            if fine.chosen and fine.found:  # measured on the object: a later run shows it at the same scale
                windows = ((plan.track_ids[0], fine.crop.window),)

        record = RunRecord(tracks=list(plan.track_ids), start_frame=plan.start_frame, mode=plan.mode, started=_now())
        index = len(session.runs)
        session.runs.append(record)
        self._hand_over(record, index, complete=False, fine_windows=windows)
        status, tracked, began, frame = COMPLETE, 0, time.perf_counter(), plan.start_frame
        try:
            with closing(iter_rgb_frames(self.video_path, frames)) as decoded:  # closed at once: Windows locks the file
                for frame, rgb in decoded:
                    if callbacks.should_cancel():
                        status = CANCELLED
                        break
                    image = view.cut(rgb)
                    if set_view is not None:
                        set_view(frame, view.box[:2], view.box[2:])
                    results = segmenter.start(image, prompts) if tracked == 0 else segmenter.step(image)
                    for track_id, result in zip(plan.track_ids, results, strict=True):
                        measured = view.measure(result, frame, session.processing.core_open_frac)
                        self.store.put(track_id, measured)
                        callbacks.frame_result(track_id, frame, measured)
                    tracked += 1
                    self.done += 1
                    self._progress()
                    if self.done % AUTOSAVE_EVERY == 0:
                        self._save(record, index, tracked, began, complete=False)
        except KeyboardInterrupt:
            status = CANCELLED
            log(f"  Stopped on frame {frame}; what was tracked so far is kept.")
        except Exception as err:
            status = FAILED
            log(f"  Tracking failed on frame {frame} ({type(err).__name__}: {err}); what was tracked before is kept.")
            log(traceback.format_exc().rstrip())
        finally:
            segmenter.close()

        if status == COMPLETE and tracked == 0:
            status = FAILED
            log(f"  The video gave no frame {plan.start_frame}: nothing was tracked.")
        elif status == COMPLETE and tracked < len(frames):
            self.total -= len(frames) - tracked
            log(f"  The video ended after frame {frames[tracked - 1]}: {tracked} of the run's {len(frames)} frames "
                f"were tracked (the clip asks for frames up to {frames[-1]}).")
            self._progress()
        record.finished = _now()
        self._save(record, index, tracked, began, complete=self._whole(status, frames, tracked))
        log(f"  {tracked} frames in {time.perf_counter() - began:.1f} s ({record.seconds_per_frame:.2f} s per frame)"
            + (f" on {record.device}." if record.device else "."))
        return status

    def _whole(self, status: str, frames: range, tracked: int) -> bool:
        """What the session's `complete` is when a run has ended (SPEC 6.4): False after a run that
        did not complete; else it is about the results, not about this job: True only if no track
        of the session is partial (`partial_tracks`). So it stays False while the job has runs to
        come, whose tracks still wait. `frames`: the run's video frames, of which the first
        `tracked` were tracked; fewer than all means the video ended there, before the clip does.
        A run that ends early because its track was ended says nothing about the other tracks."""
        if status != COMPLETE:
            return False
        if tracked < len(frames):
            self.video_end = frames[tracked - 1]
        return not partial_tracks(self.session, self.store, self.video_end)

    def _progress(self) -> None:
        s_per_frame = (time.perf_counter() - self.began) / self.done
        self.callbacks.progress(self.done, self.total, s_per_frame, s_per_frame * (self.total - self.done))

    def _save(self, record: RunRecord, index: int, tracked: int, began: float, complete: bool) -> None:
        """Write results.npz (if the run has tracked a frame), then hand the run's record over."""
        record.frames_done = tracked
        record.seconds_per_frame = (time.perf_counter() - began) / tracked if tracked else 0.0
        if tracked:
            written = self.store.save(self.results_path)
            if written != self.results_path:
                self.callbacks.log(f"  {self.results_path.name} is open in another program: the results are in "
                                   f"{written.name} next to it. Close that program, then rename the file.")
        self._hand_over(record, index, complete)

    def _hand_over(self, record: RunRecord, index: int, complete: bool,
                   fine_windows: tuple[tuple[str, int], ...] = ()) -> None:
        """Give the run's record, with the device in use now, to the session's writer."""
        record.device = str(getattr(self.segmenter, "device", None) or "")
        record.model_id = getattr(self.segmenter, "model_id", None)
        record.weights_sha256 = getattr(self.segmenter, "weights_sha256", None)
        new_hashes, self.new_hashes = self.new_hashes, ()  # with the first change of the job only
        self.save_session(SessionChanges(complete=complete, run_index=index, run=copy.deepcopy(record),
                                         frame_hashes=new_hashes, fine_windows=fine_windows))


def _save_to_run_folder(job: Job, log: Callable[[str], None]) -> Callable[[SessionChanges], None]:
    """The default `save_session`: apply the changes to the job's session and save session.json."""
    target = Path(job.run_folder) / SESSION_JSON

    def save(changes: SessionChanges) -> None:
        changes.apply(job.session)
        written = job.session.save(target)
        if written != target:
            log(f"  {target.name} is open in another program: the session is in {written.name} next to it. Close "
                "that program, then rename the file.")

    return save
