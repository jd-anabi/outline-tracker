"""Tracking jobs: from a session with prompts to the pixel-space records of results.npz (SPEC 6).

`run_job` is the one entry point, for the command line and for the GUI's worker thread. It plans
the runs (outline_tracker/tracking_plan.py), checks the start frames (tracking_guard.py), and
tracks each coarse run: every frame of the run is decoded, cut to the part the model sees, given
to the segmenter, measured (`measure.measure_mask`) and discarded. The fine runner comes next to
the coarse one.

Who writes what: `run_job` works on a copy of the job's session and writes only results.npz.
Every change it makes to the session (a run's record, whether the results are complete) is handed
to `Callbacks.save_session` as a `SessionChanges`, so that session.json has one writer.

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
from outline_tracker.tracking_guard import FrameHashMismatch, check_start_frames
from outline_tracker.tracking_plan import RunPlan, dish_box, plan_runs, run_prompts
from outline_tracker.video import iter_rgb_frames

__all__ = ["AUTOSAVE_EVERY", "Callbacks", "FrameHashMismatch", "Job", "RunPlan", "SessionChanges", "dish_box",
           "plan_runs", "run_job"]

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

    complete: False while results are partial (SPEC 6.4). run: the record of the run the change is
    about, as it is now (a copy: the receiver's own), and run_index: its place in `session.runs`.
    """

    complete: bool
    run_index: int
    run: RunRecord

    def apply(self, session: Session) -> None:
        """Make the changes in `session`: the run's record replaces the one at `run_index`, or is
        added when the session has no such run yet. No quantities, so no units."""
        session.complete = self.complete
        record = copy.deepcopy(self.run)
        if self.run_index < len(session.runs):
            session.runs[self.run_index] = record
        else:
            session.runs.append(record)


@dataclass
class Callbacks:
    """How a job reports (SPEC 6.4). All are called on the thread that runs `run_job`.

    progress(done, total, s_per_frame, eta_s): after every tracked frame. done, total: tracked
    frames and frames to track, over all runs of the job; s_per_frame: s per tracked frame so far;
    eta_s: s left at that rate.
    frame_result(track_id, frame, record): one object's `PixelRecord` on video frame `frame`, in px
    in the full frame.
    log(msg): one line of text for the log.
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
    What fails there is raised, nothing is written and `finished` is not called: ValueError
    (`core_open_frac` not finite or negative, `fps_true` not a positive number, an unknown track, a
    prompt that cannot be used, a start frame the video does not have, a results.npz of another
    version), `FrameHashMismatch`, NotImplementedError for a fine object (its runner is not written
    yet), and whatever `make_segmenter` raises, which is called last.

    Then each run is tracked, frame by frame. Records are in px in the full frame, frames are video
    frame numbers. results.npz and, through `save_session`, the session are saved when a run
    starts (the session only), after every 200 tracked frames, and when a run ends. Returns
    "complete"; "cancelled" when `should_cancel` said so or the user pressed Ctrl+C; "failed" when
    the model or the video raised an error during a run, which is logged. In all three cases what
    was tracked is saved, and the session's `complete` is True only after "complete". A video that
    ends before the clip does is tracked to its last frame, which is logged, and is "complete".
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
    fine = [plan.track_ids[0] for plan in plans if plan.mode == "fine"]
    if fine:
        raise NotImplementedError(f"Fine tracking is not available yet (fine objects: {', '.join(fine)}). Set them "
                                  "to coarse, or track the other objects by name.")
    prompts = [run_prompts(session, plan) for plan in plans]
    check_start_frames(job.video_path, session, plans, callbacks.log)

    segmenter = job.make_segmenter()
    own_log = getattr(segmenter, "log", None)
    if own_log is not None:  # so that a switch from the Apple GPU to the processor reaches the log
        segmenter.log = callbacks.log
    runner = _Runner(job, session, store, callbacks, segmenter, total=sum(len(plan.frames) for plan in plans))
    status = COMPLETE
    try:
        for number, (plan, run_prompt) in enumerate(zip(plans, prompts), start=1):
            if callbacks.should_cancel():
                status = CANCELLED
                break
            status = runner.coarse(plan, run_prompt, number, len(plans))
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


class _Runner:
    """One job while it runs: its session copy, results, segmenter and frame counts."""

    def __init__(self, job: Job, session: Session, store: ResultsStore, callbacks: Callbacks, segmenter, total: int):
        self.session, self.store, self.callbacks, self.segmenter = session, store, callbacks, segmenter
        self.video_path = Path(job.video_path)
        self.results_path = Path(job.run_folder) / RESULTS_NPZ
        self.save_session = callbacks.save_session or _save_to_run_folder(job, callbacks.log)
        self.total, self.done = total, 0  # frames to track and frames tracked, over all runs of the job
        self.began = time.perf_counter()

    def coarse(self, plan: RunPlan, prompts: list[ObjectPrompt], number: int, count: int) -> str:
        """Track one coarse run (SPEC 6.2): every frame of `plan.frames` is cut to `plan.input_box`,
        given to the segmenter, and each object's result is shifted back into the full frame and
        measured. `prompts` are in px of the box. Returns "complete", "cancelled" or "failed"."""
        session, callbacks, segmenter, log = self.session, self.callbacks, self.segmenter, self.callbacks.log
        c0, r0, width, height = plan.input_box
        whole = plan.input_box == (0, 0, session.video.width, session.video.height)
        set_view = getattr(segmenter, "set_view", None)  # ExactFake is told what each image shows
        frames = plan.frames
        log(f"Run {number} of {count}: {', '.join(plan.track_ids)} ({plan.mode}), frames {frames[0]} to {frames[-1]} "
            f"every {frames.step} ({len(frames)} frames). The model sees "
            + ("the whole frame" if whole else f"columns {c0} to {c0 + width - 1}, rows {r0} to {r0 + height - 1}")
            + f" ({width} x {height} px).")

        record = RunRecord(tracks=list(plan.track_ids), start_frame=plan.start_frame, mode=plan.mode, started=_now())
        index = len(session.runs)
        session.runs.append(record)
        self._hand_over(record, index, complete=False)
        status, tracked, began, frame = COMPLETE, 0, time.perf_counter(), plan.start_frame
        try:
            with closing(iter_rgb_frames(self.video_path, frames)) as decoded:  # closed at once: Windows locks the file
                for frame, rgb in decoded:
                    if callbacks.should_cancel():
                        status = CANCELLED
                        break
                    # without a crop the model gets the decoded frame itself, as last week (SPEC 6.2)
                    image = rgb if whole else np.ascontiguousarray(rgb[r0:r0 + height, c0:c0 + width])
                    if set_view is not None:
                        set_view(frame, (c0, r0), (width, height))
                    results = segmenter.start(image, prompts) if tracked == 0 else segmenter.step(image)
                    for track_id, result in zip(plan.track_ids, results, strict=True):
                        in_frame = replace(result, offset=(result.offset[0] + c0, result.offset[1] + r0))
                        measured = measure_mask(in_frame, frame, plan.input_box, plan.mode,
                                                session.processing.core_open_frac)
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
        self._save(record, index, tracked, began, complete=status == COMPLETE and number == count)
        log(f"  {tracked} frames in {time.perf_counter() - began:.1f} s ({record.seconds_per_frame:.2f} s per frame)"
            + (f" on {record.device}." if record.device else "."))
        return status

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

    def _hand_over(self, record: RunRecord, index: int, complete: bool) -> None:
        """Give the run's record, with the device in use now, to the session's writer."""
        record.device = str(getattr(self.segmenter, "device", None) or "")
        record.model_id = getattr(self.segmenter, "model_id", None)
        record.weights_sha256 = getattr(self.segmenter, "weights_sha256", None)
        self.save_session(SessionChanges(complete=complete, run_index=index, run=copy.deepcopy(record)))


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
