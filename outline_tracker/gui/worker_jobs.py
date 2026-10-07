"""Tracking jobs in the worker thread (SPEC 6.1, 6.4, 10.2; decision X7): Track runs
`tracking.run_job` there, with the model the worker loaded, so the window stays usable.

`Jobs` lives in the GUI thread, one per window (`jobs_of`). `start()` checks what must be there,
saves the session, and hands the worker one task; the worker thread runs it after what it is
doing, and an outline asked for meanwhile waits until the job has ended (one job or outline at a
time). The task calls `run_job` with a copy of the session, made in the GUI thread.

Who writes what. The worker thread writes results.npz and nothing else. Every change a job makes
to the session (`tracking.SessionChanges`: a run's record, whether the results are complete, a
frame hash, a fine window) crosses to the GUI thread as the value of a signal, is applied to
`controller.session` there, and is saved there. The task reports through private signals of the
`Jobs` object, which are connected to its own bound methods: emitted in the worker thread, they
are taken over by the GUI thread's event loop, and every public signal is emitted in the GUI thread.

Cancel is a `threading.Event` that `run_job` asks before every frame: the frame the model is in is
finished and stored, the next one is not started, and what was tracked stays (`"complete": false`).
Closing the window does the same through the worker's `stopping`, waits for the thread, and then
takes over what the job still had to say, so that the session on disk has the run as it ended.

A failure (loading the model, an outline, a job) is said in one plain line by whoever shows it;
its trace is kept by the worker and appended here to `<run folder>/run.log` as soon as there is a
run folder (SPEC 10.2).

Units: frames are counts of tracked frames, or video frame numbers where a name says so; times
are s. This module does not import torch.
"""

from __future__ import annotations

import copy
import numbers
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal

from outline_tracker import schema
from outline_tracker.fileio import append_block
from outline_tracker.gui import worker as worker_module
from outline_tracker.gui.click_rules import ResultsOnDisk
from outline_tracker.gui.worker import REASON_LENGTH, plain, traced
from outline_tracker.tracking import CANCELLED, FAILED, Callbacks, Job, SessionChanges, plan_runs, run_job

RUNNING = "Tracking is running."
NO_FPS = "Type the true frame rate (fps_true) in panel 2 (Time) first."
NO_OBJECT = "Add an object in panel 6 first."
NOTHING_LEFT = "Every object that has points is tracked already."
LOADING = "The model is loading. Track is ready when the model is ready."
NO_MODEL = "The model could not be loaded. {reason}"
NOT_SAVED = "The session could not be saved to the run folder, so tracking did not start."
STOPPED = "Tracking stopped with an error."
TRACE_STARTS = "Traceback (most recent call last):"  # how the entry with a failed run's trace begins


def failure_of(lines: list[str]) -> str:
    """The plain reason of a job that `run_job` ended as "failed", from the lines it logged: the
    first line of the entry before the trace, which says on which frame tracking failed and with
    what; without a trace, of the entry before the run's last one, which says what the video did
    not give. One line, cut to `REASON_LENGTH` characters."""
    before_trace = [index - 1 for index, text in enumerate(lines) if text.startswith(TRACE_STARTS) and index]
    said = lines[before_trace[-1]] if before_trace else lines[-2] if len(lines) >= 2 else ""
    said = said.strip().splitlines()
    return said[0].strip()[:REASON_LENGTH] if said else STOPPED


class _Task:
    """One job as the worker thread runs it. `cancel`: set to stop it after the current frame."""

    def __init__(self, jobs: Jobs, session, run_folder: Path, video_path: Path, stopping: threading.Event):
        self._jobs, self._session, self._run_folder, self._video_path = jobs, session, run_folder, video_path
        self._stopping, self.cancel = stopping, threading.Event()

    def __call__(self, segmenter) -> None:
        """Track with `segmenter` (the worker's loaded model) and report how it ended: in the
        worker thread. Nothing is raised."""
        jobs, lines = self._jobs, []
        callbacks = Callbacks(progress=jobs._progress.emit, frame_result=lambda track_id, frame, record: None,
                              log=lines.append, finished=lambda status: None,
                              should_cancel=lambda: self.cancel.is_set() or self._stopping.is_set(),
                              save_session=jobs._changed.emit)
        what = f"Tracking in the run folder {self._run_folder.name}."
        reason = trace = ""
        try:
            if segmenter is None:
                raise RuntimeError(NO_MODEL.format(reason="").strip())
            status = run_job(Job(self._session, self._run_folder, self._video_path, lambda: segmenter), callbacks)
            if status == FAILED:  # `run_job` logged the error and its trace, and kept what was tracked
                reason, trace = failure_of(lines), "\n".join([what, *lines])
                worker_module.log.error(trace)
        except Exception as error:  # what `run_job` refuses before it tracks, and whatever else is raised
            status, reason, trace = FAILED, plain(error), traced(what)
        jobs._ended.emit(status, reason, trace)


class Jobs(QObject):
    """Tracking for one window, from the GUI thread (see the module's text).

    `running`: a job was handed to the worker and has not reported its end. `cancelling`: it was
    asked to stop. `plans`: the runs of the job (`tracking.RunPlan`), `run_number` the one that is
    tracked (from 1). `done`, `total`: tracked frames and frames to track over all runs;
    `s_per_frame`, `eta_s`: s per tracked frame so far and s left. `status`, `reason`: how the last
    job ended ("complete", "cancelled", "failed"; "" before the first) and the plain reason of a
    failure. `results`: results.npz as it is on disk, for planning.

    Signals, all emitted in the GUI thread: `started()`; `progress(done, total, s_per_frame, eta_s)`
    after every tracked frame; `saved()` when the worker has written results.npz or a run has
    started, and the session has what the job changed; `finished(status, reason)`.
    """

    started = Signal()
    progress = Signal(int, int, float, float)
    saved = Signal()
    finished = Signal(str, str)
    _progress = Signal(int, int, float, float)
    _changed = Signal(object)
    _ended = Signal(str, str, str)

    def __init__(self, window, worker):
        super().__init__(window)
        self._controller, self.worker = window.controller, worker
        self.results = ResultsOnDisk(window.controller)
        self.running = self.cancelling = False
        self.plans: list = []
        self.run_number = self.done = self.total = 0
        self.s_per_frame = self.eta_s = 0.0
        self.status = self.reason = ""
        self._task: _Task | None = None
        self._session = None                         # the session the job is about
        self._first_run = 0                          # the place of the job's first run in `session.runs`
        self._last: SessionChanges | None = None     # the newest change of the job
        self._logged: list[str] = []                 # the traces that are in run.log
        self._progress.connect(self._on_progress)
        self._changed.connect(self._on_changed)
        self._ended.connect(self._on_ended)
        worker.state_changed.connect(self._worker_state)
        worker.preview_failed.connect(self.write_traces)
        worker.busy_changed.connect(self.write_traces)
        window.controller.saved.connect(self.write_traces)
        window.controller.video_opened.connect(self.cancel)

    # ------------------------------------------------------------------ before a job

    def pending(self) -> list:
        """The runs Track would start now (`tracking.plan_runs` of the session and the results on
        disk), none without a session. Raises ValueError, with a sentence for the user, for clicks
        or a clip that cannot be tracked, and for a results file of another version."""
        session = self._controller.session
        return [] if session is None else plan_runs(session, self.results.now())

    def refusal(self) -> str | None:
        """Why no job can start now, as one sentence that says what to do first; None when one can.
        In this order: a job runs; the student's name or the video is missing
        (`controller.refusal`); the model could not be loaded; fps_true is missing; the clicks or
        the clip cannot be tracked; nothing is left to track; the model is still loading."""
        controller, session = self._controller, self._controller.session
        if self.running:
            return RUNNING
        missing = controller.refusal("track")
        if missing is not None:
            return missing
        if self.worker.state in ("failed", "stopped"):
            return NO_MODEL.format(reason=self.worker.message).strip()
        fps = session.time.fps_true
        if not isinstance(fps, numbers.Real) or isinstance(fps, bool) or not fps > 0:
            return NO_FPS
        try:
            plans = self.pending()
        except ValueError as refused:
            return str(refused)
        if not plans:
            clicked = any(prompt.points_px for track in session.tracks for prompt in track.prompts)
            return NOTHING_LEFT if clicked else NO_OBJECT
        return None if self.worker.ready else LOADING

    def start(self) -> str | None:
        """Track the pending objects in the worker thread. Returns at once: None when the job was
        handed over (`started` was emitted, `finished` will be), else the sentence that says why
        nothing started (`refusal`, or a session that could not be saved). The session is saved
        first, here in the GUI thread."""
        refused = self.refusal()
        if refused is not None:
            return refused
        controller = self._controller
        if controller.save_now() is None:
            return controller.save_problem or NOT_SAVED
        session = controller.session
        self.plans = self.pending()
        self._session, self._first_run, self._last = session, len(session.runs), None
        self._task = _Task(self, copy.deepcopy(session), Path(controller.run_folder), Path(controller.video_path),
                           self.worker.stopping)
        self.running, self.cancelling, self.status, self.reason = True, False, "", ""
        self.run_number, self.done, self.total = 1, 0, sum(len(plan.frames) for plan in self.plans)
        self.s_per_frame = self.eta_s = 0.0
        self.worker.run(self._task)
        self.started.emit()
        return None

    def cancel(self) -> None:
        """Stop the job after the frame the model is in; what was tracked is kept. Without a job
        nothing happens."""
        if self.running and not self.cancelling:
            self.cancelling = True
            self._task.cancel.set()

    def stopped_at(self) -> int | None:
        """The first video frame that the last job did not track, from its plans and the record of
        its last run; None when it tracked every frame, or has not run."""
        if not self.plans:
            return None
        if self._last is None:
            return self.plans[0].start_frame
        index = self._last.run_index - self._first_run
        frames, tracked = self.plans[index].frames, self._last.run.frames_done
        if tracked < len(frames):
            return frames[tracked]
        return self.plans[index + 1].start_frame if index + 1 < len(self.plans) else None

    # ------------------------------------------------------------------ what the task reports

    def _on_progress(self, done: int, total: int, s_per_frame: float, eta_s: float) -> None:
        self.done, self.total, self.s_per_frame, self.eta_s = done, total, s_per_frame, eta_s
        self.progress.emit(done, total, s_per_frame, eta_s)

    def _on_changed(self, changes: SessionChanges) -> None:
        """Apply what the job changed to the session, here in the GUI thread; the session is then
        saved by its controller, as after any other change."""
        self._last = changes
        self.run_number = changes.run_index - self._first_run + 1
        if self._controller.session is self._session:
            changes.apply(self._session)
            self._controller.touch()
        self.saved.emit()

    def _on_ended(self, status: str, reason: str, trace: str) -> None:
        self.worker.keep_trace(trace)
        self.running, self.cancelling, self._task = False, False, None
        self.status, self.reason = status, reason
        if self._controller.session is self._session:
            self._controller.save_now()
        self.write_traces()
        self.finished.emit(status, reason)

    def _worker_state(self, state: str, message: str) -> None:
        """A worker that has stopped (the window closes) says nothing more by itself: take over
        what the job reported before its thread ended, so that the session is saved as it ended."""
        if state == "stopped" and self.running:
            QCoreApplication.sendPostedEvents(self, QEvent.Type.MetaCall.value)
            if self.running:  # the thread ended before the task's turn came
                self._on_ended(CANCELLED, "", "")
        self.write_traces()

    # ------------------------------------------------------------------ traces

    def write_traces(self, *_) -> None:
        """Append the traces the worker keeps and run.log does not hold yet to
        `<run folder>/run.log`, under a line with the time. Nothing is written before the run
        folder is there, nor into a folder whose session is another one's; it is tried again
        whenever the session was saved or the worker reports."""
        controller, folder = self._controller, self._controller.run_folder
        new = [trace for trace in self.worker.traces if not any(trace is old for old in self._logged)]
        if not new or folder is None or not Path(folder).is_dir() or controller.save_problem is not None:
            return
        when = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            append_block(Path(folder) / schema.RUN_LOG, [f"Errors noted in the window, {when}:", *new])
        except OSError:
            return  # the folder cannot be written now: the next save tries again
        self._logged = list(self.worker.traces)


def jobs_of(window) -> Jobs:
    """The `Jobs` of `window` (a `MainWindow`): made at the first call, with the window's worker,
    and kept as `window.jobs`, so that every panel works with the same one."""
    jobs = getattr(window, "jobs", None)
    if jobs is None:
        jobs = window.jobs = Jobs(window, worker_module.worker_of(window))
    return jobs
