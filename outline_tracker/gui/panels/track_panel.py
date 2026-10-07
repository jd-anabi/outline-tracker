"""Panel 7, Track (SPEC 6.1, 6.4, 10.1, 10.2): the estimated time before a run, Track, the progress
bar with s per frame and the time left, Cancel, and one line that says how the run ended.

The job itself is `tracking.run_job` in the worker thread (gui/worker_jobs.py, `Jobs`); what it
stored is drawn on the picture by gui/overlays.py, which this panel makes. This module is the
panel's controls:

- Before a run the hint line gives the estimate (gui/estimate.py) and how many frames and objects
  it is for. Track is off, with the reason as the hint line and as its tooltip, while something is
  missing (`Jobs.refusal`), and while a run is going.
- During a run: the bar counts the tracked frames of all runs, and the line under it says the run,
  the frame, s per frame and the time left; the line is written at most 4 times a second. Cancel
  stops after the current frame.
- After a run one line says what happened: complete, cancelled with what was kept, or the plain
  reason of a failure. A failure is also shown in a dialog (`dialogs.message`); so is a model that
  could not be loaded and an outline that could not be made. A trace is never shown: it goes to
  run.log (`Jobs.write_traces`).

The time one frame takes (`seconds_per_frame`, for one object) is taken from what this computer did
last: the session's newest run, a run in this window, or the outline made after a click.

Units: times are s; frames are counts of tracked frames, and video frame numbers where a text
names a frame. Lengths of widgets are Qt's device-independent px.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from outline_tracker.geometry import grid_frames
from outline_tracker.gui import dialogs, estimate
from outline_tracker.gui.overlays import Overlays
from outline_tracker.gui.panels.calibration_panel import CONTROL_HEIGHT, SPACING, Message, button, column
from outline_tracker.gui.worker import NO_FACTORY, worker_of
from outline_tracker.gui.worker_jobs import NOTHING_LEFT, RUNNING, STOPPED, jobs_of
from outline_tracker.results import ResultsStore
from outline_tracker.tracking import CANCELLED, COMPLETE, partial_tracks

PRIMARY_HEIGHT = 32
LINE_EVERY_MS = 250  # the progress line is written at most 4 times a second

TRACK_TIP = "Track the objects that have points (panel 6)"
CANCEL_TIP = "Stop after the current frame. The tracked frames are kept."
ESTIMATE = "Estimated time: {time} for {frames} and {objects}."
NO_ESTIMATE = "No time estimate yet: {frames} and {objects} are ready to track."
RUNNING_HINT = "Tracking is running. You can look at other frames meanwhile."
STOPPING = "Stopping after the current frame."
MODEL_LOADING = "The model is loading."
COMPLETE_TEXT = "Tracking is complete: {objects}, {frames}."
NOT_COMPLETE = "Tracking is not complete: {ids} before the end of the clip."
CANCELLED_AT = "Tracking was cancelled at frame {frame}. The {frames} were kept."
CANCELLED_TEXT = "Tracking was cancelled. The {frames} were kept."
# Dialogs: the first line says what happened, the lines after it what to do next.
JOB_DIALOG = ("Tracking stopped with an error\n{reason}\nFrames that were tracked before are kept. The details "
              "are in run.log in the run folder.")
MODEL_DIALOG = ("The model could not be loaded\n{reason}\nTracking and outlines need the model. Check the "
                "internet connection, which the first start needs, and start the app again.")
OUTLINE_DIALOG = ("The outline could not be made\n{reason}\nClick on the animal again. If it happens again, the "
                  "details are in run.log in the run folder.")


class TrackPanel(QWidget):
    """The controls of panel 7. Parts: `progress_bar`, `progress_label` (the line under it),
    `message` (how the last run ended), `track_button`, `cancel_button`. `jobs` is the window's
    `Jobs`, `worker` its `Worker`, `overlays` the `Overlays` that draw the results on the picture.
    `seconds_per_frame`: s per tracked frame of one object on this computer, None while nothing
    was timed. `clock`: gives the time in s, for timing an outline."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[6]
        self.worker, self.jobs = worker_of(window), jobs_of(window)
        self.overlays = Overlays(window, self.jobs)
        self.seconds_per_frame: float | None = None
        self.clock = time.perf_counter
        self._asked_at: float | None = None        # when the outline that is on its way was asked for
        self._outcome: tuple[str, str] | None = None  # kind and text of the line about the last run

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        font = self.progress_label.font()
        font.setFeature(QFont.Tag("tnum"), 1)  # digits of one width: the line does not jump while it counts
        self.progress_label.setFont(font)
        self.message = Message()
        self.track_button = button("Track", TRACK_TIP)
        self.track_button.setProperty("kind", "primary")
        self.track_button.setFixedHeight(PRIMARY_HEIGHT)
        self.track_button.clicked.connect(self.track)
        self.cancel_button = button("Cancel", CANCEL_TIP)
        self.cancel_button.setFixedHeight(CONTROL_HEIGHT)
        self.cancel_button.clicked.connect(self.cancel)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        buttons.addWidget(self.track_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        column(self, self.progress_bar, self.progress_label, self.message, buttons)

        self._line_timer = QTimer(self)
        self._line_timer.setInterval(LINE_EVERY_MS)
        self._line_timer.timeout.connect(self._show_progress)

        controller, jobs, worker = self._controller, self.jobs, self.worker
        controller.video_opened.connect(self._video_opened)
        controller.session_changed.connect(self.refresh)
        controller.saved.connect(self.refresh)
        self._panel.header.toggled.connect(self.refresh)  # a name typed before a video is open says nothing
        worker.state_changed.connect(self._model_state)
        worker.busy_changed.connect(self._busy_changed)
        worker.preview_done.connect(self._preview_done)
        worker.preview_failed.connect(self._preview_failed)
        jobs.started.connect(self._started)
        jobs.progress.connect(self._progress)
        jobs.finished.connect(self._finished)
        self.refresh()

    # ------------------------------------------------------------------ the buttons

    def track(self) -> None:
        """Start tracking the pending objects (the Track button). What cannot start is said in the
        panel, and nothing starts."""
        refused = self.jobs.start()
        if refused is not None:
            self._outcome = ("problem", refused)
        self.refresh()

    def cancel(self) -> None:
        """Stop the run after the current frame (the Cancel button); what was tracked is kept."""
        self.jobs.cancel()
        self.refresh()

    # ------------------------------------------------------------------ what the worker and the job report

    def _video_opened(self) -> None:
        self._outcome = None
        session = self._controller.session
        measured = None if session is None else estimate.last_run_seconds(session.runs)
        if measured is not None:
            self.seconds_per_frame = measured
        self.refresh()

    def _model_state(self, state: str, message: str) -> None:
        if state == "failed" and message != NO_FACTORY:  # a window made without a model has nothing that failed
            dialogs.message(self._window, "problem", MODEL_DIALOG.format(reason=message))
        self.refresh()

    def _busy_changed(self, busy: bool) -> None:
        # an outline asked for while the model loads would be timed with the loading
        self._asked_at = self.clock() if busy and self.worker.ready else None

    def _preview_done(self, preview) -> None:
        if self._asked_at is not None and preview.results and not self.jobs.running:
            self.seconds_per_frame = estimate.seconds_per_object_frame(self.clock() - self._asked_at,
                                                                       len(preview.results))
        self._asked_at = None
        self.refresh()

    def _preview_failed(self, serial: int, reason: str) -> None:
        if self.worker.state != "failed":  # else the dialog about the model has said it
            dialogs.message(self._window, "problem", OUTLINE_DIALOG.format(reason=reason))

    def _started(self) -> None:
        self._outcome = None
        self._line_timer.start()
        self.refresh()

    def _progress(self, done: int, total: int, s_per_frame: float, eta_s: float) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _finished(self, status: str, reason: str) -> None:
        self._line_timer.stop()
        jobs, session = self.jobs, self._controller.session
        measured = None if session is None else estimate.last_run_seconds(session.runs)
        if measured is not None and jobs.done:
            self.seconds_per_frame = measured
        frames = estimate.counted(jobs.done, "tracked frame")
        if status == COMPLETE:
            self._outcome = ("success", self._complete_text())
        elif status == CANCELLED:
            at = jobs.stopped_at()
            self._outcome = ("warning", CANCELLED_TEXT.format(frames=frames) if at is None
                             else CANCELLED_AT.format(frame=at, frames=frames))
        else:
            self._outcome = ("problem", f"{STOPPED} {reason}")
            dialogs.message(self._window, "problem", JOB_DIALOG.format(reason=reason))
        self.refresh()

    # ------------------------------------------------------------------ showing

    def _complete_text(self) -> str:
        """ "Tracking is complete: 3 objects, 60 frames.": the tracked objects of the session and
        the frames of its clip."""
        session, store = self._controller.session, self._results()
        objects = sum(track.id in store.track_ids for track in session.tracks)
        frames = len(grid_frames(session.clip.start, session.clip.end, session.clip.step))
        return COMPLETE_TEXT.format(objects=estimate.counted(objects, "object"),
                                    frames=estimate.counted(frames, "frame"))

    def _results(self) -> ResultsStore:
        """The results on disk now; none when the file cannot be read as results."""
        try:
            return self.jobs.results.now()
        except ValueError:
            return ResultsStore()

    def _standing(self) -> tuple[str | None, str, str]:
        """(why Track is off or None, the hint line, the panel's state) while no job runs."""
        refused = self.jobs.refusal()
        if refused is None:
            plans = self.jobs.pending()
            frames, objects = estimate.frames_and_objects(plans)
            words = {"frames": estimate.counted(frames, "frame"), "objects": estimate.counted(objects, "object")}
            if self.seconds_per_frame is None:
                return None, NO_ESTIMATE.format(**words), "todo"
            seconds = estimate.estimate_seconds(self.seconds_per_frame, plans)
            return None, ESTIMATE.format(time=estimate.about(seconds), **words), "todo"
        if refused != NOTHING_LEFT:
            return refused, refused, "todo"
        session = self._controller.session
        try:
            partial = partial_tracks(session, self._results())
        except ValueError as error:
            return str(error), str(error), "attention"
        if partial:
            ids = (f"{partial[0]} stops" if len(partial) == 1
                   else f"{', '.join(partial[:-1])} and {partial[-1]} stop")
            text = NOT_COMPLETE.format(ids=ids)
            return text, text, "attention"
        text = self._complete_text()
        return text, text, "done"

    def _show_progress(self) -> None:
        jobs = self.jobs
        if jobs.cancelling:
            self.progress_label.setText(STOPPING)
        elif jobs.done:
            self.progress_label.setText(estimate.progress_text(jobs.run_number, len(jobs.plans), jobs.done,
                                                               jobs.total, jobs.s_per_frame, jobs.eta_s))
        else:
            self.progress_label.setText(f"Run 1 of {len(jobs.plans)} · frame 0 of {jobs.total}")

    def refresh(self, *_) -> None:
        """Show where tracking stands: the hint line with the estimate or the reason why Track is
        off, the buttons, the bar and its line during a run, and the line about the last run."""
        jobs, loading = self.jobs, self.worker.state == "loading"
        if jobs.running:
            reason, hint, state = RUNNING, RUNNING_HINT, self._panel.state
            self.progress_bar.setRange(0, jobs.total)
            self.progress_bar.setValue(jobs.done)
            self._show_progress()
        else:
            reason, hint, state = self._standing()
            if self._outcome is not None and self._outcome[0] != "success" and state == "todo":
                state = "attention"
            if loading:
                self.progress_bar.setRange(0, 0)  # busy: no invented progress
                self.progress_label.setText(MODEL_LOADING)
        self.progress_bar.setVisible(jobs.running or loading)
        self.progress_label.setVisible(jobs.running or loading)
        self.track_button.setEnabled(reason is None)
        self.track_button.setToolTip(TRACK_TIP if reason is None else reason)
        self.cancel_button.setEnabled(jobs.running and not jobs.cancelling)
        kind, text = self._outcome or (None, "")
        self.message.show_message(kind, text, self._window.statusBar())
        if self._panel.state != state:
            self._panel.set_state(state)
        if self._panel.hint.text() != hint:
            self._panel.set_hint(hint)


def build(window) -> QWidget:
    """The controls of panel 7 for `window` (a `MainWindow`): a `TrackPanel`. Making it also makes
    the window's `Jobs` and the `Overlays` that draw the results on the picture."""
    return TrackPanel(window)
