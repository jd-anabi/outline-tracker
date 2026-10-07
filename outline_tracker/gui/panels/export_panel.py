"""Panel 9, Export (SPEC 8.1, 10.1, 10.2): the run folder, Export all, what it wrote, Open folder.

Export all is `export.export_all(run_folder, overlay=True)`: every output file of SPEC 8 from
session.json and results.npz, with overlay.mp4. It runs as one task in the window's worker thread
(`Worker.run`), after what that thread is doing, so the window stays usable. It needs no model: it
starts, runs and ends the same while a model is still loading (which another thread does) and
after a model could not be loaded. While it runs, `Jobs.writing()` says so to every panel: what a tracking run
switches off is off during an export too. This module is the panel's controls, and what they say:

- The run folder's name (the whole path is its tooltip) and Change folder, which is the File
  menu's Save session as.
- Export all, and File > Export, which is the same function. Both are off, with the reason as the
  hint line and as the button's tooltip, while an export or a tracking run is going, while the
  name or the video is missing, while nothing is tracked, and while the session has no fps_true or
  no scale. Before a video is open the hint line stays the window's own.
- Before the task is handed over, here in the GUI thread: `controller.refusal("export")`, and
  `controller.save_now()`, because the export reads session.json from the disk. A session that
  could not be saved there is not exported from the older file: the panel says why.
- While it runs: a busy bar (no invented progress) and the line `export_all` logged last.
- After it: the files written with their sizes, the time it took, and every warning of the
  export's report (a locked file whose data went to `<name>.new<ext>`, an overlay that was
  skipped). A failure is one plain sentence in the panel and in a dialog (`dialogs.message`); its
  trace goes to run.log with the worker's others (`Jobs.write_traces`), never to the window.
- The panel is done when positions.csv is there and neither results.npz nor session.json is
  newer; it needs attention when one of them is (export again: an export saves the session first,
  so a fresh export is never older), after a failure, and after an export with warnings. What the
  last export said is shown until the results change or the next export starts.
- Open folder shows the run folder in the system's file browser (`show_folder`).

No measurement is made here. Sizes of files are bytes, written in kB, MB (1 kB = 1000 bytes, as the
command line writes them); times are s, a time of day is the computer's local time; lengths of
widgets are Qt's device-independent px.
"""

from __future__ import annotations

import numbers
import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QWidget

from outline_tracker import schema
from outline_tracker.cli_export import UNUSABLE, WARNING_PREFIX, one_line, size_text
from outline_tracker.export import ExportReport, export_all
from outline_tracker.gui import dialogs, estimate, theme
from outline_tracker.gui.panels.calibration_panel import CONTROL_HEIGHT, DASH, SPACING, Message, button, column
from outline_tracker.gui.panels.track_panel import PRIMARY_HEIGHT
from outline_tracker.gui.panels.video_panel import ElidedLabel, field_rows
from outline_tracker.gui.tools import mm_per_px
from outline_tracker.gui.worker import traced, worker_of
from outline_tracker.gui.worker_jobs import jobs_of

EXPORT_TIP = "Write the CSV files, the overlay video, the log and README.txt to the run folder"
OPEN_TIP = "Show the run folder in the system's file browser"
FOLDER_TIP = "Go on in another run folder: the session and the results are copied there (File > Save session as)"
# Why Export all is off: each says what to do first.
EXPORTING = "Export all is running. You can look at other frames meanwhile."
TRACKING = "Tracking is running. Export all is ready when tracking has stopped."
NO_RESULTS = "Nothing is tracked yet. Click Track in panel 7 (Track) first."
NO_FPS = "No frame rate is set. Type fps_true in panel 2 (Time). Export needs it to write times in s."
NO_SCALE = ("No scale is set. Place the stick in panel 3 (Calibration). Export needs the scale to write positions "
            "in mm.")
NOT_SAVED = "The session could not be saved to the run folder, so nothing was exported."
NO_FOLDER = "There is no run folder yet. Type your name and open your video in panel 1 first."
NOT_SHOWN = "The run folder could not be opened. Point at its name in panel 9 (Export) to see where it is."
# The hint line once something was exported, or should be.
EXPORTED = "Exported at {time}. Export all replaces these files."
STALE = "The results changed after the last export. Click Export all again."
SESSION_STALE = "The session changed after the last export. Click Export all again."
WARNINGS_HINT = ("The export has 1 warning. Read it below.", "The export has {n} warnings. Read them below.")
FAILED_HINT = "Export all stopped. See the message below."
# While an export runs, and how it ended.
WRITING = "Writing the files to the run folder {folder}. The overlay video takes the longest."
WROTE = "Export all wrote {files} in {time}{warnings}."
STOPPED = "Export all stopped with an error."
# The dialog: the first line says what happened, the lines after it what to do next.
STOPPED_DIALOG = ("Export all stopped with an error\n{reason}\nFiles that were written before it stopped are kept. "
                  "The details are in run.log in the run folder.")
IN_FOLDER = "Export all in the run folder {folder}."  # the line above a failure's trace in run.log


def show_folder(folder) -> bool:
    """Show the folder at `folder` (a path) in the system's file browser: Explorer, Finder, or
    what the desktop has. Returns whether the system took it. The one way out to the file browser."""
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))


def duration(seconds: float) -> str:
    """A time an export took, in s, in words: "2.5 s" under a minute, then "1 min 5 s"."""
    return f"{seconds:.1f} s" if seconds < 59.95 else estimate.time_left(seconds)


def sentence_of(error: BaseException, folder: Path) -> str:
    """What stopped an export in the run folder `folder`, on one line, as the command line says it
    (`cli_export.run`): the package's own message for what it words itself, with the folder's name
    in place of its path; for anything else `cli_export.UNUSABLE` with the error's kind."""
    text = one_line(str(error))
    if text and isinstance(error, (OSError, RuntimeError, ValueError, cv2.error)):
        return text.replace(str(folder), folder.name)
    return UNUSABLE.format(reason=type(error).__name__ + (f": {text}" if text else ""))


class _Task:
    """One export as the worker thread runs it. It reports through the panel's private signals,
    which the GUI thread's event loop takes over, and raises nothing."""

    def __init__(self, panel: ExportPanel, folder: Path):
        self._panel, self._folder = panel, folder

    def __call__(self, segmenter) -> None:
        """Write every output file of the run folder, with the overlay; the model is not used."""
        panel, folder = self._panel, self._folder
        try:
            report = export_all(folder, overlay=True, log=panel._said.emit)
        except Exception as error:  # whatever stops an export: one sentence for the panel, the trace for run.log
            panel._ended.emit(None, sentence_of(error, folder), traced(IN_FOLDER.format(folder=folder.name)))
            return
        panel._ended.emit(report, "", "")


class ExportPanel(QWidget):
    """The controls of panel 9. Parts: `folder_label` (the run folder's name; its tooltip is the
    whole path), `folder_button` (Change folder), `progress_bar` and `progress_label` (while an
    export runs), `message` (how the last export ended, with its warnings; or why one did not
    start), `files_label` (the files it wrote, one a line, with their sizes), `export_button`,
    `open_button`, and `export_action`, the File menu's Export. `worker` is the window's `Worker`,
    `jobs` its `Jobs`.

    `exporting`: an export was handed to the worker and has not reported its end (kept by `jobs`,
    which tells the other panels). `report`: the
    `ExportReport` of the last export that ran to its end, None before one did and after one that
    stopped. `seconds`: how long the last export took, s. `clock`: gives the time in s."""

    _said = Signal(str)                # a line `export_all` logged
    _ended = Signal(object, str, str)  # the report, or None with the plain reason of a failure and its trace

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[8]
        self.worker, self.jobs = worker_of(window), jobs_of(window)
        self._start_hint = self._panel.hint.text()  # what Export all does: the line until something is exported
        self.report: ExportReport | None = None
        self.seconds: float | None = None
        self.clock = time.perf_counter
        self._task: _Task | None = None
        self._folder: Path | None = None              # the run folder the export writes into
        self._started_at = 0.0
        self._line = ""                               # what the running export said last
        self._refused: str | None = None              # why the last press did not start an export
        self._outcome: tuple[str, str] | None = None  # kind and text of the line about the last export
        self._outcome_of: tuple | None = None         # the results that export was of (`_results_key`)
        self._files: list[str] = []                   # the lines of `files_label`

        self.folder_label = ElidedLabel()
        self.folder_button = button("Change folder", FOLDER_TIP)
        self.folder_button.setFixedHeight(CONTROL_HEIGHT)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # busy: an export does not say how far it is
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        self.message = Message()
        self.files_label = QLabel()
        self.files_label.setProperty("role", "hint")
        self.files_label.setWordWrap(True)
        self.files_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # never wider
        self.export_button = button("Export all", EXPORT_TIP)
        self.open_button = button("Open folder", OPEN_TIP)
        self.open_button.setFixedHeight(CONTROL_HEIGHT)  # `refresh` makes one of the two the primary button
        # the name has the row to itself, as in panel 1: beside a button it would be cut to a few letters
        folder_rows = field_rows((("Run folder", self.folder_label),))
        folder_rows.addWidget(self.folder_button, 1, 1, Qt.AlignmentFlag.AlignLeft)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        column(self, folder_rows, self.progress_bar, self.progress_label, self.message, self.files_label, buttons)

        # the File menu's items that this panel has the functions for (gui/menus.py made them)
        self.export_action, self._save_as = window.menus.export_action, window.menus.save_as_action
        self.export_button.clicked.connect(self.export)
        self.export_action.triggered.connect(self.export)
        self.open_button.clicked.connect(self.open_folder)
        self.folder_button.clicked.connect(self._save_as.trigger)
        self._save_as.changed.connect(self._follow_save_as)
        self._said.connect(self._on_said)
        self._ended.connect(self._on_ended)
        controller, jobs = self._controller, self.jobs
        controller.video_opened.connect(self._session_changed)
        controller.session_changed.connect(self._session_changed)
        controller.saved.connect(self.refresh)
        jobs.started.connect(self.refresh)
        jobs.finished.connect(self.refresh)
        self.worker.state_changed.connect(self._worker_state)
        self._follow_save_as()
        self.refresh()

    # ------------------------------------------------------------------ the buttons

    exporting = property(lambda self: self.jobs.exporting, doc="Whether an export is going (`Jobs.exporting`).")

    def refusal(self) -> str | None:
        """Why no export can start now, as one sentence that says what to do first; None when one
        can. In this order: an export runs; tracking runs; the student's name or the video is
        missing (`controller.refusal`); results.npz cannot be read, or holds no track of the
        session; fps_true is missing; the scale is missing. The model is not asked for."""
        off = self._off()
        return None if off is None else off[0]

    def export(self, *_) -> None:
        """Export all (the button, and File > Export): save the session, then write every output
        file of the run folder, with the overlay, in the worker thread. Returns at once. What
        cannot start is said in the panel, and nothing starts."""
        if self.exporting:
            return
        controller, task, folder = self._controller, None, None
        refused = self.refusal()
        if refused is None and (controller.save_now() is None or controller.save_problem is not None):
            refused = controller.save_problem or NOT_SAVED  # the export would read an older session.json
        if refused is None:
            folder = Path(controller.run_folder)
            task = _Task(self, folder)
            if not self.worker.run(task, needs_model=False):
                return  # the worker has stopped: the window is closing
        if refused is not None:
            self._refused = refused
            self._window.statusBar().showMessage(refused)
            self.refresh()
            return
        self._task, self._folder, self._line = task, folder, WRITING.format(folder=folder.name)
        self.report, self.seconds = None, None
        self._refused = self._outcome = self._outcome_of = None
        self._files = []
        self._started_at = self.clock()
        self.jobs.set_exporting(True)  # the lock: every panel is told
        self.refresh()

    def open_folder(self) -> None:
        """Show the run folder in the system's file browser (the Open folder button). Without a
        run folder on the disk nothing happens."""
        folder = self._controller.run_folder
        if folder is None or not Path(folder).is_dir():
            return
        if not show_folder(Path(folder)):
            self._window.statusBar().showMessage(NOT_SHOWN)

    # ------------------------------------------------------------------ what the export reports

    def _on_said(self, line: str) -> None:
        if self.exporting and not line.startswith(WARNING_PREFIX):  # the warnings are listed at the end
            self._line = one_line(line)
            self.progress_label.setText(self._line)

    def _on_ended(self, report: ExportReport | None, reason: str, trace: str) -> None:
        self.worker.keep_trace(trace)
        folder, self._task = self._folder, None
        self.seconds = self.clock() - self._started_at
        if report is None:
            self._outcome = said = ("problem", f"{STOPPED} {reason}")
            dialogs.message(self._window, "problem", STOPPED_DIALOG.format(reason=reason))
        else:
            self.report, self._files = report, [_file_line(path, folder) for path in report.files]
            warned = f", with {estimate.counted(len(report.warnings), 'warning')}" if report.warnings else ""
            took = WROTE.format(files=estimate.counted(len(report.files), "file"), time=duration(self.seconds),
                                warnings=warned)
            said = ("warning" if report.warnings else "success", took)
            self._outcome = (said[0], "\n\n".join([took, *report.warnings]))
        self._outcome_of = _results_key(folder)
        self._window.statusBar().showMessage(said[1])
        self.jobs.write_traces()  # a failure's trace goes to run.log now
        self.jobs.set_exporting(False)
        self.refresh()

    def _worker_state(self, state: str, message: str) -> None:
        if state == "stopped" and self.exporting:  # the window closes: take over what the export reported
            QCoreApplication.sendPostedEvents(self, QEvent.Type.MetaCall.value)
            self.jobs.set_exporting(False)
        self.refresh()

    def _session_changed(self) -> None:
        self._refused = None  # it was about the session as it was
        self.refresh()

    def _follow_save_as(self) -> None:
        self.folder_button.setEnabled(self._save_as.isEnabled())  # Change folder is that item

    # ------------------------------------------------------------------ showing

    def _off(self) -> tuple[str, str | None] | None:
        """(why Export all is off, the panel's state with it) or None when it can start; the state
        is None where it is the state of what is in the run folder (`_exported`)."""
        controller, session = self._controller, self._controller.session
        if self.jobs.writing() is not None:
            return (EXPORTING if self.exporting else TRACKING), self._panel.state
        missing = controller.refusal("export")
        if missing is not None:
            return missing, "todo"
        try:
            stored = self.jobs.results.now().track_ids
        except ValueError as unreadable:
            return sentence_of(unreadable, Path(controller.run_folder)), "attention"
        if not any(track.id in stored for track in session.tracks):
            return NO_RESULTS, "todo"
        fps = session.time.fps_true
        if not isinstance(fps, numbers.Real) or isinstance(fps, bool) or not fps > 0:
            return NO_FPS, "attention"
        return (NO_SCALE, "attention") if mm_per_px(session) is None else None

    def _exported(self) -> tuple[str, str]:
        """(the hint line, the panel's state) from how the last export of this window ended and from
        the run folder: done when positions.csv is there and neither results.npz nor session.json
        is newer (the files' times of change are compared)."""
        kind = None if self._outcome is None else self._outcome[0]
        if kind == "problem":
            return FAILED_HINT, "attention"
        if kind == "warning":
            count = len(self.report.warnings)
            return WARNINGS_HINT[count > 1].format(n=count), "attention"
        folder = self._controller.run_folder
        try:
            written, tracked, saved = ((Path(folder) / name).stat().st_mtime_ns for name in (
                schema.POSITIONS_CSV, schema.RESULTS_NPZ, schema.SESSION_JSON))
        except (OSError, TypeError):  # nothing was exported yet; or there is no run folder
            return self._start_hint, "todo"
        if max(tracked, saved) > written:  # the exported files are older than what they come from
            return (STALE if tracked > written else SESSION_STALE), "attention"
        return EXPORTED.format(time=f"{datetime.fromtimestamp(written / 1e9):%H:%M}"), "done"

    def refresh(self, *_) -> None:
        """Show where the export stands: the run folder, the buttons with the reason why one is
        off, the bar and its line while an export runs, how the last one ended, and the panel's
        hint line and state."""
        controller, folder = self._controller, self._controller.run_folder
        self.folder_label.set_full_text(DASH if folder is None else Path(folder).name,
                                        "" if folder is None else str(folder))
        on_disk = folder is not None and Path(folder).is_dir()
        self.open_button.setEnabled(on_disk)
        self.open_button.setToolTip(OPEN_TIP if on_disk else NO_FOLDER)
        if self._outcome is not None and not self.exporting and self._outcome_of != _results_key(folder):
            self._outcome, self._outcome_of, self._files, self.report = None, None, [], None  # of other results
        off = self._off()
        reason, state = (None, None) if off is None else off
        hint = reason
        if controller.session is None:
            hint = state = None  # before a video is open the hint line and the state are the window's own
        elif state is None:
            told, state = self._exported()
            hint = hint or told
        if self._refused is not None and state is not None and not self.exporting:
            state = "attention"
        self.export_button.setEnabled(reason is None)
        self.export_action.setEnabled(reason is None)
        self.export_button.setToolTip(EXPORT_TIP if reason is None else reason)
        for made, primary in ((self.export_button, state != "done"), (self.open_button, state == "done")):
            if (made.property("kind") == "primary") != primary:  # the next step: Open folder once it is exported
                theme.set_property(made, "kind", "primary" if primary else "")
                made.setFixedHeight(PRIMARY_HEIGHT if primary else CONTROL_HEIGHT)
        self.progress_bar.setVisible(self.exporting)
        self.progress_label.setVisible(self.exporting)
        self.progress_label.setText(self._line if self.exporting else "")
        kind, text = ("problem", self._refused) if self._refused is not None else self._outcome or (None, "")
        self.message.show_message(None if self.exporting else kind, text)
        self.files_label.setText("\n".join(self._files))
        self.files_label.setVisible(bool(self._files) and not self.exporting)
        if state is not None and self._panel.state != state:
            self._panel.set_state(state)
        if hint is not None and self._panel.hint.text() != hint:
            self._panel.set_hint(hint)


def _file_line(path: Path, folder: Path) -> str:
    """One line of the list of files written: the name relative to the run folder, and the size."""
    try:
        size = size_text(path.stat().st_size)
    except OSError:  # gone again already
        size = DASH
    return f"{path.relative_to(folder).as_posix()} · {size}"


def _results_key(folder) -> tuple | None:
    """What tells one state of the results of the run folder `folder` from another: the folder,
    and when results.npz was written and how large it is; None without a folder or the file."""
    try:
        found = (Path(folder) / schema.RESULTS_NPZ).stat()
    except (OSError, TypeError):
        return None
    return str(folder), found.st_mtime_ns, found.st_size


def build(window) -> QWidget:
    """The controls of panel 9 for `window` (a `MainWindow`): an `ExportPanel`, connected to the
    window's controller, worker and `Jobs`, and to the File menu's Export and Save session as."""
    return ExportPanel(window)
