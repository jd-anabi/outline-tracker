"""The flags table of panel 8 (SPEC 9: "a flags table (track, frame, t, code) whose rows jump to the
frame"): its rows, how they are shown, and how they are computed.

The rows are `tracking.flags_table(run_folder)`: one row per flag on a frame, by track and then by
frame, from session.json and results.npz as they are on disk. That function derives every track
again, which takes seconds for 10 tracks of 1,200 frames: far longer than a window may stand
still. So `Listing` never has it called in the GUI thread, and the window stays usable meanwhile.
It is called in the worker thread (`Worker.run`). The worker takes a task only while its model is
ready: when the model could not be loaded, the flags are listed in a thread of the listing's own,
which is started for the first such listing and ended when the window closes. The rows are asked
for again only when what they are made of has changed: results.npz (its size, time and file
number), whether session.json is there, or the parts of the session that the flags are derived
from (fps_true, the scale and the axes, the dish circle, the settings, and the id, head click and
start frame of each track that has results). A click on an animal changes none of these, so it
never makes the worker list the flags while it should outline the click.

Who reads and writes what. The task, in either thread, reads the two files and nothing else: it
is given the run folder as a path, never the session object, and it writes nothing. session.json
is written by the controller in the GUI thread, results.npz by a tracking job or a correction;
both are replaced in one step, so a read never meets half a file. `sync()` is therefore called
when the controller has saved, when a run has ended and when the worker's state changes; while a
run is going nothing is listed (the worker is busy with the run).

When the flags cannot be listed, `Listing.problem` says so in one plain sentence and there are no
rows: that is never "no flags". A refusal of the function (no scale, no fps_true) is its own
sentence. For any other error the sentence names run.log, and the error's trace goes there with
the worker's traces (`Worker.keep_trace`, SPEC 10.2): it is never shown.

Units: frames are video frame numbers; t is frame / fps_true in s, shown with 3 decimals. Row
heights are Qt's device-independent px. Nothing here imports torch.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QApplication, QHeaderView, QTableView

from outline_tracker import schema, tracking
from outline_tracker.gui import theme
from outline_tracker.gui.worker import STACK_BYTES, plain, traced, worker_of
from outline_tracker.gui.worker_jobs import jobs_of

COLUMNS = ("Track", "Frame", "t", "Flag")
# What the table lists by default: the flags on positions and MULTI. (`schema.POSITION_FLAGS` has MULTI with the
# shape flags; here it counts with the flags that say "look at this frame".)
DEFAULT_FLAGS = ("LOST", "JUMP", "SIZE", "CONTACT", "EDGE", "MULTI")
MORE_FLAGS = ("LOWRES", "ORIENT", "HEADGUESS")  # on nearly every frame of a small object: listed on request
ROWS_SHOWN, ROW_HEIGHT = 6, 24
WIDEST = ("Track", "000000", "000.000 s")  # what the first three columns have room for; the flag takes the rest
CELL_MARGIN = 8  # at each side of a cell's text
TABLE_TIP = "The flags of the tracked frames. Click a row to go to its frame (up and down: the row before, the next)."
# Why there are no rows, for the panel's hint line; and the line above an error's trace in run.log.
NOT_LISTED = "The flags could not be listed: {reason}. The details are in run.log in the run folder."
NOT_READ = "{name} could not be read"
NO_SESSION_FILE = (f"{schema.SESSION_JSON} is not in the run folder, so the flags cannot be listed. Save the session "
                   "(File > Save session).")
TRACE_HEAD = "The flags of the run folder {folder} could not be listed."

Row = tuple[str, int, float, str]  # track id, video frame number, t in s, flag code


def meaning(code: str) -> str:
    """What the flag `code` means for the student, in the words of SPEC 9, with "outline" for "mask"."""
    return schema.FLAG_INFO[code].meaning.replace("mask", "outline")


class FlagsModel(QAbstractTableModel):
    """The rows of the flags table for a view: `rows` holds (track id, video frame number, t in s,
    flag code). A cell shows the id, the frame, t as "0.250 s" or the code; its tooltip says what
    the code means. The rows of `frame`, the video frame the window shows, have another background
    (the theme's `accentSoft`), so that the flags of the frame shown can be told."""

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.rows: list[Row] = []
        self.frame: int | None = None

    def set_rows(self, rows: list[Row]) -> None:
        """Show `rows` in place of the rows before."""
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def set_frame(self, frame: int | None) -> None:
        """Mark the rows of video frame `frame` (None: no row) in place of those marked before."""
        if frame != self.frame:
            self.frame = frame
            if self.rows:
                self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, len(COLUMNS) - 1),
                                      [Qt.ItemDataRole.BackgroundRole])

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(COLUMNS)

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        track_id, frame, t_s, code = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return (track_id, str(frame), f"{t_s:.3f} s", code)[index.column()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{code}: {meaning(code)}"
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in (1, 2):  # numbers are right-aligned
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole and frame == self.frame:
            dark = QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark
            return QBrush(QColor(theme.tokens(dark)["accentSoft"]))
        return None


class FlagsView(QTableView):
    """The table that shows a `FlagsModel`: whole rows are selected, one at a time; six rows show
    and the rest scrolls inside; the first three columns are as wide as their longest text in the
    table's font and the flag takes the rest, so the table is never wider than its panel. Left,
    right, Home and End are not the table's: they stay the keys of the window's bottom bar.

    `row_chosen(row)` is emitted when the user clicks a row or reaches it with the up or down key,
    never when a row is chosen from the program (`choose`, `set_rows`)."""

    row_chosen = Signal(int)

    def __init__(self, model: FlagsModel):
        super().__init__()
        self._setting = False  # true while the program sets rows or the chosen row: no click then
        self.setModel(model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabKeyNavigation(False)  # Tab leaves the table, as it leaves every other control
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  # no row is measured: there are many
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        header = self.horizontalHeader()
        header.setFixedHeight(ROW_HEIGHT)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for column, text in enumerate(WIDEST):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(column, self.fontMetrics().horizontalAdvance(text) + 2 * CELL_MARGIN)
        self.setFixedHeight((ROWS_SHOWN + 1) * ROW_HEIGHT + 2 * self.frameWidth())
        self.setToolTip(TABLE_TIP)
        self.clicked.connect(self._by_user)
        self.selectionModel().currentRowChanged.connect(self._by_user)

    def chosen(self) -> int:
        """The chosen row, counted from 0; -1 for none."""
        return self.currentIndex().row()

    def choose(self, row: int) -> None:
        """Make `row` the chosen row; a row the table does not have (-1) chooses none."""
        if row == self.chosen():
            return
        self._setting = True
        try:
            if 0 <= row < self.model().rowCount():
                self.selectRow(row)
                self.scrollTo(self.model().index(row, 0))
            else:
                self.clearSelection()
                self.setCurrentIndex(QModelIndex())
        finally:
            self._setting = False

    def set_rows(self, rows: list[Row]) -> None:
        """Show `rows` in place of the rows before. The row that was chosen stays the chosen one
        if it is among them."""
        was = self.model().rows[self.chosen()] if self.chosen() >= 0 else None
        self._setting = True
        try:
            self.model().set_rows(rows)
        finally:
            self._setting = False
        if was in rows:
            self.choose(rows.index(was))

    def _by_user(self, index: QModelIndex, *_) -> None:
        if not self._setting and index.isValid():
            self.row_chosen.emit(index.row())


def listed(rows: list[Row], track_ids, only: str | None, shape_flags: bool) -> list[Row]:
    """Of `rows`, those the table lists: the rows of the tracks `track_ids` (of `only` alone
    unless it is None) whose flag is one of `DEFAULT_FLAGS`, or of `MORE_FLAGS` too when
    `shape_flags` is true. The order stays."""
    codes = DEFAULT_FLAGS + (MORE_FLAGS if shape_flags else ())
    return [row for row in rows if row[3] in codes and row[0] in track_ids and only in (None, row[0])]


def reason_of(error: Exception) -> str:
    """Why the flags could not be listed, as the part of a sentence (`NOT_LISTED`) for the user,
    without a full stop: for a file that could not be read, its name without its folder; for any
    other error, `worker.plain`."""
    name = getattr(error, "filename", None) if isinstance(error, OSError) else None
    return NOT_READ.format(name=Path(name).name) if isinstance(name, str) and name else plain(error).rstrip(". ")


class _Task:
    """One listing as a thread runs it (the worker's, or the listing's own):
    `tracking.flags_table` of a run folder."""

    def __init__(self, listing: Listing, serial: int, folder: Path):
        self._listing, self._serial, self._folder = listing, serial, folder

    def __call__(self, segmenter) -> None:
        """List the flags and report the rows, or one plain sentence that says why there are none
        (results.npz was there when the listing was asked for), with the trace of an error that
        the sentence does not explain. Nothing is raised; the model (`segmenter`) is not used."""
        rows, problem, trace = [], "", ""
        try:
            rows = tracking.flags_table(self._folder)
        except ValueError as refused:  # no scale or no fps_true yet: the function's own sentence
            problem = str(refused)
        except Exception as error:  # whatever else is raised: said in one line, never raised in the thread
            if isinstance(error, FileNotFoundError) and not (self._folder / schema.SESSION_JSON).is_file():
                problem = NO_SESSION_FILE  # which says all there is to it: no trace
            else:
                problem = NOT_LISTED.format(reason=reason_of(error))
                trace = traced(TRACE_HEAD.format(folder=self._folder.name))
        self._listing._done.emit(self._serial, rows, problem, trace)


class _Aside(QObject):
    """The object in a listing's own thread: `run` is its slot and runs there."""

    @Slot(object)
    def run(self, task) -> None:
        """Call `task(None)`: no model is there. The task reports by itself and raises nothing."""
        task(None)


class Listing(QObject):
    """The flags of the window's run folder, kept up to date from the GUI thread (see the module's
    text).

    `rows`: what `tracking.flags_table` gave last, as (track id, video frame number, t in s, flag
    code); `problem`: one plain sentence that says why the flags could not be listed (that
    function's own when it refused: no scale, no fps_true), "" when they were. `version` counts
    the listings that were taken. `pending`: the files on disk are no longer what `rows` was made
    of, and the new rows have not arrived (they are being listed, or wait for the model to be
    ready or for a run to end). `held`: while true `sync` does nothing; whoever saves a correction
    and starts its run at once sets it, so that the flags are listed after that run and not
    before it. `changed` is emitted in the GUI thread whenever `rows`, `problem` or `pending` has
    changed.
    """

    changed = Signal()
    # serial, rows, problem, the trace of an error (`worker.traced`): emitted by the task, in the thread that runs it
    _done = Signal(int, object, str, str)
    _aside = Signal(object)  # a task for this listing's own thread

    def __init__(self, window):
        super().__init__(window)
        self._controller, self._worker, self._jobs = window.controller, worker_of(window), jobs_of(window)
        self.rows: list[Row] = []
        self.problem = ""
        self.version = 0
        self.held = False
        self._made_of = None   # what `rows` was made of (`_inputs`)
        self._wanted = None    # what is on disk now, as far as `sync` has seen
        self._asked: tuple[int, object] | None = None  # the listing on its way: its number, what it is made of
        self._serial = 0
        self._thread: QThread | None = None  # this listing's own: made for the first listing without a model
        self._runner: _Aside | None = None   # the object in it
        self._stopped = False                # the window has closed: nothing is listed any more
        self._done.connect(self._arrived)
        self._controller.video_opened.connect(self.sync)
        self._controller.saved.connect(self.sync)
        self._jobs.finished.connect(self.sync)
        self._worker.state_changed.connect(self.sync)
        window.closing.connect(self.stop)

    @property
    def pending(self) -> bool:
        """Whether `rows` is out of date (see the class)."""
        return self._wanted != self._made_of

    def is_running(self) -> bool:
        """Whether this listing's own thread runs: from the first listing without a model until
        the window closes. With a model the flags are listed in the worker thread, and it never does."""
        return self._thread is not None and self._thread.isRunning()

    def _inputs(self):
        """What the flags on disk are made of, for telling whether they changed; None while no
        track of the session has results. It holds no quantity that is used as one."""
        session, folder = self._controller.session, self._controller.run_folder
        if session is None or folder is None:
            return None
        try:
            found = (Path(folder) / schema.RESULTS_NPZ).stat()
        except OSError:
            return None
        try:
            tracked = set(self._jobs.results.now().track_ids)
        except ValueError:  # no results file of this version: the listing says so
            tracked = {track.id for track in session.tracks}
        tracks = tuple((track.id, None if track.head_px is None else tuple(track.head_px), track.start_frame)
                       for track in session.tracks if track.id in tracked)
        if not tracks:
            return None
        saved = (Path(folder) / schema.SESSION_JSON).is_file()  # without it nothing can be listed: see `_Task`
        return (str(folder), found.st_size, found.st_mtime_ns, found.st_ino, saved, session.time.fps_true,
                repr(session.calibration), repr(session.axes), repr(session.circle), repr(session.processing), tracks)

    def sync(self, *_) -> None:
        """List the flags again if what they are made of changed since `rows` was listed. Call it
        when session.json is up to date on disk; it returns at once, and `changed` says when the
        rows are there. Nothing is asked for during a run, while `held`, or after `stop`. The
        worker thread lists; when the model could not be loaded, this listing's own thread does."""
        if self.held or self._stopped or self._jobs.running or self._worker.state == "stopped":
            return
        before = (self.version, self.pending)
        self._wanted = self._inputs()
        if self._wanted is None:  # no results: nothing to list
            self._asked = None
            if (self._made_of, self.rows, self.problem) != (None, [], ""):
                self._take(None, [], "")
        elif not self.pending:
            self._asked = None  # what may be on its way is of files that are no longer there
        elif self._asked is None or self._asked[1] != self._wanted:
            self._serial += 1
            task = _Task(self, self._serial, Path(self._controller.run_folder))
            self._asked = (self._serial, self._wanted)
            if not self._worker.run(task):
                if self._worker.state == "failed":
                    self._list_aside(task)
                else:
                    self._asked = None  # the model is loading: `state_changed` calls this again
        if before != (self.version, self.pending):
            self.changed.emit()

    def _list_aside(self, task) -> None:
        """Have `task(None)` called in this listing's own thread, after the listing that thread may
        be in. The thread is made and started at the first call, with the stack of the worker
        thread, where the same task runs when there is a model."""
        if self._thread is None:
            self._thread, self._runner = QThread(self), _Aside()
            self._thread.setStackSize(STACK_BYTES)
            self._runner.moveToThread(self._thread)
            self._aside.connect(self._runner.run)
            self._thread.start()
        self._aside.emit(task)

    def stop(self) -> None:
        """The window closes: list nothing from now on, and end this listing's own thread if it
        runs. It waits until the listing the thread is in has returned, as `Worker.stop` waits for
        the worker's; a listing that waits behind that one is dropped. Calling it again does nothing."""
        self._stopped = True
        if self.is_running():
            self._thread.quit()
            self._thread.wait()

    def _arrived(self, serial: int, rows: list[Row], problem: str, trace: str) -> None:
        """Take what a task listed: in the GUI thread. The trace of an error is kept with the
        worker's and written to run.log with them, whichever request it is of; the rows of an
        older request are dropped."""
        if trace:
            self._worker.keep_trace(trace)
            self._jobs.write_traces()
        if self._asked is None or serial != self._asked[0]:
            return
        made_of, self._asked = self._asked[1], None
        self._take(made_of, rows, problem)
        self.changed.emit()

    def _take(self, made_of, rows: list[Row], problem: str) -> None:
        self._made_of, self.rows, self.problem = made_of, rows, problem
        self.version += 1
