"""Panel 6, Objects (SPEC 5, 10.1): the table of objects, Add and Remove, mode and fine window of
the selected object, and the buttons of the point tools Positive, Negative and Head.

What a click on the video does, and what is drawn there, is in gui/prompts.py (`Prompts`); the
model that makes the outlines runs in the worker thread (gui/worker.py). This module is the
panel's controls. Every edit of the object table goes through `outline_tracker.tracking`
(`add_object`, `remove_object`): ids and colours are decided there. `remove_object` gets None for
the results, so it reads results.npz as it is now and saves it without the object's records. After
an edit the controller's `touch()` tells the rest of the window. A refusal is shown in the panel
in the function's own words, and nothing changes.

Remove deletes an object's tracked frames, which cannot be undone: the button asks first
(`dialogs.confirm`) and says how many frames go; an object without results is removed at once.
While a tracking job or an export runs (`worker_jobs.Jobs.writing`) the objects stay as that task
got them: Add, Remove, the tools, the mode and the window are off, each with the reason as its tooltip.

The panel is done when at least one object has a click; its hint line says the next step.

Units: frames are video frame numbers; the fine window is the side of a square in px of the video
frame (SPEC 6.3). Lengths of widgets are Qt's device-independent px.
"""

from __future__ import annotations

import tempfile

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from outline_tracker import tracking
from outline_tracker.gui import dialogs, theme
from outline_tracker.gui.estimate import counted
from outline_tracker.gui.prompts import BUSY_TEXT, LOADING_TEXT, Prompts, colour_of
from outline_tracker.gui.worker import worker_of
from outline_tracker.gui.worker_jobs import jobs_of
from outline_tracker.measure import MODES
from outline_tracker.results import ResultsStore
from outline_tracker.tracking_plan import pending_from

COLUMNS = ("Object", "Mode", "Window", "Start", "Status")
NONE = "–"                 # a cell that has no value: the window of a coarse object, the start of one without clicks
ROWS_SHOWN, ROW_HEIGHT = 4, 24
CONTROL_HEIGHT, PRIMARY_HEIGHT = 28, 32
SPACING = 8                # between two rows, and between two buttons
LARGEST_WINDOW = 4096      # px: only a wide limit of the box; tracking uses the window as it is typed

ADD_HINT = "Click Add. Then click on one animal in the video. An outline appears."
NO_POINTS = "Object {id} has no points. Click on the animal, or remove the object."
READY = ("{count} ready to track. Check that each outline follows its animal. If the antennae matter, click on "
         "each antenna too. One click on the body leaves them out.")
TRACKED_MODE = ("Object {id} is tracked already, in {mode} mode. To track the animal in {other} mode, remove the "
                "object and add it again.")
MODEL_TEXTS = {"idle": "Model loading", "loading": "Model loading", "ready": "Model ready",
               "failed": "Model not loaded", "stopped": "Model not loaded"}
TOOL_TIPS = {"positive": "Click on the animal: a positive point",
             "negative": "Click on what is not the animal: a negative point (also a right click)",
             "head": "Click on the head of the animal, on the frame its track starts on"}
ADD_TIP, NO_VIDEO_TIP = "Add an object. Then click on the animal in the video.", "Open a video in panel 1 first."
# The question before Remove: the first line is the heading, the rest says what is lost.
REMOVE_QUESTION = ("Remove object {id}?\nIts points and its results on {frames} are removed. This cannot be "
                   "undone. Other objects do not change.")


def status_of(track, store: ResultsStore) -> str:
    """The status word of an object (SPEC 5), from its track and the results so far: "ready" while
    it has clicks that are not tracked yet, "ended" for a track that was ended, "tracked" when it
    has results, "no prompts" for an object without clicks and without results."""
    if pending_from(track, store) is not None:
        return "ready"
    if track.ended_at is not None:
        return "ended"
    return "tracked" if track.id in store.track_ids else "no prompts"


def has_click(track) -> bool:
    return any(prompt.points_px for prompt in track.prompts)


class ObjectsPanel(QWidget):
    """The controls of panel 6. Parts: `table`; `mode_box`, `window_box` (0 shows as "auto") and
    `auto_button` for the selected object; `positive_button`, `negative_button`, `head_button`;
    `busy_label`, `message_label`; `add_button`, `remove_button`; in the window's status bar
    `model_label`. `prompts` is the `Prompts` of the window, `worker` its `Worker`."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[5]
        self.worker, self._jobs = worker_of(window), jobs_of(window)
        self.prompts = Prompts(window, self.worker)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        header.setFixedHeight(ROW_HEIGHT)
        self.table.setFixedHeight((ROWS_SHOWN + 1) * ROW_HEIGHT + 2 * self.table.frameWidth())
        self.table.setToolTip("The objects to track. Click a row to choose the object that gets the next point.")
        self.table.itemSelectionChanged.connect(self._row_selected)

        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.setToolTip("coarse: tracked with the others on the frame. fine: tracked alone, in a window that "
                                 "follows the object.")
        self.mode_box.currentTextChanged.connect(self.set_mode)
        self.window_box = QSpinBox()
        self.window_box.setRange(0, LARGEST_WINDOW)
        self.window_box.setSpecialValueText("auto")
        self.window_box.setSuffix(" px")
        self.window_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.window_box.setKeyboardTracking(False)  # a typed number counts when it is complete
        self.window_box.setToolTip("Side of the window of a fine object. auto: three times the object's length.")
        self.window_box.valueChanged.connect(self.set_fine_window)
        self.auto_button = QPushButton("Auto")
        self.auto_button.setToolTip("Let the window follow from the object's length again")
        self.auto_button.clicked.connect(lambda: self.set_fine_window(None))
        window_row = QHBoxLayout()
        window_row.setSpacing(SPACING)
        window_row.addWidget(self.window_box, 1)
        window_row.addWidget(self.auto_button)
        fields = QFormLayout()
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(SPACING)
        fields.addRow("Mode", self.mode_box)
        fields.addRow("Fine window", window_row)

        tools = QHBoxLayout()
        tools.setSpacing(SPACING)
        self._tool_buttons = {}
        for kind in self.prompts.tools:
            button = QPushButton(kind.capitalize())
            button.setCheckable(True)
            button.setToolTip(TOOL_TIPS[kind])
            # the three share the row equally and never widen the panel, whatever the font
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked, kind=kind: self.prompts.choose_tool(kind if checked else None))
            tools.addWidget(button)
            self._tool_buttons[kind] = button
        self.positive_button, self.negative_button, self.head_button = self._tool_buttons.values()

        self.busy_label = QLabel(BUSY_TEXT)
        self.busy_label.setProperty("role", "hint")
        self.busy_label.setWordWrap(True)
        self.message_label = QLabel()
        self.message_label.setProperty("role", "msg")
        self.message_label.setWordWrap(True)

        self.add_button = QPushButton("Add")
        self.add_button.setProperty("kind", "primary")
        self.add_button.clicked.connect(self.add_object)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setProperty("kind", "destructive")
        self.remove_button.setToolTip("Remove the selected object, with its points and its results")
        self.remove_button.clicked.connect(self.ask_remove)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        buttons.addWidget(self.add_button)
        buttons.addStretch(1)
        buttons.addWidget(self.remove_button)
        for button in (*self._tool_buttons.values(), self.auto_button, self.add_button, self.remove_button):
            button.setAutoDefault(False)  # no button of the window reacts to Enter
            button.setFixedHeight(PRIMARY_HEIGHT if button is self.add_button else CONTROL_HEIGHT)

        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(SPACING)
        rows.addWidget(self.table)
        rows.addLayout(fields)
        rows.addLayout(tools)
        rows.addWidget(self.busy_label)
        rows.addWidget(self.message_label)
        rows.addLayout(buttons)

        self.model_label = QLabel()
        window.statusBar().addPermanentWidget(self.model_label)

        # each part's own tooltip, to put back when the run or the export that had it off has ended
        self._tips = {part: part.toolTip() for part in (self.mode_box, self.window_box, self.auto_button,
                                                        self.remove_button, *self._tool_buttons.values())}
        self.prompts.changed.connect(self.refresh)  # also when a run or an export starts or ends
        self.worker.state_changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ edits

    def _selected_track(self):
        session = self._controller.session
        return None if session is None else next(
            (track for track in session.tracks if track.id == self.prompts.selected), None)

    def add_object(self) -> None:
        """Add an object to the session (the next id, with its colour; coarse, automatic window),
        select it and choose the Positive tool: the next click on the video is its first point."""
        session = self._controller.session
        if session is None or self._jobs.writing():
            return
        self.prompts.selected = tracking.add_object(session).id
        self.prompts.choose_tool("positive")
        self.prompts.say("", "")
        self._controller.touch()

    def ask_remove(self) -> None:
        """The Remove button: remove the selected object, after a question when it has tracked
        frames (how many is said; only the answer Remove removes). Without results nothing is asked."""
        track_id, store = self.prompts.selected, self._results()
        frames = len(store.arrays(track_id).frames) if track_id in store.track_ids else 0
        if frames and not self._jobs.writing():
            dialogs.confirm(self._window, REMOVE_QUESTION.format(id=track_id, frames=counted(frames, "tracked frame")),
                            "Remove", lambda: self.remove(track_id))
        else:
            self.remove_selected()

    def remove_selected(self) -> None:
        """Remove the selected object, without a question (`remove`)."""
        self.remove(self.prompts.selected)

    def remove(self, track_id: str | None) -> None:
        """Remove the object `track_id`: its track from the session, and its records from
        results.npz of the run folder. The object that takes its row is selected then. Nothing
        happens for an id the session does not have, and while a tracking job or an export runs."""
        session = self._controller.session
        if session is None or self._jobs.writing() or track_id not in [track.id for track in session.tracks]:
            return
        index = [track.id for track in session.tracks].index(track_id)
        try:
            if self._controller.run_folder is None:
                # no run folder yet, so no results anywhere: the function is given a folder without any
                with tempfile.TemporaryDirectory() as without_results:
                    tracking.remove_object(session, None, track_id, without_results)
            else:
                tracking.remove_object(session, None, track_id, self._controller.run_folder)
        except (ValueError, RuntimeError) as refused:  # RuntimeError: results.npz could not be saved
            self.prompts.say("problem", str(refused))
            return
        left = session.tracks
        self.prompts.selected = left[min(index, len(left) - 1)].id if left else None
        self.prompts.say("", "")
        self._controller.touch()

    def set_mode(self, mode: str) -> None:
        """Set the selected object's mode: "coarse" or "fine". An object that has results keeps its
        mode (its records were measured in it), and the panel says what to do instead."""
        track = self._selected_track()
        if self._filling or self._jobs.writing() or track is None or mode == track.mode or mode not in MODES:
            return
        if track.id in self._results().track_ids:
            self.prompts.say("problem", TRACKED_MODE.format(id=track.id, mode=track.mode, other=mode))
            return
        track.mode = mode
        self.prompts.say("", "")
        self._controller.touch()

    def set_fine_window(self, side_px: int | None) -> None:
        """Set the selected object's fine window: the side of the square in px of the video frame,
        or None (or 0) for automatic, which is the way back from a typed value (SPEC 6.3)."""
        track = self._selected_track()
        side = int(side_px) if side_px else None
        if self._filling or self._jobs.writing() or track is None or side == track.fine_window_px:
            return
        track.fine_window_px = side
        self._controller.touch()

    def _row_selected(self) -> None:
        session, row = self._controller.session, self.table.currentRow()
        if self._filling or session is None or not 0 <= row < len(session.tracks):
            return
        if session.tracks[row].id != self.prompts.selected:
            self.prompts.select(session.tracks[row].id)

    # ------------------------------------------------------------------ showing

    _filling = False  # true while refresh() sets the widgets: their signals are no edits then

    def _results(self) -> ResultsStore:
        """The results on disk now; none when the file cannot be read (the edits say why)."""
        try:
            return self.prompts.results.now()
        except ValueError:
            return ResultsStore()

    def refresh(self, *_) -> None:
        """Show the session as it is now: the table, the selected object's mode and window, which
        buttons can be used, the message, and the panel's state and hint."""
        session = self._controller.session
        tracks = [] if session is None else session.tracks
        store, selected = self._results(), self._selected_track()
        self._filling = True
        try:
            self._fill_table(tracks, store)
            fine = selected is not None and selected.mode == "fine"
            self.mode_box.setCurrentText(selected.mode if selected is not None and selected.mode in MODES else MODES[0])
            self.window_box.setValue(selected.fine_window_px or 0 if fine else 0)
        finally:
            self._filling = False
        busy = self._jobs.writing()  # a run or an export works on the objects as they are: nothing changes now
        self.mode_box.setEnabled(selected is not None and not busy)
        self.window_box.setEnabled(fine and not busy)
        self.auto_button.setEnabled(fine and selected.fine_window_px is not None and not busy)
        self.add_button.setEnabled(session is not None and not busy)
        self.add_button.setToolTip(busy or (ADD_TIP if session is not None else NO_VIDEO_TIP))
        self.remove_button.setEnabled(selected is not None and not busy)
        for kind, button in self._tool_buttons.items():
            button.setEnabled(selected is not None and not busy)
            button.setChecked(self.prompts.tool_kind == kind)
        for part, tip in self._tips.items():
            part.setToolTip(busy or tip)

        kind, text = self.prompts.message
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))
        if self.message_label.property("kind") != kind:
            theme.set_property(self.message_label, "kind", kind)
        self.busy_label.setText(LOADING_TEXT if self.worker.state in ("idle", "loading") else BUSY_TEXT)
        self.busy_label.setVisible(self.prompts.busy)
        self.model_label.setText(MODEL_TEXTS[self.worker.state])
        self.model_label.setToolTip(self.worker.message)

        clicked = [track for track in tracks if has_click(track)]
        bare = next((track.id for track in tracks if not has_click(track)), None)
        if not tracks:
            hint = ADD_HINT
        elif bare is not None:
            hint = NO_POINTS.format(id=bare)
        else:
            hint = READY.format(count="1 object is" if len(clicked) == 1 else f"{len(clicked)} objects are")
        state = "done" if clicked else "todo"
        if self._panel.state != state:
            self._panel.set_state(state)
        if self._panel.hint.text() != hint:
            self._panel.set_hint(hint)

    def _fill_table(self, tracks, store: ResultsStore) -> None:
        self.table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            started = bool(track.prompts) or track.id in store.track_ids
            if track.mode != "fine":
                window = NONE
            else:
                window = "auto" if track.fine_window_px is None else f"{track.fine_window_px} px"
            cells = (track.id, track.mode, window, str(track.start_frame) if started else NONE, status_of(track, store))
            for column, text in enumerate(cells):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # the table shows; the controls edit
                    self.table.setItem(row, column, item)
                item.setText(text)
                if column == 0:
                    item.setData(Qt.ItemDataRole.DecorationRole, QColor(colour_of(track)))  # the colour swatch
                elif column == 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row = next((row for row, track in enumerate(tracks) if track.id == self.prompts.selected), -1)
        if row < 0:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
        elif self.table.currentRow() != row or not self.table.selectionModel().isRowSelected(row):
            self.table.selectRow(row)


def build(window) -> QWidget:
    """The controls of panel 6 for `window` (a `MainWindow`): an `ObjectsPanel`. Making it also
    makes the window's worker (`worker.worker_of`), which loads the model once the window is shown."""
    return ObjectsPanel(window)
