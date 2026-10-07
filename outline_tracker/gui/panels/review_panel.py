"""Panel 8, Review and fix (SPEC 6.6, 9, 10.1): the flags table, Previous flag and Next flag, and the
three corrections Re-track from here, End track here and Continue as new track.

- The table (gui/review_table.py) lists `tracking.flags_table` of the run folder: by default the
  flags on positions (LOST, JUMP, SIZE, CONTACT, EDGE, MULTI); "Show shape flags too" adds LOWRES,
  ORIENT and HEADGUESS, which sit on nearly every frame of a small object; "Only the selected
  object" limits it to the object selected in panel 6. A row that is clicked, or reached with the
  up and down keys, shows its frame and selects its object. Previous flag and Next flag go through
  the listed rows of the selected object, from the frame shown, and around.
- The corrections work on the object selected in panel 6 at the frame the window shows, and go
  through `outline_tracker.tracking` (`retrack_from`, `end_track`, `new_piece`), which is given
  None for the results: it reads results.npz as it is now. Re-track from here and End track here
  remove results, so they ask first (`dialogs.confirm`), and their change is saved at once; Re-track
  from here then starts the run that tracks the object again. A refusal is shown in the panel in
  the function's own words, and nothing changes. A frame that was moved onto the clip's grid is
  said, and the window goes there. After a correction the controller's `touch()` tells the rest of
  the window (the object table, the picture).
- The buttons of the corrections are off, with the reason as the end of the hint line and as their
  tooltip, during a run or an export (`Jobs.writing`), before anything is tracked, without a selected
  object, and for an object that has no results yet; Re-track from here also while the model is not ready.
- The panel is "Not started" before anything is tracked, done when tracking is complete and no
  flag on a position is listed, and needs attention otherwise; the hint line gives the number.

Units: frames are video frame numbers; t in the table is frame / fps_true in s. Lengths of widgets
are Qt's device-independent px. Nothing here imports torch.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from PySide6.QtWidgets import QCheckBox, QLabel, QSizePolicy, QWidget

from outline_tracker import tracking
from outline_tracker.gui import dialogs, estimate
from outline_tracker.gui.panels.calibration_panel import CONTROL_HEIGHT, GROUP_GAP, Message, button, column, row
from outline_tracker.gui.prompts import Prompts
from outline_tracker.gui.review_table import FlagsModel, FlagsView, Listing, listed, meaning
from outline_tracker.gui.worker import worker_of
from outline_tracker.gui.worker_jobs import NO_MODEL, jobs_of
from outline_tracker.results import ResultsStore
from outline_tracker.tracking_ids import next_piece_id

PRIMARY_HEIGHT = 32

NOT_STARTED = "Track first. The flags appear here."
NO_FLAGS = "No flags on positions. Play the video once and look at the outlines."
FLAGS = "{flags} on {tracks}. Click a row to go to its frame."
NOT_COMPLETE = "No flags on positions so far. Tracking is not complete: see panel 7 (Track)."
RUNNING = "Tracking is running. The flags are listed again when it has stopped."
LISTING = "The flags are being listed."
NO_OBJECT = "Select an object to fix its track."
NOT_TRACKED = "Object {id} is not tracked yet. Track it first in panel 7 (Track)."
MODEL_LOADING = "The model is loading. Re-track from here is ready when the model is ready."
CLICK_FIRST = "Click on animal {id} on this frame (frame {frame}). Then click Re-track from here again."
CLICK_PIECE = "Click on animal {id} on this frame (frame {frame}). Then click Track in panel 7 (Track)."
NEW_TRACK = "The new track is {id}."
NO_FLAG_OF = "Object {id} has no flag in the table."
RETRACK_ASK = ("Re-track {id} from frame {frame}?\nThe results of {id} from frame {frame} to the end are replaced. "
               "Other tracks do not change.")
END_ASK = ("End track {id} at frame {frame}?\nThe results of {id} after frame {frame} are removed. "
           "Other tracks do not change.")
TIPS = {
    "next": "Go to the next flag of the selected object",
    "previous": "Go to the previous flag of the selected object",
    "retrack": "Track the selected object again from this frame to the end of the clip",
    "end": "Remove the results of the selected object after this frame",
    "continue": "Add the next track of this animal, starting on this frame",
}


class ReviewPanel(QWidget):
    """The controls of panel 8. Parts: `table` (a view of `model`), `meaning_label` (what the flag of
    the chosen row means), `shape_box`, `selected_box`, `next_button`, `previous_button`, `message`,
    `retrack_button`, `end_button`, `continue_button`, `piece_label` (the id Continue as new track
    would give). `rows`: the rows listed, as (track id, video frame number, t in s, flag code).
    `listing` computes them (`review_table.Listing`); `prompts`, `jobs` and `worker` are the
    window's. `store`: the results as the last Re-track from here or End track here left them
    (`Edit.store`), None before one."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[7]
        self._view = window.view
        self.worker, self.jobs, self.prompts = worker_of(window), jobs_of(window), window.findChild(Prompts)
        self.listing = Listing(window)
        self.store: ResultsStore | None = None
        self.rows: list = []
        self._listed_for = None                      # what `rows` was picked for (`_pick_rows`)
        self._frames: dict[str, list[int]] = {}      # the frames of the listed rows of each track, ascending
        self._counts = (0, 0)                        # flags on positions, and the tracks that have one
        self._said: tuple[str, str] | None = None    # kind and text of the message under the table
        self._to_do: tuple[str, tuple] | None = None  # a sentence that says what to click, and what it is about

        self.model = FlagsModel(self)
        self.table = FlagsView(self.model)
        self.table.row_chosen.connect(self.go_to_row)
        self.meaning_label = QLabel()
        self.meaning_label.setProperty("role", "hint")
        self.meaning_label.setWordWrap(True)
        self.shape_box = QCheckBox("Show shape flags too")
        self.shape_box.setToolTip("Also list LOWRES, ORIENT and HEADGUESS: they are on nearly every frame of a small "
                                  "object")
        self.selected_box = QCheckBox("Only the selected object")
        self.selected_box.setToolTip("List the flags of the object selected in panel 6 only")
        for box in (self.shape_box, self.selected_box):
            box.toggled.connect(self.refresh)
        self.message = Message()

        self.next_button = self._button("Next flag", "next", self.next_flag)
        self.next_button.setProperty("kind", "primary")
        self.previous_button = self._button("Previous flag", "previous", self.previous_flag)
        self.retrack_button = self._button("Re-track from here", "retrack", self.retrack)
        self.end_button = self._button("End track here", "end", self.end_track)
        self.end_button.setProperty("kind", "destructive")
        self.continue_button = self._button("Continue as new track", "continue", self.continue_as_new)
        self.next_button.setFixedHeight(PRIMARY_HEIGHT)
        self.piece_label = QLabel()
        self.piece_label.setProperty("role", "hint")

        # one correction per row: three long labels do not fit side by side in the dock at its smallest
        column(self, self.table, self.meaning_label, self.shape_box, self.selected_box,
               row(self.next_button, self.previous_button), self.message, self.retrack_button, GROUP_GAP,
               self.end_button, self.continue_button, self.piece_label)

        self.listing.changed.connect(self.refresh)
        self.prompts.changed.connect(self.refresh)       # the selection, a click, the tool, the frame shown
        self._controller.session_changed.connect(self.refresh)
        self._controller.video_opened.connect(self._video_opened)
        self.worker.state_changed.connect(self.refresh)
        self.jobs.started.connect(self._run_started)
        self.jobs.writing_changed.connect(self.refresh)  # a run or an export has started or ended
        self.refresh()

    def _button(self, text: str, tip: str, pressed):
        made = button(text, TIPS[tip])
        made.setFixedHeight(CONTROL_HEIGHT)
        # a button takes the room its row has and never widens the panel, whatever the font
        made.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        made.clicked.connect(pressed)
        return made

    # ------------------------------------------------------------------ going to a flag

    def go_to_row(self, index: int) -> None:
        """Show the frame of the table's row `index` (counted from 0) and select its object in panel
        6. A row the table does not have changes nothing."""
        if 0 <= index < len(self.rows):
            self._go(*self.rows[index][:2], self.rows[index])

    def next_flag(self) -> None:
        """Go to the next frame after the one shown on which the selected object has a listed
        flag; after the last one, to the first (the Next flag button)."""
        frames = self._frames.get(self.prompts.selected, [])
        if frames:
            at = bisect_right(frames, self._view.frame) if self._view.frame is not None else 0
            self._go(self.prompts.selected, frames[at % len(frames)])

    def previous_flag(self) -> None:
        """Go to the last frame before the one shown on which the selected object has a listed
        flag; before the first one, to the last (the Previous flag button)."""
        frames = self._frames.get(self.prompts.selected, [])
        if frames:
            at = bisect_left(frames, self._view.frame) if self._view.frame is not None else 0
            self._go(self.prompts.selected, frames[at - 1])

    def _go(self, track_id: str, frame: int, chosen: tuple | None = None) -> None:
        """Show video frame `frame`, select the object `track_id`, and make the row `chosen` (the
        first row of that object on that frame if None) the table's current row."""
        if self._view.frame != frame:
            self._window.show_frame(frame)
        if self.prompts.selected != track_id:
            self.prompts.select(track_id)  # with "Only the selected object", the table then lists other rows
        if chosen in self.rows:
            at = self.rows.index(chosen)
        else:
            at = next((at for at, listed_row in enumerate(self.rows) if listed_row[:2] == (track_id, frame)), -1)
        self.table.choose(at)
        self.refresh()

    # ------------------------------------------------------------------ the corrections

    def retrack(self) -> None:
        """Re-track from here (SPEC 6.6): track the selected object again from the frame shown on.
        The run starts from a positive point of the object on that frame. Without one, the Positive
        tool is chosen and the hint line says to click on the animal and to press the button again.
        With one, the question is asked first, since the results from this frame on are replaced."""
        track, frame = self._selected_track(), self._view.frame
        if track is None or frame is None or self._reasons()[0] is not None:
            return
        self._said = None
        clicked = any(label == 1 for prompt in track.prompts if prompt.frame == frame for label in prompt.labels)
        if not clicked:
            self.prompts.choose_tool("positive")
            self._tell(CLICK_FIRST.format(id=track.id, frame=frame))
            return
        dialogs.confirm(self._window, RETRACK_ASK.format(id=track.id, frame=frame), "Re-track from here",
                        lambda: self._retrack(track.id, frame))

    def _retrack(self, track_id: str, frame: int) -> None:
        """The question was answered with yes: make the correction, save it, and start its run."""
        session = self._controller.session
        if session is None or self.jobs.writing():
            return
        try:
            edit = tracking.retrack_from(session, None, [track_id], frame, {}, self._controller.run_folder)
        except (ValueError, RuntimeError) as refused:  # RuntimeError: results.npz could not be saved
            self._refused(refused)
            return
        self.listing.held = True  # the flags are listed after the run, not between the correction and the run
        try:
            self._made(edit)
            not_started = self.jobs.start()
        finally:
            self.listing.held = False
        if not_started is not None:  # the correction is made; Track in panel 7 starts the run later
            self._said = ("problem", not_started)
            self.listing.sync()
        self.refresh()

    def end_track(self) -> None:
        """End track here (SPEC 6.6): remove the selected object's results after the frame shown,
        and mark the track as ended there. The question is asked first."""
        track, frame = self._selected_track(), self._view.frame
        if track is None or frame is None or self._reasons()[1] is not None:
            return
        self._said = None
        dialogs.confirm(self._window, END_ASK.format(id=track.id, frame=frame), "End track here",
                        lambda: self._end(track.id, frame))

    def _end(self, track_id: str, frame: int) -> None:
        """The question was answered with yes: make the correction and save it."""
        session = self._controller.session
        if session is None or self.jobs.writing():
            return
        try:
            edit = tracking.end_track(session, None, track_id, frame, self._controller.run_folder)
        except (ValueError, RuntimeError) as refused:
            self._refused(refused)
            return
        self._made(edit)
        self.refresh()

    def continue_as_new(self) -> None:
        """Continue as new track (SPEC 6.6): add the next piece of the selected object's animal (A2
        after A), starting on the frame shown, select it and choose the Positive tool; the hint
        line says to click on the animal and to press Track. Nothing is removed, so nothing is asked."""
        track, frame, session = self._selected_track(), self._view.frame, self._controller.session
        if track is None or frame is None or self._reasons()[1] is not None:
            return
        self._said = None
        try:
            edit = tracking.new_piece(session, track.id, frame)
        except ValueError as refused:
            self._refused(refused)
            return
        if edit.moved:  # onto the clip's grid: say so, and show the frame the piece starts on
            self._said = ("warning", edit.note)
            self._window.show_frame(edit.frame)
        (piece,) = edit.track_ids
        self.prompts.select(piece)
        self.prompts.choose_tool("positive")
        self._tell(CLICK_PIECE.format(id=piece, frame=edit.frame))
        self._controller.touch()

    def _made(self, edit) -> None:
        """Take over a correction that changed the results: keep its store, show the frame it was
        made on if that is another one, tell the window, and save the session now."""
        self.store, self._to_do = edit.store, None
        if edit.moved:
            self._said = ("warning", edit.note)
            self._window.show_frame(edit.frame)
        self._controller.touch()
        self._controller.save_now()

    def _refused(self, refused: Exception) -> None:
        self._said = ("problem", str(refused))
        self.refresh()

    def _tell(self, text: str) -> None:
        """Put `text` in the hint line until the selected object, the frame shown or the object's
        points on that frame change."""
        self._to_do = (text, self._about())
        self.refresh()

    def _about(self) -> tuple:
        track, frame = self._selected_track(), self._view.frame
        points = 0 if track is None else sum(len(prompt.points_px) for prompt in track.prompts if prompt.frame == frame)
        return (self.prompts.selected, frame, points)

    # ------------------------------------------------------------------ what the window reports

    def _video_opened(self) -> None:
        self.store = self._said = self._to_do = None
        self.refresh()

    def _run_started(self) -> None:
        self._said = self._to_do = None
        self.refresh()

    # ------------------------------------------------------------------ showing

    def _selected_track(self):
        session = self._controller.session
        return None if session is None else next(
            (track for track in session.tracks if track.id == self.prompts.selected), None)

    def _results(self) -> ResultsStore:
        """The results on disk now; none when the file cannot be read as results."""
        try:
            return self.jobs.results.now()
        except ValueError:
            return ResultsStore()

    def _reasons(self) -> tuple[str | None, str | None]:
        """Why Re-track from here cannot be used now, and why the other two corrections cannot:
        each one sentence, or None when the button can be used."""
        session, selected = self._controller.session, self.prompts.selected
        tracked = set() if session is None else {track.id for track in session.tracks} & set(self._results().track_ids)
        if self.jobs.writing():
            both = RUNNING if self.jobs.running else self.jobs.writing()  # "An export is running."
        elif not tracked:
            both = NOT_STARTED
        elif selected is None:
            both = NO_OBJECT
        elif selected not in tracked:
            both = NOT_TRACKED.format(id=selected)
        else:
            both = None
        if both is not None or self.worker.ready:
            return both, both
        loading = self.worker.state in ("idle", "loading")
        return MODEL_LOADING if loading else NO_MODEL.format(reason=self.worker.message).strip(), None

    def _pick_rows(self) -> None:
        """Bring `rows` in line with the listing, the two boxes and the session's objects."""
        session = self._controller.session
        ids = () if session is None else tuple(track.id for track in session.tracks)
        only = self.prompts.selected if self.selected_box.isChecked() else None
        wanted = (self.listing.version, ids, only, self.shape_box.isChecked())
        if wanted == self._listed_for:
            return
        self._listed_for = wanted
        on_positions = listed(self.listing.rows, ids, None, False)
        self._counts = (len(on_positions), len({track_id for track_id, *_ in on_positions}))
        self.rows = listed(self.listing.rows, ids, only, self.shape_box.isChecked())
        self._frames = {}
        for track_id, frame, _, _ in self.rows:
            frames = self._frames.setdefault(track_id, [])
            if not frames or frames[-1] != frame:
                frames.append(frame)
        self.table.set_rows(self.rows)

    def _standing(self, tracked: bool) -> tuple[str, str]:
        """(the panel's state, its hint line) from the listing and the session."""
        flags, tracks = self._counts
        if self.jobs.running:
            return self._panel.state, RUNNING
        if not tracked:
            return "todo", NOT_STARTED
        if self.listing.problem:
            return "attention", self.listing.problem
        if self.listing.pending:
            return self._panel.state, LISTING
        if flags:
            return "attention", FLAGS.format(flags=estimate.counted(flags, "flag"),
                                             tracks=estimate.counted(tracks, "track"))
        return ("done", NO_FLAGS) if self._controller.session.complete else ("attention", NOT_COMPLETE)

    def refresh(self, *_) -> None:
        """Show where the review stands: the table, which buttons can be used and why not, the
        message, and the panel's state and hint line."""
        self._pick_rows()
        self.model.set_frame(self._view.frame)
        at = self.table.chosen()
        code = self.rows[at][3] if 0 <= at < len(self.rows) else None
        self.meaning_label.setText("" if code is None else f"{code}: {meaning(code)}")

        selected = self.prompts.selected
        no_flag = None if self._frames.get(selected) else (NO_OBJECT if selected is None
                                                           else NO_FLAG_OF.format(id=selected))
        for made, tip in ((self.next_button, "next"), (self.previous_button, "previous")):
            made.setEnabled(no_flag is None)
            made.setToolTip(no_flag or TIPS[tip])
        retrack, others = self._reasons()
        for made, tip, reason in ((self.retrack_button, "retrack", retrack), (self.end_button, "end", others),
                                  (self.continue_button, "continue", others)):
            made.setEnabled(reason is None)
            made.setToolTip(reason or TIPS[tip])
        try:
            piece = "" if others is not None else NEW_TRACK.format(id=next_piece_id(self._controller.session, selected))
        except ValueError:  # an id that is not letters and a number: Continue as new track says so
            piece = ""
        self.piece_label.setText(piece)

        kind, text = self._said or (None, "")
        self.message.show_message(kind, text, self._window.statusBar())

        state, hint = self._standing(others != NOT_STARTED)
        if self._to_do is not None and self._to_do[1] != self._about():
            self._to_do = None
        if self._to_do is not None:
            hint = self._to_do[0]
        elif retrack is not None and retrack != hint and hint != self.listing.problem:
            hint = f"{hint} {retrack}"  # why a button is off, where the line does not say it yet
        if self._panel.state != state:
            self._panel.set_state(state)
        if self._panel.hint.text() != hint:
            self._panel.set_hint(hint)


def build(window) -> QWidget:
    """The controls of panel 8 for `window` (a `MainWindow`): a `ReviewPanel`. It works with the
    `Prompts` that panel 6 made, the window's `Jobs` and its worker."""
    return ReviewPanel(window)
