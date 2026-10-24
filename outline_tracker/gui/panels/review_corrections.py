"""The three corrections of panel 8 (SPEC 6.6): Re-track from here, End track here and Continue as new track.

`Corrections` is the part of `ReviewPanel` (gui/panels/review_panel.py) that makes them: methods that the panel
inherits. It is not a panel (it is not in `PANEL_MODULES`) and keeps no state of its own; it works with what the
panel has: `prompts`, `jobs`, `listing`, `store`, `_window`, `_controller`, `_view`, the notes `_said` and `_to_do`,
and the methods `refresh`, `_selected_track` and `_reasons`.

The corrections work on the object selected in panel 6 at the frame the window shows, and go through
`outline_tracker.tracking.job` (`retrack_from`, `end_track`, `new_piece`), which is given None for the results: it
reads results.npz as it is now. Re-track from here and End track here remove results, so they ask first
(`dialogs.confirm`), and their change is saved at once; Re-track from here then starts the run that tracks the object
again. A refusal is shown in the panel in the function's own words, and nothing changes. A frame that was moved onto
the clip's grid is said, and the window goes there. After a correction the controller's `touch()` tells the rest of
the window (the object table, the picture).

Units: frames are video frame numbers. Nothing here imports torch.
"""

from __future__ import annotations

from outline_tracker.gui import dialogs
from outline_tracker.tracking import job as tracking

CLICK_FIRST = "Click on animal {id} on this frame (frame {frame}). Then click Re-track from here again."
CLICK_PIECE = "Click on animal {id} on this frame (frame {frame}). Then click Track in panel 7 (Track)."
RETRACK_ASK = ("Re-track {id} from frame {frame}?\nThe results of {id} from frame {frame} to the end are replaced. "
               "Other tracks do not change.")
END_ASK = ("End track {id} at frame {frame}?\nThe results of {id} after frame {frame} are removed. "
           "Other tracks do not change.")


class Corrections:
    """The correction methods of `ReviewPanel`: `retrack`, `end_track` and `continue_as_new`, and what they use
    (`_retrack`, `_end`, `_made`, `_refused`, `_tell`, `_about`)."""

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
