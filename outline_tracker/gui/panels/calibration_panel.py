"""Panel 3, Calibration (SPEC 4.2, 4.3, 10.1): the Stick tool and its length, the scale, and the Tape check.
The scale comes with its uncertainty, and the Tape check with its true distance and its error.

The values live in the session (`calibration.stick`, `calibration.check`, SPEC 8.10) and are kept
there by the two tools (outline_tracker/gui/view/tools.py); this panel chooses the tools, takes the two
typed lengths, and shows what `geometry.stick_scale` and `geometry.tape_check` make of the session:
"32.40 µm/px ± 0.08%", and the check's error with the word "Passes" or "Does not pass" on green or
red. A length that cannot be used is refused where it is typed: the box keeps what was typed and
is marked, a message says why, and the session keeps the length it had.

The parts that panel 4 needs as well (`IconMessage`, `NumberBox`, and the small functions that build
a row) are in gui/widgets/panel_parts.py. Units: lengths typed and shown are mm, points and the stick's
length on the picture are px of the video frame (SPEC 3.1), the scale is shown in µm per px; sizes
of widgets are Qt's device-independent px.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget

from outline_tracker.core import geometry
from outline_tracker.gui.widgets.panel_parts import (DASH, GROUP_GAP, REDO_TIP, WIDEST, IconMessage, NumberBox, button, column,
                                             follow_tool, group_label, number, row, value_label)
from outline_tracker.gui.view.tools import StickTool, TapeTool, mm_per_px

NO_SCALE = "Place the stick first. The tape check needs the scale."
LIMIT = f"{100 * geometry.TAPE_TOLERANCE:g}%"  # "1%"


class CalibrationPanel(QWidget):
    """The controls of panel 3. `window` is the main window (its `controller`, `view`, `panels`).

    Parts: `stick_tool`, `tape_tool`; `stick_button`, `stick_redo`, `length_box`, `scale_value`;
    `tape_button`, `tape_redo`, `tape_note`, `true_box`, `tape_value`; `message`.
    """

    def __init__(self, window):
        super().__init__()
        self._controller, self._panel, self._status = window.controller, window.panels[2], window.statusBar()
        self._start_hint = self._panel.hint.text()
        self._shown = None                    # the session the panel shows
        self._refused: dict[str, str] = {}    # a box's name -> why its typed value was not taken
        self.stick_tool = StickTool(window.controller, window.view, self)
        self.tape_tool = TapeTool(window.controller, window.view, self)

        self.stick_button = button("Stick", "Click the two ends of a known length on the ruler", checkable=True)
        self.tape_button = button("Tape", "Check the scale on two other ruler marks", checkable=True)
        self.stick_redo, self.tape_redo = button("Redo", REDO_TIP), button("Redo", REDO_TIP)
        for redo in (self.stick_redo, self.tape_redo):
            redo.setProperty("kind", "quiet")
        self.length_box = NumberBox(" mm", 2, -WIDEST, "The true length between the two ends of the stick")
        self.true_box = NumberBox(" mm", 2, -WIDEST, "The true distance between the two ruler marks")
        self.scale_value, self.tape_value = value_label(), value_label()
        self.tape_note = QLabel(NO_SCALE)
        self.tape_note.setProperty("role", "hint")
        self.tape_note.setWordWrap(True)
        self.message = IconMessage()
        column(self,
               group_label("Stick"), row(self.stick_button, None, self.stick_redo),
               row("Length", self.length_box), row("Scale", self.scale_value),
               GROUP_GAP,
               group_label("Tape check"), row(self.tape_button, None, self.tape_redo), self.tape_note,
               row("True distance", self.true_box), row("Measured", self.tape_value),
               self.message)

        follow_tool(window.view, [(self.stick_button, self.stick_tool), (self.tape_button, self.tape_tool)])
        self.stick_redo.clicked.connect(self.stick_tool.redo)
        self.tape_redo.clicked.connect(self.tape_tool.redo)
        self.length_box.valueChanged.connect(self.set_stick_length)
        self.true_box.valueChanged.connect(self.set_tape_true)
        window.controller.video_opened.connect(self.refresh)
        window.controller.session_changed.connect(self.refresh)
        self.refresh()

    def set_stick_length(self, mm: float) -> None:
        """Take the stick's true length in mm, as typed. One that is not positive and finite is
        refused: the session keeps its length, and the box and a message say so."""
        self._take("length", self.stick_tool, mm)

    def set_tape_true(self, mm: float) -> None:
        """Take the true distance of the tape marks in mm, as typed; refused like a stick length."""
        self._take("true", self.tape_tool, mm)

    def _take(self, name: str, tool, mm: float) -> None:
        self._refused.pop(name, None)
        try:
            tool.set_value(mm)
        except ValueError as refused:
            self._refused[name] = str(refused)
        self.refresh()

    def refresh(self) -> None:
        """Show the session as it is now: the lengths, the scale, the check, and the panel's state."""
        session = self._controller.session
        self.setEnabled(session is not None)
        if session is None:
            return
        if session is not self._shown:
            self._shown = session
            self._refused.clear()
        scale, check = self.stick_tool.scale(), self.tape_tool.check()
        self.length_box.show_value(self.stick_tool.length_mm, "length" in self._refused)
        self.true_box.show_value(self.tape_tool.true_mm, "true" in self._refused)
        fitted = mm_per_px(session)  # the stick's scale, or the one fitted to a Tracker export
        scale_text = None
        if scale is not None:
            scale_text = f"{1000 * scale.k_mm_per_px:.2f} µm/px ± {100 * scale.rel_uncertainty:.2f}%"
        elif fitted is not None:
            scale_text = f"{1000 * fitted:.2f} µm/px"
        self.scale_value.setText(scale_text or DASH)
        error = None if check is None else f"{number(100 * check.rel_error, 1)}%"
        self.tape_value.setText(DASH if check is None else f"{check.measured_mm:.2f} mm, error {error}")
        self.tape_button.setEnabled(fitted is not None)
        self.tape_note.setVisible(fitted is None)

        refusal = next((text for text in (self._refused.get("length"), self._refused.get("true"),
                                          self.stick_tool.refusal, self.tape_tool.refusal) if text), None)
        result = None if check is None else (f"Tape check: {check.measured_mm:.2f} mm measured, "
                                             f"{self.tape_tool.true_mm:.2f} mm true, error {error}.")
        if refusal:
            kind, text = "problem", refusal
            state, hint = "attention", refusal
        elif check is not None and not check.ok:
            kind, text = "problem", f"{result} Does not pass (limit {LIMIT})."
            state = "attention"
            hint = f"The tape check does not pass: {error} (limit {LIMIT}). Click Redo and place the stick again."
        elif scale is not None and scale.too_short:
            kind = "warning"
            text = hint = (f"The stick is {scale.length_px:.1f} px long. Under {geometry.MIN_STICK_PX:g} px the "
                           "scale is not precise. Use a longer length on the ruler.")
            state = "attention"
        elif scale_text:
            kind, text = ("success", f"{result} Passes (limit {LIMIT}).") if check is not None else (None, "")
            state, hint = "done", f"Scale: {scale_text}."
            if check is not None:
                hint += f" Tape check: {error}, passes (limit {LIMIT})."
        else:
            kind, text, state, hint = None, "", "todo", self._start_hint
        self.message.show_message(kind, text, self._status)
        self._panel.set_state(state)
        self._panel.set_hint(hint)


def build(window) -> QWidget:
    """The controls of panel 3 for `window` (the main window), to go under the panel's hint line."""
    return CalibrationPanel(window)
