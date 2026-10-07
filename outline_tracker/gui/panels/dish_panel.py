"""Panel 4, Dish and axes (SPEC 4.4, 4.5, 10.1): the Circle tool and what the fit gives, the dish's
known diameter beside 2R, Origin to Center, the Axes tool and the axes angle, and "Crop to dish for
tracking".

The values live in the session (`circle`, `axes`, `processing.dish_crop`, SPEC 8.10) and are kept
there by the two tools (outline_tracker/gui/tools.py); this panel chooses the tools, takes the
typed values, and shows the session: the center, the radius and the RMS that `geometry.fit_circle`
returned, the radius in mm once there is a scale, and where the origin is.

Units: the center, the origin and the RMS are shown in px of the video frame (SPEC 3.1: u to the
right, v down), the radius in mm and px, the dish diameter and 2R in mm, the angle in degrees,
counterclockwise on screen from the image's rightward direction (SPEC 3.2).
"""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QWidget

from outline_tracker.gui.panels.calibration_panel import (DASH, GROUP_GAP, REDO_TIP, WIDEST, Message, NumberBox,
                                                          button, column, follow_tool, group_label, point_text, row,
                                                          value_label)
from outline_tracker.gui.tools import NEEDS_THREE, AxesTool, CircleTool, mm_per_px

NOT_KNOWN = "not known"  # what the dish diameter's box says at 0


class DishPanel(QWidget):
    """The controls of panel 4. `window` is the main window (its `controller`, `view`, `panels`).

    Parts: `circle_tool`, `axes_tool`; `circle_button`, `circle_redo`, `center_value`,
    `radius_value`, `rms_value`, `dish_box`, `two_r_value`; `axes_button`, `origin_button`,
    `origin_value`, `angle_box`; `crop_box`; `message`.
    """

    def __init__(self, window):
        super().__init__()
        self._controller, self._panel, self._status = window.controller, window.panels[3], window.statusBar()
        self._start_hint = self._panel.hint.text()
        self._shown = None                    # the session the panel shows
        self._refused: dict[str, str] = {}    # a box's name -> why its typed value was not taken
        self.axes_tool = AxesTool(window.controller, window.view, self)
        self.circle_tool = CircleTool(window.controller, window.view, self.axes_tool, self)

        self.circle_button = button("Circle", "Click points on the inner wall of the dish", checkable=True)
        self.circle_redo = button("Redo", REDO_TIP)
        self.circle_redo.setProperty("kind", "quiet")
        self.center_value, self.radius_value, self.rms_value = value_label(), value_label(), value_label()
        self.two_r_value, self.origin_value = value_label(), value_label()
        self.dish_box = NumberBox(" mm", 2, 0.0, "The dish's inner diameter, if you know it: 2R should be close to it")
        self.dish_box.setSpecialValueText(NOT_KNOWN)
        self.axes_button = button("Axes", "Click where the origin of the axes goes", checkable=True)
        self.origin_button = button("Origin to Center", "Put the origin at the center of the circle")
        self.angle_box = NumberBox("°", 1, -WIDEST, "The direction of +x: counterclockwise from the picture's right")
        self.crop_box = QCheckBox("Crop to dish for tracking")
        self.crop_box.setToolTip("Track only inside the square around the dish: faster, and sharper on the animals")
        self.message = Message()
        column(self,
               group_label("Dish"), row(self.circle_button, None, self.circle_redo),
               row("Center", self.center_value), row("Radius", self.radius_value), row("RMS", self.rms_value),
               row("Dish diameter", self.dish_box), row("2R", self.two_r_value),
               GROUP_GAP,
               group_label("Axes"), row(self.axes_button, self.origin_button, None),
               row("Origin", self.origin_value), row("Angle", self.angle_box),
               self.crop_box, self.message)

        follow_tool(window.view, [(self.circle_button, self.circle_tool), (self.axes_button, self.axes_tool)])
        self.circle_redo.clicked.connect(self.circle_tool.redo)
        self.origin_button.clicked.connect(self.axes_tool.origin_to_center)
        self.dish_box.valueChanged.connect(self.set_dish_mm)
        self.angle_box.valueChanged.connect(self.set_angle)
        self.crop_box.toggled.connect(self.set_crop)
        window.controller.video_opened.connect(self.refresh)
        window.controller.session_changed.connect(self.refresh)
        self.refresh()

    def set_dish_mm(self, mm: float) -> None:
        """Take the dish's inner diameter in mm, as typed; 0 means it is not known. A value that is
        not positive and finite is refused: the session keeps what it had, and a message says so."""
        self._take("dish", self.circle_tool.set_dish_mm, None if mm == 0 else mm)

    def set_angle(self, degrees: float) -> None:
        """Take the direction of +x in degrees, counterclockwise on screen from the image's
        rightward direction, as typed; a value that is no number is refused."""
        self._take("angle", self.axes_tool.set_angle, degrees)

    def set_crop(self, on: bool) -> None:
        """Switch "Crop to dish for tracking" (`processing.dish_crop`) on or off."""
        session = self._controller.session
        if session is not None and session.processing.dish_crop != bool(on):
            session.processing.dish_crop = bool(on)
            self._controller.touch()

    def _take(self, name: str, setter, value) -> None:
        self._refused.pop(name, None)
        try:
            setter(value)
        except ValueError as refused:
            self._refused[name] = str(refused)
        self.refresh()

    def refresh(self) -> None:
        """Show the session as it is now: the circle, the axes, the crop, and the panel's state."""
        session = self._controller.session
        self.setEnabled(session is not None)
        if session is None:
            return
        if session is not self._shown:
            self._shown = session
            self._refused.clear()
        circle, scale, origin = session.circle, mm_per_px(session), session.axes.origin_px
        radius = None
        if circle is not None:
            radius = f"{circle.radius_px:.1f} px"
            if scale:
                radius = f"{scale * circle.radius_px:.2f} mm ({radius})"
        self.center_value.setText(DASH if circle is None else point_text(circle.center_px))
        self.radius_value.setText(radius or DASH)
        self.rms_value.setText(DASH if circle is None else f"{circle.rms_px:.1f} px")
        self.two_r_value.setText(f"{2 * scale * circle.radius_px:.2f} mm" if circle is not None and scale else DASH)
        self.origin_value.setText(point_text(origin))
        self.dish_box.show_value(self.circle_tool.dish_mm or 0.0, "dish" in self._refused)
        self.angle_box.show_value(session.axes.angle_deg, "angle" in self._refused)
        self.crop_box.blockSignals(True)
        self.crop_box.setChecked(bool(session.processing.dish_crop))
        self.crop_box.blockSignals(False)
        self.origin_button.setEnabled(circle is not None)

        refusal = self._refused.get("dish") or self._refused.get("angle")
        at_origin = f"The origin is at {point_text(origin)}."
        if refusal:
            kind, text, state, hint = "problem", refusal, "attention", refusal
        elif self.circle_tool.refusal:
            text = hint = self.circle_tool.refusal
            kind, state = ("warning" if text == NEEDS_THREE else "problem"), "attention"
        elif circle is not None:
            centered = list(origin) == list(circle.center_px)
            kind, text, state = None, "", "done"
            hint = f"Dish: R = {radius}, RMS {circle.rms_px:.1f} px. " + ("The origin is at the center." if centered
                                                                         else at_origin)
        elif not session.processing.dish_crop and self.axes_tool.placed:
            kind, text, state, hint = None, "", "done", f"No dish circle. {at_origin}"
        else:
            kind, text, state, hint = None, "", "todo", self._start_hint
        self.message.show_message(kind, text, self._status)
        self._panel.set_state(state)
        self._panel.set_hint(hint)


def build(window) -> QWidget:
    """The controls of panel 4 for `window` (the main window), to go under the panel's hint line."""
    return DishPanel(window)
