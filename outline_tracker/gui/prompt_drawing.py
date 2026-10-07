"""What the point tools draw on the video (SPEC 5, 10.1): the look of the marks of the clicks and of
the outline the model gives for them, with the id and the window of a fine object. gui/prompts.py
(`Prompts`) decides what is drawn and puts the items on the view; this module makes them, and
gui/overlays.py draws the stored results with the same colours and widths. Every name here is also
importable from gui/prompts.py, where it was before the two were split.

The look is the design note's: marks told apart by shape (positive a disc with a black +, negative
a cross, head a white diamond), outlines in the track's colour on a black casing.

Coordinates: (u, v) in px of the full video frame, Tracker's convention (u to the right, v
downward, pixel centers at +0.5). A box is (c0, r0, width, height) in whole px of the full frame.
Pen widths and mark sizes are screen px. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import pyqtgraph as pg

from outline_tracker.tracking_fine import crop_box

BLACK, WHITE = "#000000", "#FFFFFF"
CASING = (0, 0, 0, 180)
MARK_SIZES = {"positive": 12, "negative": 14, "head": 13}  # screen px
MARK_SYMBOLS = {"positive": "o", "negative": "x", "head": "d"}
PLUS_SIZE = 7             # the black + on a positive mark
LINE_WIDTHS = (2, 4)      # an outline and its casing; one more each for the selected object


@dataclass
class Drawn:
    """The items that draw one object's outline: `casing` under `line` (the outline, closed),
    `label` (the id), and for a fine object `window`, its square, with `box` = (c0, r0, W, W) in
    whole px of the full frame; both None for a coarse object."""

    casing: pg.PlotCurveItem
    line: pg.PlotCurveItem
    label: pg.TextItem
    window: pg.PlotCurveItem | None = None
    box: tuple[int, int, int, int] | None = None

    def items(self) -> list:
        return [item for item in (self.window, self.casing, self.line, self.label) if item is not None]


def colour_of(track) -> str:
    """A track's colour as `#RRGGBB`; white for a track whose colour is no colour name."""
    return track.color if QColor.isValidColorName(track.color) else WHITE


def mark_spots(marks, tracks) -> tuple[list[dict], list[dict]]:
    """What two scatter items show for `marks` (`click_rules.Mark`s, (u, v) in px of the full
    frame) of `tracks`: (one spot per mark, in its track's colour, a head in white; the black +
    on each positive mark). Sizes are screen px."""
    colours = {track.id: colour_of(track) for track in tracks}
    edge = pg.mkPen(BLACK, width=1.5)
    points = [
        {"pos": (mark.u, mark.v), "size": MARK_SIZES[mark.kind], "symbol": MARK_SYMBOLS[mark.kind], "pen": edge,
         "brush": pg.mkBrush(WHITE if mark.kind == "head" else colours[mark.track_id])} for mark in marks]
    pluses = [{"pos": (mark.u, mark.v), "size": PLUS_SIZE, "symbol": "+", "pen": pg.mkPen(None),
               "brush": pg.mkBrush(BLACK)} for mark in marks if mark.kind == "positive"]
    return points, pluses


def draw_outline(track, hit, selected: bool) -> Drawn:
    """The items of the outline of `track` as the model found it (`hit`, a `worker.Found`: the
    outline and its center in px of the full frame, and the window the rule chooses, px): the
    closed line on its casing, one screen px wider each for the selected object, the id above and
    right of it, and for a fine object the dotted square of its window (the typed side in px, else
    the rule's). The items are not on a view yet."""
    colour = colour_of(track)
    line_width, casing_width = (width + selected for width in LINE_WIDTHS)
    closed = np.vstack([hit.outline, hit.outline[:1]])
    drawn = Drawn(
        casing=pg.PlotCurveItem(closed[:, 0], closed[:, 1], pen=pg.mkPen(CASING, width=casing_width),
                                antialias=True),
        line=pg.PlotCurveItem(closed[:, 0], closed[:, 1], pen=pg.mkPen(colour, width=line_width),
                              antialias=True),
        label=pg.TextItem(track.id, color=BLACK, fill=colour, anchor=(0, 1),
                          border=pg.mkPen(WHITE, width=2) if selected else pg.mkPen(BLACK)))
    drawn.label.setPos(float(closed[:, 0].max()), float(closed[:, 1].min()))  # above and right of the outline
    side = track.fine_window_px if isinstance(track.fine_window_px, int) else hit.auto_window
    if track.mode == "fine" and side and side > 0:
        drawn.box = c0, r0, _, _ = crop_box(hit.center, side)
        far_c, far_r = c0 + side, r0 + side
        corners = np.array([(c0, r0), (far_c, r0), (far_c, far_r), (c0, far_r), (c0, r0)], float)
        drawn.window = pg.PlotCurveItem(corners[:, 0], corners[:, 1], antialias=True,
                                        pen=pg.mkPen(colour, width=LINE_WIDTHS[0], style=Qt.PenStyle.DotLine))
    return drawn
