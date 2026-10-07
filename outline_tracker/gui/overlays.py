"""Results on the picture (SPEC 6.3, 7.3, 10.1): for the frame the view shows, each track's stored
outline, its centroid, its id and its head mark, in the track's colour; for a fine track also the
square the model was shown.

What is drawn comes from results.npz as it is on disk, never from a job's memory. The file is read
again (`reload`) when a job says that it was saved (after every autosave and at the end of a run),
when another video is opened, and when the session changed while no job runs; the worker replaces
the file in one step, so a read never meets half a file. Showing another frame reads nothing.

- A track is drawn on a frame only if it has a record there and was found: a frame that was not
  tracked and a lost frame draw nothing for that track. A track the session does not list is not
  drawn.
- The head mark sits where the head direction (`derive_heading.track_headings`, the rule of SPEC
  7.3, the same that export uses) leaves the stored outline: at the outline point farthest along
  it from the core's center. It is filled when the head was clicked and an edge only when the head
  side is a guess (the flag HEADGUESS). The direction does not depend on the calibration, so it is
  worked out in px with y turned upward.
- The square of a fine track on a frame is `tracking_fine.crop_box` around the object's centroid
  on the tracked frame before (the last one on which it was found), with the side the record
  holds (256 grid cells); on the track's first record, around the centroid of that record, which
  is where the model found the object on that frame.

Coordinates: (u, v) in px of the full video frame, Tracker's convention (u to the right, v
downward, the pixel in column c and row r with its center at (c + 0.5, r + 0.5)); every item drawn
is in these coordinates. A box is (c0, r0, width, height) in whole px of the full frame. Frames
are video frame numbers. Pen widths and mark sizes are screen px. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Qt

import pyqtgraph as pg

from outline_tracker.derive_heading import track_headings
from outline_tracker.geometry import WorldFrame
from outline_tracker.gui.click_rules import ResultsOnDisk
from outline_tracker.gui.prompts import BLACK, CASING, LINE_WIDTHS, colour_of
from outline_tracker.measure import GRID_CELLS
from outline_tracker.results import ResultsStore, TrackArrays
from outline_tracker.tracking_fine import crop_box

DOT_SIZE, HEAD_SIZE = 6, 9   # the centroid's dot and the head's diamond, screen px
PX_UP = WorldFrame(1.0, 0.0, 0.0, 0.0)  # px as they are, with y upward: the frame the head rule works in


@dataclass
class Shown:
    """What is drawn of one track on the frame shown. outline: [256, 2] points (u, v) along the
    stored outline; center: the stored centroid (u, v); head: where the head mark is, None when
    the frame has no head direction; head_clicked: the head side was clicked (else it is a guess);
    box: the square a fine track's model was shown, (c0, r0, W, W), None for a coarse record. All
    in px of the full frame. `casing` under `line` (the outline, closed), `label` (the id) and
    `window` (the square, None without one) are the items in the view."""

    outline: np.ndarray
    center: tuple[float, float]
    head: tuple[float, float] | None
    head_clicked: bool
    box: tuple[int, int, int, int] | None
    casing: pg.PlotCurveItem
    line: pg.PlotCurveItem
    label: pg.TextItem
    window: pg.PlotCurveItem | None = None

    def items(self) -> list:
        """The items of this track in the view, lowest first."""
        return [item for item in (self.window, self.casing, self.line, self.label) if item is not None]


def head_points(arrays: TrackArrays, head_px, start_frame: int) -> tuple[np.ndarray, bool]:
    """Where the head mark of a track goes on each of its tracked frames: ([n, 2] points (u, v) in
    px of the full frame, NaN where the frame has no head direction; whether the head side is a
    guess). `arrays`: the track's records; `head_px`: the head click (u, v) in px, or None;
    `start_frame`: the video frame the click belongs to."""
    visible = arrays.visible.astype(bool)
    core = np.stack(PX_UP.to_world(arrays.core_u, arrays.core_v), axis=-1)
    head = None if head_px is None else np.array(PX_UP.to_world(*head_px), float)
    at_start = np.flatnonzero(arrays.frames == start_frame)
    with np.errstate(invalid="ignore", divide="ignore"):
        headings = track_headings(PX_UP.cov_to_world(arrays.cov_core), core, visible, arrays.core_fallback, head,
                                  int(at_start[0]) if len(at_start) else None, 1.0)
    direction = headings.heading * (1.0, -1.0)  # back to the picture, where v points down
    outline = arrays.outline_px.astype(float)
    from_core = outline - np.stack([arrays.core_u, arrays.core_v], axis=-1)[:, None, :]
    along = (from_core * direction[:, None, :]).sum(axis=-1)
    farthest = np.where(np.isfinite(along), along, -np.inf).argmax(axis=1)
    points = outline[np.arange(len(outline)), farthest]
    points[~np.isfinite(along).any(axis=1)] = np.nan
    return points, bool(headings.headguess)


def fine_box(arrays: TrackArrays, row: int) -> tuple[int, int, int, int] | None:
    """The square the model was shown for row `row` of a track's records (an index, not a frame
    number): (c0, r0, W, W) in whole px of the full frame, or None for a coarse record (see the
    module's text for where it is centered)."""
    if arrays.mode[row] != "fine":
        return None
    found_before = np.flatnonzero(arrays.visible[:row])
    at = int(found_before[-1]) if len(found_before) else row
    if not arrays.visible[at]:
        return None
    return crop_box((float(arrays.u[at]), float(arrays.v[at])), round(float(arrays.cell_px[row]) * GRID_CELLS))


class Overlays(QObject):
    """Draws the results of the run folder on the frame the window's view shows.

    `shown`: what is drawn now, by track id, in the session's order (`Shown`). `dots`, `heads`: the
    items that draw every track's centroid dot and head mark. `jobs` (a `worker_jobs.Jobs`) says
    when results.npz was written and whether a job runs.
    """

    def __init__(self, window, jobs):
        super().__init__(window)
        self._controller, self._view, self._jobs = window.controller, window.view, jobs
        self._results = ResultsOnDisk(window.controller)
        self._store = ResultsStore()
        self._heads: dict[tuple, tuple[np.ndarray, bool]] = {}  # per track, until the results are read again
        self.shown: dict[str, Shown] = {}
        self.dots, self.heads = pg.ScatterPlotItem(pxMode=True), pg.ScatterPlotItem(pxMode=True)
        for z, item in ((8, self.dots), (9, self.heads)):  # under the clicks and the outline of a preview
            item.setZValue(z)
            self._view.add_item(item)
        self._view.frame_changed.connect(self.refresh)
        window.controller.video_opened.connect(self.reload)
        window.controller.session_changed.connect(self._session_changed)
        jobs.saved.connect(self.reload)
        jobs.finished.connect(self.reload)

    def reload(self, *_) -> None:
        """Read results.npz of the run folder again, if it changed, and draw the frame shown. A
        file that cannot be read as results draws nothing."""
        try:
            self._store = self._results.now()
        except ValueError:
            self._store = ResultsStore()
        self._heads = {}
        self.refresh()

    def _session_changed(self) -> None:
        if self._jobs.running:  # the job says when the file was written
            self.refresh()
        else:  # an object was removed, perhaps, and its records with it
            self.reload()

    def refresh(self, *_) -> None:
        """Draw the stored results of the frame the view shows, in place of what was drawn."""
        for drawn in self.shown.values():
            for item in drawn.items():
                self._view.remove_item(item)
        self.shown = {}
        session, frame = self._controller.session, self._view.frame
        tracks = [] if session is None or frame is None or self._view.source is not self._controller.source else [
            track for track in session.tracks if track.id in self._store.track_ids]
        for track in tracks:
            arrays = self._store.arrays(track.id)
            try:
                row = arrays.row(frame)
            except KeyError:  # this frame was not tracked for the track
                continue
            if arrays.visible[row]:
                self.shown[track.id] = self._draw(track, arrays, row)
        colours = {track.id: colour_of(track) for track in tracks}
        self.dots.setData([{"pos": drawn.center, "size": DOT_SIZE, "symbol": "o", "pen": pg.mkPen(BLACK, width=1),
                            "brush": pg.mkBrush(colours[track_id])} for track_id, drawn in self.shown.items()])
        self.heads.setData([
            {"pos": drawn.head, "size": HEAD_SIZE, "symbol": "d",
             "pen": pg.mkPen(BLACK, width=1) if drawn.head_clicked else pg.mkPen(colours[track_id], width=2),
             "brush": pg.mkBrush(colours[track_id]) if drawn.head_clicked else pg.mkBrush(None)}
            for track_id, drawn in self.shown.items() if drawn.head is not None])

    def _draw(self, track, arrays: TrackArrays, row: int) -> Shown:
        """Put one track's record of row `row` on the picture, and return what was drawn."""
        colour = colour_of(track)
        key = (track.id, None if track.head_px is None else tuple(track.head_px), track.start_frame)
        if key not in self._heads:
            self._heads[key] = head_points(arrays, track.head_px, track.start_frame)
        heads, guessed = self._heads[key]
        outline = arrays.outline_px[row].astype(float)
        closed = np.vstack([outline, outline[:1]])
        line_width, casing_width = LINE_WIDTHS
        drawn = Shown(
            outline=outline, center=(float(arrays.u[row]), float(arrays.v[row])),
            head=(float(heads[row, 0]), float(heads[row, 1])) if np.isfinite(heads[row]).all() else None,
            head_clicked=not guessed, box=fine_box(arrays, row),
            casing=pg.PlotCurveItem(closed[:, 0], closed[:, 1], pen=pg.mkPen(CASING, width=casing_width),
                                    antialias=True),
            line=pg.PlotCurveItem(closed[:, 0], closed[:, 1], pen=pg.mkPen(colour, width=line_width), antialias=True),
            label=pg.TextItem(track.id, color=BLACK, fill=colour, anchor=(0, 1), border=pg.mkPen(BLACK)))
        drawn.label.setPos(float(closed[:, 0].max()), float(closed[:, 1].min()))  # above and right of the outline
        if drawn.box is not None:
            c0, r0, side, _ = drawn.box
            corners = np.array([(c0, r0), (c0 + side, r0), (c0 + side, r0 + side), (c0, r0 + side), (c0, r0)], float)
            drawn.window = pg.PlotCurveItem(corners[:, 0], corners[:, 1], antialias=True,
                                            pen=pg.mkPen(colour, width=line_width, style=Qt.PenStyle.DotLine))
        for z, item in enumerate(drawn.items(), start=4):
            item.setZValue(z)
            self._view.add_item(item)
        return drawn
