"""Tracks shared by the tests of qc (tests/test_qc.py: the nine flags of SPEC 9, each where it
belongs; tests/test_qc_contact.py: the distance between two outlines and the rule of CONTACT;
tests/test_qc_summary.py: the lines for run.log).

A track is made the way tracking and export make it: an analytic shape (tests/analytic_shapes.py)
or a rectangle of pixels goes through `measure_mask`, the records go into a `ResultsStore`, the
store's arrays through `derive_track`, and `compute_flags` gets both. Which frame should carry
which flag is written in the tests from the shapes alone.

Frames and units: shapes, records, clicks and model inputs are in image pixels (u to the right,
v down, pixel centers at +0.5); a model input is a box (column, row, width, height) in full-frame
px. Speeds are in mm per s in the world frame `WORLD` (0.0324 mm per px) at `FPS` frames per
second: between two frames that are 2 apart, 100 mm/s is 25.7 px.
"""

import analytic_shapes as shapes
import derive_helpers as h
import numpy as np

from outline_tracker import qc
from outline_tracker.derive import derive_track
from outline_tracker.measure import measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.session import Processing, Track

FULL_HD = h.FULL_HD            # the whole frame as the model's input: grid cells of 7.5 px
CLOSE = (572, 272, 256, 256)   # a model input of 256 px around CENTER: grid cells of 1 px
CENTER = h.CENTER
WORLD = h.TILTED
FPS = h.FPS
TILT = 0.3                     # rad, from +u toward +v: the long axis of the usual ellipse


def window(center, side):
    """The model input of a fine track: a box of `side` px around `center` (u, v), in px."""
    return (int(center[0]) - side // 2, int(center[1]) - side // 2, side, side)


def ellipse(frame, center=CENTER, a=40.0, b=15.0, angle=TILT, box=CLOSE):
    """The record of an ellipse with semi-axes `a` (along `direction(angle)`) and `b` px, seen
    through the model input `box`. With the defaults nothing is wrong with it: 80 px and 80 grid
    cells long, one piece, clear of the input's border, a core that is far from round."""
    return h.ellipse_record(frame, center, a, b, angle, half=int(max(a, b)) + 15, input_box=box)


def disk(frame, center, radius=10.0, box=FULL_HD):
    return h.disk_record(frame, center, radius, half=int(radius) + 15, input_box=box)


def block(frame, rows, cols, at=(600, 300), box=FULL_HD):
    """The record of a rectangle of `rows` x `cols` pixels whose top-left pixel is (column, row) =
    `at`, without logits: its area is rows * cols px exactly, its centroid the rectangle's middle,
    and its outline runs through the centers of its border pixels."""
    return pixels(frame, np.ones((rows, cols), bool), at, box)


def pixels(frame, mask, at=(0, 0), box=FULL_HD):
    """The record of exactly the pixels of `mask` [row, column], its top-left pixel at `at`."""
    return measure_mask(shapes.pixel_result(mask, offset=at), frame, box, "coarse")


def lost(frame, box=CLOSE):
    """What measure_mask returns when the object was not found on `frame`."""
    return measure_mask(shapes.pixel_result(np.zeros((0, 0), bool)), frame, box, "coarse")


def head(center=CENTER, angle=TILT):
    """A head click (u, v) in px: 30 px from `center` along `direction(angle)`."""
    return h.along(center, angle, 30.0)


def tracks(records, heads=None, world=WORLD, fps=FPS, **settings):
    """What export hands to qc: (derived_by_track, arrays_by_track, processing).

    `records` = {track id: [record, ...]}; `heads` = {track id: head click (u, v) in px} for the
    tracks that have one; `settings` are fields of session.Processing. No dish circle.
    """
    store = ResultsStore()
    for track_id, rows in records.items():
        for record in rows:
            store.put(track_id, record)
    processing = Processing(**settings)
    arrays = {track_id: store.arrays(track_id) for track_id in records}
    derived = {}
    for track_id, rows in records.items():
        click = (heads or {}).get(track_id)
        track = Track(id=track_id, start_frame=rows[0].frame,
                      head_px=None if click is None else [float(click[0]), float(click[1])])
        derived[track_id] = derive_track(arrays[track_id], track, world, fps, None, processing)
    return derived, arrays, processing


def flags_of(records, heads=None, world=WORLD, fps=FPS, **settings):
    """compute_flags for tracks made of `records` (see `tracks`): {track id: one text per row}."""
    derived, arrays, processing = tracks(records, heads, world, fps, **settings)
    return qc.compute_flags(derived, arrays, world, processing)


def rows_with(code, cells):
    """The rows (0, 1, ...) of a track whose flags cell holds `code`."""
    return [row for row, cell in enumerate(cells) if code in cell.split(";")]
