"""Tracks and truths shared by the tests of derive (tests/test_derive.py: the quantities the plan
lists; tests/test_derive_rules.py: lost rows, degenerate shapes, the reference rule, settings).

A track is made the way tracking makes it: an analytic shape (tests/analytic_shapes.py) goes
through `measure_mask`, the records go into a `ResultsStore`, and `derive_track` gets the store's
arrays. Truths are written here from the shape and from SPEC 3.2 alone.

Frames and units: shapes, records, clicks and the dish circle are in image pixels (u to the right,
v down, pixel centers at +0.5); angles given to the shapes are in rad in the (u, v) plane, from +u
toward +v. World values are in mm in the axes of a `WorldFrame`, y up; world angles are in rad,
counterclockwise from +x.
"""

import analytic_shapes as shapes
import numpy as np

from outline_tracker.derive import derive_track
from outline_tracker.geometry import WorldFrame
from outline_tracker.measure import measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.session import Processing, Track

FULL_HD = (0, 0, 1920, 1080)
K = 0.0324    # mm per px, the dish view
FPS = 240.0   # frames per second
UPRIGHT = WorldFrame(K, 0.0, 960.5, 540.5)
TILTED = WorldFrame(K, np.radians(25.0), 960.5, 540.5)   # +x points 25 degrees above "right" on screen
CENTER = (700.37, 400.81)   # a sub-pixel place well inside the frame, px


def record(distance, frame, center, *, half=70, input_box=FULL_HD, mode="coarse"):
    """What measure_mask returns for a shape on video frame `frame`. `distance(u, v)` is the
    shape's signed distance in px; it is evaluated on a raster of 2 `half` px around `center`."""
    origin = (int(center[0]) - half, int(center[1]) - half)
    u, v = shapes.pixel_centers(2 * half, 2 * half, origin)
    return measure_mask(shapes.to_result(distance(u, v), origin=origin), frame, input_box, mode)


def ellipse_record(frame, center, a, b, angle, **how):
    """The record of an ellipse with semi-axes `a` (along `direction(angle)`) and `b`, in px."""
    return record(lambda u, v: shapes.ellipse(u, v, center, a, b, angle), frame, center, **how)


def disk_record(frame, center, radius, **how):
    return record(lambda u, v: shapes.disk(u, v, center, radius), frame, center, **how)


def lost_record(frame, mode="coarse"):
    """What measure_mask returns when the object was not found on `frame`."""
    return measure_mask(shapes.pixel_result(np.zeros((0, 0), bool)), frame, FULL_HD, mode)


def arrays_of(records):
    """The records as the arrays of one track of the results store."""
    store = ResultsStore()
    for one in records:
        store.put("A", one)
    return store.arrays("A")


def derive(records, world=TILTED, *, head_px=None, circle=None, fps=FPS, start_frame=None, **settings):
    """derive_track for a track made of `records`; `head_px` is the head click (u, v) in px or None,
    `settings` are fields of session.Processing."""
    track = Track(id="A", start_frame=records[0].frame if start_frame is None else start_frame,
                  head_px=None if head_px is None else [float(head_px[0]), float(head_px[1])])
    return derive_track(arrays_of(records), track, world, fps, circle, Processing(**settings))


def along(center, angle, distance):
    """The image point `distance` px from `center` along `direction(angle)`."""
    step = shapes.direction(angle)
    return (center[0] + distance * step[0], center[1] + distance * step[1])


def world_vector(world, step_px):
    """The world displacement (mm) of the image displacement `step_px` = (du, dv) in px."""
    x0, y0 = world.to_world(0.0, 0.0)
    x1, y1 = world.to_world(step_px[0], step_px[1])
    return np.array([x1 - x0, y1 - y0])


def world_angle(world, angle):
    """The world angle (rad, counterclockwise from +x) of the image direction `direction(angle)`."""
    x, y = world_vector(world, shapes.direction(angle))
    return float(np.arctan2(y, x))


def apart_deg(angle, other):
    """The angle between two directions in degrees, 0 to 180 (360 degrees apart is 0)."""
    return np.abs((np.degrees(np.asarray(angle) - other) + 180.0) % 360.0 - 180.0)


def axis_apart_deg(angle, other):
    """The angle between two axes in degrees, 0 to 90: an axis has no direction (180 apart is 0)."""
    return np.abs((np.degrees(np.asarray(angle) - other) + 90.0) % 180.0 - 90.0)


def signed_area(points):
    """Shoelace area of a closed polygon [m, 2]: positive when it runs counterclockwise with y up."""
    x, y = np.asarray(points, float).T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def chords(points):
    """Distances from each point of a closed polygon to the next (the last one back to the first)."""
    points = np.asarray(points, float)
    return np.hypot(*(np.roll(points, -1, axis=0) - points).T)
