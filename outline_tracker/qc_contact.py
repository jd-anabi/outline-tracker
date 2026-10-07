"""How close the outlines of two tracks come: the geometry behind the CONTACT flag (SPEC 9; X6).

The part of `qc` that compares tracks with each other. Two tracks are in contact on a video frame
when both are visible there and their stored outlines are at most `contact_limit` apart: 3 px, or
two grid cells of the coarser of the two images the model saw, whichever is more. Outlines that
overlap, or lie one inside the other, are 0 apart.

Units and frame: image pixels in the image plane (u to the right, v down, pixel centers at +0.5),
the frame in which the outlines are stored (`outline_px` of results.npz, 256 points per frame). A
polygon is an array [m, 2] of points; it is closed, the last point joined to the first. Distances
do not depend on the sense in which a polygon runs or on which way v points, so nothing here needs
world coordinates.

Comparing two outlines exactly takes every point of one against every edge of the other. Most
pairs of tracks are far apart on most frames, and their bounding boxes say so at once: only the
frames whose boxes are within the limit are compared exactly.

No Qt, no torch.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from outline_tracker.results import TrackArrays

LIMIT_MIN_PX = 3.0   # outlines this close are in contact whatever the grid, px (SPEC 9)
LIMIT_CELLS = 2.0    # ... and so are outlines within this many grid cells (SPEC 9; X6)
_SLACK = 1e-6        # px added to the limit for the bounding boxes, so that rounding there decides nothing
_CHUNK = 8           # frames compared at once: each takes several arrays of 256 x 256 numbers


def contact_limit(cell_a_px, cell_b_px):
    """The distance in px at or below which two outlines are in contact (decision X6):
    max(3 px, 2 grid cells), a grid cell being the larger of the two tracks' `cell_px` on that
    frame (px per cell of the image the model saw). Numbers or arrays of one shape."""
    return np.maximum(LIMIT_MIN_PX, LIMIT_CELLS * np.maximum(cell_a_px, cell_b_px))


def polygon_distance(polygon_a, polygon_b) -> float:
    """The least distance between two closed polygons [m, 2] and [n, 2], in the unit of their
    points (px for stored outlines); 0 when they touch, cross, or one lies inside the other.

    Both polygons are given in one plane, for CONTACT the image plane (u, v) in px; the result
    does not depend on the sense in which they run. Each is taken as the region its points
    enclose (even-odd rule). A polygon whose points all coincide is a point.
    """
    first, second = (np.asarray(polygon, float)[None] for polygon in (polygon_a, polygon_b))
    return float(_distances(first, second)[0])


def contact_marks(arrays_by_track: Mapping[str, TrackArrays]) -> dict[str, np.ndarray]:
    """For every track a boolean per row: on that video frame the track's outline is in contact
    with the outline of at least one other track.

    `arrays_by_track`: the tracks' records from the results store, in image px; rows of
    different tracks are matched by their video frame number (`frames`), not by position. Only
    frames on which both tracks are visible count; both tracks of a pair get the mark. The limit
    is `contact_limit` of the two tracks' `cell_px` on that frame, in px.
    """
    marks = {track_id: np.zeros(len(arrays), bool) for track_id, arrays in arrays_by_track.items()}
    boxes = {track_id: _bounding_boxes(arrays) for track_id, arrays in arrays_by_track.items()}
    ids = list(arrays_by_track)
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            rows_a, rows_b = _contact_rows(arrays_by_track[id_a], boxes[id_a], arrays_by_track[id_b], boxes[id_b])
            marks[id_a][rows_a] = True
            marks[id_b][rows_b] = True
    return marks


def _bounding_boxes(arrays: TrackArrays) -> tuple[np.ndarray, np.ndarray]:
    """(low [n, 2], high [n, 2]): the smallest and largest (u, v) of each row's outline, px; NaN
    on lost rows."""
    seen = arrays.visible.astype(bool)
    low, high = np.full((len(arrays), 2), np.nan), np.full((len(arrays), 2), np.nan)
    low[seen] = arrays.outline_px[seen].min(axis=1)
    high[seen] = arrays.outline_px[seen].max(axis=1)
    return low, high


def _contact_rows(a: TrackArrays, box_a, b: TrackArrays, box_b) -> tuple[np.ndarray, np.ndarray]:
    """The rows of track `a` and of track `b` (equally many, the same video frames) on which the
    two are in contact. `box_a`, `box_b`: their `_bounding_boxes`."""
    _, rows_a, rows_b = np.intersect1d(a.frames, b.frames, assume_unique=True, return_indices=True)
    both = a.visible[rows_a].astype(bool) & b.visible[rows_b].astype(bool)
    rows_a, rows_b = rows_a[both], rows_b[both]
    limit = contact_limit(a.cell_px[rows_a], b.cell_px[rows_b])
    # Two boxes are never farther apart than the outlines inside them.
    between = np.maximum(np.maximum(box_a[0][rows_a] - box_b[1][rows_b], box_b[0][rows_b] - box_a[1][rows_a]), 0.0)
    near = np.flatnonzero(np.hypot(between[:, 0], between[:, 1]) <= limit + _SLACK)
    touching = np.zeros(len(near), bool)
    for start in range(0, len(near), _CHUNK):
        part = near[start:start + _CHUNK]
        apart = _distances(a.outline_px[rows_a[part]].astype(np.float64), b.outline_px[rows_b[part]].astype(np.float64))
        touching[start:start + _CHUNK] = apart <= limit[part]
    return rows_a[near[touching]], rows_b[near[touching]]


def _distances(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The least distance between the closed polygons p[i] and q[i], for stacks p [c, m, 2] and
    q [c, n, 2] in one plane and unit: [c], 0 where the two overlap.

    Two polygons whose edges do not cross are nearest at a corner of one of them, so the distance
    is the least one from a point of either to an edge of the other. Edges that cross properly
    make it 0, and so does one polygon inside the other: then no edges cross and every point of
    the inner one is inside the outer one, so testing one point of each is enough.
    """
    apart = np.sqrt(np.minimum(_least_square(p, q), _least_square(q, p)))
    overlap = _edges_cross(p, q) | _inside(p[:, 0], q) | _inside(q[:, 0], p)
    return np.where(overlap, 0.0, apart)


def _least_square(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """[c]: the square of the least distance from any of `points` [c, m, 2] to any edge of the
    closed `polygon` [c, n, 2]. An edge without length is its point."""
    ax, ay = polygon[:, None, :, 0], polygon[:, None, :, 1]             # edge starts, [c, 1, n]
    following = np.roll(polygon, -1, axis=1)
    ex, ey = following[:, None, :, 0] - ax, following[:, None, :, 1] - ay
    wx, wy = points[:, :, None, 0] - ax, points[:, :, None, 1] - ay     # from the edge start to the point, [c, m, n]
    length2 = ex * ex + ey * ey
    along = np.clip((wx * ex + wy * ey) / np.where(length2 > 0, length2, 1.0), 0.0, 1.0)
    dx, dy = wx - along * ex, wy - along * ey
    return (dx * dx + dy * dy).min(axis=(1, 2))


def _edges_cross(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """[c]: an edge of p[i] crosses an edge of q[i] properly: the ends of each lie strictly on
    different sides of the other. (Edges that only touch are 0 apart by `_least_square`.)"""
    p_side, q_side = _sides(p, q), _sides(q, p)             # [c, m, n] and [c, n, m]
    p_splits = p_side * np.roll(p_side, -1, axis=1) < 0     # edge i of p has an end on each side of edge j of q
    q_splits = q_side * np.roll(q_side, -1, axis=1) < 0     # and edge j of q one on each side of edge i of p
    return (p_splits & q_splits.transpose(0, 2, 1)).any(axis=(1, 2))


def _sides(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """[c, m, n]: on which side of the line through edge j of `polygon` [c, n, 2] the point i of
    `points` [c, m, 2] lies: the cross product (edge) x (point - edge start), positive on one
    side, negative on the other, 0 on the line."""
    ax, ay = polygon[:, None, :, 0], polygon[:, None, :, 1]
    following = np.roll(polygon, -1, axis=1)
    ex, ey = following[:, None, :, 0] - ax, following[:, None, :, 1] - ay
    return ex * (points[:, :, None, 1] - ay) - ey * (points[:, :, None, 0] - ax)


def _inside(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """[c]: point[i] (x, y) lies inside the closed polygon[i] [n, 2] (even-odd rule: a ray from
    the point toward +x crosses the outline an odd number of times)."""
    x, y = point[:, None, 0], point[:, None, 1]
    x1, y1 = polygon[..., 0], polygon[..., 1]
    x2, y2 = np.roll(x1, -1, axis=1), np.roll(y1, -1, axis=1)
    spans = (y1 > y) != (y2 > y)                                      # the edge passes the point's height
    rise = np.where(spans, y2 - y1, 1.0)
    crossing = spans & (x < x1 + (y - y1) * (x2 - x1) / rise)
    return np.count_nonzero(crossing, axis=1) % 2 == 1
