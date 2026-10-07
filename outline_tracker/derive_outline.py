"""Geometry of an outline polygon: rays, the radial profile, the hull, resampling (SPEC 7.4, 7.6, 7.7).

The part of `derive` that works on polygons. A polygon is an array [m, 2] of points (x, y); it is
closed, the last point joined to the first.

Units and frame: every function works in the plane its points are given in and returns lengths in
the unit of those points and areas in that unit squared: mm for an outline mapped to world
coordinates, px for a stored outline (`outline_px` of results.npz). "Counterclockwise" and every
angle mean from +x toward +y of that plane. In world coordinates (y up) that is counterclockwise
on screen. In image coordinates (v down) it is clockwise on screen, so anything that depends on
the sense of rotation must be given world points (SPEC 3.2). Lengths, hull and Feret diameter do
not depend on it.

No Qt, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SLACK = 1e-9   # a ray through a vertex meets both of its segments: the ends count, with this margin
_CHUNK = 32     # frames worked on at once: rays x segments x frames numbers are in memory together


def ray_angles(step_deg) -> np.ndarray:
    """The angles of the radial profile (SPEC 7.6): 0, step_deg, 2 step_deg, ... below 360 degrees,
    returned in rad. Raises ValueError unless `step_deg` (the setting `radial_step_deg`, in degrees)
    is a positive number that divides 360."""
    step = float(step_deg)
    count = round(360.0 / step) if np.isfinite(step) and step > 0 else 0
    if count < 1 or abs(count * step - 360.0) > 1e-9:
        raise ValueError(f"radial_step_deg = {step_deg!r}: the step of the radial profile must be a positive "
                         f"number of degrees that divides 360.")
    return np.radians(step * np.arange(count))


def fan(heading, angles) -> np.ndarray:
    """The unit vectors at `angles` (rad, [k]) counterclockwise from `heading` ([..., 2], unit
    vectors in the points' plane): [..., k, 2]."""
    heading = np.asarray(heading, float)
    hx, hy = heading[..., None, 0], heading[..., None, 1]
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack([hx * cos - hy * sin, hx * sin + hy * cos], axis=-1)


def signed_area(polygon) -> np.ndarray:
    """Shoelace area of closed polygons [..., m, 2], in the points' unit squared: positive when the
    points run counterclockwise (from +x toward +y), negative when clockwise."""
    polygon = np.asarray(polygon, float)
    x, y = polygon[..., 0], polygon[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)


def farthest_crossings(polygon, origin, directions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where rays leave a closed polygon for good: the farthest crossing of each ray (SPEC 7.6).

    `polygon` is [..., m, 2], `origin` [..., 2] is where the rays start and `directions` [..., k, 2]
    are their unit vectors, all in one plane and unit. Every ray is tested against every segment
    at once. Returns (distance, segment, along), each [..., k]: the distance from the origin to the
    ray's farthest crossing (NaN if the ray meets no segment), the index of the segment it lies on
    (from point `segment` to the next) and how far along that segment, 0 to 1. The polygon may run
    either way; a ray along a segment does not count as crossing that segment.
    """
    polygon, origin, directions = (np.asarray(x, float) for x in (polygon, origin, directions))
    a = polygon - origin[..., None, :]
    e = np.roll(polygon, -1, axis=-2) - polygon
    ax, ay, ex, ey = a[..., None, :, 0], a[..., None, :, 1], e[..., None, :, 0], e[..., None, :, 1]
    dx, dy = directions[..., :, None, 0], directions[..., :, None, 1]
    # origin + t d = p + s e, with a = p - origin: t = (a x e) / (d x e) and s = (a x d) / (d x e)
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ax * ey - ay * ex) / den
        s = (ax * dy - ay * dx) / den
        hit = (den != 0) & (t >= 0) & (s >= -_SLACK) & (s <= 1 + _SLACK)
    t = np.where(hit, t, -np.inf)
    segment = t.argmax(axis=-1)
    far = np.take_along_axis(t, segment[..., None], axis=-1)[..., 0]
    along = np.take_along_axis(s, segment[..., None], axis=-1)[..., 0]
    found = far > -np.inf
    return np.where(found, far, np.nan), segment, np.where(found, np.clip(along, 0.0, 1.0), 0.0)


def radial_profile(polygon, center, heading, step_deg=5) -> np.ndarray:
    """The radial profile r(phi_j) of one outline polygon (SPEC 7.6), [360 / step_deg].

    `polygon` is [m, 2], `center` (x, y) is the body center and `heading` (hx, hy) the head
    direction (any length but 0), all in one plane and unit; for the profile of SPEC 7.6 that plane is
    the world frame (mm, y up). phi_j = j `step_deg` degrees is measured counterclockwise (from +x
    toward +y) from the heading. r is the distance from the center to the farthest crossing of the
    ray at phi_j with the polygon, in the points' unit: the outline itself when it is star-shaped
    about the center, else its outer envelope; NaN where the ray misses the polygon.
    Raises ValueError unless `step_deg` is a positive number that divides 360.
    """
    heading = np.asarray(heading, float)
    directions = fan(heading / np.hypot(heading[0], heading[1]), ray_angles(step_deg))
    return farthest_crossings(polygon, center, directions)[0]


def hull_area_and_feret(points) -> tuple[float, float]:
    """(area of the convex hull, maximum Feret diameter) of points [m, 2] (SPEC 7.7): the area in
    the points' unit squared and the largest distance between two hull corners in the points'
    unit. (NaN, NaN) when there is no hull: fewer than 3 points, a point that is not finite, or
    all points on one line."""
    from scipy.spatial import ConvexHull, QhullError   # imported here: only export and previews need it
    from scipy.spatial.distance import pdist

    points = np.asarray(points, float)
    if points.ndim != 2 or len(points) < 3 or not np.isfinite(points).all():
        return float("nan"), float("nan")
    try:
        hull = ConvexHull(points - points.mean(axis=0))
    except QhullError:
        return float("nan"), float("nan")
    return float(hull.volume), float(pdist(hull.points[hull.vertices]).max())


def feret_max(points) -> float:
    """The maximum Feret diameter of an outline (SPEC 7.7): the largest distance between two of
    its points, in the unit of the points. For a stored outline (`outline_px`, [256, 2] image
    points in px) the result is in px; it does not depend on the frame's orientation. NaN when
    there is no convex hull: a lost frame's NaN outline, fewer than 3 points, points on one line.
    """
    return hull_area_and_feret(points)[1]


def resample_from(polygon: np.ndarray, arc: np.ndarray, start: float, count: int) -> np.ndarray:
    """`count` points equally spaced along a closed polygon [m, 2], the first one at arc length
    `start` from the polygon's first point, going on in the polygon's own sense. `arc` [m + 1]
    holds the arc length at each point and, last, the whole perimeter. Points, `arc` and `start`
    share one unit. A polygon without length gives `count` copies of its first point."""
    total = arc[-1]
    if not total > 0:
        return np.repeat(polygon[:1], count, axis=0)
    at = (start + np.arange(count) * (total / count)) % total
    closed = np.vstack([polygon, polygon[:1]])
    return np.column_stack([np.interp(at, arc, closed[:, 0]), np.interp(at, arc, closed[:, 1])])


@dataclass(frozen=True, eq=False)
class OutlineQuantities:
    """What `outline_quantities` measures on the outlines of a track, one row per frame, in the
    unit of the polygons given (mm for world polygons) and that unit squared for the areas.

    perimeter [n]: polygon length. area [n]: polygon area (positive). hull_area [n] and feret [n]:
    area of the convex hull and largest distance between two hull corners (NaN without a hull).
    radial [n, k]: the radial profile. outline [n, count, 2]: the outline resampled from the head
    point, counterclockwise. Rows that could not be measured are NaN.
    """

    perimeter: np.ndarray
    area: np.ndarray
    hull_area: np.ndarray
    feret: np.ndarray
    radial: np.ndarray
    outline: np.ndarray


def outline_quantities(polygons, centers, headings, count: int, angles) -> OutlineQuantities:
    """Measure the outline polygon of every frame of a track (SPEC 7.4, 7.6, 7.7).

    `polygons` [n, m, 2], the body centers `centers` [n, 2] and the unit head directions
    `headings` [n, 2] are in world coordinates (mm, y up), one row per frame. `angles` [k] are
    the angles of the radial profile in rad, counterclockwise from the heading (`ray_angles`).
    Each polygon is first made to run counterclockwise (signed area > 0). The resampled outline
    has `count` points equally spaced along the polygon and starts at the head point: the farthest
    crossing of the ray from the center along the heading. If that ray meets no segment (the
    center lies outside the outline), it starts at the polygon point farthest along the heading.
    A row with a NaN in its polygon, center or heading (a lost frame) stays NaN.
    """
    polygons, centers, headings = (np.asarray(x, float) for x in (polygons, centers, headings))
    n = len(polygons)
    perimeter, area, hull_area, feret = (np.full(n, np.nan) for _ in range(4))
    radial = np.full((n, len(angles)), np.nan)
    outline = np.full((n, count, 2), np.nan)
    usable = (np.isfinite(polygons).all(axis=(1, 2)) & np.isfinite(centers).all(axis=1)
              & np.isfinite(headings).all(axis=1))
    rows = np.flatnonzero(usable)
    for chunk in (rows[i:i + _CHUNK] for i in range(0, len(rows), _CHUNK)):
        signed = signed_area(polygons[chunk])
        shapes = np.where((signed < 0)[:, None, None], polygons[chunk][:, ::-1], polygons[chunk])
        steps = np.hypot(*np.moveaxis(np.roll(shapes, -1, axis=1) - shapes, -1, 0))
        arcs = np.concatenate([np.zeros((len(chunk), 1)), np.cumsum(steps, axis=1)], axis=1)
        perimeter[chunk], area[chunk] = arcs[:, -1], np.abs(signed)
        # ray 0 is the head ray, rays 1.. are the radial profile
        rays = np.concatenate([headings[chunk][:, None, :], fan(headings[chunk], angles)], axis=1)
        distance, segment, along = farthest_crossings(shapes, centers[chunk], rays)
        radial[chunk] = distance[:, 1:]
        for j, row in enumerate(chunk):
            hull_area[row], feret[row] = hull_area_and_feret(shapes[j])
            if np.isfinite(distance[j, 0]):
                start = arcs[j, segment[j, 0]] + along[j, 0] * steps[j, segment[j, 0]]
            else:
                start = arcs[j, int(np.argmax(shapes[j] @ headings[row]))]
            outline[row] = resample_from(shapes[j], arcs[j], start, count)
    return OutlineQuantities(perimeter, area, hull_area, feret, radial, outline)
