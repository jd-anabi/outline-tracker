"""Analytic test shapes: implicit functions rasterized at pixel centers (SPEC 13.1, decision X16).

Shared by the tests of measure, derive and qc. A shape is a function d(u, v): its signed distance
to the shape's boundary in px, positive inside. The ground-truth mask is d > 0 at the pixel centers,
and d itself serves as the logits, so the 0-level set is the known boundary. Sizes given to a cv2
drawing call are never used as truth (a cv2 ellipse of nominal 80 x 30 px comes out 6% larger).

Frame and units: everything is in image pixels, Tracker's convention (SPEC 3.1): u to the right,
v downward, the pixel in column c and row r has its center at (c + 0.5, r + 0.5). Arrays are indexed
[row, column]. Angles are in radians in the (u, v) plane, from +u toward +v: `direction(angle)` is
the unit vector (cos, sin) in (u, v). Because v points down, that turn is CLOCKWISE on screen. This
module knows nothing about world coordinates: a test that needs a world angle maps the direction
vector through its own calibration.

The disk and the ellipse are the package's own shapes (outline_tracker/synthetic_shapes.py), asked in
this module's frame; tests/test_shapes_agree.py holds both to their closed-form outlines. The package's
body frame has xi toward the head and eta 90 degrees counterclockwise on screen from xi, so its heading
is minus this module's angle. The other shapes here have no equivalent in the package.
"""

from __future__ import annotations

import numpy as np

from outline_tracker.segmenter.base import MaskResult, crop_to_bbox
from outline_tracker.synthetic_shapes import Disk, Ellipse


def pixel_centers(height: int, width: int, origin: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """The pixel centers (u, v), in px, of a raster of `height` rows and `width` columns.

    `origin` = (column, row) of the raster's top-left pixel in the full frame, so the arrays hold
    full-frame coordinates: u[r, c] = origin[0] + c + 0.5 and v[r, c] = origin[1] + r + 0.5.
    """
    rows, cols = np.mgrid[0:height, 0:width]
    return cols + origin[0] + 0.5, rows + origin[1] + 0.5


def direction(angle: float) -> np.ndarray:
    """The unit vector (du, dv) at `angle` rad in the (u, v) plane, from +u toward +v (v is down)."""
    return np.array([np.cos(angle), np.sin(angle)])


def disk(u: np.ndarray, v: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    """Exact signed distance (px, positive inside) to the circle of `radius` px around `center` (u, v):
    the package's `Disk`, asked at the points as seen from its center (xi to the right, eta up)."""
    return Disk(radius).distance(u - center[0], -(v - center[1]))


def ellipse(u: np.ndarray, v: np.ndarray, center: tuple[float, float], a: float, b: float,
            angle: float) -> np.ndarray:
    """Signed distance (px, positive inside) to an ellipse, to first order near its boundary.

    `center` (u, v) and the semi-axes `a` (along `direction(angle)`) and `b` (across) are in px.
    It is the package's `Ellipse(a, b)` with its head along `direction(angle)`. With
    q = sqrt((x/a)^2 + (y/b)^2) in the ellipse's own axes, d = (1 - q) / |grad q|: exact in
    sign everywhere (inside is q < 1), and a distance within a pixel or so of the boundary. Deep
    inside it is capped at min(a, b).
    """
    e = direction(angle)
    du, dv = u - center[0], v - center[1]
    # the package's body frame: xi along the head, eta across it, counterclockwise on screen (v is down)
    return Ellipse(a, b).distance(du * e[0] + dv * e[1], du * e[1] - dv * e[0])


def capsule(u: np.ndarray, v: np.ndarray, p0: tuple[float, float], p1: tuple[float, float],
            width: float) -> np.ndarray:
    """Exact signed distance (px, positive inside) to a rod: the segment from `p0` to `p1`, both
    (u, v) in px, thickened to `width` px with round ends."""
    pu, pv = u - p0[0], v - p0[1]
    su, sv = p1[0] - p0[0], p1[1] - p0[1]
    along = np.clip((pu * su + pv * sv) / (su * su + sv * sv), 0.0, 1.0)
    return width / 2.0 - np.hypot(pu - along * su, pv - along * sv)


def union(*distances: np.ndarray) -> np.ndarray:
    """Signed distance of the union of shapes: the largest of their signed distances (px)."""
    return np.maximum.reduce(distances)


def body_with_rods(u: np.ndarray, v: np.ndarray, center: tuple[float, float], angle: float, *,
                   a: float = 23.5, b: float = 10.0, rod_angle: float = np.pi / 2, rod_reach: float = 30.0,
                   rod_width: float = 3.0, attach: float = 14.0) -> tuple[np.ndarray, np.ndarray]:
    """An ellipse body with two thin rods, like a nauplius with its antennae (SPEC 7.3, 13.1 Core).

    Returns (d_all, d_body): the signed distances (px) of the whole shape and of the body alone.
    The body is the ellipse of semi-axes `a`, `b` px around `center`, its long axis and head along
    `direction(angle)`. Each rod starts from the point `attach` px ahead of the center on the long
    axis, points `rod_angle` rad away from the head direction (one rod to each side), begins where
    that ray leaves the body and reaches `rod_reach` px beyond the body's surface; it is
    `rod_width` px thick. The defaults are the 47 x 20 px body with 3 px rods spread sideways.
    """
    head, side = direction(angle), direction(angle + np.pi / 2)
    body = ellipse(u, v, center, a, b, angle)
    parts = [body]
    for sign in (1.0, -1.0):
        # in the body's own axes (xi toward the head, eta across): start point and unit direction
        xi0, eta0 = attach, 0.0
        dxi, deta = np.cos(rod_angle), sign * np.sin(rod_angle)
        # the ray (xi0, eta0) + t (dxi, deta) leaves the ellipse where (xi/a)^2 + (eta/b)^2 = 1
        qa = (dxi / a) ** 2 + (deta / b) ** 2
        qb = 2.0 * (xi0 * dxi / a ** 2 + eta0 * deta / b ** 2)
        qc = (xi0 / a) ** 2 + (eta0 / b) ** 2 - 1.0
        t_out = (-qb + np.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
        ends = []
        for t in (t_out, t_out + rod_reach):
            xi, eta = xi0 + t * dxi, eta0 + t * deta
            ends.append((center[0] + xi * head[0] + eta * side[0], center[1] + xi * head[1] + eta * side[1]))
        parts.append(capsule(u, v, ends[0], ends[1], rod_width))
    return union(*parts), body


def to_result(distance: np.ndarray, *, origin: tuple[int, int] = (0, 0), pad: int | None = 8,
              logits: bool = True, score: float | None = None, obj_id: str = "A") -> MaskResult:
    """The `MaskResult` a perfect model would return for a shape, in full-frame pixels.

    `distance` is the shape's signed distance (px) on a raster whose top-left pixel is at `origin`
    = (column, row) of the full frame. The logits are that distance as float32 and the mask is
    logits > 0. The arrays are cut to the mask's bounding box plus `pad` px (clipped at the
    raster's border, as the model's image clips them); `pad=None` keeps the whole raster. The
    result's offset is in full-frame px. With `logits=False` the result carries no logits.
    """
    values = np.asarray(distance, np.float32)
    mask = values > 0
    if pad is None:
        result = MaskResult(obj_id, (0, 0), mask, values, score)
    else:
        result = crop_to_bbox(mask, values, pad, obj_id=obj_id, score=score)
    result.offset = (result.offset[0] + origin[0], result.offset[1] + origin[1])
    if not logits:
        result.logits = None
    return result


def blocks(shape: tuple[int, int], *rectangles: tuple[int, int, int, int]) -> np.ndarray:
    """A mask of `shape` (rows, columns) with the given pixel rectangles (row0, row1, col0, col1) set;
    row1 and col1 are one past the last pixel, as in a slice. Boolean, indexed [row, column], in px."""
    mask = np.zeros(shape, bool)
    for row0, row1, col0, col1 in rectangles:
        mask[row0:row1, col0:col1] = True
    return mask


def pixel_result(mask: np.ndarray, offset: tuple[int, int] = (0, 0), *, logits: bool = False,
                 score: float | None = None, obj_id: str = "A") -> MaskResult:
    """A `MaskResult` made of exactly the given pixels: no cropping, `offset` = (column, row) of
    the array's top-left pixel in the full frame, in px.

    With `logits=True` the logits are +1 on the mask and -1 off it, so their 0-level set runs
    along the pixel edges, halfway between a mask pixel's center and its neighbor's.
    """
    mask = np.asarray(mask, bool)
    values = np.where(mask, 1.0, -1.0).astype(np.float32) if logits else None
    return MaskResult(obj_id, (int(offset[0]), int(offset[1])), mask.copy(), values, score)


def box(u: np.ndarray, v: np.ndarray, center: tuple[float, float], half_length: float, half_width: float,
        angle: float) -> np.ndarray:
    """Exact signed distance (px, positive inside) to a rectangle around `center` (u, v), in px:
    2 `half_length` long along `direction(angle)` and 2 `half_width` wide across it."""
    e = direction(angle)
    du, dv = u - center[0], v - center[1]
    x = np.abs(du * e[0] + dv * e[1]) - half_length
    y = np.abs(-du * e[1] + dv * e[0]) - half_width
    return -(np.hypot(np.maximum(x, 0.0), np.maximum(y, 0.0)) + np.minimum(np.maximum(x, y), 0.0))


def l_shape(u: np.ndarray, v: np.ndarray, corner: tuple[float, float], size: float, arm: float,
            angle: float) -> np.ndarray:
    """Signed distance (px, positive inside) to an L: two arms `size` px long and `arm` px wide that
    share the square at the outer corner `corner` (u, v). One arm runs along p = `direction(angle)`,
    the other along q = `direction(angle + pi / 2)`. Its boundary is exact (the union of two exact
    rectangles); its area is size^2 - (size - arm)^2 px^2, and its convex hull, which closes the
    notch with one straight edge, has size^2 - (size - arm)^2 / 2 px^2.
    """
    p, q = direction(angle), direction(angle + np.pi / 2)
    half, thin = size / 2.0, arm / 2.0
    along_p = (corner[0] + half * p[0] + thin * q[0], corner[1] + half * p[1] + thin * q[1])
    along_q = (corner[0] + thin * p[0] + half * q[0], corner[1] + thin * p[1] + half * q[1])
    return union(box(u, v, along_p, half, thin, angle), box(u, v, along_q, thin, half, angle))
