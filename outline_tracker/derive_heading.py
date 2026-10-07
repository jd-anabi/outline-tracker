"""The head direction of a track, frame by frame (SPEC 7.3; decision X17).

The part of `derive` that turns the core's second moments into a heading. The core's major axis
has no direction of its own; the head side is carried from frame to frame.

Units and frame: everything here is in world coordinates (SPEC 3.2): mm, y up, angles in rad
counterclockwise from +x. Covariances are (xx, xy, yy) in mm^2, as `WorldFrame.cov_to_world` gives
them for the stored (uu, uv, vv). Nothing here may be fed image coordinates: the reflection
between the two (det J = -1) reverses every angle.

The rule, with X17's reading of SPEC 7.3:
- A frame has no axis when it is lost, when its core has no extent (lambda1 = 0), or when the core
  is nearly round (lambda2 / lambda1 > 0.8). Such a frame is never the reference.
- The first heading takes its side from the head click on the track's start frame; without a
  click, from the core centroid's displacement between the first and the 10th visible frame (the
  last one of a shorter track). If that displacement, projected on the axis, is under 1 px, or a
  click decides nothing, the head is the end of the axis with x >= 0 (y > 0 if x = 0).
- Every later heading is the axis end within 90 degrees of the reference: the heading of the last
  earlier frame that had an axis. theta = theta_ref + the signed angle between them, in
  (-90, 90] degrees, so theta is continuous and may leave (-pi, pi].
- ORIENT marks a visible frame without an axis, a frame whose axis is more than 60 degrees from
  the reference (a jump) and a frame that used the core fallback. The last two still become the
  reference, so one jump marks one frame and not the rest of the track.
- A nearly round frame still gets a heading from its weak axis, by the same rule; a frame without
  any extent keeps the reference's heading (+x before there is one). A lost frame has none (NaN)
  and is not marked: it is LOST.

No Qt, no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ROUND_RATIO = 0.8                         # lambda2 / lambda1 of the core above this: no axis (SPEC 7.3)
JUMP_COS = math.cos(math.radians(60.0))   # |h . h_ref| below this: the axis jumped (SPEC 7.3)
GUESS_FRAMES = 10                         # the head guess looks at the first 10 visible frames (SPEC 7.3)
GUESS_MIN_PX = 1.0                        # a displacement along the axis under this decides nothing, px


@dataclass(frozen=True, eq=False)
class Headings:
    """The head direction of every frame of a track, in world coordinates (y up).

    heading [n, 2]: unit vectors. theta_rad [n]: their angle in rad, counterclockwise from +x,
    unwrapped. Both are NaN on lost frames. orient [n]: the frame gets the flag ORIENT (never a
    lost frame). headguess: the head side was guessed (no head click); it holds for the whole track.
    """

    heading: np.ndarray
    theta_rad: np.ndarray
    orient: np.ndarray
    headguess: bool


def principal_axes(cov) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues and major axis of covariances [..., 3] = (xx, xy, yy), in any length unit squared.

    Returns (lambda1 [...], lambda2 [...], axis [..., 2]): lambda1 >= lambda2 >= 0 in the unit of
    the input, and the unit eigenvector of lambda1 in the same frame as the covariance, the one
    of its two directions with x >= 0. NaN rows stay NaN. A covariance without a preferred
    direction gives the axis (1, 0).
    """
    cov = np.asarray(cov, float)
    xx, xy, yy = cov[..., 0], cov[..., 1], cov[..., 2]
    mean, spread = (xx + yy) / 2.0, np.hypot((xx - yy) / 2.0, xy)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    return mean + spread, np.maximum(mean - spread, 0.0), np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def track_headings(cov_core, core_xy, visible, core_fallback, head_xy, start_row, pixel_mm) -> Headings:
    """The heading of every frame of one track (the module's text gives the rule).

    One row per tracked frame, in frame order. cov_core [n, 3]: the core's covariance (xx, xy, yy)
    in world mm^2. core_xy [n, 2]: the core centroid in world mm. visible [n] and core_fallback [n]:
    booleans from the records. head_xy: the head click in world mm (x, y), or None. start_row: the
    row of the track's start frame, where the click was made, or None if that frame has no row
    (the first visible row is used then, as it is when the start frame is lost). pixel_mm: the
    scale k in mm per px, which turns the 1 px limit of the head guess into mm.
    """
    visible = np.asarray(visible, bool)
    core_xy = np.asarray(core_xy, float)
    lambda1, lambda2, axis = principal_axes(cov_core)
    with np.errstate(invalid="ignore"):
        has_axis = visible & (lambda1 > 0) & np.isfinite(axis).all(axis=-1)
        defined = has_axis & (lambda2 <= ROUND_RATIO * lambda1)
    rows = np.flatnonzero(visible)
    guide, least = _guide(core_xy, rows, visible, head_xy, start_row, pixel_mm)

    n = len(visible)
    heading, theta, orient = np.full((n, 2), np.nan), np.full(n, np.nan), np.zeros(n, bool)
    reference = None   # (heading, theta) of the last frame that had an axis
    for i in rows:
        jumped = False
        if reference is None:
            h = _first_side(axis[i], guide, least) if has_axis[i] else np.array([1.0, 0.0])
            angle = math.atan2(h[1], h[0])
        elif has_axis[i]:
            ref_h, ref_angle = reference
            turn = math.atan2(ref_h[0] * axis[i, 1] - ref_h[1] * axis[i, 0], ref_h @ axis[i])   # in (-pi, pi]
            h = axis[i]
            if turn > math.pi / 2:
                turn, h = turn - math.pi, -h
            elif turn <= -math.pi / 2:
                turn, h = turn + math.pi, -h
            angle = ref_angle + turn
            jumped = math.cos(turn) < JUMP_COS
        else:
            h, angle = reference
        heading[i], theta[i] = h, angle
        orient[i] = (not defined[i]) or bool(core_fallback[i]) or jumped
        if defined[i]:
            reference = (h, angle)
    return Headings(heading, theta, orient, head_xy is None)


def _guide(core_xy, rows, visible, head_xy, start_row, pixel_mm):
    """What the first heading should point along, and how long its projection on the axis must be
    to decide: (vector in world mm or None, length in mm)."""
    if len(rows) == 0:
        return None, 0.0
    if head_xy is not None:
        row = start_row if start_row is not None and visible[start_row] else rows[0]
        guide, least = np.asarray(head_xy, float) - core_xy[row], 0.0
    else:
        guide = core_xy[rows[min(GUESS_FRAMES, len(rows)) - 1]] - core_xy[rows[0]]
        least = GUESS_MIN_PX * pixel_mm
    return (guide if np.isfinite(guide).all() else None), least


def _first_side(axis, guide, least):
    """The end of `axis` (a unit vector) that the guide points to; the end with x >= 0 when the
    guide's projection on the axis is shorter than `least` or is nothing at all."""
    projection = 0.0 if guide is None else float(axis @ guide)
    if projection != 0.0 and abs(projection) >= least:
        return axis if projection > 0 else -axis
    return axis if axis[0] > 0 or (axis[0] == 0 and axis[1] > 0) else -axis
