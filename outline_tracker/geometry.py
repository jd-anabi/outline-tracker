"""Image <-> world transform, calibration stick, tape check, stopwatch, circle fit, frame grid, manifest.

Coordinates and units (SPEC 3):
- Image coordinates (u, v) are in px, in Tracker's convention: the origin is the top-left corner of
  the frame, u grows to the right, v downward, and the pixel in column c and row r has its center at
  (c + 0.5, r + 0.5). Every point named `*_px` here is such a (u, v) pair.
- World coordinates (x, y) are in mm in the user's axes, with y up on screen. The axes are given by
  the scale k (mm per px), the origin (u0, v0) in px and the angle alpha: the direction of +x,
  counterclockwise on screen from the image's rightward direction (Tracker's convention), in rad.
- The map is (x, y) = k J (u - u0, v - v0) with J = [[cos a, -sin a], [-sin a, -cos a]], det J = -1.
  Because of that reflection, the sense of an angle and the sign of an area computed on raw (u, v)
  are reversed: compute angles, signed areas and polygon orientation in world coordinates.
- Frames are the video's own frame numbers (whole numbers, frame 0 first); fps_true is in frames per
  second and t_s = frame / fps_true.

This module imports from tracker_io (last week's `Calibration` and manifest reader); tracker_io never
imports this module. No Qt, no torch.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from outline_tracker.tracker_io import Calibration, _fps_from_manifest

CLICK_SIGMA_PX = 0.5  # assumed precision of one click (SPEC 4.2), px
MIN_STICK_PX = 300.0  # a shorter calibration stick gets a warning (SPEC 4.2), px
TAPE_TOLERANCE = 0.01  # the tape check passes at |relative error| <= 1 % (SPEC 4.3)
MANIFEST_RELPATH = Path("data") / "manifest.csv"  # the course's manifest, inside a group's folder


# --------------------------------------------------------------------------- image <-> world

class MirroredCalibrationError(ValueError):
    """A fitted `Calibration` in the unflipped form: y grows downward on screen. Not a world frame."""


@dataclass(frozen=True)
class WorldFrame:
    """The user's axes on the image (SPEC 3.2).

    k_mm_per_px: the scale, mm per px, positive.
    alpha_rad: the direction of +x in rad, counterclockwise on screen from the image's rightward
        direction. With 0, x points right and y points up on screen.
    u0, v0: the origin of the axes in image px (Tracker's convention, see the module text).
    """

    k_mm_per_px: float
    alpha_rad: float
    u0: float
    v0: float

    def __post_init__(self) -> None:
        k, alpha, u0, v0 = (float(getattr(self, name)) for name in ("k_mm_per_px", "alpha_rad", "u0", "v0"))
        if not (math.isfinite(k) and k > 0):
            raise ValueError(f"The scale must be a positive number of mm per px, not {k}.")
        if not all(math.isfinite(value) for value in (alpha, u0, v0)):
            raise ValueError(f"The axis angle ({alpha} rad) and the origin ({u0}, {v0}) px must be finite numbers.")
        for name, value in (("k_mm_per_px", k), ("alpha_rad", alpha), ("u0", u0), ("v0", v0)):
            object.__setattr__(self, name, value)  # plain floats, whatever number type came in

    @property
    def J(self) -> np.ndarray:
        """The 2 x 2 matrix J of SPEC 3.2, no units: (x, y) = k J (u - u0, v - v0). It is a
        reflection (det J = -1, J J = 1), so (u - u0, v - v0) = J (x, y) / k."""
        c, s = math.cos(self.alpha_rad), math.sin(self.alpha_rad)
        return np.array([[c, -s], [-s, -c]])

    def to_world(self, u, v):
        """Image px (u, v) -> world mm (x, y), y up. Numbers or arrays of one shape; NaN stays NaN."""
        du, dv = np.asarray(u, float) - self.u0, np.asarray(v, float) - self.v0
        c, s, k = math.cos(self.alpha_rad), math.sin(self.alpha_rad), self.k_mm_per_px
        return k * (du * c - dv * s), -k * (du * s + dv * c)

    def to_px(self, x, y):
        """World mm (x, y) -> image px (u, v), the inverse of `to_world`. Numbers or arrays."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        c, s, k = math.cos(self.alpha_rad), math.sin(self.alpha_rad), self.k_mm_per_px
        return self.u0 + (x * c - y * s) / k, self.v0 - (x * s + y * c) / k

    def cov_to_world(self, cov_px) -> np.ndarray:
        """A covariance of image points in px^2 -> the covariance of the same points in world mm^2:
        k^2 J cov J^T (SPEC 7.3).

        Two layouts, told apart by the shape, and the result has the layout of the input:
        - [..., 3]: the stored form of results.npz (`cov_full`, `cov_core`), the elements uu, uv, vv
          along the last axis; the result holds xx, xy, yy. One row per frame; NaN rows stay NaN.
        - [..., 2, 2]: matrices [[uu, uv], [uv, vv]]; the result is [[xx, xy], [xy, yy]].
        The cross term changes sign with the reflection (alpha = 0: xy = -k^2 uv).
        """
        cov = np.asarray(cov_px, float)
        packed = cov.ndim >= 1 and cov.shape[-1] == 3
        if packed:
            cov = cov[..., [0, 1, 1, 2]].reshape(*cov.shape[:-1], 2, 2)
        elif cov.shape[-2:] != (2, 2):
            raise ValueError(f"A covariance must have the shape [..., 3] or [..., 2, 2], not {list(cov.shape)}.")
        m = self.k_mm_per_px * self.J
        world = m @ cov @ m.T
        return np.stack([world[..., 0, 0], world[..., 0, 1], world[..., 1, 1]], axis=-1) if packed else world


def world_frame_from_calibration(cal: Calibration) -> WorldFrame:
    """Last week's fitted `Calibration` (image px -> mm, flip form) as a `WorldFrame` (SPEC 3.2):
    k = hypot(a, c) in mm per px, alpha = atan2(-c, a) in rad, in (-pi, pi], and (u0, v0) =
    cal.to_px(0, 0) in image px.

    Raises `MirroredCalibrationError` (a ValueError) for the unflipped form, where y grows downward
    on screen: Tracker's y points up, so such a fit means the data are not Tracker's.
    """
    if not cal.flip:
        raise MirroredCalibrationError(
            "The calibration is mirrored: in these data y grows downward on screen, while Tracker's y axis "
            "points up. Export the track from Tracker again with its own x, y, pixelx and pixely columns."
        )
    k = math.hypot(cal.a, cal.c)
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"The calibration has no usable scale ({k} mm per px).")
    u0, v0 = cal.to_px(0.0, 0.0)
    return WorldFrame(k_mm_per_px=k, alpha_rad=math.atan2(-cal.c, cal.a), u0=float(u0), v0=float(v0))


def calibration_from_world_frame(wf: WorldFrame, rms_mm: float = 0.0) -> Calibration:
    """A `WorldFrame` as last week's `Calibration` in the flip form (image px -> mm, SPEC 3.2):
    a = k cos(alpha), c = -k sin(alpha), tx = -(a u0 + c v0), ty = -(c u0 - a v0), so that
    `Calibration.to_mm` equals `WorldFrame.to_world`.

    rms_mm fills the `Calibration`'s residual field, in mm: 0 for axes that were set, not fitted;
    pass the fit's residual to get a fitted calibration back unchanged.
    """
    a, c = wf.k_mm_per_px * math.cos(wf.alpha_rad), -wf.k_mm_per_px * math.sin(wf.alpha_rad)
    return Calibration(a=a, c=c, tx=-(a * wf.u0 + c * wf.v0), ty=-(c * wf.u0 - a * wf.v0), flip=True,
                       rms_mm=float(rms_mm))


# --------------------------------------------------------------------------- stick and tape

@dataclass(frozen=True)
class StickScale:
    """The scale from the calibration stick (SPEC 4.2).

    k_mm_per_px: mm per px. rel_uncertainty: sigma_k / k from click precision alone, a fraction
    (0.0008 is 0.08 %). length_px: the stick's length on the image, px. too_short: True when the
    stick spans fewer than 300 px (the user gets a warning; the scale is still valid).
    """

    k_mm_per_px: float
    rel_uncertainty: float
    length_px: float
    too_short: bool


@dataclass(frozen=True)
class TapeCheck:
    """The tape-measure check (SPEC 4.3).

    measured_mm: the distance between the two marks at the current scale, mm. rel_error: (measured -
    true) / true, signed, a fraction (0.005 is +0.5 %). ok: True at |rel_error| <= 1 %.
    """

    measured_mm: float
    rel_error: float
    ok: bool


def _point(p, what: str) -> tuple[float, float]:
    """An image point as two finite floats (px); ValueError naming `what` otherwise."""
    try:
        u, v = (float(c) for c in p)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a point (u, v) in px, not {p!r}.") from None
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError(f"{what} must be a point (u, v) in px with finite numbers, not {p!r}.")
    return u, v


def _positive(value, what: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{what} must be a positive number, not {value}.")
    return value


def stick_scale(p1, p2, length_mm, sigma_click_px=CLICK_SIGMA_PX) -> StickScale:
    """The scale from the two clicked ends of the calibration stick (SPEC 4.2).

    p1, p2: the ends, image px (u, v). length_mm: the stick's true length, mm. sigma_click_px: the
    precision of one click, px (an assumption, 0.5 by default). Returns k = length_mm / |p2 - p1| in
    mm per px and sigma_k / k = sqrt(2) sigma_click / |p2 - p1| (click precision only).
    Raises ValueError for a length that is not positive and for two identical ends.
    """
    (u1, v1), (u2, v2) = _point(p1, "The first end of the stick"), _point(p2, "The second end of the stick")
    length_mm = _positive(length_mm, "The stick length in mm")
    sigma = float(sigma_click_px)
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(f"The click precision sigma must be zero or a positive number of px, not {sigma}.")
    length_px = math.hypot(u2 - u1, v2 - v1)
    if length_px == 0:
        raise ValueError("The two ends of the stick are the same point. Click one end, then the other.")
    return StickScale(k_mm_per_px=length_mm / length_px, rel_uncertainty=math.sqrt(2) * sigma / length_px,
                      length_px=length_px, too_short=length_px < MIN_STICK_PX)


def tape_check(q1, q2, true_mm, k) -> TapeCheck:
    """Check the scale on two other ruler marks (SPEC 4.3).

    q1, q2: the marks, image px (u, v). true_mm: their true distance, mm. k: the scale, mm per px.
    Returns the measured distance k |q2 - q1| in mm, its signed relative error, and whether it
    passes (|error| <= 1 %). Raises ValueError for a distance or scale that is not positive and for
    two identical marks.
    """
    (u1, v1), (u2, v2) = _point(q1, "The first tape mark"), _point(q2, "The second tape mark")
    true_mm = _positive(true_mm, "The true tape distance in mm")
    k = _positive(k, "The scale in mm per px")
    length_px = math.hypot(u2 - u1, v2 - v1)
    if length_px == 0:
        raise ValueError("The two tape marks are the same point. Click one mark, then the other.")
    measured_mm = k * length_px
    rel_error = (measured_mm - true_mm) / true_mm
    return TapeCheck(measured_mm=measured_mm, rel_error=rel_error, ok=abs(rel_error) <= TAPE_TOLERANCE)


# --------------------------------------------------------------------------- time

def fps_from_stopwatch(frame_a, time_a_s, frame_b, time_b_s) -> float:
    """fps_true in frames per second from a filmed stopwatch (SPEC 4.1): (frame_b - frame_a) /
    (time_b_s - time_a_s), with the video's frame numbers and the stopwatch readings in s. The two
    moments may be given in either order.

    Raises ValueError when the readings are equal, and when the result is not a positive number.
    """
    frame_a, time_a_s, frame_b, time_b_s = (float(value) for value in (frame_a, time_a_s, frame_b, time_b_s))
    if not all(math.isfinite(value) for value in (frame_a, time_a_s, frame_b, time_b_s)):
        raise ValueError("The two frame numbers and the two stopwatch readings must be finite numbers.")
    if time_a_s == time_b_s:
        raise ValueError(f"The two stopwatch readings are equal ({time_a_s} s). Pick two frames that show "
                         "different times, several seconds apart.")
    fps = (frame_b - frame_a) / (time_b_s - time_a_s)
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"These frames and stopwatch readings give {fps} frames per second. The later frame "
                         "must show the later time.")
    return fps


fps_from_manifest = _fps_from_manifest  # (video: Path, manifest: Path) -> fps_true or None, as last week


# --------------------------------------------------------------------------- circle

@dataclass(frozen=True)
class CircleFit:
    """A circle fitted to clicked points (SPEC 4.4), all in image px: center_px = (u, v), radius_px,
    and rms_px, the RMS of the points' distances from the circle."""

    center_px: tuple[float, float]
    radius_px: float
    rms_px: float


def fit_circle(points_px) -> CircleFit:
    """Fit a circle to three or more points (the dish wall, SPEC 4.4).

    points_px: [n, 2] image points (u, v) in px. An algebraic (Kasa) fit gives the start; the result
    minimizes the geometric distances, sum (|p_i - c| - R)^2. With three points it is the circle
    through them. Returns the center (u, v), the radius and the RMS residual, all in px.
    Raises ValueError for fewer than three points and for points on one line.
    """
    from scipy.optimize import least_squares  # imported here: half a second that most commands never need

    pts = np.asarray(points_px, float)
    n = len(pts) if pts.ndim else 0
    if n < 3:
        raise ValueError(f"A circle needs at least 3 points; there {'is' if n == 1 else 'are'} {n}.")
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("The circle points must be points (u, v) in px with finite numbers.")
    mean = pts.mean(axis=0)
    q = pts - mean  # centered: better conditioned, and the same circle
    spread = np.linalg.svd(q, compute_uv=False)  # extent along and across the points' best line
    if spread[1] <= 1e-9 * spread[0]:
        raise ValueError("The circle points lie on one line (or are the same point), so no circle fits them. "
                         "Click points spread around the wall.")
    # Kasa: |q|^2 = 2 q.c + (R^2 - |c|^2) is linear in c and in the bracket
    sol, *_ = np.linalg.lstsq(np.column_stack([2.0 * q, np.ones(n)]), (q ** 2).sum(axis=1), rcond=None)
    start = np.array([sol[0], sol[1], math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0))])

    def residuals(p):
        return np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]) - p[2]

    def jacobian(p):
        d = q - p[:2]
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), np.finfo(float).tiny)
        return np.column_stack([-d[:, 0] / dist, -d[:, 1] / dist, -np.ones(n)])

    fit = least_squares(residuals, start, jac=jacobian, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    cu, cv, radius = (float(value) for value in fit.x)
    if not (fit.success and math.isfinite(cu) and math.isfinite(cv) and math.isfinite(radius) and radius > 0):
        raise ValueError("No circle could be fitted to these points. Click points spread around the wall.")
    rms = float(np.sqrt(np.mean(residuals(fit.x) ** 2)))
    return CircleFit(center_px=(float(mean[0]) + cu, float(mean[1]) + cv), radius_px=radius, rms_px=rms)


# --------------------------------------------------------------------------- the frame grid

def _whole(value, what: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be a whole number, not {value!r}.") from None


def grid_frames(start, end, step) -> range:
    """The clip's frame grid (SPEC 3.4): the frames start, start + step, ... up to and including
    `end` when it is on the grid. start and end are the clip's first and last frame (video frame
    numbers, end inclusive), step is the clip step in frames. Returns a `range`.

    Raises ValueError for a step below 1 and for an end before the start.
    """
    start, end, step = _whole(start, "The start frame"), _whole(end, "The end frame"), _whole(step, "The step")
    if step < 1:
        raise ValueError(f"The step must be at least 1 frame, not {step}.")
    if end < start:
        raise ValueError(f"The end frame ({end}) is before the start frame ({start}).")
    return range(start, end + 1, step)


def snap_to_grid(frame, start, step, end) -> tuple[int, bool]:
    """Move a typed frame number onto the clip's grid (SPEC 3.4, decision X20).

    Returns (grid frame, moved): the first grid frame at or after `frame`, clamped to the clip (a
    frame before the start gives the start, a frame after the last grid frame gives the last grid
    frame), and whether that differs from `frame`. All are video frame numbers; note the argument
    order (frame, start, step, end). Raises ValueError like `grid_frames`.
    """
    grid = grid_frames(start, end, step)
    frame = _whole(frame, "The frame")
    ahead = -(-(frame - grid.start) // grid.step)  # grid steps from the start, rounded up
    snapped = min(max(grid.start + ahead * grid.step, grid[0]), grid[-1])
    return snapped, snapped != frame


# --------------------------------------------------------------------------- manifest

def _manifest_row(video: Path, manifest: Path) -> pd.Series | None:
    """The one row of a manifest for this video, found as last week's `_fps_from_manifest` finds it:
    the `video_file` column and the video's name are compared without folder and extension, after
    dropping a trailing "_tracker" from the video's name. None when the file is missing or cannot be
    read, has no `video_file` column, or has no such row or several."""
    try:
        table = pd.read_csv(manifest, dtype=str)
    except Exception:  # missing, a folder, locked, not a table: no manifest (last week's rule)
        return None
    if "video_file" not in table.columns:
        return None
    stem = re.sub(r"_tracker$", "", video.stem)
    rows = table[table["video_file"].fillna("").map(lambda name: Path(name).stem == stem)]
    return rows.iloc[0] if len(rows) == 1 else None


def find_manifest(video, export=None, cwd=None) -> Path | None:
    """The manifest that lists this video (SPEC 4.1, decision X9), or None.

    Looks for `data/manifest.csv`, in this order: in the current folder (`cwd`; last week's rule, the
    folder itself only); then in the folder of the Tracker export and every folder above it; then in
    the video's folder and every folder above it. A manifest counts only if it has exactly one row
    for the video (matched like `fps_from_manifest`), so an unrelated manifest found first does not
    hide the right one. video, export and cwd are paths (relative ones are taken from `cwd`); the
    files themselves need not exist. Returns an absolute path.
    """
    here = Path.cwd() if cwd is None else Path(cwd)
    video = Path(video)
    folders = [here]
    for path in (export, video):
        if path is not None:
            folder = (here / path).resolve().parent
            folders += [folder, *folder.parents]
    for folder in dict.fromkeys(folders):  # each folder once, in order
        candidate = folder / MANIFEST_RELPATH
        if _manifest_row(video, candidate) is not None:
            return candidate.resolve()
    return None


def dish_mm_from_manifest(video, manifest) -> float | None:
    """The dish's inner diameter in mm from the manifest's `dish_mm` column, for the video's row
    (SPEC 4.4), or None when the manifest, the row or a positive number is missing. video and
    manifest are paths; the row is matched like `fps_from_manifest`."""
    row = _manifest_row(Path(video), Path(manifest))
    if row is None or "dish_mm" not in row.index:
        return None
    try:
        dish_mm = float(row["dish_mm"])
    except (TypeError, ValueError):
        return None
    return dish_mm if math.isfinite(dish_mm) and dish_mm > 0 else None
