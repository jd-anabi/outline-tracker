"""World quantities from the stored pixel records and the current calibration (SPEC 7).

Tracking stores what it measured in pixels (results.npz). `derive_track` turns one track's arrays
into everything that carries a unit: positions, areas, axes, the heading, the outline in world and
body coordinates, the radial profile, the shape numbers, the resolution indicators and the
distances to the dish wall. Nothing here looks at a video or a model, so a new calibration, a new
fps_true or a new head click costs a call of `derive_track` and no tracking.

Units and frames:
- Input: image pixels (u, v), Tracker's convention (SPEC 3.1): u to the right, v down, pixel
  centers at +0.5; the records' covariances are (uu, uv, vv) in px^2.
- Output: world coordinates (SPEC 3.2): mm in the user's axes, y up; t_s = frame / fps_true in s.
  Every angle, signed area and sense of rotation is computed in world coordinates, because the
  map from the image is a reflection (det J = -1) and reverses them. theta is in rad,
  counterclockwise from +x.
- Body frame (SPEC 7.5): xi = (r - r_core) . h points to the head, eta is 90 degrees
  counterclockwise from xi in world coordinates (y up).

Which number comes from where:
- `area_mm2` is k^2 times the mask's pixel count. Solidity and circularity use the area of the
  outline polygon instead.
- Perimeter, Feret diameter, wall distance of the outline and the radial rays use the stored
  256-point outline, mapped to world coordinates and turned counterclockwise there.
- `major_mm`, `minor_mm` and `eccentricity` come from the full mask's moments (the stored
  `cov_full`); the heading and `core_x_mm`, `core_y_mm` from the core's (`derive_heading`).
- The resolution indicators `px_along_major`, `cells_along_major` and `shape_ok` are of the
  largest piece of the mask, the piece the outline goes around (`measure.largest_piece`), so that
  a few stray pixels far from the object cannot lengthen them; this replaces the full mask of
  SPEC 7.8 (decision 26 of docs/ROADMAP.md). With one piece that is the full mask, and `cov_full`
  gives them. With more pieces (`n_components` > 1) the row's stored mask crop is unpacked and its
  largest piece is measured: of pieces of equal size the one that the row's stored outline runs
  along. A crop without a pixel keeps the value of `cov_full`.

The polygon geometry is in `derive_outline`; its `radial_profile` and `feret_max` (the maximum
Feret diameter of an outline, also for one in px) are offered here too. No Qt, no torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from outline_tracker.derive_heading import principal_axes, track_headings
from outline_tracker.derive_outline import feret_max, outline_quantities, radial_profile, ray_angles
from outline_tracker.geometry import WorldFrame
from outline_tracker.measure import largest_piece
from outline_tracker.results import TrackArrays
from outline_tracker.session import Circle, Processing, Track

__all__ = ["DerivedTrack", "derive_track", "feret_max", "radial_profile"]


@dataclass(frozen=True, eq=False)
class DerivedTrack:
    """One track in world units: one row per tracked frame, frames ascending, lost frames kept.

    Every column of positions.csv and shapes.csv (`schema.POSITIONS`, `schema.SHAPES`) except
    `track_id` and `flags` is a field of the same name, an array [n]:
    - frame: video frame numbers. t_s: frame / fps_true, s. visible: 1 or 0. mode: "coarse", "fine".
    - x_mm, y_mm: the mask's area centroid in world mm (y up); u_px, v_px: the same point in
      image px. area_mm2: k^2 times the pixel count (0 on a lost frame).
    - major_mm, minor_mm, eccentricity: 4 sqrt(lambda) and sqrt(1 - lambda2 / lambda1) of the full
      mask. perimeter_mm, solidity, circularity, feret_max_mm: from the outline polygon.
    - theta_rad: the heading, rad counterclockwise from +x in world coordinates, unwrapped.
      core_x_mm, core_y_mm: the core centroid, world mm. core_frac, n_components,
      largest_fraction: as stored.
    - px_along_major: the major axis 4 sqrt(lambda1) of the largest piece of the mask, in camera
      px (with one piece: of the full mask). cells_along_major: the same in model grid cells.
      shape_ok: 1 if the smaller of the two reaches `shape_ok_min`, else 0.
    - wall_dist_centroid_mm, wall_dist_min_mm: dish radius minus the centroid's distance, and
      minus the outline's largest distance, from the dish center, mm; negative outside the
      circle; NaN without a circle.
    On a lost frame the counts and `shape_ok` are 0 and every other number is NaN.

    Beyond the CSV columns:
    - outline_xy_mm [n, N, 2]: the outline in world mm, N points equally spaced along it,
      counterclockwise (y up), starting at the head point. outline_xieta_mm [n, N, 2]: the same
      points in the body frame (xi toward the head, eta 90 degrees counterclockwise), mm.
    - radial_mm [n, 360 / radial_step_deg]: the radial profile, mm; column j is the radius at
      j steps counterclockwise from the heading; NaN where a ray misses the outline.
    - heading [n, 2]: the head direction as unit vectors in world coordinates.
    - orient [n], headguess [n]: the frame gets the flag ORIENT / HEADGUESS (SPEC 9). `orient` is
      never set on a lost frame; `headguess` is the same on every frame of the track.
    """

    track_id: str
    frame: np.ndarray
    t_s: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    u_px: np.ndarray
    v_px: np.ndarray
    area_mm2: np.ndarray
    visible: np.ndarray
    mode: np.ndarray
    perimeter_mm: np.ndarray
    major_mm: np.ndarray
    minor_mm: np.ndarray
    eccentricity: np.ndarray
    theta_rad: np.ndarray
    core_x_mm: np.ndarray
    core_y_mm: np.ndarray
    core_frac: np.ndarray
    solidity: np.ndarray
    circularity: np.ndarray
    feret_max_mm: np.ndarray
    n_components: np.ndarray
    largest_fraction: np.ndarray
    px_along_major: np.ndarray
    cells_along_major: np.ndarray
    shape_ok: np.ndarray
    wall_dist_centroid_mm: np.ndarray
    wall_dist_min_mm: np.ndarray
    outline_xy_mm: np.ndarray
    outline_xieta_mm: np.ndarray
    radial_mm: np.ndarray
    heading: np.ndarray
    orient: np.ndarray
    headguess: np.ndarray


def derive_track(arrays: TrackArrays, track: Track, world_frame: WorldFrame, fps_true: float,
                 circle: Circle | None, processing: Processing) -> DerivedTrack:
    """Every world quantity of one track (SPEC 7.1-7.9) from its pixel records.

    arrays: the track's records from the results store, in image px, one row per tracked frame.
    A stored mask crop is unpacked only on a row whose mask has more than one piece.
    track: its entry in the session; `head_px` (the head click, image px, or None) and
    `start_frame` (the frame the click belongs to) fix the head side, `id` names the result.
    world_frame: the calibration: scale in mm per px, origin in px, axis angle in rad.
    fps_true: frames per second of the recording. circle: the dish wall in image px, or None.
    processing: `outline_points` (N), `radial_step_deg` (degrees), `shape_ok_min` (px or cells).

    Returns a `DerivedTrack`: mm, s and rad in world coordinates, y up (see the class). The
    heading follows `derive_heading`. Raises ValueError for an fps_true that is not a positive
    number, fewer than 3 outline points, or a radial step that does not divide 360 degrees.
    """
    fps = float(fps_true)
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps_true = {fps_true!r}: the frame rate must be a positive number of frames per second.")
    count = int(processing.outline_points)
    if count != processing.outline_points or count < 3:
        raise ValueError(f"outline_points = {processing.outline_points!r}: an outline needs a whole number of "
                         f"points, 3 or more.")
    angles = ray_angles(processing.radial_step_deg)

    k = world_frame.k_mm_per_px
    visible = arrays.visible.astype(bool)
    x, y = world_frame.to_world(arrays.u, arrays.v)
    core = np.stack(world_frame.to_world(arrays.core_u, arrays.core_v), axis=-1)

    lambda1, lambda2, _ = principal_axes(world_frame.cov_to_world(arrays.cov_full))
    major = 4.0 * np.sqrt(lambda1)
    with np.errstate(invalid="ignore", divide="ignore"):
        eccentricity = np.sqrt(1.0 - lambda2 / lambda1)   # no extent: NaN
        px_along_major = major / k
    for row in np.flatnonzero(visible & (arrays.n_components > 1)):   # several pieces: the largest one alone
        crop, offset = arrays.mask(row)
        if crop.any():
            px_along_major[row] = _largest_piece_px(crop, arrays.outline_px[row].astype(np.float64) - offset)
    with np.errstate(invalid="ignore", divide="ignore"):
        cells_along_major = px_along_major / arrays.cell_px
        shape_ok = np.minimum(px_along_major, cells_along_major) >= processing.shape_ok_min   # NaN: not ok

    head = None if track.head_px is None else np.array(world_frame.to_world(*track.head_px), float)
    at_start = np.flatnonzero(arrays.frames == track.start_frame)
    headings = track_headings(world_frame.cov_to_world(arrays.cov_core), core, visible, arrays.core_fallback, head,
                              int(at_start[0]) if len(at_start) else None, k)

    stored = arrays.outline_px.astype(np.float64)
    polygons = np.stack(world_frame.to_world(stored[..., 0], stored[..., 1]), axis=-1)
    shape = outline_quantities(polygons, core, headings.heading, count, angles)
    with np.errstate(invalid="ignore", divide="ignore"):
        solidity = shape.area / shape.hull_area
        circularity = 4.0 * np.pi * shape.area / shape.perimeter ** 2   # no length: NaN
    from_core = shape.outline - core[:, None, :]
    hx, hy = headings.heading[:, None, 0], headings.heading[:, None, 1]
    xieta = np.stack([from_core[..., 0] * hx + from_core[..., 1] * hy,
                      -from_core[..., 0] * hy + from_core[..., 1] * hx], axis=-1)

    wall_centroid, wall_min = np.full(len(arrays), np.nan), np.full(len(arrays), np.nan)
    if circle is not None:
        cx, cy = world_frame.to_world(circle.center_px[0], circle.center_px[1])
        radius = k * float(circle.radius_px)
        wall_centroid = radius - np.hypot(x - cx, y - cy)
        wall_min = radius - np.hypot(polygons[..., 0] - cx, polygons[..., 1] - cy).max(axis=1)

    return DerivedTrack(
        track_id=track.id, frame=arrays.frames.copy(), t_s=arrays.frames / fps, x_mm=x, y_mm=y,
        u_px=arrays.u.copy(), v_px=arrays.v.copy(), area_mm2=k * k * arrays.area_px, visible=visible.astype(np.int32),
        mode=arrays.mode.copy(), perimeter_mm=shape.perimeter, major_mm=major, minor_mm=4.0 * np.sqrt(lambda2),
        eccentricity=eccentricity, theta_rad=headings.theta_rad, core_x_mm=core[:, 0], core_y_mm=core[:, 1],
        core_frac=arrays.core_frac.copy(), solidity=solidity, circularity=circularity, feret_max_mm=shape.feret,
        n_components=arrays.n_components.copy(), largest_fraction=arrays.largest_fraction.copy(),
        px_along_major=px_along_major, cells_along_major=cells_along_major, shape_ok=shape_ok.astype(np.int32),
        wall_dist_centroid_mm=wall_centroid, wall_dist_min_mm=wall_min, outline_xy_mm=shape.outline,
        outline_xieta_mm=xieta, radial_mm=shape.radial, heading=headings.heading, orient=headings.orient,
        headguess=np.full(len(arrays), headings.headguess),
    )


def _largest_piece_px(crop: np.ndarray, outline: np.ndarray) -> float:
    """L1 = 4 sqrt(lambda1) in px of the largest piece of a stored mask crop [row, column] with at
    least one pixel: lambda1 is the larger eigenvalue of the covariance of the piece's pixel
    centers, the formula of the full mask's major axis on fewer pixels. `outline` is the stored
    outline of the same row, (u, v) in px from the crop's top-left corner (the full-frame points
    minus the crop's offset): of pieces of equal size it says which one the outline was taken
    from, and that one is measured (see `measure.largest_piece`)."""
    piece, _ = largest_piece(crop, outline)
    rows, cols = np.nonzero(piece)
    du, dv = cols - cols.mean(), rows - rows.mean()
    # (uu, uv, vv) in image px: only the eigenvalue is used, which is the same in image and world axes
    lambda1, _, _ = principal_axes([np.mean(du * du), np.mean(du * dv), np.mean(dv * dv)])
    return 4.0 * math.sqrt(lambda1)
