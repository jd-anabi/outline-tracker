"""Measuring a mask: where an object is, and its pixel-space record for results.npz (SPEC 7, 8.12).

`mask_center` is last week's function from shrimp.segment, moved over unchanged and checked against
the reference copy by tests/test_port_fidelity.py and tests/test_port_equivalence.py.

`measure_mask` turns one object's mask (and logits) on one frame into a `PixelRecord`: area,
centroid, second moments, the core mask's moments, the outline, the components, and what the model
saw. Everything in a record is in pixels; world units are derived from records at export, so a new
calibration needs no new tracking (SPEC 7). A record's field names are the names of
`schema.RESULTS_KEYS`, with `frame` for one entry of `frames`.

Coordinates: masks are arrays indexed [row, column]. Positions are in Tracker's image coordinates,
in px: the origin is the top-left corner of the frame, u grows to the right, v downward, and the
pixel in column c and row r has its center at (c + 0.5, r + 0.5) (SPEC 3.1). Second moments are in
px^2, in the same (u, v) axes.

Rules fixed here; they are frozen into results.npz at tracking time:
- Components are 8-connected everywhere: two pixels that touch at a corner belong together. The
  largest one is what `largest_piece` returns.
- The core (SPEC 7.3) is the largest component of the mask's opening with the disk of pixels
  (x, y), x^2 + y^2 <= r^2, r = max(1, floor(core_open_frac * L1 + 0.5)) px, L1 = 4 sqrt(lambda1)
  of the full mask. If the opening removes more than half of the mask's pixels, the core is the
  mask itself (`core_fallback`).
- The outline (SPEC 7.4) is the 0-level set of the logits around the largest component, linearly
  interpolated between pixel centers (marching squares), resampled to 256 points equally spaced
  along it. Holes are ignored. Where the crop ends on a mask pixel (the border of the model's
  image), the outline runs along that pixel's outer edge, so it is always closed. Without logits
  it is the chain of the component's border pixel centers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import find_contours

from outline_tracker.schema import STORED_OUTLINE_POINTS
from outline_tracker.segmenter.base import MaskResult

MODES = ("coarse", "fine")  # what the model saw: the frame or the dish crop, or a crop that follows one object
GRID_CELLS = 256            # the model predicts its masks on a 256 x 256 grid, whatever its input (SPEC 6.2)


def mask_center(mask: np.ndarray) -> tuple[float, float, int]:
    """Center of a boolean mask in Tracker's image coordinates, and its area in pixels.

    The mean column and row of the mask's pixels, plus 0.5: in Tracker, pixel (column c, row r) is the
    square from c to c + 1 and r to r + 1, so its center is at (c + 0.5, r + 0.5). NaN if empty.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return float("nan"), float("nan"), 0
    return float(cols.mean() + 0.5), float(rows.mean() + 0.5), int(len(rows))


@dataclass(frozen=True, eq=False)
class PixelRecord:
    """What was measured for one object on one frame, in pixels (SPEC 8.12; decisions X5, X15).

    Positions (u, v) are in px in full-frame image coordinates (pixel centers at +0.5); the
    covariances are (mu_uu, mu_uv, mu_vv) of pixel centers in px^2. The field names, and the dtypes
    that results.npz gives them, are those of `schema.RESULTS_KEYS`.

    frame: video frame number. visible: the mask is non-empty. area_px: its pixel count.
    u, v: area centroid of the whole mask (`mask_center`). cov_full: covariance of the whole mask.
    core_u, core_v, cov_core: centroid and covariance of the core, or of the whole mask when
    `core_fallback` is set. core_frac: pixels of the largest piece that the opening left, over
    area_px; the fallback does not change it, so it is below 0.5 whenever the fallback was used.
    core_r_px: radius of the opening disk in px. outline_px: 256 points (u, v) along the largest
    component's outline, float32, equally spaced, in no fixed sense of rotation.
    n_components: number of 8-connected pieces. largest_fraction: largest piece over area_px.
    second_fraction: second-largest piece over the largest, 0 with one piece.
    cell_px: one model grid cell in px, max(width, height) / 256 of the model's input.
    edge: the mask has a pixel on the first or last row or column of the model's input.
    mode: "coarse" or "fine". score: the model's score for the object, NaN if it has none.
    mask_offset, mask_shape, mask_bits: the mask cut to its bounding box: (column, row) of the
    box's top-left pixel in the full frame, its (rows, columns), and its pixels packed by
    `pack_mask`.

    For an empty mask: visible False, area_px, n_components and core_r_px 0, the two flags False,
    an empty crop, and NaN in every measured number; frame, mode, score and cell_px are kept.
    """

    frame: int
    visible: bool
    area_px: int
    u: float
    v: float
    cov_full: np.ndarray
    cov_core: np.ndarray
    core_u: float
    core_v: float
    core_frac: float
    core_fallback: bool
    core_r_px: int
    outline_px: np.ndarray
    n_components: int
    largest_fraction: float
    second_fraction: float
    cell_px: float
    edge: bool
    mode: str
    score: float
    mask_offset: tuple[int, int]
    mask_shape: tuple[int, int]
    mask_bits: np.ndarray

    def mask(self) -> np.ndarray:
        """The mask cut to its bounding box: a boolean array [row, column] of shape `mask_shape`,
        whose top-left pixel is at `mask_offset` = (column, row) of the full frame, in px."""
        return unpack_mask(self.mask_bits, self.mask_shape)


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """A boolean mask [row, column] as bytes: flattened row by row and packed with numpy.packbits,
    the layout of `mask_bits` in results.npz: ceil(rows * columns / 8) bytes (uint8). No units."""
    return np.packbits(np.asarray(mask, bool).ravel())


def unpack_mask(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `pack_mask`: the boolean mask [row, column] of `shape` = (rows, columns), in
    pixels, from its packed bytes."""
    rows, cols = int(shape[0]), int(shape[1])
    return np.unpackbits(np.asarray(bits, np.uint8), count=rows * cols).reshape(rows, cols).astype(bool)


def largest_piece(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The largest 8-connected piece of a non-empty boolean mask [row, column], and the pixel
    count of every piece.

    Returns (piece, sizes): `piece` has the mask's shape and is true on the pixels of the piece
    with the most pixels; `sizes` holds one count per piece. `measure_mask` takes the outline from
    this piece, and `derive` takes the size check from it (`px_along_major`), so that both are of
    the same pixels.

    Of pieces with exactly the same number of pixels, the first in OpenCV's label order is taken.
    That order depends on the row the array begins with, because OpenCV labels in blocks of 2 x 2
    pixels. `measure_mask` gives the mask's bounding box with one pixel around it. For the same
    piece from a stored crop, give the crop with one pixel around it too (`np.pad(crop, 1)`). One
    case stays open: a mask on the first row of the model's image had no row above it, and the
    crop does not say so. Only there, and only between pieces of equal size, can the two differ.
    """
    labels, sizes = _components(mask)
    return labels == 1 + int(np.argmax(sizes)), sizes


def measure_mask(result: MaskResult, frame: int, input_box: tuple[int, int, int, int], mode: str,
                 core_open_frac: float = 0.1) -> PixelRecord:
    """Measure one object's mask on one frame: its pixel-space record (SPEC 7.1-7.4, 7.8, 8.12).

    `result` must be in full-frame pixels: `result.offset` = (column, row) of the top-left pixel
    of its arrays in the full frame. A runner that gave the model a crop adds the crop's corner to
    the segmenter's offset first (SPEC 6.2, 6.3). `result.mask` is boolean [row, column];
    `result.logits`, if present, has the same shape and is positive exactly on the mask.
    `frame` is the video frame number. `input_box` = (c0, r0, width, height), in full-frame px, is
    the image the model saw: the whole frame, the dish crop or a fine crop. It gives `cell_px` and
    `edge`. `mode` is "coarse" or "fine". `core_open_frac` is the opening radius as a fraction of
    the mask's major axis L1 (SPEC 7.3).

    Returns a `PixelRecord`: positions (u, v) in px in the full frame, pixel centers at +0.5;
    second moments in px^2. Only the mask's bounding box is worked on, plus one pixel around it
    for the logits; `result` is not changed. An empty mask gives the lost record described in
    `PixelRecord`. Raises ValueError for an unknown mode, an input box without a size, a
    `core_open_frac` that is negative or not finite, or logits of another shape than the mask.
    """
    col0, row0, width, height = (int(x) for x in input_box)
    if width <= 0 or height <= 0:
        raise ValueError(f"input_box = {tuple(input_box)}: the width and height of the model's input must be positive.")
    if mode not in MODES:
        raise ValueError(f"mode = {mode!r}: it must be 'coarse' or 'fine'.")
    if not (math.isfinite(core_open_frac) and core_open_frac >= 0):
        raise ValueError(f"core_open_frac = {core_open_frac!r}: it must be a finite number, 0 or more.")
    mask = np.asarray(result.mask, bool)
    logits = None if result.logits is None else np.asarray(result.logits)
    if logits is not None and logits.shape != mask.shape:
        raise ValueError(f"logits of shape {logits.shape} do not match the mask of shape {mask.shape}.")
    frame = int(frame)
    cell_px = max(width, height) / GRID_CELLS
    score = float("nan") if result.score is None else float(result.score)

    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return _lost_record(frame, cell_px, mode, score)
    top, bottom, left, right = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    box = mask[top:bottom, left:right]
    box_col, box_row = int(result.offset[0]) + left, int(result.offset[1]) + top  # the box in the full frame

    # Everything is computed in the frame of a window: the box and one pixel around it, where the
    # arrays have that pixel (the level set runs between a mask pixel and its neighbor).
    window_top, window_left = max(top - 1, 0), max(left - 1, 0)
    window = np.s_[window_top:min(bottom + 1, mask.shape[0]), window_left:min(right + 1, mask.shape[1])]
    work = np.ascontiguousarray(mask[window])
    shift_u, shift_v = box_col - (left - window_left), box_row - (top - window_top)

    u, v, area = mask_center(work)
    cov_full = _covariance(work)

    largest, sizes = largest_piece(work)
    ordered = np.sort(sizes)[::-1]

    radius = max(1, math.floor(core_open_frac * _major_axis(cov_full) + 0.5))
    opened = _opening(work, radius)
    core_fallback = bool(2 * int(opened.sum()) < area)
    core_frac = 0.0
    core_u, core_v, cov_core = u, v, cov_full.copy()
    if opened.any():
        core_labels, core_sizes = _components(opened)
        core_frac = int(core_sizes.max()) / area
        if not core_fallback:
            core = core_labels == 1 + int(np.argmax(core_sizes))
            core_u, core_v, _ = mask_center(core)
            cov_core = _covariance(core)

    outline = _resample_closed(_outline(largest, None if logits is None else logits[window]), STORED_OUTLINE_POINTS)
    outline_px = (outline + (shift_u, shift_v)).astype(np.float32)

    edge = (box_col <= col0 or box_row <= row0
            or box_col + box.shape[1] >= col0 + width or box_row + box.shape[0] >= row0 + height)
    return PixelRecord(
        frame=frame, visible=True, area_px=area, u=u + shift_u, v=v + shift_v, cov_full=cov_full,
        cov_core=cov_core, core_u=core_u + shift_u, core_v=core_v + shift_v, core_frac=core_frac,
        core_fallback=core_fallback, core_r_px=radius, outline_px=outline_px, n_components=len(sizes),
        largest_fraction=int(ordered[0]) / area,
        second_fraction=int(ordered[1]) / int(ordered[0]) if len(ordered) > 1 else 0.0,
        cell_px=cell_px, edge=bool(edge), mode=mode, score=score, mask_offset=(box_col, box_row),
        mask_shape=(box.shape[0], box.shape[1]), mask_bits=pack_mask(box),
    )


def _lost_record(frame: int, cell_px: float, mode: str, score: float) -> PixelRecord:
    """The record of a frame on which the object was not found (see `PixelRecord`)."""
    nan = float("nan")
    return PixelRecord(
        frame=frame, visible=False, area_px=0, u=nan, v=nan, cov_full=np.full(3, nan), cov_core=np.full(3, nan),
        core_u=nan, core_v=nan, core_frac=nan, core_fallback=False, core_r_px=0,
        outline_px=np.full((STORED_OUTLINE_POINTS, 2), nan, np.float32), n_components=0, largest_fraction=nan,
        second_fraction=nan, cell_px=cell_px, edge=False, mode=mode, score=score, mask_offset=(0, 0),
        mask_shape=(0, 0), mask_bits=np.zeros(0, np.uint8),
    )


def _covariance(mask: np.ndarray) -> np.ndarray:
    """(mu_uu, mu_uv, mu_vv), px^2: the covariance of the pixel centers of a non-empty mask."""
    rows, cols = np.nonzero(mask)
    du, dv = cols - cols.mean(), rows - rows.mean()
    return np.array([np.mean(du * du), np.mean(du * dv), np.mean(dv * dv)])


def _major_axis(cov: np.ndarray) -> float:
    """L1 = 4 sqrt(lambda1) in px, lambda1 the larger eigenvalue of the covariance (uu, uv, vv)."""
    uu, uv, vv = (float(x) for x in cov)
    return 4.0 * math.sqrt((uu + vv) / 2.0 + math.hypot((uu - vv) / 2.0, uv))


def _components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 8-connected components of a mask: labels (0 off the mask, 1..n on it) and the pixel
    count of each label 1..n."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels, stats[1:count, cv2.CC_STAT_AREA]


def _opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening of a mask with the disk of pixels x^2 + y^2 <= radius^2 (radius in px).

    Pixels outside the array count as background: the mask is padded with zeros first. A disk
    wider than the array fits nowhere, so the opening is empty.
    """
    if 2 * radius + 1 > min(mask.shape):
        return np.zeros(mask.shape, bool)
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y <= radius * radius).astype(np.uint8)
    pad = radius + 1
    padded = np.pad(mask.astype(np.uint8), pad)
    opened = cv2.dilate(cv2.erode(padded, disk), disk)
    return opened[pad:-pad, pad:-pad].astype(bool)


def _outline(component: np.ndarray, logits: np.ndarray | None) -> np.ndarray:
    """The closed outline of one component: points (u, v) in px in the array's own frame (pixel
    centers at +0.5), the last one joined to the first. From the logits if there are any and they
    give a contour, else from the component's border pixels."""
    contour = None if logits is None else _level_set(component, logits)
    return _pixel_boundary(component) if contour is None else contour


def _level_set(component: np.ndarray, logits: np.ndarray) -> np.ndarray | None:
    """The 0-level set of the logits around one component, or None if there is no closed one.

    Logits off the component are made negative, so other components leave no contour. One ring is
    added around the array, each value minus the absolute value of its neighbor inside: a mask
    pixel on the array's border is then closed off exactly on its outer edge. Of the closed
    contours, the one with the largest area is kept: holes are smaller than the outer contour.
    """
    level = np.where(component, logits, -np.abs(logits)).astype(np.float64)
    padded = np.pad(level, 1, mode="edge")
    for ring in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        padded[ring] = -np.abs(padded[ring])
    best, best_area = None, 0.0
    for contour in find_contours(padded, 0.0, fully_connected="high"):
        if len(contour) < 4 or not np.array_equal(contour[0], contour[-1]):
            continue
        points = contour[:-1, ::-1] - 0.5  # (row, col) of the padded array -> (u, v) = (col - 1 + 0.5, row - 1 + 0.5)
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
        if area > best_area:
            best, best_area = points, area
    return best


def _pixel_boundary(component: np.ndarray) -> np.ndarray:
    """The chain of the centers of one component's border pixels (cv2.findContours, every pixel
    kept): it runs half a pixel inside the true boundary."""
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=len)[:, 0, :].astype(np.float64) + 0.5


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """`count` points equally spaced along a closed polygon, starting at its first vertex; the
    same units and frame as `points`. A polygon without length gives `count` copies of its point."""
    closed = np.vstack([points, points[:1]])
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    if along[-1] <= 0:
        return np.repeat(closed[:1], count, axis=0)
    at = np.arange(count) * (along[-1] / count)
    return np.column_stack([np.interp(at, along, closed[:, 0]), np.interp(at, along, closed[:, 1])])
