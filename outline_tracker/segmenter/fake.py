"""Stand-in segmenters for the tests (SPEC 12, 13.2): no model, no torch.

Both implement the protocol of segmenter/base.py (`start`, `step`, `preview`, `close`) and its rule
for prompts: points outside the image are not used, and an object needs a positive point inside it
(`points_in_image`). Both return `MaskResult`s cut to the bounding box +/- 8 px, with logits whose
positive part is exactly the mask, and without a score (None: there is no model).

- `ThresholdFake` is the template's `DiskFinder` (the stand-in of last week's tests) behind the
  protocol: for each object, the dark pixels around its last position. It reads the image and
  nothing else, so it works on any clip of dark objects on a light background.
- `ExactFake` returns the ground truth of a synthetic clip (outline_tracker/synthetic.py). It does
  not read the image: the runner tells it which frame, and which part of it, the image shows, by
  calling `set_view` before `start`, `step` or `preview`.

Coordinates: an image is an RGB array indexed [row, column]. Prompt points are px in the pixel
frame of that image, Tracker's convention: u to the right, v downward, and the pixel in column c
and row r has its center at (c + 0.5, r + 0.5) (SPEC 3.1). A result's `offset` = (col0, row0) is
the top-left pixel of its crop in the image, in px. Logits are signed distances in px, positive
inside the object.
"""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import MaskResult, ObjectPrompt, crop_to_bbox, points_in_image

if TYPE_CHECKING:  # only for the annotation: importing the module here would load pandas and scikit-image
    from outline_tracker.synthetic import GroundTruth

DARK_BELOW = 128  # a pixel is dark if its red value (0 to 255) is below this
NEAR_PX = 20      # a dark group belongs to an object if its center is less than this far from it, px
PAD_PX = 8        # margin of a result's crop around the mask's bounding box, px


# ---------------------------------------------------------------------------------------------
# ThresholdFake


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the outline of a pixel mask, in px, positive inside: float32, same shape.

    The outline runs along the pixel edges between the mask and the rest. A pixel center is taken
    to be as far from it as from the nearest pixel center of the other kind, less half a pixel:
    exact along rows and columns, a little too far diagonally. So the values are at least 0.5
    inside, at most -0.5 outside, and `distance > 0` is the mask. A mask without pixels of the
    other kind has no outline: OpenCV's distance is then 65536 px.
    """
    if mask.size == 0:
        return np.zeros(mask.shape, np.float32)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(mask, inside - 0.5, 0.5 - outside).astype(np.float32)


def _start_positions(image: np.ndarray, prompts: list[ObjectPrompt]) -> list[tuple[float, float]]:
    """Each object's first positive point inside the image, (u, v) in px of the image."""
    positions = []
    for prompt in prompts:
        kept = points_in_image(prompt, image.shape[0], image.shape[1])
        positions.append(next(tuple(point) for point, label in zip(kept.points_px, kept.labels) if label == 1))
    return positions


def _find(image: np.ndarray, names: list[str],
          positions: list[tuple[float, float]]) -> tuple[list[MaskResult], list[tuple[float, float]]]:
    """`DiskFinder`'s step: one result per object, and the objects' new last positions.

    `image` is an RGB array [row, column, 3]; positions are (u, v) in px of the image (SPEC 3.1),
    result offsets likewise.
    """
    _, labels, _, centers = cv2.connectedComponentsWithStats((image[:, :, 0] < DARK_BELOW).astype(np.uint8))
    results, moved = [], []
    for name, (u, v) in zip(names, positions):
        # label 0 is the light background; centers are (column, row) with pixel centers at whole numbers
        near = np.hypot(centers[1:, 0] - int(u), centers[1:, 1] - int(v)) < NEAR_PX
        mask = np.isin(labels, np.flatnonzero(near) + 1)
        new_u, new_v, area = mask_center(mask)
        moved.append((new_u, new_v) if area else (u, v))
        cut = crop_to_bbox(mask, None, PAD_PX, obj_id=name)
        results.append(replace(cut, logits=_signed_distance(cut.mask)))
    return results, moved


class ThresholdFake:
    """The template's `DiskFinder` behind the Segmenter protocol: thresholding instead of a model.

    In every image the dark pixels (red value below 128) are grouped into 8-connected components.
    An object's mask is every component whose center is less than 20 px from the object's last
    position; the last position then becomes the center of that mask. Without such a component the
    object is lost in this image (an empty result) and its last position stays. An object's first
    position is its first positive point inside the image; its other points are not used.

    As in `DiskFinder`, the 20 px are measured between the component's center counted with pixel
    centers at whole numbers and the last position cut down to whole numbers, so the masks are the
    same as last week's, pixel for pixel. Logits are the signed distance to the mask's outline from
    a distance transform (px, positive inside): fine for positions, a staircase for outlines.

    A run is `start`, any number of `step`s, `close`; `preview` can be called at any time. Images
    are RGB arrays [row, column, 3]; coordinates are px in the pixel frame of the image of the same
    call (see the module text).
    """

    def __init__(self):
        self._names: list[str] | None = None        # the run's objects, in the order of the prompts
        self._last: list[tuple[float, float]] = []  # where each was last found: (u, v), px of the image

    def start(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """First image of a run. `image` is an RGB array [row, column, 3]; prompt points are px in
        its pixel frame (SPEC 3.1). Each object starts at its first positive point inside the image;
        an object without one raises `PromptError`. Returns one result per prompt, in the same
        order, with offsets in px of `image`."""
        positions = _start_positions(image, prompts)
        self._names, self._last = [prompt.obj_id for prompt in prompts], positions
        return self.step(image)

    def step(self, image: np.ndarray) -> list[MaskResult]:
        """The next image of the run, an RGB array [row, column, 3]. Returns one result per object,
        in the order of the prompts, with offsets in px of `image`; a lost object's result is empty."""
        if self._names is None:
            raise RuntimeError("ThresholdFake.step() needs a run: call start() first.")
        results, self._last = _find(image, self._names, self._last)
        return results

    def preview(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """The masks `start` would give for this image and these prompts (points in px of `image`,
        SPEC 3.1), with offsets in px of `image`; a run in progress is not changed."""
        return _find(image, [prompt.obj_id for prompt in prompts], _start_positions(image, prompts))[0]

    def close(self) -> None:
        """End the run: forget its objects and their positions. No quantities, so no units."""
        self._names, self._last = None, []


# ---------------------------------------------------------------------------------------------
# ExactFake


def _whole(value, what: str) -> int:
    """`value` as a Python int; a `ValueError` naming `what` if it is not a whole number."""
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"ExactFake.set_view: {what} must be a whole number, not {value!r}.") from None


class ExactFake:
    """Returns the ground truth of a synthetic clip, in whatever part of the frame it is shown.

    `ground_truth` is the `GroundTruth` of the clip (outline_tracker/synthetic.py); an object's
    `obj_id` is the `track_id` of the scene's object it stands for. The image itself is not read,
    only its size is checked: before `start`, `step` or `preview`, the runner says what the image
    shows with `set_view(frame, offset, size)`. The full-frame ground-truth mask and the analytic
    signed-distance logits of that frame are then sliced with the view's whole-pixel offset (never
    drawn again in the view's coordinates), so a position measured on a result, plus the result's
    offset, plus the view's offset, is the position in the full frame, exactly.

    Where a view hangs over the frame's border (a fine crop pads there by repeating the edge
    pixels, SPEC 6.3) nothing is inside: the mask is False and each logit is minus the size of the
    nearest frame pixel's logit, so an object cut by the border has its outline on the border.

    Coordinates: px, Tracker's convention (SPEC 3.1). Prompt points and result offsets are in the
    pixel frame of the image given, that is, of the view; the view's own offset is in the full frame.
    """

    def __init__(self, ground_truth: GroundTruth):
        self.ground_truth = ground_truth
        self._view: tuple[int, tuple[int, int], tuple[int, int]] | None = None  # frame, offset, size
        self._names: list[str] | None = None  # the run's objects, in the order of the prompts

    def set_view(self, frame: int, offset: tuple[int, int], size: tuple[int, int]) -> None:
        """Say what the next images show: frame number `frame` of the clip, and the part of it whose
        top-left pixel is `offset` = (column, row) in the full frame and whose `size` is
        (width, height), all in whole px. The full frame is offset (0, 0) with the clip's size. The
        view may reach beyond the frame. It stays until the next call."""
        frame = _whole(frame, "the frame number")
        col0, row0 = (_whole(value, "a view's offset in px") for value in offset)
        width, height = (_whole(value, "a view's size in px") for value in size)
        if width <= 0 or height <= 0:
            raise ValueError(f"ExactFake.set_view: a view's width and height must be positive, not {width} x "
                             f"{height} px.")
        self._view = (frame, (col0, row0), (width, height))

    def _truth(self, names: list[str]) -> list[MaskResult]:
        """The ground truth of the named tracks in the current view, one result per name."""
        frame, (col0, row0), (width, height) = self._view
        frame_width, frame_height = self.ground_truth.scene.size
        # Rows and columns of the frame under the view's pixels; beyond the border, the edge's.
        rows, cols = np.arange(row0, row0 + height), np.arange(col0, col0 + width)
        in_frame = ((rows >= 0) & (rows < frame_height))[:, None] & ((cols >= 0) & (cols < frame_width))[None, :]
        nearest = np.ix_(np.clip(rows, 0, frame_height - 1), np.clip(cols, 0, frame_width - 1))
        results = []
        for name in names:
            mask = self.ground_truth.mask(name, frame)[nearest] & in_frame
            logits = self.ground_truth.logits(name, frame)[nearest]
            results.append(crop_to_bbox(mask, np.where(in_frame, logits, -np.abs(logits)), PAD_PX, obj_id=name))
        return results

    def _check(self, image: np.ndarray) -> None:
        """Raise unless `set_view` was called and `image` has the size of that view."""
        if self._view is None:
            raise RuntimeError("ExactFake does not read the image: call set_view(frame, offset, size) first.")
        width, height = self._view[2]
        if image.shape[:2] != (height, width):
            raise ValueError(f"ExactFake was told a view of {width} x {height} px, but the image is "
                             f"{image.shape[1]} x {image.shape[0]} px.")

    def start(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """First image of a run: the view set by `set_view`, an array [row, column, 3] of that size.
        Prompt points are px in its pixel frame (SPEC 3.1); they are checked as the model would
        check them (`PromptError` without a positive point inside the image) and otherwise not
        used: each `obj_id` names its track. Returns one result per prompt, in the same order, with
        offsets in px of `image`. An unknown track raises `KeyError`."""
        results = self.preview(image, prompts)
        self._names = [prompt.obj_id for prompt in prompts]
        return results

    def step(self, image: np.ndarray) -> list[MaskResult]:
        """The next image of the run: the view last set by `set_view`, an array of that size.
        Returns one result per object, in the order of the prompts, with offsets in px of `image`;
        an object that is not visible in the view gives an empty result."""
        if self._names is None:
            raise RuntimeError("ExactFake.step() needs a run: call start() first.")
        self._check(image)
        return self._truth(self._names)

    def preview(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """The results `start` would give for the current view and these prompts (points in px of
        `image`, SPEC 3.1), with offsets in px of `image`; a run in progress is not changed."""
        self._check(image)
        prompts = [points_in_image(prompt, image.shape[0], image.shape[1]) for prompt in prompts]
        return self._truth([prompt.obj_id for prompt in prompts])

    def close(self) -> None:
        """End the run: forget its objects (the view stays). No quantities, so no units."""
        self._names = None
