"""The segmenter protocol and its two records (SPEC 12, plus `preview`, decision X21).

A segmenter is given one image at a time. `start` receives the first image of a run together with
the prompts of every object; `step` receives each later image; both return one `MaskResult` per
object, in the order of the prompts. `preview` returns the masks for one image without keeping any
tracking state. No torch here: the stand-ins of the tests implement the same protocol.

Coordinates: an image is an RGB array indexed [row, column]. Prompt points and boxes are in px in
the pixel frame of that image, Tracker's convention: u to the right, v downward, and the pixel in
column c and row r has its center at (c + 0.5, r + 0.5) (SPEC 3.1). A result's mask and logits are
arrays indexed [row, column], cropped to the object's bounding box +/- 8 px; `offset` = (col0, row0)
is the crop's top-left pixel in the image, so a position in the image is a position in the crop
plus the offset. Logits have no unit; the mask is logits > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ObjectPrompt:       # coordinates in the pixel frame of `image` (§3.1)
    obj_id: str
    points_px: list[tuple[float, float]]
    labels: list[int]     # 1 positive, 0 negative
    box_px: tuple[float, float, float, float] | None = None


@dataclass
class MaskResult:         # mask and logits cropped to bbox ± 8 px, with the crop's offset in `image`
    obj_id: str
    offset: tuple[int, int]          # (col0, row0)
    mask: np.ndarray                 # bool
    logits: np.ndarray | None        # float32, same shape as mask
    score: float | None


class Segmenter(Protocol):
    """What tracking needs from a model. Images are RGB arrays [row, column]; all coordinates are
    px in the pixel frame of the image given in the same call (see the module text)."""

    def start(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]: ...
    def step(self, image: np.ndarray) -> list[MaskResult]: ...
    def preview(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]: ...
    def close(self) -> None: ...


def crop_to_bbox(mask: np.ndarray, logits: np.ndarray | None, pad: int = 8, *, obj_id: str = "",
                 score: float | None = None) -> MaskResult:
    """Cut a full-image mask (and its logits) down to the mask's bounding box plus `pad` px.

    `mask` is a boolean array indexed [row, column] with the shape of the image; `logits` has the
    same shape, or is None. `pad` is in px on every side, clipped at the image border. The result's
    `offset` = (col0, row0) is the crop's top-left pixel in the image, in px, so that
    `mask_center(result.mask)` plus the offset is the position in the image. The crops are copies:
    they do not keep the full-image arrays alive (SPEC 6.5). An empty mask gives arrays of shape
    (0, 0) with offset (0, 0). `obj_id` and `score` are stored as given.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        empty_logits = None if logits is None else np.zeros((0, 0), np.float32)
        return MaskResult(obj_id, (0, 0), np.zeros((0, 0), bool), empty_logits, score)
    row0, row1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + 1 + pad, mask.shape[0])
    col0, col1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + 1 + pad, mask.shape[1])
    cut = None if logits is None else np.array(logits[row0:row1, col0:col1], np.float32)
    return MaskResult(obj_id, (col0, row0), np.array(mask[row0:row1, col0:col1], bool), cut, score)
