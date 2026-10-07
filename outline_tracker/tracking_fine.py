"""Fine mode: a W x W crop that follows one object (SPEC 6.3).

A fine object has a run and a session of the model of its own. The model is shown a square window
of W x W px around the object, so that one of its 256 grid cells is W / 256 px whatever the size
of the frame. The parts:

- the window rule: W = clip(ceil(3 F), 96, 512), F the largest diameter of the object's outline on
  its start frame (`fine_window_px`, `fine_window`); a window set by hand is used as it is;
- how a fine run starts (`fine_starts`): the model is asked for the object's mask on the start
  frame as coarse tracking would show it (the frame, or the dish square); the mask gives F and the
  first center, and the clicks are shifted into the first window;
- the crop (`FollowCrop`): centered on the object's centroid on the tracked frame before, kept
  where it is while the object is lost, its corner rounded to whole pixels, and of the same size
  where it hangs over the frame, where the edge pixels are repeated (`cut_crop`).

`tracking.run_job` runs the frames of a fine run with the same loop as a coarse one.

Units and coordinates (SPEC 3.1): px in Tracker's convention, the origin at the top-left corner of
the frame, u to the right, v downward, the pixel in column c and row r with its center at
(c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the full frame; the corner of
a fine crop may be negative, and its far side may lie beyond the frame. Frames are video frame
numbers. No Qt, no torch.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace

import cv2
import numpy as np

from outline_tracker.derive_outline import feret_max
from outline_tracker.measure import PixelRecord, mask_center, measure_mask
from outline_tracker.segmenter.base import MaskResult, ObjectPrompt, PromptError, Segmenter
from outline_tracker.session import Session, Track
from outline_tracker.tracking_plan import RunPlan, run_prompts
from outline_tracker.video import iter_rgb_frames

WINDOW_MIN, WINDOW_MAX = 96, 512  # the smallest and the largest window the rule chooses, px (SPEC 6.3)


def fine_window_px(feret_px: float, factor: float = 3.0) -> int:
    """The side W of the fine window for an object, in px: clip(ceil(factor * F), 96, 512).

    `feret_px` is F, the object's maximum Feret diameter in px; `factor` is the setting
    `fine_window_factor` (3: the object then spans about 85 of the model's 256 grid cells). An F
    that is NaN (no outline to measure) or not positive gives the smallest window, 96 px. Raises
    ValueError for a factor that is not a positive, finite number.
    """
    _check_factor(factor)
    wanted = factor * float(feret_px)
    if not wanted > WINDOW_MIN:  # also NaN
        return WINDOW_MIN
    return WINDOW_MAX if wanted >= WINDOW_MAX else math.ceil(wanted)


def _check_factor(factor) -> None:
    if (not isinstance(factor, numbers.Real) or isinstance(factor, bool) or not math.isfinite(factor)
            or factor <= 0):
        raise ValueError(f"fine_window_factor must be a positive number (3 unless changed: the fine window is that "
                         f"many times the object's length), not {factor!r}.")


def fine_window(preview_result: MaskResult, factor: float = 3.0) -> int:
    """The fine window for an object from its mask on the start frame: `fine_window_px` of F, the
    maximum Feret diameter of the mask's outline in px.

    `preview_result` is the object's result from `segmenter.preview`; its offset does not matter.
    F is `derive_outline.feret_max` of the 256-point outline that `measure.measure_mask` stores,
    the diameter shapes.csv reports, here in px. Returns W in px: 96 for an empty mask and for an
    outline without a hull (all of it on one line). `factor` as in `fine_window_px`.
    """
    outline = measure_mask(preview_result, 0, (0, 0, 1, 1), "fine").outline_px  # frame and box do not shape the outline
    return fine_window_px(feret_max(outline), factor)


def crop_box(center_px: tuple[float, float], window: int) -> tuple[int, int, int, int]:
    """The fine crop around a point: (c0, r0, window, window) in whole px of the full frame.

    `center_px` = (u, v), px in the full frame, is where the crop is centered; `window` is its
    side W in px. The corner (u - W / 2, v - W / 2) is rounded to whole pixels (halves upward), so
    the crop's middle is within half a pixel of the point. The corner may be negative and the
    crop may reach beyond the frame (`cut_crop`).
    """
    window = int(window)
    return (math.floor(center_px[0] - window / 2 + 0.5), math.floor(center_px[1] - window / 2 + 0.5), window, window)


def cut_crop(rgb: np.ndarray, corner: tuple[int, int], window: int) -> np.ndarray:
    """The W x W crop of a frame that the model is shown, always of that size (SPEC 6.3).

    `rgb` is the frame, an array [row, column, 3]; `corner` = (c0, r0), whole px of the full
    frame, is the crop's top-left pixel and may lie outside the frame; `window` is W in px. Where
    the crop reaches beyond the frame it is padded by repeating the frame's edge pixels
    (`cv2.BORDER_REPLICATE`). Returns a new array [W, W, 3] in one piece of memory. Raises
    ValueError for a crop that has no pixel in common with the frame.
    """
    (c0, r0), window = corner, int(window)
    height, width = rgb.shape[:2]
    left, top, right, bottom = max(c0, 0), max(r0, 0), min(c0 + window, width), min(r0 + window, height)
    if right <= left or bottom <= top:
        raise ValueError(f"A {window} x {window} px window at column {c0}, row {r0} lies outside the "
                         f"{width} x {height} px frame.")
    inside = np.ascontiguousarray(rgb[top:bottom, left:right])
    if inside.shape[:2] == (window, window):
        return inside
    return cv2.copyMakeBorder(inside, top - r0, r0 + window - bottom, left - c0, c0 + window - right,
                              cv2.BORDER_REPLICATE)


def _in_frame(result: MaskResult, frame_size: tuple[int, int]) -> MaskResult:
    """`result`, whose offset is in px of the full frame, without the rows and columns that lie
    outside the frame of `frame_size` = (width, height) px: what a model finds in the padding of
    a crop is not part of the object. Nothing left gives an empty result."""
    width, height = frame_size
    (col0, row0), (rows, cols) = result.offset, result.mask.shape
    top, left = max(-row0, 0), max(-col0, 0)
    bottom, right = min(rows, height - row0), min(cols, width - col0)
    if (top, left, bottom, right) == (0, 0, rows, cols):
        return result
    if bottom <= top or right <= left:
        offset, cut = (0, 0), np.s_[:0, :0]
    else:
        offset, cut = (col0 + left, row0 + top), np.s_[top:bottom, left:right]
    return replace(result, offset=offset, mask=result.mask[cut],
                   logits=None if result.logits is None else result.logits[cut])


class FollowCrop:
    """Where the model looks during one fine run: a `window` x `window` px crop that follows the
    object (SPEC 6.3).

    `center` = (u, v), px in the full frame, is where the crop is centered on the next frame: the
    centroid of the start frame's preview mask at first, then the object's centroid on the tracked
    frame before; while the object is lost it stays. `frame_size` = (width, height) of the video's
    frames, px. `box` is the crop for the next frame, (c0, r0, window, window) in whole px of the
    full frame.
    """

    mode = "fine"

    def __init__(self, window: int, center: tuple[float, float], frame_size: tuple[int, int]):
        self.window, self.frame_size = int(window), (int(frame_size[0]), int(frame_size[1]))
        self.center = (float(center[0]), float(center[1]))

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The crop for the next frame: (c0, r0, window, window), whole px of the full frame."""
        return crop_box(self.center, self.window)

    def describe(self) -> str:
        """What the model is shown, for the log; sizes in px."""
        return f"a {self.window} x {self.window} px window that follows the object"

    def cut(self, rgb: np.ndarray) -> np.ndarray:
        """The image for the model: the crop of the frame `rgb` ([row, column, 3]) at `box`, padded
        where it hangs over the frame (`cut_crop`)."""
        return cut_crop(rgb, self.box[:2], self.window)

    def measure(self, result: MaskResult, frame: int, core_open_frac: float) -> PixelRecord:
        """The record of the object on video frame `frame`, from the model's result for the crop.

        `result` has its offset in px of the crop. It is shifted back by the crop's corner, cut to
        the frame (the padding is not part of the picture), and measured with the crop as the
        model's input, so `cell_px` is window / 256. `edge` is also set when the mask touches the
        border of the frame, where the crop cannot show what lies beyond. Returns the record, in
        px of the full frame, and moves the crop's center to the object's centroid; a lost frame
        leaves the center where it is.
        """
        box = self.box
        shifted = replace(result, offset=(result.offset[0] + box[0], result.offset[1] + box[1]))
        record = measure_mask(_in_frame(shifted, self.frame_size), frame, box, self.mode, core_open_frac)
        if record.visible:
            (col, row), (rows, cols) = record.mask_offset, record.mask_shape
            width, height = self.frame_size
            if not record.edge and (col <= 0 or row <= 0 or col + cols >= width or row + rows >= height):
                record = replace(record, edge=True)
            self.center = (record.u, record.v)
        return record


@dataclass
class FineStart:
    """How one fine run starts. crop: the window and its first center. prompts: the object's clicks
    on the start frame, in px of the first crop (one prompt). chosen: the window was chosen by the
    rule (the track had none). found: the model found the object on the start frame; if not, the
    window lies around the first positive click. A chosen window is stored in the session only
    when the object was found (`tracking.run_job`): the 96 px chosen without a mask were measured
    on nothing, and a stored window is used as it is by every later run of the track."""

    crop: FollowCrop
    prompts: list[ObjectPrompt]
    chosen: bool
    found: bool


def user_window(track: Track) -> int | None:
    """The fine window a track has: `track.fine_window_px`, in px, set by hand or stored by an
    earlier job that chose it from the object's mask; None for automatic. It is used as it is,
    also below 96 or above 512 px. Raises ValueError unless it is a whole number of at least 1."""
    window = track.fine_window_px
    if window is None:
        return None
    if not isinstance(window, numbers.Integral) or isinstance(window, bool) or window < 1:
        raise ValueError(f"Track {track.id}: the fine window must be a whole number of pixels (or empty, for "
                         f"automatic), not {window!r}.")
    return int(window)


def check_fine_settings(session: Session, plans: Sequence[RunPlan]) -> None:
    """Refuse, before a model is loaded, what the fine runs among `plans` could not use: a window
    set by hand that is not a whole number of px (`user_window`), and a `fine_window_factor` that is
    not a positive number. Raises ValueError; without a fine run nothing is checked."""
    tracks = {track.id: track for track in session.tracks}
    for plan in plans:
        if plan.mode == "fine":
            user_window(tracks[plan.track_ids[0]])
            _check_factor(session.processing.fine_window_factor)


def fine_starts(video_path, session: Session, plans: Sequence[RunPlan], segmenter: Segmenter) -> dict[str, FineStart]:
    """Prepare the fine runs among `plans`: for each, the window, its first center and the clicks
    in the first crop. Returns them by track id; without a fine run the video is not opened.

    Each fine run's start frame (a video frame number) is decoded once, as tracking decodes, and
    cut to the plan's `input_box`, what coarse tracking would show the model. `segmenter.preview`
    gives the object's mask there, from its clicks on that frame. No run may be under way.
    - Window: the track's own (`user_window`), else `fine_window` of the preview mask with the
      session's `fine_window_factor`; 96 px when the mask is empty.
    - First center: the centroid of the preview mask, px in the full frame; for an empty mask the
      object's first positive click.
    - Clicks: those of the start frame, shifted into the first crop; the ones outside it are
      dropped. Raises `PromptError`, a ValueError, when no positive click is left in the crop: the
      model found something else than what was clicked on.
    """
    waiting: dict[int, list[RunPlan]] = {}
    for plan in plans:
        if plan.mode == "fine":
            waiting.setdefault(plan.start_frame, []).append(plan)
    if not waiting:
        return {}
    starts: dict[str, FineStart] = {}
    with closing(iter_rgb_frames(video_path, sorted(waiting))) as frames:  # closed at once: Windows locks the file
        for frame, rgb in frames:
            for plan in waiting[frame]:
                starts[plan.track_ids[0]] = _fine_start(session, plan, rgb, segmenter)
    return starts


def _fine_start(session: Session, plan: RunPlan, rgb: np.ndarray, segmenter: Segmenter) -> FineStart:
    """How the fine run `plan` starts, from `rgb`, its start frame as decoded ([row, column, 3])."""
    (track_id,), frame, (c0, r0, width, height) = plan.track_ids, plan.start_frame, plan.input_box
    (track,) = [track for track in session.tracks if track.id == track_id]
    (prompt,) = run_prompts(session, plan)  # the clicks in px of the coarse view
    set_view = getattr(segmenter, "set_view", None)  # ExactFake is told what each image shows
    if set_view is not None:
        set_view(frame, (c0, r0), (width, height))
    (result,) = segmenter.preview(np.ascontiguousarray(rgb[r0:r0 + height, c0:c0 + width]), [prompt])
    u, v, area = mask_center(result.mask)
    if area:
        center = (u + result.offset[0] + c0, v + result.offset[1] + r0)
    else:
        click = next(point for point, label in zip(prompt.points_px, prompt.labels) if label == 1)
        center = (click[0] + c0, click[1] + r0)
    window = user_window(track)
    chosen = window is None
    if chosen:
        window = fine_window(result, session.processing.fine_window_factor)
    crop = FollowCrop(window, center, (session.video.width, session.video.height))
    try:
        in_crop = run_prompts(session, replace(plan, input_box=crop.box))
    except PromptError as err:
        raise PromptError(
            f"Track {track_id} (fine): on frame {frame} the model found an object around ({center[0]:.0f}, "
            f"{center[1]:.0f}) px, and the {window} x {window} px window around it holds no positive click of this "
            "track. Click on the object itself; a negative click on what was found instead helps.") from err
    return FineStart(crop, in_crop, chosen, bool(area))
