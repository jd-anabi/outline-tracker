"""Brightness probes: the mean color inside named rectangles on every frame (SPEC 4.6, 8.7).

The tool's version of Tracker's RGB Region, made to time a stimulus LED. It is its own pass over
the clip: one sequential decode, no model. `measure_probes` returns the rows of probes.csv; the
command that calls it writes the file.

Units and coordinates: a rectangle is (u0, v0, u1, v1) in image px in Tracker's convention
(SPEC 3.1): u to the right, v down, origin at the top-left corner of the frame, the pixel in
column c and row r being the square [c, c + 1) x [r, r + 1) with its center at (c + 0.5, r + 0.5).
Frames are the video's own frame numbers, counted from 0 in the order of the sequential decode
(SPEC 3.5); t_s = frame / fps_true in s (SPEC 3.3). Brightness is on the video's 0 to 255 scale.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from outline_tracker import schema, video

GRAY_WEIGHTS = (0.299, 0.587, 0.114)  # gray = 0.299 R + 0.587 G + 0.114 B (SPEC 4.6)


def measure_probes(video_path, boxes: Mapping[str, Sequence[float]], start: int, end: int, fps_true: float,
                   log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Mean R, G, B and gray inside each box, on every frame from `start` to `end` of a video.

    `boxes` maps a probe's name to its rectangle (u0, v0, u1, v1): two opposite corners, in either
    order, in image px (u to the right, v down, pixel centers at +0.5; SPEC 3.1). A box holds the
    pixels whose centers lie inside it; a center on its low edge counts, one on its high edge does
    not. So whole-number corners u0 < u1, v0 < v1 select the array slice [v0:v1, u0:u1]. A box that
    reaches beyond the frame holds the pixels inside the frame.

    `start` and `end` are video frame numbers counted from 0, both included; every frame between
    them is measured (step 1, whatever the tracking step). The video is decoded once, from its
    beginning, in the order that defines frame numbers (SPEC 3.5). `fps_true` is the real frame
    rate in frames per s; the frame rate written in the file is not used.

    Returns one row per frame and probe, sorted by frame, the probes of a frame in the order of
    `boxes`, with the columns of probes.csv (`schema.PROBES`): `frame`; `t_s` = frame / fps_true
    in s; `probe`, the name; `r`, `g`, `b`, the means over the box's pixels on the 0 to 255 scale;
    `gray` = 0.299 r + 0.587 g + 0.114 b.

    A video can end before `end`, also when the file reports enough frames: the pass then stops at
    the last frame that decodes, returns the rows it has (none if `start` is past that frame), and
    says so in one message to `log`, a function taking a text (no `log`: nothing is said).

    Raises ValueError, before the pass starts, for a box that holds no pixel of the frame, a box
    that is not four finite numbers, an empty name, no boxes at all, `start` < 0, `end` < `start`,
    or an `fps_true` that is not a positive finite number; FileNotFoundError or IOError if the
    video cannot be opened.
    """
    if start < 0:
        raise ValueError(f"start must be a frame number, 0 or more, not {start}.")
    if end < start:
        raise ValueError(f"The clip's end (frame {end}) lies before its start (frame {start}).")
    if not (math.isfinite(fps_true) and fps_true > 0):
        raise ValueError(f"fps_true must be a positive number of frames per second, not {fps_true}.")
    if not boxes:
        raise ValueError("No probe rectangle was given: at least one probe is needed.")
    info = video.probe(video_path)
    slices = [_pixels(name, box, info.width, info.height) for name, box in boxes.items()]

    frames, means = [], []
    for frame, rgb in video.iter_rgb_frames(video_path, range(start, end + 1)):
        frames.append(frame)
        means.extend(rgb[rows, cols].mean(axis=(0, 1)) for rows, cols in slices)
    if log is not None and len(frames) < end - start + 1:
        log(_ended_early(frames, start, end, info.n_frames))

    frame_column = np.repeat(np.asarray(frames, dtype=np.int64), len(slices))
    rgb_means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    table = pd.DataFrame({
        "frame": frame_column,
        "t_s": frame_column / fps_true,
        "probe": pd.Series(list(boxes) * len(frames), dtype="str"),
        "r": rgb_means[:, 0],
        "g": rgb_means[:, 1],
        "b": rgb_means[:, 2],
        "gray": rgb_means @ np.asarray(GRAY_WEIGHTS),
    })
    return table[[column.name for column in schema.PROBES]]


def _pixels(name: str, box: Sequence[float], width: int, height: int) -> tuple[slice, slice]:
    """The (rows, columns) slices of a frame of `width` x `height` px that the box (u0, v0, u1, v1),
    image px, holds: the pixels whose centers (c + 0.5, r + 0.5) lie inside it, low edge included."""
    if not name:
        raise ValueError("A probe needs a name (for example LED1); an empty name was given.")
    try:
        corners = [float(value) for value in box]
    except (TypeError, ValueError):
        corners = []
    if len(corners) != 4 or not all(math.isfinite(value) for value in corners):
        raise ValueError(f"Probe {name!r}: a rectangle is four numbers u0,v0,u1,v1 in image px, not {box!r}.")
    u0, v0, u1, v1 = corners
    cols = _span(min(u0, u1), max(u0, u1), width)
    rows = _span(min(v0, v1), max(v0, v1), height)
    if cols.start >= cols.stop or rows.start >= rows.stop:
        raise ValueError(f"Probe {name!r}: the rectangle from ({u0:g}, {v0:g}) to ({u1:g}, {v1:g}) px holds no "
                         f"pixel of the {width} x {height} px frame. Give two opposite corners u0,v0,u1,v1 in "
                         f"image px (u to the right, v down).")
    return rows, cols


def _span(low: float, high: float, size: int) -> slice:
    """Indices i in 0 ... size - 1 with low <= i + 0.5 < high, as a slice (empty if start >= stop)."""
    return slice(max(math.ceil(low - 0.5), 0), min(math.ceil(high - 0.5), size))


def _ended_early(frames: list[int], start: int, end: int, n_reported: int) -> str:
    """The message for a pass that got fewer frames than start ... end (`frames`: those it got)."""
    first_missing = frames[-1] + 1 if frames else start
    missing = f"frame {end}" if first_missing == end else f"frames {first_missing} to {end}"
    reported = f"the file reports {n_reported} frames"
    if not frames:
        return (f"The video ends before frame {start} ({reported}): {missing} could not be read, so there are "
                f"no probe rows.")
    return (f"The video ends at frame {frames[-1]} ({reported}): {missing} could not be read. The probes stop "
            f"at frame {frames[-1]}.")
