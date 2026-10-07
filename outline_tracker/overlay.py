"""overlay.mp4: the clip with each track's outline, centroid dot and id drawn on it (SPEC 8.8, P0 content).

The overlay is made from the stored results by decoding the clip again, not while tracking, so a
correction shows as soon as results.npz holds it. `draw_overlay_frame` is the drawing alone;
`write_overlay` reads a run folder, decodes the tracked frames and encodes the video as last week:
H.264, yuv420p, 30 frames per s whatever the clip's own rate, 960 px wide with an even height.
The look is last week's too (`shrimp.segment._overlay`): the outline as a 1 px anti-aliased line,
a filled dot of radius 2 px at the centroid, the id next to it, and the stamp
`t = 0.000 s   frame 0` in white in the top-left corner.

Coordinates and units (SPEC 3.1): an image is an RGB uint8 array indexed [row, column, 3], colors
are (red, green, blue) from 0 to 255. Points are (u, v) in px of the full video frame, Tracker's
convention: u to the right, v downward, the pixel in column c and row r covers c to c + 1 and r to
r + 1, so its center is (c + 0.5, r + 0.5). The overlay frame is the full frame resized; a point
(u, v) of a frame of W x H px lies at (u * W' / W, v * H' / H) of the overlay of W' x H' px, in the
same convention. Frames are video frame numbers; times are s, t_s = frame / fps_true.

No Qt, no torch.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from outline_tracker.fileio import atomic_write
from outline_tracker.results import ResultsStore
from outline_tracker.schema import OVERLAY_MP4, RESULTS_NPZ, SESSION_JSON
from outline_tracker.session import Session
from outline_tracker.video import iter_rgb_frames

WIDTH_PX = 960      # the overlay's default width (SPEC 8.8)
PLAYBACK_FPS = 30   # frames per s of the file: a 240 fps clip at step 2 plays 4 times slower than real time

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_SHIFT = 4          # the outline and the dot are placed to 1/16 px (OpenCV's fixed-point coordinates)
_HEX_COLOR = re.compile(r"#?([0-9a-fA-F]{6})")


@dataclass(frozen=True)
class OverlayItem:
    """What is drawn for one track on one frame.

    track_id: the id written next to the dot. color: (red, green, blue), 0 to 255.
    outline_px: the outline as an (n, 2) array of points (u, v), a closed polygon, as results.npz
    stores it. center_px: the centroid (u, v). Both are px of the full video frame, pixel centers
    at +0.5. On a frame where the track is lost they are NaN, and then nothing is drawn.
    """

    track_id: str
    color: tuple[int, int, int]
    outline_px: np.ndarray
    center_px: tuple[float, float]


def draw_overlay_frame(rgb: np.ndarray, items: Iterable[OverlayItem], frame: int, t_s: float,
                       width: int = WIDTH_PX) -> np.ndarray:
    """One frame of the overlay: the video frame resized, with the items and the stamp drawn on it.

    rgb: the full video frame, an RGB uint8 array [row, column, 3]; it is not changed.
    items: one `OverlayItem` per track, positions in px of the full frame (pixel centers at +0.5).
    They are drawn one after the other, a later one over an earlier one, each as its outline (a
    1 px line), a dot of radius 2 px at the centroid and the id beside the dot, in the item's
    color. An outline or a centroid with a value that is not finite (a lost track) is not drawn.
    frame, t_s: the video frame number and its time in s, for the stamp `t = 1.234 s   frame 296`.
    width: the overlay's width in px, an even number (yuv420p).

    Returns a new RGB uint8 array [row, column, 3], `width` px wide; its height is the even number
    of px nearest to the frame's height times width / frame width. Raises ValueError for an image
    that is not RGB uint8, or a width that is not an even number of at least 2.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        raise ValueError(f"The frame must be a uint8 image (0 to 255), not {rgb.dtype}.")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError(f"The frame must be an RGB image [row, column, 3], not an array of shape {rgb.shape}.")
    if not isinstance(width, (int, np.integer)) or width < 2 or width % 2:
        raise ValueError(f"width = {width!r}: the overlay's width must be an even number of px, 2 or more.")
    rows, cols = rgb.shape[:2]
    width = int(width)
    height = max(2, int(round(rows * (width / cols) / 2)) * 2)  # last week's rounding
    image = cv2.resize(rgb, (width, height))
    scale = np.array([width / cols, height / rows])

    def place(points_px) -> np.ndarray:
        """Full-frame (u, v) in px as OpenCV's fixed-point coordinates in the overlay, where the
        center of the pixel in column c and row r is (c, r)."""
        return np.rint((np.asarray(points_px, float) * scale - 0.5) * (1 << _SHIFT)).astype(np.int32)

    for item in items:
        color = tuple(int(part) for part in item.color)
        outline = np.asarray(item.outline_px, float).reshape(-1, 2)
        if len(outline) >= 2 and np.isfinite(outline).all():
            cv2.polylines(image, [place(outline)], True, color, 1, cv2.LINE_AA, _SHIFT)
        center = np.asarray(item.center_px, float)
        if np.isfinite(center).all():
            x, y = (int(part) for part in place(center))
            cv2.circle(image, (x, y), 2 << _SHIFT, color, -1, cv2.LINE_AA, _SHIFT)
            col, row = round(x / (1 << _SHIFT)), round(y / (1 << _SHIFT))
            cv2.putText(image, item.track_id, (col + 6, row - 6), _FONT, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"t = {t_s:.3f} s   frame {frame}", (10, 22), _FONT, 0.6, _WHITE, 2, cv2.LINE_AA)
    return image


def write_overlay(run_folder, video_path, out_path=None, width: int = WIDTH_PX,
                  log: Callable[[str], object] | None = None) -> tuple[Path, int]:
    """Write the overlay video of a run folder from its results.npz and session.json.

    run_folder: the folder that holds results.npz and session.json. video_path: the session's video
    (find it with `Session.locate_video`). out_path: where to write, a name ending in .mp4;
    `<run_folder>/overlay.mp4` by default. width: the overlay's width in px, an even number.
    log: called with one line of text if the video ends before the last tracked frame.

    The tracks are the session's, in the session's order and colors (RGB hex such as "#FFFF00");
    the frames are the video frames that results.npz holds for any of them, ascending: one overlay
    frame each, also where a track is lost (it is then not drawn). Each is decoded from the video,
    counted from the start of the file as tracking counts them, drawn by `draw_overlay_frame` from
    the stored outline and centroid (px of the full frame), and stamped with t_s = frame / fps_true
    of the session. The video plays at 30 frames per s. It is written through
    `fileio.atomic_write`: `out_path` holds the old video or the complete new one. If the video
    ends early, the overlay ends with its last decodable frame.

    Returns (path, number of frames written): the path is `out_path`, or `<stem>.new.mp4` next to
    it when `out_path` stayed locked by another program (Windows), which the caller should report.
    Raises ValueError, before anything is written, when `out_path` does not end in .mp4, fps_true is
    not known, a track's color is not RGB hex, or no track of the session has results;
    FileNotFoundError when there is no video at `video_path`; OSError when the video cannot be
    opened or holds none of the tracked frames (`out_path` is then left as it was); and what
    `Session.load` and `ResultsStore.load` raise.
    """
    import imageio_ffmpeg  # imported here, as convert.ffmpeg_exe does

    run_folder, video_path = Path(run_folder), Path(video_path)
    out_path = run_folder / OVERLAY_MP4 if out_path is None else Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        raise ValueError(f"The overlay is an .mp4 video: {out_path.name} must end in .mp4.")
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")
    session = Session.load(run_folder / SESSION_JSON)
    store = ResultsStore.load(run_folder / RESULTS_NPZ)
    fps_true = session.time.fps_true
    if fps_true is None:
        raise ValueError("The overlay's time stamp needs fps_true, and this session has none yet: take it from "
                         "the manifest or the stopwatch clip, or type it in.")
    stored = set(store.track_ids)
    tracks = [(track.id, _rgb(track.id, track.color), store.arrays(track.id))
              for track in session.tracks if track.id in stored]
    row_of = [{int(frame): row for row, frame in enumerate(arrays.frames)} for _, _, arrays in tracks]
    frames = sorted(set().union(*row_of))
    if not frames:
        raise ValueError(f"No track of the session has results in {run_folder / RESULTS_NPZ}: there is nothing "
                         "to draw. Track first.")
    drawn: list[int] = []

    def encode(tmp: Path) -> None:
        writer = None
        try:
            for frame, rgb in iter_rgb_frames(video_path, frames):
                items = [OverlayItem(track_id, color, arrays.outline_px[rows[frame]],
                                     (arrays.u[rows[frame]], arrays.v[rows[frame]]))
                         for (track_id, color, arrays), rows in zip(tracks, row_of) if frame in rows]
                image = draw_overlay_frame(rgb, items, frame, frame / fps_true, width)
                if writer is None:  # last week's encoding: it plays in PowerPoint and Google Slides
                    writer = imageio_ffmpeg.write_frames(
                        str(tmp), (image.shape[1], image.shape[0]), fps=PLAYBACK_FPS, codec="libx264",
                        pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-crf", "23"], quality=None)
                    writer.send(None)
                writer.send(np.ascontiguousarray(image))
                drawn.append(frame)
        finally:
            if writer is not None:
                writer.close()
        if not drawn:
            raise OSError(f"{video_path.name} holds none of the tracked frames (the first is frame {frames[0]}): "
                          "no overlay was written. Is it the video that was tracked?")

    written = atomic_write(out_path, encode)
    if len(drawn) < len(frames) and log is not None:
        log(f"{video_path.name} ended before the last tracked frame ({frames[-1]}): the overlay stops at frame "
            f"{drawn[-1]}, with {len(drawn)} of the {len(frames)} tracked frames.")
    return written, len(drawn)


def _rgb(track_id: str, color) -> tuple[int, int, int]:
    """A track's color of session.json, RGB hex such as "#FFFF00" (either case), as (red, green,
    blue) from 0 to 255. Raises ValueError, naming the track, for anything else."""
    match = _HEX_COLOR.fullmatch(color) if isinstance(color, str) else None
    if match is None:
        raise ValueError(f'Track {track_id} has the color {color!r} in the session file; a color is RGB hex '
                         'such as "#FFFF00".')
    digits = match.group(1)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)
