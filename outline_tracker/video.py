"""Reading phone videos safely. Ported from the course's shrimp.video (unchanged behavior).

The functions below are last week's, moved over unchanged and checked against the reference copy
by tests/test_port_fidelity.py. Coordinates: frames are arrays indexed [row, column], so x is the
column and y is the row counted downward (pixel units; see SPEC 3.1 for the pixel-center rule).

Two rules this module exists to enforce:

1. Never load a whole video into memory. One minute at 240 fps and 1920x1080 is
   14,400 frames x 2 MB = about 30 GB as grayscale. iter_frames() yields one frame
   at a time instead.

2. Never trust the frame rate stored in the file for physics. Phones and sharing apps
   often re-time slow-motion videos. Real time comes from your stopwatch clip
   (fps_true in data/manifest.csv). check_video() only warns when a file looks wrong.

Note (SPEC 3.5): iter_frames(start > 0), read_frame and frame_changes seek with OpenCV, which is
not guaranteed to deliver the frame asked for. They stay as last week's, for check_video and for
occasional access; do not use them to display a frame or to sample a probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """What the file says about itself (not necessarily the truth)."""

    path: str
    width: int              # pixels, as delivered by iter_frames (after automatic rotation)
    height: int
    fps_container: float    # frame rate written in the file; NOT used for physics
    n_frames: int
    duration_s: float       # n_frames / fps_container: file time, not real time
    codec: str
    rotation_deg: int       # rotation metadata; OpenCV applies it automatically


@dataclass
class VideoCheck:
    info: VideoInfo
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def _open(path) -> cv2.VideoCapture:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {p}. See 'Troubleshooting' in README.md.")
    return cap


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def probe(path) -> VideoInfo:
    """Read the file's own description of itself (fast; decodes one frame)."""
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        rotation = 0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            rotation = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"Could not decode the first frame of {path}.")
        height, width = frame.shape[:2]
    finally:
        cap.release()
    duration = n / fps if fps > 0 else float("nan")
    return VideoInfo(str(path), width, height, fps, n, duration, codec or "unknown", rotation)


def iter_frames(path, start: int = 0, stop: int | None = None, step: int = 1,
                gray: bool = True) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) one frame at a time, from `start` up to (not including) `stop`.

    gray=True returns 2-D uint8 arrays (rows = y, columns = x). Frames skipped by `step`
    are not decoded into arrays, so step > 1 is faster.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    cap = _open(path)
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while stop is None or i < stop:
            if (i - start) % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, (_to_gray(frame) if gray else frame)
            elif not cap.grab():
                break
            i += 1
    finally:
        cap.release()


def read_frame(path, index: int, gray: bool = True) -> np.ndarray:
    """Return a single frame by index (seeks; fine for occasional access)."""
    cap = _open(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IndexError(f"Could not read frame {index} of {path}.")
    return _to_gray(frame) if gray else frame


def frame_changes(path, indices, scale: float = 0.25) -> np.ndarray:
    """Mean absolute gray-level change between frame i and frame i+1, for each i in `indices`.

    Used by check_video() to spot re-timed (rendered) slow motion.
    """
    cap = _open(path)
    out = []
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok_a, a = cap.read()
            ok_b, b = cap.read()
            if not (ok_a and ok_b):
                out.append(np.nan)
                continue
            a = cv2.resize(_to_gray(a), None, fx=scale, fy=scale).astype(np.float32)
            b = cv2.resize(_to_gray(b), None, fx=scale, fy=scale).astype(np.float32)
            out.append(float(np.mean(np.abs(a - b))))
    finally:
        cap.release()
    return np.asarray(out, dtype=float)


def _ramp_ratio(path, info: VideoInfo, n_pairs: int = 15) -> float | None:
    """Motion per frame near the ends of the file divided by motion per frame in the middle.

    Rendered slow motion plays the first and last ~1 s at normal speed and the middle
    slowed down, so consecutive frames differ much more near the ends (ratio >> 1).
    The first and last 0.25 s are skipped so that bumping the phone does not count.
    """
    fps, n = info.fps_container, info.n_frames
    if fps <= 0 or n < 40 or n < 4 * fps:
        return None
    skip = max(int(round(0.25 * fps)), 1)
    win = max(int(round(1.0 * fps)), 5)
    first = np.linspace(skip, skip + win - 2, min(n_pairs, win - 1)).astype(int)
    last = np.linspace(n - skip - win, n - skip - 2, min(n_pairs, win - 1)).astype(int)
    middle = np.linspace(int(0.25 * n), int(0.75 * n), 2 * n_pairs).astype(int)
    c_first = np.nanmedian(frame_changes(path, first))
    c_last = np.nanmedian(frame_changes(path, last))
    c_mid = np.nanmedian(frame_changes(path, middle))
    if not np.isfinite(c_mid) or c_mid <= 1e-6 or not (np.isfinite(c_first) and np.isfinite(c_last)):
        return None
    return float(min(c_first, c_last) / c_mid)


def check_video(path, min_slowmo_fps: float = 100.0, min_side_px: int = 720) -> VideoCheck:
    """Look for the common ways a phone video gets silently damaged before analysis."""
    info = probe(path)
    check = VideoCheck(info)
    if info.fps_container < min_slowmo_fps:
        check.warnings.append(
            f"The file says {info.fps_container:.1f} fps. A slow-motion original should say "
            f"120-240 fps. This is probably a re-timed 'compatible' copy made by the phone or "
            f"an app. Transfer the ORIGINAL file (README: 'Getting the original video off your phone')."
        )
    ratio = _ramp_ratio(path, info)
    if ratio is not None and ratio > 2.0:
        check.warnings.append(
            f"Motion per frame is {ratio:.1f}x larger near the start and end of the file than in "
            f"the middle. That is the signature of rendered slow motion (normal speed at the ends, "
            f"slowed in the middle), so frame times are not uniform. Do not analyze this file."
        )
    if min(info.width, info.height) < min_side_px:
        check.warnings.append(
            f"Low resolution ({info.width}x{info.height}). Shrimp may be only a few pixels long."
        )
    if info.rotation_deg:
        check.notes.append(
            f"The file has {info.rotation_deg} degree rotation metadata; frames are delivered "
            f"already rotated ({info.width}x{info.height}). Read every clip, including the ruler "
            f"clip, with Outline Tracker so all coordinates use the same orientation."
        )
    check.notes.append(
        "Real time comes from your stopwatch clip (fps_true in data/manifest.csv), not from this file."
    )
    return check
