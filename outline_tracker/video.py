"""Reading phone videos safely. Ported from the course's shrimp.video (unchanged behavior).

The functions down to `check_video` are last week's, moved over unchanged, with last week's tests
(tests/test_video.py); so is `iter_rgb_frames`, from last week's
shrimp.segment. Coordinates: frames are arrays indexed [row, column], so x is the column and y is
the row counted downward (pixel units; see SPEC 3.1 for the pixel-center rule).

Two rules this module exists to enforce:

1. Never load a whole video into memory. One minute at 240 fps and 1920x1080 is
   14,400 frames x 2 MB = about 30 GB as grayscale. iter_frames() yields one frame
   at a time instead.

2. Never trust the frame rate stored in the file for physics. Phones and sharing apps
   often re-time slow-motion videos. Real time comes from your stopwatch clip
   (fps_true in data/manifest.csv). check_video() only warns when a file looks wrong.

Note (SPEC 3.5): iter_frames(start > 0), read_frame and frame_changes seek with OpenCV, which is
not guaranteed to deliver the frame asked for (measured: one or two frames off, early or late,
when the file's timestamps have gaps). They stay as last week's, for check_video and for
occasional access; do not use them to display a frame or to sample a probe.

Frame numbers (SPEC 3.5): frame n is the n-th frame, counted from 0, of the sequential decode from
the start of the file, which is `iter_rgb_frames`. Tracking reads the video with it. A single
frame by number comes from `FrameSource` (outline_tracker/frame_source.py), which uses
`frame_timestamps` to make sure of the frame it delivers. `frame_hash` and `decoder_tag` let a
session store which frame the user saw (decision X8).
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
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


# ---------------------------------------------------------------------------------------------
# The sequential decode that defines frame numbers (last week's, from shrimp.segment, unchanged)


def iter_rgb_frames(path, frames):
    """Yield (frame number, RGB image) for the requested frame numbers (increasing), reading the video
    from the start so the numbering is exactly the decoding order Tracker uses."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}. Use the ..._tracker.mp4 copy made by shrimp.convert.")
    wanted = list(frames)
    i, k = 0, 0
    try:
        while k < len(wanted):
            if i < wanted[k]:
                if not cap.grab():
                    break
            else:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                k += 1
            i += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------------------------
# Frame identity: the time stamped on every frame, and a hash of a decoded frame

HIDDEN_PACKET = 0x4  # bit of a packet's flags in ffmpeg's list (AV_PKT_FLAG_DISCARD): decoded, never shown


def frame_timestamps(path) -> np.ndarray | None:
    """Time stamped on every frame of the file's first video stream, read without decoding.

    Returns a float64 array in s of file time (not real time), sorted, frame 0 at 0: entry k
    belongs to frame k of the sequential decode (a file with B-frames stores its frames in another
    order). Packets that the file marks as not to be shown are left out, as the decoder leaves
    their frames out (a copy cut without re-encoding has them, before the cut). None if the
    bundled ffmpeg cannot list the file's packets.
    """
    try:
        import imageio_ffmpeg  # imported here, as convert.ffmpeg_exe does

        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                   "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"]
        # CREATE_NO_WINDOW exists on Windows only: no console window flashes up under the GUI.
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, RuntimeError):
        return None
    if done.returncode != 0:
        return None
    time_base, stamps = None, []
    try:
        for line in done.stdout.splitlines():
            if line.startswith("#tb"):  # "#tb 0: 1/15360": s per timestamp unit
                numerator, denominator = line.split(":")[1].split("/")
                time_base = int(numerator) / int(denominator)
            elif line and not line.startswith("#"):  # "stream, dts, pts, duration, size, checksum[, F=0x4]"
                fields = [part.strip() for part in line.split(",")]
                flags = next((int(part[2:], 16) for part in fields[6:] if part.startswith("F=")), 0)
                if not flags & HIDDEN_PACKET:
                    stamps.append(int(fields[2]))
    except (ValueError, IndexError, ZeroDivisionError):
        return None
    if time_base is None or not stamps:
        return None
    stamps = np.sort(np.asarray(stamps, dtype=np.int64))
    return (stamps - stamps[0]) * time_base


def count_timestamp_gaps(times_s) -> int:
    """Number of gaps in a table of frame times (s, sorted, as `frame_timestamps` returns it).

    A gap is a step from one frame to the next that is longer than 1.5 times the median step: what
    a dropped frame leaves behind. An evenly timed file has none.
    """
    steps = np.diff(np.asarray(times_s, dtype=float))
    if steps.size == 0:
        return 0
    return int(np.count_nonzero(steps > 1.5 * np.median(steps)))


def frame_hash(rgb: np.ndarray) -> str:
    """Identity of a decoded frame: "sha256:" followed by the SHA-256 (hex) of its bytes, row by row.

    `rgb` is the full frame as `iter_rgb_frames` and `FrameSource.get` return it: a uint8 array
    [row, column, 3], before any crop. Frames that differ in one pixel have different hashes.
    """
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest()


def decoder_tag() -> str:
    """What decodes the frames on this computer: OpenCV's version, the platform and the machine,
    for example "opencv-5.0.0/darwin/arm64". Frame hashes compare only under the same tag (X8)."""
    return f"opencv-{cv2.__version__}/{sys.platform}/{platform.machine()}"
