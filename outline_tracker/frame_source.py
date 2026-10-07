"""Exact random access to the frames of a video (SPEC 3.5, 6.5, 13.6; decision X4).

`FrameSource.get(k)` returns frame k of the sequential decode from the start of the file
(`video.iter_rgb_frames`, the numbering tracking and Tracker use), bit for bit, without decoding
the whole file for every jump. Plain OpenCV seeking does not do that: when the file's timestamps
have gaps it delivers a frame one or two frames early or late and still reports the frame asked
for (measured; `convert` keeps the phone's timestamps, so real files may have gaps).

The strategy, with one capture kept open:
- up to `FORWARD_MAX` frames ahead of the frame last decoded: step forward;
- otherwise seek `SEEK_BACK` frames early, read the time stamped on the frame that arrived, look
  it up in the table of all frame times (`video.frame_timestamps`, made once, no decoding) to
  learn which frame it is, and step forward from there to k;
- whenever something does not match (no table, a table that does not fit the file, a backend other
  than FFmpeg, a time that is in no table entry's reach, a frame that will not decode): open the
  file again and decode forward from frame 0, which is the definition of frame k.

Frames are RGB uint8 arrays [row, column, 3] of the full frame (SPEC 3.1: the pixel in column c and
row r has its center at (c + 0.5, r + 0.5) px). Frame numbers count from 0. Times are s of file
time, not real time. Nothing here imports Qt; a FrameSource is used from one thread at a time.
"""

from __future__ import annotations

import operator
from collections import Counter, OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from outline_tracker import video

CACHE_BYTES = 400_000_000  # the cache never holds more than this (64 frames of 1920 x 1080 px are 398 MB)
FORWARD_MAX = 64  # up to this many frames ahead, step forward instead of seeking
SEEK_BACK = 4  # a seek asks for the frame this many frames before the wanted one


class FrameSource:
    """One open video that returns any frame by number, exactly (see the module text).

    `info`: what the file says about itself (`video.probe`): frame size in px, frame count, fps of
    file time. `timestamps_s`: the table of frame times in use (s, frame 0 at 0), or None when
    there is none, and every jump back decodes from the start of the file. `stats` counts, since
    the file was opened: "cache" (frames served from the cache), "forward" (frames stepped over
    going forward), "seek" (seeks made) and "from_zero" (decodes restarted at frame 0).

    Close it when done, or use it in a `with` block: on Windows the file stays locked while open.
    """

    def __init__(self, path, cache_frames: int = 64):
        """Open the video at `path`. The cache holds at most `cache_frames` frames and never more
        than `CACHE_BYTES` bytes, but always the frame returned last."""
        self.info = video.probe(path)  # FileNotFoundError or IOError, with a message for the user
        self.cache_frames = cache_frames
        self.stats: Counter[str] = Counter()
        self.timestamps_s: np.ndarray | None = None
        self._path = str(path)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._held: int | None = -1  # number of the frame grabbed last (-1: none yet; None: not known)
        self._end: int | None = None  # the video has no frame with this number or a higher one, if known
        self._capture: cv2.VideoCapture | None = self._open()  # as iter_rgb_frames opens it
        try:
            self._use_timestamps()
        except BaseException:
            self.close()
            raise

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            capture.release()
            raise IOError(f"OpenCV could not open {self._path}.")
        return capture

    def _use_timestamps(self) -> None:
        """Take the table of frame times into use, if it fits this file and this decoder."""
        if self._capture.getBackendName() != "FFMPEG":
            return  # only FFmpeg's seeking and time stamps were measured
        times = video.frame_timestamps(self._path)
        if times is None or len(times) < 2 or len(times) != self.info.n_frames or not (np.diff(times) > 0).all():
            return  # no table, not one entry per frame, or two frames with the same time
        if not self._capture.grab():
            self._held = None
            return
        self._held, self._end = 0, len(times)
        self._first_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)  # what the decoder calls frame 0's time
        self._times_ms = times * 1000.0
        self._reach_ms = 0.5 * float(np.diff(self._times_ms).min())  # half the shortest step between frames
        self.timestamps_s = times

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Release the file and drop the cached frames. Calling it again does nothing."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._cache.clear()

    @property
    def cached_frames(self) -> list[int]:
        """Numbers of the frames in the cache, the least recently used first."""
        return list(self._cache)

    def get(self, k: int) -> np.ndarray:
        """Frame `k` (counted from 0) as an RGB uint8 array [row, column, 3], read-only.

        It equals the k-th frame of `video.iter_rgb_frames`, bit for bit. Raises IndexError if the
        video has no frame k (also when it ends before the frame count the file reports).
        """
        k = operator.index(k)
        if self._capture is None:
            raise ValueError(f"This FrameSource is closed ({self._path}).")
        if k in self._cache:
            self._cache.move_to_end(k)
            self.stats["cache"] += 1
            return self._cache[k]
        if k < 0 or (self._end is not None and k >= self._end) or not self._go_to(k):
            raise IndexError(f"{self._path} has no frame {k}.")
        ok, bgr = self._capture.retrieve()
        if not ok:
            self._held = None
            raise IndexError(f"{self._path} has no frame {k}.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # the array is shared with the cache
        self._cache[k] = rgb
        limit = max(1, min(self.cache_frames, CACHE_BYTES // rgb.nbytes))
        while len(self._cache) > limit:
            self._cache.popitem(last=False)
        return rgb

    def _seek(self, frame: int) -> None:
        """Ask OpenCV to deliver `frame` with the next grab. Every seek goes through this method
        (the tests replace it to land elsewhere), and what it delivers is never taken on trust."""
        self.stats["seek"] += 1
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def _go_to(self, k: int) -> bool:
        """Make frame `k` the frame grabbed last; False if the video has no such frame."""
        ahead = -1 if self._held is None else k - self._held
        if 0 <= ahead <= FORWARD_MAX or (ahead > 0 and self.timestamps_s is None):
            if self._forward(ahead):
                return True
        elif self.timestamps_s is not None and self._seek_to(k):
            return True
        return self._from_zero(k)

    def _forward(self, count: int) -> bool:
        """Grab the next `count` frames; False, and the position unknown, if one does not come."""
        for _ in range(count):
            if not self._capture.grab():
                self._held = None
                return False
        self._held += count
        self.stats["forward"] += count
        return True

    def _seek_to(self, k: int) -> bool:
        """Seek a little before frame `k`, find out which frame arrived, and step forward to `k`.

        False if the frame that arrived cannot be told from its time stamp, or lies after `k` even
        for a seek to frame 0, or a frame does not come: the caller then decodes from the start.
        """
        back = SEEK_BACK
        while True:
            target = max(k - back, 0)
            self._seek(target)
            self._held = None
            if not self._capture.grab():
                return False
            arrived = self._arrived()
            if arrived is None:
                return False
            if arrived <= k:
                self._held = arrived
                return self._forward(k - arrived)
            if target == 0:
                return False
            back *= 4  # it landed after k: ask for an earlier frame

    def _arrived(self) -> int | None:
        """Number of the frame grabbed last, from its time stamp and the table; None if no entry
        of the table is within half the shortest frame step of that time."""
        time_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC) - self._first_ms
        nearest = int(np.argmin(np.abs(self._times_ms - time_ms)))
        return nearest if abs(self._times_ms[nearest] - time_ms) < self._reach_ms else None

    def _from_zero(self, k: int) -> bool:
        """Open the file again and grab frames 0 to `k`: slow, and exact by definition.

        False if the video ends before frame `k`; where it ends is then known for later calls.
        """
        self.stats["from_zero"] += 1
        self._capture.release()
        self._capture = self._open()
        self._held = None
        for frame in range(k + 1):
            if not self._capture.grab():
                self._end = frame
                return False
        self._held = k
        return True


@dataclass(frozen=True)
class SeekCheck:
    """Result of `check_seek`. `n_frames`: the frame count the file reports; `tested`: how many
    frames were compared; `exact`: how many of them were bit-identical; `gaps`: the number of gaps
    in the file's timestamps (`video.count_timestamp_gaps`), None when no table is in use."""

    n_frames: int
    tested: int
    exact: int
    gaps: int | None


def check_seek(path, count: int = 20) -> SeekCheck:
    """Compare `count` random frames from `FrameSource` with the sequential decode of the file.

    The frame numbers (counted from 0) are the same on every call for a given frame count, and are
    read in their random order, as jumps; a clip with fewer frames is tested whole. Only hashes are
    kept, so the memory does not grow with the file.
    """
    with FrameSource(path) as source:
        n_frames = source.info.n_frames
        frames = [int(k) for k in np.random.default_rng(0).permutation(max(n_frames, 0))[:count]]
        jumped = {}
        for k in frames:
            try:
                jumped[k] = video.frame_hash(source.get(k))
            except IndexError:
                pass  # counts as not exact
        gaps = None if source.timestamps_s is None else video.count_timestamp_gaps(source.timestamps_s)
    exact = sum(jumped.get(k) == video.frame_hash(rgb) for k, rgb in video.iter_rgb_frames(path, sorted(frames)))
    return SeekCheck(n_frames, len(frames), exact, gaps)
