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
  than FFmpeg, a time that is no table entry's, a frame that will not decode): open the file again
  and decode forward from frame 0, which is the definition of frame k.

Counting from frame 0 is the truth; the table and the decoder's time stamps are evidence, and are
checked wherever that costs little:
- a time stamp is a table entry's only if the two agree to `MATCH` of the shortest step between two
  frames (they are the same whole number of ticks of the file's clock, so they agree to rounding
  or not at all), never merely because that entry is the nearest;
- when the file is opened, the decoder's times for frames 0 and 1 must be the table's, or the table
  is not used;
- before the first jump is trusted, the far end of the table is looked at through a seek: the last
  frame of the file is the frame after which none comes, whatever its number, and it must carry the
  table's last time. A table that is whole frame steps off from some frame on, or a decoder whose
  times are whole frame steps off after a seek, agrees with itself at every single frame and fails
  here. Until this has succeeded, every jump decodes from frame 0;
- while the table is in use, every frame stepped to, and so every frame returned, must carry the
  table's time for its number; if one does not, frame k is counted from frame 0;
- if a frame counted from frame 0 carries another time than the table gives it, the table is not
  this file's: it is not used from then on, and the cached frames are dropped, because some may
  have been placed with it.
What these checks cannot see: a table that fits the decoder at both ends of the file and at every
frame compared, and still numbers a stretch in between differently (the decoder leaves one frame
out and the table another, with evenly spaced times around them). A jump into such a stretch is
misplaced until a count passes over it. `check_seek` compares the frames themselves, on a given
file.

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
MATCH = 0.01  # a time stamp is a table entry's time only within this fraction of the shortest frame step


class FrameSource:
    """One open video that returns any frame by number, exactly (see the module text).

    `info`: what the file says about itself (`video.probe`): frame size in px, frame count, fps of
    file time. `timestamps_s`: the table of frame times in use (s, frame 0 at 0). It is None when
    there is no table or the table does not fit the decoder's time stamps, which is found when the
    file is opened or later, while reading; every jump back then decodes from the start of the
    file. `stats` counts, since the file was opened: "cache" (frames served from the cache),
    "forward" (frames stepped over going forward), "seek" (seeks made, the one that looks at the
    far end of the table included) and "from_zero" (decodes restarted at frame 0).

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
        self._end_fits = False  # whether the table's last entry was seen to be the decoder's last frame
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
        """Take the table of frame times into use, if it fits this file and this decoder.

        That is checked on frames 0 and 1, which are known here by counting: the decoder's time for
        frame 0 is where its times start, and one frame later it must report the table's time for
        frame 1. Frame 1 is the frame held afterwards.
        """
        if self._capture.getBackendName() != "FFMPEG":
            return  # only FFmpeg's seeking and time stamps were measured
        times = video.frame_timestamps(self._path)
        if times is None or len(times) < 2 or len(times) != self.info.n_frames or not (np.diff(times) > 0).all():
            return  # no table, not one entry per frame, or two frames with the same time
        self._times_ms = times * 1000.0
        self._match_ms = MATCH * float(np.diff(self._times_ms).min())
        for frame in (0, 1):
            if not self._capture.grab():
                self._held = None
                return
            self._held = frame
            if frame == 0:
                self._first_ms = self._time_ms()  # what the decoder calls frame 0's time; the table calls it 0
            if not self._time_is(frame):
                return  # the table does not start at 0, or frame 1 does not come one table step later
        self.timestamps_s, self._end = times, len(times)

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
        """Make frame `k` the frame grabbed last; False if the video has no such frame.

        While the table is in use, the frame reached is taken only if it, and every frame stepped
        to on the way, carries the table's time for its number; if not, frame `k` is counted from
        frame 0.
        """
        ahead = -1 if self._held is None else k - self._held
        if 0 <= ahead <= FORWARD_MAX or (ahead > 0 and self.timestamps_s is None):
            reached = self._forward(ahead)
        else:
            reached = self.timestamps_s is not None and self._seek_to(k)
        if reached and (self.timestamps_s is None or self._time_is(k)):
            return True
        return self._from_zero(k)

    def _forward(self, count: int) -> bool:
        """Grab the next `count` frames. False if one does not come (the position is then unknown)
        or if, with the table in use, one carries another time than the table gives its number."""
        for _ in range(count):
            if not self._capture.grab():
                self._held = None
                return False
            self._held += 1
            self.stats["forward"] += 1
            if self.timestamps_s is not None and not self._time_is(self._held):
                return False
        return True

    def _seek_to(self, k: int) -> bool:
        """Jump to frame `k` with a seek; False if the jump cannot be trusted (the caller then
        decodes from the start).

        Frames 0 and 1 fit the table since the file was opened, but a frame reached by a seek is
        placed by its time stamp alone, and times that are whole frame steps off agree with each
        other. So before the first jump is trusted, the far end is looked at in the same way: a
        seek to the table's last frame must reach a frame that carries the table's last time, with
        no frame after it. That is tried again at each jump until it has succeeded once.
        """
        if not self._end_fits:
            fits = self._land_on(len(self._times_ms) - 1) and not self._capture.grab()
            self._held = None  # past the last frame, or somewhere unknown
            if not fits:
                return False
            self._end_fits = True
        return self._land_on(k)

    def _land_on(self, k: int) -> bool:
        """Seek a little before frame `k`, find out which frame arrived, and step forward to `k`.

        False if the frame that arrived cannot be told from its time stamp, or a frame stepped to
        carries another time than the table's, or no frame at or before `k` arrives even for a
        seek to frame 0.
        """
        back = SEEK_BACK
        while True:
            target = max(k - back, 0)
            self._seek(target)
            self._held = None
            if self._capture.grab():
                arrived = self._arrived()
                if arrived is None:
                    return False
                if arrived <= k:
                    self._held = arrived
                    return self._forward(k - arrived)
            if target == 0:
                return False
            back *= 4  # it landed after k, or after the last frame: ask for an earlier frame

    def _arrived(self) -> int | None:
        """Number of the frame grabbed last, from its time stamp and the table; None if that time
        is no entry of the table (being nearest to one is not enough)."""
        time_ms = self._time_ms() - self._first_ms
        nearest = int(np.argmin(np.abs(self._times_ms - time_ms)))
        return nearest if abs(self._times_ms[nearest] - time_ms) < self._match_ms else None

    def _time_is(self, k: int) -> bool:
        """Whether the frame grabbed last carries the time the table gives for frame `k`."""
        return abs(self._time_ms() - self._first_ms - self._times_ms[k]) < self._match_ms

    def _time_ms(self) -> float:
        """Time the decoder stamps on the frame grabbed last, in ms of file time. Every time stamp is
        read through this method (the tests replace it to model a decoder that reports other times)."""
        return self._capture.get(cv2.CAP_PROP_POS_MSEC)

    def _from_zero(self, k: int) -> bool:
        """Open the file again and grab frames 0 to `k`: slow, and exact by definition.

        False if the video ends before frame `k`; where it ends is then known for later calls.
        Every frame is known here by counting: a table that gives one of them another time than the
        decoder does is not this file's. It is not used from there on, and the cached frames are
        dropped: some may have been placed with it.
        """
        self.stats["from_zero"] += 1
        self._capture.release()
        self._capture = self._open()
        self._held = None
        for frame in range(k + 1):
            if not self._capture.grab():
                self._end = frame
                return False
            if self.timestamps_s is not None and not self._time_is(frame):
                self.timestamps_s = self._end = None  # the end was the table's too
                self._cache.clear()
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
