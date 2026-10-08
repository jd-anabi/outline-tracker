"""What the tests of exact random access share (tests/test_frame_source.py,
tests/test_frame_source_times.py, tests/test_frame_source_jumps.py): the two clips, the frames that are
read, and two faults made on purpose.

The clips are the fixtures `disk_clip` and `gapped_clip` of tests/conftest.py, named here by the names
of those fixtures: 120 frames each, 1/240 s per frame, `gapped_clip` with three gaps in its time
stamps. Frame numbers count from 0; times are s or ms of file time, as each name says. A frame is an
RGB uint8 array [row, column, 3], and the expected one comes from the sequential decode
`video.iter_rgb_frames` (the fixtures `sequential_frames` and `clip_with_frames` of tests/conftest.py).
"""

import numpy as np
from helpers import GAPS_BEFORE

from outline_tracker import video

CLIPS = ["disk_clip", "gapped_clip"]
SKIPS = {"disk_clip": (), "gapped_clip": GAPS_BEFORE}
N = 120  # frames in each clip
STEP_S = 1.0 / 240.0  # one frame step in s
STEP_MS = 1000.0 * STEP_S  # and in ms
LATER = 50  # the faults that begin inside the clip begin at this frame
FAR = N - 3  # read first: more than 64 frames ahead of where a source is after opening, so a jump


def frames_to_test(n=N, seed=13):
    """20 seeded random frame numbers plus 0, 1, 23, 24, 25, n - 2, n - 1: 27 different ones, shuffled."""
    fixed = [0, 1, 23, 24, 25, n - 2, n - 1]
    rng = np.random.default_rng(seed)
    others = [k for k in range(n) if k not in fixed]
    frames = fixed + [int(k) for k in rng.choice(others, 20, replace=False)]
    rng.shuffle(frames)
    return frames


def shift_seeks(source, shift, monkeypatch):
    """Make every seek of `source` ask for a frame `shift` frames away; returns the list of its calls."""
    calls, seek = [], source._seek

    def shifted(frame):
        calls.append(frame)
        seek(max(frame + shift, 0))

    monkeypatch.setattr(source, "_seek", shifted)
    return calls


def same(frame, expected):
    """Whether `frame` is a uint8 array that equals `expected`, value for value."""
    return frame is not None and frame.dtype == np.uint8 and np.array_equal(frame, expected)


def change_the_table(monkeypatch, change):
    """Make `video.frame_timestamps` return `change(times_s)` of the file's real table."""
    real = video.frame_timestamps
    monkeypatch.setattr(video, "frame_timestamps", lambda path: change(real(path)))


def every_tested_frame_is_exact(source, frames):
    """Read frame 117 (a jump, if the table is in use) and then the 27 frames in their shuffled
    order, which begins with steps forward; `frames` is the clip's sequential decode."""
    for k in [FAR, *frames_to_test()]:
        assert same(source.get(k), frames[k]), f"frame {k} differs"
    return True
