"""`video.check_video`'s test for rendered slow motion: what its two helpers say they do
(docs/ROADMAP.md, W1 step 5).

`frame_changes` and `_ramp_ratio` are tested in tests/test_video.py only through `check_video`, on
one file with even motion and one that is fast for 1.5 s at both ends. Those two tests still passed
with another pair of frames compared, with the change not made absolute, with the faster end
taken instead of the slower, with nothing skipped and with a window of 2 s. Each test here holds
one sentence of a docstring or of the warning:

- `frame_changes`: "Mean absolute gray-level change between frame i and frame i+1, for each i in
  `indices`." The clip's frames have one gray level each, typed by hand, so the change between two
  frames is the difference of their levels;
- the warning: "larger near the start and end of the file than in the middle". A file that is fast
  at one end only is not that;
- `_ramp_ratio`: "The first and last 0.25 s are skipped". Which frames are read is recorded;
- `_ramp_ratio`: "Rendered slow motion plays the first and last ~1 s at normal speed". A file made
  so, with ends of 1 s, gets the warning.

Frames are counted from 0; a frame rate here is the one written in the file (frames per s of file
time); gray levels are on the 0-255 scale; speeds are px per frame along a circle.
"""

import cv2
import numpy as np
import pytest
from test_video import _circle_path, _write_video

from outline_tracker import video

# The codec is lossy. It gave a frame of one gray level back within 2 levels (measured with OpenCV
# 5.0.0 on macOS: 50 -> 48, 90 -> 89, 30 -> 28), so the change between two such frames was within 4
# of the difference of their levels. The limit is twice that, for another build of the codec; the
# changes looked for are 40 and 60.
CODEC_LEVELS = 8.0


def _write_levels(path, levels):
    """Write a 160 x 160 px video with one frame per entry of `levels`: every pixel of the frame has
    that gray level."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240, (160, 160))
    assert out.isOpened(), "OpenCV could not create a test video"
    for level in levels:
        out.write(np.full((160, 160, 3), level, np.uint8))
    out.release()


def _about_slow_motion(check):
    """The warnings of a `VideoCheck` that say the slow motion was rendered."""
    return [warning for warning in check.warnings if "slow motion" in warning]


def test_frame_changes_is_the_absolute_change_from_a_frame_to_the_next(tmp_path):
    # frames 0-5 have the level 50, frames 6-11 the level 90, frames 12-17 the level 30: the level
    # goes up by 40 from frame 5 to frame 6 and down by 60 from frame 11 to frame 12
    path = tmp_path / "levels.mp4"
    _write_levels(path, [50] * 6 + [90] * 6 + [30] * 6)
    changes = video.frame_changes(path, [0, 4, 5, 6, 10, 11, 12, 16])
    assert changes == pytest.approx([0, 0, 40, 0, 0, 60, 0, 0], abs=CODEC_LEVELS)
    # one value per index, in the order asked for
    assert video.frame_changes(path, [11, 3, 5]) == pytest.approx([60, 0, 40], abs=CODEC_LEVELS)


@pytest.mark.parametrize("fast_end", ["start", "end"])
def test_a_file_that_is_fast_at_one_end_only_is_not_rendered_slow_motion(tmp_path, fast_end):
    # The rendered file of tests/test_video.py without one of its fast ends: 8 px per frame for 45
    # frames at one end, 1 px per frame in the middle and at the other end. That end moves like the
    # middle, so motion is not larger "near the start and end".
    steps = [8.0] * 45 + [1.0] * 345
    path = tmp_path / "one_end.mp4"
    _write_video(path, 30, _circle_path(steps if fast_end == "start" else steps[::-1]))
    check = video.check_video(path, min_side_px=100)
    assert _about_slow_motion(check) == []


def test_the_first_and_last_quarter_second_are_not_read(tmp_path, monkeypatch):
    # 10 s at 240 fps, 2400 frames: 0.25 s is 60 frames, so the frames 0-59 and 2340-2399 are skipped
    path = tmp_path / "original.mp4"
    _write_video(path, 240, _circle_path([1.0] * 2400))
    pairs = []  # i of every pair (frame i, frame i + 1) that is compared
    real = video.frame_changes

    def recorded(file, indices):
        pairs.extend(int(i) for i in indices)
        return real(file, indices)

    monkeypatch.setattr(video, "frame_changes", recorded)
    video.check_video(path, min_side_px=100)
    assert pairs, "no pair of frames was compared"
    assert min(pairs) >= 60 and max(pairs) + 1 <= 2339


def test_check_flags_slow_motion_rendered_with_one_second_at_normal_speed(tmp_path):
    # 30 fps file: normal speed (8 px/frame) for the first and the last 30 frames, 1 s each, and
    # slowed 8x (1 px/frame) for the 300 frames between
    path = tmp_path / "rendered.mp4"
    _write_video(path, 30, _circle_path([8.0] * 30 + [1.0] * 300 + [8.0] * 30))
    check = video.check_video(path, min_side_px=100)
    assert len(_about_slow_motion(check)) == 1, check.warnings
