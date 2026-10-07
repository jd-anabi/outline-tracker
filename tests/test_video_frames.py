"""What video.py adds to last week's functions (SPEC 3.5; decisions X4 and X8): the sequential decode
that defines frame numbers, the table of frame timestamps, the frame hash and the decoder tag.

The ported functions are tested in tests/test_video.py (the template's file) and `FrameSource` in
tests/test_frame_source.py. Frame numbers count from 0; times are s of file time. Expected
timestamps come from how the clips are written (`frame_times` in tests/helpers.py), expected
hashes from hashlib on bytes written out here.
"""

import hashlib
import platform
import subprocess
import sys

import cv2
import numpy as np
import pytest
from helpers import GAPS_BEFORE, frame_times

from outline_tracker import video

CLIPS = ["disk_clip", "gapped_clip"]
SKIPS = {"disk_clip": (), "gapped_clip": GAPS_BEFORE}
N = 120  # frames in each clip
SHAPE = (240, 320, 3)  # rows, columns, RGB


# ---------------------------------------------------------------------------------------------
# The sequential decode and the timestamp table


def test_iter_rgb_frames_yields_the_requested_frames_of_the_sequential_decode(disk_clip):
    wanted = [0, 5, 6, 119, 500]  # the clip ends at 119: the decode stops there
    capture = cv2.VideoCapture(str(disk_clip.path))
    decoded = []
    try:
        for _ in range(N):
            decoded.append(capture.read()[1])
    finally:
        capture.release()
    got = list(video.iter_rgb_frames(disk_clip.path, wanted))
    assert [k for k, _ in got] == [0, 5, 6, 119]
    for k, rgb in got:
        assert rgb.dtype == np.uint8 and rgb.shape == SHAPE
        assert np.array_equal(rgb, decoded[k][:, :, ::-1])  # RGB = OpenCV's BGR reversed


@pytest.mark.parametrize("name", CLIPS)
def test_frame_timestamps_lists_the_time_of_every_frame(name, request):
    times = video.frame_timestamps(request.getfixturevalue(name).path)
    assert times.shape == (N,) and times.dtype == np.float64
    assert times == pytest.approx(frame_times(SKIPS[name]), abs=1e-6)  # sorted: entry k is frame k
    assert video.count_timestamp_gaps(times) == len(set(SKIPS[name]))  # 0, and 3


def test_count_timestamp_gaps_counts_steps_longer_than_the_usual_one():
    assert video.count_timestamp_gaps(np.arange(10) / 240.0) == 0
    assert video.count_timestamp_gaps(frame_times((3, 7, 7), n_frames=12)) == 2  # one of 2 and one of 3 durations
    assert video.count_timestamp_gaps(np.array([0.0])) == 0  # one frame: no step


def test_frame_timestamps_is_none_when_ffmpeg_cannot_read_the_file(tmp_path):
    text = tmp_path / "notes.mp4"
    text.write_text("this is text, not a video\n")
    assert video.frame_timestamps(text) is None
    assert video.frame_timestamps(tmp_path / "missing.mp4") is None


def test_ffmpeg_is_started_without_a_console_window_on_windows(disk_clip, monkeypatch):
    # CREATE_NO_WINDOW exists only on Windows; there, the table must not flash a console window.
    started = {}

    def run(command, **options):
        started.update(options)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    assert video.frame_timestamps(disk_clip.path) is None  # the stand-in reports a failure
    assert started["creationflags"] == 0x08000000


# ---------------------------------------------------------------------------------------------
# The frame hash and the decoder tag (decision X8)


def test_frame_hash_is_sha256_of_the_rgb_bytes():
    black = np.zeros((2, 2, 3), np.uint8)
    assert video.frame_hash(black) == "sha256:" + hashlib.sha256(bytes(12)).hexdigest()
    one_pixel = black.copy()
    one_pixel[1, 0, 2] = 1
    assert video.frame_hash(one_pixel) != video.frame_hash(black)
    # byte 8 of the 12 = row 1, column 0, blue
    assert video.frame_hash(one_pixel) == "sha256:" + hashlib.sha256(bytes(8) + b"\x01" + bytes(3)).hexdigest()


def test_frame_hash_reads_the_rows_in_order_whatever_the_memory_layout():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    mirrored = frame[:, ::-1]  # a view that is not C-contiguous
    assert not mirrored.flags.c_contiguous
    assert video.frame_hash(mirrored) == "sha256:" + hashlib.sha256(mirrored.tobytes()).hexdigest()
    assert video.frame_hash(mirrored) != video.frame_hash(frame)


def test_decoder_tag_names_opencv_the_platform_and_the_machine():
    tag = video.decoder_tag()
    assert tag == video.decoder_tag()
    assert cv2.__version__ in tag and sys.platform in tag and platform.machine() in tag
