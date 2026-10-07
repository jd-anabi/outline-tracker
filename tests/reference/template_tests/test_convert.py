"""Tests for the provided converter (a copy of the video that Tracker can open). These must always pass."""

import cv2
import numpy as np
import pytest

from shrimp import convert, video


def _write_video(path, n_frames=120, fps=240):
    """A dark disk moving 1 px to the right every frame, so each frame can be recognized."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert out.isOpened(), "OpenCV could not create a test video"
    for i in range(n_frames):
        frame = np.full((120, 160, 3), 200, np.uint8)
        cv2.circle(frame, (20 + i, 60), 8, (40, 40, 40), -1)
        out.write(frame)
    out.release()


def _disk_x(frame):
    rows, cols = np.nonzero(frame < 120)
    return cols.mean()


def test_copy_keeps_every_frame_in_order(tmp_path):
    src = tmp_path / "clip.mp4"
    _write_video(src)
    dst = convert.convert_for_tracker(src)
    assert dst == tmp_path / "clip_tracker.mp4"
    info = video.probe(dst)
    assert info.n_frames == 120
    assert info.fps_container == pytest.approx(240, rel=1e-3)
    assert info.codec.lower() in ("avc1", "h264")
    for i in (0, 37, 119):
        assert _disk_x(video.read_frame(dst, i)) == pytest.approx(20 + i, abs=0.5), f"frame {i}"


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert.convert_for_tracker(tmp_path / "does_not_exist.MOV")
