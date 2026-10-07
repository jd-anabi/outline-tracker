"""Tests for the provided video module. These must always pass."""

import cv2
import numpy as np
import pytest

from shrimp import video


def _write_video(path, fps, positions, size=(160, 160), radius=10):
    """Write a dark disk on a light background, one frame per (x, y) position."""
    width, height = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert out.isOpened(), "OpenCV could not create a test video"
    for x, y in positions:
        frame = np.full((height, width, 3), 200, np.uint8)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, (40, 40, 40), -1)
        out.write(frame)
    out.release()


def _circle_path(steps_px, center=(80, 80), r=50):
    """Positions along a circle, advancing by the given arc length (px) each frame."""
    angle, positions = 0.0, []
    for step in steps_px:
        positions.append((center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)))
        angle += step / r
    return positions


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, 240, [(20 + 0.5 * i, 80) for i in range(120)])
    return path


def test_probe_reports_what_the_file_says(clip):
    info = video.probe(clip)
    assert info.fps_container == pytest.approx(240, rel=1e-3)
    assert info.n_frames == 120
    assert (info.width, info.height) == (160, 160)


def test_iter_frames_streams_every_frame_as_gray(clip):
    frames = video.iter_frames(clip)
    assert not isinstance(frames, list), "frames must be streamed, not loaded all at once"
    indices = []
    for i, frame in frames:
        assert frame.shape == (160, 160) and frame.dtype == np.uint8
        indices.append(i)
    assert indices == list(range(120))


def test_iter_frames_start_stop_step(clip):
    assert [i for i, _ in video.iter_frames(clip, start=10, stop=50, step=5)] == list(range(10, 50, 5))


def test_read_frame_matches_iter_frames(clip):
    from_iter = next(f for i, f in video.iter_frames(clip) if i == 37)
    assert np.array_equal(video.read_frame(clip, 37), from_iter)


def test_x_is_column_and_y_is_row(tmp_path):
    path = tmp_path / "corner.mp4"
    _write_video(path, 240, [(120, 30)] * 5)
    frame = video.read_frame(path, 2)
    rows, cols = np.nonzero(frame < 120)
    assert cols.mean() == pytest.approx(120, abs=1.0)  # x = column index
    assert rows.mean() == pytest.approx(30, abs=1.0)   # y = row index, counted downward


def test_check_accepts_a_true_slow_motion_file(tmp_path):
    path = tmp_path / "original.mp4"
    _write_video(path, 240, _circle_path([1.0] * 1000))
    check = video.check_video(path, min_side_px=100)
    assert check.ok, check.warnings


def test_check_flags_rendered_slow_motion(tmp_path):
    # 30 fps file: normal speed at both ends (8 px/frame), slowed 8x in the middle (1 px/frame)
    path = tmp_path / "rendered.mp4"
    _write_video(path, 30, _circle_path([8.0] * 45 + [1.0] * 300 + [8.0] * 45))
    check = video.check_video(path, min_side_px=100)
    assert not check.ok
    assert any("fps" in w for w in check.warnings)
    assert any("slow motion" in w for w in check.warnings)


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.probe(tmp_path / "does_not_exist.MOV")
