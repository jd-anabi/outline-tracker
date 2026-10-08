"""Small videos written frame by frame with OpenCV, for the tests of outline_tracker/video.py
(tests/test_video.py, tests/test_video_check.py).

A position is (x, y) in px of the frame, x to the right and y down, as OpenCV draws; a frame rate is
the one written in the file, in frames per s of file time.
"""

import cv2
import numpy as np


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
