"""Test data and checks shared by the overlay tests (tests/test_overlay.py).

The truth is the synthetic scene: `record` measures a ground-truth mask as tracking would store it,
and `make_run` writes those records and a matching session into a run folder, without the tracking
runner. What must be on a drawn frame is then known from the records the test made itself.

Coordinates: records are in px of the full frame, Tracker's convention (u to the right, v downward,
pixel centers at +0.5). An overlay frame is the full frame resized, so the point (u, v) of a frame
of W x H px lies at (u * W' / W, v * H' / H) in an overlay of W' x H' px (`scaled`), in the same
convention: the pixel in column c and row r of the overlay array covers c to c + 1 and r to r + 1.
Colors are (red, green, blue), 0 to 255.
"""

import math
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from outline_tracker import overlay, schema
from outline_tracker.measure import measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import crop_to_bbox
from outline_tracker.session import Clip, Session, TimeSettings, Track

FPS_TRUE = 239.6  # frames per s; not the clips' 240, so a stamp's time shows where it came from
COLORS = {"A": "#FFFF00", "B": "#FF00FF", "C": "#0080FF"}  # as session.json holds them: RGB hex
RGB = {"A": (255, 255, 0), "B": (255, 0, 255), "C": (0, 128, 255)}  # the same colors as numbers


# ---------------------------------------------------------------------------------------------
# A run folder without the tracking runner


def record(truth, track_id, frame, stored_as=None):
    """What tracking stores for one object on one frame: the ground-truth mask of `track_id` on
    video frame `frame`, measured in full-frame px. `stored_as` gives the record another frame
    number (the scene has no frames after the clip's last one)."""
    width, height = truth.scene.size
    result = crop_to_bbox(truth.mask(track_id, frame), truth.logits(track_id, frame), obj_id=track_id, score=10.0)
    return measure_mask(result, frame if stored_as is None else stored_as, (0, 0, width, height), "coarse")


def lost_record(truth, frame):
    """What tracking stores for an object that was not found on `frame`: an empty mask, measured."""
    width, height = truth.scene.size
    return measure_mask(crop_to_bbox(np.zeros((height, width), bool), None), frame, (0, 0, width, height), "coarse")


def make_run(folder, records, colors=COLORS, fps_true=FPS_TRUE):
    """Write results.npz and session.json into `folder` (created if needed) and return the store.

    `records` maps a track id to its records (video frames ascending); the session lists these
    tracks in this order, each with its color of `colors` (RGB hex), and has fps_true `fps_true`
    (frames per s).
    """
    store = ResultsStore()
    for track_id, rows in records.items():
        for row in rows:
            store.put(track_id, row)
    folder.mkdir(parents=True, exist_ok=True)
    store.save(folder / schema.RESULTS_NPZ)
    frames = sorted({row.frame for rows in records.values() for row in rows})
    step = frames[1] - frames[0] if len(frames) > 1 else 1
    session = Session(
        student="test", clip=Clip(start=frames[0], end=frames[-1], step=step), time=TimeSettings(fps_true=fps_true),
        tracks=[Track(id=track_id, color=colors[track_id], start_frame=rows[0].frame)
                for track_id, rows in records.items()])
    session.save(folder / schema.SESSION_JSON)
    return store


class Drawn:
    """Stands in for `overlay.draw_overlay_frame` while `write_overlay` runs, and keeps what it saw.

    Every call is passed on to the real function. `calls` lists (frame, t_s) in the order of the
    calls; `frames[frame]` is the decoded RGB frame that was given; `images[frame]` is the pair
    (drawn, undrawn): the array that went into the video, and the same frame drawn with no tracks.
    """

    def __init__(self, monkeypatch):
        self.calls, self.frames, self.images = [], {}, {}
        real = overlay.draw_overlay_frame

        def spy(rgb, items, frame, t_s, width=960):
            image = real(rgb, items, frame, t_s, width)
            self.calls.append((frame, t_s))
            self.frames[frame] = rgb.copy()
            self.images[frame] = (image, real(rgb, [], frame, t_s, width))
            return image

        monkeypatch.setattr(overlay, "draw_overlay_frame", spy)


# ---------------------------------------------------------------------------------------------
# What is on a drawn frame


def scaled(points_px, frame_size, image):
    """Full-frame points (u, v) in px, for a frame of `frame_size` = (width, height) px, as
    positions (x, y) in px of the overlay array `image` ([row, column, 3]): float64, shape (n, 2)."""
    rows, cols = image.shape[:2]
    return np.asarray(points_px, float).reshape(-1, 2) * (cols / frame_size[0], rows / frame_size[1])


def marks(drawn, undrawn, color):
    """The pixels that the drawing moved toward `color`: a bool array [row, column], True where
    `drawn` differs from `undrawn` and is nearer `color` (distance in RGB) than `undrawn` is."""
    color = np.asarray(color, float)
    changed = np.any(drawn != undrawn, axis=2)
    nearer = np.linalg.norm(drawn - color, axis=2) < np.linalg.norm(undrawn - color, axis=2)
    return changed & nearer


def fraction_marked(points, marked):
    """The share of `points` ((x, y) in overlay px) that have a marked pixel within 1 px: a pixel
    of the bool array `marked` whose center (column + 0.5, row + 0.5) is at most 1 px away."""
    rows, cols = marked.shape
    hits = 0
    for x, y in points:
        col0, row0 = math.floor(x), math.floor(y)
        hits += any(
            marked[row, col] and math.hypot(col + 0.5 - x, row + 0.5 - y) <= 1.0
            for row in range(max(row0 - 1, 0), min(row0 + 2, rows))
            for col in range(max(col0 - 1, 0), min(col0 + 2, cols)))
    return hits / len(points)


def near(points, shape, reach):
    """The pixels of an image of `shape` = (rows, columns) whose center is at most `reach` px from
    one of `points` ((x, y) in overlay px): a bool array [row, column]."""
    points = np.asarray(points, float).reshape(-1, 2)
    rows, cols = shape
    row0, row1 = max(math.floor(points[:, 1].min() - reach), 0), min(math.ceil(points[:, 1].max() + reach), rows)
    col0, col1 = max(math.floor(points[:, 0].min() - reach), 0), min(math.ceil(points[:, 0].max() + reach), cols)
    mask = np.zeros(shape, bool)
    if row0 < row1 and col0 < col1:
        dx = (np.arange(col0, col1) + 0.5)[None, :, None] - points[None, None, :, 0]
        dy = (np.arange(row0, row1) + 0.5)[:, None, None] - points[None, None, :, 1]
        mask[row0:row1, col0:col1] = (dx ** 2 + dy ** 2).min(axis=2) <= reach ** 2
    return mask


# ---------------------------------------------------------------------------------------------
# What a video file holds


def decoded_frames(path):
    """Every frame of a video as an RGB uint8 array [row, column, 3], in the order of a sequential
    decode from the start (frame 0 first), read with OpenCV alone."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return frames
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def ffmpeg_report(path):
    """What the bundled ffmpeg says about a video: (number of frames it decodes, its line about
    the video stream, which names the codec, the pixel format, the size in px and the frame rate)."""
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-vf", "showinfo",
                           "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stderr
    stream = next(line for line in done.stderr.splitlines() if "Video:" in line)
    return len(re.findall(r"type:[IPB]", done.stderr)), stream
