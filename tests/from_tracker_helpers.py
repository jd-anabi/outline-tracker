"""Clips, Tracker exports, a run with the stand-in model, and readers, shared by the from-tracker tests
(tests/test_from_tracker*.py).

`disk_video` is the template's (tests/reference/template_tests/test_segment.py), copied unchanged;
tests/test_port_fidelity.py compares the two. `tracker_map`, `MM_PER_PX`, `W` and `H` are the
template's too, taken from tests/test_tracker_io.py, where they are ported verbatim.

Coordinates: a clip is 320 x 240 px at 240 frames per s. `disk_video` takes disk centers with pixel
centers at whole numbers, so a disk drawn at (x, y) is at (x + 0.5, y + 0.5) px in Tracker's
convention (SPEC 3.1: u to the right, v downward, pixel centers at +0.5), which is what an export's
pixelx and pixely hold. An export's x and y are mm in the user's axes with y up; `tracker_map` is
Tracker's map with the origin at (160, 120) px and 0.05 mm per px. Frames are video frame numbers.
"""

import hashlib
import json

import cv2
import numpy as np
import pandas as pd
from conftest import java_sci
from test_tracker_io import MM_PER_PX, H, W, tracker_map

from outline_tracker.from_tracker import from_tracker
from outline_tracker.segmenter.fake import ThresholdFake

FPS = 240.0  # what Tracker's t column of the exports made here stands for, frames per s
# what a run leaves in its folder, with the overlay and without probes (SPEC 8.1); MODEL is a folder
MODEL = "stand-in"
RUN_FILES = ["README.txt", "outlines.npz", "overlay.mp4", "positions.csv", "radial.csv", "results.npz", "run.log",
             "session.json", "shapes.csv", MODEL]


def run(video_path, export, lines=None, **options):
    """`from_tracker` with the stand-in model, without the overlay unless asked for; console lines go
    to `lines`."""
    options.setdefault("overlay", False)
    options.setdefault("segmenter", ThresholdFake())
    log = (lambda *a: None) if lines is None else lines.append
    return from_tracker(video_path, export, model=MODEL, log=log, **options)


def names_in(folder):
    """The names of the files and folders directly in a folder, sorted."""
    return sorted(p.name for p in folder.iterdir())


def disk_video(path, centers, n=200):
    """A 240 fps video of dark disks; centers(n) gives their (x, y) in frame n (pixel centers at integers)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240, (W, H))
    for f in range(n):
        img = np.full((H, W, 3), 220, np.uint8)
        for x, y in centers(f):
            cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), 6 * 16, (40, 40, 40), -1, cv2.LINE_AA, 4)
        out.write(img)
    out.release()


def mirrored_map(px, py):
    """A map no Tracker export has: x to the right and y DOWN the screen, 0.05 mm per px."""
    return MM_PER_PX * (np.asarray(px) - 160.0), MM_PER_PX * (np.asarray(py) - 120.0)


def rotated_map(angle_deg, origin):
    """Tracker's map with the origin at `origin` (px) and +x turned `angle_deg` counterclockwise on screen."""
    return lambda px, py: tracker_map(px, py, angle_deg, origin)


def track_text(rows, to_mm=tracker_map, name="mass A"):
    """One point mass as Tracker exports it, like the template's `export_text`, for any pixel -> mm map.
    rows: (t in s, frame, pixelx, pixely)."""
    lines = [f",{name},,,,,", "t,frame,x,y,pixelx,pixely,"]
    for t, f, px, py in rows:
        x, y = to_mm(px, py)
        lines.append(",".join(java_sci(float(v)) for v in (t, f, x, y, px, py)) + ",")
    return "\n".join(lines) + "\n"


def write_start_file(path, marks, frame=40, to_mm=tracker_map):
    """Several point masses marked once on one frame and exported together, as the template's tests write
    it: `#multi:`, a line of names, the columns repeated. marks: {name: (pixelx, pixely)}. The folder is
    created. Returns the path."""
    rows = ["#multi:", "," + "".join(f"{name},,,,," for name in marks), "t," + "frame,x,y,pixelx,pixely," * len(marks)]
    cells = [java_sci(0.0)]
    for px, py in marks.values():
        cells += [java_sci(float(v)) for v in (frame, *to_mm(px, py), px, py)]
    rows.append(",".join(cells) + ",")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
    return path


def one_disk(folder, frames=range(40, 61, 4), n=70, student="ana", to_mm=tracker_map):
    """The clip of the template's whole-run test, shortened, and its Tracker track: a disk at
    (60.5 + 0.5 f, 100.5) px in frame f, marked on `frames`, with Tracker's t counted from the first of
    them at 240 frames per s. The video is `<folder>/clip_tracker.mp4` (n frames), the export
    `<folder>/<student>/A.csv`. Returns (video, export)."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, lambda f: [(60 + 0.5 * f, 100)], n)
    export = folder / student / "A.csv"
    export.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    export.write_text(track_text([((f - first) / FPS, f, 60.5 + 0.5 * f, 100.5) for f in frames], to_mm, name="A"))
    return video, export


def two_disks(folder, n=70, student="ana"):
    """The clip of the template's many-shrimp test, shortened, and its start file: A at
    (60.5 + 0.5 f, 60.5) px and B at (250.5 - 0.25 f, 170.5) px in frame f, both marked on frame 40. The
    video is `<folder>/clip_tracker.mp4`, the export `<folder>/<student>/extra/start.csv`.
    Returns (video, export)."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, lambda f: [(60 + 0.5 * f, 60), (250 - 0.25 * f, 170)], n)
    marks = {"A": (60.5 + 20, 60.5), "B": (250.5 - 10, 170.5)}
    return video, write_start_file(folder / student / "extra" / "start.csv", marks)


def read_track(path):
    """A Tracker-format file as a table with the columns t, frame, x, y, pixelx, pixely."""
    return pd.read_csv(path, skiprows=1)


def session_json(run_folder):
    """The session.json of a run folder, parsed."""
    return json.loads((run_folder / "session.json").read_text(encoding="utf-8"))


def frame_sha256(video, frame):
    """"sha256:" and the SHA-256 of video frame `frame` as RGB bytes, read here with OpenCV alone: every
    frame from the start of the file, in order."""
    capture = cv2.VideoCapture(str(video))
    try:
        for _ in range(frame + 1):
            ok, bgr = capture.read()
            assert ok, f"{video} has no frame {frame}"
    finally:
        capture.release()
    return "sha256:" + hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


class StopsAfter(ThresholdFake):
    """A stand-in that raises `error` when it is given its image number `images` + 1: the first `images`
    images of the job are tracked. With KeyboardInterrupt it is Ctrl+C during that frame."""

    def __init__(self, images, error=KeyboardInterrupt):
        super().__init__()
        self.left, self.error = images, error

    def step(self, image):
        if self.left == 0:
            raise self.error("stopped by the test")
        self.left -= 1
        return super().step(image)


class Loaded(ThresholdFake):
    """A stand-in with the facts a loaded model has: the device it runs on."""

    device = "test-device"
