"""Synthetic test clips with exact ground truth (SPEC 13.2), and last week's selftest clip.

A scene is a few objects moving over a plain background. Every shape is an implicit function: a
signed distance d in px, positive inside (decision X16; the shapes, and the paths they move on,
are in outline_tracker/synthetic_shapes.py). The ground-truth mask is d > 0 at the pixel centers,
the ground-truth logits are d itself, and the frames are drawn from the same function
(coverage = clip(d + 1/2, 0, 1)). So the truth is known to a small fraction of a pixel, which a
shape drawn with OpenCV at a nominal size is not. `render` writes the clip as H.264 with libx264,
GOP 24, B-frames and yuv420p, like the `_tracker.mp4` copies made by `convert`.

Coordinates (SPEC 3.1): px in the full frame, u to the right, v downward; the pixel in column c
and row r has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. Angles are in
rad, counterclockwise on screen from the image's rightward direction (y up), which is the world
angle when the axis angle alpha is 0. A shape's body frame has its origin at the object's center,
xi toward the head and eta 90 degrees counterclockwise from xi. Time is frame / fps, in s.

`selftest_clip` is the clip-making part of last week's `shrimp.segment.selftest`, moved over
unchanged and checked against the reference copy by tests/test_port_fidelity.py.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import find_contours

from outline_tracker.convert import ffmpeg_exe
from outline_tracker.measure import mask_center
from outline_tracker.synthetic_shapes import Arc, Disk, Ellipse, Shrimp, Straight, shrimp_shape
from outline_tracker.tracker_io import write_tracker_file

FPS = 240.0  # frames per second of the ready-made scenes; for a synthetic clip this is also fps_true
SWIM_MM_S = 5.0  # swimming speed of the ready-made shrimp, mm/s (as in the selftest clip)
TABLE_COLUMNS = ["track_id", "frame", "u_px", "v_px", "area_px", "heading_rad", "solidity"]


# ---------------------------------------------------------------------------------------------
# Scenes


@dataclass(frozen=True)
class SceneObject:
    """One object: its track id, shape, path, and gray level (0 to 255) in the frames."""

    track_id: str
    shape: Disk | Ellipse | Shrimp
    path: Straight | Arc
    gray: int = 70


@dataclass(frozen=True)
class Led:
    """A rectangle that changes color at `onset_frame`: `off_rgb` before it, `on_rgb` from it on.

    `box_px` = (u0, v0, u1, v1), integer px: the pixels of the array slice [v0:v1, u0:u1].
    """

    box_px: tuple[int, int, int, int]
    onset_frame: int
    off_rgb: tuple[int, int, int] = (150, 140, 140)
    on_rgb: tuple[int, int, int] = (255, 240, 180)


@dataclass(frozen=True)
class Contact:
    """Two tracks that pass each other: closest in frame `frame`, `gap_px` px apart, outline to outline."""

    track_ids: tuple[str, str]
    frame: int
    gap_px: float


@dataclass(frozen=True)
class Scene:
    """What a clip shows. `size` = (width, height) in px; `fps` in frames per s (also fps_true).
    `objects` is a tuple, drawn in its order (a scene must be hashable: its results are cached).

    `dish` = (u, v, radius) of the dish wall in px, drawn as the gray level `background` inside and
    `outside` outside; without a dish the whole frame is `background`. `mm_per_px` is the scale the
    object sizes stand for. `contact` describes a contact event among the objects, if there is one.
    """

    size: tuple[int, int]
    fps: float
    n_frames: int
    objects: tuple[SceneObject, ...]
    mm_per_px: float
    dish: tuple[float, float, float] | None = None
    led: Led | None = None
    contact: Contact | None = None
    background: int = 205
    outside: int = 160


def _window(scene: Scene, obj: SceneObject, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the frame that can hold the object (its reach plus 2 px); empty if outside."""
    u, v, _ = obj.path.pose(frame)
    reach = obj.shape.reach_px + 2.0
    cols = np.arange(max(int(np.floor(u - reach)), 0), min(int(np.ceil(u + reach)), scene.size[0]))
    rows = np.arange(max(int(np.floor(v - reach)), 0), min(int(np.ceil(v + reach)), scene.size[1]))
    return rows, cols


def _distance(scene: Scene, obj: SceneObject, frame: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Signed distance of the object (px, float32, positive inside) at the centers of rows x cols."""
    u, v, heading = obj.path.pose(frame)
    x, y = (cols + 0.5 - u)[None, :], -(rows + 0.5 - v)[:, None]  # y up
    cos, sin = np.cos(heading), np.sin(heading)
    return np.asarray(obj.shape.distance(x * cos + y * sin, -x * sin + y * cos, frame / scene.fps), np.float32)


@lru_cache(maxsize=4)
def _background(scene: Scene) -> np.ndarray:
    """The frame without objects, gray levels as float32 [row, column] (shared: copy before drawing)."""
    width, height = scene.size
    image = np.full((height, width), scene.background, np.float32)
    if scene.dish is not None:
        cu, cv, radius = scene.dish
        to_wall = radius - np.hypot((np.arange(width) + 0.5 - cu)[None, :], (np.arange(height) + 0.5 - cv)[:, None])
        image = scene.outside + np.clip(to_wall + 0.5, 0.0, 1.0) * (scene.background - scene.outside)
    return image.astype(np.float32)


def render_frame(scene: Scene, frame: int) -> np.ndarray:
    """Frame `frame` of the scene as an RGB uint8 array [row, column, 3], before any compression.

    Each object covers a pixel by clip(d + 1/2, 0, 1), d being its signed distance at the pixel
    center in px, and is blended onto the background with its gray level.
    """
    image = _background(scene).copy()
    for obj in scene.objects:
        rows, cols = _window(scene, obj, frame)
        if rows.size and cols.size:
            part = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            part += np.clip(_distance(scene, obj, frame, rows, cols) + 0.5, 0.0, 1.0) * (obj.gray - part)
    rgb = np.repeat(np.rint(image).astype(np.uint8)[:, :, None], 3, axis=2)
    if scene.led is not None:
        u0, v0, u1, v1 = scene.led.box_px
        rgb[v0:v1, u0:u1] = scene.led.on_rgb if frame >= scene.led.onset_frame else scene.led.off_rgb
    return rgb


# ---------------------------------------------------------------------------------------------
# Ground truth


def _polygon_area(points: np.ndarray) -> float:
    """Area enclosed by a closed polygon given as an (n, 2) array (shoelace formula), in its units squared."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def true_solidity(shape: Disk | Ellipse | Shrimp, t_s: float = 0.0) -> float:
    """Solidity of a shape at time `t_s` (s): the area of its outline / the area of the outline's hull.

    The outline is the zero level of the shape's signed distance, sampled in the body frame on a
    grid 4 times finer than the pixels (0.25 px). It does not depend on where the object is in the
    frame, and it is not the solidity of the pixel mask. No unit.
    """
    half = int(np.ceil(4 * (shape.reach_px + 1.0)))
    axis = np.arange(-half, half + 1) / 4.0
    closed = [c for c in find_contours(shape.distance(axis[None, :], axis[:, None], t_s), 0.0)
              if np.array_equal(c[0], c[-1])]
    outline = max(closed, key=_polygon_area)
    return _polygon_area(outline) / float(ConvexHull(outline).volume)


@lru_cache(maxsize=8)
def _table(scene: Scene) -> pd.DataFrame:
    """The ground-truth table of a scene (see `GroundTruth.table`); shared, so callers get copies."""
    records = []
    for obj in scene.objects:
        beats = isinstance(obj.shape, Shrimp)
        solidity = float("nan") if beats else true_solidity(obj.shape)
        for frame in range(scene.n_frames):
            rows, cols = _window(scene, obj, frame)
            u, v, area = float("nan"), float("nan"), 0
            if rows.size and cols.size:
                u, v, area = mask_center(_distance(scene, obj, frame, rows, cols) > 0)
                u, v = u + cols[0], v + rows[0]
            if beats:
                solidity = true_solidity(obj.shape, frame / scene.fps)
            records.append((obj.track_id, frame, u, v, area, obj.path.pose(frame)[2], solidity))
    return pd.DataFrame(records, columns=TABLE_COLUMNS)


@dataclass(frozen=True)
class GroundTruth:
    """The exact answer for a scene, and the clip's path once `render` has written it."""

    scene: Scene
    path: Path | None = None

    def _object(self, track_id: str, frame: int) -> SceneObject:
        found = [obj for obj in self.scene.objects if obj.track_id == track_id]
        if not found:
            raise KeyError(f"No track {track_id!r} in this scene.")
        if not 0 <= frame < self.scene.n_frames:
            raise IndexError(f"frame {frame} is outside the clip (frames 0 to {self.scene.n_frames - 1}).")
        return found[0]

    def logits(self, track_id: str, frame: int) -> np.ndarray:
        """The object's signed distance in px, positive inside, at every pixel center of the frame.

        A float32 array [row, column] of the full frame; `logits > 0` is exactly `mask`.
        """
        obj = self._object(track_id, frame)
        width, height = self.scene.size
        return _distance(self.scene, obj, frame, np.arange(height), np.arange(width))

    def mask(self, track_id: str, frame: int) -> np.ndarray:
        """The object's true mask in the full frame, a bool array [row, column]: signed distance > 0
        at the pixel center (c + 0.5, r + 0.5). All False if the object is outside the frame."""
        obj = self._object(track_id, frame)
        mask = np.zeros(self.scene.size[::-1], bool)
        rows, cols = _window(self.scene, obj, frame)
        if rows.size and cols.size:
            mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = _distance(self.scene, obj, frame, rows, cols) > 0
        return mask

    @property
    def table(self) -> pd.DataFrame:
        """One row per track and frame, sorted by track (in scene order), then frame; a new DataFrame
        on every access.

        `u_px`, `v_px`: `mask_center` of the true mask, px in Tracker's convention (NaN if the mask
        is empty); `area_px`: its pixel count; `heading_rad`: the head direction, rad counterclockwise
        on screen from the image's rightward direction (not wrapped); `solidity`: `true_solidity`
        of the shape at that frame's time.
        """
        return _table(self.scene).copy()


def render(scene: Scene, path, crf: int = 23, skip_before: tuple[int, ...] = ()) -> GroundTruth:
    """Write the scene as an H.264 clip at `path` and return its ground truth.

    Encoded by the bundled ffmpeg with libx264, GOP 24, B-frames and yuv420p at quality `crf`
    (lower is closer to the drawn frames; `disk_scene` needs 10). The frame size, in px, must be
    even in both directions, and the folder must exist.
    `skip_before` lists frame numbers (1 to n_frames - 1) before which the timestamps skip one frame
    duration (twice the same number: two), as when a phone drops a frame. The frames and their
    numbers stay as they are; frame k is stamped (k + skips up to k) / fps s.
    """
    path = Path(path)
    width, height = scene.size
    if width % 2 or height % 2:
        raise ValueError(f"yuv420p needs an even width and height, not {width} x {height} px.")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"No such folder: {path.parent}")
    if any(not 1 <= before < scene.n_frames for before in skip_before):
        raise ValueError(f"skip_before must name frames 1 to {scene.n_frames - 1}, not {tuple(skip_before)}.")
    steps = "".join(f"+gte(N,{before})" for before in skip_before)  # N: frame number; TB: time base in s
    timing = ["-vf", f"setpts='(N{steps})/({scene.fps:g}*TB)'", "-fps_mode", "passthrough"] if steps else []
    command = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{scene.fps:g}", "-i", "-", *timing, "-c:v", "libx264",
               "-preset", "veryfast", "-crf", str(crf), "-g", "24", "-bf", "2", "-b_strategy", "0",
               "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in range(scene.n_frames):
            encoder.stdin.write(render_frame(scene, frame).tobytes())
    except OSError:
        pass  # ffmpeg stopped reading; its message is reported below
    _, errors = encoder.communicate()
    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg could not write {path}:\n{errors.decode(errors='replace').strip()}")
    return GroundTruth(scene, path)


# ---------------------------------------------------------------------------------------------
# Ready-made scenes


def dish_scene(size: tuple[int, int] = (1920, 1080), n_frames: int = 480, mm_per_px: float = 0.0324) -> Scene:
    """The dish-scale scene: three objects in a dish, a contact event and an LED (240 fps).

    `size` = (width, height) in px; `mm_per_px` sets the object sizes (body about 14 px at the
    default). The dish wall is a circle around the frame center with radius 0.45 x the shorter side.
    - A: a shrimp swimming at 5 mm/s along the wall, its center 12 px inside it.
    - B, C: two plain bodies (ellipses) on straight paths, B to the right at 5 mm/s, C to the left
      at 4 mm/s. In frame `n_frames // 2` they are side by side with a gap of 0.6 px (`contact`).
    - an LED box in the top-left corner, outside the dish, switching on at frame `n_frames // 3 + 1`.
    """
    width, height = size
    cu, cv, radius = width / 2 + 0.3, height / 2 - 0.2, 0.45 * min(size)
    shrimp = shrimp_shape(mm_per_px)
    body = Ellipse(shrimp.a_px, shrimp.b_px)
    step = SWIM_MM_S / mm_per_px / FPS  # px per frame
    contact = Contact(("B", "C"), n_frames // 2, 0.6)
    v_b = cv + 0.37
    objects = (
        SceneObject("A", shrimp, Arc((cu, cv), radius - 12.0, np.radians(200.0), step / (radius - 12.0))),
        SceneObject("B", body, Straight((cu + 0.25 - step * contact.frame, v_b), (step, 0.0))),
        SceneObject("C", body, Straight((cu + 0.25 + 0.8 * step * contact.frame, v_b + 2 * body.b_px + contact.gap_px),
                                        (-0.8 * step, 0.0))),
    )
    corner = min(size) / 240.0
    box = tuple(int(round(edge * corner)) for edge in (8, 8, 40, 28))
    return Scene(size, FPS, n_frames, objects, mm_per_px, dish=(cu, cv, radius), led=Led(box, n_frames // 3 + 1),
                 contact=contact)


def closeup_scene(size: tuple[int, int] = (1920, 1080), n_frames: int = 480, mm_per_px: float = 0.010) -> Scene:
    """The close-up scene: a shrimp beating its antennae at 9 Hz, and the same body without antennae (240 fps).

    `size` = (width, height) in px; `mm_per_px` sets the object sizes (body 47 x 20 px, antennae
    30 px long and 3 px wide at the default). Both objects circle inside the frame for any number
    of frames, so the heading turns slowly and the positions are sub-pixel.
    - A: the shrimp, at 5 mm/s on a circle of radius 0.2 x the shorter side, right of the middle.
    - B: the plain body (an ellipse), at 2 mm/s on a small circle near the left edge.
    """
    width, height = size
    shrimp = shrimp_shape(mm_per_px)
    step = SWIM_MM_S / mm_per_px / FPS  # px per frame
    radius_a, radius_b = 0.2 * min(size), 0.07 * min(size)
    objects = (
        SceneObject("A", shrimp, Arc((0.62 * width + 0.3, 0.5 * height - 0.2), radius_a, np.radians(20.0),
                                     step / radius_a)),
        SceneObject("B", Ellipse(shrimp.a_px, shrimp.b_px),
                    Arc((0.15 * width + 0.4, 0.5 * height + 0.3), radius_b, np.radians(250.0), -0.4 * step / radius_b)),
    )
    return Scene(size, FPS, n_frames, objects, mm_per_px)


def disk_scene(size: tuple[int, int] = (320, 240), n_frames: int = 120) -> Scene:
    """Three dark disks of radius 12 px on a light background, for `ThresholdFake` (240 fps).

    Render it with `crf=10`: then thresholding the decoded frames at gray 128 gives each center
    within 0.25 px. The disks move 0.5, 0.47 and 2.2 px per frame on straight paths and stay inside
    a frame of at least 320 x 240 px (`size`, width and height in px) for 120 frames.
    """
    moves = [("A", (60.3, 200.5), (0.5, 0.0)), ("B", (40.7, 110.1), (0.37, 0.29)), ("C", (30.2, 30.6), (2.08, 0.71))]
    objects = tuple(SceneObject(track_id, Disk(12.0), Straight(start, step), gray=40)
                    for track_id, start, step in moves)
    return Scene(size, FPS, n_frames, objects, mm_per_px=0.0324, background=220)


def shapes_scene(size: tuple[int, int] = (640, 480), n_frames: int = 60, mm_per_px: float = 0.010) -> Scene:
    """Two shapes with known descriptors inside a dish circle (240 fps), for the export tests.

    - A: a plain ellipse with semi-axes 40 and 15 px, at 0.5 px per frame on a circle of radius
      60 px, so it turns by 0.48 degrees per frame.
    - B: a disk of radius 50 px, drifting by (0.21, -0.13) px per frame.
    The sizes are in px whatever `mm_per_px` says. `size` = (width, height) must be at least
    640 x 480 px; the dish circle has radius 0.45 x the shorter side around the frame center.
    """
    width, height = size
    if width < 640 or height < 480:
        raise ValueError(f"shapes_scene needs a frame of at least 640 x 480 px, not {width} x {height} px.")
    cu, cv, radius = width / 2 + 0.3, height / 2 - 0.4, 0.45 * min(size)
    objects = (
        SceneObject("A", Ellipse(40.0, 15.0), Arc((cu - 100.0, cv + 0.25), 60.0, np.radians(100.0), 0.5 / 60.0)),
        SceneObject("B", Disk(50.0), Straight((cu + 110.3, cv + 20.6), (0.21, -0.13))),
    )
    return Scene(size, FPS, n_frames, objects, mm_per_px, dish=(cu, cv, radius))


# ---------------------------------------------------------------------------------------------
# Last week's selftest clip


def selftest_clip(folder) -> dict:
    """Write last week's selftest clip into `folder`: a made-up 1080p video of one shrimp-sized dark
    ellipse swimming 5 mm/s (`selftest_tracker.mp4`, 40 frames at 240 fps, 0.0324 mm/px), and a
    Tracker export of its true positions on every 2nd frame (`selftest.csv`).

    Returns `video` and `export` (the two paths), `frames` (the exported frame numbers), and
    `pixelx`, `pixely`: the true centers on those frames, px in Tracker's convention (pixel
    centers at +0.5). The export's x and y are in mm, origin at the frame center, y up.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    video, export = folder / "selftest_tracker.mp4", folder / "selftest.csv"
    w, h, mm_per_px, n = 1920, 1080, 0.0324, 40
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 205.0 - 15.0 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.49 * h) ** 2
    truth = [(700.0 + 0.6 * f, 500.0 + 0.2 * f) for f in range(n)]  # array coordinates: pixel centers at integers
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 240, (w, h))
    angle = float(np.degrees(np.arctan2(0.2, 0.6)))
    for x, y in truth:
        img = base.copy()
        cv2.ellipse(img, (int(round(x * 16)), int(round(y * 16))), (8 * 16, 3 * 16), angle, 0, 360, 70.0, -1,
                    cv2.LINE_AA, 4)
        img = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    out.release()
    frames = list(range(0, n, 2))
    tpx = np.array([truth[f][0] + 0.5 for f in frames])  # Tracker's pixel convention (+0.5)
    tpy = np.array([truth[f][1] + 0.5 for f in frames])
    write_tracker_file(export, "selftest", frames, np.array(frames) / 240.0, (tpx - w / 2) * mm_per_px,
                       -(tpy - h / 2) * mm_per_px, tpx, tpy)
    return {"video": video, "export": export, "frames": frames, "pixelx": tpx, "pixely": tpy}
