"""Sessions, callbacks and segmenter wrappers shared by the tracking tests (tests/test_tracking_*.py).

A test builds a session for a synthetic clip (`make_session`, `track`), runs it (`run`) with a
stand-in segmenter and a `Recorder` for the callbacks, and compares results.npz and session.json
with the clip's ground truth. `Watched` wraps a stand-in and writes down how the runner used it; the
stand-in still does the segmenting. `Renaming` is `ExactFake` answering for one track with another
object's ground truth: a piece (A2 is the animal A), or a track that jumped to another animal.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. A box is
(c0, r0, width, height) in whole px of the full frame: the part of it the model is shown. Frames
are video frame numbers.
"""

from contextlib import closing
from dataclasses import replace

import numpy as np

from outline_tracker.measure import mask_center
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Circle, Clip, Processing, Prompt, Session, TimeSettings, Track, VideoRef
from outline_tracker.tracking import Callbacks, Job, run_job
from outline_tracker.video import decoder_tag, frame_hash, iter_rgb_frames

FULL_SMALL = (0, 0, 320, 240)  # the whole frame of the fast tests' clips (helpers.SMALL)

# The dish of `dish_scene` at 320 x 240 px: center (160.3, 119.8), R = 0.45 * 240 = 108 px. With
# m = 0.03 R = 3.24 px the square of SPEC 6.2 is [49.06, 271.54] x [8.56, 231.04]: floor and ceil
# give columns 49 to 271 and rows 8 to 231.
DISH_BOX = (49, 8, 223, 224)


def dish_circle(scene):
    """The dish wall of a scene as the session stores it: center (u, v) and radius, px."""
    cu, cv, radius = scene.dish
    return Circle(center_px=[cu, cv], radius_px=radius)


def track_at(track_id, frame, points, labels, mode="coarse"):
    """A track that starts on `frame` with the given clicks: points (u, v) in px of the full frame,
    labels 1 (positive) or 0 (negative)."""
    prompt = Prompt(frame=frame, points_px=[list(point) for point in points], labels=list(labels))
    return Track(id=track_id, mode=mode, start_frame=frame, prompts=[prompt])


def center(clip, track_id, frame):
    """Where the scene puts the center of an object's body in a frame: (u, v) in px, from its path."""
    (obj,) = [obj for obj in clip.scene.objects if obj.track_id == track_id]
    u, v, _ = obj.path.pose(frame)
    return float(u), float(v)


def track(clip, track_id, frame=0, mode="coarse"):
    """A track with one positive click on the center of the scene's object of that name."""
    return track_at(track_id, frame, [center(clip, track_id, frame)], [1], mode)


def frame_identity(clip, frame):
    """What the GUI knows of the frame a click is placed on: (frame hash, decoder tag) of video
    frame `frame` as this computer decodes it."""
    with closing(iter_rgb_frames(clip.path, [frame])) as frames:
        ((_, rgb),) = frames
    return frame_hash(rgb), decoder_tag()


def clicks_at(clip, track_id, frame, on=None):
    """The clicks of a correction: one positive click on the center of the scene's object `on`
    (`track_id` itself if None) in video frame `frame`, with that frame's hash and decoder tag."""
    found, tag = frame_identity(clip, frame)
    return Prompt(frame=frame, frame_hash=found, decoder=tag,
                  points_px=[list(center(clip, on or track_id, frame))], labels=[1])


def make_session(clip, run_folder, tracks, *, start=0, end=None, step=2, circle=None, dish_crop=True):
    """A session for a synthetic clip: the clip's frames `start` to `end` (the last frame of the
    video if None) every `step`, fps_true = the scene's frame rate, and the given tracks."""
    scene = clip.scene
    width, height = scene.size
    video = VideoRef.from_file(clip.path, run_folder, width=width, height=height, n_frames=scene.n_frames,
                               fps_container=scene.fps)
    return Session(video=video, clip=Clip(start, scene.n_frames - 1 if end is None else end, step),
                   time=TimeSettings(fps_true=scene.fps, source="typed"), circle=circle,
                   processing=Processing(dish_crop=dish_crop), tracks=list(tracks))


def abc_session(clip, run_folder):
    """A session with a click on each of the dish scene's objects A, B and C in frame 0, tracked on
    the whole frame every 2nd frame, with a scale (a stick of 100 px for 3.24 mm) and the origin in
    the middle of a 320 x 240 px frame."""
    session = make_session(clip, run_folder, [track(clip, name) for name in "ABC"], dish_crop=False)
    session.calibration.stick = {"p1_px": [10.5, 20.5], "p2_px": [110.5, 20.5], "length_mm": 3.24}
    session.axes.origin_px = [160.5, 120.5]
    return session


class Recorder:
    """Callbacks that write down everything a job reports.

    progress: (done, total, s_per_frame, eta_s) of every call. results: (track id, frame, record).
    log: the lines. finished: the statuses. changes: what `save_session` was given, when
    `callbacks(record_session=True)` is used. With `cancel_after_frame`, `should_cancel` answers
    True from the moment a result for that video frame (or a later one) has been reported.
    """

    def __init__(self, cancel_after_frame=None):
        self.progress, self.results, self.log, self.finished, self.changes = [], [], [], [], []
        self.cancel_after_frame = cancel_after_frame

    def _cancelled(self):
        return self.cancel_after_frame is not None and any(
            frame >= self.cancel_after_frame for _, frame, _ in self.results)

    def callbacks(self, record_session=False):
        return Callbacks(progress=lambda *call: self.progress.append(call),
                         frame_result=lambda *call: self.results.append(call), log=self.log.append,
                         finished=self.finished.append, should_cancel=self._cancelled,
                         save_session=self.changes.append if record_session else None)


class Watched:
    """A stand-in segmenter that writes down how the runner used it.

    starts: (image shape, prompts) of every `start`; previews: the same of every `preview`.
    calls: "preview", "start", "step" and "close" in order. shapes: the shape of every image given
    to `start` or `step`. views: (frame, offset, size) of every `set_view`, which exists only if
    the stand-in has it. made: how often `make` (the job's factory) was called. Any other attribute
    is the stand-in's.
    """

    def __init__(self, inner):
        self.inner = inner
        self.starts, self.previews, self.calls, self.shapes, self.views, self.made = [], [], [], [], [], 0

    def make(self):
        self.made += 1
        return self

    def __getattr__(self, name):  # only for what this class does not define itself
        value = getattr(self.inner, name)
        if name != "set_view":
            return value

        def set_view(frame, offset, size):
            self.views.append((frame, tuple(offset), tuple(size)))
            value(frame, offset, size)

        return set_view

    def start(self, image, prompts):
        self.starts.append((image.shape, prompts))
        self.shapes.append(image.shape)
        self.calls.append("start")
        return self.inner.start(image, prompts)

    def step(self, image):
        self.shapes.append(image.shape)
        self.calls.append("step")
        return self.inner.step(image)

    def preview(self, image, prompts):
        self.previews.append((image.shape, prompts))
        self.calls.append("preview")
        return self.inner.preview(image, prompts)

    def close(self):
        self.calls.append("close")
        self.inner.close()


class Renaming:
    """`ExactFake` behind other names: a track id in `names` is answered with the ground truth of
    the scene's object it names, from video frame `from_frame` on (before it, with its own).
    {"A2": "A"}: the piece A2 is the animal A. {"A": "C"} from frame 40: the track A jumps to the
    animal C there. The runner must name each frame with `set_view`, as it does for `ExactFake`.
    """

    def __init__(self, ground_truth, names, from_frame=0):
        self.inner, self.names, self.from_frame = ExactFake(ground_truth), dict(names), from_frame
        self._frame, self._prompts = None, None

    def set_view(self, frame, offset, size):
        self._frame = frame
        self.inner.set_view(frame, offset, size)

    def _answer(self, image, prompts):
        names = self.names if self._frame >= self.from_frame else {}
        asked = [replace(prompt, obj_id=names.get(prompt.obj_id, prompt.obj_id)) for prompt in prompts]
        return [replace(result, obj_id=prompt.obj_id)
                for result, prompt in zip(self.inner.preview(image, asked), prompts, strict=True)]

    def start(self, image, prompts):
        self._prompts = list(prompts)
        return self._answer(image, prompts)

    def step(self, image):
        return self._answer(image, self._prompts)

    def preview(self, image, prompts):
        return self._answer(image, prompts)

    def close(self):
        self._prompts = None


def run(clip, session, run_folder, segmenter, recorder=None, *, track_ids=None, record_session=False):
    """Run a job on a clip with a stand-in segmenter: (status, the recorder of its callbacks)."""
    recorder = Recorder() if recorder is None else recorder
    make = segmenter.make if isinstance(segmenter, Watched) else (lambda: segmenter)
    job = Job(session, run_folder, clip.path, make, track_ids)
    return run_job(job, recorder.callbacks(record_session)), recorder


def table_truth(clip, track_id, frames):
    """The ground-truth table's centroids of one track on the given frames: (u_px, v_px, area_px)
    as arrays, px in the full frame (`mask_center` of the true mask)."""
    table = clip.table
    rows = table[table.track_id == track_id].set_index("frame").loc[list(frames)]
    return rows.u_px.to_numpy(), rows.v_px.to_numpy(), rows.area_px.to_numpy()


def truth_in_box(clip, track_id, frame, box):
    """What a model that is shown only `box` can see of an object's true mask in one frame:
    (u, v, area, edge). u, v: centroid of the part of the true mask inside the box, px in the full
    frame (NaN if nothing is inside); area: its pixels; edge: it has a pixel on the first or last
    row or column of the box. A box that hangs over the frame (a fine crop, SPEC 6.3) shows nothing
    of the object out there: it counts as ending at the frame's border."""
    c0, r0, width, height = box
    frame_width, frame_height = clip.scene.size
    c1, r1 = min(c0 + width, frame_width), min(r0 + height, frame_height)
    c0, r0 = max(c0, 0), max(r0, 0)
    part = clip.mask(track_id, frame)[r0:r1, c0:c1]
    u, v, area = mask_center(part)
    edge = bool(part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any())
    return u + c0, v + r0, area, edge


def box_truth(clip, track_id, frames, box):
    """`truth_in_box` for several frames, as four arrays: u, v, area, edge."""
    u, v, area, edge = zip(*(truth_in_box(clip, track_id, frame, box) for frame in frames))
    return np.array(u), np.array(v), np.array(area), np.array(edge)


def assert_centered(view, window, center, slack=0.01):
    """A fine crop, as the runner named it to the stand-in (`view` = (frame, offset, size)), is
    `window` px wide and high, has a whole-pixel corner, and is centered on `center` = (u, v), px
    in the full frame, to within half a pixel: the corner is rounded (SPEC 6.3). `slack`, px, is
    what the center itself may be off by (0.01 px for a centroid of the ground truth)."""
    _, (c0, r0), size = view
    assert size == (window, window)
    assert type(c0) is int and type(r0) is int
    assert abs(c0 + window / 2 - center[0]) <= 0.5 + slack
    assert abs(r0 + window / 2 - center[1]) <= 0.5 + slack
