"""Sessions, callbacks and segmenter wrappers shared by the tracking tests (tests/test_tracking_*.py).

A test builds a session for a synthetic clip (`make_session`, `track`), runs it (`run`) with a
stand-in segmenter and a `Recorder` for the callbacks, and compares results.npz and session.json
with the clip's ground truth. `Watched` wraps a stand-in and writes down how the runner used it; the
stand-in still does the segmenting.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. A box is
(c0, r0, width, height) in whole px of the full frame: the part of it the model is shown. Frames
are video frame numbers.
"""

import numpy as np

from outline_tracker.measure import mask_center
from outline_tracker.session import Circle, Clip, Processing, Prompt, Session, TimeSettings, Track, VideoRef
from outline_tracker.tracking import Callbacks, Job, run_job

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

    starts: (image shape, prompts) of every `start`. calls: "start", "step" and "close" in order.
    views: (frame, offset, size) of every `set_view`, which exists only if the stand-in has it.
    made: how often `make` (the job's factory) was called. Any other attribute is the stand-in's.
    """

    def __init__(self, inner):
        self.inner = inner
        self.starts, self.calls, self.views, self.made = [], [], [], 0

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
        self.calls.append("start")
        return self.inner.start(image, prompts)

    def step(self, image):
        self.calls.append("step")
        return self.inner.step(image)

    def preview(self, image, prompts):
        return self.inner.preview(image, prompts)

    def close(self):
        self.calls.append("close")
        self.inner.close()


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
    row or column of the box."""
    c0, r0, width, height = box
    part = clip.mask(track_id, frame)[r0:r0 + height, c0:c0 + width]
    u, v, area = mask_center(part)
    edge = bool(part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any())
    return u + c0, v + r0, area, edge


def box_truth(clip, track_id, frames, box):
    """`truth_in_box` for several frames, as four arrays: u, v, area, edge."""
    u, v, area, edge = zip(*(truth_in_box(clip, track_id, frame, box) for frame in frames))
    return np.array(u), np.array(v), np.array(area), np.array(edge)
