"""Run folders and exact answers shared by the export tests (tests/test_export*.py).

A run folder is made the way the tool makes one: coarse tracks by `tracking.run_job` with `ExactFake`
(`coarse_run`), a fine track by `add_fine`, which stores what the fine runner stores: the scene's
ground-truth mask and logits cut to a W x W window centered on the object, measured with
`measure_mask(..., input_box=<that window>, mode="fine")`. Every session here has the same stick,
so the scale is known: 300 px for 30 mm, K = 0.1 mm per px.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). World coordinates are mm, y up (SPEC 3.2); with the
axis angle 0 used here, x = K (u - u0) and y = -K (v - v0), and an angle on screen (counterclockwise
from the image's rightward direction) is the same angle in the world. Frames are video frame numbers.
"""

import copy
import math
from dataclasses import replace

import numpy as np
import pandas as pd
from overlay_helpers import make_run
from tracking_helpers import center, dish_circle, make_session, run, track

from outline_tracker import schema
from outline_tracker.measure import measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import ObjectPrompt
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import RunRecord, Session, Track

STICK = {"p1_px": [10.0, 20.0], "p2_px": [310.0, 20.0], "length_mm": 30.0}  # 300 px long
K = 0.1                  # mm per px with that stick
ORIGIN = (100.5, 60.25)  # the origin of the axes, px (the axis angle stays 0)
MODEL = "edgetam"        # `processing.model` of a new session: the name of the Tracker-format folder

# the files `export_all` leaves in a run folder that was tracked, without probes and without the overlay
RUN_FILES = {schema.SESSION_JSON, schema.RESULTS_NPZ, schema.POSITIONS_CSV, schema.SHAPES_CSV, schema.RADIAL_CSV,
             schema.OUTLINES_NPZ, schema.RUN_LOG, schema.README_TXT, MODEL}


def world(u, v):
    """Image px (u, v) -> world mm (x, y) of the sessions made here; numbers or arrays."""
    return K * (np.asarray(u, float) - ORIGIN[0]), -K * (np.asarray(v, float) - ORIGIN[1])


def calibrate(session):
    """Give a session the stick and the axes of this module; returns it."""
    session.calibration.stick = copy.deepcopy(STICK)
    session.axes.origin_px = list(ORIGIN)
    return session


def load(run_folder):
    """(session, store) of a run folder."""
    return Session.load(run_folder / schema.SESSION_JSON), ResultsStore.load(run_folder / schema.RESULTS_NPZ)


def coarse_run(clip, run_folder, track_ids, *, dish_crop=True, heads=None, model=MODEL, step=2):
    """Track the named objects of a synthetic clip, coarse, with `ExactFake` through `run_job`, into
    `run_folder` (results.npz and session.json). The session has the scene's dish circle if the
    scene has a dish, the stick of this module, and for the ids in `heads` a head click (u, v) in
    px. Returns the session."""
    scene = clip.scene
    circle = dish_circle(scene) if scene.dish is not None else None
    tracks = [track(clip, track_id) for track_id in track_ids]
    for one in tracks:
        one.color = "#FFFF00"  # the overlay needs a color
        if heads and one.id in heads:
            one.head_px = [float(value) for value in heads[one.id]]
    session = calibrate(make_session(clip, run_folder, tracks, step=step, circle=circle, dish_crop=dish_crop))
    session.processing.model = model
    status, _ = run(clip, session, run_folder, ExactFake(clip))
    assert status == "complete"
    return session


def empty_run(clip, run_folder):
    """A run folder with a calibrated session for the clip and no track yet. Returns the session."""
    scene = clip.scene
    session = calibrate(make_session(clip, run_folder, [], circle=dish_circle(scene) if scene.dish else None))
    run_folder.mkdir(parents=True, exist_ok=True)
    session.save(run_folder / schema.SESSION_JSON)
    return session


def store_run(run_folder, records):
    """A calibrated run folder made without the tracking runner: `records` maps a track id (any
    text) to its records, as `overlay_helpers.make_run` takes them; fps_true is 239.6 there, and the
    session names no video. Returns the session."""
    make_run(run_folder, records, colors=dict.fromkeys(records, "#FFFF00"))
    session = calibrate(Session.load(run_folder / schema.SESSION_JSON))
    session.save(run_folder / schema.SESSION_JSON)
    return session


def fine_records(clip, track_id, frames, window):
    """What the fine runner stores for one object: on each frame the ground truth inside a
    `window` x `window` px square centered on the object (whole-pixel corner), measured as "fine"."""
    fake = ExactFake(clip)
    image = np.zeros((window, window, 3), np.uint8)
    records = []
    for frame in frames:
        u, v = center(clip, track_id, frame)
        c0, r0 = math.floor(u - window / 2), math.floor(v - window / 2)
        fake.set_view(frame, (c0, r0), (window, window))
        (result,) = fake.preview(image, [ObjectPrompt(track_id, [(u - c0, v - r0)], [1])])
        in_frame = replace(result, offset=(result.offset[0] + c0, result.offset[1] + r0))
        records.append(measure_mask(in_frame, frame, (c0, r0, window, window), "fine"))
    return records


def add_fine(clip, run_folder, track_id, window, head_px=None):
    """Add a fine track of the scene's object `track_id` to a run folder: its records on every
    frame of the session's clip, its entry in the session (with the head click `head_px`, (u, v)
    in px, if given) and the record of its run."""
    session = Session.load(run_folder / schema.SESSION_JSON)
    results = run_folder / schema.RESULTS_NPZ
    store = ResultsStore.load(results) if results.is_file() else ResultsStore()
    frames = range(session.clip.start, session.clip.end + 1, session.clip.step)
    for record in fine_records(clip, track_id, frames, window):
        store.put(track_id, record)
    store.save(results)
    session.tracks.append(Track(id=track_id, color="#FFFF00", mode="fine", fine_window_px=window,
                                start_frame=frames[0], head_px=None if head_px is None else list(head_px)))
    session.runs.append(RunRecord(tracks=[track_id], start_frame=frames[0], mode="fine",
                                  started="2026-10-07T03:00:00-07:00", finished="2026-10-07T03:00:30-07:00",
                                  frames_done=len(frames), seconds_per_frame=30.0 / len(frames), device="cpu"))
    session.save(run_folder / schema.SESSION_JSON)


def table(path):
    """A CSV file of the run folder as a DataFrame: track ids and flags as text ("" for no flag)."""
    frame = pd.read_csv(path, dtype={"track_id": str, "flags": str, "mode": str}, keep_default_na=True)
    if "flags" in frame:
        frame["flags"] = frame["flags"].fillna("")
    return frame


def cells(path):
    """A CSV file as text cells: (header, rows), each row a dict from column name to the cell's text."""
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[-1] == ""  # the file ends with a line end
    header = lines[0].split(",")
    return header, [dict(zip(header, line.split(","), strict=True)) for line in lines[1:-1]]


def true_pose(clip, track_id, frame):
    """(u, v, heading) of a scene's object: its center in px and its head direction in rad,
    counterclockwise on screen from the image's rightward direction."""
    (obj,) = [obj for obj in clip.scene.objects if obj.track_id == track_id]
    return obj.path.pose(frame)


def head_point(clip, track_id, frame, ahead_px):
    """The point `ahead_px` px in front of an object's center along its heading: (u, v) in px."""
    u, v, heading = true_pose(clip, track_id, frame)
    return u + ahead_px * math.cos(heading), v - ahead_px * math.sin(heading)


def ray_ellipse(origin, angle, ellipse_center, a, b, heading):
    """How far a ray travels before it leaves an ellipse: the distance from `origin` (x, y) along
    the direction at `angle` (rad, counterclockwise from +x) to the far crossing with the ellipse
    of semi-axes `a` (along `heading`, rad) and `b` around `ellipse_center`. All in one plane with
    y up and one unit. The origin must be inside the ellipse."""
    px, py = origin[0] - ellipse_center[0], origin[1] - ellipse_center[1]
    cos, sin = math.cos(heading), math.sin(heading)
    ox, oy = px * cos + py * sin, -px * sin + py * cos
    dx, dy = math.cos(angle - heading), math.sin(angle - heading)
    qa = dx * dx / a ** 2 + dy * dy / b ** 2
    qb = 2.0 * (ox * dx / a ** 2 + oy * dy / b ** 2)
    qc = ox * ox / a ** 2 + oy * oy / b ** 2 - 1.0
    return (-qb + math.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)


def ellipse_gap(clip, id_a, id_b, frame, samples=720):
    """The least distance in px between the true outlines of two ellipse objects of a scene on one
    frame, from `samples` points on each outline (so it is at most about 0.03 px too large)."""
    outlines = []
    for track_id in (id_a, id_b):
        (obj,) = [obj for obj in clip.scene.objects if obj.track_id == track_id]
        u, v, heading = obj.path.pose(frame)
        s = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        xi, eta = obj.shape.a_px * np.cos(s), obj.shape.b_px * np.sin(s)
        outlines.append(np.column_stack([u + xi * np.cos(heading) - eta * np.sin(heading),
                                         v - (xi * np.sin(heading) + eta * np.cos(heading))]))
    apart = outlines[0][:, None, :] - outlines[1][None, :, :]
    return float(np.hypot(apart[..., 0], apart[..., 1]).min())
