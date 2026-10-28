"""Tests of outline_tracker/core/session.py: session.json <-> dataclasses, optional and unknown keys, world frame.

Expected values come from docs/design/SPEC-v0.1.md itself (the session.json example of section 8.10, a file of
version 1, is read from the file and must become the records), from docs/design/formats.md (its examples of
version 2 must come back unchanged), from decision X18 (the optional keys), and from SPEC 3.2 and 4.2 worked by
hand (the world frame). Nothing is copied from the output of the code under test.

Coordinates: (u, v) are image pixels in Tracker's convention (origin at the top-left corner of the
frame, u to the right, v down, pixel centers at +0.5); (x, y) are mm in the user's axes, y up.

The rest of the tests of the session are beside this file: the file on disk, odd and refused files and
the version in tests/core/test_session_file.py, the video's path and finding the video again in
tests/core/test_session_video.py. What the three share is in tests/support/session_json_helpers.py.
"""

import copy
import json
import math
from pathlib import Path

import pytest
from session_json_helpers import VIDEO_NAME, _write_json, assert_same_json, formats_example, spec_example

import outline_tracker
from outline_tracker.core.geometry import WorldFrame
from outline_tracker.core.session import (
    Axes,
    Boundary,
    CalibrationSettings,
    Clip,
    Correction,
    ProbeBox,
    Processing,
    Prompt,
    RunRecord,
    Session,
    TimeSettings,
    Track,
    VideoRef,
)


def tracker_fit_example() -> dict:
    """A session as `from-tracker` makes it (X18), as a file of version 2: no stick and no check, the scale
    from the fit to the Tracker export, fps from the export's t column, and the facts `export` needs without
    torch. The new keys sit where the session file puts them: `tracker_fit` after `check`, `decoder` after
    `frame_hash`, `model_id` and `weights_sha256` at the end of a run entry."""
    data = formats_example(1, 2)
    data["time"] = {"fps_true": 240.0096, "source": "tracker-export", "stopwatch": None}
    data["calibration"] = {"stick": None, "click_sigma_px": 0.5, "check": None,
                           "tracker_fit": {"mm_per_px": 0.0323999942, "rms_mm": 5.0e-06, "n_points": 8}}
    data["axes"] = {"origin_px": [960.50006, 540.49999], "angle_deg": -5.83e-06}
    data["boundary"] = None
    data["probes"] = []
    data["tracks"][0]["prompts"][0] = {"frame": 96, "frame_hash": "5f" * 32, "decoder": "opencv 5.0.0; darwin; arm64",
                                       "points_px": [[812.5, 377.25]], "labels": [1]}
    data["runs"][0]["model_id"] = "a-model-id"
    data["runs"][0]["weights_sha256"] = "c4" * 32
    return data


def _through_a_file(tmp_path: Path, data: dict) -> dict:
    """Load `data` from a file, save the session to another file, and return that file's JSON."""
    loaded = Session.load(_write_json(tmp_path / "in" / "session.json", data))
    out = tmp_path / "out" / "session.json"
    assert loaded.save(out) == out
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the SPEC 8.10 example

def test_the_example_is_the_one_in_the_spec():
    example = spec_example()
    # typed from SPEC 8.10: if the extraction above went wrong, this fails
    assert list(example) == ["format", "schema_version", "tool_version", "student", "notes", "complete", "video",
                             "clip", "time", "calibration", "axes", "circle", "processing", "probes", "tracks",
                             "runs", "corrections"]
    assert example["format"] == "outline-tracker-session"
    assert example["schema_version"] == 1
    assert example["video"]["relpath"] == "../" + VIDEO_NAME
    assert example["corrections"][0]["action"] == "retrack|end|new_piece"


@pytest.mark.parametrize("number", [1, 2])
def test_the_version_2_examples_round_trip_without_loss(tmp_path, number):
    example = formats_example(number, 2)
    saved = _through_a_file(tmp_path, example)
    assert saved == example  # as JSON
    assert_same_json(saved, example)  # and with the key order and the number types of the page


def test_spec_example_becomes_the_dataclasses():
    example = spec_example()
    s = Session.from_json(example)
    assert isinstance(s, Session)
    assert (s.tool_version, s.student, s.notes, s.complete) == ("0.1.0", "", "", True)
    assert isinstance(s.video, VideoRef)
    assert s.video.relpath == "../" + VIDEO_NAME
    assert (s.video.abspath, s.video.size, s.video.sha256_first_64mib) == ("...", 0, "...")
    assert (s.video.width, s.video.height, s.video.n_frames, s.video.fps_container) == (1920, 1080, 0, 240.0)
    assert isinstance(s.clip, Clip) and (s.clip.start, s.clip.end, s.clip.step) == (0, 2399, 2)
    assert isinstance(s.time, TimeSettings)
    assert (s.time.fps_true, s.time.source) == (239.6, "manifest|stopwatch|typed")  # the text is not "manifest"
    assert s.time.stopwatch == {"frame_a": 0, "time_a_s": 0.0, "frame_b": 0, "time_b_s": 0.0}
    assert isinstance(s.calibration, CalibrationSettings)
    assert s.calibration.stick == {"p1_px": [0, 0], "p2_px": [0, 0], "length_mm": 30.0}
    assert s.calibration.click_sigma_px == 0.5
    assert s.calibration.check == {"p1_px": [0, 0], "p2_px": [0, 0], "true_mm": 20.0}
    assert s.calibration.tracker_fit is None
    assert isinstance(s.axes, Axes) and (s.axes.origin_px, s.axes.angle_deg) == ([960.5, 540.5], 0.0)
    assert isinstance(s.boundary, Boundary)
    assert (s.boundary.points_px, s.boundary.center_px, s.boundary.radius_px) == ([[0, 0]], [0, 0], 0.0)
    assert (s.boundary.shape, s.boundary.rms_px, s.boundary.diameter_mm) == ("circle", 0.0, None)
    assert isinstance(s.processing, Processing)
    assert (s.processing.boundary_crop, s.processing.jump_mm_s, s.processing.tracker_format) == (True, 100.0, True)
    assert len(s.probes) == 1 and isinstance(s.probes[0], ProbeBox)
    assert (s.probes[0].name, s.probes[0].rect_px) == ("LED1", [0, 0, 0, 0])
    (track,) = s.tracks
    assert isinstance(track, Track)
    assert (track.id, track.color, track.mode, track.fine_window_px) == ("A", "#00FFFF", "coarse", None)
    assert (track.start_frame, track.head_px, track.ended_at) == (0, None, None)
    (prompt,) = track.prompts
    assert isinstance(prompt, Prompt)
    assert (prompt.frame, prompt.frame_hash, prompt.decoder) == (0, "...", None)
    assert (prompt.points_px, prompt.labels) == ([[0, 0]], [1])
    (run,) = s.runs
    assert isinstance(run, RunRecord)
    assert (run.tracks, run.start_frame, run.mode, run.started, run.finished) == (["A"], 0, "coarse", "ISO-8601", "ISO-8601")
    assert (run.frames_done, run.seconds_per_frame, run.device) == (0, 0.0, "cpu")
    assert (run.model_id, run.weights_sha256) == (None, None)
    (correction,) = s.corrections
    assert isinstance(correction, Correction)
    assert (correction.time, correction.tracks, correction.action) == ("ISO-8601", ["A"], "retrack|end|new_piece")
    assert (correction.frame, correction.prompts) == (0, [])
    # every key of the example is a field: nothing was only carried along as an unknown key
    records = [s, s.video, s.clip, s.time, s.calibration, s.axes, s.boundary, s.processing, s.probes[0], track,
               prompt, run, correction]
    assert [type(r).__name__ for r in records if r.extra] == []


def test_defaults_are_those_of_the_spec_and_of_version_2():
    # SPEC 8.10 (the processing block), 4.2 (click precision 0.5 px), 4.6 (LED1), 7.4, 7.6, 7.8; formats.md
    p = Processing()
    assert (p.model, p.device, p.boundary_crop) == ("edgetam", "auto", True)
    assert (p.outline_points, p.radial_step_deg, p.shape_ok_min) == (128, 5, 20)
    assert (p.core_open_frac, p.jump_mm_s, p.fine_window_factor, p.shape_files_for_coarse) == (0.1, 100.0, 3.0, False)
    assert p.tracker_format is False  # a new session (decision 34)
    spec = {("boundary_crop" if key == "dish_crop" else key): value for key, value in spec_example()["processing"].items()}
    assert_same_json(Processing().to_json(), {**spec, "tracker_format": False})
    assert Boundary().shape == "circle"
    assert CalibrationSettings().click_sigma_px == 0.5
    assert Clip().step == 2
    assert Axes().angle_deg == 0.0
    assert ProbeBox().name == "LED1"
    track = Track()
    assert (track.mode, track.fine_window_px, track.head_px, track.ended_at, track.prompts) == ("coarse", None, None, None, [])
    s = Session()
    assert (s.tool_version, s.student, s.notes, s.complete) == (outline_tracker.__version__, "", "", True)
    assert (s.boundary, s.probes, s.tracks, s.runs, s.corrections) == (None, [], [], [], [])
    assert s.time.fps_true is None  # not known yet
    assert (s.calibration.stick, s.calibration.check, s.calibration.tracker_fit) == (None, None, None)
    data = s.to_json()
    assert list(data) == list(formats_example(1, 2))  # the top-level keys of version 2, in their order
    assert (data["format"], data["schema_version"]) == ("outline-tracker-session", 2)


def test_default_sessions_share_nothing():
    a, b = Session(), Session()
    a.tracks.append(Track(id="A"))
    a.processing.boundary_crop = False
    a.video.size = 5
    assert (b.tracks, b.processing.boundary_crop, b.video.size) == ([], True, 0)
    assert Track().prompts is not Track().prompts
    assert a != b and Session() == Session()


def test_to_json_and_from_json_do_not_share_lists_with_the_session():
    data = spec_example()
    s = Session.from_json(data)
    data["axes"]["origin_px"][0] = -1.0
    assert s.axes.origin_px == [960.5, 540.5]
    out = s.to_json()
    out["tracks"][0]["prompts"][0]["points_px"][0][0] = -1.0
    out["calibration"]["stick"]["length_mm"] = -1.0
    assert s.tracks[0].prompts[0].points_px == [[0, 0]]
    assert s.calibration.stick["length_mm"] == 30.0


# --------------------------------------------------------------------------- the optional keys (X18)

def test_tracker_fit_session_round_trips(tmp_path):
    example = tracker_fit_example()
    saved = _through_a_file(tmp_path, example)
    assert saved == example
    assert_same_json(saved, example)
    s = Session.from_json(example)
    assert s.calibration.tracker_fit == {"mm_per_px": 0.0323999942, "rms_mm": 5.0e-06, "n_points": 8}
    assert (s.calibration.stick, s.calibration.check, s.boundary) == (None, None, None)
    assert s.time.source == "tracker-export"
    assert s.tracks[0].prompts[0].decoder == "opencv 5.0.0; darwin; arm64"
    assert (s.runs[0].model_id, s.runs[0].weights_sha256) == ("a-model-id", "c4" * 32)
    assert [type(r).__name__ for r in (s, s.calibration, s.time, s.tracks[0].prompts[0], s.runs[0]) if r.extra] == []


def test_optional_keys_are_written_only_when_set():
    s = Session(tracks=[Track(id="A", prompts=[Prompt(frame=0, frame_hash="ab")])], runs=[RunRecord(tracks=["A"])])
    data = s.to_json()
    assert "tracker_fit" not in data["calibration"]
    assert list(data["calibration"]) == ["stick", "click_sigma_px", "check"]
    assert "decoder" not in data["tracks"][0]["prompts"][0]
    assert "model_id" not in data["runs"][0] and "weights_sha256" not in data["runs"][0]

    s.calibration.tracker_fit = {"mm_per_px": 0.05, "rms_mm": 0.0, "n_points": 3}
    s.tracks[0].prompts[0].decoder = "tag"
    s.runs[0].model_id, s.runs[0].weights_sha256 = "m", "w"
    data = s.to_json()
    assert data["calibration"]["tracker_fit"] == {"mm_per_px": 0.05, "rms_mm": 0.0, "n_points": 3}
    prompt_keys = list(data["tracks"][0]["prompts"][0])
    assert prompt_keys[prompt_keys.index("frame_hash") + 1] == "decoder"  # next to the frame hash
    assert (data["runs"][0]["model_id"], data["runs"][0]["weights_sha256"]) == ("m", "w")


# --------------------------------------------------------------------------- unknown keys

def test_unknown_keys_survive_load_and_save(tmp_path):
    example = formats_example(1, 2)
    example["later_addition"] = {"a": [1, 2.5, None], "b": "text"}
    example["extra"] = "a key that happens to be called extra"
    example["video"]["duration_s"] = 10.0
    example["calibration"]["stick"]["note"] = "the 30 mm side"
    example["calibration"]["future_fit"] = [1, 2]
    example["processing"]["new_switch"] = True
    example["probes"][0]["color"] = "#FF0000"
    example["tracks"][0]["label"] = "the big one"
    example["tracks"][0]["prompts"][0]["box_px"] = [0, 0, 5, 5]
    example["runs"][0]["host"] = "laptop"
    example["corrections"][0]["why"] = "swapped"
    saved = _through_a_file(tmp_path, example)
    assert saved == example
    assert_same_json(saved, example)  # unknown keys were put last here, so the order is kept too

    s = Session.from_json(example)
    assert s.extra == {"later_addition": {"a": [1, 2.5, None], "b": "text"}, "extra": "a key that happens to be called extra"}
    assert s.video.extra == {"duration_s": 10.0}
    assert s.tracks[0].prompts[0].extra == {"box_px": [0, 0, 5, 5]}
    assert s.runs[0].extra == {"host": "laptop"}


def test_unknown_keys_between_known_ones_are_kept(tmp_path):
    example = formats_example(1, 2)
    example["clip"] = {"start": 0, "note": "trimmed", "end": 2399, "step": 2}
    saved = _through_a_file(tmp_path, example)
    assert saved == example
    assert saved["clip"]["note"] == "trimmed"


# --------------------------------------------------------------------------- the world frame

def _calibrated(angle_deg=0.0) -> Session:
    s = Session()
    # a 30 mm stick that spans 926 px (SPEC 4.2): k = 30 / 926 mm per px
    s.calibration.stick = {"p1_px": [100.5, 300.5], "p2_px": [1026.5, 300.5], "length_mm": 30.0}
    s.axes = Axes(origin_px=[960.5, 540.5], angle_deg=angle_deg)
    return s


def test_world_frame_from_the_stick_and_the_axes():
    wf = _calibrated().world_frame()
    assert isinstance(wf, WorldFrame)
    assert wf.k_mm_per_px == pytest.approx(30.0 / 926.0, rel=1e-12)
    assert (wf.u0, wf.v0, wf.alpha_rad) == (960.5, 540.5, 0.0)
    assert wf.to_world(960.5, 540.5) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert wf.to_world(960.5 + 926.0, 540.5) == pytest.approx((30.0, 0.0), abs=1e-9)  # one stick to the right
    assert wf.to_world(960.5, 540.5 - 463.0) == pytest.approx((0.0, 15.0), abs=1e-9)  # half a stick up: y is up


def test_world_frame_turns_with_the_axis_angle():
    # 90 degrees: +x points up on screen, so +y points left
    wf = _calibrated(angle_deg=90.0).world_frame()
    assert wf.alpha_rad == pytest.approx(math.pi / 2, rel=1e-15)
    assert wf.to_world(960.5, 540.5 - 463.0) == pytest.approx((15.0, 0.0), abs=1e-9)
    assert wf.to_world(960.5 + 926.0, 540.5) == pytest.approx((0.0, -30.0), abs=1e-9)
    assert _calibrated(angle_deg=-30.0).world_frame().alpha_rad == pytest.approx(-math.pi / 6, rel=1e-15)


def test_world_frame_from_the_tracker_fit_when_there_is_no_stick():
    s = Session()
    s.calibration.tracker_fit = {"mm_per_px": 0.05, "rms_mm": 1e-6, "n_points": 8}
    s.axes = Axes(origin_px=[100.0, 80.0], angle_deg=0.0)
    wf = s.world_frame()
    assert wf.k_mm_per_px == 0.05
    # 10 px to the right of the origin and 20 px above it on screen
    assert wf.to_world(110.0, 60.0) == pytest.approx((0.5, 1.0), abs=1e-12)


def test_the_stick_wins_over_the_tracker_fit():
    s = _calibrated()
    s.calibration.tracker_fit = {"mm_per_px": 0.05, "rms_mm": 0.0, "n_points": 8}
    assert s.world_frame().k_mm_per_px == pytest.approx(30.0 / 926.0, rel=1e-12)


def test_world_frame_of_the_tracker_fit_example_file():
    wf = Session.from_json(tracker_fit_example()).world_frame()
    assert wf.k_mm_per_px == 0.0323999942
    assert (wf.u0, wf.v0) == (960.50006, 540.49999)
    assert wf.alpha_rad == pytest.approx(math.radians(-5.83e-06), rel=1e-12)


def test_world_frame_without_a_scale_is_a_plain_error():
    with pytest.raises(ValueError, match="scale"):
        Session().world_frame()


@pytest.mark.parametrize("stick", [
    {"p1_px": [100.5, 300.5], "p2_px": [100.5, 300.5], "length_mm": 30.0},  # the same point twice
    {"p1_px": [100.5, 300.5], "p2_px": [1026.5, 300.5], "length_mm": 0.0},  # no length
    {"p1_px": [100.5, 300.5], "p2_px": [1026.5, 300.5], "length_mm": -30.0},
    {"p1_px": [100.5, 300.5], "length_mm": 30.0},  # an end is missing
    {"p1_px": [100.5, 300.5], "p2_px": None, "length_mm": 30.0},
    {"p1_px": [100.5, 300.5], "p2_px": [1026.5, 300.5], "length_mm": None},
])
def test_world_frame_with_a_nonsense_stick_is_a_plain_error(stick):
    s = _calibrated()
    s.calibration.stick = stick
    with pytest.raises(ValueError, match="stick"):
        s.world_frame()


# --------------------------------------------------------------------------- the file's constants, and comparing sessions

def test_format_and_version_are_constants_of_the_file(tmp_path):
    s = Session()
    s.extra.update({"format": "something-else", "schema_version": 7, "kept": 1})
    data = s.to_json()
    assert (data["format"], data["schema_version"], data["kept"]) == ("outline-tracker-session", 2, 1)
    assert list(data)[:2] == ["format", "schema_version"]
    s.save(tmp_path / "session.json")
    assert Session.load(tmp_path / "session.json").extra == {"kept": 1}


def test_sessions_compare_by_value():
    a, b = Session.from_json(spec_example()), Session.from_json(spec_example())
    assert a == b
    b.tracks[0].prompts[0].points_px = [[1, 0]]
    assert a != b
    c = copy.deepcopy(a)
    c.extra["later"] = 1
    assert a != c
