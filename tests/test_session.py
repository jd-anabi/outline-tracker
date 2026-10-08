"""Tests of outline_tracker/session.py: session.json <-> dataclasses, and finding the video again.

Expected values come from SPEC.md itself (the session.json example of section 8.10 is read from the
file and must come back unchanged), from decision X18 (the optional keys), from SPEC 3.2 and 4.2
worked by hand (the world frame), and from hashlib (the video's identity). Nothing is copied from
the output of the code under test.

Coordinates: (u, v) are image pixels in Tracker's convention (origin at the top-left corner of the
frame, u to the right, v down, pixel centers at +0.5); (x, y) are mm in the user's axes, y up.
"""

import copy
import hashlib
import json
import math
import ntpath
import os
import posixpath
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from helpers import _names
from results_helpers import names_number

import outline_tracker
from outline_tracker import session as session_module
from outline_tracker.fileio import relative_path
from outline_tracker.geometry import WorldFrame
from outline_tracker.session import (
    Axes,
    CalibrationSettings,
    Circle,
    Clip,
    Correction,
    ProbeBox,
    Processing,
    Prompt,
    RunRecord,
    Session,
    SessionVersionError,
    TimeSettings,
    Track,
    VideoNotFoundError,
    VideoRef,
    WrongVideoError,
)

REPO = Path(__file__).resolve().parents[1]
VIDEO_NAME = "groupB_2026-10-06_1325_main_tracker.mp4"


def spec_example() -> dict:
    """The session.json example of SPEC 8.10, parsed from SPEC.md."""
    text = (REPO / "SPEC.md").read_text(encoding="utf-8")
    section = text.split("### 8.10 session.json", 1)[1]
    return json.loads(section.split("```json", 1)[1].split("```", 1)[0])


def tracker_fit_example() -> dict:
    """A session as `from-tracker` makes it (X18): no stick and no check, the scale from the fit to
    the Tracker export, fps from the export's t column, and the facts `export` needs without torch.
    The new keys sit where the session file puts them: `tracker_fit` after `check`, `decoder` after
    `frame_hash`, `model_id` and `weights_sha256` at the end of a run entry."""
    data = spec_example()
    data["time"] = {"fps_true": 240.0096, "source": "tracker-export", "manifest_path": None, "stopwatch": None}
    data["calibration"] = {"stick": None, "click_sigma_px": 0.5, "check": None,
                           "tracker_fit": {"mm_per_px": 0.0323999942, "rms_mm": 5.0e-06, "n_points": 8}}
    data["axes"] = {"origin_px": [960.50006, 540.49999], "angle_deg": -5.83e-06}
    data["circle"] = None
    data["probes"] = []
    data["tracks"][0]["prompts"][0] = {"frame": 96, "frame_hash": "5f" * 32, "decoder": "opencv 5.0.0; darwin; arm64",
                                       "points_px": [[812.5, 377.25]], "labels": [1]}
    data["runs"][0]["model_id"] = "a-model-id"
    data["runs"][0]["weights_sha256"] = "c4" * 32
    return data


def assert_same_json(a, b, where="$"):
    """Equal as JSON, and strictly: the same keys in the same order, and 0 is not 0.0."""
    assert type(a) is type(b), f"{where}: {a!r} is not {b!r}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{where}: keys {list(a)} are not {list(b)}"
        for key in a:
            assert_same_json(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{where}: {a!r} is not {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_json(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} is not {b!r}"


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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


def test_spec_example_round_trips_without_loss(tmp_path):
    example = spec_example()
    saved = _through_a_file(tmp_path, example)
    assert saved == example  # as JSON
    assert_same_json(saved, example)  # and with the key order and the number types of the spec


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
    assert (s.time.fps_true, s.time.source, s.time.manifest_path) == (239.6, "manifest|stopwatch|typed", None)
    assert s.time.stopwatch == {"frame_a": 0, "time_a_s": 0.0, "frame_b": 0, "time_b_s": 0.0}
    assert isinstance(s.calibration, CalibrationSettings)
    assert s.calibration.stick == {"p1_px": [0, 0], "p2_px": [0, 0], "length_mm": 30.0}
    assert s.calibration.click_sigma_px == 0.5
    assert s.calibration.check == {"p1_px": [0, 0], "p2_px": [0, 0], "true_mm": 20.0}
    assert s.calibration.tracker_fit is None
    assert isinstance(s.axes, Axes) and (s.axes.origin_px, s.axes.angle_deg) == ([960.5, 540.5], 0.0)
    assert isinstance(s.circle, Circle)
    assert (s.circle.points_px, s.circle.center_px, s.circle.radius_px) == ([[0, 0]], [0, 0], 0.0)
    assert (s.circle.rms_px, s.circle.dish_mm) == (0.0, None)
    assert isinstance(s.processing, Processing)
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
    records = [s, s.video, s.clip, s.time, s.calibration, s.axes, s.circle, s.processing, s.probes[0], track,
               prompt, run, correction]
    assert [type(r).__name__ for r in records if r.extra] == []


def test_defaults_are_those_of_the_spec():
    # SPEC 8.10 (the processing block), 4.2 (click precision 0.5 px), 4.6 (LED1), 7.4, 7.6, 7.8
    p = Processing()
    assert (p.model, p.device, p.dish_crop) == ("edgetam", "auto", True)
    assert (p.outline_points, p.radial_step_deg, p.shape_ok_min) == (128, 5, 20)
    assert (p.core_open_frac, p.jump_mm_s, p.fine_window_factor, p.shape_files_for_coarse) == (0.1, 100.0, 3.0, False)
    assert Processing().to_json() == spec_example()["processing"]
    assert CalibrationSettings().click_sigma_px == 0.5
    assert Clip().step == 2
    assert Axes().angle_deg == 0.0
    assert ProbeBox().name == "LED1"
    track = Track()
    assert (track.mode, track.fine_window_px, track.head_px, track.ended_at, track.prompts) == ("coarse", None, None, None, [])
    s = Session()
    assert (s.tool_version, s.student, s.notes, s.complete) == (outline_tracker.__version__, "", "", True)
    assert (s.circle, s.probes, s.tracks, s.runs, s.corrections) == (None, [], [], [], [])
    assert s.time.fps_true is None  # not known yet
    assert (s.calibration.stick, s.calibration.check, s.calibration.tracker_fit) == (None, None, None)
    data = s.to_json()
    assert list(data) == list(spec_example())  # the top-level keys of SPEC 8.10, in its order
    assert (data["format"], data["schema_version"]) == ("outline-tracker-session", 1)


def test_default_sessions_share_nothing():
    a, b = Session(), Session()
    a.tracks.append(Track(id="A"))
    a.processing.dish_crop = False
    a.video.size = 5
    assert (b.tracks, b.processing.dish_crop, b.video.size) == ([], True, 0)
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
    assert (s.calibration.stick, s.calibration.check, s.circle) == (None, None, None)
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
    example = spec_example()
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
    example = spec_example()
    example["clip"] = {"start": 0, "note": "trimmed", "end": 2399, "step": 2}
    saved = _through_a_file(tmp_path, example)
    assert saved == example
    assert saved["clip"]["note"] == "trimmed"


# --------------------------------------------------------------------------- versions and other files

def _names_number(message: str, number: int) -> bool:
    """True when the text holds this whole number on its own, not as part of "0.1.0", "0.2.0.dev0" or "12"."""
    return re.search(rf"(?<![\w.]){number}(?!\d|\.\d)", message) is not None


@pytest.mark.parametrize("found", [0, 2])
def test_other_schema_version_raises_and_names_both_versions(tmp_path, found):
    data = spec_example()
    data["schema_version"] = found
    data["tool_version"] = "9.8.7"
    path = _write_json(tmp_path / "session.json", data)
    with pytest.raises(SessionVersionError) as err:
        Session.load(path)
    assert "session.json" in str(err.value)
    message = str(err.value).replace(str(path), "<the file>")  # a temp folder's name may hold any digit
    assert _names_number(message, found), message  # the file's version
    assert _names_number(message, 1), message  # the version this tool reads
    assert (err.value.found, err.value.supported) == (found, 1)
    assert isinstance(err.value, ValueError)  # one "bad input" family for the command line to catch
    with pytest.raises(SessionVersionError):
        Session.from_json(data)


def test_names_number_helper():
    assert _names_number("has version 2, but reads only version 1. Update.", 2)
    assert _names_number("has version 2, but reads only version 1. Update.", 1)
    assert not _names_number("this outline-tracker (0.1.0) reads version 12", 0)
    assert not _names_number("this outline-tracker (0.1.0) reads version 12", 1)
    assert not _names_number("this outline-tracker (0.1.0) reads version 12", 2)
    assert not _names_number("this outline-tracker (0.2.0.dev0) reads version 12", 0)
    # the second copy of the helper (tests/results_helpers.py) has no test of its own
    assert not names_number("this outline-tracker (0.2.0.dev0) reads version 12", 0)


@pytest.mark.parametrize("found", ["missing", "1", None, 1.5])
def test_missing_or_odd_schema_version_is_a_version_error(found):
    data = spec_example()
    if found == "missing":
        del data["schema_version"]
    else:
        data["schema_version"] = found
    with pytest.raises(SessionVersionError) as err:
        Session.from_json(data)
    assert _names_number(str(err.value), 1), str(err.value)


@pytest.mark.parametrize("content", [
    '{"format": "something-else", "schema_version": 1}',
    '{"schema_version": 1}',
    "[1, 2, 3]",
    '"outline-tracker-session"',
    '{"format": "outline-tracker-session", "schema_version": 1, ',  # cut off
    "t,frame,x,y\n0,0,1,2\n",  # a CSV given by mistake
])
def test_a_file_that_is_no_session_gives_a_plain_error(tmp_path, content):
    path = tmp_path / "positions.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as err:
        Session.load(path)
    assert not isinstance(err.value, SessionVersionError)
    assert "positions.json" in str(err.value)
    assert "session" in str(err.value)


def test_a_missing_file_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Session.load(tmp_path / "session.json")


@pytest.mark.parametrize(("key", "value", "needed"), [
    ("clip", 5, "object"), ("video", None, "object"), ("processing", "edgetam", "object"), ("tracks", [3], "object"),
    ("tracks", {"id": "A"}, "list"), ("runs", None, "list"), ("probes", "LED1", "list"),
])
def test_a_block_of_the_wrong_kind_gives_a_plain_error(key, value, needed):
    data = spec_example()
    data[key] = value
    with pytest.raises(ValueError) as err:
        Session.from_json(data)
    assert f'"{key}' in str(err.value)  # which block
    assert needed in str(err.value)  # and what it must be
    assert not isinstance(err.value, SessionVersionError)


def test_null_is_fine_where_a_block_may_be_absent():
    data = spec_example()
    data["circle"] = None
    data["calibration"]["stick"] = None
    data["time"]["stopwatch"] = None
    s = Session.from_json(data)
    assert (s.circle, s.calibration.stick, s.time.stopwatch) == (None, None, None)
    assert s.to_json() == data


def test_a_file_saved_with_a_byte_order_mark_loads(tmp_path):
    # older Windows editors put one in front of UTF-8 text; students edit session.json by hand
    example = spec_example()
    path = tmp_path / "session.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(example).encode("utf-8"))
    assert Session.load(path).to_json() == example


# --------------------------------------------------------------------------- the file on disk

def test_saved_file_is_utf8_with_lf_line_ends_and_loads_back_equal(tmp_path):
    s = Session.from_json(spec_example())
    s.student = "Zoë Müller"
    s.notes = "two lines\nof notes, with a tab\t and a \\ backslash"
    path = tmp_path / "vidéo test ü" / "session.json"  # the folder does not exist yet
    assert s.save(path) == path
    raw = path.read_bytes()
    assert b"\r" not in raw  # the same bytes on Windows and macOS
    assert raw.endswith(b"\n")
    assert len(raw.splitlines()) > 20  # one key per line: readable, and editable by hand
    assert "Zoë Müller" in raw.decode("utf-8")
    loaded = Session.load(path)
    assert loaded == s
    assert (loaded.student, loaded.notes) == (s.student, s.notes)
    assert Session.load(str(path)) == s
    assert _names(path.parent) == ["session.json"]


def test_numpy_numbers_and_arrays_are_stored_as_plain_json(tmp_path):
    s = Session()
    s.clip = Clip(start=np.int64(96), end=np.int64(2494), step=2)
    s.time.fps_true = np.float64(239.6)
    s.axes.origin_px = np.array([960.5, 540.5])
    s.tracks.append(Track(id="A", color="#FFFF00", start_frame=np.int32(96), prompts=[
        Prompt(frame=96, points_px=np.array([[812.5, 377.25]], dtype=np.float32), labels=np.array([1]))]))
    path = tmp_path / "session.json"
    s.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_same_json(data["clip"], {"start": 96, "end": 2494, "step": 2})
    assert_same_json(data["time"]["fps_true"], 239.6)
    assert_same_json(data["axes"]["origin_px"], [960.5, 540.5])
    assert_same_json(data["tracks"][0]["start_frame"], 96)
    assert_same_json(data["tracks"][0]["prompts"][0]["points_px"], [[812.5, 377.25]])
    assert_same_json(data["tracks"][0]["prompts"][0]["labels"], [1])


def test_save_keeps_the_old_file_and_returns_the_new_name_when_the_target_is_locked(tmp_path, lock_file):
    path = tmp_path / "session.json"
    first = Session(student="ana")
    first.save(path)
    old = path.read_bytes()
    lock_file(path)  # as Windows does while a sync client holds session.json
    written = Session(student="ben").save(path)
    assert written == tmp_path / "session.new.json"
    assert path.read_bytes() == old
    assert Session.load(written).student == "ben"
    assert _names(tmp_path) == ["session.json", "session.new.json"]


# --------------------------------------------------------------------------- nonsense is not saved

@pytest.mark.parametrize("fps", [0, 0.0, -239.6, math.nan, math.inf, -math.inf, "240", True])
def test_fps_true_not_positive_raises_and_writes_no_file(tmp_path, fps):
    s = Session.from_json(spec_example())
    s.time.fps_true = fps
    with pytest.raises(ValueError, match="fps_true"):
        s.save(tmp_path / "run" / "session.json")
    assert _names(tmp_path) == []  # no file, no folder, no temporary file


def test_fps_true_not_positive_leaves_an_existing_file_alone(tmp_path):
    path = tmp_path / "session.json"
    s = Session.from_json(spec_example())
    s.save(path)
    before = path.read_bytes()
    s.time.fps_true = 0.0
    with pytest.raises(ValueError, match="fps_true"):
        s.save(path)
    assert path.read_bytes() == before
    assert _names(tmp_path) == ["session.json"]


@pytest.mark.parametrize("fps", [0, -1.0])
def test_fps_true_not_positive_in_a_file_is_refused_on_load(tmp_path, fps):
    data = spec_example()
    data["time"]["fps_true"] = fps
    with pytest.raises(ValueError, match="fps_true"):
        Session.load(_write_json(tmp_path / "session.json", data))


def test_unknown_fps_true_can_be_saved(tmp_path):
    s = Session()  # a new session: the user has not given fps_true yet
    path = tmp_path / "session.json"
    s.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["time"]["fps_true"] is None
    assert Session.load(path).time.fps_true is None


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_number_that_json_cannot_hold_raises_and_writes_no_file(tmp_path, bad):
    s = Session.from_json(spec_example())
    s.axes.origin_px = [bad, 540.5]
    with pytest.raises(ValueError):
        s.save(tmp_path / "session.json")
    assert _names(tmp_path) == []


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


# --------------------------------------------------------------------------- relative paths

def test_relative_path_from_the_run_folder():
    run = "/data/videos/clip_outline_ana"
    assert relative_path("/data/videos/clip.mp4", run, posixpath) == "../clip.mp4"
    assert relative_path("/data/videos/clip_outline_ana/clip.mp4", run, posixpath) == "clip.mp4"
    assert relative_path("/media/day 1/clip.mp4", run, posixpath) == "../../../media/day 1/clip.mp4"


def test_relative_path_on_windows_is_stored_with_forward_slashes():
    run = "C:\\data\\videos\\clip_outline_ana"
    assert relative_path("C:\\data\\videos\\clip.mp4", run, ntpath) == "../clip.mp4"
    assert relative_path("C:\\data\\videos\\day 1\\clip.mp4", "C:\\data\\runs\\ana", ntpath) == "../../videos/day 1/clip.mp4"
    assert relative_path("c:\\DATA\\videos\\clip.mp4", run, ntpath) == "../clip.mp4"  # one drive, whatever the case


def test_a_video_on_another_windows_drive_has_no_relative_path():
    run = "C:\\outline\\runs\\clip_outline_ana"
    assert relative_path("G:\\My Drive\\videos\\clip.mp4", run, ntpath) is None
    assert relative_path("\\\\server\\share\\videos\\clip.mp4", run, ntpath) is None  # a network share


def test_a_video_without_a_relative_path_stores_null_and_is_found_by_its_absolute_path(tmp_path, monkeypatch):
    video = tmp_path / "videos" / VIDEO_NAME
    _video(video)
    run = tmp_path / "runs" / "clip_outline_ana"
    monkeypatch.setattr(session_module, "relative_path", lambda *args, **kwargs: None)  # as on another drive
    s = Session(video=VideoRef.from_file(video, run, width=1920, height=1080, n_frames=2400, fps_container=240.0))
    assert s.video.relpath is None
    s.save(run / "session.json")
    stored = json.loads((run / "session.json").read_text(encoding="utf-8"))["video"]
    assert stored["relpath"] is None  # null in the file
    assert stored["abspath"] == str(video)
    assert Session.load(run / "session.json").locate_video(run) == video


# --------------------------------------------------------------------------- finding the video again

def _video(path: Path, seed: int = 0, size: int = 5000) -> bytes:
    """A stand-in for a video file: identity is its size and its first bytes, never its content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).bytes(size)
    path.write_bytes(data)
    return data


def _saved_run(root: Path) -> tuple[Path, Path, bytes]:
    """`root/videos/<video>` and the default run folder next to it (SPEC 8.1), with a saved session."""
    video = root / "videos" / VIDEO_NAME
    data = _video(video)
    run = video.parent / (video.stem + "_outline_ana")
    s = Session(student="ana", clip=Clip(start=0, end=2399, step=2), time=TimeSettings(fps_true=239.6, source="typed"),
                video=VideoRef.from_file(video, run, width=1920, height=1080, n_frames=2400, fps_container=240.0))
    assert s.save(run / "session.json") == run / "session.json"
    return video, run, data


def test_video_is_stored_relative_to_the_run_folder_with_its_identity(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    v = Session.load(run / "session.json").video
    assert v.relpath == "../" + VIDEO_NAME  # relative to the run folder, forward slash
    assert v.abspath == str(video)
    assert v.size == 5000
    assert v.sha256_first_64mib == hashlib.sha256(data).hexdigest()
    assert (v.width, v.height, v.n_frames, v.fps_container) == (1920, 1080, 2400, 240.0)
    stored = json.loads((run / "session.json").read_text(encoding="utf-8"))["video"]
    assert stored["sha256_first_64MiB"] == hashlib.sha256(data).hexdigest()  # the key as SPEC 8.10 spells it
    assert Session.load(run / "session.json").locate_video(run) == video


def test_video_path_given_relative_to_the_current_folder(tmp_path, monkeypatch):
    video, run, _ = _saved_run(tmp_path / "a")
    monkeypatch.chdir(video.parent)
    ref = VideoRef.from_file(VIDEO_NAME, run.name, width=1920, height=1080, n_frames=2400, fps_container=240.0)
    assert ref.relpath == "../" + VIDEO_NAME
    assert Path(ref.abspath) == Path.cwd() / VIDEO_NAME
    found = Session.load(Path(run.name) / "session.json").locate_video(run.name)
    assert found.is_absolute() and found == Path.cwd() / VIDEO_NAME


def test_the_whole_folder_can_be_moved(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    shutil.move(tmp_path / "a", tmp_path / "moved here")  # video and run folder together, e.g. to another disk
    new_run = tmp_path / "moved here" / "videos" / run.name
    s = Session.load(new_run / "session.json")
    assert not Path(s.video.abspath).exists()  # the absolute path is stale
    assert s.locate_video(new_run) == tmp_path / "moved here" / "videos" / VIDEO_NAME


def test_a_run_folder_moved_alone_finds_the_video_by_its_absolute_path(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_run = tmp_path / "elsewhere" / "deep" / run.name
    new_run.parent.mkdir(parents=True)
    shutil.move(run, new_run)
    s = Session.load(new_run / "session.json")
    assert not (new_run / s.video.relpath).exists()  # the relative path leads nowhere now
    assert s.locate_video(new_run) == video


def test_a_moved_video_is_reported_missing_then_accepted_where_the_user_points(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_video = tmp_path / "drive" / "renamed.mp4"
    new_video.parent.mkdir()
    shutil.move(video, new_video)
    s = Session.load(run / "session.json")
    with pytest.raises(VideoNotFoundError) as err:
        s.locate_video(run)
    assert isinstance(err.value, FileNotFoundError)
    assert VIDEO_NAME in str(err.value)  # which video to look for
    assert str(video) in str(err.value)  # and where it was looked for

    s.video.check(new_video)  # the file the user picked is the session's video: no error
    s.video.point_to(new_video, run)
    assert s.video.relpath == "../../../drive/renamed.mp4"
    assert s.video.abspath == str(new_video)
    s.save(run / "session.json")
    assert Session.load(run / "session.json").locate_video(run) == new_video


def test_a_different_file_in_the_videos_place_is_refused(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    s = Session.load(run / "session.json")

    _video(video, seed=1)  # the same size, other content (say, converted again)
    with pytest.raises(WrongVideoError) as err:
        s.locate_video(run)
    assert isinstance(err.value, ValueError)
    assert VIDEO_NAME in str(err.value)
    assert "content" in str(err.value)

    video.write_bytes(data[:4000])  # another size (say, a download that stopped)
    with pytest.raises(WrongVideoError) as err:
        s.locate_video(run)
    assert VIDEO_NAME in str(err.value)
    sizes = re.sub(r"[,. ]", "", str(err.value).replace(str(tmp_path), ""))
    assert "4000" in sizes and "5000" in sizes  # both sizes, in bytes, however the digits are grouped

    video.write_bytes(data)  # the right file again
    assert s.locate_video(run) == video


def test_check_refuses_another_file_and_a_missing_one(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    s = Session.load(run / "session.json")
    other = tmp_path / "other.mp4"
    _video(other, seed=2)
    with pytest.raises(WrongVideoError, match="other.mp4"):
        s.video.check(other)
    with pytest.raises(VideoNotFoundError, match="nothing.mp4"):
        s.video.check(tmp_path / "nothing.mp4")
    copy_of_it = tmp_path / "copy.mp4"
    copy_of_it.write_bytes(data)
    s.video.check(copy_of_it)  # a copy of the same file is the same video
    s.video.check(str(copy_of_it))


def test_a_wrong_file_at_the_relative_path_does_not_hide_the_right_one(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_run = tmp_path / "b" / run.name
    new_run.parent.mkdir()
    shutil.move(run, new_run)
    _video(new_run.parent / VIDEO_NAME, seed=5)  # another video of the same name next to the moved run folder
    assert Session.load(new_run / "session.json").locate_video(new_run) == video


def test_a_session_without_a_video_is_reported_missing(tmp_path):
    with pytest.raises(VideoNotFoundError):
        Session().locate_video(tmp_path)


def test_a_path_from_another_kind_of_computer_is_reported_missing_by_name(tmp_path):
    # a run folder made on Windows and opened on a Mac, or the other way round, without its video
    foreign = "/Volumes/lab/videos/clip.mp4" if os.name == "nt" else "G:\\My Drive\\videos\\clip.mp4"
    s = Session(video=VideoRef(relpath=None, abspath=foreign, size=5000, sha256_first_64mib="ab" * 32))
    with pytest.raises(VideoNotFoundError) as err:
        s.locate_video(tmp_path)
    assert "The video clip.mp4 " in str(err.value)  # the file's name, whichever slashes the path has
    assert "videos" in str(err.value)  # and where it was


def test_format_and_version_are_constants_of_the_file(tmp_path):
    s = Session()
    s.extra.update({"format": "something-else", "schema_version": 7, "kept": 1})
    data = s.to_json()
    assert (data["format"], data["schema_version"], data["kept"]) == ("outline-tracker-session", 1, 1)
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
