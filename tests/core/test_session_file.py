"""Tests of outline_tracker/core/session.py: the file on disk, its version, odd files, what is not saved.

Expected values are the session.json example of SPEC 8.10 (read from docs/design/SPEC-v0.1.md), the examples
of docs/design/formats.md, and values typed in each test. Nothing is copied from the output of the code under
test.

The rest of the tests of the session are in tests/core/test_session.py (session.json <-> dataclasses,
the optional and unknown keys, the world frame) and tests/core/test_session_video.py (the video's path
and finding the video again). What the three share is in tests/support/session_json_helpers.py.
"""

import json
import math

import numpy as np
import pytest
from helpers import names
from results_helpers import names_number
from session_json_helpers import _write_json, assert_same_json, formats_example, spec_example

from outline_tracker.core.session import (
    Clip,
    Prompt,
    Session,
    SessionVersionError,
    Track,
)


# --------------------------------------------------------------------------- versions and other files

@pytest.mark.parametrize("found", [0, 3])
def test_other_schema_version_raises_and_names_it_and_the_versions_read(tmp_path, found):
    data = spec_example()
    data["schema_version"] = found
    data["tool_version"] = "9.8.7"
    path = _write_json(tmp_path / "session.json", data)
    with pytest.raises(SessionVersionError) as err:
        Session.load(path)
    assert "session.json" in str(err.value)
    message = str(err.value).replace(str(path), "<the file>")  # a temp folder's name may hold any digit
    assert names_number(message, found), message  # the file's version
    assert names_number(message, 1) and names_number(message, 2), message  # the versions this tool reads
    assert (err.value.found, err.value.supported) == (found, (1, 2))
    assert ("newer" if found == 3 else "older") in message
    assert isinstance(err.value, ValueError)  # one "bad input" family for the command line to catch
    with pytest.raises(SessionVersionError):
        Session.from_json(data)


def test_names_number_helper():
    assert names_number("has version 2, but reads only version 1. Update.", 2)
    assert names_number("has version 2, but reads only version 1. Update.", 1)
    assert not names_number("this outline-tracker (0.1.0) reads version 12", 0)
    assert not names_number("this outline-tracker (0.1.0) reads version 12", 1)
    assert not names_number("this outline-tracker (0.1.0) reads version 12", 2)
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
    assert names_number(str(err.value), 1), str(err.value)


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
    data = formats_example(1, 2)
    data["boundary"] = None
    data["calibration"]["stick"] = None
    data["time"]["stopwatch"] = None
    s = Session.from_json(data)
    assert (s.circle, s.calibration.stick, s.time.stopwatch) == (None, None, None)
    assert s.to_json() == data


@pytest.mark.parametrize("version", [1, 2])
def test_a_file_saved_with_a_byte_order_mark_loads(tmp_path, version):
    # older Windows editors put one in front of UTF-8 text; students edit session.json by hand
    path = tmp_path / "session.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(formats_example(1, version)).encode("utf-8"))
    assert Session.load(path).to_json() == formats_example(1, 2)


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
    assert names(path.parent) == ["session.json"]


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
    assert names(tmp_path) == ["session.json", "session.new.json"]


# --------------------------------------------------------------------------- nonsense is not saved

@pytest.mark.parametrize("fps", [0, 0.0, -239.6, math.nan, math.inf, -math.inf, "240", True])
def test_fps_true_not_positive_raises_and_writes_no_file(tmp_path, fps):
    s = Session.from_json(spec_example())
    s.time.fps_true = fps
    with pytest.raises(ValueError, match="fps_true"):
        s.save(tmp_path / "run" / "session.json")
    assert names(tmp_path) == []  # no file, no folder, no temporary file


def test_fps_true_not_positive_leaves_an_existing_file_alone(tmp_path):
    path = tmp_path / "session.json"
    s = Session.from_json(spec_example())
    s.save(path)
    before = path.read_bytes()
    s.time.fps_true = 0.0
    with pytest.raises(ValueError, match="fps_true"):
        s.save(path)
    assert path.read_bytes() == before
    assert names(tmp_path) == ["session.json"]


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
    assert names(tmp_path) == []
