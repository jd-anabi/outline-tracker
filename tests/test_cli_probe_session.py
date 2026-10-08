"""`outline-tracker probe SESSION.json`, and the entry every `probe` appends to run.log (SPEC 4.6, 8.9, 11).

A session gives the rectangles (`probes`), the frames (`clip.start` to `clip.end`, every frame whatever the
clip's step), fps_true and the video; probes.csv goes into the session's run folder, the folder that holds
session.json. Expected values: see tests/cli_probe_helpers.py. The video form is in tests/test_cli_probe.py.
"""

import re
import shutil

import pytest
from cli_probe_helpers import (FPS, LED, LED_BOX, N, ONSET, WALL, WALL_BOX, assert_rows, decode, onset_frame, probe,
                               table, write_manifest)
from helpers import error_line

from outline_tracker import __version__, provenance
from outline_tracker.session import Clip, ProbeBox, Session, TimeSettings, VideoRef


@pytest.fixture(scope="module")
def frames(dish_clip):
    """The 120 frames of the dish clip, decoded here and not by the package."""
    decoded = decode(dish_clip.path)
    assert len(decoded) == N and dish_clip.scene.led.box_px == LED_BOX and dish_clip.scene.led.onset_frame == ONSET
    return decoded


def make_session(video_path, name="session.json", **changes):
    """A saved session for the video in the default run folder next to it: frames 30 to 59 with step 2,
    fps_true 239.6 from the stopwatch, the probes WALL and LED1 (in that order). Returns the session file."""
    run = video_path.parent / "dish_tracker_outline_ana"
    settings = dict(
        student="ana", clip=Clip(start=30, end=59, step=2), time=TimeSettings(fps_true=FPS, source="stopwatch"),
        probes=[ProbeBox(name="WALL", rect_px=list(WALL_BOX)), ProbeBox(name="LED1", rect_px=list(LED_BOX))],
        video=VideoRef.from_file(video_path, run, width=320, height=240, n_frames=N, fps_container=240.0),
    ) | changes
    return Session(**settings).save(run / name)


def log_lines(folder, start) -> list[str]:
    """The lines of the folder's run.log that begin with `start`."""
    return [line for line in (folder / "run.log").read_text(encoding="utf-8").splitlines() if line.startswith(start)]


# --------------------------------------------------------------------------- probe SESSION.json

@pytest.mark.parametrize("name", ["session.json", "ana copy.JSON"])
def test_probe_session_uses_the_sessions_boxes_clip_and_fps(name, clip_in_odd_folder, frames, tmp_path, monkeypatch,
                                                            capsys):
    session_file = make_session(clip_in_odd_folder.path, name)
    before = session_file.read_bytes()
    write_manifest(tmp_path, "238.0")  # the session's fps_true counts, not a manifest in the current folder
    monkeypatch.chdir(tmp_path)
    assert probe(session_file) == 0
    run = session_file.parent
    rows = table(run)
    assert rows["probe"].tolist() == ["WALL", "LED1"] * 30  # the session's probes, in its order
    assert_rows(rows[rows["probe"] == "WALL"], frames, range(30, 60), WALL_BOX)  # every frame: the step is not used
    assert_rows(rows[rows["probe"] == "LED1"], frames, range(30, 60), LED_BOX)
    assert onset_frame(rows[rows["probe"] == "LED1"], clip_in_odd_folder.scene.led) == ONSET
    assert sorted(path.name for path in run.iterdir()) == sorted([name, "probes.csv", "run.log"])
    assert session_file.read_bytes() == before  # the session is read, never written
    shown = capsys.readouterr()
    assert shown.err == "" and str(run / "probes.csv") in shown.out


def test_the_run_folder_can_have_moved_with_its_video(clip_in_odd_folder, frames, tmp_path):
    session_file = make_session(clip_in_odd_folder.path)
    moved = tmp_path / "moved here"
    shutil.move(clip_in_odd_folder.path.parent, moved)  # the stored absolute path is now stale (review focus 5)
    run = moved / session_file.parent.name
    assert probe(run / "session.json") == 0
    assert_rows(table(run)[lambda rows: rows["probe"] == "LED1"], frames, range(30, 60), LED_BOX)


@pytest.mark.parametrize("option, value", [("--rect", LED), ("--rect", WALL), ("--start", 0), ("--end", 100),
                                            ("--fps", 240), ("--out", "elsewhere"), ("--student", "ben")])
def test_an_option_of_the_video_form_is_an_error_with_a_session(option, value, clip_in_odd_folder, tmp_path,
                                                                 monkeypatch, capsys):
    session_file = make_session(clip_in_odd_folder.path)
    monkeypatch.chdir(tmp_path)
    assert probe(session_file, option, value) == 1
    message = error_line(capsys)
    assert option in message and "session" in message
    assert [path.name for path in session_file.parent.iterdir()] == ["session.json"]
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("changes, said", [
    ({"probes": []}, "no probe rectangle"),
    ({"time": TimeSettings(fps_true=None)}, "fps_true"),
    ({"probes": [ProbeBox(name="LED1", rect_px=list(LED_BOX)), ProbeBox(name="LED1", rect_px=list(WALL_BOX))]}, "LED1"),
    ({"probes": [ProbeBox(name="LED1", rect_px=[400, 300, 420, 320])]}, "LED1"),  # outside the 320 x 240 px frame
    ({"probes": [ProbeBox(name="LED1", rect_px=[8, 8, 40])]}, "LED1"),
    ({"clip": Clip(start=50, end=40, step=2)}, "before its start"),
    ({"clip": Clip(start=30.5, end=59, step=2)}, "whole frame numbers"),  # a session edited by hand
])
def test_a_session_that_cannot_be_probed_is_an_error_and_writes_nothing(changes, said, clip_in_odd_folder, capsys):
    session_file = make_session(clip_in_odd_folder.path, **changes)
    assert probe(session_file) == 1
    assert said in error_line(capsys)
    assert [path.name for path in session_file.parent.iterdir()] == ["session.json"]


def test_a_session_whose_video_is_gone_is_an_error(clip_in_odd_folder, capsys):
    session_file = make_session(clip_in_odd_folder.path)
    clip_in_odd_folder.path.unlink()
    assert probe(session_file) == 1
    assert "dish_tracker.mp4 was not found" in error_line(capsys)
    assert [path.name for path in session_file.parent.iterdir()] == ["session.json"]


def test_another_file_in_the_videos_place_is_refused(clip_in_odd_folder, capsys):
    session_file = make_session(clip_in_odd_folder.path)
    clip_in_odd_folder.path.write_bytes(clip_in_odd_folder.path.read_bytes()[:-100])  # e.g. converted again
    assert probe(session_file) == 1
    assert "is not the video of this session" in error_line(capsys)
    assert [path.name for path in session_file.parent.iterdir()] == ["session.json"]


def test_a_missing_session_file_is_one_error_line(tmp_path, capsys):
    missing = tmp_path / "run" / "session.json"
    assert probe(missing) == 1
    assert capsys.readouterr().err == f"ERROR: No such file: {missing}\n"
    assert not missing.parent.exists()


def test_a_json_file_that_is_no_session_is_an_error(tmp_path, capsys):
    other = tmp_path / "notes.json"
    other.write_text('{"fps_true": 239.6}\n', encoding="utf-8")
    assert probe(other) == 1
    assert "is not an outline-tracker session file" in error_line(capsys)
    assert [path.name for path in tmp_path.iterdir()] == ["notes.json"]


# --------------------------------------------------------------------------- run.log (SPEC 8.9)

def test_run_log_says_what_was_measured(dish_clip, tmp_path):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--rect", WALL, "--fps", FPS, "--out", out) == 0
    data = (out / "run.log").read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")  # LF line ends on every platform
    lines = data.decode("utf-8").splitlines()
    assert "probe" in lines[0] and f"outline-tracker {__version__}" in lines[0]
    (video_line,) = log_lines(out, "video: ")
    assert str(dish_clip.path) in video_line and "320 x 240 px" in video_line
    (fps_line,) = log_lines(out, "fps_true: ")
    assert fps_line.startswith("fps_true: 239.6 ") and "--fps" in fps_line
    (probes_line,) = log_lines(out, "probes: ")
    assert "LED1 8,8,40,28" in probes_line and "WALL 150,110,170,130" in probes_line and "px" in probes_line
    assert log_lines(out, "frames: ") == ["frames: 0 to 119, every frame (120 frames)"]
    (wrote_line,) = log_lines(out, "wrote: ")
    assert "probes.csv" in wrote_line and "240 rows" in wrote_line  # 120 frames x 2 probes


def test_run_log_entry_is_headed_by_the_version_line_with_the_commit(dish_clip, tmp_path, monkeypatch):
    # SPEC 8.9: "tool version and commit"; the line `--version` prints and an export's block starts with
    monkeypatch.setattr(provenance, "tool_version", lambda: "outline-tracker 0.1.0 (commit abc1234)")
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 3) == 0
    head = (out / "run.log").read_text(encoding="utf-8").splitlines()[0]
    assert re.fullmatch(r"==== probe, \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}, "
                        r"outline-tracker 0\.1\.0 \(commit abc1234\) ====", head), head


def test_run_log_names_the_manifest_fps_came_from(dish_clip, tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, "238.0")
    monkeypatch.chdir(tmp_path)
    assert probe(dish_clip.path, "--rect", LED, "--out", tmp_path / "out", "--end", 3) == 0
    (fps_line,) = log_lines(tmp_path / "out", "fps_true: ")
    assert fps_line.startswith("fps_true: 238.0 ") and "manifest" in fps_line and str(manifest.resolve()) in fps_line


def test_run_log_of_a_session_names_the_sessions_fps_source(clip_in_odd_folder):
    session_file = make_session(clip_in_odd_folder.path)
    assert probe(session_file) == 0
    run = session_file.parent
    (fps_line,) = log_lines(run, "fps_true: ")
    assert fps_line.startswith("fps_true: 239.6 ") and "session" in fps_line and "stopwatch" in fps_line
    assert log_lines(run, "frames: ") == ["frames: 30 to 59, every frame (30 frames)"]
    (video_line,) = log_lines(run, "video: ")
    assert str(clip_in_odd_folder.path) in video_line


@pytest.mark.parametrize("earlier, gap", [
    (b"an earlier entry\nof two lines\n", b"\n"),  # a blank line, then the new entry
    (b"no line end at the end", b"\n\n"),  # its last line is ended first
    (b"", b""),
])
def test_each_probe_appends_one_entry_and_keeps_what_was_there(earlier, gap, dish_clip, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.log").write_bytes(earlier)
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 3) == 0
    once = (out / "run.log").read_bytes()
    assert once.startswith(earlier + gap + b"==== probe")
    assert probe(dish_clip.path, "--rect", WALL, "--fps", 120, "--out", out, "--start", 2, "--end", 5) == 0
    twice = (out / "run.log").read_bytes()
    assert twice.startswith(once)  # appended: nothing that was there is rewritten
    assert log_lines(out, "frames: ") == ["frames: 0 to 3, every frame (4 frames)",
                                          "frames: 2 to 5, every frame (4 frames)"]
    assert [line.split()[1] for line in log_lines(out, "fps_true: ")] == ["239.6", "120.0"]


def test_run_log_says_when_the_video_ended_early_and_a_low_fps(dish_clip, tmp_path):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", 30, "--out", out, "--start", 100, "--end", 500) == 0
    assert log_lines(out, "frames: ") == ["frames: 100 to 119, every frame (20 frames)"]  # what was measured
    (note,) = log_lines(out, "note: ")
    assert "ends at frame 119" in note
    (warning,) = log_lines(out, "warning: ")
    assert "fps_true" in warning and "100" in warning
