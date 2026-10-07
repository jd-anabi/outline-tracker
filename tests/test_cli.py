"""Tests for the command line (SPEC 11): the ported `convert` and `check` subcommands.

The expected sizes, frame counts and messages come from how the synthetic clips are built and from
the template's `main` functions, whose messages the subcommands keep. Errors become one
`ERROR: ...` line on stderr and exit code 1, never a traceback.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from outline_tracker import cli, video

REPO = Path(__file__).resolve().parents[1]


def _write_clip(path, n_frames, fps, size):
    """A dark disk moving 1 px to the right per frame on a light background (size = width, height)."""
    width, height = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert out.isOpened(), "OpenCV could not create a test video"
    for i in range(n_frames):
        frame = np.full((height, width, 3), 200, np.uint8)
        cv2.circle(frame, (20 + i, height // 2), 8, (40, 40, 40), -1)
        out.write(frame)
    out.release()
    return path


@pytest.fixture
def good_clip(tmp_path):
    """720p at 240 fps: nothing for `check` to warn about (48 frames: too short for the ramp test)."""
    return _write_clip(tmp_path / "good.mp4", n_frames=48, fps=240, size=(1280, 720))


@pytest.fixture
def small_clip(tmp_path):
    return _write_clip(tmp_path / "small.mp4", n_frames=60, fps=240, size=(160, 120))


@pytest.fixture
def not_a_video(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("this is text, not a video\n")
    return path


# ---------------------------------------------------------------------------------------------
# check


def test_check_reports_a_good_clip(good_clip, capsys):
    assert cli.main(["check", str(good_clip)]) == 0
    shown = capsys.readouterr()
    assert str(good_clip) in shown.out
    assert "frames delivered as 1280 x 720 px" in shown.out
    assert "48 frames" in shown.out
    assert "OK: no problems found. Now measure fps_true from your stopwatch clip." in shown.out
    assert "WARNING" not in shown.out
    assert shown.err == ""


def test_check_warns_about_a_small_clip_and_exits_1(tmp_path, capsys):
    path = _write_clip(tmp_path / "phone.mp4", n_frames=60, fps=30, size=(160, 160))
    assert cli.main(["check", str(path)]) == 1
    shown = capsys.readouterr()
    warnings = [line for line in shown.out.splitlines() if line.startswith("  WARNING: ")]
    assert len(warnings) == 2  # 30 fps is not slow motion; 160 px is low resolution
    assert "fps" in warnings[0]
    assert "Low resolution (160x160)" in warnings[1]
    assert "OK:" not in shown.out


def test_check_missing_file_is_one_error_line(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.MOV"
    assert cli.main(["check", str(missing)]) == 1
    shown = capsys.readouterr()
    assert shown.err == f"ERROR: No such file: {missing}\n"
    assert shown.out == ""


def test_check_file_that_is_not_a_video_is_an_error(not_a_video, capsys):
    assert cli.main(["check", str(not_a_video)]) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith("ERROR: ")
    assert str(not_a_video) in shown.err
    assert "Traceback" not in shown.err


def test_check_goes_on_with_the_next_file_after_an_error(tmp_path, good_clip, capsys):
    missing = tmp_path / "does_not_exist.MOV"
    assert cli.main(["check", str(missing), str(good_clip)]) == 1
    shown = capsys.readouterr()
    assert shown.err.count("ERROR:") == 1
    assert "OK: no problems found" in shown.out  # the good clip was still checked


def test_check_needs_a_video(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["check"])
    assert stopped.value.code == 2
    assert "VIDEO" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# convert


def test_convert_writes_the_copy_and_says_so(small_clip, tmp_path, capsys):
    assert cli.main(["convert", str(small_clip)]) == 0
    copy = tmp_path / "small_tracker.mp4"
    info = video.probe(copy)
    assert (info.n_frames, info.width, info.height) == (60, 160, 120)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Converting {small_clip} (a minute of 240 fps video takes a few minutes) ..."
    assert lines[1].startswith(f"  wrote {copy}: 60 frames, 160 x 120 px, ")
    assert lines[1].endswith(" fps in the file")
    assert lines[2] == "  Open this copy in Tracker. Real time still comes from fps_true (your stopwatch clip)."
    assert len(lines) == 3


def test_convert_missing_file_is_one_error_line(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.MOV"
    assert cli.main(["convert", str(missing)]) == 1
    shown = capsys.readouterr()
    assert shown.err == f"ERROR: No such file: {missing}\n"
    assert not (tmp_path / "does_not_exist_tracker.mp4").exists()


def test_convert_ffmpeg_failure_is_an_error_line_not_a_traceback(not_a_video, capsys):
    assert cli.main(["convert", str(not_a_video)]) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith(f"ERROR: ffmpeg could not convert {not_a_video}")
    assert "Traceback" not in shown.err
    assert not (not_a_video.parent / "notes_tracker.mp4").exists()


def test_convert_goes_on_with_the_next_file_after_an_error(tmp_path, small_clip, capsys):
    missing = tmp_path / "does_not_exist.MOV"
    assert cli.main(["convert", str(missing), str(small_clip)]) == 1
    assert (tmp_path / "small_tracker.mp4").exists()
    assert capsys.readouterr().err.count("ERROR:") == 1


def test_convert_needs_a_video(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["convert"])
    assert stopped.value.code == 2
    assert "VIDEO" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# the program itself


def test_help_shows_a_usage_example_for_each_command(capsys):
    for command in ("convert", "check"):
        with pytest.raises(SystemExit) as stopped:
            cli.main([command, "--help"])
        assert stopped.value.code == 0
        assert f"outline-tracker {command} " in capsys.readouterr().out


def test_without_a_command_the_help_lists_the_commands(capsys):
    assert cli.main([]) == 0
    shown = capsys.readouterr().out
    assert "convert" in shown and "check" in shown


def test_exit_code_reaches_the_shell(tmp_path):
    missing = tmp_path / "does_not_exist.MOV"
    done = subprocess.run(
        [sys.executable, "-m", "outline_tracker.cli", "check", str(missing)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert done.returncode == 1
    assert done.stderr == f"ERROR: No such file: {missing}\n"


# ---------------------------------------------------------------------------------------------
# synth (hidden, decision X10): writes a synthetic clip at the spec's scale, 1920 x 1080 px, 240 fps


@pytest.mark.parametrize("scene", ["dish", "closeup"])
def test_synth_writes_a_clip_and_creates_its_folder(scene, tmp_path, capsys):
    out = tmp_path / "new folder" / f"{scene}_tracker.mp4"
    assert cli.main(["synth", scene, str(out), "--seconds", "0.1"]) == 0
    info = video.probe(out)
    assert (info.n_frames, info.width, info.height) == (24, 1920, 1080)  # 0.1 s at 240 fps
    assert info.fps_container == pytest.approx(240.0)
    shown = capsys.readouterr()
    assert shown.out.splitlines()[0] == f"wrote {out}: 24 frames, 1920 x 1080 px, 240 fps"
    assert shown.err == ""


def test_synth_lasts_two_seconds_unless_told_otherwise():
    args = cli.build_parser().parse_args(["synth", "closeup", "closeup_tracker.mp4"])
    assert (args.scene, args.out, args.seconds) == ("closeup", "closeup_tracker.mp4", 2.0)


def test_synth_is_not_shown_in_the_help(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "convert" in shown and "synth" not in shown
    assert cli.main([]) == 0
    assert "synth" not in capsys.readouterr().out


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_synth_refuses_a_length_that_is_not_positive(seconds, tmp_path, capsys):
    out = tmp_path / "clips" / "dish_tracker.mp4"
    assert cli.main(["synth", "dish", str(out), "--seconds", seconds]) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith("ERROR: ") and "--seconds" in shown.err
    assert not out.parent.exists()  # nothing written, no folder made


def test_synth_knows_only_its_two_scenes(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["synth", "ocean", str(tmp_path / "x.mp4")])
    assert stopped.value.code == 2
    assert "dish" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------
# check --seek (hidden, decision X19): 20 random frames through FrameSource against the sequential
# decode, and the number of gaps in the file's timestamps. The seek line comes last.

SEEK_OK = "  seek: 20 of 20 frames exact"


def test_check_seek_ends_with_the_gap_count_and_the_seek_line(good_clip, capsys):
    assert cli.main(["check", str(good_clip), "--seek"]) == 0
    shown = capsys.readouterr()
    lines = shown.out.splitlines()
    assert "  OK: no problems found. Now measure fps_true from your stopwatch clip." in lines
    assert lines[-2:] == ["  timestamps: 0 gaps in 48 frames", SEEK_OK]  # written evenly timed, 48 frames
    assert shown.err == ""


def test_check_seek_counts_the_gaps_of_a_clip_with_gaps(gapped_clip, capsys):
    # 320 x 240 px: `check` warns about the low resolution (exit code 1) and goes on to the seek check.
    assert cli.main(["check", "--seek", str(gapped_clip.path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("  WARNING: ") for line in lines) == 1
    assert lines[-2:] == ["  timestamps: 3 gaps in 120 frames", SEEK_OK]


def test_check_seek_reports_each_video(good_clip, small_clip, capsys):
    assert cli.main(["check", "--seek", str(good_clip), str(small_clip)]) == 1  # 160 x 120 px: low resolution
    lines = capsys.readouterr().out.splitlines()
    assert lines.count(SEEK_OK) == 2 and lines[-1] == SEEK_OK
    assert "  timestamps: 0 gaps in 48 frames" in lines and "  timestamps: 0 gaps in 60 frames" in lines


def test_check_seek_says_when_there_is_no_timestamp_table(good_clip, monkeypatch, capsys):
    monkeypatch.setattr(video, "frame_timestamps", lambda path: None)
    assert cli.main(["check", str(good_clip), "--seek"]) == 0  # still exact, only slower
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].startswith("  timestamps: no usable table") and "slower" in lines[-2]
    assert lines[-1] == SEEK_OK


def test_check_seek_with_a_wrong_frame_exits_1_and_says_how_many(good_clip, monkeypatch, capsys):
    from outline_tracker.frame_source import FrameSource

    get = FrameSource.get
    monkeypatch.setattr(FrameSource, "get", lambda self, k: get(self, (k + 1) % 48))  # always the next frame
    assert cli.main(["check", str(good_clip), "--seek"]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "  seek: 0 of 20 frames exact"


def test_check_seek_on_a_missing_file_is_one_error_line(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.MOV"
    assert cli.main(["check", str(missing), "--seek"]) == 1
    shown = capsys.readouterr()
    assert shown.err == f"ERROR: No such file: {missing}\n"
    assert shown.out == ""


def test_check_without_seek_does_not_run_the_seek_check(good_clip, capsys):
    assert cli.main(["check", str(good_clip)]) == 0
    shown = capsys.readouterr().out
    assert "seek:" not in shown and "timestamps:" not in shown


def test_check_seek_is_not_shown_in_the_help(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["check", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "VIDEO" in shown and "--seek" not in shown
