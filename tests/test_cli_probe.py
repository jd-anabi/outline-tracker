"""`outline-tracker probe VIDEO --rect ...` (SPEC 4.6, 8.7, 11; decisions X9 and X19; review focus 1 to 3).

Expected values: see tests/cli_probe_helpers.py (the scene's LED, frames decoded with OpenCV alone,
t_s = frame / fps_true). An error is one `ERROR: ...` line on stderr and exit code 1, and writes nothing;
a command line argparse cannot read is exit code 2. The session form is in tests/test_cli_probe_session.py.
"""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from cli_probe_helpers import (COLUMNS, FPS, LED, LED_BOX, N, ONSET, UNKNOWN_FPS, WALL, WALL_BOX, assert_rows, decode,
                               error_line, onset_frame, probe, table, write_manifest)

from outline_tracker import cli, video

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frames(dish_clip):
    """The 120 frames of the dish clip, decoded here and not by the package."""
    decoded = decode(dish_clip.path)
    assert len(decoded) == N and dish_clip.scene.led.box_px == LED_BOX and dish_clip.scene.led.onset_frame == ONSET
    return decoded


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """The current folder is an empty folder with no manifest: fps_true can only come from --fps."""
    folder = tmp_path / "nowhere"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


# --------------------------------------------------------------------------- the file

def test_onset_frame_in_probes_csv_is_exact(dish_clip, frames, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out) == 0
    rows = table(out)
    assert list(rows.columns) == COLUMNS
    assert (rows["probe"] == "LED1").all()
    assert onset_frame(rows, dish_clip.scene.led) == ONSET
    assert_rows(rows, frames, range(N), LED_BOX)  # every frame of the video, first to last
    shown = capsys.readouterr()
    assert shown.err == ""
    assert str(out / "probes.csv") in shown.out
    assert "NOTE" not in shown.out and "WARNING" not in shown.out  # the default end is the last frame, not beyond


def test_the_file_is_written_as_spec_8_7_says(dish_clip, tmp_path):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", "240", "--out", out, "--end", 48) == 0
    data = (out / "probes.csv").read_bytes()
    lines = data.decode("utf-8").split("\n")
    assert b"\r" not in data and lines[-1] == ""  # LF line ends on every platform, the last line ended
    assert lines[0] == "frame,t_s,probe,r,g,b,gray"
    assert len(lines) == 1 + 49 + 1
    assert lines[1].startswith("0,0.0000000,LED1,") and lines[49].startswith("48,0.2000000,LED1,")  # 48 / 240
    assert all(len(cell.split(".")[1]) == 3 for cell in lines[1].split(",")[3:])  # means with 3 decimals


def test_two_rect_options_give_two_probes(dish_clip, frames, tmp_path):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", WALL, "--rect", LED, "--fps", FPS, "--out", out) == 0
    rows = table(out)
    assert rows["probe"].tolist() == ["WALL", "LED1"] * N  # every frame, the probes in the order given
    assert_rows(rows[rows["probe"] == "WALL"], frames, range(N), WALL_BOX)
    assert_rows(rows[rows["probe"] == "LED1"], frames, range(N), LED_BOX)
    assert onset_frame(rows[rows["probe"] == "LED1"], dish_clip.scene.led) == ONSET


def test_start_and_end_are_both_included(dish_clip, frames, tmp_path):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--start", 38, "--end", 44, "--fps", FPS, "--out", out) == 0
    assert_rows(table(out), frames, range(38, 45), LED_BOX)  # frames keep the video's numbers and times


def test_fractional_corners_and_a_name_with_a_colon(dish_clip, frames, tmp_path):
    # pixel centers at +0.5: u from 7.6 to 39.6 holds columns 8 to 39, v from 7.6 to 27.6 holds rows 8 to 27
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", "led:left:7.6,7.6,39.6,27.6", "--fps", FPS, "--out", out, "--end", 5) == 0
    rows = table(out)
    assert (rows["probe"] == "led:left").all()
    assert_rows(rows, frames, range(6), LED_BOX)


# --------------------------------------------------------------------------- fps_true (X9)

def test_a_missing_fps_gives_the_ported_message_and_writes_nothing(dish_clip, nowhere, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--out", out) == 1  # the file says 240 fps: never used
    assert capsys.readouterr().err == UNKNOWN_FPS
    assert not out.exists()


def test_fps_comes_from_the_manifest_in_the_current_folder(dish_clip, frames, nowhere, tmp_path, capsys):
    manifest = write_manifest(nowhere, "238.0")
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--out", out) == 0
    assert_rows(table(out), frames, range(N), LED_BOX, fps_true=238.0)  # frame 119 is at 0.5 s, not 119 / 240
    shown = capsys.readouterr().out
    assert "238" in shown and str(manifest.resolve()) in shown  # says which fps_true, and from where


def test_fps_comes_from_a_manifest_above_the_video(clip_in_odd_folder, frames, nowhere, tmp_path):
    write_manifest(clip_in_odd_folder.path.parent.parent, "120.5", video_file="videos/dish.mov")
    out = tmp_path / "out"
    assert probe(clip_in_odd_folder.path, "--rect", LED, "--out", out, "--end", 9) == 0
    assert_rows(table(out), frames, range(10), LED_BOX, fps_true=120.5)


def test_fps_option_comes_before_the_manifest(dish_clip, frames, nowhere, tmp_path):
    write_manifest(nowhere, "238.0")
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 9) == 0
    assert_rows(table(out), frames, range(10), LED_BOX, fps_true=FPS)


@pytest.mark.parametrize("cell", ["", "0", "-240", "nan", "inf", "about 240"])
def test_a_manifest_without_a_usable_fps_counts_as_unknown(cell, dish_clip, nowhere, tmp_path, capsys):
    write_manifest(nowhere, cell)
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--out", out) == 1
    assert capsys.readouterr().err == UNKNOWN_FPS
    assert not out.exists()


@pytest.mark.parametrize("fps", ["0", "-239.6", "nan", "inf"])
def test_an_fps_option_that_is_not_a_positive_number_is_an_error(fps, dish_clip, nowhere, tmp_path, capsys):
    write_manifest(nowhere, "238.0")  # not used instead: what was typed is wrong, so say so
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, f"--fps={fps}", "--out", out) == 1
    assert "--fps" in error_line(capsys)
    assert not out.exists()


def test_a_low_fps_gets_a_warning_and_is_used(dish_clip, frames, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", 30, "--out", out, "--end", 9) == 0  # SPEC 3.3: warn only
    assert_rows(table(out), frames, range(10), LED_BOX, fps_true=30.0)
    shown = capsys.readouterr()
    warnings = [line for line in shown.out.splitlines() if line.startswith("WARNING: ")]
    assert len(warnings) == 1 and "fps_true" in warnings[0] and "100" in warnings[0]


def test_a_normal_fps_gets_no_warning(dish_clip, tmp_path, capsys):
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", tmp_path / "out", "--end", 9) == 0
    assert "WARNING" not in capsys.readouterr().out


# --------------------------------------------------------------------------- where the file goes (SPEC 8.1)

def test_student_gives_the_run_folder_next_to_the_video(clip_in_odd_folder, frames, capsys):
    clip = clip_in_odd_folder.path  # in a folder with a space and non-ASCII characters (review focus 1)
    assert probe(clip, "--rect", LED, "--fps", FPS, "--student", "Zoë Müller") == 0
    run = clip.parent / "dish_tracker_outline_Zoë_Müller"
    assert run.is_dir() and len(list(clip.parent.iterdir())) == 2  # the video and its run folder
    assert sorted(path.name for path in run.iterdir()) == ["probes.csv", "run.log"]  # no session.json is made
    assert_rows(table(run), frames, range(N), LED_BOX)
    assert str(run / "probes.csv") in capsys.readouterr().out


def test_out_comes_before_student(clip_in_odd_folder, tmp_path):
    out = tmp_path / "chosen folder"
    clip = clip_in_odd_folder.path
    assert probe(clip, "--rect", LED, "--fps", FPS, "--student", "ana", "--out", out, "--end", 3) == 0
    assert (out / "probes.csv").is_file()
    assert [path.name for path in clip.parent.iterdir()] == ["dish_tracker.mp4"]


def test_without_out_and_student_the_message_shows_both_forms(clip_in_odd_folder, capsys):
    clip = clip_in_odd_folder.path
    assert probe(clip, "--rect", LED, "--fps", FPS) == 1
    message = error_line(capsys)
    assert "--out DIR" in message and "--student NAME" in message
    assert "dish_tracker_outline_NAME" in message  # where --student would put it
    assert [path.name for path in clip.parent.iterdir()] == ["dish_tracker.mp4"]


def test_an_empty_student_name_is_an_error(clip_in_odd_folder, capsys):
    assert probe(clip_in_odd_folder.path, "--rect", LED, "--fps", FPS, "--student", "  ") == 1
    assert "name" in error_line(capsys)
    assert [path.name for path in clip_in_odd_folder.path.parent.iterdir()] == ["dish_tracker.mp4"]


@pytest.mark.parametrize("option", ["--out", "--student"])
def test_a_folder_of_tracker_files_is_not_written_into(option, clip_in_odd_folder, capsys):
    clip = clip_in_odd_folder.path
    folder = clip.parent / "dish_tracker_outline_ana"
    folder.mkdir()
    (folder / "A.csv").write_text(",A,,,,,\nt,frame,x,y,pixelx,pixely\n", encoding="utf-8")
    where = folder if option == "--out" else "ana"
    assert probe(clip, "--rect", LED, "--fps", FPS, option, where) == 1
    message = error_line(capsys)
    assert "Tracker files" in message and str(folder) in message
    assert [path.name for path in folder.iterdir()] == ["A.csv"]


def test_a_run_folder_with_earlier_output_is_written_again(dish_clip, frames, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for name in ("positions.csv", "README.txt", "probes.csv"):
        (out / name).write_text("old\n", encoding="utf-8")
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 3) == 0
    assert_rows(table(out), frames, range(4), LED_BOX)  # the old probes.csv is replaced
    assert (out / "positions.csv").read_text(encoding="utf-8") == "old\n"  # nothing else is touched
    assert sorted(path.name for path in out.iterdir()) == ["README.txt", "positions.csv", "probes.csv", "run.log"]


def test_a_locked_probes_csv_keeps_the_rows_in_a_new_file_and_says_so(dish_clip, frames, tmp_path, monkeypatch,
                                                                      lock_file, capsys):
    out = tmp_path / "run"
    out.mkdir()
    (out / "probes.csv").write_text("old\n", encoding="utf-8")
    lock_file(out / "probes.csv")  # as on Windows while probes.csv is open in a spreadsheet program
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 3) == 0
    assert (out / "probes.csv").read_text(encoding="utf-8") == "old\n"
    assert_rows(pd.read_csv(out / "probes.new.csv"), frames, range(4), LED_BOX)
    shown = capsys.readouterr().out
    assert str(out / "probes.new.csv") in shown
    notes = [line for line in shown.splitlines() if line.startswith("NOTE: ")]
    assert len(notes) == 1 and "probes.csv is open in another program" in notes[0] and "probes.new.csv" in notes[0]
    assert "wrote: probes.new.csv" in (out / "run.log").read_text(encoding="utf-8")
    monkeypatch.undo()
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 5) == 0  # the folder is still ours
    assert_rows(table(out), frames, range(6), LED_BOX)


# --------------------------------------------------------------------------- the video and the frames

def test_a_missing_video_is_one_error_line(tmp_path, capsys):
    missing = tmp_path / "does_not_exist_tracker.mp4"
    assert probe(missing, "--rect", LED, "--fps", FPS, "--out", tmp_path / "out") == 1
    assert capsys.readouterr().err == f"ERROR: No such file: {missing}\n"
    assert not (tmp_path / "out").exists()


def test_an_end_past_the_video_stops_at_the_last_frame_and_says_so(dish_clip, frames, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--start", 100, "--end", 500) == 0
    assert_rows(table(out), frames, range(100, N), LED_BOX)  # review focus 2: keeps what it has
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("NOTE: ")]
    assert len(notes) == 1 and "ends at frame 119" in notes[0]


@pytest.mark.parametrize("frames_asked, said", [
    (("--start", 500), "120 frames (0 to 119), so there is no frame 500"),  # with the default end
    (("--start", 500, "--end", 600), "ends before frame 500"),  # nothing to read at all
    (("--start", 50, "--end", 40), "before its start"),
    (("--start=-1",), "start"),
    (("--end=-5",), "end"),
])
def test_frames_that_make_no_sense_are_an_error_and_write_nothing(frames_asked, said, dish_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, *frames_asked) == 1
    assert said in error_line(capsys)
    assert not out.exists()


def test_a_video_that_does_not_say_its_length_needs_an_end(dish_clip, frames, tmp_path, monkeypatch, capsys):
    read = video.probe
    monkeypatch.setattr(video, "probe", lambda path: replace(read(path), n_frames=0))  # as some containers do
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out) == 1
    assert "--end" in error_line(capsys)
    assert not out.exists()
    assert probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", out, "--end", 9) == 0  # with an end it runs
    assert_rows(table(out), frames, range(10), LED_BOX)


# --------------------------------------------------------------------------- the rectangles

def test_a_rectangle_outside_the_frame_is_an_error_and_writes_nothing(dish_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", "LED1:400,300,420,320", "--fps", FPS, "--out", out) == 1  # 320 x 240 px
    assert "LED1" in error_line(capsys)
    assert not out.exists()


def test_without_a_rectangle_there_is_nothing_to_measure(dish_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--fps", FPS, "--out", out) == 1
    assert "--rect NAME:u0,v0,u1,v1" in error_line(capsys)
    assert not out.exists()


def test_two_rectangles_with_one_name_are_an_error(dish_clip, tmp_path, capsys):
    out = tmp_path / "out"
    assert probe(dish_clip.path, "--rect", LED, "--rect", "LED1:150,110,170,130", "--fps", FPS, "--out", out) == 1
    assert "LED1" in error_line(capsys)
    assert not out.exists()


@pytest.mark.parametrize("rect", ["LED1:8,8,40", "LED1:8,8,40,28,3", "8,8,40,28", ":8,8,40,28", "LED1:a,b,c,d",
                                  "LED1:8,8,40,nan", "LED1:8,8,inf,28", "LED1", "LED1:8 8 40 28"])
def test_a_rect_that_cannot_be_read_is_a_usage_error(rect, dish_clip, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        probe(dish_clip.path, f"--rect={rect}", "--fps", FPS, "--out", tmp_path / "out")
    assert stopped.value.code == 2
    shown = capsys.readouterr().err
    assert "argument --rect" in shown and rect in shown and "NAME:u0,v0,u1,v1" in shown
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("option", ["--start", "--end", "--fps"])
def test_a_number_that_cannot_be_read_is_a_usage_error(option, dish_clip, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        probe(dish_clip.path, "--rect", LED, "--fps", FPS, "--out", tmp_path / "out", option, "many")
    assert stopped.value.code == 2
    assert option in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- the program

def test_probe_needs_a_video_or_a_session(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["probe"])
    assert stopped.value.code == 2
    assert "VIDEO" in capsys.readouterr().err


def test_help_shows_both_forms_and_every_option(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["probe", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "outline-tracker probe VIDEO --rect NAME:u0,v0,u1,v1" in shown
    assert "outline-tracker probe SESSION.json" in shown
    for option in ("--rect", "--start", "--end", "--fps", "--out", "--student"):
        assert option in shown
    assert cli.main([]) == 0
    assert "probe" in capsys.readouterr().out  # listed with the other commands


def test_probe_loads_neither_the_model_nor_qt(dish_clip, tmp_path):
    # in a fresh process, so that what this test session has already imported does not count
    code = ("import sys\n"
            "from outline_tracker import cli\n"
            "status = cli.main(sys.argv[1:])\n"
            "print('HEAVY', sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph') if m in sys.modules))\n"
            "raise SystemExit(status)\n")
    done = subprocess.run(
        [sys.executable, "-c", code, "probe", str(dish_clip.path), "--rect", LED, "--fps", str(FPS), "--out",
         str(tmp_path / "out")],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "HEAVY []"
    assert onset_frame(table(tmp_path / "out"), dish_clip.scene.led) == ONSET
