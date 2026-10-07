"""The `from-tracker` command (SPEC 11, X19): its options, its console, its errors, and odd folder names.

The command is `from_tracker` behind argparse, so these tests run it through `cli.main` with the
loading of the model replaced by a stand-in (`ThresholdFake` with a device name): everything but the
model itself is the real command. Expected positions are where the clip's objects were drawn.

Coordinates: px in Tracker's convention (pixel centers at +0.5); frames are video frame numbers.
"""

import numpy as np
import pandas as pd
import pytest
from from_tracker_helpers import FPS, Loaded, mirrored_map, one_disk, read_track, session_json, track_text, two_disks
from from_tracker_helpers import write_start_file

from outline_tracker import cli, cli_from_tracker, from_tracker

OPTIONS = ("--fine", "--step", "--seconds", "--fps", "--out", "--student", "--model", "--device", "--no-overlay")


@pytest.fixture
def stand_in(monkeypatch):
    """The command with a stand-in instead of the model; returns the list of (model, device) it was asked for."""
    asked = []

    def load(model, device):
        asked.append((model, device))
        return Loaded()

    monkeypatch.setattr(from_tracker, "load_segmenter", load)
    return asked


def test_the_model_is_made_from_its_name_and_the_device(monkeypatch):
    from outline_tracker.segmenter import hf  # the module alone loads no torch; making the class would

    made = []
    monkeypatch.setattr(hf, "HFSegmenter", lambda *given: made.append(given) or "the segmenter")
    assert from_tracker.load_segmenter("sam2", "cpu") == "the segmenter"
    assert made == [("sam2", "cpu")]


def command(*args) -> int:
    """Run `outline-tracker from-tracker ARGS...` in this process and return its exit code."""
    return cli.main(["from-tracker", *(str(arg) for arg in args)])


def test_help_lists_the_options(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["from-tracker", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "outline-tracker from-tracker " in shown  # a usage example
    assert "VIDEO" in shown and "EXPORT" in shown
    for option in OPTIONS:
        assert option in shown
    assert "edgetam" in shown and "sam2" in shown


def test_the_command_is_listed_with_the_others(capsys):
    assert cli.main([]) == 0
    assert "from-tracker" in capsys.readouterr().out


def test_options_reach_the_function_with_their_meaning():
    parse = cli.build_parser().parse_args
    args = parse(["from-tracker", "clip_tracker.mp4", "start.csv"])
    assert cli_from_tracker.options(args) == dict(
        fine_ids=[], step=None, seconds=None, fps=None, out=None, student=None, model="edgetam", device="auto",
        overlay=True)
    args = parse(["from-tracker", "clip_tracker.mp4", "start.csv", "--fine", "B, mass C", "--step", "4", "--seconds",
                  "2.5", "--fps", "239.6", "--out", "runs/one", "--student", "ana", "--model", "sam2", "--device",
                  "cpu", "--no-overlay"])
    assert cli_from_tracker.options(args) == dict(
        fine_ids=["B", "mass C"], step=4, seconds=2.5, fps=239.6, out="runs/one", student="ana", model="sam2",
        device="cpu", overlay=False)
    # last week's spelling of --no-overlay
    assert cli_from_tracker.options(parse(["from-tracker", "v.mp4", "e.csv", "--no-video"]))["overlay"] is False


@pytest.mark.parametrize("wrong", [["--model", "sam3"], ["--device", "gpu"], ["--step", "two"]])
def test_a_wrong_option_is_a_usage_error(wrong, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["from-tracker", "clip_tracker.mp4", "start.csv", *wrong])
    assert stopped.value.code == 2
    assert wrong[0] in capsys.readouterr().err


def test_a_run_from_the_command_line(tmp_path, stand_in, capsys):
    clip, export = two_disks(tmp_path)
    assert command(clip, export, "--seconds", 8 / FPS, "--step", 4, "--fps", FPS, "--fine", "B") == 0
    shown = capsys.readouterr()
    assert shown.err == ""
    run = tmp_path / "clip_tracker_outline_ana"
    lines = shown.out.splitlines()
    assert lines[0].startswith("edgetam: 2 shrimp (A, B), frames 40-44 every 4 (2 frames, ")
    assert "  loading the model (the first time this downloads it) ..." in lines  # last week's two lines
    assert "  running on: test-device" in lines
    assert any("ana" in line and "--student" in line for line in lines) and any(str(run) in line for line in lines)
    assert stand_in == [("edgetam", "auto")]  # the model was loaded once, for both runs
    assert sorted(p.name for p in (run / "edgetam").iterdir()) == ["A.csv", "B.csv"]
    assert (run / "overlay.mp4").stat().st_size > 0
    session = session_json(run)
    assert [(t["id"], t["mode"]) for t in session["tracks"]] == [("A", "coarse"), ("B", "fine")]
    assert [one["device"] for one in session["runs"]] == ["test-device", "test-device"]
    for name, (x0, vx, y0) in zip("AB", [(60.5, 0.5, 60.5), (250.5, -0.25, 170.5)]):
        got = read_track(run / "edgetam" / f"{name}.csv")
        assert list(got["frame"]) == [40, 44]
        assert np.allclose(got["pixelx"], x0 + vx * got["frame"], atol=0.25)
        assert np.allclose(got["pixely"], y0, atol=0.25)


def test_folders_with_spaces_and_other_alphabets(clip_in_odd_folder, stand_in, capsys):
    # review focus 1: the video, the export, the student's name and the run folder all have such names
    clip = clip_in_odd_folder
    truth = clip.table[clip.table.track_id == "A"].set_index("frame")
    frames = list(range(0, 20, 2))
    export = clip.path.parent / "élève ü" / "mes exports" / "A.csv"
    export.parent.mkdir(parents=True)
    export.write_text(track_text([(f / FPS, f, truth.u_px[f], truth.v_px[f]) for f in frames], name="A"))
    out = clip.path.parent / "résultats de Zoë"
    assert command(clip.path, export, "--student", "Zoë Ünal", "--out", out) == 0
    assert capsys.readouterr().err == ""
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "README.txt", "outlines.npz", "overlay.mp4", "positions.csv", "radial.csv", "results.npz", "run.log",
        "session.json", "shapes.csv", "edgetam"])
    assert session_json(out)["student"] == "Zoë Ünal"
    assert session_json(out)["video"]["relpath"] == "../dish_tracker.mp4"
    positions = pd.read_csv(out / "positions.csv")
    assert positions.frame.tolist() == frames
    # the shrimp of the dish clip through H.264 and a threshold: within a pixel of where it was drawn
    assert np.allclose(positions.u_px, truth.u_px[frames], atol=1.0)
    assert np.allclose(positions.v_px, truth.v_px[frames], atol=1.0)

    # and the default run folder, next to the video, from the export's own folder name
    assert command(clip.path, export, "--no-overlay") == 0
    default = clip.path.parent / "dish_tracker_outline_mes_exports"
    assert session_json(default)["student"] == "mes exports"
    assert read_track(default / "edgetam" / "A.csv")["frame"].tolist() == frames


def error_line(capsys) -> str:
    """The one `ERROR: ...` line the command printed on stderr (asserted to be exactly one line)."""
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and err.endswith("\n") and err.count("\n") == 1, err
    return err


def test_errors_are_one_line_and_exit_code_1(tmp_path, stand_in, capsys):
    clip, export = one_disk(tmp_path)
    missing = tmp_path / "nothing_tracker.mp4"
    assert command(missing, export) == 1
    assert error_line(capsys) == f"ERROR: No such video: {missing}\n"
    assert command(clip, export, "--fine", "B") == 1
    assert "--fine" in error_line(capsys)
    assert command(clip, export, "--out", export.parent) == 1
    assert "folder of Tracker files" in error_line(capsys)
    assert command(clip, export, "--fps", "0") == 1
    assert "--fps" in error_line(capsys)
    mirrored = write_start_file(tmp_path / "bea" / "extra" / "start.csv",
                                {"A": (80.5, 100.5), "B": (240.5, 170.5), "C": (100.5, 200.5)}, to_mm=mirrored_map)
    assert command(clip, mirrored, "--fps", FPS) == 1
    assert "mirrored" in error_line(capsys)
    assert stand_in == []  # no model was loaded for any of them
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ana", "bea", "clip_tracker.mp4"]  # and nothing written
