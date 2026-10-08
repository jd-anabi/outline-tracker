"""The `selftest` command (SPEC 11, 13.4) with stand-in models: last week's selftest, through `from_tracker`.

The first test is last week's, ported, its two assertions unchanged: `selftest` with `ThresholdFake`
stands where `segment.selftest` with `DiskFinder` stood, and one assertion is added, for where SPEC 8.1
puts the file the positions are read from.

The clip is last week's: one dark ellipse on a 1080p frame, its center at (700.5 + 0.6 f, 500.5 + 0.2 f)
px in frame f, tracked on the 20 frames 0, 2, ..., 38. Expected values come from that and from last
week's numbers, which stay: the limit of 3 px, 1,200 frames for 10 s at 240 frames per s and step 2,
half as much time again for every further shrimp with EdgeTAM, and the wording of the last two lines.
A stand-in that thresholds finds the ellipse within 1 px (the template's own criterion), so one whose
masks are moved by 5 px is between 4 and 6 px off. `ok` also needs the shrimp on every frame.

The command is run in this process through `cli.main`, with the loading of the model replaced by a
stand-in (`from_tracker.load_segmenter`): everything but the model is the real command.

Coordinates: px in Tracker's convention (pixel centers at +0.5); frames are video frame numbers.
"""

import contextlib
import io
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from from_tracker_helpers import Loaded, StopsAfter, read_track, session_json
from helpers import error_line

from outline_tracker import cli, from_tracker
from outline_tracker.segmenter.base import crop_to_bbox
from outline_tracker.segmenter.fake import ThresholdFake
from outline_tracker.selftest import selftest

REPO = Path(__file__).resolve().parents[1]
RUN_FOLDER = "selftest_tracker_outline_selftest"  # SPEC 8.1: <video stem>_outline_<student>, next to the clip
CLIP_FILES = ["selftest_tracker.mp4", "selftest.csv"]  # the clip and its Tracker export, as last week
MODEL_MODULES = ("torch", "torchvision", "transformers", "timm")
QT_MODULES = ("PySide6", "pyqtgraph")
DEFERRED = (*MODEL_MODULES, *QT_MODULES, "numpy", "cv2", "pandas", "scipy", "skimage", "imageio_ffmpeg")
VERDICT = (r"{verdict}: {model} followed the test shrimp within (\d+\.\d) pixels \(should be under 3\)\. "
           r"(\d+\.\d\d) s per frame here\.")
ESTIMATE = (r"Estimate for 10 s at step 2 \(1,200 frames\): one shrimp about (\d+) min; "
            r"10 shrimp together about (\d+) min\.")


def test_selftest_runs_and_reports_the_time(tmp_path):
    res = selftest(model="stand-in", segmenter=ThresholdFake(), folder=tmp_path, log=lambda *a: None)
    assert res["ok"] and res["max_error_px"] < 1.0
    assert res["minutes_ten"] > res["minutes_one"] > 0
    # SPEC 8.1: the run has a run folder of its own next to the clip, <video stem>_outline_<student>/, and
    # the positions were read from its Tracker-format file, in a folder named after the model
    assert (tmp_path / "selftest_tracker_outline_selftest" / "stand-in" / "selftest.csv").is_file()


# ---------------------------------------------------------------------------------------------
# The function: last week's numbers and last week's two lines


def test_the_verdict_and_the_estimate_are_last_weeks(tmp_path):
    lines = []
    res = selftest(model="edgetam", segmenter=ThresholdFake(), folder=tmp_path, log=lines.append)
    assert sorted(res) == ["max_error_px", "minutes_one", "minutes_ten", "ok", "seconds_per_frame"]
    spf = res["seconds_per_frame"]
    assert spf > 0
    # 10 s at 240 frames per s and step 2 are 1,200 frames; s per frame times 1,200, in minutes
    assert res["minutes_one"] == pytest.approx(1200 * spf / 60)
    # with EdgeTAM every further shrimp takes half as long again (last week's measured factor): 1 + 9 * 0.5
    assert res["minutes_ten"] == pytest.approx(5.5 * res["minutes_one"])
    assert lines[-2:] == [
        f"\nOK: edgetam followed the test shrimp within {res['max_error_px']:.1f} pixels (should be under 3). "
        f"{spf:.2f} s per frame here.",
        f"Estimate for 10 s at step 2 (1,200 frames): one shrimp about {res['minutes_one']:.0f} min; "
        f"10 shrimp together about {res['minutes_ten']:.0f} min."]


def test_the_run_is_a_coarse_from_tracker_run_at_240_fps_without_an_overlay(tmp_path):
    selftest(model="stand-in", segmenter=ThresholdFake(), folder=tmp_path, log=lambda *a: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(CLIP_FILES + [RUN_FOLDER])
    run = tmp_path / RUN_FOLDER
    assert not (run / "overlay.mp4").exists()
    session = session_json(run)
    assert [(track["id"], track["mode"]) for track in session["tracks"]] == [("selftest", "coarse")]
    assert session["time"]["fps_true"] == 240.0 and session["time"]["source"] == "typed"  # no manifest is read
    assert (session["clip"]["start"], session["clip"]["end"], session["clip"]["step"]) == (0, 38, 2)
    got = read_track(run / "stand-in" / "selftest.csv")
    assert got["frame"].tolist() == list(range(0, 40, 2))
    # the true centers of the clip, px in Tracker's convention; the stand-in is within 1 px of them
    assert (got["pixelx"] - (700.5 + 0.6 * got["frame"])).abs().max() < 1.0
    assert (got["pixely"] - (500.5 + 0.2 * got["frame"])).abs().max() < 1.0


class FivePixelsOff(Loaded):
    """A stand-in whose masks lie 5 px to the right of the dark pixels it found."""

    def step(self, image):
        return [replace(result, offset=(result.offset[0] + 5, result.offset[1])) for result in super().step(image)]


def test_a_mask_5_px_away_is_a_problem(tmp_path):
    lines = []
    res = selftest(model="stand-in", segmenter=FivePixelsOff(), folder=tmp_path, log=lines.append)
    assert res["ok"] is False
    assert 4.0 < res["max_error_px"] < 6.0  # 5 px, and the stand-in's own error of less than 1 px
    assert res["minutes_ten"] > res["minutes_one"] > 0  # the time is reported all the same
    assert lines[-2] == (f"\nPROBLEM: stand-in followed the test shrimp within {res['max_error_px']:.1f} pixels "
                         f"(should be under 3). {res['seconds_per_frame']:.2f} s per frame here.")
    assert re.fullmatch(ESTIMATE, lines[-1])


class LostHalfway(ThresholdFake):
    """A stand-in that finds the shrimp in its first 10 images and nothing in the images after them."""

    def __init__(self):
        super().__init__()
        self.images = 0

    def step(self, image):
        self.images += 1
        results = super().step(image)
        if self.images <= 10:
            return results
        nothing = np.zeros(image.shape[:2], bool)
        return [crop_to_bbox(nothing, None, 8, obj_id=result.obj_id) for result in results]


def test_a_shrimp_that_is_lost_is_a_problem(tmp_path):
    # found on the frames 0 to 18 and lost on the frames 20 to 38: within 1 px where found, but not followed
    lines = []
    res = selftest(model="stand-in", segmenter=LostHalfway(), folder=tmp_path, log=lines.append)
    assert res["ok"] is False
    assert res["max_error_px"] < 1.0
    assert lines[-2].startswith("\nPROBLEM: stand-in followed the test shrimp within ")
    # last week's CHECK line says why: the first lost frame is frame 20, at 20 / 240 s
    assert "  CHECK: selftest: lost in 10 of 20 frames (first at t = 0.083 s)" in lines


def test_a_run_that_was_stopped_has_no_verdict(tmp_path):
    # Ctrl+C on the 4th of the 20 frames: `from_tracker` keeps the 3 tracked frames and returns
    lines = []
    with pytest.raises(RuntimeError, match="3 of 20 frames"):
        selftest(model="stand-in", segmenter=StopsAfter(3), folder=tmp_path, log=lines.append)
    assert not any("OK:" in line or "PROBLEM:" in line or "Estimate" in line for line in lines)


# ---------------------------------------------------------------------------------------------
# The command


def stand_in_for_the_model(patch, segmenter=Loaded):
    """Replace the loading of the model by a stand-in; returns the list of (model, device) asked for."""
    asked = []

    def load(model, device):
        asked.append((model, device))
        return segmenter()

    patch.setattr(from_tracker, "load_segmenter", load)
    return asked


@pytest.fixture
def temp_folder(tmp_path, monkeypatch):
    """This test's own folder in place of the computer's folder for temporary files."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    """`outline-tracker selftest`, run once in this process with a stand-in for the model: the exit
    code, what was printed, the folder that stood in for the temporary files, the (model, device) the
    loader was asked for, and the names of the modules that were imported during the run."""
    temp = tmp_path_factory.mktemp("temporary files")
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        asked = stand_in_for_the_model(patch)
        patch.setattr(tempfile, "tempdir", str(temp))
        before = set(sys.modules)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["selftest"])
        imported = set(sys.modules) - before
    return SimpleNamespace(code=code, out=out.getvalue(), err=err.getvalue(), temp=temp, asked=asked,
                           imported=imported)


def test_the_command_ends_with_last_weeks_two_lines(plain_run):
    assert plain_run.code == 0 and plain_run.err == ""
    assert plain_run.asked == [("edgetam", "auto")]  # the defaults; the model is loaded once
    lines = plain_run.out.splitlines()
    assert lines[-3] == ""  # an empty line sets the verdict apart, as last week
    verdict = re.fullmatch(VERDICT.format(verdict="OK", model="edgetam"), lines[-2])
    assert verdict, lines[-2]
    assert float(verdict[1]) < 1.0  # the stand-in is within 1 px
    estimate = re.fullmatch(ESTIMATE, lines[-1])
    assert estimate, lines[-1]
    assert int(estimate[2]) >= int(estimate[1]) >= 0


def test_without_a_folder_the_command_works_in_a_new_temporary_folder(plain_run):
    (folder,) = plain_run.temp.iterdir()  # one new folder, and nothing next to it
    assert folder.is_dir()
    assert sorted(p.name for p in folder.iterdir()) == sorted(CLIP_FILES + [RUN_FOLDER])
    assert (folder / RUN_FOLDER / "edgetam" / "selftest.csv").is_file()
    assert str(folder / RUN_FOLDER) in plain_run.out  # the console says where the files are


def test_with_a_stand_in_the_command_loads_neither_torch_nor_qt(plain_run):
    # in this process: pytest-qt has loaded PySide6 already (and on Windows torch is loaded first), so
    # what counts is that the command added nothing, not a submodule either
    heavy = sorted(name for name in plain_run.imported if name.split(".")[0] in (*MODEL_MODULES, *QT_MODULES))
    assert heavy == []
    assert plain_run.code == 0 and plain_run.asked == [("edgetam", "auto")]  # and the command did run


def test_model_and_device_reach_the_run(temp_folder, monkeypatch, capsys):
    asked = stand_in_for_the_model(monkeypatch)
    assert cli.main(["selftest", "--model", "sam2", "--device", "cpu"]) == 0
    assert asked == [("sam2", "cpu")]
    lines = capsys.readouterr().out.splitlines()
    assert re.fullmatch(VERDICT.format(verdict="OK", model="sam2"), lines[-2]), lines[-2]
    (folder,) = temp_folder.iterdir()
    assert (folder / RUN_FOLDER / "sam2" / "selftest.csv").is_file()  # the folder is named after the model
    processing = session_json(folder / RUN_FOLDER)["processing"]
    assert (processing["model"], processing["device"]) == ("sam2", "cpu")


def test_a_problem_is_exit_code_1(temp_folder, monkeypatch, capsys):
    stand_in_for_the_model(monkeypatch, FivePixelsOff)
    assert cli.main(["selftest"]) == 1
    shown = capsys.readouterr()
    assert shown.err == ""  # a verdict, not an error
    lines = shown.out.splitlines()
    verdict = re.fullmatch(VERDICT.format(verdict="PROBLEM", model="edgetam"), lines[-2])
    assert verdict, lines[-2]
    assert 4.0 < float(verdict[1]) < 6.0
    assert re.fullmatch(ESTIMATE, lines[-1]), lines[-1]


class StoppedAtOnce(StopsAfter):
    """Ctrl+C on the first frame, with the device name a loaded model has."""

    device = "test-device"

    def __init__(self):
        super().__init__(0)


def test_a_stopped_run_is_one_error_line_and_exit_code_1(temp_folder, monkeypatch, capsys):
    stand_in_for_the_model(monkeypatch, StoppedAtOnce)
    assert cli.main(["selftest"]) == 1
    assert "0 of 20 frames" in error_line(capsys)


def test_a_model_that_cannot_be_loaded_is_one_error_line_and_exit_code_1(temp_folder, monkeypatch, capsys):
    def load(model, device):
        raise OSError("Could not load the model 'edgetam': the first run needs internet once.\n(offline)")

    monkeypatch.setattr(from_tracker, "load_segmenter", load)
    assert cli.main(["selftest"]) == 1
    shown = capsys.readouterr()
    # one line, although the reason had two
    assert shown.err == "ERROR: Could not load the model 'edgetam': the first run needs internet once. (offline)\n"
    assert "OK:" not in shown.out and "PROBLEM:" not in shown.out


def test_help_shows_the_options_and_a_usage_example(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["selftest", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "outline-tracker selftest" in shown and "--model" in shown and "--device" in shown
    assert "edgetam" in shown and "sam2" in shown and "mps" in shown


def test_the_command_is_listed_with_the_others(capsys):
    assert cli.main([]) == 0
    assert any(line.split()[:1] == ["selftest"] for line in capsys.readouterr().out.splitlines())


@pytest.mark.parametrize("wrong", [["--model", "sam3"], ["--device", "gpu"], ["clip.mp4"]])
def test_a_wrong_option_is_a_usage_error(wrong, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["selftest", *wrong])
    assert stopped.value.code == 2
    assert wrong[0] in capsys.readouterr().err


def test_building_the_parser_loads_nothing_heavy():
    # cli.py imports cli_selftest for every command: numpy, OpenCV and the rest come only when `selftest` runs
    code = ("import sys\n"
            "from outline_tracker import cli\n"
            "args = cli.build_parser().parse_args(['selftest', '--device', 'cpu'])\n"
            "print(args.model, args.device, sorted(m for m in sys.argv[1:] if m in sys.modules))\n")
    done = subprocess.run([sys.executable, "-c", code, *DEFERRED], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "edgetam cpu []"
