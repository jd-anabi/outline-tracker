"""Export: run.log (SPEC 8.9). One block is appended per export; what is there is never rewritten.

Every number looked for comes from the session the test wrote: the stick of tests/export_helpers.py
(300 px for 30 mm: k = 0.1 mm per px, and sqrt(2) x 0.5 / 300 = 0.24 % from click precision), a tape
check of 201 px for 20 mm (measured 20.1 mm, +0.50 %), the dish circle of the scene (R = 108 px, so
2R = 21.6 mm), the video's own facts, and the run records that tracking stored. The text is plain
ASCII (um, mm^2, deg).
"""

import platform
import re
import shutil
from importlib import metadata

import pytest
from export_helpers import MODEL, add_fine, coarse_run, load

from outline_tracker import cli, export, provenance, qc
from outline_tracker.session import Correction, ProbeBox

VERSION_LINE = "outline-tracker 0.1.0 (commit abc1234)"
SECTIONS = ["software:", "machine:", "libraries:", "device:", "model:", "video:", "time:", "calibration:", "runs:",
            "results:", "corrections:", "qc summary:", "outputs:"]
WEIGHTS = "ab" * 32


@pytest.fixture(scope="module")
def tracked(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse inside the dish crop, with a tape check, a dish
    diameter, the model's facts in the run record and one correction in the session."""
    folder = tmp_path_factory.mktemp("log") / "run"
    session = coarse_run(dish_clip, folder, ["A", "B", "C"])
    session.calibration.check = {"p1_px": [10.0, 40.0], "p2_px": [211.0, 40.0], "true_mm": 20.0}
    session.circle.dish_mm = 22.0
    session.circle.rms_px = 0.4
    session.runs[0].device = "cpu"
    session.runs[0].model_id = "yonigozlan/EdgeTAM-hf"
    session.runs[0].weights_sha256 = WEIGHTS
    session.corrections.append(Correction(time="2026-10-07T03:10:00-07:00", tracks=["B"], action="retrack", frame=60))
    session.save(folder / "session.json")
    return folder


@pytest.fixture
def run_folder(tracked, tmp_path, monkeypatch):
    """This test's own copy of the run folder; the tool says it is commit abc1234."""
    monkeypatch.setattr(provenance, "tool_version", lambda: VERSION_LINE)
    return shutil.copytree(tracked, tmp_path / "run")


def _log(run_folder):
    export.export_all(run_folder, log=lambda line: None)
    data = (run_folder / "run.log").read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    return data.decode("utf-8")


def _line(text, start):
    """The one line of the log that starts with `start` (after its indent)."""
    (line,) = [line for line in text.split("\n") if line.strip().startswith(start)]
    return line


def test_the_block_has_every_section_in_order_in_plain_ascii(run_folder):
    text = _log(run_folder)
    assert text.isascii()
    lines = text.split("\n")
    assert re.fullmatch(r"==== export, \d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}, "
                        + re.escape(VERSION_LINE) + " ====", lines[0]), lines[0]
    starts = [next(i for i, line in enumerate(lines) if line.startswith(section)) for section in SECTIONS]
    assert starts == sorted(starts)


def test_software_machine_and_model(run_folder):
    text = _log(run_folder)
    assert _line(text, "software:") == f"software: {VERSION_LINE}"
    machine = _line(text, "machine:")
    assert platform.platform() in machine and f"Python {platform.python_version()}" in machine
    libraries = _line(text, "libraries:")
    for name in ("torch", "transformers", "numpy"):
        assert f"{name} {metadata.version(name)}" in libraries
    assert _line(text, "device:") == "device: cpu"
    model = _line(text, "model:")
    assert MODEL in model and "yonigozlan/EdgeTAM-hf" in model and WEIGHTS in model


def test_video_time_and_calibration(run_folder, dish_clip):
    session, _ = load(run_folder)
    text = _log(run_folder)
    video = _line(text, "video:")
    assert "dish_tracker.mp4" in video and f"{dish_clip.path.stat().st_size:,} bytes" in video
    assert session.video.sha256_first_64mib in video and len(session.video.sha256_first_64mib) == 64
    assert "240.00 fps" in video and "120 frames" in video and "320 x 240 px" in video
    time = _line(text, "time:")
    assert "fps_true = 240" in time and "typed" in time
    scale = _line(text, "scale:")
    assert "0.1000000 mm per px" in scale and "100.000 um per px" in scale and "0.24 %" in scale
    stick = _line(text, "stick:")
    assert "(10.000, 20.000)" in stick and "(310.000, 20.000)" in stick and "30 mm" in stick
    tape = _line(text, "tape check:")
    assert "true 20 mm" in tape and "measured 20.1000 mm" in tape and "+0.50 %" in tape
    circle = _line(text, "circle:")
    assert "(160.300, 119.800)" in circle and "R = 108.000 px" in circle and "rms 0.400 px" in circle
    assert "2R = 21.6000 mm" in circle and "dish_mm = 22" in circle
    axes = _line(text, "axes:")
    assert "(100.500, 60.250)" in axes and "alpha = 0 deg" in axes


def test_runs_corrections_qc_summary_and_outputs(run_folder):
    session, store = load(run_folder)
    text = _log(run_folder)
    run = _line(text, "run 1:")
    for part in ("A, B, C", "coarse", "start frame 0", "step 2", "60 frames", "dish", "224",
                 f"{session.runs[0].seconds_per_frame:.3f} s per frame"):
        assert part in run, part
    assert _line(text, "results:") == "results: complete"
    correction = _line(text, "2026-10-07T03:10:00-07:00")
    assert "retrack" in correction and "B" in correction and "frame 60" in correction
    data = export.derive_all(session, store)
    summary = qc.summary_lines(data.flags, data.derived)
    assert "B: LOWRES in 60 of 60 frames (first at frame 0, t = 0.000 s)" in summary  # a body 14.5 px long
    lines = text.split("\n")
    first = lines.index("qc summary:") + 1
    assert [line.strip() for line in lines[first:first + len(summary)]] == summary
    for name in ("positions.csv", "shapes.csv", "radial.csv", "outlines.npz", "README.txt", f"{MODEL}/A.csv",
                 f"{MODEL}/B.csv", f"{MODEL}/C.csv"):
        assert _line(text, f"{name}:").strip() == f"{name}: {(run_folder / name).stat().st_size:,} bytes"


def test_each_export_appends_a_block_and_keeps_what_was_there(run_folder):
    (run_folder / "run.log").write_text("==== probe, an earlier entry ====\nwrote: probes.csv\n", encoding="utf-8",
                                        newline="\n")
    first = _log(run_folder)
    assert first.startswith("==== probe, an earlier entry ====\nwrote: probes.csv\n\n==== export, ")
    second = _log(run_folder)
    assert second.startswith(first + "\n==== export, ")
    assert second.count("==== export, ") == 2 and second.count("==== probe, ") == 1


def test_probe_and_export_head_their_entries_with_one_version_line(tracked, tmp_path):
    # one run.log, two commands: both name the tool by the line `--version` prints, commit included
    run_folder = shutil.copytree(tracked, tmp_path / "run")
    session, _ = load(run_folder)
    session.probes = [ProbeBox(name="LED1", rect_px=[8, 8, 40, 28])]
    session.save(run_folder / "session.json")
    assert cli.main(["probe", str(run_folder / "session.json")]) == 0
    export.export_all(run_folder, log=lambda line: None)
    heads = [line for line in (run_folder / "run.log").read_text(encoding="utf-8").split("\n")
             if line.startswith("====")]
    assert [head.split(",")[0] for head in heads] == ["==== probe", "==== export"]
    versions = [head.split(", ", 2)[2].removesuffix(" ====") for head in heads]
    assert versions[0] == versions[1]
    assert re.fullmatch(r"outline-tracker 0\.2\.0\.dev0 \(commit ([0-9a-f]{7,40}|unknown)\)", versions[0]), versions


def test_a_run_without_a_crop_and_a_fine_run_say_what_the_model_saw(shapes_clip, tmp_path):
    run_folder = tmp_path / "run"
    coarse_run(shapes_clip, run_folder, ["B"], dish_crop=False)
    add_fine(shapes_clip, run_folder, "A", 240)
    text = _log(run_folder)
    coarse, fine = _line(text, "run 1:"), _line(text, "run 2:")
    assert "tracks B" in coarse and "coarse" in coarse and "whole frame" in coarse and "640 x 480 px" in coarse
    assert "tracks A" in fine and "fine" in fine and "W = 240 px" in fine and "30 frames" in fine
    assert "1.000 s per frame" in fine and "30.0 s in all" in fine  # as the run record says
    assert _line(text, "corrections:") == "corrections: none"
    assert "tape check: none" in text
    assert _line(text, "device:") == "device: cpu"  # ExactFake names no device; the fine run's record does


def test_a_session_from_a_tracker_export_has_a_fitted_scale_and_no_stick(run_folder):
    session, _ = load(run_folder)
    session.calibration.stick = None
    session.calibration.tracker_fit = {"mm_per_px": 0.0324, "rms_mm": 5.0e-6, "n_points": 12}
    session.time.source = "tracker-export"
    session.circle = None
    session.save(run_folder / "session.json")
    text = _log(run_folder)
    scale = _line(text, "scale:")
    assert "0.0324000 mm per px" in scale and "Tracker export" in scale and "12 points" in scale
    assert "stick: none" in text and "circle: none" in text
    assert "tracker-export" in _line(text, "time:")
