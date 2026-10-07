"""`from_tracker`: where fps_true comes from (decision X9), and what the console says (last week's lines).

fps_true is looked for in this order: `--fps`; `data/manifest.csv` in the current folder (last week's
rule); the same file upward from the export's folder, then upward from the video's; the t column of
the export. The session and run.log name the source. Expected times are frame / fps_true with the
fps_true written into the test's own manifests.

Frames are video frame numbers; fps_true is in frames per second; t in s.
"""

import re

import numpy as np
import pytest
from from_tracker_helpers import FPS, MODEL, names_in, one_disk, read_track, run, session_json, two_disks


# --------------------------------------------------------------------------- fps_true (X9)

def manifest(folder, fps, video_file="clip.MOV"):
    """<folder>/data/manifest.csv with one row for the clip's original and `fps` in its fps_true cell."""
    path = folder / "data" / "manifest.csv"
    path.parent.mkdir(parents=True)
    path.write_text(f"video_file,group,fps_true,stick_mm\n{video_file},B,{fps},30\n", encoding="utf-8")
    return path.resolve()


def places(tmp_path):
    """A group's folder as the current folder, the export in a folder of its own, the video in a third."""
    here, tracks, videos = tmp_path / "repo", tmp_path / "tracks", tmp_path / "videos"
    for folder in (here, videos / "day1"):
        folder.mkdir(parents=True)
    clip, export = one_disk(videos / "day1")
    moved = tracks / "clip" / "ana" / "A.csv"
    moved.parent.mkdir(parents=True)
    export.replace(moved)
    export.parent.rmdir()
    return here, tracks, videos, clip, moved


def fps_of(result):
    """The `time` block of the session, the `time:` line of this run in run.log (the log is added to by
    every run into the folder: the last one), and frame / t of the Tracker-format file."""
    time = session_json(result.run_folder)["time"]
    line = [line for line in (result.run_folder / "run.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("time:")][-1]
    track = read_track(result.files[0])  # t = frame / fps_true, and no frame here is frame 0
    return time, line, float(np.median(track["frame"] / track["t"]))


def test_fps_sources_in_their_order(tmp_path, monkeypatch):
    here, tracks, videos, clip, export = places(tmp_path)
    monkeypatch.chdir(here)
    in_cwd, by_export, by_video = manifest(here, 238.0), manifest(tracks, 237.0), manifest(videos, 236.0)

    time, line, in_file = fps_of(run(clip, export, fps=239.6))  # typed in: before every manifest
    assert time == {"fps_true": 239.6, "source": "typed", "manifest_path": None, "stopwatch": None}
    assert "239.6" in line and "source: typed" in line and in_file == pytest.approx(239.6)

    for found, fps in ((in_cwd, 238.0), (by_export, 237.0), (by_video, 236.0)):
        lines = []
        time, line, in_file = fps_of(run(clip, export, lines))
        assert time == {"fps_true": fps, "source": "manifest", "manifest_path": str(found), "stopwatch": None}
        assert "source: manifest" in line and str(found) in line and in_file == pytest.approx(fps)
        assert any(str(found) in said for said in lines)  # the console names the manifest too
        found.unlink()  # the next one in the order is found now

    lines = []
    time, line, in_file = fps_of(run(clip, export, lines))  # no manifest left: Tracker's t column, 240
    assert (time["source"], time["manifest_path"]) == ("tracker-export", None)
    assert time["fps_true"] == pytest.approx(FPS, rel=1e-5) and in_file == pytest.approx(FPS, rel=1e-5)
    assert "source: tracker-export" in line
    assert any("t column" in said for said in lines)  # the printed note


def test_a_manifest_row_without_a_usable_fps_is_no_source(tmp_path, monkeypatch):
    here, tracks, videos, clip, export = places(tmp_path)
    monkeypatch.chdir(here)
    manifest(here, "")  # the row is there, the cell is empty
    manifest(tracks, 0)
    time, _, in_file = fps_of(run(clip, export))
    assert time["source"] == "tracker-export" and in_file == pytest.approx(FPS, rel=1e-5)


@pytest.mark.parametrize("fps", [0.0, -240.0, float("nan"), float("inf")])
def test_a_typed_fps_must_be_a_positive_number(tmp_path, fps):
    clip, export = one_disk(tmp_path)
    with pytest.raises(ValueError, match="--fps"):
        run(clip, export, fps=fps)
    assert names_in(tmp_path) == ["ana", "clip_tracker.mp4"]


def test_a_start_file_without_any_fps_is_last_weeks_error(tmp_path):
    clip, export = two_disks(tmp_path)  # one row: its t column gives no frame rate
    with pytest.raises(ValueError, match=re.escape("Unknown fps_true: give it with --fps (e.g. --fps 239.6)")):
        run(clip, export)
    assert not (tmp_path / "clip_tracker_outline_ana").exists()


def test_an_fps_below_100_is_a_warning_and_still_runs(tmp_path):
    clip, export = one_disk(tmp_path)
    lines = []
    result = run(clip, export, lines, fps=30.0)
    (warning,) = [line for line in lines if "WARNING" in line]
    assert "fps_true = 30" in warning and "below 100" in warning
    got = read_track(result.files[0])
    assert list(got["frame"]) == [40, 44, 48, 52, 56, 60] and np.allclose(got["t"], got["frame"] / 30.0)
    lines = []
    run(clip, export, lines, fps=100.0)
    assert not [line for line in lines if "WARNING" in line]


# --------------------------------------------------------------------------- the console

def test_the_console_lines_are_last_weeks(tmp_path):
    clip, export = one_disk(tmp_path, frames=range(40, 161, 4), n=200)
    lines = []
    result = run(clip, export, lines, fps=FPS, overlay=True)
    # 31 frames at step 4 are 124 frames of video: 0.5 s at 240 frames per s; 0.05 mm per px is 50 um
    assert lines[0] == (f"{MODEL}: 1 shrimp (A), frames 40-160 every 4 (31 frames, 0.5 s at fps_true = 240); "
                        "scale 50.00 um per pixel")
    progress = [line for line in lines if line.startswith("  frame ")]
    assert len(progress) == 3  # after frames 1 and 5 and at the end, as last week
    for line, done in zip(progress, (1, 5, 31)):
        assert re.fullmatch(rf"  frame {done}/31: \d+\.\d\d s per frame, about \d+ min left", line)
    assert lines[-2] == f"  saved {result.files[0]} and {result.overlay}"
    assert str(result.run_folder) in lines[-1] and "positions.csv" in lines[-1]
    assert not [line for line in lines if "loading the model" in line or "running on" in line]  # none was loaded
    assert all(line.isascii() for line in lines if str(tmp_path) not in line)


def test_a_calibration_that_does_not_fit_is_last_weeks_warning(tmp_path):
    clip, export = one_disk(tmp_path)
    text = export.read_text().splitlines()
    cells = text[4].split(",")
    cells[2] = "9.000000E0"  # one x that no single map explains: 9 mm
    text[4] = ",".join(cells)
    export.write_text("\n".join(text) + "\n")
    lines = []
    run(clip, export, lines)
    (warning,) = [line for line in lines if "WARNING" in line]
    assert warning.startswith("  WARNING: the pixel and mm columns of A.csv do not fit one calibration (rms ")
    assert warning.endswith(" um). Were they exported from the same .trk?")
