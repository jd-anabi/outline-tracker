"""Last week's end-to-end tests, through `from_tracker` (decision X3), and last week's script itself
as the judge of the Tracker-format files (SPEC 13.3).

The first two tests are the template's (tests/reference/template_tests/test_segment.py). Their
assertions on frames, times and positions are unchanged; `from_tracker` with `ThresholdFake` stands
where `segment.track_video` with `DiskFinder` stood, and the files are looked for in the run folder of
SPEC 8.1. tests/test_port_fidelity.py lists every changed piece of text and compares the rest with the
template.

The third runs the reference `track_video` with the reference `DiskFinder`, and `from_tracker` with
`ThresholdFake`, on the same clip and export: the Tracker-format files must be the same bytes on this
computer, and the CHECK messages the same list.

Coordinates: px in Tracker's convention (pixel centers at +0.5); mm in Tracker's axes, y up; a disk
drawn at (x, y) by `disk_video` is at (x + 0.5, y + 0.5) px. Frames are video frame numbers.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import java_sci
from from_tracker_helpers import disk_video, write_start_file
from test_tracker_io import MM_PER_PX, export_text, tracker_map

from outline_tracker.from_tracker import from_tracker
from outline_tracker.segmenter.fake import ThresholdFake


def test_whole_run_with_a_stand_in_model(tmp_path):
    # a dark disk moving 0.5 px per frame to the right; frame n shows it at x = 60 + 0.5 n (pixel centers)
    video = tmp_path / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 100)])
    export = tmp_path / "A.csv"
    frames = range(40, 161, 4)  # tracked in Tracker every 4th frame from frame 40
    export.write_text(export_text([((f - 40) / 240.0, f, 60.5 + 0.5 * f, 100.5) for f in frames]))
    res = from_tracker(video, export, model="stand-in", segmenter=ThresholdFake(), log=lambda *a: None)
    got = pd.read_csv(res.files[0], skiprows=1)
    assert list(got["frame"]) == list(frames), "same frames as the Tracker track"
    assert np.allclose(got["t"], got["frame"] / 240.0)
    ex, ey = tracker_map(60.5 + 0.5 * got["frame"], 100.5)
    assert np.allclose(got["x"], ex, atol=0.1 * MM_PER_PX), "x in mm, from Tracker's calibration"
    assert np.allclose(got["y"], ey, atol=0.1 * MM_PER_PX)
    assert res.overlay.exists() and res.overlay.stat().st_size > 0
    # SPEC 8.1: the files are in the run folder, <video folder>/<video stem>_outline_<student>/; the
    # student is the name of the export's folder, and the tracks are in a folder named after the model
    run = tmp_path / f"clip_tracker_outline_{tmp_path.name}"
    assert (run / "run.log").exists()  # not .txt: load_tracks reads .csv and .txt
    assert res.files == [run / "stand-in" / "A.csv"] and res.overlay == run / "overlay.mp4"
    assert res.flags == []


def test_many_shrimp_from_a_start_file_in_extra(tmp_path):
    # two disks; both marked once on frame 40 and exported together into ana/extra/start.csv
    video = tmp_path / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 60), (250 - 0.25 * n, 170)])
    start = tmp_path / "ana" / "extra" / "start.csv"
    start.parent.mkdir(parents=True)
    rows = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    cells = [java_sci(0.0)]
    for px, py in [(60.5 + 20, 60.5), (250.5 - 10, 170.5)]:
        cells += [java_sci(float(v)) for v in (40, *tracker_map(px, py), px, py)]
    rows.append(",".join(cells) + ",")
    start.write_text("\n".join(rows) + "\n")
    res = from_tracker(video, start, model="stand-in", seconds=100 / 240, step=4, fps=240.0,
                       segmenter=ThresholdFake(), overlay=False, log=lambda *a: None)
    # SPEC 8.1: the run folder is next to the video, <video stem>_outline_<student>/; the student is the
    # folder that holds extra/, as last week, and the tracks are in a folder named after the model
    assert [f.relative_to(tmp_path).as_posix() for f in res.files] == [
        "clip_tracker_outline_ana/stand-in/A.csv", "clip_tracker_outline_ana/stand-in/B.csv"]
    for f, (x0, vx, y0) in zip(res.files, [(60.5, 0.5, 60.5), (250.5, -0.25, 170.5)]):
        got = pd.read_csv(f, skiprows=1)
        assert list(got["frame"]) == list(range(40, 140, 4))
        ex, ey = tracker_map(x0 + vx * got["frame"], y0)
        # thresholding a slowly moving disk puts its center within ~0.2 px of the truth
        assert np.allclose(got["x"], ex, atol=0.25 * MM_PER_PX) and np.allclose(got["y"], ey, atol=0.25 * MM_PER_PX)


# ---------------------------------------------------------------------------------------------
# Last week's script on the same clip and export


def _one_track(folder):
    """The template's whole-run case: one disk, its Tracker track on every 4th frame from 40 to 160."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 100)])
    export = folder / "A.csv"
    export.write_text(export_text([((f - 40) / 240.0, f, 60.5 + 0.5 * f, 100.5) for f in range(40, 161, 4)]))
    return video, export, {}, ["A"]


def _start_file_and_a_lost_disk(folder):
    """Two disks from a start file; B is gone from frame 100 on, so its rows from there are empty."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 60)] + ([(250 - 0.25 * n, 170)] if n < 100 else []))
    export = write_start_file(folder / "ana" / "extra" / "start.csv", {"A": (80.5, 60.5), "B": (240.5, 170.5)})
    return video, export, {"seconds": 100 / 240, "step": 4, "fps": 240.0}, ["A", "B"]


def _every_frame_and_a_jump(folder):
    """One disk marked on every frame from 10 to 40; from frame 25 on it is 12 px further right."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n + (12 if n >= 25 else 0), 100)], n=60)
    export = folder / "A.csv"
    export.write_text(export_text([((f - 10) / 240.0, f, 60.5 + 0.5 * f + (12 if f >= 25 else 0), 100.5)
                                   for f in range(10, 41)]))
    return video, export, {}, ["A"]


@pytest.mark.parametrize("case", [_one_track, _start_file_and_a_lost_disk, _every_frame_and_a_jump])
def test_the_tracker_format_files_are_last_weeks_bytes(tmp_path, case):
    from shrimp import segment as reference
    from template_tests.test_segment import DiskFinder

    video, export, options, names = case(tmp_path)
    old = reference.track_video(video, export, model="stand-in", segmenter=DiskFinder(), overlay=False,
                                out=tmp_path / "reference", manifest=tmp_path / "none.csv", log=lambda *a: None,
                                **options)
    new = from_tracker(video, export, model="stand-in", segmenter=ThresholdFake(), overlay=False,
                       out=tmp_path / "new", log=lambda *a: None, **options)
    assert [f.name for f in new.files] == [f.name for f in old["files"]] == [f"{name}.csv" for name in names]
    for new_file, old_file in zip(new.files, old["files"]):
        assert new_file.parent == tmp_path / "new" / "stand-in"
        assert new_file.read_bytes() == old_file.read_bytes(), new_file.name
    assert new.flags == old["flags"]
    assert (new.plan.start, new.plan.step, new.plan.n, new.plan.fps) == (
        old["plan"].start, old["plan"].step, old["plan"].n, old["plan"].fps)


def test_the_check_messages_are_last_weeks(tmp_path):
    # B is gone from frame 100 on: of the 25 frames 40, 44, ..., 136 the 10 frames 100 to 136 are lost,
    # the first at t = 100 / 240 s
    video, export, options, _ = _start_file_and_a_lost_disk(tmp_path)
    lines = []
    new = from_tracker(video, export, model="stand-in", segmenter=ThresholdFake(), overlay=False, log=lines.append,
                       **options)
    assert new.flags == ["B: lost in 10 of 25 frames (first at t = 0.417 s)"]
    assert [line for line in lines if "CHECK" in line] == ["  CHECK: B: lost in 10 of 25 frames (first at t = 0.417 s)"]
    lost = pd.read_csv(new.files[1], skiprows=1).set_index("frame")
    assert lost.loc[100:, ["x", "y", "pixelx", "pixely"]].isna().all().all()  # the rows stay, empty
    assert lost.loc[:96, ["x", "y", "pixelx", "pixely"]].notna().all().all()


def test_a_jump_is_a_check_message_as_last_week(tmp_path):
    # 12.5 px in one frame at 0.05 mm per px and 240 frames per s is 150 mm/s, above last week's 100 mm/s
    video, export, options, _ = _every_frame_and_a_jump(tmp_path)
    new = from_tracker(video, export, model="stand-in", segmenter=ThresholdFake(), overlay=False,
                       log=lambda *a: None, **options)
    (message,) = new.flags
    assert message.startswith("A: jumps 0.6")  # 12.5 px are 0.625 mm
    assert message.endswith(" mm at t = 0.104 s (frame 25): check the video there")  # 25 / 240 s
