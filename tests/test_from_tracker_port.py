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

The fourth compares what `from_tracker` writes with the frozen files of the same three cases,
tests/data/tracker_format/<case>/<name>.csv, and needs nothing of the template. The third is what
writes them, only when asked and only after the template and the package agreed in that run
(tests/frozen_helpers.py). The plan and the CHECK messages of each case are typed values
(`REPORTED`): the fourth holds the tool's run to them, and the third the template's.

The frozen bytes belong to the decoder that made them. The positions come from a clip that OpenCV
encodes and decodes on the computer that runs the test, and another build of OpenCV gives other digits
at a few frames. So the bytes are asserted only where the decoder is the one that the files' HEADER.txt
names (`same_decoder`). There the third test is their independent check: the template's bytes equal
the frozen bytes too. A frozen file has LF line ends; the tool writes the system's line ends (CRLF on
Windows, docs/OUTPUTS.md), so the frozen bytes are compared after LF is replaced by those. With every
other decoder the fourth test asserts what does not depend on the decoder (`compare_with_golden`): the
text that does not come from pixels is the frozen file's, every position is within 0.25 px of the
disk's true center, and the mm columns follow from the pixel columns.

Three tests after the fourth: that second comparison is run on every system, so on the one that froze
the files too; the frozen positions themselves are held to the true centers, without a clip; and
changed copies of a frozen file, made up in the test's own folder, show that each difference is found.

Coordinates: px in Tracker's convention (pixel centers at +0.5); mm in Tracker's axes, y up; a disk
drawn at (x, y) by `disk_video` is at (x + 0.5, y + 0.5) px. Frames are video frame numbers.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest
from conftest import java_sci
from from_tracker_helpers import FPS, disk_video, write_start_file
from frozen_helpers import (GOLDEN, SIDECAR, compare_with_golden, frozen_files, read_frozen, same_decoder,
                            write_frozen, write_listing)
from test_tracker_io import MM_PER_PX, export_text, tracker_map

from outline_tracker.from_tracker import from_tracker
from outline_tracker.segmenter.fake import ThresholdFake
from outline_tracker.video import decoder_tag

FREEZE_COMMAND = "OUTLINE_TRACKER_FREEZE=1 uv run pytest tests/test_from_tracker_port.py -q"  # what writes them


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

# Where each of the three cases below draws its disks, by the name of the case and of the track:
# frame -> (x, y) of the disk's center as `disk_video` takes it (px, pixel centers at whole numbers), or
# None on a frame without that disk
DISKS = {
    "_one_track": {"A": lambda n: (60 + 0.5 * n, 100)},
    "_start_file_and_a_lost_disk": {"A": lambda n: (60 + 0.5 * n, 60),
                                    "B": lambda n: (250 - 0.25 * n, 170) if n < 100 else None},
    "_every_frame_and_a_jump": {"A": lambda n: (60 + 0.5 * n + (12 if n >= 25 else 0), 100)},
}


def drawn(case):
    """What `disk_video` takes for a case: frame -> the centers of the disks that the frame shows."""
    disks = DISKS[case.__name__].values()
    return lambda n: [center for center in (disk(n) for disk in disks) if center is not None]


def true_center(case, name):
    """Where a case's clip has the disk of the track `name`: frame -> (pixelx, pixely) of its true
    center, px in Tracker's convention, which is where the case draws it (`DISKS`) plus 0.5; None on a
    frame that does not show it."""
    def center(frame):
        at = DISKS[case.__name__][name](frame)
        return None if at is None else (at[0] + 0.5, at[1] + 0.5)
    return center


def _one_track(folder):
    """The template's whole-run case: one disk, its Tracker track on every 4th frame from 40 to 160."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, drawn(_one_track))
    export = folder / "A.csv"
    export.write_text(export_text([((f - 40) / 240.0, f, 60.5 + 0.5 * f, 100.5) for f in range(40, 161, 4)]))
    return video, export, {}, ["A"]


def _start_file_and_a_lost_disk(folder):
    """Two disks from a start file; B is gone from frame 100 on, so its rows from there are empty."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, drawn(_start_file_and_a_lost_disk))
    export = write_start_file(folder / "ana" / "extra" / "start.csv", {"A": (80.5, 60.5), "B": (240.5, 170.5)})
    return video, export, {"seconds": 100 / 240, "step": 4, "fps": 240.0}, ["A", "B"]


def _every_frame_and_a_jump(folder):
    """One disk marked on every frame from 10 to 40; from frame 25 on it is 12 px further right."""
    video = folder / "clip_tracker.mp4"
    disk_video(video, drawn(_every_frame_and_a_jump), n=60)
    export = folder / "A.csv"
    export.write_text(export_text([((f - 10) / 240.0, f, 60.5 + 0.5 * f + (12 if f >= 25 else 0), 100.5)
                                   for f in range(10, 41)]))
    return video, export, {}, ["A"]


CASES = [_one_track, _start_file_and_a_lost_disk, _every_frame_and_a_jump]

# What a run of each case reports besides its files: the plan as (start, step, n), and its CHECK
# messages as patterns that a whole message must fit.
# - `_one_track`: marked on every 4th frame from 40 to 160: (160 - 40) / 4 + 1 = 31 frames;
# - `_start_file_and_a_lost_disk`: marked on frame 40, tracked with step 4 for 100 / 240 s at 240 frames
#   per s: 100 / 4 = 25 frames, 40 to 136. B is gone from frame 100 on: the 10 frames 100 to 136, the
#   first at t = 100 / 240 = 0.417 s;
# - `_every_frame_and_a_jump`: marked on every frame from 10 to 40: 31 frames. From frame 25 on the disk
#   is 12 px further right, so it moves 12.5 px in that one frame: 0.625 mm at 0.05 mm per px, at
#   t = 25 / 240 = 0.104 s. The stand-in finds each center within 0.25 px: 0.60 to 0.65 mm.
REPORTED = {
    "_one_track": ((40, 4, 31), []),
    "_start_file_and_a_lost_disk": ((40, 4, 25), [r"B: lost in 10 of 25 frames \(first at t = 0\.417 s\)"]),
    "_every_frame_and_a_jump": ((10, 1, 31),
                                [r"A: jumps 0\.6[0-5] mm at t = 0\.104 s \(frame 25\): check the video there"]),
}


def assert_the_plan_and_the_messages(case, plan, flags) -> None:
    """A run of a case reported what `REPORTED` holds for it: the plan's start, step and n, 240
    frames per s, and its CHECK messages, one for each pattern and in that order."""
    start_step_n, messages = REPORTED[case.__name__]
    assert (plan.start, plan.step, plan.n) == start_step_n
    # Two of the cases take fps_true from the times of their export, which have 7 digits there
    # (`java_sci`): 240 to a few parts in a million, not to the last bit.
    assert plan.fps == pytest.approx(FPS, rel=1e-5)
    assert len(flags) == len(messages) and all(map(re.fullmatch, messages, flags)), flags


def as_written_here(frozen: bytes) -> bytes:
    """The bytes of a frozen Tracker-format file (LF line ends) with this system's line ends, which is
    how `write_tracker_file` writes it here."""
    return frozen.replace(b"\n", os.linesep.encode())


def golden_decoder_here() -> bool:
    """Whether this computer's decoder is the one that froze the golden files (`same_decoder`): only
    then are their bytes asserted. Read when asked, because a run that freezes writes the header."""
    return same_decoder(read_frozen(GOLDEN / SIDECAR)[0], decoder_tag())


def run_and_compare(folder, case, strict: bool):
    """Run a case with the stand-in model in `folder` and judge its Tracker-format files against the
    case's golden files: the names, then each file by its bytes when `strict` (this is the decoder that
    froze them), and by what holds with every decoder otherwise (`compare_with_golden`). Returns what
    `from_tracker` returned."""
    video, export, options, names = case(folder)
    new = from_tracker(video, export, model="stand-in", segmenter=ThresholdFake(), overlay=False,
                       out=folder / "new", log=lambda *a: None, **options)
    frozen = GOLDEN / case.__name__
    assert [f.name for f in new.files] == [f.name for f in frozen_files(frozen)] == [f"{name}.csv" for name in names]
    for new_file in new.files:
        if strict:
            assert new_file.read_bytes() == as_written_here((frozen / new_file.name).read_bytes()), new_file.name
        else:
            compare_with_golden(f"{case.__name__}, {new_file.name}", new_file, frozen / new_file.name,
                                true_center(case, new_file.stem), tracker_map)
    return new


@pytest.mark.parametrize("case", CASES)
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
    # the typed values of the next test, proved here against the template's own run
    assert_the_plan_and_the_messages(case, old["plan"], old["flags"])
    # The template and the package agree. Only now, and only when asked, their bytes are frozen (with LF).
    frozen = GOLDEN / case.__name__
    for new_file in new.files:
        write_frozen(frozen / new_file.name, new_file.read_bytes().replace(os.linesep.encode(), b"\n"))
    write_listing(GOLDEN, FREEZE_COMMAND, {
        "agreement with the template": "the same bytes, file by file, asserted in the run that wrote them",
        "line ends": "LF; the tool writes the system's line ends, and a test compares after putting those in"})
    # The independent check of the frozen files, with the decoder that froze them: the template writes
    # their bytes. With another decoder the template's bytes are still this run's (asserted above), and
    # this run's are judged by the next test.
    if golden_decoder_here():
        for old_file in old["files"]:
            assert old_file.read_bytes() == as_written_here((frozen / old_file.name).read_bytes()), old_file.name


@pytest.mark.parametrize("case", CASES)
def test_the_tracker_format_files_are_the_frozen_bytes(tmp_path, case):
    # the bytes with the decoder that froze them; with another, the fixed text and the true centers
    new = run_and_compare(tmp_path, case, strict=golden_decoder_here())
    assert_the_plan_and_the_messages(case, new.plan, new.flags)


@pytest.mark.parametrize("case", CASES)
def test_with_another_decoder_the_files_are_judged_by_the_fixed_text_and_the_true_centers(tmp_path, case):
    # The comparison for another decoder does not depend on the decoder, so it holds here too. On the
    # computer that froze the files the test above takes the bytes; this one asks for the other comparison.
    run_and_compare(tmp_path, case, strict=False)


def test_the_frozen_positions_are_within_a_quarter_px_of_the_true_centers(tmp_path):
    # No clip and no run: each frozen file, with this system's line ends, is judged as a run's file is
    # with another decoder. Its text is its own, so what is asserted is the geometry: the pixel columns
    # against the true centers, and the mm columns against the pixel columns.
    for case in CASES:
        for golden in frozen_files(GOLDEN / case.__name__):
            what, center = f"{case.__name__}, {golden.name}", true_center(case, golden.stem)
            copy = tmp_path / case.__name__ / golden.name
            copy.parent.mkdir(exist_ok=True)
            copy.write_bytes(as_written_here(golden.read_bytes()))
            compare_with_golden(what, copy, golden, center, tracker_map)
            # and a row is lost on the frames that do not show the disk, on no other
            track = pd.read_csv(golden, skiprows=1)
            assert list(track["pixelx"].isna()) == [center(frame) is None for frame in track["frame"]], what


def test_with_another_decoder_each_difference_from_the_frozen_file_is_found(tmp_path):
    # Made-up files in this test's folder: B of the case with the lost disk, with one change each. The
    # clip has B at (250.5 - 0.25 f, 170.5) px on frame f up to frame 99 and not from frame 100 on; the
    # case's map is 0.05 mm per px from (160, 120) px with y up, so B is at y = -0.05 (170.5 - 120) mm.
    case = _start_file_and_a_lost_disk
    golden = GOLDEN / case.__name__ / "B.csv"
    frozen = golden.read_bytes()
    found = b"0.2000000,48,3.925000,-2.525000,238.500,170.500\n"  # frame 48: the true center, (238.5, 170.5) px
    lost = b"0.4166667,100,,,,\n"  # frame 100: the first row without the disk
    assert frozen.count(found) == 1 and frozen.count(lost) == 1

    def judge(old, new, golden=golden, line_end=os.linesep.encode()):
        """Judge the frozen file with `old` replaced by `new`, written with `line_end`, against `golden`."""
        written = tmp_path / "written.csv"
        written.write_bytes(frozen.replace(old, new).replace(b"\n", line_end))
        compare_with_golden("made up", written, golden, true_center(case, "B"), tracker_map)

    def row(cells: str) -> bytes:
        return b"0.2000000,48," + cells.encode("ascii") + b"\n"

    judge(found, found)  # no change: this is the file that the test above judges

    # a position 0.2 px to the right of the true center, with x 0.2 * 0.05 = 0.010 mm further: within
    # the limit, though not the frozen bytes
    judge(found, row("3.935000,-2.525000,238.700,170.500"))
    # 0.3 px to the right, with x 0.015 mm further: over the limit of 0.25 px
    with pytest.raises(AssertionError, match=r"frame 48: 0\.300 px from the true center"):
        judge(found, row("3.940000,-2.525000,238.800,170.500"))
    # 238.500 stands for 238.4995 to 238.5005 px, which the map takes to 3.924975 to 3.925025 mm: an x
    # one step of its last decimal beyond that is not this position's x, and the same for y
    judge(found, row("3.925025,-2.525000,238.500,170.500"))
    with pytest.raises(AssertionError, match=r"frame 48: x = 3\.925026 mm is not what the calibration gives"):
        judge(found, row("3.925026,-2.525000,238.500,170.500"))
    judge(found, row("3.925000,-2.525025,238.500,170.500"))
    with pytest.raises(AssertionError, match=r"frame 48: y = -2\.525026 mm is not what the calibration gives"):
        judge(found, row("3.925000,-2.525026,238.500,170.500"))

    # a lost row made found, where the disk would be on frame 100, (225.5, 170.5) px; a found row made lost
    with pytest.raises(AssertionError, match="frame 100: found here, lost in the golden file"):
        judge(lost, b"0.4166667,100,3.275000,-2.525000,225.500,170.500\n")
    with pytest.raises(AssertionError, match="frame 48: lost here, found in the golden file"):
        judge(found, row(",,,"))
    # the text that does not come from pixels: a t cell, a frame cell, the name line, a row less
    with pytest.raises(AssertionError, match="frame 48: t and frame are"):
        judge(found, found.replace(b"0.2000000,48,", b"0.2000001,48,"))
    with pytest.raises(AssertionError, match="frame 48: t and frame are"):
        judge(found, found.replace(b"0.2000000,48,", b"0.2000000,49,"))
    with pytest.raises(AssertionError, match="the two lines above the rows"):
        judge(b",B,,,,,\n", b",C,,,,,\n")
    with pytest.raises(AssertionError, match="24 rows, the golden file has 25"):
        judge(lost, b"")
    # a number with another count of decimals: one more for pixelx, one fewer for x, an exponent for y
    for cells in ("3.925000,-2.525000,238.5000,170.500", "3.92500,-2.525000,238.500,170.500",
                  "3.925000,-2.525e+00,238.500,170.500"):
        with pytest.raises(AssertionError, match="frame 48: other decimals than the golden row"):
            judge(found, row(cells))
    # the line ends of the other kind of system, and a file without its last line end
    with pytest.raises(AssertionError, match="line ends"):
        judge(found, found, line_end=b"\n" if os.linesep == "\r\n" else b"\r\n")
    with pytest.raises(AssertionError, match="line ends"):
        judge(b"0.5666667,136,,,,\n", b"0.5666667,136,,,,")

    # A golden file that held such a row would not pass the test above, whatever a run gave: the
    # 0.3 px, and a found row on a frame without the disk.
    shifted = tmp_path / "shifted.csv"
    shifted.write_bytes(frozen.replace(found, row("3.940000,-2.525000,238.800,170.500")))
    with pytest.raises(AssertionError, match=r"frame 48: 0\.300 px from the true center"):
        judge(found, row("3.940000,-2.525000,238.800,170.500"), golden=shifted)
    no_disk = tmp_path / "no_disk.csv"
    no_disk.write_bytes(frozen.replace(lost, b"0.4166667,100,3.275000,-2.525000,225.500,170.500\n"))
    with pytest.raises(AssertionError, match="frame 100: found, but the clip does not show"):
        judge(lost, b"0.4166667,100,3.275000,-2.525000,225.500,170.500\n", golden=no_disk)


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
