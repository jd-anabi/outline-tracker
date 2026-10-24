"""Tests of outline_tracker/core/geometry.py: the frame grid and the manifest of a class (SPEC 3.4, 4.1, 4.4).

Expected values come from SPEC 3.4 worked by hand, from brute force over the definition of the grid,
and from the manifest files that each test writes. Nothing is copied from the output of the code under
test.

Frames are video frame numbers counted from 0; fps_true is in frames per second and dish_mm in mm.

The rest of the tests of the module are in tests/core/test_geometry.py (the transform, its
covariances, its tie to Tracker's calibration) and tests/core/test_geometry_calibration.py (the stick,
the tape check, the stopwatch, the circle fit).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from outline_tracker.core import geometry, tracker_io


# --------------------------------------------------------------------------- the frame grid (SPEC 3.4, X20)

def test_grid_frames():
    assert list(geometry.grid_frames(96, 104, 2)) == [96, 98, 100, 102, 104]
    assert list(geometry.grid_frames(96, 105, 2)) == [96, 98, 100, 102, 104]  # the end is off the grid
    assert list(geometry.grid_frames(96, 110, 5)) == [96, 101, 106]
    assert list(geometry.grid_frames(7, 7, 3)) == [7]
    grid = geometry.grid_frames(0, 2399, 2)  # the clip of SPEC 8.10
    assert (grid[0], grid[-1], len(grid)) == (0, 2398, 1200)


@pytest.mark.parametrize("frame, expected", [
    (101, (102, True)),
    (103, (104, True)),
    (100, (100, False)),
    (90, (96, True)),
    (201, (200, True)),
    (96, (96, False)),
    (200, (200, False)),
    (199, (200, True)),
])
def test_snap_forward_step_2(frame, expected):
    assert geometry.snap_to_grid(frame, 96, 2, 200) == expected


@pytest.mark.parametrize("frame, expected", [
    (97, (101, True)),
    (101, (101, False)),
    (196, (196, False)),  # the last grid frame: 96 + 20 * 5
    (197, (196, True)),  # no grid frame at or after 197 inside the clip
    (200, (196, True)),
    (300, (196, True)),
])
def test_snap_forward_step_5(frame, expected):
    assert geometry.snap_to_grid(frame, 96, 5, 200) == expected


@pytest.mark.parametrize("step", [1, 2, 3, 5, 7, 200])
def test_snap_against_brute_force(step):
    start, end = 96, 200
    grid = [f for f in range(start, end + 1) if (f - start) % step == 0]  # SPEC 3.4, written out
    for frame in range(80, 215):
        later = [g for g in grid if g >= frame]
        want = later[0] if later else grid[-1]
        got, moved = geometry.snap_to_grid(frame, start, step, end)
        assert got == want, frame
        assert moved is (want != frame), frame
        assert type(got) is int


def test_snap_accepts_numpy_integers():
    assert geometry.snap_to_grid(np.int64(101), np.int64(96), np.int64(2), np.int64(200)) == (102, True)


@pytest.mark.parametrize("step", [0, -2])
def test_step_must_be_at_least_one(step):
    with pytest.raises(ValueError, match="step"):
        geometry.grid_frames(96, 200, step)
    with pytest.raises(ValueError, match="step"):
        geometry.snap_to_grid(100, 96, step, 200)


def test_end_must_not_be_before_start():
    with pytest.raises(ValueError, match="end"):
        geometry.grid_frames(200, 96, 2)
    with pytest.raises(ValueError, match="end"):
        geometry.snap_to_grid(100, 200, 2, 96)


def test_frames_must_be_whole_numbers():
    with pytest.raises(ValueError, match="whole number"):
        geometry.grid_frames(96, 200, 2.5)
    with pytest.raises(ValueError, match="whole number"):
        geometry.snap_to_grid(100.5, 96, 2, 200)


# --------------------------------------------------------------------------- manifest (SPEC 4.1, 4.4, X9)

HEADER = "video_file,group,fps_true,stick_mm,check_mm,dish_mm\n"
VIDEO_NAME = "groupB_2026-10-06_1325_main_tracker.mp4"
ROW = "groupB_2026-10-06_1325_main.MOV,B,{fps},30,20,{dish}\n"


def write_manifest(folder: Path, fps="239.6", dish="35", rows=None) -> Path:
    """Write <folder>/data/manifest.csv with one row for VIDEO_NAME's original (or the given rows)."""
    path = folder / "data" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + (ROW.format(fps=fps, dish=dish) if rows is None else rows), encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path):
    """Three places a manifest can be: the current folder, above the export, above the video."""
    cwd = tmp_path / "here"
    export = tmp_path / "group repo" / "ana" / "tracks" / "A.csv"
    video = tmp_path / "vidéos" / "day 1" / VIDEO_NAME
    for folder in (cwd, export.parent, video.parent):
        folder.mkdir(parents=True)
    return cwd, export, video


def test_fps_from_manifest_is_the_ported_function(tmp_path):
    assert geometry.fps_from_manifest is tracker_io._fps_from_manifest
    manifest = write_manifest(tmp_path, fps="238.0")
    assert geometry.fps_from_manifest(tmp_path / VIDEO_NAME, manifest) == pytest.approx(238.0)


def test_manifest_in_the_current_folder_comes_first(tree):
    cwd, export, video = tree
    here = write_manifest(cwd)
    write_manifest(export.parents[2])
    write_manifest(video.parents[1])
    found = geometry.find_manifest(video, export=export, cwd=cwd)
    assert found.is_absolute()
    assert found.samefile(here)


def test_then_upward_from_the_export(tree):
    cwd, export, video = tree
    above_export = write_manifest(export.parents[2])  # "group repo"/data/manifest.csv, two folders up
    write_manifest(video.parents[1])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_export)


def test_then_upward_from_the_video(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_video)
    assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)  # no export given


def test_upward_search_starts_in_the_files_own_folder(tree):
    cwd, export, video = tree
    beside_export = write_manifest(export.parent)
    write_manifest(export.parents[2])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(beside_export)


def test_no_manifest_gives_none(tree):
    cwd, export, video = tree
    assert geometry.find_manifest(video, export=export, cwd=cwd) is None


def test_current_folder_is_not_searched_upward(tree):
    cwd, export, video = tree
    write_manifest(cwd)
    deeper = cwd / "sub"  # last week's rule looks in the current folder itself, not above it
    deeper.mkdir()
    assert geometry.find_manifest(video, export=export, cwd=deeper) is None


def test_a_manifest_without_this_video_is_passed_over(tree):
    cwd, export, video = tree
    write_manifest(cwd, rows="another_clip.MOV,B,120.0,30,20,35\n")  # exists, but no row for the video
    above_export = write_manifest(export.parents[2])
    assert geometry.find_manifest(video, export=export, cwd=cwd).samefile(above_export)


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX folder permissions")
def test_a_data_folder_that_cannot_be_opened_is_passed_over(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    locked = video.parent / "data"  # nearer to the video than the manifest, and closed to us
    locked.mkdir()
    locked.chmod(0)
    try:
        if os.access(locked, os.X_OK):
            pytest.skip("folder permissions are not enforced for this user")
        assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)
        assert geometry.dish_mm_from_manifest(video, locked / "manifest.csv") is None
    finally:
        locked.chmod(0o755)


def test_a_folder_named_like_the_manifest_is_passed_over(tree):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    (video.parent / "data" / "manifest.csv").mkdir(parents=True)
    assert geometry.find_manifest(video, cwd=cwd).samefile(above_video)


def test_default_current_folder_and_relative_paths(tree, monkeypatch):
    cwd, export, video = tree
    above_video = write_manifest(video.parents[1])
    monkeypatch.chdir(video.parent)
    assert geometry.find_manifest(VIDEO_NAME).samefile(above_video)  # a bare file name
    here = write_manifest(video.parent)
    assert geometry.find_manifest(VIDEO_NAME).samefile(here)  # ./data/manifest.csv


def test_dish_mm_from_manifest(tmp_path):
    manifest = write_manifest(tmp_path, dish="34.6")
    assert geometry.dish_mm_from_manifest(tmp_path / VIDEO_NAME, manifest) == pytest.approx(34.6)
    # the original video's name works too (the ported rule drops "_tracker" and the extension)
    assert geometry.dish_mm_from_manifest(Path("groupB_2026-10-06_1325_main.MOV"), manifest) == pytest.approx(34.6)


@pytest.mark.parametrize("dish", ["", "about 35", "0", "-35", "nan", "inf"])
def test_dish_mm_that_is_not_a_diameter_gives_none(tmp_path, dish):
    manifest = write_manifest(tmp_path, dish=dish)
    assert geometry.dish_mm_from_manifest(tmp_path / VIDEO_NAME, manifest) is None


def test_dish_mm_without_file_column_or_row_gives_none(tmp_path):
    video = tmp_path / VIDEO_NAME
    assert geometry.dish_mm_from_manifest(video, tmp_path / "none.csv") is None
    manifest = write_manifest(tmp_path)
    assert geometry.dish_mm_from_manifest(tmp_path / "another_tracker.mp4", manifest) is None
    twice = write_manifest(tmp_path / "twice", rows=2 * ROW.format(fps="239.6", dish="35"))
    assert geometry.dish_mm_from_manifest(video, twice) is None
    no_dish = tmp_path / "no_dish.csv"
    no_dish.write_text("video_file,fps_true\ngroupB_2026-10-06_1325_main.MOV,239.6\n", encoding="utf-8")
    assert geometry.dish_mm_from_manifest(video, no_dish) is None
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert geometry.dish_mm_from_manifest(video, empty) is None


def test_manifest_rows_are_matched_like_last_week(tmp_path):
    # dish_mm and fps_true must come from the same row: wherever the ported reader finds the video,
    # so does dish_mm_from_manifest, and the other way round
    cases = {
        "plain": ROW.format(fps="239.6", dish="35"),
        "folder in video_file": "videos/" + ROW.format(fps="239.6", dish="35"),
        "row names the tracker copy": VIDEO_NAME + ",B,239.6,30,20,35\n",
        "other video": "another_clip.MOV,B,239.6,30,20,35\n",
        "two rows": 2 * ROW.format(fps="239.6", dish="35"),
        "no rows": "",
    }
    video = tmp_path / VIDEO_NAME
    found = {}
    for i, (name, rows) in enumerate(cases.items()):
        manifest = write_manifest(tmp_path / str(i), rows=rows)
        fps, dish = geometry.fps_from_manifest(video, manifest), geometry.dish_mm_from_manifest(video, manifest)
        assert (fps is None) == (dish is None), name
        found[name] = dish
    assert found["plain"] == pytest.approx(35.0) and found["folder in video_file"] == pytest.approx(35.0)
    assert found["other video"] is None and found["two rows"] is None and found["no rows"] is None
