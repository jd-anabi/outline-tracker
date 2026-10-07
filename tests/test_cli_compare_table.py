"""Tests for `tracker_io.compare_tracks`, the table behind the hidden `compare-tracks` command (X10).

The folders, the positions and where the expected values come from are described in
tests/cli_compare_helpers.py: straight tracks written with the ported `write_tracker_file`, shifted by a
known offset or moved on one frame. Differences are px in Tracker's image coordinates (SPEC 3.1);
frames are the video's own frame numbers.
"""

import numpy as np
import pandas as pd
import pytest
from cli_compare_helpers import B_START, FRAMES, SHIFT, folders, jump_on_136, path_px, write_track

from outline_tracker import tracker_io

COLUMNS = ["track", "old_track", "n_common", "first", "last", "rms_px", "max_px", "at_frame",
           "new_first", "new_last", "old_first", "old_last"]


def test_compare_tracks_returns_the_numbers_behind_the_table(tmp_path):
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    bx, by = path_px(FRAMES, B_START)
    jump_x, jump_y = jump_on_136(bx, by)
    write_track(old, "A", FRAMES, ax, ay)
    write_track(new, "A", FRAMES, ax, ay, shift=SHIFT)
    write_track(old, "B", FRAMES[2:], bx[2:], by[2:])  # frames 104 to 148
    write_track(new, "B", FRAMES[:-3], jump_x[:-3], jump_y[:-3])  # frames 100 to 142; both have 20 frames
    write_track(new, "C", FRAMES, *path_px(FRAMES, start=(50.5, 150.5)))  # only in the new folder
    write_track(old, "D", FRAMES, *path_px(FRAMES, start=(250.5, 150.5)))  # only in the old folder
    table = tracker_io.compare_tracks(new, old)
    assert isinstance(table, pd.DataFrame) and list(table.columns) == COLUMNS
    a, b, c, d = table.to_dict("records")  # the new tracks by name, then the old track left over

    assert (a["track"], a["old_track"], a["n_common"], a["first"], a["last"]) == ("A", "A", 25, 100, 148)
    assert a["rms_px"] == pytest.approx(0.3, abs=1e-9) and a["max_px"] == pytest.approx(0.3, abs=1e-9)
    assert a["at_frame"] in FRAMES  # the same difference on every frame
    assert (a["new_first"], a["new_last"], a["old_first"], a["old_last"]) == (100, 148, 100, 148)

    assert (b["track"], b["old_track"], b["n_common"], b["first"], b["last"]) == ("B", "B", 20, 104, 142)
    assert b["rms_px"] == pytest.approx(50.0 / np.sqrt(20), abs=1e-9)
    assert b["max_px"] == pytest.approx(50.0, abs=1e-9) and b["at_frame"] == 136
    assert (b["new_first"], b["new_last"], b["old_first"], b["old_last"]) == (100, 142, 104, 148)

    assert c["track"] == "C" and pd.isna(c["old_track"]) and c["n_common"] == 0
    assert np.isnan(c["rms_px"]) and np.isnan(c["max_px"])
    assert all(pd.isna(c[key]) for key in ("first", "last", "at_frame", "old_first", "old_last"))
    assert (c["new_first"], c["new_last"]) == (100, 148)

    assert pd.isna(d["track"]) and d["old_track"] == "D" and d["n_common"] == 0
    assert (d["old_first"], d["old_last"]) == (100, 148) and pd.isna(d["new_first"])


def test_compare_tracks_by_position_takes_the_nearest_old_track(tmp_path):
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    write_track(new, "A", FRAMES, ax, ay)
    write_track(old, "near", FRAMES, ax, ay, shift=(3.0, 4.0))  # 5 px away
    write_track(old, "nearer", FRAMES, ax, ay, shift=(0.0, -2.0))  # 2 px away
    table = tracker_io.compare_tracks(str(new), str(old), by_position=True)  # folders as text, as the command has them
    assert table["track"].tolist()[:1] == ["A"] and table["old_track"].tolist() == ["nearer", "near"]
    assert table["rms_px"].iloc[0] == pytest.approx(2.0, abs=1e-9)
    assert pd.isna(table["track"].iloc[1])  # "near" is left over


def test_two_empty_folders_give_an_empty_table_with_every_column(tmp_path):
    new, old = folders(tmp_path)
    new.mkdir(parents=True)
    old.mkdir(parents=True)
    table = tracker_io.compare_tracks(new, old)
    assert list(table.columns) == COLUMNS and len(table) == 0


def test_a_frame_that_is_in_a_file_twice_is_refused(tmp_path):
    new, old = folders(tmp_path)
    frames = np.array([100, 102, 102, 104])
    write_track(new, "A", frames, *path_px(frames))
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    with pytest.raises(ValueError, match=r"A\.csv.* frame 102 "):
        tracker_io.compare_tracks(new, old)


def test_two_tracks_of_one_name_in_a_folder_are_refused(tmp_path):
    # A.csv, and a file of several point masses (Tracker's "#multi:" layout) that also holds an A
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES))
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    several = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2,
               "0.4166667,100,5.025,-4.025,100.500,80.500,100,10.025,-2.025,200.500,40.500,"]
    (new / "start.csv").write_text("\n".join(several) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two tracks named A"):
        tracker_io.compare_tracks(new, old)


def test_a_folder_that_is_not_there_is_named(tmp_path):
    new, old = folders(tmp_path)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    with pytest.raises(FileNotFoundError) as refused:
        tracker_io.compare_tracks(new, old)
    assert str(refused.value) == f"No such folder: {new}"
