"""`tracker_io` on files and tracks that the test itself made: read, plan, write (docs/ROADMAP.md, W1 step 5).

Every expected value here is one of two things: the values that the test wrote into a file, or text
typed by hand from the format of SPEC 8.3 (the frames and fps_true of a plan follow from the written
values by arithmetic). None is taken from what the package returns, and nothing is read back with the
package's own reader alone.

- `read_tracker_export`: one known track in every layout that Tracker and a spreadsheet give it:
  comma, tab or semicolon; LF or CRLF; with and without a byte order mark;
- `make_plan`: its options on a known track, by arithmetic on the frame numbers, and fps_true from
  the times of two marked frames;
- `write_tracker_file`: the lines of the file, typed by hand.

Coordinates: pixelx and pixely are px in Tracker's convention (pixelx to the right, pixely downward,
pixel centers at +0.5); x and y are mm in Tracker's axes, y up; frames are video frame numbers; t
is in s.

The tests of `fit_calibration` on known maps are in tests/core/test_tracker_io_known_maps.py.
"""

import numpy as np
import pandas as pd
import pytest
from helpers import java_sci

from outline_tracker.core import tracker_io


# ---------------------------------------------------------------------------------------------
# read_tracker_export on a known track in every layout

# One point mass at 0.05 mm per px with the origin at (160, 120) px and y up, 250 frames per s: each
# row is (t, frame, x, y, pixelx, pixely), or (t,) where Tracker has a time and no mark. Every number
# has at most 7 digits, so Tracker's "Full Precision" (`java_sci`) writes it as it is.
TRACK = [
    (0.0, 300, -2.975, 1.975, 100.5, 80.5),
    (0.008, 302, -2.9, 1.95, 102.0, 81.0),
    (0.016,),
    (0.024, 306, -2.75, 1.9, 105.0, 82.0),
    (0.032,),
    (0.04, 310, -2.5625, 1.8125, 108.75, 83.75),
]
COLUMNS = ["t", "frame", "x", "y", "pixelx", "pixely"]


def export_bytes(rows, sep=",", newline="\n", bom=False, name="mass A") -> bytes:
    """One point mass as Tracker exports it: a line with its name, the column names, one line per
    row, each line ended by a separator. `rows` as `TRACK`; the cells of a row without a mark are
    empty. `bom` puts the UTF-8 byte order mark first."""
    lines = [sep + name + sep * 5, sep.join(COLUMNS) + sep]
    for t, *mark in rows:
        cells = [java_sci(float(value)) for value in (t, *mark)] + [""] * (5 - len(mark))
        lines.append(sep.join(cells) + sep)
    return (b"\xef\xbb\xbf" if bom else b"") + (newline.join(lines) + newline).encode("utf-8")


@pytest.mark.parametrize("bom", [False, True], ids=["no mark", "byte order mark"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("sep", [",", "\t", ";"], ids=["comma", "tab", "semicolon"])
def test_one_known_track_reads_the_same_in_every_layout(tmp_path, sep, newline, bom):
    path = tmp_path / "A.csv"
    path.write_bytes(export_bytes(TRACK, sep, newline, bom))
    tracks = tracker_io.read_tracker_export(path)
    assert list(tracks) == ["A"]  # one point mass is named after its file
    table = tracks["A"]
    assert list(table.columns) == COLUMNS
    marked = [row for row in TRACK if len(row) == 6]  # the two rows without a mark are dropped
    assert pd.api.types.is_integer_dtype(table["frame"]) and table["frame"].tolist() == [300, 302, 306, 310]
    for k, column in enumerate(COLUMNS):  # as written, to the last bit that reading a decimal number leaves open
        assert table[column].tolist() == pytest.approx([row[k] for row in marked], rel=1e-12, abs=1e-15), column


def test_a_multi_header_whose_columns_cannot_be_split_evenly_is_refused(tmp_path):
    # two names, and nine columns after t: the second point mass has no pixely
    path = tmp_path / "start.csv"
    path.write_text("#multi:\n,A,,,,,B,,,,\nt,frame,x,y,pixelx,pixely,frame,x,y,pixelx,\n")
    refused = r"start\.csv: 9 columns cannot be split between the point masses \['A', 'B'\]"
    with pytest.raises(ValueError, match=refused):
        tracker_io.read_tracker_export(path)


# ---------------------------------------------------------------------------------------------
# make_plan on a known track


def moving_point(frames, marked=None):
    """Rows as `TRACK` for a point that moves 0.5 px per frame to the right along pixely = 100.5, at
    250 frames per s with t = 0 on the first frame and the map of `TRACK`. A frame that is not in
    `marked` has a time and no mark; without `marked` every frame has one."""
    rows = []
    for frame in frames:
        t = (frame - frames[0]) / 250.0
        if marked is None or frame in marked:
            px, py = 60.5 + 0.5 * frame, 100.5
            rows.append((t, frame, 0.05 * (px - 160.0), -0.05 * (py - 120.0), px, py))
        else:
            rows.append((t,))
    return rows


@pytest.fixture
def every_2nd_frame(tmp_path):
    """The export of one track marked on every 2nd frame from 100 to 140: 21 frames."""
    path = tmp_path / "A.csv"
    path.write_bytes(export_bytes(moving_point(range(100, 141, 2))))
    return path


def test_without_options_the_plan_is_the_track(every_2nd_frame):
    plan = tracker_io.make_plan(every_2nd_frame)
    assert (plan.start, plan.step, plan.n) == (100, 2, 21) and plan.frames[-1] == 140
    assert plan.names == ["A"] and plan.points_px == [pytest.approx((110.5, 100.5))]  # 60.5 + 0.5 * 100
    assert plan.fps == pytest.approx(250.0, rel=1e-5)  # from the times, which the export has with 7 digits


@pytest.mark.parametrize("step, n, last", [(5, 9, 140), (3, 14, 139), (1, 41, 140)])
def test_a_given_step_replaces_the_tracks_spacing(every_2nd_frame, step, n, last):
    # n = (140 - 100) // step + 1: from the first marked frame to the last one, never beyond it
    plan = tracker_io.make_plan(every_2nd_frame, step=step)
    assert (plan.start, plan.step, plan.n) == (100, step, n) and plan.frames[-1] == last


def test_step_0_becomes_1(every_2nd_frame):
    plan = tracker_io.make_plan(every_2nd_frame, step=0)
    assert (plan.start, plan.step, plan.n) == (100, 1, 41) and plan.frames == list(range(100, 141))


def test_seconds_0_gives_one_frame(every_2nd_frame):
    plan = tracker_io.make_plan(every_2nd_frame, seconds=0)
    assert (plan.start, plan.step, plan.n) == (100, 2, 1) and plan.frames == [100]


def test_a_track_with_unmarked_frames_takes_the_median_spacing_and_ends_on_its_last_frame(tmp_path):
    # Every 3rd frame from 100 to 121, of which 103 and 115 have no mark. The marked frames are 100,
    # 106, 109, 112, 118, 121: their spacings are 6, 3, 3, 6, 3, and the median of those is 3 (the
    # first is 6 and the mean 4.2). n = (121 - 100) // 3 + 1 = 8: the plan holds the unmarked frames too.
    path = tmp_path / "A.csv"
    path.write_bytes(export_bytes(moving_point(range(100, 122, 3), marked={100, 106, 109, 112, 118, 121})))
    plan = tracker_io.make_plan(path)
    assert (plan.start, plan.step, plan.n) == (100, 3, 8)
    assert plan.frames == [100, 103, 106, 109, 112, 115, 118, 121]
    assert plan.fps == pytest.approx(250.0, rel=1e-5)


def test_two_marked_frames_are_enough_for_fps_true(tmp_path):
    # The frames 100 and 110, with t = 0 and t = 0.04 s in the export: 10 frames in 0.04 s are 250
    # frames per s. One spacing of 10 frames, and n = (110 - 100) // 10 + 1 = 2.
    path = tmp_path / "A.csv"
    path.write_bytes(export_bytes(moving_point([100, 110])))
    plan = tracker_io.make_plan(path)
    assert plan.fps == pytest.approx(250.0, rel=1e-5)
    assert (plan.start, plan.step, plan.n) == (100, 10, 2) and plan.frames == [100, 110]


def test_a_point_mass_without_any_marked_frame_is_refused(tmp_path):
    # A is marked on the frames 100 and 110 (enough for the calibration), B on none
    path = tmp_path / "start.csv"
    lines = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    for t, *mark in moving_point([100, 110]):
        lines.append(",".join([java_sci(float(value)) for value in (t, *mark)] + [""] * 5) + ",")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"No marked frame for \['B'\] in start\.csv"):
        tracker_io.make_plan(path, fps=250.0)


# ---------------------------------------------------------------------------------------------
# write_tracker_file, line by line


def test_the_written_file_is_the_text_typed_by_hand(tmp_path):
    # SPEC 8.3: t with 7 decimals, the frame as a whole number, x and y with 6 decimals, pixelx and
    # pixely with 3, and an empty cell for a value that is not finite. The first frame number is a
    # float, as a table read by pandas may give it. 1 / 6 = 0.16666..., 1.23456789 rounds to 1.234568.
    path = tmp_path / "A.csv"
    result = tracker_io.write_tracker_file(
        path, "mass A", frames=[40.0, 44, 48, 52], t=[1 / 6, 0.1833333333, 0.2, 0.2166666667],
        x=[1.23456789, np.inf, -0.5, np.nan], y=[-2.0, -2.1, np.nan, -np.inf],
        px=[100.5, np.inf, 104.0627, np.nan], py=[80.25, 80.0, np.nan, 81.9994])
    assert result is None
    # Compared line by line: the writer uses the line ends of the system, and reading as text takes them away.
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n") and text.splitlines() == [
        ",mass A,,,,,",
        "t,frame,x,y,pixelx,pixely",
        "0.1666667,40,1.234568,-2.000000,100.500,80.250",
        "0.1833333,44,,-2.100000,,80.000",
        "0.2000000,48,-0.500000,,104.063,",
        "0.2166667,52,,,,81.999",
    ]


def written_lines(path):
    """The lines of a file that `write_tracker_file` wrote. Read as text, which takes the line ends
    away: the writer uses the line ends of the system, so lines are compared, not bytes."""
    return path.read_text(encoding="ascii").splitlines()


def test_an_empty_name_leaves_an_empty_cell_in_the_name_line(tmp_path):
    # SPEC 8.3: the first line is a comma, the name and five commas (`,A,,,,,`). Without a name that
    # is six commas, and the rest of the file is as with a name.
    path = tmp_path / "unnamed.csv"
    tracker_io.write_tracker_file(path, "", frames=[7], t=[0.028], x=[1.5], y=[-2.25], px=[190.0], py=[165.0])
    assert written_lines(path) == [
        ",,,,,,",
        "t,frame,x,y,pixelx,pixely",
        "0.0280000,7,1.500000,-2.250000,190.000,165.000",
    ]


@pytest.mark.parametrize("empty", [[], np.array([])], ids=["lists", "numpy arrays"])
def test_a_track_without_rows_is_the_name_line_and_the_column_names(tmp_path, empty):
    # SPEC 8.3 and the docstring: the name line, the column names, then one row per frame. No frame, no row.
    path = tmp_path / "A.csv"
    tracker_io.write_tracker_file(path, "A", frames=empty, t=empty, x=empty, y=empty, px=empty, py=empty)
    assert written_lines(path) == [",A,,,,,", "t,frame,x,y,pixelx,pixely"]


# One track of three frames, lost on the second, in the map of `TRACK` (0.05 mm per px from (160, 120)
# px, y up) at 240 frames per s: frames, t, x, y, pixelx, pixely. 16 / 240 = 0.0666..., 20 / 240 = 0.0833...
NAN = float("nan")
THREE_FRAMES = ([12, 16, 20], [0.05, 16 / 240, 20 / 240], [0.25, NAN, -1.125], [3.0, NAN, 2.0],
                [165.0, NAN, 137.5], [60.0, NAN, 80.0])
GIVEN_AS = {
    "lists": list,
    "numpy arrays": np.array,  # the frames as integers
    # as the columns of rows 5, 9 and 13 of a table read by pandas: every column holds floats, the frames too
    "pandas Series": lambda values: pd.Series(values, index=[5, 9, 13], dtype=float),
}


@pytest.mark.parametrize("kind", GIVEN_AS)
def test_lists_numpy_arrays_and_pandas_series_give_the_same_text(tmp_path, kind):
    path = tmp_path / "A.csv"
    tracker_io.write_tracker_file(path, "A", *(GIVEN_AS[kind](values) for values in THREE_FRAMES))
    assert written_lines(path) == [
        ",A,,,,,",
        "t,frame,x,y,pixelx,pixely",
        "0.0500000,12,0.250000,3.000000,165.000,60.000",
        "0.0666667,16,,,,",
        "0.0833333,20,-1.125000,2.000000,137.500,80.000",
    ]


def test_very_small_and_large_values_keep_the_decimals_of_the_format(tmp_path):
    # The format gives a number of decimals, not of digits (SPEC 8.2, "as in `segment.write_tracker_file`"):
    # 7 for t, 6 for mm, 3 for px. So 1e-9 mm is 0.000000 and 1e4 mm is 10000.000000, a pixel beyond
    # column 1000 keeps its three decimals (1234.5678 rounds to 1234.568, 1000.0004 to 1000.000), and no
    # number is written with an exponent. 100000 / 240 = 416.66666..., which rounds to 416.6666667.
    path = tmp_path / "A.csv"
    tracker_io.write_tracker_file(path, "A", frames=[0, 100000], t=[0.0, 100000 / 240], x=[1e-9, 1e4],
                                  y=[-1e4, 1e-9], px=[1234.5678, 1919.5], py=[1079.5, 1000.0004])
    assert written_lines(path) == [
        ",A,,,,,",
        "t,frame,x,y,pixelx,pixely",
        "0.0000000,0,0.000000,-10000.000000,1234.568,1079.500",
        "416.6666667,100000,10000.000000,0.000000,1919.500,1000.000",
    ]
