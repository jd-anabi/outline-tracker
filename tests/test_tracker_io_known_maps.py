"""`tracker_io` on maps, files and tracks that the test itself made (docs/ROADMAP.md, W1 step 5).

Every expected value here is one of three things: the map that the test made the points with, the
values that the test wrote into a file, or text typed by hand from the format of SPEC 8.3. None is
taken from what the package returns, and nothing is read back with the package's own reader alone.

- `fit_calibration`: 200 random maps of each form, made from a scale, an angle and a shift that
  the test drew (seeded numpy); rows without a value; the refusal below 1 px; the choice of the
  form under noise; points moved off a typed map by a typed distance to both sides, where the fit
  is still that map and `rms_mm` follows from the distance by arithmetic;
- `read_tracker_export`: one known track in every layout that Tracker and a spreadsheet give it:
  comma, tab or semicolon; LF or CRLF; with and without a byte order mark;
- `make_plan`: its options on a known track, by arithmetic on the frame numbers, and fps_true from
  the times of two marked frames;
- `write_tracker_file`: the lines of the file, typed by hand.

Coordinates: pixelx and pixely are px in Tracker's convention (pixelx to the right, pixely downward,
pixel centers at +0.5); x and y are mm in Tracker's axes, y up; frames are video frame numbers; t
is in s. A map is given by its scale in mm per px, the angle of its x axis in radians and its shift
in mm.
"""

import numpy as np
import pandas as pd
import pytest
from helpers import java_sci

from outline_tracker import tracker_io

N_RANDOM = 200
FRAME_PX = (1920.0, 1080.0)  # the points of a map are drawn inside a frame of this size, px
SAME_MM, SAME_PX = 0.5e-6, 0.5e-3  # half of the last decimal that a Tracker-format file has: 6 for mm, 3 for px


# ---------------------------------------------------------------------------------------------
# fit_calibration on known maps


def known_map(rng, mirrored=False):
    """A random pixel -> mm map, as (to_mm, scale, matrix).

    Tracker's map is a turn by `angle` after a flip of the vertical axis (its y points up, image rows
    go down), times `scale`, plus a shift; a mirrored map is the same without the flip:
        Tracker's:  x = a px + c py + tx,   y = c px - a py + ty
        mirrored:   x = a px - c py + tx,   y = c px + a py + ty
    with a = scale cos(angle) and c = scale sin(angle). The scale is between 0.003 and 1 mm per px,
    the angle is any, and the shift is up to 300 mm either way. `to_mm(px, py)` takes arrays;
    `matrix` is the 2 x 2 part of the map, mm per px.
    """
    scale = 10 ** rng.uniform(-2.5, 0.0)
    angle = rng.uniform(-np.pi, np.pi)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = scale * (turn if mirrored else turn @ np.diag([1.0, -1.0]))
    tx, ty = rng.uniform(-300, 300, 2)

    def to_mm(px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        return matrix[0, 0] * px + matrix[0, 1] * py + tx, matrix[1, 0] * px + matrix[1, 1] * py + ty

    return to_mm, scale, matrix


def assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng):
    """A fitted calibration is the map that made its points: the scale, the 2 x 2 part, no residual,
    and at five points in and around the frame `to_mm` gives the map's mm and `to_px` gives the px
    back from the map's own mm. "The same", and "no residual", is half of the last decimal that a
    Tracker-format file has, 0.5e-6 mm and 0.5e-3 px: rounding in the fit grows with the size of
    the numbers (here up to 2500 mm), and no file could show a difference below that."""
    assert cal.mm_per_px == pytest.approx(scale, rel=1e-6)
    assert cal.matrix() == pytest.approx(matrix, rel=0, abs=1e-6 * scale)
    assert cal.rms_mm < SAME_MM  # zero up to rounding
    probe_px, probe_py = rng.uniform(-100, 2000, 5), rng.uniform(-100, 1200, 5)
    probe_x, probe_y = to_mm(probe_px, probe_py)
    x, y = cal.to_mm(probe_px, probe_py)
    assert x == pytest.approx(probe_x, rel=0, abs=SAME_MM) and y == pytest.approx(probe_y, rel=0, abs=SAME_MM)
    px, py = cal.to_px(probe_x, probe_y)
    assert px == pytest.approx(probe_px, rel=0, abs=SAME_PX) and py == pytest.approx(probe_py, rel=0, abs=SAME_PX)


def test_fit_calibration_recovers_trackers_map_whatever_its_scale_angle_and_shift():
    rng = np.random.default_rng(2026)
    for case in range(N_RANDOM):
        to_mm, scale, matrix = known_map(rng)
        n = int(rng.integers(2, 41))  # two points are enough for Tracker's form
        px, py = rng.uniform(0, FRAME_PX[0], n), rng.uniform(0, FRAME_PX[1], n)
        x, y = to_mm(px, py)
        given = (px, py, x, y) if case % 2 else tuple(values.tolist() for values in (px, py, x, y))  # arrays, lists
        cal = tracker_io.fit_calibration(*given)
        assert cal.flip is True, case
        assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_fit_calibration_recovers_a_mirrored_map_from_points_that_are_not_on_one_line():
    rng = np.random.default_rng(2027)
    corners = np.array([(100.5, 100.5), (1800.5, 150.5), (900.5, 1000.5)])  # a triangle: never on one line
    for case in range(N_RANDOM):
        to_mm, scale, matrix = known_map(rng, mirrored=True)
        n_more = int(rng.integers(0, 38))
        px = np.concatenate([corners[:, 0], rng.uniform(0, FRAME_PX[0], n_more)])
        py = np.concatenate([corners[:, 1], rng.uniform(0, FRAME_PX[1], n_more)])
        x, y = to_mm(px, py)
        given = (px, py, x, y) if case % 2 else tuple(values.tolist() for values in (px, py, x, y))
        cal = tracker_io.fit_calibration(*given)
        assert cal.flip is False, case
        assert cal.matrix()[0, 0] == cal.matrix()[1, 1] and cal.matrix()[0, 1] == -cal.matrix()[1, 0]  # no flip
        assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_rows_with_a_missing_value_change_nothing():
    rng = np.random.default_rng(5)
    to_mm, scale, matrix = known_map(rng)
    px, py = rng.uniform(0, FRAME_PX[0], 12), rng.uniform(0, FRAME_PX[1], 12)
    x, y = to_mm(px, py)
    complete = tracker_io.fit_calibration(px, py, x, y)
    # Four more rows, each without one of its four values, and far off the map with the other three
    # (1e6 mm): a fit that used one of them would be another fit.
    nan = float("nan")
    extra = np.array([(nan, 500.5, 1e6, 1e6), (900.5, nan, 1e6, -1e6), (900.5, 500.5, nan, 1e6),
                      (900.5, 500.5, -1e6, nan)])
    at = [0, 5, 5, 12]  # where they stand among the complete rows: first, in the middle, last
    with_gaps = [np.insert(values, at, extra[:, k]) for k, values in enumerate((px, py, x, y))]
    cal = tracker_io.fit_calibration(*with_gaps)
    assert cal == complete
    assert cal.flip is True
    assert_the_fit_is_the_map(cal, to_mm, scale, matrix, rng)


def test_points_in_one_place_or_under_1_px_apart_are_refused_and_1_px_is_enough():
    to_mm, scale, _ = known_map(np.random.default_rng(6))

    def fit(px, py):
        return tracker_io.fit_calibration(px, py, *to_mm(px, py))

    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.full(7, 640.5), np.full(7, 360.5))  # seven points in one place
    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.array([640.5, 641.0]), np.array([360.5, 360.5]))  # 0.5 px apart
    with pytest.raises(ValueError, match="at least two points at least 1 px apart"):
        fit(np.array([640.5, 640.5]), np.array([360.5, 361.25]))  # 0.75 px apart, along pixely
    cal = fit(np.array([640.5, 641.5]), np.array([360.5, 360.5]))  # exactly 1.0 px apart
    assert cal.flip is True and cal.mm_per_px == pytest.approx(scale, rel=1e-6)


def test_mirrored_data_keep_the_mirrored_form_under_small_noise_and_lose_it_under_noise_far_larger_than_the_map():
    # The unflipped form is kept only if it fits clearly better. Noise 1000 times smaller than the
    # map's extent leaves it far better; under noise 1000 times larger neither form fits, so neither
    # fits clearly better, and Tracker's form stands.
    rng = np.random.default_rng(11)
    to_mm, scale, _ = known_map(rng, mirrored=True)
    px, py = rng.uniform(0, FRAME_PX[0], 40), rng.uniform(0, FRAME_PX[1], 40)
    x, y = to_mm(px, py)
    extent_mm = scale * float(np.hypot(*FRAME_PX))  # the frame's diagonal, as the map sees it
    small, large = rng.normal(0, 1e-3 * extent_mm, (2, 40)), rng.normal(0, 1e3 * extent_mm, (2, 40))
    assert tracker_io.fit_calibration(px, py, x + small[0], y + small[1]).flip is False
    assert tracker_io.fit_calibration(px, py, x + large[0], y + large[1]).flip is True


# Tracker's map at 0.05 mm per px with its x axis turned by 12 degrees and a shift of (-31.5, 27.25)
# mm, and seven points in px that are not on one line. E_MM is how far each mm value is moved.
TYPED_SCALE, TYPED_ANGLE_DEG, TYPED_SHIFT = 0.05, 12.0, (-31.5, 27.25)
TYPED_POINTS = [(100.5, 80.5), (1800.5, 150.5), (900.5, 1000.5), (400.5, 600.5), (1500.5, 900.5),
                (250.5, 950.5), (1200.5, 300.5)]
PROBES = [(-100.0, -100.0), (2000.0, 1200.0), (960.0, 540.0), (0.5, 1079.5), (1919.5, 0.5)]  # px
E_MM = 0.02


@pytest.mark.parametrize("order", ["all +e, then all -e", "alternating"])
def test_points_moved_by_e_to_both_sides_along_x_give_the_typed_map_and_an_rms_of_e_over_root_2(order):
    # Each of the seven points is given twice: with its mm value on the typed map moved along x by
    # +e, and by -e. For a map that gives p where the typed map gives m,
    #     (p - (m + e))^2 + (p - (m - e))^2 = 2 (p - m)^2 + 2 e^2,
    # so the sum of squares over all rows is twice the sum over the unmoved values plus a constant.
    # The least-squares fit over all rows is therefore the map of the unmoved values: the typed
    # map. A fit over a part of the rows is another map: seven is odd, so in neither order is the
    # first half of the rows a set of whole pairs.
    a, c = TYPED_SCALE * np.cos(np.radians(TYPED_ANGLE_DEG)), TYPED_SCALE * np.sin(np.radians(TYPED_ANGLE_DEG))
    tx, ty = TYPED_SHIFT

    def typed_map(px, py):  # Tracker's form
        return a * px + c * py + tx, c * px - a * py + ty

    px, py = np.array(TYPED_POINTS).T
    x, y = typed_map(px, py)
    if order == "alternating":  # point 1 by +e, point 1 by -e, point 2 by +e, ...
        rows = px.repeat(2), py.repeat(2), x.repeat(2) + np.tile([E_MM, -E_MM], len(px)), y.repeat(2)
    else:
        rows = np.tile(px, 2), np.tile(py, 2), np.concatenate([x + E_MM, x - E_MM]), np.tile(y, 2)
    cal = tracker_io.fit_calibration(*rows)

    assert cal.flip is True
    assert cal.mm_per_px == pytest.approx(TYPED_SCALE, rel=1e-6)
    assert (cal.a, cal.c) == pytest.approx((a, c), rel=0, abs=1e-6 * TYPED_SCALE)  # the scale and the angle
    assert (cal.tx, cal.ty) == pytest.approx((tx, ty), rel=0, abs=SAME_MM)
    probe_px, probe_py = np.array(PROBES).T
    probe_x, probe_y = typed_map(probe_px, probe_py)
    fitted_x, fitted_y = cal.to_mm(probe_px, probe_py)
    assert fitted_x == pytest.approx(probe_x, rel=0, abs=SAME_MM)
    assert fitted_y == pytest.approx(probe_y, rel=0, abs=SAME_MM)
    # rms_mm is the root mean square over the x and the y residuals of all rows together (the
    # docstring of fit_calibration). At the typed map the 14 x residuals are +e or -e and the 14 y
    # residuals are 0: sqrt(14 e^2 / 28) = e / sqrt(2), about 0.01414 mm. The root mean square of
    # the distances would be e, and the mean of the residuals' sizes e / 2.
    assert cal.rms_mm == pytest.approx(E_MM / np.sqrt(2.0), rel=1e-9)


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
