"""Tests of outline_tracker/schema.py, the single source of truth for every output file (SPEC 8, 9).

Expected values come from the specification: column lists typed from SPEC 8.2, 8.4, 8.5, 8.7, the
SPEC.md tables parsed directly, and rows built cell by cell by hand. Nothing is copied from the
output of the code under test.
"""

import io
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import normalized

from outline_tracker import schema, tracker_io

REPO = Path(__file__).resolve().parents[1]
NAN = float("nan")

# ---------------------------------------------------------------------------------------------
# Column lists typed from the specification

POSITIONS_NAMES = [
    "track_id", "frame", "t_s", "x_mm", "y_mm", "u_px", "v_px", "area_mm2", "visible", "mode", "flags",
]
SHAPES_NAMES = [
    "track_id", "frame", "t_s", "area_mm2", "perimeter_mm", "major_mm", "minor_mm", "eccentricity",
    "theta_rad", "core_x_mm", "core_y_mm", "core_frac", "solidity", "circularity", "feret_max_mm",
    "n_components", "largest_fraction", "px_along_major", "cells_along_major", "shape_ok",
    "wall_dist_centroid_mm", "wall_dist_min_mm", "flags",
]
RADIAL_NAMES = ["track_id", "frame", "t_s"] + [f"r_{deg:03d}" for deg in range(0, 360, 5)]
PROBES_NAMES = ["frame", "t_s", "probe", "r", "g", "b", "gray"]

POSITIONS_DTYPES = dict(
    track_id="str", frame="int", t_s="float", x_mm="float", y_mm="float", u_px="float", v_px="float",
    area_mm2="float", visible="int", mode="str", flags="str",
)
SHAPES_DTYPES = dict(
    track_id="str", frame="int", t_s="float", area_mm2="float", perimeter_mm="float", major_mm="float",
    minor_mm="float", eccentricity="float", theta_rad="float", core_x_mm="float", core_y_mm="float",
    core_frac="float", solidity="float", circularity="float", feret_max_mm="float", n_components="int",
    largest_fraction="float", px_along_major="float", cells_along_major="float", shape_ok="int",
    wall_dist_centroid_mm="float", wall_dist_min_mm="float", flags="str",
)
RADIAL_DTYPES = dict(track_id="str", frame="int", t_s="float", **{n: "float" for n in RADIAL_NAMES[3:]})
PROBES_DTYPES = dict(frame="int", t_s="float", probe="str", r="float", g="float", b="float", gray="float")

FLAG_CODES = ["LOST", "JUMP", "SIZE", "CONTACT", "EDGE", "MULTI", "LOWRES", "ORIENT", "HEADGUESS"]
POSITION_FLAGS = ["LOST", "JUMP", "SIZE", "CONTACT", "EDGE"]
SHAPE_FLAGS = ["MULTI", "LOWRES", "ORIENT", "HEADGUESS"]

# SPEC 8.1, in the order of the tree drawn there; (name, goes into git per SPEC 8.13).
FILE_NAMES = [
    ("session.json", True), ("positions.csv", True), ("<model>/<id>.csv", True), ("shapes.csv", True),
    ("radial.csv", True), ("outlines.npz", True), ("probes.csv", True), ("overlay.mp4", False),
    ("results.npz", False), ("run.log", True), ("README.txt", True),
]

# SPEC 8.12 plus X5 (core_fallback, second_fraction, core_r_px) and X15 (bit-packed mask crops).
SPEC_RESULTS_KEYS = [
    "frames", "visible", "area_px", "u", "v", "cov_full", "cov_core", "core_u", "core_v", "core_frac",
    "outline_px", "n_components", "largest_fraction", "cell_px", "edge", "mode", "score",
]
EXTRA_RESULTS_KEYS = ["core_fallback", "second_fraction", "core_r_px", "mask_bits", "mask_shape", "mask_offset"]


def spec_section(heading_start: str) -> list[str]:
    """Lines of SPEC.md from the heading that starts with `heading_start` to the next heading."""
    lines = (REPO / "SPEC.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading_start))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return lines[start + 1 : end]


def spec_table(heading_start: str) -> list[list[str]]:
    """Body rows of the first markdown table of a SPEC.md section, as lists of stripped cells."""
    rows = []
    for line in spec_section(heading_start):
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[2:]  # drop the header row and the separator row


def spec_code_list(heading_start: str) -> list[str]:
    """The first inline code span with a comma-separated list, split into its items."""
    for line in spec_section(heading_start):
        match = re.search(r"`([^`]*,[^`]*)`", line)
        if match:
            return [item.strip() for item in match.group(1).split(",")]
    raise AssertionError(f"no column list in SPEC section {heading_start}")


def read_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def names(columns) -> list[str]:
    return [c.name for c in columns]


# ---------------------------------------------------------------------------------------------
# Column names, order, dtypes

def test_positions_columns_exact():
    assert names(schema.POSITIONS) == POSITIONS_NAMES


def test_shapes_columns_exact():
    assert names(schema.SHAPES) == SHAPES_NAMES


def test_radial_columns_exact():
    assert names(schema.RADIAL) == RADIAL_NAMES
    assert len(schema.RADIAL) == 3 + 72
    assert [n for n in names(schema.RADIAL) if n.startswith("r_")][:2] == ["r_000", "r_005"]
    assert names(schema.RADIAL)[-1] == "r_355"


def test_probes_columns_exact():
    assert names(schema.PROBES) == PROBES_NAMES


def test_dtypes():
    for columns, expected in [
        (schema.POSITIONS, POSITIONS_DTYPES), (schema.SHAPES, SHAPES_DTYPES),
        (schema.RADIAL, RADIAL_DTYPES), (schema.PROBES, PROBES_DTYPES),
    ]:
        assert {c.name: c.dtype for c in columns} == expected


def test_columns_match_the_spec_tables():
    """The spec's own table of positions.csv, and its lists for the other three files."""
    table = spec_table("### 8.2")
    assert [row[0] for row in table] == names(schema.POSITIONS)
    assert [row[1] for row in table] == [c.dtype for c in schema.POSITIONS]
    assert [row[2].replace("²", "^2") for row in table] == [c.unit for c in schema.POSITIONS]
    assert spec_code_list("### 8.4") == names(schema.SHAPES)
    radial = spec_code_list("### 8.5")  # "track_id, frame, t_s, r_000, r_005, ..., r_355"
    assert names(schema.RADIAL)[:4] == radial[:4] and names(schema.RADIAL)[-1] == radial[-1] == "r_355"
    assert spec_code_list("### 8.7") == names(schema.PROBES)


def test_units_follow_the_name_suffix():
    suffix_unit = {"_s": "s", "_mm": "mm", "_mm2": "mm^2", "_rad": "rad", "_px": "px"}
    for columns in (schema.POSITIONS, schema.SHAPES, schema.RADIAL, schema.PROBES):
        for col in columns:
            matches = [u for s, u in suffix_unit.items() if col.name.endswith(s)]
            if matches:
                assert col.unit == matches[0], col.name
    radial = {c.name: c for c in schema.RADIAL}
    assert radial["r_090"].unit == "mm"
    # counts of camera pixels and of grid cells carry their unit although the name has no suffix
    shapes = {c.name: c for c in schema.SHAPES}
    assert shapes["px_along_major"].unit == "px" and shapes["cells_along_major"].unit == "cells"


def test_every_column_is_documented_in_ascii():
    for columns in (schema.POSITIONS, schema.SHAPES, schema.RADIAL, schema.PROBES, schema.TRACKER_COLUMNS):
        for col in columns:
            assert col.meaning.strip(), col.name
            for field in (col.name, col.dtype, col.unit, col.fmt, col.meaning):
                assert field.isascii(), (col.name, field)


def test_number_formats_per_unit():
    """SPEC 8.2: t with 7 decimals, mm with 6, px with 3; ratios and angles 6; integers as integers."""
    for columns in (schema.POSITIONS, schema.SHAPES, schema.RADIAL, schema.PROBES):
        for col in columns:
            if col.dtype == "int":
                assert col.fmt == "d", col.name
            elif col.dtype == "str":
                assert col.fmt == "", col.name
            elif col.name.endswith("_s"):
                assert col.fmt == ".7f", col.name
            elif col.name.endswith(("_mm", "_mm2", "_rad")):
                assert col.fmt == ".6f", col.name
            elif col.name.endswith("_px"):
                assert col.fmt == ".3f", col.name
    shapes = {c.name: c.fmt for c in schema.SHAPES}
    for ratio in ("eccentricity", "core_frac", "solidity", "circularity", "largest_fraction"):
        assert shapes[ratio] == ".6f", ratio
    for count in ("px_along_major", "cells_along_major"):
        assert shapes[count] == ".3f", count
    assert {c.name: c.fmt for c in schema.PROBES if c.dtype == "float"} == {
        "t_s": ".7f", "r": ".3f", "g": ".3f", "b": ".3f", "gray": ".3f"
    }
    assert all(c.fmt == ".6f" for c in schema.RADIAL if c.name.startswith("r_"))


def test_integer_columns():
    ints = lambda columns: [c.name for c in columns if c.dtype == "int"]  # noqa: E731
    assert ints(schema.POSITIONS) == ["frame", "visible"]
    assert ints(schema.SHAPES) == ["frame", "n_components", "shape_ok"]
    assert ints(schema.RADIAL) == ["frame"]
    assert ints(schema.PROBES) == ["frame"]


def test_tracker_columns_are_last_weeks_file():
    assert names(schema.TRACKER_COLUMNS) == ["t", "frame", "x", "y", "pixelx", "pixely"]
    assert [c.fmt for c in schema.TRACKER_COLUMNS] == [".7f", "d", ".6f", ".6f", ".3f", ".3f"]


def test_tracker_columns_reproduce_the_ported_writer(tmp_path):
    """format_row on the Tracker-format columns gives the same text as tracker_io.write_tracker_file,
    for a tracked row and for a lost row (x, y, pixelx, pixely empty)."""
    path = tmp_path / "A.csv"
    tracker_io.write_tracker_file(
        path, "A", frames=[0, 2], t=[0.0, 2 / 239.6], x=[1.0, NAN], y=[-2.5, NAN],
        px=[10.5, NAN], py=[20.5, NAN],
    )
    lines = path.read_text().splitlines()
    assert lines[0] == ",A,,,,,"
    assert lines[1] == ",".join(names(schema.TRACKER_COLUMNS))
    rows = [
        dict(t=0.0, frame=0, x=1.0, y=-2.5, pixelx=10.5, pixely=20.5),
        dict(t=2 / 239.6, frame=2, x=NAN, y=NAN, pixelx=NAN, pixely=NAN),
    ]
    assert [schema.format_row(schema.TRACKER_COLUMNS, r) for r in rows] == lines[2:]


# ---------------------------------------------------------------------------------------------
# Flags, files, results keys, versions

def test_flags_in_spec_order():
    assert schema.FLAGS == FLAG_CODES
    table_codes = [re.match(r"`([A-Z]+)`", row[0]).group(1) for row in spec_table("## 9.")]
    assert table_codes == schema.FLAGS


def test_flag_scopes_and_texts():
    assert schema.POSITION_FLAGS == POSITION_FLAGS
    assert schema.SHAPE_FLAGS == SHAPE_FLAGS
    assert list(schema.FLAG_INFO) == FLAG_CODES
    for code, info in schema.FLAG_INFO.items():
        assert info.code == code
        assert info.scope == ("position" if code in POSITION_FLAGS else "shape")
        assert info.condition.strip() and info.meaning.strip()
        assert info.condition.isascii() and info.meaning.isascii()


def test_join_flags_orders_and_checks():
    assert schema.join_flags([]) == ""
    assert schema.join_flags(["HEADGUESS", "JUMP", "LOST", "JUMP"]) == "LOST;JUMP;HEADGUESS"
    assert schema.join_flags(iter(["LOWRES", "EDGE"])) == "EDGE;LOWRES"
    with pytest.raises(ValueError, match="NOPE"):
        schema.join_flags(["LOST", "NOPE"])


def test_files_match_the_run_folder_of_the_spec():
    assert [(f.name, f.in_git) for f in schema.FILES] == FILE_NAMES
    assert [f.name for f in schema.FILES if f.optional] == ["probes.csv"]
    for f in schema.FILES:
        assert f.meaning.strip() and f.meaning.isascii(), f.name
    by_name = {f.name: f for f in schema.FILES}
    assert by_name["positions.csv"].columns is schema.POSITIONS
    assert by_name["shapes.csv"].columns is schema.SHAPES
    assert by_name["radial.csv"].columns is schema.RADIAL
    assert by_name["probes.csv"].columns is schema.PROBES
    assert by_name["<model>/<id>.csv"].columns is schema.TRACKER_COLUMNS
    assert by_name["session.json"].columns is None


def test_file_name_constants():
    assert schema.SESSION_JSON == "session.json"
    assert schema.POSITIONS_CSV == "positions.csv"
    assert schema.SHAPES_CSV == "shapes.csv"
    assert schema.RADIAL_CSV == "radial.csv"
    assert schema.OUTLINES_NPZ == "outlines.npz"
    assert schema.PROBES_CSV == "probes.csv"
    assert schema.OVERLAY_MP4 == "overlay.mp4"
    assert schema.RESULTS_NPZ == "results.npz"
    assert schema.RUN_LOG == "run.log"
    assert schema.README_TXT == "README.txt"


def test_versions_and_text_rules():
    assert schema.RESULTS_VERSION == 1
    assert schema.SESSION_SCHEMA_VERSION == 1
    assert schema.CSV_ENCODING == "utf-8"
    assert schema.CSV_NEWLINE == "\n"


def test_results_keys_cover_the_spec_and_the_extras():
    keys = [k.name for k in schema.RESULTS_KEYS]
    assert len(keys) == len(set(keys))
    spec_tokens = {
        t for line in spec_section("### 8.12") for t in re.findall(r"`([a-z_]+)(?:\[[^`]*\])?`", line)
    }
    assert set(SPEC_RESULTS_KEYS) <= spec_tokens  # the list above is what the spec says
    assert set(keys) == set(SPEC_RESULTS_KEYS) | set(EXTRA_RESULTS_KEYS)
    assert schema.RESULTS_VERSION_KEY == "version"


def test_results_key_shapes_and_dtypes():
    by_name = {k.name: k for k in schema.RESULTS_KEYS}
    assert by_name["outline_px"].dtype == "float32" and by_name["outline_px"].shape == ("n", 256, 2)
    assert by_name["cov_full"].shape == ("n", 3) and by_name["cov_core"].shape == ("n", 3)
    assert by_name["frames"].shape == ("n",) and by_name["frames"].dtype == "int32"
    assert by_name["visible"].dtype == "bool" and by_name["edge"].dtype == "bool"
    assert by_name["core_fallback"].dtype == "bool"
    assert by_name["mask_offset"].shape == ("n", 2) and by_name["mask_shape"].shape == ("n", 2)
    assert by_name["mask_bits"].dtype == "uint8" and by_name["mask_bits"].shape == ("total_bytes",)
    for key in schema.RESULTS_KEYS:
        assert np.dtype(key.dtype).kind in "biuf" or key.name == "mode"
        assert key.meaning.strip() and key.meaning.isascii()
    assert by_name["mode"].dtype == "U6"  # "coarse" or "fine"


def test_outlines_keys_follow_the_spec():
    by_name = {k.name: k for k in schema.OUTLINES_KEYS}
    assert list(by_name) == ["frames", "xy_mm", "xieta_mm"]
    assert by_name["frames"].dtype == "int32" and by_name["frames"].shape == ("n",)
    assert by_name["xy_mm"].dtype == "float32" and by_name["xy_mm"].shape == ("n", "N", 2)
    assert by_name["xieta_mm"].dtype == "float32" and by_name["xieta_mm"].shape == ("n", "N", 2)
    assert schema.OUTLINES_META_KEY == "meta"
    assert schema.OUTLINE_POINTS == 128 and schema.STORED_OUTLINE_POINTS == 256


def test_npz_keys_round_trip():
    assert schema.npz_key("A", "xy_mm") == "A__xy_mm"
    assert schema.npz_key("A2", "frames") == "A2__frames"
    assert schema.split_npz_key("A2__frames") == ("A2", "frames")
    assert schema.split_npz_key("my__track__xy_mm") == ("my__track", "xy_mm")  # export names are verbatim
    assert schema.split_npz_key("meta") is None
    assert schema.split_npz_key("version") is None


# ---------------------------------------------------------------------------------------------
# Writing rows

def positions_row(**changes) -> dict:
    row = dict(track_id="A", frame=12, t_s=0.05, x_mm=1.5, y_mm=-2.25, u_px=100.5, v_px=200.25,
               area_mm2=0.1234567, visible=1, mode="coarse", flags="JUMP;SIZE")
    row.update(changes)
    return row


def test_format_row_positions_visible():
    assert schema.format_row(schema.POSITIONS, positions_row()) == (
        "A,12,0.0500000,1.500000,-2.250000,100.500,200.250,0.123457,1,coarse,JUMP;SIZE"
    )


def test_format_row_positions_lost_has_empty_cells():
    nan = NAN
    row = positions_row(track_id="B", frame=40, t_s=0.1666667, x_mm=nan, y_mm=nan, u_px=nan, v_px=nan,
                        area_mm2=nan, visible=0, flags="LOST")
    cells = ["B", "40", "0.1666667", "", "", "", "", "", "0", "coarse", "LOST"]
    assert schema.format_row(schema.POSITIONS, row) == ",".join(cells)


def test_format_row_accepts_a_sequence_in_column_order():
    row = positions_row()
    values = [row[name] for name in POSITIONS_NAMES]
    assert schema.format_row(schema.POSITIONS, values) == schema.format_row(schema.POSITIONS, row)
    with pytest.raises(ValueError, match="11"):
        schema.format_row(schema.POSITIONS, values[:-1])


def test_format_row_mapping_needs_every_column_and_ignores_extras():
    row = positions_row(extra_column=3.0)
    assert schema.format_row(schema.POSITIONS, row).startswith("A,12,")
    del row["mode"]
    with pytest.raises(KeyError, match="mode"):
        schema.format_row(schema.POSITIONS, row)


def test_format_row_numpy_scalars_and_none():
    row = positions_row(frame=np.int64(12), t_s=np.float32(0.5), x_mm=np.float64(1.5), visible=np.bool_(True),
                        y_mm=None, flags=None)
    assert schema.format_row(schema.POSITIONS, row) == "A,12,0.5000000,1.500000,,100.500,200.250,0.123457,1,coarse,"


def test_non_finite_floats_become_empty_cells():
    row = positions_row(x_mm=float("inf"), y_mm=-float("inf"), u_px=NAN)
    assert schema.format_row(schema.POSITIONS, row).split(",")[3:6] == ["", "", ""]


def test_integer_columns_refuse_nan_and_fractions():
    with pytest.raises(ValueError, match="visible"):
        schema.format_row(schema.POSITIONS, positions_row(visible=NAN))
    with pytest.raises(ValueError, match="visible"):
        schema.format_row(schema.POSITIONS, positions_row(visible=None))
    with pytest.raises(ValueError, match="frame"):
        schema.format_row(schema.POSITIONS, positions_row(frame=12.5))
    # a whole number held as a float (a DataFrame row) is fine and written without a decimal point
    assert schema.format_row(schema.POSITIONS, positions_row(frame=12.0)).startswith("A,12,")


def test_text_with_a_comma_or_quote_is_quoted():
    line = schema.format_row(schema.POSITIONS, positions_row(track_id='a,"b"'))
    assert line.startswith('"a,""b""",12,')
    df = read_csv(schema.csv_text(schema.POSITIONS, [positions_row(track_id='a,"b"')]))
    assert df["track_id"].tolist() == ['a,"b"']


def test_text_cells_must_be_text():
    with pytest.raises(ValueError, match="flags"):
        schema.format_row(schema.POSITIONS, positions_row(flags=3))


def test_float_cells_must_be_numbers():
    with pytest.raises(ValueError, match="x_mm"):
        schema.format_row(schema.POSITIONS, positions_row(x_mm="1.5"))


def test_header_line():
    assert schema.header_line(schema.PROBES) == "frame,t_s,probe,r,g,b,gray"
    assert schema.header_line(schema.RADIAL).startswith("track_id,frame,t_s,r_000,r_005,")


def test_csv_text_uses_lf_and_ends_with_one():
    text = schema.csv_text(schema.POSITIONS, [positions_row(), positions_row(frame=14)])
    assert "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")
    assert text.splitlines()[0] == ",".join(POSITIONS_NAMES)
    assert len(text.splitlines()) == 3
    text.encode(schema.CSV_ENCODING)


def test_csv_text_with_no_rows_is_the_header_only():
    """radial.csv of a run without fine tracks (X12): the file exists and holds its header."""
    assert schema.csv_text(schema.RADIAL, []) == ",".join(RADIAL_NAMES) + "\n"


# ---------------------------------------------------------------------------------------------
# Reading back

def shapes_visible_row() -> dict:
    row = {c.name: 1.25 for c in schema.SHAPES if c.dtype == "float"}
    row.update(track_id="A", frame=10, t_s=0.0417362, n_components=1, shape_ok=1, flags="")
    return row


def shapes_lost_row() -> dict:
    row = {c.name: NAN for c in schema.SHAPES if c.dtype == "float"}
    row.update(track_id="B", frame=40, t_s=0.1666667, n_components=0, shape_ok=0, flags="LOST;HEADGUESS")
    return row


def test_lost_shapes_row_has_empty_floats_and_zero_integers():
    cells = ["B", "40", "0.1666667"] + [""] * 12 + ["0", "", "", "", "0", "", "", "LOST;HEADGUESS"]
    assert len(cells) == len(SHAPES_NAMES)
    assert schema.format_row(schema.SHAPES, shapes_lost_row()) == ",".join(cells)


@pytest.mark.parametrize(
    ("columns", "dtypes"), [(schema.POSITIONS, POSITIONS_DTYPES), (schema.SHAPES, SHAPES_DTYPES)]
)
def test_integer_columns_stay_integer_when_a_row_is_lost(columns, dtypes):
    if columns is schema.POSITIONS:
        rows = [positions_row(flags=""), positions_row(frame=14, x_mm=NAN, y_mm=NAN, u_px=NAN, v_px=NAN,
                                                       area_mm2=NAN, visible=0, flags="LOST")]
    else:
        rows = [shapes_visible_row(), shapes_lost_row()]
    df = read_csv(schema.csv_text(columns, rows))
    assert df.columns.tolist() == names(columns)
    for name, kind in dtypes.items():
        if kind == "int":
            assert pd.api.types.is_integer_dtype(df[name]), name
        elif kind == "float":
            assert pd.api.types.is_float_dtype(df[name]), name
    lost = df.iloc[1]
    for name, kind in dtypes.items():
        if kind == "float" and name not in ("t_s",):
            assert math.isnan(lost[name]), name
    assert df["flags"].fillna("").tolist()[0] == ""
    assert df["flags"].fillna("").tolist()[1].startswith("LOST")


def test_the_reading_recipe_of_the_readme_works():
    """An all-empty flags column is read as NaN, and fillna("") makes str.contains safe."""
    df = read_csv(schema.csv_text(schema.POSITIONS, [positions_row(flags=""), positions_row(frame=14, flags="")]))
    flags = df["flags"].fillna("")
    assert flags.tolist() == ["", ""]
    assert not flags.str.contains("JUMP").any()
    df = read_csv(schema.csv_text(schema.POSITIONS, [positions_row(flags="JUMP;SIZE"), positions_row(flags="")]))
    assert df["flags"].fillna("").str.contains("JUMP").tolist() == [True, False]


def test_radial_row_round_trip():
    row = {name: 50.0 + i for i, name in enumerate(RADIAL_NAMES[3:])}
    row.update(track_id="A", frame=3, t_s=0.0125)
    lost = {name: NAN for name in RADIAL_NAMES[3:]}
    lost.update(track_id="A", frame=4, t_s=0.0166667)
    df = read_csv(schema.csv_text(schema.RADIAL, [row, lost]))
    assert df.shape == (2, 75)
    assert df["r_000"].tolist()[0] == 50.0 and df["r_355"].tolist()[0] == 50.0 + 71
    assert df.iloc[1, 3:].isna().all()
    assert pd.api.types.is_integer_dtype(df["frame"])


def test_probes_row():
    row = dict(frame=12, t_s=0.05, probe="LED1", r=200.0, g=100.5, b=50.25, gray=124.522)
    assert schema.format_row(schema.PROBES, row) == "12,0.0500000,LED1,200.000,100.500,50.250,124.522"


# ---------------------------------------------------------------------------------------------
# README.txt

@pytest.fixture(scope="module")
def readme() -> str:
    return schema.readme_text()


def section(text: str, heading: str) -> list[str]:
    """Lines under `== HEADING ==` up to the next `==` heading."""
    lines = text.splitlines()
    start = lines.index(f"== {heading} ==")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("== ")), len(lines))
    return lines[start + 1 : end]


def test_readme_is_plain_ascii_with_lf(readme):
    assert readme.isascii()
    assert "\r" not in readme and "\t" not in readme
    assert readme.endswith("\n") and not readme.endswith("\n\n")
    assert readme == schema.readme_text()  # deterministic: generated, not edited


def test_readme_names_every_column(readme):
    for columns in (schema.POSITIONS, schema.SHAPES, schema.RADIAL, schema.PROBES, schema.TRACKER_COLUMNS):
        for col in columns:
            assert re.search(rf"(?<![A-Za-z0-9_]){re.escape(col.name)}(?![A-Za-z0-9_])", readme), col.name


def test_readme_gives_every_column_its_meaning_and_unit(readme):
    flat = normalized(readme)
    for columns in (schema.POSITIONS, schema.SHAPES, schema.PROBES, schema.TRACKER_COLUMNS):
        for col in columns:
            assert normalized(col.meaning) in flat, col.name
            if col.unit:
                assert f"{col.dtype}, {col.unit}" in flat, col.name
    radial = [c for c in schema.RADIAL if c.name.startswith("r_")]
    assert normalized(radial[0].meaning) in flat  # one shared text for the 72 radial columns
    assert flat.count("r_000") >= 1 and "72 columns" in flat


def test_readme_names_every_flag_with_its_meaning(readme):
    flat = normalized(readme)
    for code, info in schema.FLAG_INFO.items():
        assert re.search(rf"(?<![A-Z]){code}(?![A-Z])", readme), code
        assert normalized(info.condition) in flat, code
        assert normalized(info.meaning) in flat, code


def test_readme_names_every_file_and_the_model_folder(readme):
    for spec in schema.FILES:
        assert spec.name in readme, spec.name
        assert normalized(spec.meaning) in normalized(readme), spec.name
    assert "edgetam/A.csv" in readme and "A2" in readme


def test_readme_says_which_flags_concern_positions_and_which_only_shapes(readme):
    lines = readme.splitlines()
    [position_line] = [line for line in lines if "LOST, JUMP, SIZE, CONTACT, EDGE" in line]
    [shape_line] = [line for line in lines if "MULTI, LOWRES, ORIENT, HEADGUESS" in line]
    assert "position" in position_line.lower() and "shape" not in position_line.lower()
    assert "only" in shape_line.lower() and "shape" in shape_line.lower()
    flat = normalized(readme)
    assert "not on \"flags is empty\"" in flat  # filter on specific codes


def test_readme_says_fine_tracks_by_default_for_radial_and_outlines(readme):
    fine = normalized(" ".join(section(readme, "SHAPE FILES")))
    assert "radial.csv and outlines.npz hold the fine tracks by default" in fine
    assert "shape_files_for_coarse" in fine
    assert "only its header line" in fine and "only the key meta" in fine
    assert "outer envelope" in fine  # SPEC 7.6: outlines that are not star-shaped
    assert "--fine" in fine  # how to get a fine track, so an empty file explains itself
    flat = normalized(readme)
    for key in schema.OUTLINES_KEYS:
        assert f"<id>__{key.name}" in flat
    for field in schema.OUTLINES_META_FIELDS:
        assert field in flat


def test_readme_lists_what_goes_into_git(readme):
    body = section(readme, "WHAT GOES INTO GIT")
    split = next(i for i, line in enumerate(body) if line.strip().startswith("Not in git"))
    in_git, not_in_git = "\n".join(body[:split]), "\n".join(body[split:])
    for spec in schema.FILES:
        assert (spec.name in in_git) is spec.in_git, spec.name
        assert (spec.name in not_in_git) is (not spec.in_git), spec.name
    assert "calibration.json" in in_git
    assert "!**/outlines.npz" in in_git  # the only full-shape output must be committed
    assert "*.npz" in in_git
    assert "Drive" in not_in_git


def test_readme_states_the_conventions(readme):
    flat = normalized(readme)
    assert "(c + 0.5, r + 0.5)" in flat  # pixel centers
    assert "y points up" in flat
    assert "t_s = frame / fps_true" in flat
    assert "empty cell" in flat and "UTF-8" in flat and "LF" in flat
    assert 'df["flags"].fillna("")' in flat
    assert "not df.flags" in flat
    for decimals in ("7 decimals", "6 decimals", "3 decimals"):
        assert decimals in flat
    assert "visible = 0" in flat and "n_components = 0" in flat and "shape_ok = 0" in flat


def test_readme_says_what_to_do_after_a_crash_or_a_locked_file(readme):
    flat = normalized(readme)
    assert "outline-tracker export" in flat and "session.json" in flat
    assert ".new" in flat


def test_readme_sections_in_order(readme):
    headings = [line for line in readme.splitlines() if line.startswith("== ")]
    assert headings == [
        "== FILES ==", "== CONVENTIONS ==", "== COLUMNS ==", "== FLAGS ==", "== SHAPE FILES ==",
        "== READING THE FILES ==", "== WHAT GOES INTO GIT ==", "== IF SOMETHING WENT WRONG ==",
    ]


# ---------------------------------------------------------------------------------------------
# The module stays light

def test_importing_schema_loads_neither_torch_nor_qt():
    """In a subprocess: pytest-qt has already imported PySide6 into this one."""
    code = (
        "import sys; import outline_tracker.schema; "
        "bad = sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph', 'pandas') if m in sys.modules); "
        "print(bad)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
