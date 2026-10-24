"""Tests of outline_tracker/core/schema.py and csv_format.py: the rows as CSV text and read back, and README.txt.

Expected values come from the specification: column lists typed from SPEC 8.2, 8.4, 8.5, 8.7, and rows
built cell by cell by hand. Nothing is copied from the output of the code under test.

The rest of the tests of the module are in tests/core/test_schema.py: the tables of columns, flags, files
and keys. The column lists and dtypes that both files use are in tests/support/schema_helpers.py.
"""

import io
import math
import re

import numpy as np
import pandas as pd
import pytest
from helpers import normalized
from schema_helpers import (
    NAN,
    POSITIONS_DTYPES,
    POSITIONS_NAMES,
    RADIAL_NAMES,
    SHAPES_DTYPES,
    SHAPES_NAMES,
    names,
)

from outline_tracker.core import csv_format, schema


def read_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------------------------
# Writing rows

def positions_row(**changes) -> dict:
    row = dict(track_id="A", frame=12, t_s=0.05, x_mm=1.5, y_mm=-2.25, u_px=100.5, v_px=200.25,
               area_mm2=0.1234567, visible=1, mode="coarse", flags="JUMP;SIZE")
    row.update(changes)
    return row


def test_format_row_positions_visible():
    assert csv_format.format_row(schema.POSITIONS, positions_row()) == (
        "A,12,0.0500000,1.500000,-2.250000,100.500,200.250,0.123457,1,coarse,JUMP;SIZE"
    )


def test_format_row_positions_lost_has_empty_cells():
    nan = NAN
    row = positions_row(track_id="B", frame=40, t_s=0.1666667, x_mm=nan, y_mm=nan, u_px=nan, v_px=nan,
                        area_mm2=nan, visible=0, flags="LOST")
    cells = ["B", "40", "0.1666667", "", "", "", "", "", "0", "coarse", "LOST"]
    assert csv_format.format_row(schema.POSITIONS, row) == ",".join(cells)


def test_format_row_accepts_a_sequence_in_column_order():
    row = positions_row()
    values = [row[name] for name in POSITIONS_NAMES]
    assert csv_format.format_row(schema.POSITIONS, values) == csv_format.format_row(schema.POSITIONS, row)
    with pytest.raises(ValueError, match="11"):
        csv_format.format_row(schema.POSITIONS, values[:-1])


def test_format_row_mapping_needs_every_column_and_ignores_extras():
    row = positions_row(extra_column=3.0)
    assert csv_format.format_row(schema.POSITIONS, row).startswith("A,12,")
    del row["mode"]
    with pytest.raises(KeyError, match="mode"):
        csv_format.format_row(schema.POSITIONS, row)


def test_format_row_numpy_scalars_and_none():
    row = positions_row(frame=np.int64(12), t_s=np.float32(0.5), x_mm=np.float64(1.5), visible=np.bool_(True),
                        y_mm=None, flags=None)
    assert csv_format.format_row(schema.POSITIONS, row) == "A,12,0.5000000,1.500000,,100.500,200.250,0.123457,1,coarse,"


def test_non_finite_floats_become_empty_cells():
    row = positions_row(x_mm=float("inf"), y_mm=-float("inf"), u_px=NAN)
    assert csv_format.format_row(schema.POSITIONS, row).split(",")[3:6] == ["", "", ""]


def test_integer_columns_refuse_nan_and_fractions():
    with pytest.raises(ValueError, match="visible"):
        csv_format.format_row(schema.POSITIONS, positions_row(visible=NAN))
    with pytest.raises(ValueError, match="visible"):
        csv_format.format_row(schema.POSITIONS, positions_row(visible=None))
    with pytest.raises(ValueError, match="frame"):
        csv_format.format_row(schema.POSITIONS, positions_row(frame=12.5))
    # a whole number held as a float (a DataFrame row) is fine and written without a decimal point
    assert csv_format.format_row(schema.POSITIONS, positions_row(frame=12.0)).startswith("A,12,")


def test_text_with_a_comma_or_quote_is_quoted():
    line = csv_format.format_row(schema.POSITIONS, positions_row(track_id='a,"b"'))
    assert line.startswith('"a,""b""",12,')
    df = read_csv(csv_format.csv_text(schema.POSITIONS, [positions_row(track_id='a,"b"')]))
    assert df["track_id"].tolist() == ['a,"b"']


def test_text_cells_must_be_text():
    with pytest.raises(ValueError, match="flags"):
        csv_format.format_row(schema.POSITIONS, positions_row(flags=3))


def test_float_cells_must_be_numbers():
    with pytest.raises(ValueError, match="x_mm"):
        csv_format.format_row(schema.POSITIONS, positions_row(x_mm="1.5"))


def test_header_line():
    assert csv_format.header_line(schema.PROBES) == "frame,t_s,probe,r,g,b,gray"
    assert csv_format.header_line(schema.RADIAL).startswith("track_id,frame,t_s,r_000,r_005,")


def test_csv_text_uses_lf_and_ends_with_one():
    text = csv_format.csv_text(schema.POSITIONS, [positions_row(), positions_row(frame=14)])
    assert "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")
    assert text.splitlines()[0] == ",".join(POSITIONS_NAMES)
    assert len(text.splitlines()) == 3
    text.encode(schema.CSV_ENCODING)


def test_csv_text_with_no_rows_is_the_header_only():
    """radial.csv of a run without fine tracks (X12): the file exists and holds its header."""
    assert csv_format.csv_text(schema.RADIAL, []) == ",".join(RADIAL_NAMES) + "\n"


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
    assert csv_format.format_row(schema.SHAPES, shapes_lost_row()) == ",".join(cells)


@pytest.mark.parametrize(
    ("columns", "dtypes"), [(schema.POSITIONS, POSITIONS_DTYPES), (schema.SHAPES, SHAPES_DTYPES)]
)
def test_integer_columns_stay_integer_when_a_row_is_lost(columns, dtypes):
    if columns is schema.POSITIONS:
        rows = [positions_row(flags=""), positions_row(frame=14, x_mm=NAN, y_mm=NAN, u_px=NAN, v_px=NAN,
                                                       area_mm2=NAN, visible=0, flags="LOST")]
    else:
        rows = [shapes_visible_row(), shapes_lost_row()]
    df = read_csv(csv_format.csv_text(columns, rows))
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
    df = read_csv(csv_format.csv_text(schema.POSITIONS, [positions_row(flags=""), positions_row(frame=14, flags="")]))
    flags = df["flags"].fillna("")
    assert flags.tolist() == ["", ""]
    assert not flags.str.contains("JUMP").any()
    df = read_csv(csv_format.csv_text(schema.POSITIONS, [positions_row(flags="JUMP;SIZE"), positions_row(flags="")]))
    assert df["flags"].fillna("").str.contains("JUMP").tolist() == [True, False]


def test_radial_row_round_trip():
    row = {name: 50.0 + i for i, name in enumerate(RADIAL_NAMES[3:])}
    row.update(track_id="A", frame=3, t_s=0.0125)
    lost = {name: NAN for name in RADIAL_NAMES[3:]}
    lost.update(track_id="A", frame=4, t_s=0.0166667)
    df = read_csv(csv_format.csv_text(schema.RADIAL, [row, lost]))
    assert df.shape == (2, 75)
    assert df["r_000"].tolist()[0] == 50.0 and df["r_355"].tolist()[0] == 50.0 + 71
    assert df.iloc[1, 3:].isna().all()
    assert pd.api.types.is_integer_dtype(df["frame"])


def test_probes_row():
    row = dict(frame=12, t_s=0.05, probe="LED1", r=200.0, g=100.5, b=50.25, gray=124.522)
    assert csv_format.format_row(schema.PROBES, row) == "12,0.0500000,LED1,200.000,100.500,50.250,124.522"


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
