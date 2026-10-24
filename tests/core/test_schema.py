"""Tests of outline_tracker/core/schema.py, the single source of truth for every output file (SPEC 8, 9): its tables.

Expected values come from the specification: column lists typed from SPEC 8.2, 8.4, 8.5, 8.7 and the
docs/design/SPEC-v0.1.md tables parsed directly. Nothing is copied from the output of the code under test.

The rest of the tests of the module are in tests/core/test_schema_texts.py: the rows written as CSV text
and read back, and README.txt. The column lists and dtypes that both files use are in
tests/support/schema_helpers.py.
"""

import re
import subprocess
import sys

import numpy as np
import pytest
from layout_helpers import REPO
from layout_helpers import path as table_path
from schema_helpers import (
    NAN,
    POSITIONS_DTYPES,
    POSITIONS_NAMES,
    PROBES_DTYPES,
    PROBES_NAMES,
    RADIAL_DTYPES,
    RADIAL_NAMES,
    SHAPES_DTYPES,
    SHAPES_NAMES,
    names,
)

from outline_tracker.core import csv_format, schema, tracker_io


# ---------------------------------------------------------------------------------------------
# Flags, files and results keys typed from the specification

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
    """Lines of docs/design/SPEC-v0.1.md from the heading that starts with `heading_start` to the next heading."""
    lines = table_path("spec").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading_start))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return lines[start + 1 : end]


def spec_table(heading_start: str) -> list[list[str]]:
    """Body rows of the first markdown table of a section of docs/design/SPEC-v0.1.md, as lists of stripped cells."""
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
    assert [csv_format.format_row(schema.TRACKER_COLUMNS, r) for r in rows] == lines[2:]


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
# The module stays light

def test_importing_schema_loads_neither_torch_nor_qt():
    """In a subprocess: pytest-qt has already imported PySide6 into this one."""
    code = (
        "import sys; import outline_tracker.core.schema; "
        "bad = sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph', 'pandas') if m in sys.modules); "
        "print(bad)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
