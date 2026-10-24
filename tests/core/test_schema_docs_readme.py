"""Tests of outline_tracker/core/schema_docs.py: the wording of the README text and the units both texts give.

The tests check the sentences that a review of the README text found wrong or missing, and that every unit
that a column name does not show is given under Units, in README.txt and in docs/OUTPUTS.md. Expected values
are typed from the specification or computed from the schema tables, never copied from the output under test.

The rest of the tests of the module are beside this file: what must not change when the builders moved out
of core/schema.py in tests/core/test_schema_docs.py, outputs_markdown() and the committed docs/OUTPUTS.md in
tests/core/test_schema_docs_outputs.py. The fixtures `schema_docs`, `readme` and `outputs` are in
tests/core/conftest.py; what this file and the last one share is in tests/support/schema_docs_helpers.py.
"""

import dataclasses
import re

import pytest
from helpers import normalized
from schema_docs_helpers import RADIAL_COLUMNS, cells, md_section, plain

from outline_tracker.core import csv_format, schema, schema_prose


# ---------------------------------------------------------------------------------------------
# Wording slips that the review of A05 noted in the README text (fixed in the second commit of B6a)

NAN = float("nan")


def convention(readme: str, label: str) -> str:
    """The item of the CONVENTIONS section that starts with `label:`, wrapped lines joined."""
    lines = readme.splitlines()
    start = lines.index("== CONVENTIONS ==") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("== ") and i > start)
    items: list[str] = []
    for line in lines[start:end]:
        if line.startswith("  ") and not line.startswith("    "):  # an item starts at two spaces
            items.append(line.strip())
        elif line.strip() and items:
            items[-1] += " " + line.strip()
    [item] = [i for i in items if i.startswith(f"{label}:")]
    return item


def has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) is not None


def lost_row(columns: list[schema.Column]) -> dict:
    """A lost frame, built by hand as SPEC 8.2 and 8.4 describe it: floats empty (NaN), integers 0."""
    row: dict = {c.name: NAN for c in columns if c.dtype == "float"}
    row.update({c.name: 0 for c in columns if c.dtype == "int"})
    row.update(track_id="B", frame=40, t_s=0.1666667, mode="coarse", flags="LOST")
    return row


def test_the_readme_says_that_a_lost_row_keeps_frame_and_t_s(readme):
    """The numbers that a lost row keeps are found from the real cells, not typed: format_row writes the
    row and the cells that are not empty are the kept ones. The Rows convention must name each one."""
    rows = convention(readme, "Rows")
    kept_somewhere = set()
    for columns in (schema.POSITIONS, schema.SHAPES):
        cells = dict(zip((c.name for c in columns), csv_format.format_row(columns, lost_row(columns)).split(",")))
        kept = [c.name for c in columns if c.dtype in ("int", "float") and cells[c.name] != ""]
        assert {"frame", "t_s"} <= set(kept)  # SPEC 8.2: a lost frame keeps its row
        for name in kept:
            assert has_word(rows, name), f"{name} stays filled in a lost row but Rows does not say so"
        kept_somewhere.update(kept)
    assert kept_somewhere == {"frame", "t_s", "visible", "n_components", "shape_ok"}
    assert "every number is an empty cell" not in normalized(readme)


def test_the_readme_does_not_say_that_every_number_has_its_unit_in_its_name(readme):
    flat = normalized(readme)
    assert "The unit of every number is in its column name" not in flat
    units = convention(readme, "Units")
    for name in ("r", "g", "b", "gray"):  # SPEC 8.7: probe means on the 0-255 scale
        assert has_word(units, name), name
    assert "0-255" in units
    assert "ratio" in units and "no unit" in units
    for name in ("eccentricity", "core_frac", "solidity", "circularity", "largest_fraction"):
        assert has_word(units, name), name  # the ratios, which have no unit


# SPEC 3.4: units appear in column names as these suffixes (the unit of the tables, then its suffix).
UNIT_SUFFIX = {"s": "_s", "mm": "_mm", "mm^2": "_mm2", "px": "_px", "rad": "_rad"}
# What the clause about a column must say of its unit.
UNIT_WORD = {"s": "seconds", "mm": "millimetres", "px": "image pixels", "cells": "model grid cells",
             "0-255": "0-255 range"}
TRACKER_FILE = "<model>/<id>.csv"


def unit_hidden(files: list[schema.FileSpec]) -> list[tuple[str, schema.Column]]:
    """(file, column) of every table column that has a unit that its name does not show: the unit has no
    suffix (cells, 0-255), or the name does not end with the suffix of its unit."""
    return [(f.name, c) for f in files for c in f.columns or []
            if c.unit and (c.unit not in UNIT_SUFFIX or not c.name.endswith(UNIT_SUFFIX[c.unit]))]


def units_item(text: str) -> str:
    """The Units item of the conventions, as plain text, from README.txt or from docs/OUTPUTS.md."""
    if text.startswith("OUTLINE TRACKER"):
        return convention(text, "Units")
    [item] = [line[2:] for line in md_section(text, "## Conventions") if line.startswith("- Units:")]
    return plain(item)


def test_the_finder_of_hidden_units_sees_the_groups_that_the_spec_tables_show():
    """SPEC 8.2 to 8.7 and 8.3: these columns carry a unit that their names do not show. Without them the
    tests below would pass for the wrong reason."""
    expected = {(TRACKER_FILE, "t"): "s", (TRACKER_FILE, "x"): "mm", (TRACKER_FILE, "y"): "mm",
                (TRACKER_FILE, "pixelx"): "px", (TRACKER_FILE, "pixely"): "px",
                ("shapes.csv", "px_along_major"): "px", ("shapes.csv", "cells_along_major"): "cells",
                ("probes.csv", "r"): "0-255", ("probes.csv", "g"): "0-255", ("probes.csv", "b"): "0-255",
                ("probes.csv", "gray"): "0-255", **{("radial.csv", name): "mm" for name in RADIAL_COLUMNS}}
    found = {(f, c.name): c.unit for f, c in unit_hidden(schema.FILES)}
    assert expected.items() <= found.items()
    assert ("positions.csv", "x_mm") not in found and ("positions.csv", "t_s") not in found  # named by their unit


@pytest.mark.parametrize("source", ["readme", "outputs"])
def test_every_unit_that_a_column_name_does_not_show_is_given_under_units(source, readme, outputs):
    """The README says that the unit of a number is in its column name except for what Units lists, so Units
    must list every column whose unit is not in its name, with its unit, and name the file it belongs to."""
    units = units_item(readme if source == "readme" else outputs)
    for file, col in unit_hidden(schema.FILES):
        radius = col.name in RADIAL_COLUMNS  # the 72 radius columns are named by the first and the last
        name = RADIAL_COLUMNS[0] if radius else col.name
        word = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"  # the name as a whole word
        found = re.search(rf"{word}.*?\bin ([^,;.]*)", units)  # the first "in <unit words>" after it
        assert found, f"Units does not mention {name} of {file}, whose unit {col.unit} its name does not show"
        assert UNIT_WORD[col.unit] in found.group(1), f"Units gives {name} the unit {found.group(1)!r}, not {col.unit}"
        assert file in units[: found.start()], f"Units does not name {file} before {name}"
        if radius:
            assert has_word(units, RADIAL_COLUMNS[-1]) and "(72 columns)" in units


def test_a_new_column_with_a_hidden_unit_is_listed_under_units(schema_docs, monkeypatch):
    """The exceptions are read from the tables: a column that this test adds, with a unit its name does not
    show, appears in Units of both texts, and one with a unit that its name shows does not."""
    odd = schema.Column("zz_odd", "float", "mm", ".6f", "a column that only this test adds")
    shown = schema.Column("zz_shown_mm", "float", "mm", ".6f", "a column that only this test adds")
    files = [dataclasses.replace(f, columns=[*f.columns, odd, shown]) if f.name == "shapes.csv" else f
             for f in schema.FILES]
    monkeypatch.setattr(schema_docs, "FILES", files)
    monkeypatch.setattr(schema_prose, "FILES", files)  # the Units sentence is made in schema_prose
    for units in (units_item(schema_docs.readme_text()), units_item(schema_docs.outputs_markdown())):
        assert re.search(r"in shapes\.csv, .*\bzz_odd is in millimetres", units), units
        assert "zz_shown_mm" not in units


def test_core_frac_is_described_for_the_fallback_case(readme):
    """SPEC 7.3: if the opening removes more than half the area the full mask is the core, and the stored
    core_frac stays the fraction the opening left (tests/measure/test_measure_core.py), so it is below 0.5."""
    shapes = {c.name: c.meaning for c in schema.SHAPES}["core_frac"]
    stored = {k.name: k.meaning for k in schema.RESULTS_KEYS}["core_frac"]
    for meaning in (shapes, stored):
        text = normalized(meaning)
        assert "fallback" in text and "0.5" in text and "not 1" in text, text
    assert normalized(shapes) != "core area / full mask area"
    assert normalized(stored) != "core area / mask area"
    assert normalized(shapes) in normalized(readme)  # the README is built from the table


def test_px_along_major_is_described_for_the_largest_piece(readme, outputs):
    """Decision 26 of docs/ROADMAP.md: the size checks are of the largest piece of the mask, so that a few
    stray pixels far from the object cannot lengthen them. The three axis columns stay on the full mask.

    L1 is the name that major_mm gives to the full mask's axis. The two size columns do not use it: read
    with major_mm, "L1 / k" would say px_along_major = major_mm / k, which is false on a mask with stray
    pixels. cells_along_major is given from px_along_major instead."""
    meanings = {c.name: normalized(c.meaning) for c in schema.SHAPES}
    assert "largest piece" in meanings["px_along_major"] and "full mask" not in meanings["px_along_major"]
    for name in ("major_mm", "minor_mm", "eccentricity"):
        assert "full mask" in meanings[name], name
    assert "L1 =" in meanings["major_mm"]  # where the name is defined, for the full mask
    assert "L1" not in meanings["px_along_major"]
    assert "px_along_major / cell" in meanings["cells_along_major"]
    assert "L1" not in meanings["cells_along_major"]
    for name in ("px_along_major", "cells_along_major"):
        assert meanings[name] in normalized(readme), name  # both texts are built from the table
        assert meanings[name] in plain(outputs), name
