"""The prose that README.txt and docs/OUTPUTS.md share, and the helpers that make it from the schema tables.

`core/schema_docs.py` builds the two texts and takes what it needs from here. Backticks mark names, formulas and
code: OUTPUTS.md keeps them, README.txt drops them (`_plain`). Every file, column, flag, key and number comes from
the tables of core/schema.py, so a new column or flag appears in both texts without editing prose. The names start
with an underscore because only schema_docs.py reads them. This module only builds text; it does no geometry and
reads no file. Units: every number is given with its unit, and the Units convention lists each column whose name
does not show it (SPEC 3).
"""

from __future__ import annotations

import textwrap

from outline_tracker.core.schema import (
    FILES, OUTLINE_POINTS, OUTLINES_META_KEY, POSITION_FLAGS, POSITIONS, POSITIONS_CSV, PROBES_CSV,
    RADIAL_CSV, SHAPE_FLAGS, SHAPES, SHAPES_CSV, TRACKER_FILE, Column, npz_key,
)

_WIDTH = 96


def _plain(text: str) -> str:
    return text.replace("`", "")


def _para(text: str, indent: int = 0, hang: int = 0) -> str:
    """`text` without code marks, wrapped to the README width; continuation lines are indented `hang` more."""
    return textwrap.fill(
        _plain(text), width=_WIDTH, initial_indent=" " * indent, subsequent_indent=" " * (indent + hang),
        break_long_words=False, break_on_hyphens=False,
    )


_ROW_TEXT = {
    POSITIONS_CSV: "one row per track and grid frame, from the track's first frame to its last",
    SHAPES_CSV: "the same rows as `positions.csv`",
    RADIAL_CSV: "one row per fine track and frame (the same rows as `positions.csv` for those tracks)",
    PROBES_CSV: "one row per probe and frame, for every frame of the clip (step 1, whatever the tracking step)",
    TRACKER_FILE: 'first line ",A,,,,," (a comma, the track name, then commas), then the column names, then one '
                  "row per tracked frame; written exactly as last week's files were",
}


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _radii_run(radii: list[Column]) -> str:  # `r_000`, `r_005`, ..., `r_355` (72 columns)
    return f"`{radii[0].name}`, `{radii[1].name}`, ..., `{radii[-1].name}` ({len(radii)} columns)"


# SPEC 3.4: the suffix that a column name carries for a unit, and the words for it ("" if no name shows it).
_UNITS = {"s": ("_s", "seconds"), "mm": ("_mm", "millimetres"), "mm^2": ("_mm2", "square millimetres"),
          "rad": ("_rad", "radians"), "px": ("_px", "image pixels"), "cells": ("", "model grid cells"),
          "0-255": ("", "the 0-255 range of image brightness")}


def _unit_not_in_name() -> str:
    """Prose on the columns whose unit their names do not show, by file and unit, read from the tables."""
    clauses = []
    for spec in FILES:
        groups: dict[str, list[Column]] = {}
        for c in spec.columns or []:
            suffix = _UNITS.get(c.unit, ("",))[0]
            if c.unit and not (suffix and c.name.endswith(suffix)):
                groups.setdefault(c.unit, []).append(c)
        parts = []
        for unit, cols in groups.items():
            radii = _radii(cols)
            names = [f"`{c.name}`" for c in cols if c not in radii] + ([_radii_run(radii)] if radii else [])
            parts.append(f"{_and(names)} {'is' if len(cols) == 1 else 'are'} in {_UNITS.get(unit, ('', unit))[1]}")
        if parts:
            clauses.append(f"in `{spec.name}`, {_and(parts)}")
    return "; ".join(clauses)


def _decimals(*units: str) -> str:
    """The decimals written for the float columns of these units, from the tables (they must agree)."""
    found = {c.fmt[1:-1] for f in FILES for c in f.columns or [] if c.dtype == "float" and c.unit in units}
    if len(found) != 1:
        raise ValueError(f"the columns with unit {units} are written with {sorted(found)} decimals, not one number")
    return found.pop()


def _conventions() -> list[str]:
    ratios = ", ".join(f"`{c.name}`" for c in SHAPES if c.dtype == "float" and not c.unit)
    suffixes = ", ".join(f"`{suffix}` {words}" for suffix, words in _UNITS.values() if suffix)
    return [
        f"Units: {suffixes}. Not every number has its unit in its column name: ratios ({ratios}) have no unit, and "
        f"these columns have a unit that their names do not show: {_unit_not_in_name()}. The description of every "
        "column below gives its unit.",
        "Image coordinates (`u_px`, `v_px`; Tracker's `pixelx`, `pixely`): the origin is the top-left corner of the "
        "frame, u grows to the right, v downward, and the pixel in column c and row r has its center at "
        "`(c + 0.5, r + 0.5)`.",
        "Your axes (`x_mm`, `y_mm`): the origin and the direction of +x were set in the calibration, and y points "
        "up. With k the scale in mm per pixel, alpha the angle of +x counterclockwise on screen from the "
        "image's rightward direction, and (u0, v0) the origin: `x = k ((u - u0) cos alpha - (v - v0) sin alpha)` "
        "and `y = -k ((u - u0) sin alpha + (v - v0) cos alpha)`. Angles in your axes are counterclockwise.",
        "Time: `t_s = frame / fps_true`, the frame rate you gave or measured; the frame rate stored in the video "
        "file is never used. Start frame and step choose frames but never shift `t_s`.",
        "Frames: counted from 0 in a sequential decode of the file, as Tracker counts. The tracked frames are "
        "the grid `start + n * step`, so all tracks share frames.",
        "Rows: for each track, every grid frame from its first to its last, sorted by `track_id` (text order) "
        "and `frame`. A lost frame keeps its row, with its `track_id`, `frame` and `t_s`: the measured numbers are "
        "empty cells, except `visible = 0`, `n_components = 0` and `shape_ok = 0` (integers stay integers), and "
        "`flags` holds `LOST`.",
        f"Numbers: written with {_decimals('s')} decimals for seconds (`_s`), {_decimals('mm', 'mm^2', 'rad', '')} "
        f"decimals for `_mm`, `_mm2`, `_rad` and ratios, and {_decimals('px', 'cells', '0-255')} decimals for "
        "`_px`, pixel and cell counts and probe means. A number that does not exist is an empty cell, never the "
        "text `nan`.",
        "Text files: UTF-8 with LF line ends on every platform. The files in the model folder keep your "
        "computer's line ends, as last week.",
    ]


_SHAPE_FILES = [
    "`radial.csv` and `outlines.npz` hold the fine tracks by default: the outline of a coarse track is a blob of "
    "a few grid cells (flag `LOWRES`). The setting `shape_files_for_coarse` (off by default) adds the coarse "
    "tracks. A fine track comes from `--fine` on the command line, or from the object's fine mode in the "
    "app. With no fine track, `radial.csv` holds only its header line and `outlines.npz` only the key `meta`, "
    "so both files can always be read.",
    "The radial profile `r_ddd` is the distance from the body center to the farthest crossing of the ray at "
    "ddd degrees (counterclockwise from the head direction) with the outline. For an outline that is "
    "star-shaped around the body center this is the boundary itself; otherwise it is the outer envelope. "
    "`r_000` is the head radius.",
]
_OUTLINES_INTRO = (
    f"`outlines.npz` has these arrays for every track, named `{npz_key('<id>', '<name>')}`, for example "
    f"`{npz_key('A', 'xy_mm')}` (`n` is the number of tracked frames, `N` the number of points):"
)
_META_INTRO = f"`{OUTLINES_META_KEY}` is a JSON text with these fields:"
_FLAGS_INTRO = (
    "The column `flags` of `positions.csv` and `shapes.csv` holds the quality codes of the row, separated by "
    "`';'` in the order below (empty if none). It is the same text in both files."
)
_FLAGS_ADVICE = (
    'Filter on the codes you care about, not on "flags is empty": at dish scale every row of a coarse track '
    "carries `LOWRES`, and a track without a head click carries `HEADGUESS` on every frame."
)
_IN_GIT, _NOT_IN_GIT = "In git (small files):", "Not in git (large; share them through Drive):"
_GIT_EXTRA = "`calibration.json` (if you exported one)"
_GIT_NOTE = (
    "`outlines.npz` is the only full-shape output, so it must be committed. A `.gitignore` that excludes `*.npz` "
    "hides it: add the line `!**/outlines.npz` below the line `*.npz`."
)
_EXPORT_TEXT = (
    "If a run stopped early, or a file is missing, run this command; it rebuilds every file from what was "
    "tracked, without running the model:"
)
_EXPORT_COMMAND = "outline-tracker export <run folder>/session.json"
_LOCKED_TEXT = (
    "If a file is locked (for example a CSV open in Excel), the tool writes the new data next to it as "
    "`<name>.new<extension>`, for example `positions.new.csv`, and tells you. Close the program that holds the "
    "file and export again."
)


def _scopes() -> list[str]:
    return [f"Flags that concern positions: {', '.join(POSITION_FLAGS)}.",
            f"Flags that concern only shapes: {', '.join(SHAPE_FLAGS)}."]


def _reading_code() -> list[str]:
    xy_key = npz_key("A", "xy_mm")
    return [
        "import json",
        "import numpy as np",
        "import pandas as pd",
        'df = pd.read_csv("positions.csv")',
        'df["flags"] = df["flags"].fillna("")          # an empty cell is read as NaN',
        'jumps = df[df["flags"].str.contains("JUMP")]',
        'z = np.load("outlines.npz")',
        f'xy = z["{xy_key}"]                            # [n, {OUTLINE_POINTS}, 2], mm',
        'meta = json.loads(str(z["meta"]))',
    ]


def _reading_note() -> str:
    ints = [c.name for columns in (POSITIONS, SHAPES) for c in columns if c.dtype == "int"]
    return (
        'Write `df["flags"]`, not `df.flags`: pandas has its own attribute called flags. Lost frames are empty '
        f"cells, which pandas reads as NaN; the integer columns ({', '.join(dict.fromkeys(ints))}) have no empty "
        "cells and stay integers."
    )


def _radii(columns: list[Column]) -> list[Column]:
    return [c for c in columns if c.name.startswith("r_")]
