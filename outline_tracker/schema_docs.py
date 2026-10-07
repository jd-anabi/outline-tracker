"""The two texts that describe a run folder, built from the tables of `outline_tracker.schema`.

`readme_text()` is README.txt (SPEC 8.11), written into every run folder. `outputs_markdown()` is docs/OUTPUTS.md (SPEC
8.1 and 16), the contract for the analysis template; the committed file is its output, and a test fails when it is
stale. Regenerate it after a change of the schema with `uv run python -m outline_tracker.schema_docs docs/OUTPUTS.md`.

Both texts say the same things: the prose is written once below, with code marks (backticks) that the plain-text README
drops. Every file, column, flag, key and number comes from the schema tables, so a new column or flag appears in both
without editing prose. This module only builds text; it does no geometry and reads no file. It was split out of
schema.py (SPEC 12: split past about 400 lines); `outline_tracker.schema.readme_text` is still importable and is the
same function. Units: every number is given with its unit, and the Units convention lists each column whose name does
not show it (SPEC 3).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from outline_tracker.schema import (
    FILES, FLAG_INFO, OUTLINE_POINTS, OUTLINES_KEYS, OUTLINES_META_FIELDS, OUTLINES_META_KEY, OUTLINES_NPZ,
    POSITION_FLAGS, POSITIONS, POSITIONS_CSV, PROBES_CSV, RADIAL_CSV, README_TXT, RESULTS_KEYS, RESULTS_NPZ,
    RESULTS_VERSION, RESULTS_VERSION_KEY, SHAPE_FLAGS, SHAPES, SHAPES_CSV, TRACKER_FILE, ArrayKey, Column,
    FileSpec, npz_key,
)

# --------------------------------------------------------------------------- prose shared by both texts
# Backticks mark names, formulas and code: OUTPUTS.md keeps them, README.txt drops them (_plain).

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


# --------------------------------------------------------------------------- README.txt (SPEC 8.11)


def _entry(col: Column, name: str | None = None) -> str:
    """One README line for a column; `name` replaces the column's name (for the 72 radial columns)."""
    kind = col.dtype + (f", {col.unit}" if col.unit else "")
    if col.dtype == "float":
        kind += f", {col.fmt[1:-1]} decimals"
    return _para(f"{name or col.name}: {kind}. {col.meaning}", indent=2, hang=4)


def _columns_section() -> list[str]:
    lines: list[str] = []
    for spec in (f for f in FILES if f.columns is not None):
        columns = spec.columns or []
        rows = _ROW_TEXT[spec.name] + ("; see SHAPE FILES" if spec.name == RADIAL_CSV else "")
        radii = _radii(columns)
        lines += ["", spec.name, _para(rows, indent=2), ""] + [_entry(c) for c in columns if c not in radii]
        if radii:
            lines.append(_entry(radii[0], _radii_run(radii)))
            lines.append(_para("The columns, in order: " + " ".join(c.name for c in radii), indent=2, hang=4))
    return lines


def _flags_section() -> list[str]:
    lines = [_para(_FLAGS_INTRO, indent=2), "", *(f"  {line}" for line in _scopes()), ""]
    lines += [_para(f"{i.code}: {i.condition}. Meaning: {i.meaning}.", indent=2, hang=4) for i in FLAG_INFO.values()]
    return lines + ["", _para(_FLAGS_ADVICE, indent=2)]


def _shape_files_section() -> list[str]:
    lines: list[str] = []
    for text in _SHAPE_FILES:
        lines += [_para(text, indent=2), ""]
    lines += [_para(_OUTLINES_INTRO, indent=2), ""]
    for key in OUTLINES_KEYS:
        shape = ", ".join(str(s) for s in key.shape)
        unit = f", {key.unit}" if key.unit else ""
        name = npz_key("<id>", key.name)
        lines.append(_para(f"{name}: {key.dtype} [{shape}]{unit}. {key.meaning}", indent=4, hang=4))
    lines += ["", _para(_META_INTRO, indent=2)]
    return lines + [_para(f"{field}: {meaning}", indent=4, hang=4) for field, meaning in OUTLINES_META_FIELDS.items()]


def _git_section() -> list[str]:
    lines = [f"  {_IN_GIT}"] + [f"    {f.name}" for f in FILES if f.in_git]
    lines += [f"    {_plain(_GIT_EXTRA)}", "", _para(_GIT_NOTE, indent=2), "", f"  {_NOT_IN_GIT}"]
    return lines + [f"    {f.name}" for f in FILES if not f.in_git]


def readme_text() -> str:
    """The text of README.txt: ASCII, LF line ends, ending with one line end.

    Describes every file and column (with units, decimals and meaning), the conventions of SPEC 3 (units
    and coordinate frames), the flags of SPEC 9 and what goes into git; the lists come from the tables.
    """
    out = [
        "OUTLINE TRACKER: what is in this folder",
        "",
        _para("This folder holds one run of Outline Tracker on one video. The tool writes this file; do not "
              "edit it. In the CSV files the unit of a number is in its column name, except for the numbers that "
              "Units, under CONVENTIONS, lists."),
    ]

    def section(title: str, lines: list[str]) -> None:
        out.extend(["", f"== {title} ==", ""] + lines)

    section("FILES", [_para(f"{f.name}: {f.meaning}", indent=2, hang=4) for f in FILES])
    section("CONVENTIONS", [_para(text, indent=2, hang=2) for text in _conventions()])
    section("COLUMNS", _columns_section()[1:])
    section("FLAGS", _flags_section())
    section("SHAPE FILES", _shape_files_section())
    section("READING THE FILES", ["  " + line for line in _reading_code()] + ["", _para(_reading_note(), indent=2)])
    section("WHAT GOES INTO GIT", _git_section())
    section("IF SOMETHING WENT WRONG", [
        _para(_EXPORT_TEXT, indent=2), "", f"    {_EXPORT_COMMAND}", "", _para(_LOCKED_TEXT, indent=2),
    ])
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- docs/OUTPUTS.md (SPEC 8.1, 16)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table, ended by a blank line; a bar inside a cell is escaped."""
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
    return [line(header), line(["---"] * len(header)), *map(line, rows), ""]


def _column_rows(columns: list[Column]) -> list[list[str]]:
    """One row per column: name, type (floats with their decimals), unit ("none" for ratios), meaning. The
    72 radius columns share one row, as in README.txt."""
    def row(c: Column) -> list[str]:
        kind = c.dtype + (f", {c.fmt[1:-1]} decimals" if c.dtype == "float" else "")
        return [f"`{c.name}`", kind, c.unit or "none", c.meaning]
    radii = _radii(columns)
    rows = [row(c) for c in columns if c not in radii]
    if radii:
        rows.append([_radii_run(radii), *row(radii[0])[1:]])
    return rows


def _key_table(keys: list[ArrayKey]) -> list[str]:
    rows = [[f"`{npz_key('<id>', k.name)}`", k.dtype, f"[{', '.join(str(n) for n in k.shape)}]", k.unit or "none",
             k.meaning] for k in keys]
    return _table(["key", "dtype", "shape", "unit", "meaning"], rows)


def _file_body(spec: FileSpec) -> list[str]:
    columns = spec.columns or []
    if spec.name == RADIAL_CSV:
        names = ", ".join(f"`{c.name}`" for c in _radii(columns))
        notes = [line for text in _SHAPE_FILES for line in (text, "")]
        return [f"The {len(_radii(columns))} radius columns, in order: {names}", "", *notes]
    if spec.name == OUTLINES_NPZ:
        return [_OUTLINES_INTRO, "", *_key_table(OUTLINES_KEYS),
                f"Every outline has `N` = {OUTLINE_POINTS} points. Which tracks the file holds is explained under "
                f"`{RADIAL_CSV}`.", "", _META_INTRO, "",
                *(f"- `{field}`: {meaning}" for field, meaning in OUTLINES_META_FIELDS.items()), ""]
    if spec.name == RESULTS_NPZ:
        return [f"The arrays of each track are stored under the keys `{npz_key('<id>', '<name>')}`; `n` is the "
                "number of tracked frames of the track, and `U6` is text of up to 6 characters. The key "
                f"`{RESULTS_VERSION_KEY}` holds the format version, a whole number; it is currently "
                f"{RESULTS_VERSION}.", "", *_key_table(RESULTS_KEYS)]
    return []


def _file_section(spec: FileSpec) -> list[str]:
    what = spec.meaning
    if spec.name == README_TXT:  # the meaning in the tables says "this file", which is this page here
        what = "the same information as this page, as plain text; written into every run folder"
    lines = [f"### `{spec.name}`", "", f"- **What it holds:** {what}."]
    if spec.name in _ROW_TEXT:
        lines.append(f"- **Rows:** {_ROW_TEXT[spec.name]}.")
    lines += [f"- **In git:** {'yes' if spec.in_git else 'no (large; share it through Drive)'}.", ""]
    if spec.columns is not None:
        lines += _table(["column", "type", "unit", "meaning"], _column_rows(spec.columns))
    return lines + _file_body(spec)


def _glossary() -> list[str]:
    terms = [
        ("Tracker", "the video-analysis program you used last week (physlets.org/tracker); this tool follows its "
                    "pixel and file conventions."),
        ("frame", "one picture of the video, counted from 0 as Tracker counts."),
        ("track", "one animal followed through the video. Its id is a capital letter such as `A`; later pieces "
                  "of the same animal are `A2`, `A3`."),
        ("mask", "the pixels of one frame that the model says belong to the animal."),
        ("centroid", "the average position of all pixels of the mask, like its balance point. It is the official "
                     "position of the animal."),
        ("outline", f"the line around the mask, kept as {OUTLINE_POINTS} points per frame for fine tracks."),
        ("coarse and fine", "in coarse mode the model looks at the whole frame (or the dish); in fine mode it "
                            "looks at a small crop that follows the animal, which gives a sharper outline."),
        ("core", "the mask with thin parts such as antennae cut away; its centroid is the body center."),
        ("heading", "the direction from the body center to the head, as an angle in your axes."),
        ("empty cell", "how a CSV file says that a number does not exist (for example, the animal is lost on that "
                       "frame). pandas reads it as `NaN`, short for not a number."),
    ]
    return [f"- **{term}**: {text}" for term, text in terms] + [""]


def outputs_markdown() -> str:
    """The text of docs/OUTPUTS.md: ASCII, LF line ends, ending with one line end.

    One section per file of SPEC 8.1 in that order (a table of columns or array keys where the file has
    them), the conventions of SPEC 3 (pixel centers at +0.5, mm, y up, `t_s = frame / fps_true`, the frame
    grid), the flags of SPEC 9, the recipe for reading the files and what goes into git (SPEC 8.13). Every
    file, column, flag, key and number is read from the schema tables; units are given for every quantity.
    """
    flag_rows = [[f"`{i.code}`", i.scope, i.condition, i.meaning] for i in FLAG_INFO.values()]
    out = [
        '<!-- Generated from outline_tracker/schema.py by "uv run python -m outline_tracker.schema_docs '
        'docs/OUTPUTS.md". Do not edit by hand. -->',
        "",
        "# Outline Tracker: the files of a run",
        "",
        "This page describes every file that Outline Tracker writes for one run on one video: its columns, their "
        "units, and what the numbers mean. It is the contract between the tracker and the analysis template, which "
        "may rely on the names, the order and the formats written here.",
        "",
        "It is generated from the schema of the tool, and a test fails when it is out of date, so do not edit it by "
        "hand. Every run folder holds the same information as plain text in `README.txt`.",
        "",
        "## Words used on this page", "", *_glossary(),
        "## Conventions", "", *(f"- {text}" for text in _conventions()), "",
        "## The files", "",
        "The run folder holds these files, in this order. `<model>` is the name of the model, for example "
        "`edgetam`, and `<id>` is the track id. The type `str` is text, `int` a whole number and `float` a number "
        "with decimals.", "",
    ]
    for spec in FILES:
        out += _file_section(spec)
    out += ["## Quality flags", "", _FLAGS_INTRO, "", *(f"- {line}" for line in _scopes()), "",
            *_table(["code", "concerns", "condition", "meaning"], flag_rows), _FLAGS_ADVICE, "",
            "## Reading the files", "", "To read the main files with Python:", "", "```python", *_reading_code(),
            "```", "", _reading_note(), "",
            "## What goes into git", "", f"**{_IN_GIT}**", "", *(f"- `{f.name}`" for f in FILES if f.in_git),
            f"- {_GIT_EXTRA}", "", _GIT_NOTE, "", f"**{_NOT_IN_GIT}**", "",
            *(f"- `{f.name}`" for f in FILES if not f.in_git), "",
            "## If something went wrong", "", _EXPORT_TEXT, "", "```", _EXPORT_COMMAND, "```", "", _LOCKED_TEXT]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """`python -m outline_tracker.schema_docs OUTPUT.md` writes `outputs_markdown()` to that file as UTF-8 with
    LF line ends on every platform. Returns the exit code: 0, or 2 (with a usage line) for a wrong call."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m outline_tracker.schema_docs OUTPUT.md", file=sys.stderr)
        return 2
    Path(args[0]).write_bytes(outputs_markdown().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
