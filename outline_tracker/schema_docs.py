"""The text of README.txt (SPEC 8.11), built from the tables of `outline_tracker.schema`.

The text is generated from the columns, flags, files and npz keys, so a new column or flag appears in
it without editing prose. This module only builds text; it does no
geometry and reads no file. It was split out of schema.py, which had grown past the 400 lines SPEC 12
asks for; `outline_tracker.schema.readme_text` still works and is this function.

Units and coordinates: every number the texts mention carries its unit as the schema tables give it
(_s seconds, _mm millimetres, _mm2 square millimetres, _rad radians, _px image pixels). Image
coordinates (u, v) follow Tracker (pixel centers at +0.5, v down); world coordinates (x, y) are in mm in
the user's axes with y up (SPEC 3).
"""

from __future__ import annotations

import textwrap

from outline_tracker.schema import (
    FILES,
    FLAG_INFO,
    OUTLINES_KEYS,
    OUTLINES_META_FIELDS,
    OUTLINES_META_KEY,
    POSITION_FLAGS,
    POSITIONS_CSV,
    PROBES,
    PROBES_CSV,
    RADIAL_CSV,
    SHAPE_FLAGS,
    SHAPES,
    SHAPES_CSV,
    TRACKER_FILE,
    Column,
)

# --------------------------------------------------------------------------- README.txt (SPEC 8.11)

_WIDTH = 96


def _para(text: str, indent: int = 0, hang: int = 0) -> str:
    """`text` wrapped to the README width; continuation lines are indented `hang` more spaces."""
    return textwrap.fill(
        text, width=_WIDTH, initial_indent=" " * indent, subsequent_indent=" " * (indent + hang),
        break_long_words=False, break_on_hyphens=False,
    )


_ROW_TEXT = {
    POSITIONS_CSV: "one row per track and grid frame, from the track's first frame to its last",
    SHAPES_CSV: "the same rows as positions.csv",
    RADIAL_CSV: "one row per fine track and frame (the same rows as positions.csv for those tracks); "
                "see SHAPE FILES",
    PROBES_CSV: "one row per probe and frame, for every frame of the clip (step 1, whatever the tracking step)",
    TRACKER_FILE: 'first line ",A,,,,," (a comma, the track name, then commas), then the column names, then one '
                  "row per tracked frame; written exactly as last week's files were",
}


def _and(names: list[str]) -> str:
    """'a, b and c' from a list of names (text only)."""
    return ", ".join(names[:-1]) + " and " + names[-1]


# The numbers whose unit is not spelled in their column name (SPEC 8.4, 8.7): ratios, the 0-255 probe
# means, and the two counts of camera pixels and grid cells.
_RATIOS = [c.name for c in SHAPES if c.dtype == "float" and not c.unit]
_MEANS = [c.name for c in PROBES if c.unit == "0-255"]
_COUNTS = [c.name for c in SHAPES if c.unit in ("px", "cells") and not c.name.endswith("_px")]

_CONVENTIONS = [
    "Units: _s seconds, _mm millimetres, _mm2 square millimetres, _rad radians, _px image pixels. Not every "
    f"number has a unit in its column name: the probe means {_and(_MEANS)} are on the 0-255 scale of image "
    f"brightness, ratios ({', '.join(_RATIOS)}) have no unit, and {_and(_COUNTS)} count camera pixels and "
    "model grid cells.",
    "Image coordinates (u_px, v_px; Tracker's pixelx, pixely): the origin is the top-left corner of the "
    "frame, u grows to the right, v downward, and the pixel in column c and row r has its center at "
    "(c + 0.5, r + 0.5).",
    "Your axes (x_mm, y_mm): the origin and the direction of +x were set in the calibration, and y points "
    "up. With k the scale in mm per pixel, alpha the angle of +x counterclockwise on screen from the "
    "image's rightward direction, and (u0, v0) the origin: x = k ((u - u0) cos alpha - (v - v0) sin alpha) "
    "and y = -k ((u - u0) sin alpha + (v - v0) cos alpha). Angles in your axes are counterclockwise.",
    "Time: t_s = frame / fps_true, the frame rate you gave or measured; the frame rate stored in the video "
    "file is never used. Start frame and step choose frames but never shift t_s.",
    "Frames: counted from 0 in a sequential decode of the file, as Tracker counts. The tracked frames are "
    "the grid start + n * step, so all tracks share frames.",
    "Rows: for each track, every grid frame from its first to its last, sorted by track_id (text order) "
    "and frame. A lost frame keeps its row, with its track_id, frame and t_s: the measured numbers are empty "
    "cells, except visible = 0, n_components = 0 and shape_ok = 0 (integers stay integers), and flags holds "
    "LOST.",
    "Numbers: written with 7 decimals for seconds (_s), 6 decimals for _mm, _mm2, _rad and ratios, and "
    "3 decimals for _px, pixel and cell counts and probe means. A number that does not exist is an empty "
    "cell, never the text nan.",
    "Text files: UTF-8 with LF line ends on every platform. The files in the model folder keep your "
    "computer's line ends, as last week.",
]

_SHAPE_FILES = [
    "radial.csv and outlines.npz hold the fine tracks by default: the outline of a coarse track is a blob of "
    "a few grid cells (flag LOWRES). The setting shape_files_for_coarse (off by default) adds the coarse "
    "tracks. A fine track comes from --fine on the command line, or from the object's fine mode in the "
    "app. With no fine track, radial.csv holds only its header line and outlines.npz only the key meta, "
    "so both files can always be read.",
    "The radial profile r_ddd is the distance from the body center to the farthest crossing of the ray at "
    "ddd degrees (counterclockwise from the head direction) with the outline. For an outline that is "
    "star-shaped around the body center this is the boundary itself; otherwise it is the outer envelope. "
    "r_000 is the head radius.",
]

_READING_CODE = [
    "import json",
    "import numpy as np",
    "import pandas as pd",
    'df = pd.read_csv("positions.csv")',
    'df["flags"] = df["flags"].fillna("")          # an empty cell is read as NaN',
    'jumps = df[df["flags"].str.contains("JUMP")]',
    'z = np.load("outlines.npz")',
    'xy = z["A__xy_mm"]                            # [n, 128, 2], mm',
    'meta = json.loads(str(z["meta"]))',
]


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
        lines += ["", spec.name, _para(_ROW_TEXT[spec.name], indent=2), ""]
        if spec.name == RADIAL_CSV:
            radii = [c for c in columns if c.name.startswith("r_")]
            lines += [_entry(c) for c in columns if c not in radii]
            lines.append(_entry(radii[0], f"{radii[0].name}, {radii[1].name}, ..., {radii[-1].name} "
                                          f"({len(radii)} columns)"))
            lines.append(_para("The columns, in order: " + " ".join(c.name for c in radii), indent=2, hang=4))
        else:
            lines += [_entry(c) for c in columns]
    return lines


def _flags_section() -> list[str]:
    lines = [
        _para("The column flags of positions.csv and shapes.csv holds the quality codes of the row, separated by "
              "';' in the order below (empty if none). It is the same text in both files.", indent=2),
        "",
        f"  Flags that concern positions: {', '.join(POSITION_FLAGS)}.",
        f"  Flags that concern only shapes: {', '.join(SHAPE_FLAGS)}.",
        "",
    ]
    lines += [_para(f"{i.code}: {i.condition}. Meaning: {i.meaning}.", indent=2, hang=4) for i in FLAG_INFO.values()]
    lines += ["", _para('Filter on the codes you care about, not on "flags is empty": at dish scale every row of a '
                        "coarse track carries LOWRES, and a track without a head click carries HEADGUESS on every "
                        "frame.", indent=2)]
    return lines


def _shape_files_section() -> list[str]:
    lines: list[str] = []
    for text in _SHAPE_FILES:
        lines += [_para(text, indent=2), ""]
    lines += [_para("outlines.npz has these arrays for every track, named <id>__<name>, for example A__xy_mm "
                    "(n is the number of tracked frames, N the number of points):", indent=2), ""]
    for key in OUTLINES_KEYS:
        shape = ", ".join(str(s) for s in key.shape)
        unit = f", {key.unit}" if key.unit else ""
        lines.append(_para(f"<id>__{key.name}: {key.dtype} [{shape}]{unit}. {key.meaning}", indent=4, hang=4))
    lines += ["", _para(f"{OUTLINES_META_KEY} is a JSON text with these fields:", indent=2)]
    lines += [_para(f"{field}: {meaning}", indent=4, hang=4) for field, meaning in OUTLINES_META_FIELDS.items()]
    return lines


def _git_section() -> list[str]:
    lines = ["  In git (small files):"]
    lines += [f"    {f.name}" for f in FILES if f.in_git]
    lines += [
        "    calibration.json (if you exported one)",
        "",
        _para("outlines.npz is the only full-shape output, so it must be committed. A .gitignore that excludes "
              "*.npz hides it: add the line !**/outlines.npz below the line *.npz.", indent=2),
        "",
        "  Not in git (large; share them through Drive):",
    ]
    lines += [f"    {f.name}" for f in FILES if not f.in_git]
    return lines


def readme_text() -> str:
    """The text of README.txt: ASCII, LF line ends, ending with one line end.

    Describes every file and column (with units, decimals and meaning), the conventions of SPEC 3,
    the flags of SPEC 9 and what goes into git; it names the units and coordinate frames of every
    column. The file, column, flag and key lists are read from this module, so a new column or flag
    appears in it without editing prose.
    """
    out = [
        "OUTLINE TRACKER: what is in this folder",
        "",
        _para("This folder holds one run of Outline Tracker on one video. The tool writes this file; do not "
              "edit it. The unit of a number is in its column name, except for the few numbers that Units, under "
              "CONVENTIONS, lists."),
    ]

    def section(title: str, lines: list[str]) -> None:
        out.extend(["", f"== {title} ==", ""] + lines)

    section("FILES", [_para(f"{f.name}: {f.meaning}", indent=2, hang=4) for f in FILES])
    section("CONVENTIONS", [_para(text, indent=2, hang=2) for text in _CONVENTIONS])
    section("COLUMNS", _columns_section()[1:])
    section("FLAGS", _flags_section())
    section("SHAPE FILES", _shape_files_section())
    section("READING THE FILES", ["  " + line for line in _READING_CODE] + [
        "",
        _para('Write df["flags"], not df.flags: pandas has its own attribute called flags. Lost frames are empty '
              "cells, which pandas reads as NaN; the integer columns (frame, visible, n_components, shape_ok) "
              "have no empty cells and stay integers.", indent=2),
    ])
    section("WHAT GOES INTO GIT", _git_section())
    section("IF SOMETHING WENT WRONG", [
        _para("If a run stopped early, or a file is missing, run this command; it rebuilds every file from what "
              "was tracked, without running the model:", indent=2),
        "",
        "    outline-tracker export <run folder>/session.json",
        "",
        _para("If a file is locked (for example a CSV open in Excel), the tool writes the new data next to it as "
              "<name>.new<extension>, for example positions.new.csv, and tells you. Close the program that holds "
              "the file and export again.", indent=2),
    ])
    return "\n".join(out) + "\n"
