"""The two texts that describe a run folder, README.txt and docs/OUTPUTS.md, built from the schema tables.

`readme_text()` is README.txt (SPEC 8.11), written into every run folder. `outputs_markdown()` is docs/OUTPUTS.md (SPEC
8.1 and 16), the contract for the analysis template; the committed file is its output, and a test fails when it is
stale. Regenerate it after a change of the schema with `uv run python -m outline_tracker.core.schema_docs docs/OUTPUTS.md`.

Both texts say the same things: the prose is written once, in core/schema_prose.py, with code marks (backticks) that
the plain-text README drops. Every file, column, flag, key and number comes from the schema tables, so a new column or
flag appears in both without editing prose. This module only builds text; it does no geometry and reads no file. It
was split out of core/schema.py (SPEC 12: split past about 400 lines); `outline_tracker.core.schema.readme_text` is
still importable and is the same function. Units: every number is given with its unit, and the Units convention lists
each column whose name does not show it (SPEC 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

from outline_tracker.core.schema import (
    FILES, FLAG_INFO, OUTLINE_POINTS, OUTLINES_KEYS, OUTLINES_META_FIELDS, OUTLINES_NPZ, RADIAL_CSV,
    README_TXT, RESULTS_KEYS, RESULTS_NPZ, RESULTS_VERSION, RESULTS_VERSION_KEY, ArrayKey, Column, FileSpec,
    npz_key,
)
from outline_tracker.core.schema_prose import (
    _EXPORT_COMMAND, _EXPORT_TEXT, _FLAGS_ADVICE, _FLAGS_INTRO, _GIT_EXTRA, _GIT_NOTE, _IN_GIT, _LOCKED_TEXT,
    _META_INTRO, _NOT_IN_GIT, _OUTLINES_INTRO, _ROW_TEXT, _SHAPE_FILES, _conventions, _para, _plain, _radii,
    _radii_run, _reading_code, _reading_note, _scopes,
)

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
        '<!-- Generated from outline_tracker/core/schema.py by "uv run python -m outline_tracker.core.schema_docs '
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
    """`python -m outline_tracker.core.schema_docs OUTPUT.md` writes `outputs_markdown()` to that file as UTF-8 with
    LF line ends on every platform. Returns the exit code: 0, or 2 (with a usage line) for a wrong call."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m outline_tracker.core.schema_docs OUTPUT.md", file=sys.stderr)
        return 2
    Path(args[0]).write_bytes(outputs_markdown().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
