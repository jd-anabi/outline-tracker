"""Tests of outline_tracker/core/schema_docs.py: outputs_markdown() and the committed docs/OUTPUTS.md (SPEC 8.1, 16).

docs/OUTPUTS.md is the contract for the analysis template. Expected values are typed from the specification
or computed from the schema tables, never copied from the output under test.

The rest of the tests of the module are beside this file: what must not change when the builders moved out
of core/schema.py in tests/core/test_schema_docs.py, the wording of the README text and the units in
tests/core/test_schema_docs_readme.py. The fixtures `schema_docs`, `readme` and `outputs` are in
tests/core/conftest.py; what this file and the last one share is in tests/support/schema_docs_helpers.py.
"""

import dataclasses
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from helpers import normalized
from layout_helpers import REPO, module
from layout_helpers import path as table_path
from schema_docs_helpers import RADIAL_COLUMNS, cells, md_section, plain

from outline_tracker.core import csv_format, schema, schema_prose, tracker_io


# ---------------------------------------------------------------------------------------------
# docs/OUTPUTS.md: outputs_markdown(), the contract for the analysis template (SPEC 8.1, 16)

OUTPUTS_MD = REPO / "docs" / "OUTPUTS.md"
REGENERATE = f"uv run python -m {module('schema docs')} docs/OUTPUTS.md"

# SPEC 8.1, in the order of the tree drawn there.
FILE_ORDER = [
    "session.json", "positions.csv", "<model>/<id>.csv", "shapes.csv", "radial.csv", "outlines.npz",
    "probes.csv", "overlay.mp4", "results.npz", "run.log", "README.txt",
]
FLAG_CODES = ["LOST", "JUMP", "SIZE", "CONTACT", "EDGE", "MULTI", "LOWRES", "ORIENT", "HEADGUESS"]
POSITION_FLAGS = FLAG_CODES[:5]  # SPEC 9: the ones about where the object is
SHAPE_FLAGS = FLAG_CODES[5:]  # SPEC 9: the ones that only concern the outline and the heading

# A path under a home folder, as tests/repo/test_repo_rules.py spells it; put together from pieces so that
# this file, which that test scans, does not contain what it looks for.
_USERS = "Users"
_TMP = "pytest-of" + "-"
HOME_PATH = re.compile(rf"[/\\]{_USERS}[/\\]+[^/\\\s]|{_TMP}[^/\\\s]|[A-Za-z]:\\\\?{_USERS}")


def md_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Header cells and body rows of the first table in `lines`."""
    table = [line for line in lines if line.startswith("|")]
    assert len(table) >= 3, "no table"
    width = len(cells(table[0]))
    assert all(len(cells(line)) == width for line in table), "a row has more or fewer cells than the header"
    assert set(cells(table[1])) == {"---"}
    return cells(table[0]), [cells(line) for line in table[2:]]


def code(name: str) -> str:
    return f"`{name}`"


def file_section(outputs: str, name: str) -> list[str]:
    return md_section(outputs, f"### `{name}`")


def test_outputs_markdown_is_ascii_with_lf_and_deterministic(schema_docs, outputs):
    assert outputs.startswith("<!-- ")  # the generated-file note comes first, the title second
    assert outputs.isascii()
    assert "\r" not in outputs and "\t" not in outputs
    assert outputs.endswith("\n") and not outputs.endswith("\n\n")
    assert outputs == schema_docs.outputs_markdown()


def test_there_is_one_section_per_file_in_the_order_of_the_spec(outputs):
    headings = [line for line in md_section(outputs, "## The files") if line.startswith("### ")]
    assert [h[len("### `") : -1] for h in headings] == FILE_ORDER
    assert FILE_ORDER == [f.name for f in schema.FILES]


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("positions.csv", schema.POSITIONS), ("shapes.csv", schema.SHAPES), ("probes.csv", schema.PROBES),
        ("<model>/<id>.csv", schema.TRACKER_COLUMNS),
    ],
)
def test_every_csv_table_has_exactly_the_columns_of_its_file(outputs, name, columns):
    header, rows = md_table(file_section(outputs, name))
    assert header == ["column", "type", "unit", "meaning"]
    assert [row[0] for row in rows] == [code(c.name) for c in columns]
    for row, col in zip(rows, columns):
        assert row[1].split(",")[0] == col.dtype, col.name
        assert row[2] == (col.unit or "none"), col.name
        assert row[3] == col.meaning, col.name


def test_the_decimals_of_each_number_are_given_in_the_type_cell(outputs):
    """SPEC 8.2: t with 7 decimals, mm with 6, px with 3; integers and text have none."""
    _, rows = md_table(file_section(outputs, "positions.csv"))
    types = {row[0]: row[1] for row in rows}
    assert types[code("t_s")] == "float, 7 decimals"
    assert types[code("x_mm")] == "float, 6 decimals"
    assert types[code("u_px")] == "float, 3 decimals"
    assert types[code("frame")] == "int" and types[code("flags")] == "str"


def test_the_positions_table_agrees_with_the_spec_table(outputs):
    """The same names, types and units as the table in SPEC 8.2 itself."""
    spec = table_path("spec").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(spec) if line.startswith("### 8.2"))
    end = next(i for i in range(start + 1, len(spec)) if spec[i].startswith("#"))
    spec_rows = [cells(line) for line in spec[start:end] if line.startswith("|")][2:]
    _, rows = md_table(file_section(outputs, "positions.csv"))
    assert [r[0].strip("`") for r in rows] == [r[0] for r in spec_rows]
    assert [r[1].split(",")[0] for r in rows] == [r[1] for r in spec_rows]
    assert [r[2] for r in rows] == [r[2].replace("\u00b2", "^2") or "none" for r in spec_rows]


def test_the_radial_section_names_all_72_columns(outputs):
    section = "\n".join(file_section(outputs, "radial.csv"))
    assert "72 columns" in section
    for name in RADIAL_COLUMNS:
        assert re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", section), name
    [listing] = [line for line in section.splitlines() if "radius columns, in order:" in line]
    assert re.findall(r"`(r_\d{3})`", listing) == RADIAL_COLUMNS  # all 72, in the order of SPEC 8.5
    _, rows = md_table(file_section(outputs, "radial.csv"))
    assert [row[0] for row in rows[:3]] == [code("track_id"), code("frame"), code("t_s")]
    assert "r_000" in rows[3][0] and "r_355" in rows[3][0]
    assert rows[3][2] == "mm" and "head radius" in rows[3][3]
    assert "hold the fine tracks by default" in plain(section)


def test_the_results_keys_table(outputs):
    section = file_section(outputs, "results.npz")
    header, rows = md_table(section)
    assert header == ["key", "dtype", "shape", "unit", "meaning"]
    assert [row[0] for row in rows] == [code(f"<id>__{k.name}") for k in schema.RESULTS_KEYS]
    by_key = {row[0]: row for row in rows}
    outline = by_key[code("<id>__outline_px")]  # SPEC 8.12: [n, 256, 2] float32
    assert outline[1:4] == ["float32", "[n, 256, 2]", "px"]
    assert by_key[code("<id>__cov_full")][1:4] == ["float64", "[n, 3]", "px^2"]
    assert by_key[code("<id>__frames")][1:3] == ["int32", "[n]"]
    text = plain(" ".join(section))
    assert f"{schema.RESULTS_VERSION_KEY} holds the format version" in text and "currently 1" in text


def test_the_outlines_keys_table_and_the_meta_fields(outputs):
    section = file_section(outputs, "outlines.npz")
    header, rows = md_table(section)
    assert header == ["key", "dtype", "shape", "unit", "meaning"]
    assert [row[:4] for row in rows] == [  # SPEC 8.6
        [code("<id>__frames"), "int32", "[n]", "none"],
        [code("<id>__xy_mm"), "float32", "[n, N, 2]", "mm"],
        [code("<id>__xieta_mm"), "float32", "[n, N, 2]", "mm"],
    ]
    text = plain(" ".join(section))
    assert "N = 128" in text and "counterclockwise" in text and "head point" in text
    for field, meaning in schema.OUTLINES_META_FIELDS.items():
        assert f"- {code(field)}: {meaning}" in section


def test_the_files_without_columns_say_what_they_are(outputs):
    for spec in schema.FILES:
        text = plain(" ".join(file_section(outputs, spec.name)))
        if spec.name != schema.README_TXT:
            assert normalized(spec.meaning) in text, spec.name
        assert f"In git: {'yes' if spec.in_git else 'no'}" in text.replace("**", ""), spec.name
    assert "plain text" in plain(" ".join(file_section(outputs, "README.txt")))  # not "this file"


def test_the_tracker_format_first_line_is_the_one_of_the_ported_writer(outputs, tmp_path):
    path = tmp_path / "A.csv"
    tracker_io.write_tracker_file(path, "A", frames=[0], t=[0.0], x=[1.0], y=[2.0], px=[3.5], py=[4.5])
    first_line = path.read_text().splitlines()[0]
    assert first_line == ",A,,,,,"
    assert first_line in " ".join(file_section(outputs, "<model>/<id>.csv"))


def test_the_conventions_of_spec_3(outputs):
    text = plain(" ".join(md_section(outputs, "## Conventions")))
    assert "(c + 0.5, r + 0.5)" in text  # pixel centers at +0.5 (SPEC 3.1)
    assert "top-left corner" in text and "v downward" in text
    assert "y points up" in text and "counterclockwise" in text  # SPEC 3.2
    assert "x = k ((u - u0) cos alpha - (v - v0) sin alpha)" in text
    assert "y = -k ((u - u0) sin alpha + (v - v0) cos alpha)" in text
    assert "t_s = frame / fps_true" in text and "never used" in text  # SPEC 3.3
    assert "grid start + n * step" in text and "all tracks share frames" in text  # SPEC 3.4
    assert "millimetres" in text and "mm per pixel" in text
    assert "UTF-8" in text and "LF" in text


def test_the_flag_table_and_the_scopes(outputs):
    section = md_section(outputs, "## Quality flags")
    header, rows = md_table(section)
    assert header == ["code", "concerns", "condition", "meaning"]
    assert [row[0] for row in rows] == [code(c) for c in FLAG_CODES]
    assert [row[1] for row in rows] == ["position"] * 5 + ["shape"] * 4
    for row in rows:
        info = schema.FLAG_INFO[row[0].strip("`")]
        assert row[2:] == [info.condition, info.meaning]
    [position_line] = [line for line in section if "concern positions" in line]
    [shape_line] = [line for line in section if "concern only shapes" in line]
    assert ", ".join(POSITION_FLAGS) in position_line and ", ".join(SHAPE_FLAGS) in shape_line
    assert "not on \"flags is empty\"" in plain(" ".join(section))
    assert "separated by ';'" in plain(" ".join(section))


def test_the_reading_recipe_runs(outputs, tmp_path, monkeypatch):
    """The code block of 'Reading the files' is run on a small run folder built by hand."""
    section = md_section(outputs, "## Reading the files")
    start = section.index("```python")
    recipe = "\n".join(section[start + 1 : section.index("```", start + 1)])
    row = dict(track_id="A", frame=0, t_s=0.0, x_mm=1.0, y_mm=2.0, u_px=3.5, v_px=4.5, area_mm2=0.5, visible=1,
               mode="coarse", flags="")
    rows = [row, dict(row, frame=2, flags="JUMP;SIZE"), dict(row, frame=4, flags="LOWRES")]
    (tmp_path / "positions.csv").write_text(csv_format.csv_text(schema.POSITIONS, rows), encoding="utf-8")
    meta = json.dumps({"n_points": 128, "tracks": ["A"]})
    np.savez(tmp_path / "outlines.npz", A__xy_mm=np.zeros((3, 128, 2), np.float32), meta=np.array(meta))
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(recipe, namespace)  # noqa: S102
    assert isinstance(namespace["df"], pd.DataFrame)
    assert namespace["jumps"]["frame"].tolist() == [2]
    assert namespace["xy"].shape == (3, 128, 2)
    assert namespace["meta"]["n_points"] == 128
    text = plain(" ".join(section))
    assert 'Write df["flags"], not df.flags' in text and "stay integers" in text


def test_the_git_section_lists_files_as_the_spec_does(outputs):
    section = md_section(outputs, "## What goes into git")
    split = next(i for i, line in enumerate(section) if "Not in git" in line)
    in_git, not_in_git = "\n".join(section[:split]), "\n".join(section[split:])
    for spec in schema.FILES:  # SPEC 8.13: overlay.mp4 and results.npz stay out
        assert (code(spec.name) in in_git) is spec.in_git, spec.name
        assert (code(spec.name) in not_in_git) is (not spec.in_git), spec.name
    assert "`calibration.json`" in in_git
    assert "`!**/outlines.npz`" in in_git and "`*.npz`" in in_git
    assert "Drive" in not_in_git


def test_the_words_used_are_explained(outputs):
    glossary = "\n".join(md_section(outputs, "## Words used on this page"))
    for term in ("frame", "track", "mask", "centroid", "outline", "coarse and fine", "core", "heading", "empty cell"):
        assert f"**{term}**" in glossary, term


def test_the_crop_of_coarse_mode_is_the_boundarys_square(outputs):
    # decisions 8 and 44: the square that coarse mode may show the model is the boundary's
    glossary = "\n".join(md_section(outputs, "## Words used on this page"))
    assert "the model looks at the whole frame (or the boundary's square); in fine mode" in glossary
    assert ("| `mode` | str | none | coarse (the model saw the whole frame or the boundary's square) or fine (a "
            "crop that follows the object) |") in outputs
    assert "the mask touches the border of the model's input (frame, boundary crop or fine crop)" in outputs


def test_the_page_says_how_to_read_it_and_regenerate_it(outputs):
    top = "\n".join(outputs.splitlines()[:12])
    assert REGENERATE in top and "do not edit" in top.lower()
    assert "README.txt" in top


def test_markdown_does_not_hide_text_or_break_tables(outputs):
    """A name in angle brackets outside a code span is read as an HTML tag and vanishes."""
    body = re.sub(r"<!--.*?-->", "", outputs, flags=re.S)
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"`[^`\n]*`", "", body)
    assert re.findall(r"<[A-Za-z/!?]", body) == []
    assert outputs.count("```") % 2 == 0
    for heading in re.findall(r"^#+ .*$", outputs, flags=re.M):
        assert heading.count("`") % 2 == 0, heading
    tables = re.split(r"\n\n", outputs)
    for block in (b for b in tables if b.startswith("|")):
        md_table(block.splitlines())


def test_the_numbers_of_the_text_come_from_the_tables(schema_docs, monkeypatch):
    """A column, a key and the number of outline points that this test adds or changes appear in the text."""
    extra = schema.Column("zz_extra_mm", "float", "mm", ".6f", "a column that only this test adds")
    files = [dataclasses.replace(f, columns=[*f.columns, extra]) if f.name == "positions.csv" else f
             for f in schema.FILES]
    monkeypatch.setattr(schema_docs, "FILES", files)
    monkeypatch.setattr(schema_prose, "FILES", files)  # the shared prose reads FILES in its own module
    key = schema.ArrayKey("zz_key", "int32", ("n",), "", "a key that only this test adds")
    monkeypatch.setattr(schema_docs, "RESULTS_KEYS", [*schema.RESULTS_KEYS, key])
    monkeypatch.setattr(schema_docs, "OUTLINE_POINTS", 64)
    monkeypatch.setattr(schema_prose, "OUTLINE_POINTS", 64)  # the reading recipe is made in schema_prose
    text = schema_docs.outputs_markdown()
    assert f"| {code('zz_extra_mm')} | float, 6 decimals | mm | {extra.meaning} |" in file_section(text, "positions.csv")
    assert f"| {code('<id>__zz_key')} | int32 | [n] | none | {key.meaning} |" in file_section(text, "results.npz")
    outlines_meaning = {f.name: f.meaning for f in schema.FILES}["outlines.npz"]  # a table text with its own "128"
    assert "128" not in text.replace(outlines_meaning, "")
    assert "N = 64" in plain(text) and "[n, 64, 2]" in text and "64 points per frame" in text
    assert "zz_extra_mm: float, mm, 6 decimals. a column that only this test adds" in schema_docs.readme_text()


def test_outputs_and_readme_say_the_same_things(outputs, readme):
    """SPEC 16: docs/OUTPUTS.md has the same content as README.txt. Sentences typed from the README."""
    for sentence in (
        "A lost frame keeps its row, with its track_id, frame and t_s",
        "Filter on the codes you care about",
        "outlines.npz is the only full-shape output, so it must be committed",
        "star-shaped around the body center this is the boundary itself",
        "Start frame and step choose frames but never shift t_s",
        "it rebuilds every file from what was tracked, without running the model",
        "If a file is locked (for example a CSV open in Excel)",
        "The files in the model folder keep your computer's line ends, as last week",
    ):
        assert sentence in normalized(readme), sentence
        assert sentence in plain(outputs), sentence


def test_docs_outputs_md_is_the_current_output(outputs):
    """The committed file cannot go stale: if the schema changes, regenerate it (REGENERATE)."""
    assert OUTPUTS_MD.is_file(), f"docs/OUTPUTS.md is missing; run: {REGENERATE}"
    assert b"\r" not in OUTPUTS_MD.read_bytes()
    assert OUTPUTS_MD.read_text(encoding="utf-8") == outputs, f"docs/OUTPUTS.md is out of date; run: {REGENERATE}"


def test_the_command_writes_the_file(tmp_path):
    target = tmp_path / "OUTPUTS.md"
    done = subprocess.run(
        [sys.executable, "-m", module("schema docs"), str(target)],
        capture_output=True, text=True, cwd=REPO, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == "" and done.stdout == ""
    assert target.read_bytes() == OUTPUTS_MD.read_bytes()  # LF line ends, UTF-8, on every platform


def test_the_command_needs_one_path():
    done = subprocess.run(
        [sys.executable, "-m", module("schema docs")], capture_output=True, text=True, cwd=REPO, timeout=60
    )
    assert done.returncode == 2
    assert "usage" in done.stderr.lower() and "OUTPUT.md" in done.stderr


def test_the_texts_hold_no_personal_paths(outputs, readme):
    for text in (outputs, readme, OUTPUTS_MD.read_text(encoding="utf-8")):
        assert HOME_PATH.search(text) is None
    assert HOME_PATH.search("/".join(["", _USERS, "someone", "Movies"]))  # the pattern does match one
    assert HOME_PATH.search("\\".join(["C:", _USERS, "someone"]))
    assert not HOME_PATH.search("the " + _USERS + " folder")


def test_the_page_points_to_no_section_that_only_the_readme_has(outputs, readme):
    """Table texts are shared by README.txt and OUTPUTS.md; a pointer such as "see FLAGS" names a README
    section, which OUTPUTS.md does not have."""
    readme_sections = [line[3:-3] for line in readme.splitlines() if line.startswith("== ")]
    assert "FLAGS" in readme_sections
    for name in readme_sections:
        assert re.search(rf"\b{name}\b", outputs) is None, name
    assert "see the quality flags" in outputs and "see the quality flags" in readme
