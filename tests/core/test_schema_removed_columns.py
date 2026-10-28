"""The columns that shapes.csv lost in W3 step 2 (decisions 8 and 40 of docs/ROADMAP.md; section 2, rule 5).

A removed column has no reader. README.txt, docs/OUTPUTS.md and CHANGELOG.md name each one with its unit and the
formula of the version that last wrote it, so that whoever needs the number can compute it from the other files.
The expected names, units and formulas are the table "The removed columns" of docs/design/formats.md, read by
`schema_helpers.removed_columns`; nothing is copied from the texts under test. The fixtures `readme` and `outputs`
(the texts of README.txt and docs/OUTPUTS.md) are in tests/core/conftest.py.
"""

import dataclasses

from helpers import normalized
from layout_helpers import REPO
from schema_docs_helpers import md_section, plain
from schema_helpers import removed_columns

from outline_tracker.core import schema
from outline_tracker.measure.derive import DerivedTrack

SINCE = "Columns removed since 0.1.0"
# what the formulas are made of, as formats.md names it: the boundary in session.json, the outline in results.npz
SOURCES = ("boundary.center_px", "boundary.radius_px", schema.npz_key("<id>", "outline_px"))


def test_the_schemas_table_of_removed_columns_is_the_one_of_formats_md():
    expected = removed_columns()
    assert [name for name, _, _ in expected] == ["wall_dist_centroid_mm", "wall_dist_min_mm"]  # decision 8
    assert [(c.name, c.unit, c.formula) for c in schema.REMOVED_COLUMNS] == expected
    assert {(c.file, c.version) for c in schema.REMOVED_COLUMNS} == {(schema.SHAPES_CSV, "0.1.0")}


def test_no_file_and_no_derived_track_holds_a_removed_column():
    removed = {name for name, _, _ in removed_columns()}
    written = {c.name for spec in schema.FILES for c in spec.columns or []}
    assert written & removed == set()
    assert {field.name for field in dataclasses.fields(DerivedTrack)} & removed == set()


def test_readme_names_each_removed_column_under_shapes_csv_with_its_unit_and_formula(readme):
    lines = readme.splitlines()
    start, end = lines.index("== COLUMNS =="), lines.index("== FLAGS ==")
    columns = lines[start:end]
    shapes, radial = columns.index(schema.SHAPES_CSV), columns.index(schema.RADIAL_CSV)
    block = normalized("\n".join(columns[shapes:radial]))
    assert SINCE in block
    for name, unit, formula in removed_columns():
        assert f"{name} ({unit}): {formula}" in block, name
    for source in SOURCES:
        assert source in block, source


def test_outputs_md_names_each_removed_column_under_shapes_csv_with_its_unit_and_formula(outputs):
    section = md_section(outputs, f"### `{schema.SHAPES_CSV}`")
    text = "\n".join(section)
    assert f"**{SINCE}.**" in text
    for name, unit, formula in removed_columns():
        assert f"| `{name}` | {unit} | {formula} |" in section, name
    for source in SOURCES:
        assert f"`{source}`" in text, source


def test_the_changelog_names_each_removed_column_with_its_formula():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = plain(changelog.split("\n## Unreleased\n", 1)[1].split("\n## ", 1)[0])
    for name, unit, formula in removed_columns():
        assert f"{name} ({unit}): {formula}" in unreleased, name
