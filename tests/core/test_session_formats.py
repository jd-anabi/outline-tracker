"""Tests of session.json version 2 as outline_tracker/core/session.py reads and writes it, against the contract
docs/design/formats.md: the records have the keys of the table's column "Version 2" and its defaults, the
examples are read and written as the page shows, and the versions this build does not read are refused.

Expected values come from that page, written by hand and approved by the owner before the code (decision 12),
parsed by tests/support/session_json_helpers.py, and from values typed in each test. Nothing is copied from the
output of the code under test. The reader of version 1 on its own is tested in tests/core/test_session_v1.py.
"""

import types
from typing import Union, get_args, get_origin

import pytest
from results_helpers import names_number
from session_json_helpers import _write_json, assert_same_json, formats_example, formats_rows

from outline_tracker import __version__
from outline_tracker.core.session import Session, SessionVersionError
from outline_tracker.core.session_parts import _layout, _Record  # the layout of the records is what is checked
from outline_tracker.core.session_v1 import MixedVersionError, SessionKeyError

ROWS = formats_rows()
LEFT_OUT = object()  # a key that the file written does not hold
SKIP = object()  # a default that is no value to test
PROMPT_ITEMS = ("tracks[].prompts[].", "corrections[].prompts[].")  # the table's rows name the first only


def record_in(hint):
    """The record class that a field's type holds (itself, in a list, or with None), and whether in a list."""
    if get_origin(hint) in (Union, types.UnionType):
        (hint,) = (arg for arg in get_args(hint) if arg is not type(None))
    listed = get_origin(hint) is list
    item = get_args(hint)[0] if listed else hint
    is_record = isinstance(item, type) and get_origin(item) is None and issubclass(item, _Record)
    return (item if is_record else None), listed


def layout_paths(cls, prefix: str = "") -> set[str]:
    """The key paths of a record class and the records in it, as the table writes them."""
    paths = set()
    for _, key, _, hint in _layout(cls):
        paths.add(prefix + key)
        item, listed = record_in(hint)
        if item is not None:
            paths |= layout_paths(item, prefix + key + ("[]." if listed else "."))
    return paths


def file_without(path: str, version: int) -> dict:
    """The smallest session file of `version` whose objects lead to the key `path` and lack only it: every
    object and list on the way holds one empty object."""
    data = {"format": "outline-tracker-session", "schema_version": version}
    node = data
    for part in path.replace("[]", "[].").split(".")[:-1]:
        if part.endswith("[]"):
            node[part[:-2]] = [{}]
            node = node[part[:-2]][0]
        elif part:
            node[part] = {}
            node = node[part]
    return data


def value_at(data: dict, path: str):
    """The value at `path` in a parsed file (the first item of each list on the way), or LEFT_OUT."""
    node = data
    for part in (part for part in path.replace("[]", "[].").split(".") if part):
        key = part.removesuffix("[]")
        if key not in node:
            return LEFT_OUT
        node = node[key][0] if part.endswith("[]") else node[key]
    return node


def expected_default(row):
    """The value that the column "Default" names for a missing key: a JSON value, LEFT_OUT, or SKIP when it
    names none to test here ("required", or "below": see the test of the blocks)."""
    if row.default == "left out":
        return LEFT_OUT
    if row.default == "the reading build's version":
        return __version__
    value = row.json_default()
    return SKIP if value is NotImplemented else value


def default_rows(column: str):
    """The rows with a key in `column` ("v1" or "v2") and in version 2 whose default names a value."""
    return [pytest.param(row, id=getattr(row, column)) for row in ROWS
            if getattr(row, column) and row.v2 and expected_default(row) is not SKIP]


def assert_default(written: dict, row) -> None:
    """The file `written` holds the default of the row's key of version 2."""
    expected, found = expected_default(row), value_at(written, row.v2)
    if expected is LEFT_OUT:
        assert found is LEFT_OUT, found
    else:
        assert found is not LEFT_OUT, row.v2
        assert_same_json(found, expected)


# --------------------------------------------------------------------------- the records are the table

def test_the_records_have_the_keys_of_the_column_version_2():
    found = layout_paths(Session) | {"format", "schema_version"}  # the two that to_json writes itself
    prompts = {path.removeprefix(PROMPT_ITEMS[0]) for path in found if path.startswith(PROMPT_ITEMS[0])}
    assert {path.removeprefix(PROMPT_ITEMS[1]) for path in found if path.startswith(PROMPT_ITEMS[1])} == prompts
    found = {path for path in found if not path.startswith(PROMPT_ITEMS[1])}
    assert found == {row.v2 for row in ROWS if row.v2}


@pytest.mark.parametrize("row", default_rows("v2"))
def test_a_key_missing_from_a_version_2_file_reads_as_the_column_default(row):
    assert_default(Session.from_json(file_without(row.v2, 2)).to_json(), row)


@pytest.mark.parametrize("row", default_rows("v1"))
def test_a_key_missing_from_a_version_1_file_reads_as_the_column_default(row):
    assert_default(Session.from_json(file_without(row.v1, 1)).to_json(), row)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("row", [pytest.param(row, id=row.v2) for row in ROWS if row.default == "below"])
def test_a_missing_block_reads_as_the_block_with_all_its_keys_missing(row, version):
    key = (row.v1 if version == 1 else row.v2)
    head = {"format": "outline-tracker-session", "schema_version": version}
    assert_same_json(Session.from_json(head).to_json(), Session.from_json({**head, key: {}}).to_json())


def test_a_new_session_writes_version_2_without_the_tracker_format_files():
    written = Session().to_json()
    assert written["schema_version"] == 2
    assert written["boundary"] is None
    assert written["processing"]["tracker_format"] is False
    assert {"student", "circle"}.isdisjoint(written) and "manifest_path" not in written["time"]


# --------------------------------------------------------------------------- the examples

@pytest.mark.parametrize("number", [1, 2])
def test_each_version_2_example_is_read_and_written_back_unchanged(number):
    assert_same_json(Session.from_json(formats_example(number, 2)).to_json(), formats_example(number, 2))


@pytest.mark.parametrize("number", [1, 2])
def test_each_version_1_example_file_is_read_as_its_version_2_example(tmp_path, number):
    path = _write_json(tmp_path / "session.json", formats_example(number, 1))
    assert_same_json(Session.load(path).to_json(), formats_example(number, 2))


def test_a_version_1_file_keeps_the_tool_version_that_made_it():
    data = formats_example(1, 1)
    assert data["tool_version"] == "0.1.0" != __version__
    assert Session.from_json(data).tool_version == "0.1.0"


# --------------------------------------------------------------------------- what is refused

@pytest.mark.parametrize("found", [3, 4, 99])
def test_a_newer_schema_version_is_refused_naming_the_file_and_the_versions_read(tmp_path, found):
    data = formats_example(1, 2)
    data["schema_version"], data["tool_version"] = found, "9.8.7"
    path = _write_json(tmp_path / "session.json", data)
    with pytest.raises(SessionVersionError) as info:
        Session.load(path)
    message = str(info.value)
    assert str(path) in message and "9.8.7" in message and "newer" in message
    message = message.replace(str(path), "<the file>")  # a temporary folder's name may hold any digit
    assert names_number(message, found) and "versions 1 and 2" in message, message
    assert (info.value.found, info.value.supported) == (found, (1, 2))


@pytest.mark.parametrize("shape", ["ellipse", "Circle", "", None, 1])
def test_a_boundary_of_another_shape_is_refused_naming_the_file_and_the_key(tmp_path, shape):
    data = formats_example(1, 2)
    data["boundary"]["shape"] = shape
    path = _write_json(tmp_path / "session.json", data)
    with pytest.raises(SessionKeyError) as info:
        Session.load(path)
    assert str(path) in str(info.value) and '"boundary.shape"' in str(info.value)
    assert info.value.key == "boundary.shape" and isinstance(info.value, ValueError)
    assert not isinstance(info.value, MixedVersionError)


def test_a_version_2_boundary_without_a_shape_is_a_circle():
    data = formats_example(1, 2)
    del data["boundary"]["shape"]
    assert_same_json(Session.from_json(data).to_json(), formats_example(1, 2))


def test_a_version_1_file_with_a_key_of_version_2_is_refused_naming_the_file(tmp_path):
    data = formats_example(1, 1)
    data["processing"]["tracker_format"] = False
    path = _write_json(tmp_path / "session.json", data)
    with pytest.raises(MixedVersionError) as info:
        Session.load(path)
    assert str(path) in str(info.value) and '"processing.tracker_format"' in str(info.value)
    assert isinstance(info.value, SessionKeyError)


@pytest.mark.parametrize("version, name", [(1, "circle"), (2, "boundary")])
def test_a_boundary_block_of_the_wrong_kind_is_refused_under_the_files_own_name(version, name):
    data = formats_example(1, version)
    data[name] = [1, 2]
    with pytest.raises(ValueError, match=f'"{name}" must be an object'):
        Session.from_json(data)
