"""Tests of outline_tracker/core/session_v1.py: a session.json of version 1 turned into one of version 2.

Expected values come from docs/design/formats.md, written by hand and approved by the owner before the code
(decision 12): its four examples and its table, parsed by tests/support/session_json_helpers.py. The other
values are typed in each test. Nothing is copied from the output of the code under test.

How a whole session file of version 1 is opened, saved and kept (rule 5 of docs/ROADMAP.md) is tested with
the session itself, in tests/core/test_session_file.py.
"""

import copy

import pytest
from session_json_helpers import (
    assert_same_json,
    formats_example,
    formats_rows,
    key_paths,
    spec_example,
)

from outline_tracker.core.session_v1 import MixedVersionError, upgrade

WHERE = "runs/groupB/session.json"  # a name for the file in messages, typed here
ROWS = formats_rows()
# the objects that the table gives one row, not one row per key: "the same object (...)"
FREE_OBJECTS = frozenset(path for row in ROWS for path in (row.v1, row.v2)
                         if path and "the same object" in row.reading)


def minimal_v1(**keys) -> dict:
    """The smallest file of version 1 that 0.1.0 reads, with `keys` added at the top."""
    return {"format": "outline-tracker-session", "schema_version": 1, "tool_version": "0.1.0", **keys}


# --------------------------------------------------------------------------- the page itself

def test_the_column_version_1_has_every_key_of_the_version_1_examples_and_of_spec_8_10():
    found = set()
    for data in (formats_example(1, 1), formats_example(2, 1), spec_example()):
        found |= key_paths(data, FREE_OBJECTS)
    assert {row.v1 for row in ROWS if row.v1} == found


def test_the_column_version_2_has_every_key_of_the_version_2_examples():
    found = key_paths(formats_example(1, 2), FREE_OBJECTS) | key_paths(formats_example(2, 2), FREE_OBJECTS)
    assert {row.v2 for row in ROWS if row.v2} == found


def test_each_key_has_one_row():
    for column in ("v1", "v2"):
        keys = [getattr(row, column) for row in ROWS if getattr(row, column)]
        assert len(keys) == len(set(keys)), column


@pytest.mark.parametrize("number", [1, 2])
def test_the_examples_of_the_page_follow_its_table(number):
    """The keys of each version 2 example are those of its version 1 example, renamed and dropped as the
    table says, and the table's new keys where their object is there."""
    before = key_paths(formats_example(number, 1), FREE_OBJECTS)
    after = key_paths(formats_example(number, 2), FREE_OBJECTS)
    renamed = {row.v2 for row in ROWS if row.v1 in before and row.v2}
    new = {row.v2 for row in ROWS if row.v1 is None}
    assert renamed <= after
    assert after - renamed <= new
    assert not {row.v1 for row in ROWS if row.v2 is None} & after


# --------------------------------------------------------------------------- the examples

@pytest.mark.parametrize("number", [1, 2])
def test_each_version_1_example_reads_as_its_version_2_example(number):
    assert_same_json(upgrade(formats_example(number, 1), WHERE), formats_example(number, 2))


def test_the_spec_example_reads_as_version_2():
    """SPEC 8.10's example is a file of version 1 with every key: the rows that change, by hand."""
    v1 = spec_example()
    v2 = upgrade(v1, WHERE)
    assert v2["schema_version"] == 2
    assert v2["label"] == v1["student"] and list(v2).index("label") == list(v1).index("student")
    assert v2["boundary"] == {"shape": "circle", **{("diameter_mm" if key == "dish_mm" else key): value
                                                    for key, value in v1["circle"].items()}}
    assert v2["processing"]["boundary_crop"] == v1["processing"]["dish_crop"]
    assert list(v2["processing"])[-1] == "tracker_format" and v2["processing"]["tracker_format"] is True
    assert "manifest_path" not in v2["time"]
    assert {"student", "circle"}.isdisjoint(v2)


def test_the_reader_leaves_its_input_as_it_was_and_shares_nothing_with_it():
    v1 = formats_example(1, 1)
    before = copy.deepcopy(v1)
    v2 = upgrade(v1, WHERE)
    assert_same_json(v1, before)
    v2["tracks"][0]["prompts"][0]["points_px"][0][0] = -1.0
    v2["calibration"]["stick"]["p1_px"][0] = -1.0
    v2["boundary"]["points_px"].clear()
    assert_same_json(v1, before)


# --------------------------------------------------------------------------- the rows that change

def test_student_becomes_label_in_its_place():
    v2 = upgrade(minimal_v1(student="ana", notes="dish 3"), WHERE)
    assert list(v2) == ["format", "schema_version", "tool_version", "label", "notes",
                        "processing"]
    assert v2["label"] == "ana"


def test_a_file_of_another_version_1_tool_keeps_its_tool_version():
    assert upgrade(minimal_v1(tool_version="0.1.1"), WHERE)["tool_version"] == "0.1.1"
    assert "tool_version" not in upgrade({"format": "outline-tracker-session", "schema_version": 1}, WHERE)


def test_a_null_circle_becomes_a_null_boundary_in_its_place():
    v2 = upgrade(minimal_v1(circle=None, probes=[]), WHERE)
    assert list(v2)[3:5] == ["boundary", "probes"] and v2["boundary"] is None


def test_a_circle_becomes_a_boundary_whose_first_key_is_its_shape():
    circle = {"points_px": [[10.5, 20.5], [30.5, 20.5], [20.5, 30.5]], "center_px": [20.5, 20.5],
              "radius_px": 10.0, "rms_px": 0.0, "dish_mm": 90.0}
    v2 = upgrade(minimal_v1(circle=circle), WHERE)
    assert_same_json(v2["boundary"], {"shape": "circle", "points_px": circle["points_px"],
                                      "center_px": [20.5, 20.5], "radius_px": 10.0, "rms_px": 0.0,
                                      "diameter_mm": 90.0})


@pytest.mark.parametrize("diameter", [None, 50, 87.5])
def test_dish_mm_becomes_diameter_mm_with_the_same_value(diameter):
    v2 = upgrade(minimal_v1(circle={"radius_px": 3.0, "dish_mm": diameter}), WHERE)
    assert_same_json(v2["boundary"], {"shape": "circle", "radius_px": 3.0, "diameter_mm": diameter})


def test_a_circle_without_dish_mm_gets_no_diameter():
    v2 = upgrade(minimal_v1(circle={"radius_px": 3.0}), WHERE)
    assert_same_json(v2["boundary"], {"shape": "circle", "radius_px": 3.0})


@pytest.mark.parametrize("value", [[1, 2], "round", 5, True])
def test_a_circle_that_is_not_an_object_is_refused_under_its_version_1_name(value):
    with pytest.raises(ValueError, match='"circle" must be an object') as info:
        upgrade(minimal_v1(circle=value), WHERE)
    assert "boundary" not in str(info.value)


@pytest.mark.parametrize("crop", [True, False])
def test_dish_crop_becomes_boundary_crop_in_its_place(crop):
    processing = {"model": "edgetam", "dish_crop": crop, "outline_points": 64}
    v2 = upgrade(minimal_v1(processing=processing), WHERE)
    assert_same_json(v2["processing"], {"model": "edgetam", "boundary_crop": crop, "outline_points": 64,
                                        "jump_mm_s": 100.0, "tracker_format": True})


def test_a_missing_jump_mm_s_is_written_as_the_100_mm_s_of_0_1_0_and_a_given_one_is_kept():
    assert upgrade(minimal_v1(processing={}), WHERE)["processing"]["jump_mm_s"] == 100.0
    v2 = upgrade(minimal_v1(processing={"jump_mm_s": 250.0}), WHERE)
    assert_same_json(v2["processing"], {"jump_mm_s": 250.0, "tracker_format": True})


def test_tracker_format_is_written_true_as_the_last_key_of_processing():
    processing = {"model": "edgetam", "x_lab_setting": "kept", "jump_mm_s": 80.0}
    v2 = upgrade(minimal_v1(processing=processing), WHERE)
    assert list(v2["processing"]) == ["model", "x_lab_setting", "jump_mm_s", "tracker_format"]
    assert v2["processing"]["tracker_format"] is True


def test_a_file_without_processing_gets_the_values_of_0_1_0():
    v2 = upgrade(minimal_v1(), WHERE)
    assert_same_json(v2, {"format": "outline-tracker-session", "schema_version": 2, "tool_version": "0.1.0",
                          "processing": {"jump_mm_s": 100.0, "tracker_format": True}})


def test_the_values_the_reader_writes_are_those_of_the_column_default():
    """The reader writes jump_mm_s, probes[].name and tracker_format itself (formats.md, "Keys missing
    from a version 1 file"). Their 0.1.0 values are typed here; the table must name the same."""
    by_v2 = {row.v2: row for row in ROWS}
    assert by_v2["processing.jump_mm_s"].json_default() == 100.0
    assert by_v2["probes[].name"].json_default() == "LED1"
    v2 = upgrade(minimal_v1(probes=[{"rect_px": [0, 0, 4, 4]}]), WHERE)
    assert v2["processing"]["jump_mm_s"] == by_v2["processing.jump_mm_s"].json_default()
    assert v2["probes"][0]["name"] == by_v2["probes[].name"].json_default()
    # tracker_format: a new session has false, a file of 0.1.0 always true (decisions 7, 34)
    assert by_v2["processing.tracker_format"].json_default() is False
    assert v2["processing"]["tracker_format"] is True


def test_manifest_path_is_dropped_and_a_manifest_rate_reads_as_typed():
    time = {"fps_true": 239.6, "source": "manifest", "manifest_path": "/data/videos/data/manifest.csv",
            "stopwatch": None}
    v2 = upgrade(minimal_v1(time=time), WHERE)
    assert_same_json(v2["time"], {"fps_true": 239.6, "source": "typed", "stopwatch": None})


@pytest.mark.parametrize("source", ["stopwatch", "typed", "tracker-export", "a lab's own word", ""])
def test_every_other_time_source_stays_as_it_is(source):
    v2 = upgrade(minimal_v1(time={"fps_true": 120.0, "source": source}), WHERE)
    assert_same_json(v2["time"], {"fps_true": 120.0, "source": source})


def test_a_null_manifest_path_is_dropped_too():
    v2 = upgrade(minimal_v1(time={"source": "typed", "manifest_path": None}), WHERE)
    assert_same_json(v2["time"], {"source": "typed"})


def test_a_probe_without_a_name_is_named_led1_and_a_named_one_keeps_its_name():
    probes = [{"rect_px": [1, 2, 3, 4]}, {"name": "LED2", "rect_px": [5, 6, 7, 8]}]
    v2 = upgrade(minimal_v1(probes=probes), WHERE)
    assert_same_json(v2["probes"], [{"name": "LED1", "rect_px": [1, 2, 3, 4]},
                                    {"name": "LED2", "rect_px": [5, 6, 7, 8]}])


# --------------------------------------------------------------------------- unknown keys and odd blocks

def test_unknown_keys_are_kept_in_their_objects():
    v1 = formats_example(1, 1)
    v1["x_top"] = {"a": [1, 2]}
    v1["circle"]["x_circle"] = [3]
    v1["time"]["x_time"] = None
    v1["processing"]["x_processing"] = "p"
    v1["probes"][0]["x_probe"] = 2.5
    v1["tracks"][0]["x_track"] = True
    v1["tracks"][0]["prompts"][0]["x_prompt"] = "q"
    expected = formats_example(1, 2)
    expected["x_top"] = {"a": [1, 2]}
    expected["boundary"]["x_circle"] = [3]
    expected["time"]["x_time"] = None
    tracker_format = expected["processing"].pop("tracker_format")
    expected["processing"].update(x_processing="p", tracker_format=tracker_format)
    expected["probes"][0]["x_probe"] = 2.5
    expected["tracks"][0]["x_track"] = True
    expected["tracks"][0]["prompts"][0]["x_prompt"] = "q"
    assert_same_json(upgrade(v1, WHERE), expected)


@pytest.mark.parametrize("key, value", [("processing", [1, 2]), ("processing", None), ("time", "fast"),
                                        ("time", None), ("probes", {"name": "LED1"}), ("probes", [3, None]),
                                        ("tracks", "A")])
def test_another_block_of_the_wrong_kind_passes_unchanged_for_the_records_to_refuse(key, value):
    v2 = upgrade(minimal_v1(**{key: value}), WHERE)
    assert_same_json(v2[key], value)


@pytest.mark.parametrize("path", ["label", "boundary", "circle.shape", "circle.diameter_mm",
                                  "processing.boundary_crop", "processing.tracker_format"])
def test_a_version_1_file_that_holds_a_key_of_version_2_is_refused_naming_the_file_and_the_key(path):
    v1 = formats_example(1, 1)
    *parent, key = path.split(".")
    (v1[parent[0]] if parent else v1)[key] = "circle" if key == "shape" else 1
    with pytest.raises(MixedVersionError) as info:
        upgrade(v1, WHERE)
    assert isinstance(info.value, ValueError)
    assert WHERE in str(info.value) and f'"{path}"' in str(info.value)
    assert info.value.key == path
