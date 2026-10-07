"""Tests of outline_tracker/results.py: the store behind results.npz (SPEC 8.12, 6.6; X5, X15): `put`,
`arrays` and `TrackArrays`, the two corrections `replace_from` and `truncate_after`, and saving
and loading. What may go wrong with the file is in tests/test_results_file.py.

Every expected array is written out from the records given to `put` (tests/results_helpers.py), and
every expected mask crop is the boolean array the test made itself. The frames that `replace_from`
and `truncate_after` leave are worked out here from the frame numbers. The file is also read with
numpy alone, key by key. Nothing is copied from the output of the code under test.

Records are in image pixels, Tracker's convention: (u, v) from the top-left corner of the frame,
pixel centers at +0.5; a mask crop is indexed [row, column] and its offset is (column, row) of its
top-left pixel in the full frame. Frames are video frame numbers.
"""

import builtins
from pathlib import Path

import numpy as np
import pytest
from results_helpers import (
    DISK_ORIGIN,
    assert_holds,
    crop,
    disk_distance,
    expected_column,
    lost,
    made_up,
    measured,
    names,
    names_number,
    read_npz,
    same,
    two_tracks,
    write_npz,
)

from outline_tracker import schema
from outline_tracker.measure import PixelRecord
from outline_tracker.results import ResultsStore, ResultsVersionError

PER_FRAME = [key for key in schema.RESULTS_KEYS if key.name not in ("frames", "mask_bits")]


def assert_same_arrays(a, b):
    for key in schema.RESULTS_KEYS:
        assert same(getattr(a, key.name), getattr(b, key.name)), key.name


# --------------------------------------------------------------------------- put and arrays


def test_arrays_have_the_names_dtypes_and_shapes_of_the_schema():
    store, records = two_tracks()
    arrays = store.arrays("A")
    assert list(arrays.__dataclass_fields__) == [key.name for key in schema.RESULTS_KEYS]
    n = len(records["A"])
    assert n == 7 and len(arrays) == 7
    sizes = {"n": n, "total_bytes": sum((r.mask_shape[0] * r.mask_shape[1] + 7) // 8 for r in records["A"])}
    for key in schema.RESULTS_KEYS:
        column = getattr(arrays, key.name)
        assert isinstance(column, np.ndarray) and column.dtype == np.dtype(key.dtype), key.name
        assert column.shape == tuple(sizes.get(side, side) for side in key.shape), key.name


def test_each_record_is_one_row_with_every_field_in_its_own_column():
    store, records = two_tracks()
    for track_id in ("A", "B"):
        arrays = store.arrays(track_id)
        assert arrays.frames.tolist() == [record.frame for record in records[track_id]]
        for row, record in enumerate(records[track_id]):
            for key in PER_FRAME:
                # one entry of a text column is a text of its own length: compare both in the column's dtype
                stored = np.asarray(getattr(arrays, key.name)[row], dtype=key.dtype)
                assert same(stored, np.asarray(getattr(record, key.name), dtype=key.dtype)), key.name
        assert_holds(arrays, records[track_id])
    assert set(store.arrays("A").mode) == {"coarse"} and set(store.arrays("B").mode) == {"fine"}


def test_rows_are_sorted_by_frame_whatever_the_order_of_put():
    records = [made_up(frame, frame) for frame in (30, 4, 18, 6, 100, 0)]
    store = ResultsStore()
    for record in records:
        store.put("A", record)
    arrays = store.arrays("A")
    assert arrays.frames.tolist() == [0, 4, 6, 18, 30, 100]
    assert_holds(arrays, sorted(records, key=lambda record: record.frame))


def test_put_replaces_a_frame_that_is_already_there():
    store, records = two_tracks()
    new = made_up(16, 77)                 # another crop size than the old frame 16: the bytes after it move
    assert new.mask_shape != records["A"][3].mask_shape
    store.put("A", new)
    assert_holds(store.arrays("A"), records["A"][:3] + [new] + records["A"][4:])
    store.put("A", lost(16))              # and a found frame may become a lost one
    assert_holds(store.arrays("A"), records["A"][:3] + [lost(16)] + records["A"][4:])
    assert_holds(store.arrays("B"), records["B"])


def test_mask_gives_back_each_frames_crop_and_its_offset():
    center, radius = (341.3, 229.6), 9.0
    inside = disk_distance(center, radius) > 0
    rows, cols = np.nonzero(inside)
    disk_crop = inside[rows.min():rows.max() + 1, cols.min():cols.max() + 1]   # the disk's bounding box
    disk_offset = (DISK_ORIGIN[0] + int(cols.min()), DISK_ORIGIN[1] + int(rows.min()))
    assert disk_crop.shape == (18, 18) and disk_crop.size % 8 != 0

    seeds = [5, 6, 7, 8]
    records = [made_up(2 * i, seed) for i, seed in enumerate(seeds)]
    records[2:2] = [lost(3)]                                    # an empty crop between two others
    records.append(measured(40, center, radius=radius))
    store = ResultsStore()
    for record in records:
        store.put("A", record)
    arrays = store.arrays("A")
    truth = [crop(5), crop(6), np.zeros((0, 0), bool), crop(7), crop(8), disk_crop]
    assert len({mask.shape for mask in truth}) == len(truth)    # all of different sizes
    assert arrays.mask_bits.dtype == np.uint8 and len(arrays.mask_bits) == sum((m.size + 7) // 8 for m in truth)
    assert np.array_equal(arrays.mask_bits, np.concatenate([np.packbits(m.ravel()) for m in truth]))
    for row in (3, 0, 5, 2, 1, 4):                              # in any order
        mask, offset = arrays.mask(row)
        assert mask.dtype == bool and np.array_equal(mask, truth[row]) and mask.shape == truth[row].shape
        assert offset == records[row].mask_offset
    assert arrays.mask(5)[1] == disk_offset
    assert arrays.mask(-1)[1] == disk_offset                    # rows count from the end too, as in numpy
    with pytest.raises(IndexError):
        arrays.mask(6)


def test_row_finds_a_video_frame_and_refuses_one_that_was_not_tracked():
    store, _ = two_tracks()
    arrays = store.arrays("A")              # frames 10, 12, ..., 22
    assert [arrays.row(frame) for frame in (10, 16, 22)] == [0, 3, 6]
    assert type(arrays.row(16)) is int
    for frame in (9, 11, 23, -1, 0):
        with pytest.raises(KeyError, match=str(frame)):
            arrays.row(frame)


def test_a_lost_frame_keeps_its_row():
    store, _ = two_tracks()
    arrays = store.arrays("A")
    row = arrays.row(14)
    assert not arrays.visible[row]
    assert (arrays.area_px[row], arrays.n_components[row], arrays.core_r_px[row]) == (0, 0, 0)
    for name in ("u", "v", "core_u", "core_v", "core_frac", "largest_fraction", "second_fraction"):
        assert np.isnan(getattr(arrays, name)[row]), name
    assert np.isnan(arrays.cov_full[row]).all() and np.isnan(arrays.cov_core[row]).all()
    assert np.isnan(arrays.outline_px[row]).all()
    assert arrays.mode[row] == "coarse" and arrays.score[row] == -3.0 and arrays.cell_px[row] == 7.5
    assert arrays.mask_shape[row].tolist() == [0, 0]
    mask, _ = arrays.mask(row)
    assert mask.shape == (0, 0) and mask.dtype == bool


def test_arrays_are_copies_changing_them_does_not_change_the_store():
    store, _ = two_tracks()
    store.put("C", made_up(8, 60))       # a track of one frame, too
    for track_id in ("A", "C"):
        arrays = store.arrays(track_id)
        for key in schema.RESULTS_KEYS:
            getattr(arrays, key.name)[...] = np.zeros((), dtype=key.dtype)
    truth = {**two_tracks()[1], "C": [made_up(8, 60)]}   # made again, so untouched by the lines above
    for track_id in ("A", "C"):
        assert_holds(store.arrays(track_id), truth[track_id])


def test_track_ids_are_the_tracks_that_have_records_in_text_order():
    store = ResultsStore()
    assert store.track_ids == []
    for track_id in ("B", "A2", "AA", "A"):
        store.put(track_id, made_up(4, 1))
    assert store.track_ids == ["A", "A2", "AA", "B"]
    with pytest.raises(KeyError, match="C") as err:
        store.arrays("C")
    assert "A2" in str(err.value)     # the message says which tracks there are


@pytest.mark.parametrize("track_id", ["", None, 3])
def test_put_refuses_a_track_id_that_is_not_a_name(track_id):
    store = ResultsStore()
    with pytest.raises(ValueError, match="track id"):
        store.put(track_id, made_up(4, 1))
    assert store.track_ids == []


def fitting(**changes):
    """A record with a 5 x 7 px crop (35 bits, so 5 bytes), with `changes` made to its fields."""
    good = made_up(4, 1)
    fields = {name: getattr(good, name) for name in good.__dataclass_fields__}
    fields.update(mask_shape=(5, 7), mask_bits=np.packbits(np.ones((5, 7), bool).ravel()))
    fields.update(changes)
    return PixelRecord(**fields)


@pytest.mark.parametrize("changes, names_it", [
    ({"mode": "coarse_dish"}, "mode"),                      # stored as 6 characters: it would come back as "coarse"
    ({"mask_bits": np.zeros(3, np.uint8)}, "mask_bits"),    # 35 bits need 5 bytes
    ({"mask_bits": np.zeros(6, np.uint8)}, "mask_bits"),
    ({"mask_shape": (6, 7)}, "mask_bits"),                  # 42 bits need 6 bytes, and the record has 5
    ({"outline_px": np.zeros((128, 2), np.float32)}, "outline_px"),
    ({"cov_full": np.zeros((2, 2))}, "cov_full"),
    ({"mask_offset": (1, 2, 3)}, "mask_offset"),
])
def test_put_refuses_a_record_that_does_not_fit_the_file(changes, names_it):
    store, records = two_tracks()
    with pytest.raises(ValueError, match=names_it):
        store.put("A", fitting(**changes))
    assert_holds(store.arrays("A"), records["A"])    # nothing was stored
    store.put("A", fitting())                         # the same record without the change is taken
    assert store.arrays("A").frames.tolist() == [4, 10, 12, 14, 16, 18, 20, 22]


# --------------------------------------------------------------------------- corrections (SPEC 6.6)


@pytest.mark.parametrize("k, kept", [
    (16, [10, 12, 14]),                  # a tracked frame: it goes too
    (17, [10, 12, 14, 16]),              # between two tracked frames
    (22, [10, 12, 14, 16, 18, 20]),      # the last frame
    (23, [10, 12, 14, 16, 18, 20, 22]),  # past the end: nothing to remove
    (11, [10]),
])
def test_replace_from_removes_frames_from_k_on_of_that_track_only(k, kept):
    store, records = two_tracks()
    store.replace_from("A", k)
    assert store.arrays("A").frames.tolist() == kept
    assert_holds(store.arrays("A"), [record for record in records["A"] if record.frame < k])
    assert_holds(store.arrays("B"), records["B"])   # B has frames 14 to 30: none of them is touched
    assert store.track_ids == ["A", "B"]


def test_frames_put_after_replace_from_take_the_place_of_the_old_ones():
    store, records = two_tracks()
    store.replace_from("A", 16)
    again = [made_up(frame, 500 + frame) for frame in (16, 18, 20, 22, 24)]   # the new run goes further
    for record in again:
        store.put("A", record)
    assert_holds(store.arrays("A"), records["A"][:3] + again)
    assert_holds(store.arrays("B"), records["B"])


@pytest.mark.parametrize("k, kept", [
    (20, [14, 16, 18, 20]),                           # a tracked frame: it stays
    (21, [14, 16, 18, 20]),                           # between two tracked frames
    (14, [14]),                                       # the first frame
    (30, [14, 16, 18, 20, 22, 24, 26, 28, 30]),       # the last frame: nothing to remove
    (99, [14, 16, 18, 20, 22, 24, 26, 28, 30]),
])
def test_truncate_after_keeps_the_frames_up_to_k(k, kept):
    store, records = two_tracks()
    store.truncate_after("B", k)
    assert store.arrays("B").frames.tolist() == kept
    assert_holds(store.arrays("B"), [record for record in records["B"] if record.frame <= k])
    assert_holds(store.arrays("A"), records["A"])
    assert store.track_ids == ["A", "B"]


def test_a_track_that_loses_every_frame_is_gone_and_may_be_asked_to_lose_them_again():
    store, records = two_tracks()
    store.replace_from("A", 10)          # A starts at frame 10
    assert store.track_ids == ["B"]
    with pytest.raises(KeyError):
        store.arrays("A")
    store.replace_from("A", 10)          # nothing at frames >= 10 before, nothing after
    store.truncate_after("B", 13)        # B starts at frame 14
    assert store.track_ids == []
    store.truncate_after("B", 13)
    store.replace_from("never there", 0)
    store.put("A", records["A"][0])      # a new run of A starts from nothing
    assert_holds(store.arrays("A"), records["A"][:1])


# --------------------------------------------------------------------------- the file


def test_save_and_load_give_equal_arrays_for_two_tracks(tmp_path):
    store, records = two_tracks()
    path = tmp_path / "vidéo test ü" / "results.npz"     # a folder that does not exist yet (review focus 1)
    assert store.save(path) == path
    assert names(path.parent) == ["results.npz"]
    loaded = ResultsStore.load(path)
    assert isinstance(loaded, ResultsStore) and loaded is not store
    assert loaded.track_ids == ["A", "B"]
    for track_id in ("A", "B"):
        assert_same_arrays(loaded.arrays(track_id), store.arrays(track_id))
        assert_holds(loaded.arrays(track_id), records[track_id])     # every array, and every mask crop
    assert_holds(store.arrays("A"), records["A"])                    # saving changed nothing


def test_the_file_holds_the_version_and_one_array_per_track_and_key(tmp_path):
    store, records = two_tracks()
    data = read_npz(store.save(tmp_path / "results.npz"))
    assert sorted(data) == sorted(["version"] + [f"{track_id}__{key.name}" for track_id in ("A", "B")
                                                  for key in schema.RESULTS_KEYS])
    assert data["version"].shape == () and data["version"].dtype.kind == "i" and int(data["version"]) == 1
    for track_id in ("A", "B"):
        for key in schema.RESULTS_KEYS:
            array = data[schema.npz_key(track_id, key.name)]
            assert array.dtype == np.dtype(key.dtype), key.name
            assert same(array, expected_column(records[track_id], key)), key.name
    assert all(array.dtype != object for array in data.values())     # readable without pickle


def test_a_store_that_was_loaded_can_be_changed_and_saved_again(tmp_path):
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    loaded = ResultsStore.load(path)
    loaded.replace_from("A", 18)
    loaded.truncate_after("B", 20)
    loaded.put("A", made_up(18, 900))
    loaded.put("C", lost(18, "fine"))
    assert loaded.save(path) == path
    again = ResultsStore.load(path)
    assert again.track_ids == ["A", "B", "C"]
    assert_holds(again.arrays("A"), records["A"][:4] + [made_up(18, 900)])
    assert_holds(again.arrays("B"), records["B"][:4])
    assert_holds(again.arrays("C"), [lost(18, "fine")])


def test_an_empty_store_saves_and_loads(tmp_path):
    path = ResultsStore().save(tmp_path / "results.npz")
    assert list(read_npz(path)) == ["version"]
    assert ResultsStore.load(path).track_ids == []


def test_the_file_is_closed_when_load_returns(tmp_path, monkeypatch):
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    opened = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file) == path:
            opened.append(f)
        return f

    monkeypatch.setattr(builtins, "open", tracking_open)
    loaded = ResultsStore.load(path)
    monkeypatch.undo()
    assert opened                         # the test saw the file being opened
    assert all(f.closed for f in opened)
    path.unlink()                         # the arrays are in memory: they do not need the file
    assert_holds(loaded.arrays("A"), records["A"])
    assert_holds(loaded.arrays("B"), records["B"])


def test_the_file_can_be_saved_again_right_after_loading(tmp_path):
    # Windows refuses to replace a file that a program holds open, this program too: if load left
    # results.npz open, the next autosave would end up in results.new.npz after 5 s of retries.
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    loaded = ResultsStore.load(path)
    loaded.put("A", made_up(24, 50))
    assert loaded.save(path) == path
    assert names(tmp_path) == ["results.npz"]
    assert_holds(ResultsStore.load(path).arrays("A"), records["A"] + [made_up(24, 50)])


@pytest.mark.parametrize("found", [0, 2, 99])
def test_an_unknown_version_raises_a_clear_error(tmp_path, found):
    store, _ = two_tracks()
    data = read_npz(store.save(tmp_path / "results.npz"))
    data["version"] = np.array(found)
    path = write_npz(tmp_path / "other.npz", data)
    with pytest.raises(ResultsVersionError) as err:
        ResultsStore.load(path)
    assert "other.npz" in str(err.value)
    message = str(err.value).replace(str(path), "<the file>")   # a temp folder's name may hold any digit
    assert names_number(message, found), message              # the file's version
    assert names_number(message, 1), message                  # the version this tool reads
    assert (err.value.found, err.value.supported) == (found, 1)
    assert isinstance(err.value, ValueError)   # one "bad input" family for the command line to catch
