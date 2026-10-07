"""Tests of outline_tracker/results.py: results.npz itself (SPEC 8.12, 8.1): `save` and `load`, the
version key, and files that cannot be read. The store in memory is in tests/test_results.py.

What a file must hold is written out from the records that were put into the store
(tests/results_helpers.py); the files that must be refused are made here, array by array. A locked
or failing disk is imitated as in tests/test_fileio.py. Nothing is copied from the output of the
code under test.
"""

import builtins
import errno
import re
from pathlib import Path

import numpy as np
import pytest
from results_helpers import assert_holds, expected_column, lost, made_up, same, two_tracks

from outline_tracker import fileio, results, schema
from outline_tracker.results import ResultsStore, ResultsVersionError


def assert_same_arrays(a, b):
    for key in schema.RESULTS_KEYS:
        assert same(getattr(a, key.name), getattr(b, key.name)), key.name


def read_npz(path):
    with np.load(path) as z:
        return {name: z[name] for name in z.files}


def write_npz(path, data):
    with open(path, "wb") as f:
        np.savez(f, **data)
    return path


def names(folder):
    return sorted(p.name for p in folder.iterdir())


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


@pytest.mark.parametrize("name", ["results.npz", "results", "results.backup", "A.npz.tmp"])
def test_save_writes_exactly_the_file_it_was_given(tmp_path, name):
    # numpy.savez adds ".npz" to a path that does not end in it; save writes into an open file instead.
    store, records = two_tracks()
    path = tmp_path / name
    assert store.save(str(path)) == path
    assert names(tmp_path) == [name]
    assert_holds(ResultsStore.load(str(path)).arrays("B"), records["B"])


def test_a_failed_save_leaves_the_old_file_whole(tmp_path, monkeypatch):
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    before = path.read_bytes()
    store.put("A", made_up(24, 50))

    def savez_that_fails_halfway(file, **arrays):
        file.write(b"PK\x03\x04 half a file")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(results.np, "savez", savez_that_fails_halfway)
    with pytest.raises(OSError) as err:
        store.save(path)
    assert err.value.errno == errno.ENOSPC
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert names(tmp_path) == ["results.npz"]       # no temporary file is left
    assert_holds(ResultsStore.load(path).arrays("A"), records["A"])


def test_save_returns_the_new_file_when_the_target_stays_locked(tmp_path, monkeypatch):
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    before = path.read_bytes()
    store.truncate_after("B", 20)
    real_replace = fileio.os.replace

    def replace(src, dst):   # what Windows does while another program holds results.npz open
        if Path(dst) == path:
            raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)
    written = store.save(path)
    assert written == tmp_path / "results.new.npz"
    assert path.read_bytes() == before
    assert names(tmp_path) == ["results.new.npz", "results.npz"]
    assert_holds(ResultsStore.load(written).arrays("B"), records["B"][:4])


# --------------------------------------------------------------------------- files it cannot read


def _names_number(message, number):
    """True when the text holds this whole number on its own, not as part of "0.1.0" or "12"."""
    return re.search(rf"(?<![\d.]){number}(?!\d|\.\d)", message) is not None


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
    assert _names_number(message, found), message              # the file's version
    assert _names_number(message, 1), message                  # the version this tool reads
    assert (err.value.found, err.value.supported) == (found, 1)
    assert isinstance(err.value, ValueError)   # one "bad input" family for the command line to catch


@pytest.mark.parametrize("version", ["missing", np.array("1"), np.array(1.0), np.array([1, 1]), np.array(True)])
def test_a_missing_or_odd_version_is_a_version_error(tmp_path, version):
    store, _ = two_tracks()
    data = read_npz(store.save(tmp_path / "results.npz"))
    del data["version"]
    if not isinstance(version, str):
        data["version"] = version
    path = write_npz(tmp_path / "outlines.npz", data)
    with pytest.raises(ResultsVersionError) as err:
        ResultsStore.load(path)
    assert "outlines.npz" in str(err.value)
    assert _names_number(str(err.value).replace(str(path), "<the file>"), 1)


def test_the_version_is_checked_before_anything_else_is_read(tmp_path):
    # A newer format may name or shape its arrays differently: say "newer version", not "damaged".
    data = {"version": np.array(2), "A__frames": np.arange(3), "A__xy": np.zeros(3)}
    path = write_npz(tmp_path / "results.npz", data)
    with pytest.raises(ResultsVersionError):
        ResultsStore.load(path)


@pytest.mark.parametrize("damage, names_it", [
    ("an array is missing", "A__outline_px"),
    ("a column is shorter", "A__u"),
    ("a column of another type", "B__area_px"),
    ("the frames are one number", "B__frames"),
    ("the frames are not ascending", "A__frames"),
    ("the mask bytes are cut off", "A__mask_bits"),
])
def test_a_damaged_file_is_refused_with_the_name_of_what_is_wrong(tmp_path, damage, names_it):
    store, _ = two_tracks()
    data = read_npz(store.save(tmp_path / "good.npz"))
    if damage == "an array is missing":
        del data[names_it]
    elif damage == "a column is shorter":
        data[names_it] = data[names_it][:-1]
    elif damage == "a column of another type":
        data[names_it] = data[names_it].astype(np.float64)
    elif damage == "the frames are one number":
        data[names_it] = np.array(14, np.int32)
    elif damage == "the frames are not ascending":
        data[names_it] = data[names_it][::-1].copy()
    else:
        data[names_it] = data[names_it][:-1]
    path = write_npz(tmp_path / "results.npz", data)
    with pytest.raises(ValueError, match=names_it) as err:
        ResultsStore.load(path)
    assert "results.npz" in str(err.value)
    assert not isinstance(err.value, ResultsVersionError)


def test_a_file_that_is_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ResultsStore.load(tmp_path / "results.npz")
