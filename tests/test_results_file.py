"""Tests of outline_tracker/results.py: what may go wrong with results.npz (SPEC 8.1, 8.12): a save
that fails or finds the file locked, a file of an odd version, a damaged file. Saving and loading
a good file are in tests/test_results.py.

The files that must be refused are made here, array by array, from a good one. A locked or failing
disk is imitated as in tests/test_fileio.py. Nothing is copied from the output of the code under
test.
"""

import errno

import numpy as np
import pytest
from results_helpers import assert_holds, made_up, names, names_number, read_npz, two_tracks, write_npz

from outline_tracker import results
from outline_tracker.results import ResultsStore, ResultsVersionError


# --------------------------------------------------------------------------- saving


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


def test_save_returns_the_new_file_when_the_target_stays_locked(tmp_path, lock_file):
    store, records = two_tracks()
    path = store.save(tmp_path / "results.npz")
    before = path.read_bytes()
    store.truncate_after("B", 20)
    lock_file(path)  # what Windows does while another program holds results.npz open
    written = store.save(path)
    assert written == tmp_path / "results.new.npz"
    assert path.read_bytes() == before
    assert names(tmp_path) == ["results.new.npz", "results.npz"]
    assert_holds(ResultsStore.load(written).arrays("B"), records["B"][:4])


# --------------------------------------------------------------------------- files it cannot read


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
    assert names_number(str(err.value).replace(str(path), "<the file>"), 1)


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
