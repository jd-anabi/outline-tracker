"""Tests of outline_tracker/core/fileio.py: atomic writes, the stand-in for a lock, and the video hash.

Expected behavior comes from SPEC 8.1 ("temporary file, then rename ... retry for a few seconds if
the target is locked ... if it stays locked, write <name>.new.csv") and from the definition of
SHA-256: hashlib over the same bytes, and the published digests of the empty message and of "abc"
(FIPS 180-2). Nothing is copied from the output of the code under test.

A lock is imitated on every platform by the `lock_file` fixture (tests/conftest.py): an `os.replace`
that refuses one destination with PermissionError, which is what Windows does while another program
holds that file open. The fixture's own tests are here.

The tests of a target that is locked (the retry, the .new copy, the rename alone, a block added to
run.log) are in tests/core/test_fileio_retry.py. One test there, for Windows only, holds a real file
open.
"""

import errno
import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from helpers import names

from outline_tracker.core import fileio
from outline_tracker.core.fileio import atomic_write, sha256_first_64mib

MIB = 1024 * 1024


# --------------------------------------------------------------------------- the normal path

def test_writes_the_file_and_returns_its_path(tmp_path):
    target = tmp_path / "positions.csv"
    written = atomic_write(target, lambda tmp: tmp.write_bytes(b"frame,t_s\n0,0.0000000\n"))
    assert written == target
    assert isinstance(written, Path)
    assert target.read_bytes() == b"frame,t_s\n0,0.0000000\n"
    assert names(tmp_path) == ["positions.csv"]  # the temporary file is gone


def test_accepts_a_path_given_as_text(tmp_path):
    target = tmp_path / "run.log"
    assert atomic_write(str(target), lambda tmp: tmp.write_text("ok\n", encoding="utf-8")) == target
    assert target.read_text(encoding="utf-8") == "ok\n"


def test_replaces_an_existing_file(tmp_path):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n")) == target
    assert target.read_bytes() == b"new\n"
    assert names(tmp_path) == ["positions.csv"]


@pytest.mark.parametrize("name", ["positions.csv", "overlay.mp4", "results.npz", "session.json", "run.log", "README"])
def test_temporary_file_is_in_the_same_folder_and_keeps_the_suffix(tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"old")
    seen = {}

    def write(tmp):
        seen["tmp"] = tmp
        seen["target_during_write"] = target.read_bytes()
        tmp.write_bytes(b"new")

    atomic_write(target, write)
    tmp = seen["tmp"]
    assert isinstance(tmp, Path)
    assert tmp.parent == target.parent  # same folder, so the rename never crosses a drive
    assert tmp.name != target.name
    assert tmp.name != f"{target.stem}.new{target.suffix}"  # not the name kept for a locked target
    assert tmp.suffix == target.suffix  # ffmpeg and numpy choose the format from the name
    assert seen["target_during_write"] == b"old"  # the old file is whole until the rename
    assert target.read_bytes() == b"new"
    assert names(tmp_path) == [name]


def test_rename_happens_once_the_writer_has_finished(tmp_path, monkeypatch):
    target = tmp_path / "shapes.csv"
    real_replace = os.replace
    calls = []

    def spy(src, dst):
        calls.append((Path(src), Path(dst), Path(src).read_bytes()))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", spy)
    atomic_write(target, lambda tmp: tmp.write_bytes(b"x" * 1000))
    assert len(calls) == 1
    src, dst, data_at_rename = calls[0]
    assert dst == target
    assert src.parent == tmp_path
    assert data_at_rename == b"x" * 1000


@pytest.mark.parametrize("through_file_object", [False, True])
def test_numpy_writes_through_the_temporary_name(tmp_path, through_file_object):
    # np.savez adds ".npz" to a name that does not end in it; the temporary name ends in ".npz"
    target = tmp_path / "results.npz"
    frames = np.arange(96, 106, 2)

    def write(tmp):
        if through_file_object:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, frames=frames)
        else:
            np.savez_compressed(tmp, frames=frames)

    assert atomic_write(target, write) == target
    assert names(tmp_path) == ["results.npz"]
    with np.load(target) as stored:
        np.testing.assert_array_equal(stored["frames"], frames)


def test_creates_the_folder_and_copes_with_spaces_and_accents(tmp_path):
    target = tmp_path / "vidéo test ü" / "edgetam" / "A.csv"
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"t,frame\n")) == target
    assert target.read_bytes() == b"t,frame\n"
    assert names(target.parent) == ["A.csv"]


# --------------------------------------------------------------------------- a writer that fails

@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("old", [b"old\n", None])
def test_a_failing_writer_leaves_the_old_file_and_no_litter(tmp_path, error, old):
    target = tmp_path / "positions.csv"
    if old is not None:
        target.write_bytes(old)

    def write(tmp):
        tmp.write_bytes(b"half a fi")
        raise error("stopped in the middle")

    with pytest.raises(error, match="stopped in the middle"):
        atomic_write(target, write)
    if old is None:
        assert names(tmp_path) == []
    else:
        assert target.read_bytes() == old
        assert names(tmp_path) == ["positions.csv"]


# --------------------------------------------------------------------------- the stand-in for a lock
# `lock_file` (tests/conftest.py) is the locked file of every test that needs one. Its own tests call
# `os.replace` themselves, without fileio: how often a rename is refused is counted from what the test
# asked for, and the files hold the bytes written here.

def _finished_and_old(folder: Path) -> tuple[Path, Path]:
    """A finished file and the file it is to replace, in `folder`: (source, target) of a rename."""
    src, target = folder / "finished.tmp", folder / "positions.csv"
    src.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    return src, target


def test_lock_file_refuses_the_renames_asked_for_and_lets_the_next_one_through(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    held = lock_file(target, times=3)
    for _ in range(3):
        with pytest.raises(PermissionError):
            os.replace(src, target)
        assert (src.read_bytes(), target.read_bytes()) == (b"new\n", b"old\n")  # both as they were
    os.replace(str(src), str(target))  # the fourth, with the paths as text
    assert target.read_bytes() == b"new\n" and names(tmp_path) == ["positions.csv"]
    assert held.attempts == [target] * 4  # every rename tried, as a Path


def test_lock_file_with_a_bare_name_locks_that_name_in_every_folder_and_no_other_file(tmp_path, lock_file):
    held = lock_file("A.csv")
    tried = []
    for folder in (tmp_path / "run", tmp_path / "vidéo test ü" / "edgetam"):
        folder.mkdir(parents=True)
        src = folder / "finished.tmp"
        src.write_bytes(b"new\n")
        with pytest.raises(PermissionError):
            os.replace(src, folder / "A.csv")
        assert names(folder) == ["finished.tmp"]
        os.replace(src, folder / "A.csv.new")  # another name, though it begins with the locked one
        assert names(folder) == ["A.csv.new"] and (folder / "A.csv.new").read_bytes() == b"new\n"
        tried += [folder / "A.csv", folder / "A.csv.new"]
    assert held.attempts == tried  # the free renames are recorded as well


def test_lock_file_with_paths_locks_those_files_only_and_counts_for_each_of_them(tmp_path, lock_file):
    here, there, src = tmp_path / "here", tmp_path / "there", tmp_path / "finished.tmp"
    here.mkdir()
    there.mkdir()

    def rename_onto(target):
        src.write_bytes(b"new\n")
        os.replace(src, target)

    lock_file(here / "A.csv", here / "B.csv", times=1)
    rename_onto(there / "A.csv")  # the same name in another folder is free
    rename_onto(here / "C.csv")  # and so is another file of the same folder
    for locked in (here / "A.csv", here / "B.csv"):
        with pytest.raises(PermissionError):
            rename_onto(locked)  # one refusal for A.csv, and one for B.csv
    rename_onto(here / "A.csv")
    rename_onto(here / "B.csv")
    assert names(here) == ["A.csv", "B.csv", "C.csv"] and names(there) == ["A.csv"]


def test_lock_file_skips_the_waits_between_the_tries_and_records_them(tmp_path, lock_file, monkeypatch):
    waited = []
    monkeypatch.setattr(time, "sleep", waited.append)  # where a wait would go if it were passed on
    held = lock_file(tmp_path / "positions.csv", waits="skip")
    assert fileio.time.sleep(0.4) is None
    assert held.sleeps == [0.4] and waited == []  # recorded, in s, and not waited for


def test_lock_file_with_waits_none_leaves_fileio_one_try_and_no_wait(tmp_path, lock_file, monkeypatch):
    delays, sleep = fileio.RETRY_DELAYS_S, time.sleep
    lock_file(tmp_path / "positions.csv", waits="none")
    assert fileio.RETRY_DELAYS_S == () and time.sleep is sleep  # no wait to make, so none to skip
    monkeypatch.undo()
    assert fileio.RETRY_DELAYS_S == delays


def test_lock_file_takes_no_other_word_for_the_waits_and_then_locks_nothing(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    with pytest.raises(ValueError, match="waits"):
        lock_file(target, waits="real")
    os.replace(src, target)
    assert target.read_bytes() == b"new\n" and names(tmp_path) == ["positions.csv"]


def test_lock_file_lets_go_when_it_is_released(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    held = lock_file(target)
    for _ in range(10):  # locked for good: more often than fileio ever tries
        with pytest.raises(PermissionError):
            os.replace(src, target)
    held.release()  # the other program closes the file
    os.replace(src, target)
    assert target.read_bytes() == b"new\n" and names(tmp_path) == ["positions.csv"]


def test_lock_file_lets_go_when_the_test_undoes_its_monkeypatch(tmp_path, lock_file, monkeypatch):
    src, target = _finished_and_old(tmp_path)
    real_replace, real_sleep = os.replace, time.sleep
    lock_file(target)
    with pytest.raises(PermissionError):
        os.replace(src, target)
    monkeypatch.undo()
    assert os.replace is real_replace and time.sleep is real_sleep
    os.replace(src, target)
    assert target.read_bytes() == b"new\n" and names(tmp_path) == ["positions.csv"]


def test_lock_file_raises_the_error_it_is_given(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    busy = OSError(errno.EBUSY, "Device or resource busy")
    lock_file(target, error=busy)
    with pytest.raises(OSError) as raised:
        os.replace(src, target)
    assert raised.value is busy
    assert (src.read_bytes(), target.read_bytes()) == (b"new\n", b"old\n")


def test_lock_file_refuses_a_rename_made_in_another_thread(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    held = lock_file(target)
    refused = []

    def rename():  # as the window's worker thread renames a file it has written
        try:
            os.replace(src, target)
        except PermissionError as error:
            refused.append(error)

    thread = threading.Thread(target=rename)
    thread.start()
    thread.join()
    assert len(refused) == 1 and held.attempts == [target]
    assert (src.read_bytes(), target.read_bytes()) == (b"new\n", b"old\n")


# --------------------------------------------------------------------------- the video hash

def test_sha256_of_small_files_is_the_plain_sha256(tmp_path):
    empty, abc, other = tmp_path / "empty.bin", tmp_path / "abc.bin", tmp_path / "vidéo ü.bin"
    empty.write_bytes(b"")
    abc.write_bytes(b"abc")
    data = np.random.default_rng(3).bytes(3 * MIB + 17)  # several read blocks and a partial one
    other.write_bytes(data)
    # the two digests published with the standard
    assert sha256_first_64mib(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_first_64mib(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_first_64mib(other) == hashlib.sha256(data).hexdigest()
    assert sha256_first_64mib(str(other)) == hashlib.sha256(data).hexdigest()


def test_sha256_covers_exactly_the_first_64_mib(tmp_path):
    path = tmp_path / "long.bin"
    block = np.random.default_rng(4).bytes(MIB)
    head = hashlib.sha256()
    try:
        with open(path, "wb") as f:
            for _ in range(64):
                f.write(block)
                head.update(block)
            f.write(b"\x01" * MIB)  # 65 MiB in all
        expected = head.hexdigest()
        assert sha256_first_64mib(path) == expected

        with open(path, "r+b") as f:  # the first byte after 64 MiB does not count
            f.seek(64 * MIB)
            f.write(b"\xfe")
        assert sha256_first_64mib(path) == expected

        with open(path, "r+b") as f:  # the last byte before 64 MiB does
            f.seek(64 * MIB - 1)
            f.write(bytes([block[-1] ^ 0xFF]))
        changed = hashlib.sha256()
        for _ in range(63):
            changed.update(block)
        changed.update(block[:-1] + bytes([block[-1] ^ 0xFF]))
        assert sha256_first_64mib(path) == changed.hexdigest()
        assert changed.hexdigest() != expected
    finally:
        path.unlink(missing_ok=True)  # do not leave 65 MiB in the temp folder
