"""Tests of outline_tracker/fileio.py: atomic writes with the Windows lock retry, the rename alone under
names the caller gives, adding a block to run.log, and the video hash.

Expected behavior comes from SPEC 8.1 ("temporary file, then rename ... retry for a few seconds if
the target is locked ... if it stays locked, write <name>.new.csv") and from the definition of
SHA-256: hashlib over the same bytes, and the published digests of the empty message and of "abc"
(FIPS 180-2). Nothing is copied from the output of the code under test.

A lock is imitated on every platform by the `lock_file` fixture (tests/conftest.py): an `os.replace`
that refuses one destination with PermissionError, which is what Windows does while another program
holds that file open. The fixture's own tests are here too. One test, for Windows only, holds a real
file open.
"""

import errno
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from helpers import _names

from outline_tracker import fileio
from outline_tracker.fileio import atomic_write, sha256_first_64mib

MIB = 1024 * 1024


# --------------------------------------------------------------------------- the normal path

def test_writes_the_file_and_returns_its_path(tmp_path):
    target = tmp_path / "positions.csv"
    written = atomic_write(target, lambda tmp: tmp.write_bytes(b"frame,t_s\n0,0.0000000\n"))
    assert written == target
    assert isinstance(written, Path)
    assert target.read_bytes() == b"frame,t_s\n0,0.0000000\n"
    assert _names(tmp_path) == ["positions.csv"]  # the temporary file is gone


def test_accepts_a_path_given_as_text(tmp_path):
    target = tmp_path / "run.log"
    assert atomic_write(str(target), lambda tmp: tmp.write_text("ok\n", encoding="utf-8")) == target
    assert target.read_text(encoding="utf-8") == "ok\n"


def test_replaces_an_existing_file(tmp_path):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n")) == target
    assert target.read_bytes() == b"new\n"
    assert _names(tmp_path) == ["positions.csv"]


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
    assert _names(tmp_path) == [name]


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
    assert _names(tmp_path) == ["results.npz"]
    with np.load(target) as stored:
        np.testing.assert_array_equal(stored["frames"], frames)


def test_creates_the_folder_and_copes_with_spaces_and_accents(tmp_path):
    target = tmp_path / "vidéo test ü" / "edgetam" / "A.csv"
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"t,frame\n")) == target
    assert target.read_bytes() == b"t,frame\n"
    assert _names(target.parent) == ["A.csv"]


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
        assert _names(tmp_path) == []
    else:
        assert target.read_bytes() == old
        assert _names(tmp_path) == ["positions.csv"]


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
    assert target.read_bytes() == b"new\n" and _names(tmp_path) == ["positions.csv"]
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
        assert _names(folder) == ["finished.tmp"]
        os.replace(src, folder / "A.csv.new")  # another name, though it begins with the locked one
        assert _names(folder) == ["A.csv.new"] and (folder / "A.csv.new").read_bytes() == b"new\n"
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
    assert _names(here) == ["A.csv", "B.csv", "C.csv"] and _names(there) == ["A.csv"]


def test_lock_file_skips_the_waits_between_the_tries_and_records_them(tmp_path, lock_file, monkeypatch):
    waited = []
    monkeypatch.setattr(time, "sleep", waited.append)  # where a wait would go if it were passed on
    held = lock_file(tmp_path / "positions.csv", waits="skip")
    # What `fileio` calls between two tries is the fixture's recorder now. It is called under another name
    # here: the rule that no test waits by a delay goes by the names of the calls (tests/test_repo_rules.py),
    # and this call waits for nothing.
    between_tries = fileio.time.sleep
    assert between_tries(0.4) is None
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
    assert target.read_bytes() == b"new\n" and _names(tmp_path) == ["positions.csv"]


def test_lock_file_lets_go_when_it_is_released(tmp_path, lock_file):
    src, target = _finished_and_old(tmp_path)
    held = lock_file(target)
    for _ in range(10):  # locked for good: more often than fileio ever tries
        with pytest.raises(PermissionError):
            os.replace(src, target)
    held.release()  # the other program closes the file
    os.replace(src, target)
    assert target.read_bytes() == b"new\n" and _names(tmp_path) == ["positions.csv"]


def test_lock_file_lets_go_when_the_test_undoes_its_monkeypatch(tmp_path, lock_file, monkeypatch):
    src, target = _finished_and_old(tmp_path)
    real_replace, real_sleep = os.replace, time.sleep
    lock_file(target)
    with pytest.raises(PermissionError):
        os.replace(src, target)
    monkeypatch.undo()
    assert os.replace is real_replace and time.sleep is real_sleep
    os.replace(src, target)
    assert target.read_bytes() == b"new\n" and _names(tmp_path) == ["positions.csv"]


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


# --------------------------------------------------------------------------- a locked target

def test_survives_replace_failing_three_times(tmp_path, lock_file):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    held = lock_file(target, times=3)
    attempts, sleeps = held.attempts, held.sleeps
    written = atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
    assert written == target
    assert target.read_bytes() == b"new\n"
    assert attempts == [target] * 4  # three refusals, then it went through
    assert len(sleeps) == 3  # one wait after each refusal
    assert all(wait > 0 for wait in sleeps)
    assert sum(sleeps) < 5.0
    assert _names(tmp_path) == ["positions.csv"]  # no .new file, no temporary file


@pytest.mark.parametrize(("name", "fallback"), [
    ("positions.csv", "positions.new.csv"),
    ("session.json", "session.new.json"),
    ("overlay.mp4", "overlay.new.mp4"),
    ("results.npz", "results.new.npz"),
    ("README", "README.new"),
])
def test_a_target_that_stays_locked_keeps_the_data_as_new_file(tmp_path, lock_file, name, fallback):
    target = tmp_path / name
    target.write_bytes(b"old\n")
    held = lock_file(target)
    attempts, sleeps = held.attempts, held.sleeps
    written = atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
    assert written == tmp_path / fallback  # <stem>.new<suffix>, returned so the caller can say so
    assert written.read_bytes() == b"new\n"
    assert target.read_bytes() == b"old\n"  # the old file is intact
    assert 4.0 <= sum(sleeps) <= 6.0  # "retry for about 5 s"
    assert all(wait > 0 for wait in sleeps)
    assert attempts == [target] * (len(sleeps) + 1) + [tmp_path / fallback]  # a try after every wait
    assert _names(tmp_path) == sorted([name, fallback])


def test_an_older_new_file_is_replaced(tmp_path, lock_file):
    target, fallback = tmp_path / "positions.csv", tmp_path / "positions.new.csv"
    target.write_bytes(b"old\n")
    fallback.write_bytes(b"from the last locked export\n")
    lock_file(target)
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n")) == fallback
    assert fallback.read_bytes() == b"new\n"
    assert target.read_bytes() == b"old\n"


def test_target_and_new_file_both_locked_is_an_error_without_litter(tmp_path, lock_file):
    target, fallback = tmp_path / "positions.csv", tmp_path / "positions.new.csv"
    target.write_bytes(b"old\n")
    fallback.write_bytes(b"older\n")
    lock_file(target, fallback)
    with pytest.raises(PermissionError) as err:
        atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
    assert "positions.csv" in str(err.value)
    assert "positions.new.csv" in str(err.value)
    assert target.read_bytes() == b"old\n"
    assert fallback.read_bytes() == b"older\n"
    assert _names(tmp_path) == ["positions.csv", "positions.new.csv"]


def test_only_a_permission_error_is_retried(tmp_path, monkeypatch):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    sleeps = []

    def replace(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as err:
        atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
    assert err.value.errno == errno.ENOSPC
    assert sleeps == []
    assert target.read_bytes() == b"old\n"
    assert _names(tmp_path) == ["positions.csv"]


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file that a program holds open")
def test_on_windows_a_target_held_open_gives_the_new_file(tmp_path, monkeypatch):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    sleeps = []
    monkeypatch.setattr(fileio.time, "sleep", sleeps.append)  # the file stays open: no need to wait for real
    with open(target, "rb") as held:  # as a spreadsheet program holds a CSV: others may not replace it
        written = atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
        assert held.read() == b"old\n"
    assert written == tmp_path / "positions.new.csv"
    assert written.read_bytes() == b"new\n"
    assert target.read_bytes() == b"old\n"
    assert 4.0 <= sum(sleeps) <= 6.0
    assert _names(tmp_path) == ["positions.csv", "positions.new.csv"]


# --------------------------------------------------------------------------- the rename alone
# For a folder where `atomic_write`'s own names will not do: the Tracker-format folder may hold no file
# ending in .csv but the tracks, so its temporary file and its fallback are named by the caller.

@pytest.mark.parametrize("refusals", [0, 3])
def test_replace_with_retry_renames_onto_a_target_that_is_or_becomes_free(tmp_path, lock_file, refusals):
    tmp, target = tmp_path / "A.csv.tmp", tmp_path / "A.csv"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    held = lock_file(target, times=refusals)
    attempts, sleeps = held.attempts, held.sleeps
    written = fileio.replace_with_retry(str(tmp), str(target), str(tmp_path / "A.csv.new"))
    assert written == target and isinstance(written, Path)
    assert target.read_bytes() == b"new\n"
    assert attempts == [target] * (refusals + 1) and len(sleeps) == refusals  # one wait after each refusal
    assert _names(tmp_path) == ["A.csv"]


def test_replace_with_retry_keeps_the_data_under_the_callers_fallback_name(tmp_path, lock_file):
    tmp, target, fallback = tmp_path / "A.csv.tmp", tmp_path / "A.csv", tmp_path / "A.csv.new"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    held = lock_file(target)
    attempts, sleeps = held.attempts, held.sleeps
    assert fileio.replace_with_retry(tmp, target, fallback) == fallback
    assert fallback.read_bytes() == b"new\n" and target.read_bytes() == b"old\n"
    assert 4.0 <= sum(sleeps) <= 6.0 and all(wait > 0 for wait in sleeps)  # "retry for a few seconds"
    assert attempts == [target] * (len(sleeps) + 1) + [fallback]  # a try after every wait, then the fallback
    assert _names(tmp_path) == ["A.csv", "A.csv.new"]  # not A.new.csv, which a loader would read as a track


def test_replace_with_retry_leaves_the_finished_file_to_the_caller_when_both_are_locked(tmp_path, lock_file):
    tmp, target, fallback = tmp_path / "B.csv.tmp", tmp_path / "B.csv", tmp_path / "spare.new"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    lock_file(target, fallback)
    with pytest.raises(PermissionError) as err:
        fileio.replace_with_retry(tmp, target, fallback)
    assert "B.csv" in str(err.value) and "spare.new" in str(err.value)
    assert tmp.read_bytes() == b"new\n" and target.read_bytes() == b"old\n"  # nothing lost, nothing replaced
    assert _names(tmp_path) == ["B.csv", "B.csv.tmp"]


# --------------------------------------------------------------------------- a block added to run.log
# SPEC 8.9: run.log is appended per run and per export. `probe` and `export` both add to it with this.

@pytest.mark.parametrize("earlier", [
    pytest.param(None, id="no log yet"),
    pytest.param(b"", id="an empty log"),
    pytest.param(b"==== probe, an earlier entry ====\nwrote: probes.csv\n", id="one entry"),
    pytest.param(b"a log that ends without a line end", id="no line end"),
    pytest.param("a note by Zo\u00eb\n\n".encode(), id="a blank line at the end, and UTF-8"),
])
def test_append_block_adds_the_lines_after_a_blank_line_and_keeps_what_was_there(tmp_path, earlier):
    log = tmp_path / "run.log"
    if earlier is not None:
        log.write_bytes(earlier)
    kept = earlier or b""
    if kept and not kept.endswith(b"\n"):
        kept += b"\n"
    entry = "==== an entry ====\nwrote: 12 um, by Zo\u00eb\n".encode()
    assert fileio.append_block(log, ["==== an entry ====", "wrote: 12 um, by Zo\u00eb"]) == log
    assert log.read_bytes() == kept + (b"\n" if kept else b"") + entry  # what was there, a blank line, the entry
    assert _names(tmp_path) == ["run.log"]


def test_append_block_to_a_locked_log_keeps_the_whole_log_next_to_it(tmp_path, lock_file):
    log = tmp_path / "run.log"
    log.write_bytes(b"an earlier entry\n")
    lock_file(log)
    written = fileio.append_block(str(log), ["a new entry"])
    assert written == tmp_path / "run.new.log"
    assert written.read_bytes() == b"an earlier entry\n\na new entry\n"
    assert log.read_bytes() == b"an earlier entry\n"


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
