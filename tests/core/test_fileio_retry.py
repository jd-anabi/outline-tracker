"""Tests of outline_tracker/core/fileio.py: a locked target: the retry, the .new copy, the rename alone, run.log blocks.

Expected behavior comes from SPEC 8.1 ("temporary file, then rename ... retry for a few seconds if
the target is locked ... if it stays locked, write <name>.new.csv") and from SPEC 8.9 (run.log is
appended per run and per export). Nothing is copied from the output of the code under test.

A lock is imitated on every platform by the `lock_file` fixture (tests/conftest.py), whose own tests are in
tests/core/test_fileio.py with the other tests of atomic writes and the video hash. One test here, for
Windows only, holds a real file open.
"""

import errno
import sys
from pathlib import Path

import pytest
from helpers import names

from outline_tracker.core import fileio
from outline_tracker.core.fileio import atomic_write


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
    assert names(tmp_path) == ["positions.csv"]  # no .new file, no temporary file


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
    assert names(tmp_path) == sorted([name, fallback])


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
    assert names(tmp_path) == ["positions.csv", "positions.new.csv"]


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
    assert names(tmp_path) == ["positions.csv"]


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
    assert names(tmp_path) == ["positions.csv", "positions.new.csv"]


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
    assert names(tmp_path) == ["A.csv"]


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
    assert names(tmp_path) == ["A.csv", "A.csv.new"]  # not A.new.csv, which a loader would read as a track


def test_replace_with_retry_leaves_the_finished_file_to_the_caller_when_both_are_locked(tmp_path, lock_file):
    tmp, target, fallback = tmp_path / "B.csv.tmp", tmp_path / "B.csv", tmp_path / "spare.new"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    lock_file(target, fallback)
    with pytest.raises(PermissionError) as err:
        fileio.replace_with_retry(tmp, target, fallback)
    assert "B.csv" in str(err.value) and "spare.new" in str(err.value)
    assert tmp.read_bytes() == b"new\n" and target.read_bytes() == b"old\n"  # nothing lost, nothing replaced
    assert names(tmp_path) == ["B.csv", "B.csv.tmp"]


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
    assert names(tmp_path) == ["run.log"]


def test_append_block_to_a_locked_log_keeps_the_whole_log_next_to_it(tmp_path, lock_file):
    log = tmp_path / "run.log"
    log.write_bytes(b"an earlier entry\n")
    lock_file(log)
    written = fileio.append_block(str(log), ["a new entry"])
    assert written == tmp_path / "run.new.log"
    assert written.read_bytes() == b"an earlier entry\n\na new entry\n"
    assert log.read_bytes() == b"an earlier entry\n"
