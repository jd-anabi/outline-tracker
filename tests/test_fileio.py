"""Tests of outline_tracker/fileio.py: atomic writes with the Windows lock retry, the rename alone under
names the caller gives, adding a block to run.log, and the video hash.

Expected behavior comes from SPEC 8.1 ("temporary file, then rename ... retry for a few seconds if
the target is locked ... if it stays locked, write <name>.new.csv") and from the definition of
SHA-256: hashlib over the same bytes, and the published digests of the empty message and of "abc"
(FIPS 180-2). Nothing is copied from the output of the code under test.

A lock is imitated on every platform by an `os.replace` that refuses one destination with
PermissionError, which is what Windows does while another program holds that file open. One test,
for Windows only, holds a real file open.
"""

import errno
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from outline_tracker import fileio
from outline_tracker.fileio import atomic_write, sha256_first_64mib

MIB = 1024 * 1024


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _refuse(monkeypatch, refused: dict[Path, float]):
    """Make `os.replace` raise PermissionError for each destination in `refused` the given number of
    times (math.inf: always), and record the waits instead of sleeping.

    Returns (attempts, sleeps): the destinations tried, in order, and the waits asked for, in s.
    """
    real_replace = os.replace
    left = dict(refused)
    attempts: list[Path] = []
    sleeps: list[float] = []

    def replace(src, dst):
        dst = Path(dst)
        attempts.append(dst)
        if left.get(dst, 0) > 0:
            left[dst] -= 1
            raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", sleeps.append)
    return attempts, sleeps


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


# --------------------------------------------------------------------------- a locked target

def test_survives_replace_failing_three_times(tmp_path, monkeypatch):
    target = tmp_path / "positions.csv"
    target.write_bytes(b"old\n")
    attempts, sleeps = _refuse(monkeypatch, {target: 3})
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
def test_a_target_that_stays_locked_keeps_the_data_as_new_file(tmp_path, monkeypatch, name, fallback):
    target = tmp_path / name
    target.write_bytes(b"old\n")
    attempts, sleeps = _refuse(monkeypatch, {target: math.inf})
    written = atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n"))
    assert written == tmp_path / fallback  # <stem>.new<suffix>, returned so the caller can say so
    assert written.read_bytes() == b"new\n"
    assert target.read_bytes() == b"old\n"  # the old file is intact
    assert 4.0 <= sum(sleeps) <= 6.0  # "retry for about 5 s"
    assert all(wait > 0 for wait in sleeps)
    assert attempts == [target] * (len(sleeps) + 1) + [tmp_path / fallback]  # a try after every wait
    assert _names(tmp_path) == sorted([name, fallback])


def test_an_older_new_file_is_replaced(tmp_path, monkeypatch):
    target, fallback = tmp_path / "positions.csv", tmp_path / "positions.new.csv"
    target.write_bytes(b"old\n")
    fallback.write_bytes(b"from the last locked export\n")
    _refuse(monkeypatch, {target: math.inf})
    assert atomic_write(target, lambda tmp: tmp.write_bytes(b"new\n")) == fallback
    assert fallback.read_bytes() == b"new\n"
    assert target.read_bytes() == b"old\n"


def test_target_and_new_file_both_locked_is_an_error_without_litter(tmp_path, monkeypatch):
    target, fallback = tmp_path / "positions.csv", tmp_path / "positions.new.csv"
    target.write_bytes(b"old\n")
    fallback.write_bytes(b"older\n")
    _refuse(monkeypatch, {target: math.inf, fallback: math.inf})
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
def test_replace_with_retry_renames_onto_a_target_that_is_or_becomes_free(tmp_path, monkeypatch, refusals):
    tmp, target = tmp_path / "A.csv.tmp", tmp_path / "A.csv"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    attempts, sleeps = _refuse(monkeypatch, {target: refusals})
    written = fileio.replace_with_retry(str(tmp), str(target), str(tmp_path / "A.csv.new"))
    assert written == target and isinstance(written, Path)
    assert target.read_bytes() == b"new\n"
    assert attempts == [target] * (refusals + 1) and len(sleeps) == refusals  # one wait after each refusal
    assert _names(tmp_path) == ["A.csv"]


def test_replace_with_retry_keeps_the_data_under_the_callers_fallback_name(tmp_path, monkeypatch):
    tmp, target, fallback = tmp_path / "A.csv.tmp", tmp_path / "A.csv", tmp_path / "A.csv.new"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    attempts, sleeps = _refuse(monkeypatch, {target: math.inf})
    assert fileio.replace_with_retry(tmp, target, fallback) == fallback
    assert fallback.read_bytes() == b"new\n" and target.read_bytes() == b"old\n"
    assert 4.0 <= sum(sleeps) <= 6.0 and all(wait > 0 for wait in sleeps)  # "retry for a few seconds"
    assert attempts == [target] * (len(sleeps) + 1) + [fallback]  # a try after every wait, then the fallback
    assert _names(tmp_path) == ["A.csv", "A.csv.new"]  # not A.new.csv, which a loader would read as a track


def test_replace_with_retry_leaves_the_finished_file_to_the_caller_when_both_are_locked(tmp_path, monkeypatch):
    tmp, target, fallback = tmp_path / "B.csv.tmp", tmp_path / "B.csv", tmp_path / "spare.new"
    tmp.write_bytes(b"new\n")
    target.write_bytes(b"old\n")
    _refuse(monkeypatch, {target: math.inf, fallback: math.inf})
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


def test_append_block_to_a_locked_log_keeps_the_whole_log_next_to_it(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    log.write_bytes(b"an earlier entry\n")
    _refuse(monkeypatch, {log: math.inf})
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
