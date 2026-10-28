"""A save over a session.json of version 1 keeps that file beside the new one (rule 5 of docs/ROADMAP.md
section 2, decisions 36 and 39; docs/design/formats.md, "Writing over a version 1 file").

Every old file is built in its test from literal values: an example of docs/design/formats.md, written with a
byte order mark and CRLF line ends, so that only a byte-for-byte copy equals it. The names of the kept files
and what a note says come from that page and from decision 36. Nothing is copied from the output of the code
under test.
"""

import builtins
import codecs
import errno
import json
import os
from pathlib import Path

import pytest
from session_json_helpers import _write_json, assert_same_json, formats_example

from outline_tracker.core.session import Session

OLD_NS = 1_700_000_000 * 10**9  # the old file's time: a whole second, which every file system keeps
OTHER_NS = 1_710_000_000 * 10**9


def old_bytes(number: int = 1) -> bytes:
    """Example `number` of formats.md in version 1, as bytes that no save of this build writes."""
    text = json.dumps(formats_example(number, 1), indent=1, ensure_ascii=False).replace("\n", "\r\n")
    return codecs.BOM_UTF8 + text.encode("utf-8")


def old_file(tmp_path, number: int = 1):
    """`<tmp_path>/run/session.json` holding `old_bytes(number)`, with the time `OLD_NS`."""
    path = tmp_path / "run" / "session.json"
    path.parent.mkdir()
    path.write_bytes(old_bytes(number))
    os.utime(path, ns=(OLD_NS, OLD_NS))
    return path


def listing(folder) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# --------------------------------------------------------------------------- the first save


@pytest.mark.parametrize("number", [1, 2])
def test_the_first_save_keeps_the_old_file_beside_the_new_one(tmp_path, number):
    path = old_file(tmp_path, number)
    notes = []
    assert Session.load(path).save(path, on_kept=notes.append) == path
    assert listing(path.parent) == ["session.json", "session.v1.json"]
    kept = path.with_name("session.v1.json")
    assert kept.read_bytes() == old_bytes(number)  # byte for byte
    assert kept.stat().st_mtime_ns == OLD_NS  # and with the time of the old file
    assert_same_json(json.loads(path.read_bytes()), formats_example(number, 2))  # the new file is version 2


def test_the_note_names_the_kept_file_and_how_0_1_0_opens_it(tmp_path):
    path = old_file(tmp_path)
    notes = []
    Session.load(path).save(path, on_kept=notes.append)
    (note,) = notes
    assert "session.v1.json" in note and "session.json" in note
    assert "0.1.0" in note and "renamed to session.json" in note
    assert str(tmp_path) not in note and "\n" not in note  # one line, with names and no folders


def test_a_later_save_writes_session_json_alone(tmp_path):
    path = old_file(tmp_path)
    session = Session.load(path)
    session.save(path)
    kept = path.with_name("session.v1.json")
    os.utime(kept, ns=(OTHER_NS, OTHER_NS))
    session.notes = "third try"
    notes = []
    assert session.save(path, on_kept=notes.append) == path
    assert notes == [] and listing(path.parent) == ["session.json", "session.v1.json"]
    assert kept.read_bytes() == old_bytes() and kept.stat().st_mtime_ns == OTHER_NS  # not written again
    assert json.loads(path.read_bytes())["notes"] == "third try"


@pytest.mark.parametrize("there", [False, True])
def test_nothing_is_kept_without_a_file_or_over_a_file_of_version_2(tmp_path, there):
    path = tmp_path / "run" / "session.json"
    if there:
        _write_json(path, formats_example(1, 2))
    notes = []
    assert Session.from_json(formats_example(2, 2)).save(path, on_kept=notes.append) == path
    assert notes == [] and listing(path.parent) == ["session.json"]
    assert_same_json(json.loads(path.read_bytes()), formats_example(2, 2))


# --------------------------------------------------------------------------- the kept name is taken (decision 36)


def test_the_same_bytes_under_the_kept_name_are_not_written_again(tmp_path):
    path = old_file(tmp_path)
    kept = path.with_name("session.v1.json")
    kept.write_bytes(old_bytes())
    os.utime(kept, ns=(OTHER_NS, OTHER_NS))
    notes = []
    Session.load(path).save(path, on_kept=notes.append)
    assert listing(path.parent) == ["session.json", "session.v1.json"]
    assert kept.read_bytes() == old_bytes() and kept.stat().st_mtime_ns == OTHER_NS  # nothing more written
    (note,) = notes
    assert "session.v1.json" in note


def test_other_bytes_under_the_kept_name_send_the_old_file_to_the_next_free_name(tmp_path):
    path = old_file(tmp_path)
    taken = {"session.v1.json": old_bytes(2), "session.v1-2.json": old_bytes(2) + b"\r\n"}
    for name, data in taken.items():
        path.with_name(name).write_bytes(data)
    notes = []
    Session.load(path).save(path, on_kept=notes.append)
    assert listing(path.parent) == ["session.json", "session.v1-2.json", "session.v1-3.json", "session.v1.json"]
    for name, data in taken.items():
        assert path.with_name(name).read_bytes() == data  # a kept file is never overwritten
    assert path.with_name("session.v1-3.json").read_bytes() == old_bytes()
    (note,) = notes
    assert "session.v1-3.json" in note


def test_the_same_bytes_under_a_later_name_are_not_written_again(tmp_path):
    # what a save leaves when session.json stayed locked after its copy: the next save makes no second copy
    path = old_file(tmp_path)
    path.with_name("session.v1.json").write_bytes(old_bytes(2))
    path.with_name("session.v1-2.json").write_bytes(old_bytes())
    notes = []
    Session.load(path).save(path, on_kept=notes.append)
    assert listing(path.parent) == ["session.json", "session.v1-2.json", "session.v1.json"]
    assert path.with_name("session.v1.json").read_bytes() == old_bytes(2)
    assert path.with_name("session.v1-2.json").read_bytes() == old_bytes()
    (note,) = notes
    assert "session.v1-2.json" in note


# --------------------------------------------------------------------------- when the copy or the save fails


def test_when_the_copy_fails_session_json_is_not_replaced_and_the_error_says_why(tmp_path, monkeypatch):
    path = old_file(tmp_path)
    session = Session.load(path)
    real_open = builtins.open

    class Full:
        """A file on a disk that is full: it is made, and nothing can be written into it."""

        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.file.close()

        def write(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):
            self.file.close()

    def disk_full_for_kept_files(file, mode="r", *args, **kwargs):
        opened = real_open(file, mode, *args, **kwargs)
        return Full(opened) if os.fspath(file).endswith("session.v1.json") else opened

    monkeypatch.setattr(builtins, "open", disk_full_for_kept_files)
    notes = []
    with pytest.raises(OSError) as error:
        session.save(path, on_kept=notes.append)
    monkeypatch.undo()
    assert path.read_bytes() == old_bytes() and path.stat().st_mtime_ns == OLD_NS  # not replaced
    assert listing(path.parent) == ["session.json"]  # and no half copy is left
    message = str(error.value)
    assert "session.json" in message and "session.v1.json" in message and "not replaced" in message
    assert "No space left on device" in message and str(tmp_path) not in message
    assert notes == []


def test_a_locked_session_json_keeps_the_old_file_once_and_tells_once_it_is_replaced(tmp_path, lock_file):
    path = old_file(tmp_path)
    session = Session.load(path)
    lock = lock_file(path)
    notes = []
    for _ in range(2):  # the delayed saves of the window while another program holds session.json
        assert session.save(path, on_kept=notes.append) == path.with_name("session.new.json")
        assert path.read_bytes() == old_bytes() and notes == []  # session.json is not replaced yet
        assert listing(path.parent) == ["session.json", "session.new.json", "session.v1.json"]
    assert path.with_name("session.v1.json").read_bytes() == old_bytes()
    lock.release()
    assert session.save(path, on_kept=notes.append) == path
    assert listing(path.parent) == ["session.json", "session.new.json", "session.v1.json"]
    assert len(notes) == 1 and "session.v1.json" in notes[0]
    assert json.loads(path.read_bytes())["schema_version"] == 2


@pytest.mark.parametrize("version", [1, 2])
def test_a_session_json_that_cannot_be_read_is_not_replaced_and_the_session_goes_beside_it(tmp_path, monkeypatch,
                                                                                         version):
    # Windows refuses to read a file that another program holds open without sharing it: what it holds cannot
    # be told, so it is neither kept nor replaced, and the session goes where a locked file sends it
    # (`fileio.atomic_write`: <stem>.new<suffix>)
    path = old_file(tmp_path)
    if version == 2:
        path.write_bytes(json.dumps(formats_example(1, 2)).encode("utf-8"))
    before = path.read_bytes()
    session = Session.load(path)
    real = Path.read_bytes

    def held(self):
        if self.name == "session.json":
            raise PermissionError(errno.EACCES, "The process cannot access the file because it is being used by "
                                                "another process", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", held)
    notes = []
    assert session.save(path, on_kept=notes.append) == path.with_name("session.new.json")
    monkeypatch.undo()
    assert path.read_bytes() == before and notes == []
    assert listing(path.parent) == ["session.json", "session.new.json"]
    assert_same_json(json.loads(path.with_name("session.new.json").read_bytes()), formats_example(1, 2))
