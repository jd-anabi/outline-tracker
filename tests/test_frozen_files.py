"""Frozen reference files: the helper that writes one only when asked (tests/frozen_helpers.py), and
the rules that every file under tests/data/ keeps.

A frozen file holds what the code itself gave, in a run that an independent check confirmed
(docs/ROADMAP.md, section 2, rule 2). It says how it was made: in `# key: value` lines at its start,
or, for files that are compared byte for byte, in a `HEADER.txt` beside them that also lists each
file's size and SHA-256. Every file is ASCII with LF line ends, in every checkout (`.gitattributes`).

The files of the first three tests are made up in the test's own temporary folder; the last test reads
tests/data/ itself. No units or coordinates here: text and bytes only.
"""

import hashlib
import re
from datetime import date
from importlib import metadata

import frozen_helpers
import pytest
from frozen_helpers import (DATA, SIDECAR, SWITCH, freeze_asked, frozen_files, frozen_text, header_lines, machine_name,
                            read_frozen, write_frozen, write_listing)
from test_repo_rules import HOME_PATH

from outline_tracker import provenance, video

COMMAND = "OUTLINE_TRACKER_FREEZE=1 uv run pytest tests/test_frozen_files.py -q"  # made up, for the headers here
NEEDED = ("made", "commit", "tool", "system", "machine", "libraries", "command")  # every header has these keys
HEADER = "".join(f"# {key}: made up\n" for key in NEEDED).encode("ascii")  # a header that keeps the rule


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rule_breaks(folder) -> list[str]:
    """What breaks the rules of tests/data/ in `folder`, sorted, one line per finding:
    `<path below the folder>: <what is wrong>`. The rules: every file is ASCII, has no CR byte and ends
    with LF; it has `# key: value` header lines of its own, or a `HEADER.txt` in its folder or in a
    folder above lists it with its size in bytes and its SHA-256, and both are the file's; every header
    has the keys of `NEEDED`."""
    files = {path: path.read_bytes() for path in frozen_files(folder)}
    found = []
    listed = {}  # a listed path -> (size in bytes, SHA-256), from every HEADER.txt

    def name(path):
        return path.relative_to(folder).as_posix()

    for sidecar, data in files.items():
        if sidecar.name != SIDECAR or not data.isascii():
            continue
        for row in read_frozen(sidecar)[1]:
            cells = row.split(",")
            if len(cells) == 3 and cells[1].isdigit():
                listed[sidecar.parent / cells[0]] = (int(cells[1]), cells[2])
            else:
                found.append(f"{name(sidecar)}: the row {row!r} is not path,size,sha256")
    for path, data in files.items():
        if not data.isascii():
            found.append(f"{name(path)}: not ASCII")
            continue
        if b"\r" in data:
            found.append(f"{name(path)}: has a CR byte")
        if not data.endswith(b"\n"):
            found.append(f"{name(path)}: does not end with LF")
        header = read_frozen(path)[0]
        if path in listed:
            if listed[path] != (len(data), sha256(data)):
                found.append(f"{name(path)}: not the size and SHA-256 that its {SIDECAR} lists")
        elif not header:
            found.append(f"{name(path)}: no header of its own, and no {SIDECAR} lists it")
        missing = [key for key in NEEDED if key not in header]
        if header and missing:
            found.append(f"{name(path)}: its header lacks {', '.join(missing)}")
    found += [f"{name(path)}: listed in a {SIDECAR}, but there is no such file" for path in listed if path not in files]
    return sorted(found)


def test_a_frozen_file_is_written_only_when_asked_and_carries_its_header(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen_helpers, "_package_status", lambda: "")  # the package is as committed
    monkeypatch.setattr(frozen_helpers, "LIBRARIES", (*frozen_helpers.LIBRARIES, "no-such-library"))
    first_day = date.today()
    data = frozen_text(header_lines(COMMAND, {"device": "cpu"}), ["frame,u_px,v_px", "0,700.5000,500.5000"])
    table = tmp_path / "data" / "numbers" / "table.csv"
    raw = tmp_path / "data" / "raw" / "case" / "A.csv"  # a file without a header: its folder's HEADER.txt has it

    # not asked: nothing is written, whatever else the variable holds
    monkeypatch.delenv(SWITCH, raising=False)
    for value in (None, "", "0", "yes", "true"):
        if value is not None:
            monkeypatch.setenv(SWITCH, value)
        assert freeze_asked() is False, value
        assert write_frozen(table, data) is False, value
        assert write_listing(raw.parents[1], COMMAND) is False, value
    assert not (tmp_path / "data").exists()

    monkeypatch.setenv(SWITCH, "1")
    assert freeze_asked() is True
    assert write_frozen(table, data) is True
    written = table.read_bytes()
    assert written == data
    assert b"\r" not in written and written.endswith(b"\n") and written.count(b"\n") == 11

    header, rows = read_frozen(table)
    assert list(header) == ["made", "commit", "tool", "system", "machine", "decoder", "libraries", "command", "device"]
    assert first_day <= date.fromisoformat(header["made"]) <= date.today()
    assert re.fullmatch(r"[0-9a-f]{7,40}", header["commit"])
    assert header["tool"] == provenance.tool_version()
    assert header["system"] == provenance.machine_text()
    assert header["machine"].strip() and header["machine"] == machine_name()
    assert header["decoder"] == video.decoder_tag()
    libraries = header["libraries"].split("; ")
    for name in ("numpy", "opencv-python-headless"):
        assert f"{name} {metadata.version(name)}" in libraries
    assert "no-such-library" not in header["libraries"] and "not installed" not in header["libraries"]
    assert header["command"] == COMMAND and header["device"] == "cpu"
    assert rows == ["frame,u_px,v_px", "0,700.5000,500.5000"]
    assert [line for line in written.decode("ascii").splitlines() if HOME_PATH.search(line)] == []

    # data with other line ends, or without the last one, is refused and not written
    for wrong in (b"0,700.5000\r\n", b"0,700.5000"):
        with pytest.raises(ValueError, match="LF"):
            write_frozen(tmp_path / "data" / "wrong.csv", wrong)
    assert not (tmp_path / "data" / "wrong.csv").exists()

    # files without a header inside: HEADER.txt beside them says how they were made and lists them
    raw_data = b"t,frame\n0.0000000,40\n"  # 8 and 13 bytes
    assert write_frozen(raw, raw_data) is True
    (raw.parent / ".DS_Store").write_bytes(b"\x00\x01Bud1")  # what an operating system leaves is no frozen file
    assert write_listing(raw.parents[1], COMMAND, {"checked": "by hand"}) is True
    listing = (raw.parents[1] / SIDECAR).read_bytes()
    assert b"\r" not in listing and listing.endswith(b"\n")
    header, rows = read_frozen(raw.parents[1] / SIDECAR)
    assert list(header)[:8] == ["made", "commit", "tool", "system", "machine", "decoder", "libraries", "command"]
    assert list(header)[8:] == ["each row", "checked"] and header["checked"] == "by hand"
    assert rows == [f"case/A.csv,21,{sha256(raw_data)}"]
    assert [line for line in listing.decode("ascii").splitlines() if HOME_PATH.search(line)] == []
    assert write_listing(raw.parents[1], COMMAND, {"checked": "by hand"}) is True  # made again: it does not list itself
    assert read_frozen(raw.parents[1] / SIDECAR)[1] == rows
    assert rule_breaks(tmp_path / "data") == []


def test_freezing_is_refused_when_the_package_differs_from_the_commit(tmp_path, monkeypatch):
    # what git says in this checkout names files of the package only (nothing at all, in a clean one)
    assert [line for line in frozen_helpers._package_status().splitlines() if "outline_tracker/" not in line] == []
    target = tmp_path / "table.csv"
    monkeypatch.setattr(frozen_helpers, "_package_status", lambda: " M outline_tracker/measure.py")
    monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(RuntimeError, match="outline_tracker/measure.py") as refused:
        write_frozen(target, b"0,700.5000\n")
    assert "commit" in str(refused.value)  # the sentence says why
    with pytest.raises(RuntimeError, match="outline_tracker/measure.py"):
        write_listing(tmp_path, COMMAND)
    assert list(tmp_path.iterdir()) == []
    # an ordinary run does not freeze: it asks git nothing, and a package that is being worked on does not stop it
    monkeypatch.delenv(SWITCH)
    monkeypatch.setattr(frozen_helpers, "_git", lambda *args: pytest.fail(f"an ordinary run asked git: {args}"))
    assert write_frozen(target, b"0,700.5000\n") is False
    assert write_listing(tmp_path, COMMAND) is False
    assert list(tmp_path.iterdir()) == []


def test_the_check_of_tests_data_finds_each_broken_rule(tmp_path):
    # The check passes trivially on an empty folder. Here it gets a folder with files that keep every
    # rule, and beside them one file for each way of breaking one.
    listed = b"t,frame\n0.0000000,40\n"
    files = {
        "numbers/table.csv": HEADER + b"frame,u_px\n0,700.5000\n",
        "raw/HEADER.txt": HEADER + f"case/A.csv,{len(listed)},{sha256(listed)}\n".encode("ascii"),
        "raw/case/A.csv": listed,
        "raw/case/.DS_Store": b"\x00\x01Bud1",  # left by an operating system: not a frozen file
        "bad/crlf.csv": HEADER.replace(b"\n", b"\r\n") + b"0,700.5000\r\n",
        "bad/open_end.csv": HEADER + b"0,700.5000",
        "bad/umlaut.csv": HEADER + "0,700.5000 µm\n".encode("utf-8"),
        "bad/bare.csv": b"0,700.5000\n",
        "bad/short.csv": HEADER.replace(b"# machine: made up\n", b"").replace(b"# command: made up\n", b"") + b"0\n",
        "bad/listed/HEADER.txt": b"# made: made up\n"
                                 + f"other_bytes.csv,{len(listed)},{sha256(listed)}\n".encode("ascii")
                                 + f"other_size.csv,{len(listed) + 1},{sha256(listed)}\n".encode("ascii")
                                 + f"gone.csv,{len(listed)},{sha256(listed)}\n".encode("ascii")
                                 + b"no size and no hash\n",
        "bad/listed/other_bytes.csv": listed.replace(b"40", b"41"),
        "bad/listed/other_size.csv": listed,
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert rule_breaks(tmp_path) == [
        "bad/bare.csv: no header of its own, and no HEADER.txt lists it",
        "bad/crlf.csv: has a CR byte",
        "bad/listed/HEADER.txt: its header lacks commit, tool, system, machine, libraries, command",
        "bad/listed/HEADER.txt: the row 'no size and no hash' is not path,size,sha256",
        "bad/listed/gone.csv: listed in a HEADER.txt, but there is no such file",
        "bad/listed/other_bytes.csv: not the size and SHA-256 that its HEADER.txt lists",
        "bad/listed/other_size.csv: not the size and SHA-256 that its HEADER.txt lists",
        "bad/open_end.csv: does not end with LF",
        "bad/short.csv: its header lacks machine, command",
        "bad/umlaut.csv: not ASCII",
    ]


def test_every_file_under_tests_data_has_lf_line_ends_and_is_listed():
    assert rule_breaks(DATA) == []
    # the check saw the folder: the Tracker-format files of the three stand-in cases and their HEADER.txt
    names = [path.relative_to(DATA).as_posix() for path in frozen_files(DATA)]
    assert [name for name in names if name.startswith("tracker_format/")] == [
        "tracker_format/HEADER.txt",
        "tracker_format/_every_frame_and_a_jump/A.csv",
        "tracker_format/_one_track/A.csv",
        "tracker_format/_start_file_and_a_lost_disk/A.csv",
        "tracker_format/_start_file_and_a_lost_disk/B.csv",
    ]
