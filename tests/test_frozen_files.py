"""Frozen reference files: the helper that writes one only when asked (tests/frozen_helpers.py), and
the rules that every file under tests/data/ keeps.

A frozen file holds what the code itself gave, in a run that an independent check confirmed
(docs/ROADMAP.md, section 2, rule 2). It says how it was made: in `# key: value` lines at its start,
or, for files that are compared byte for byte, in a `HEADER.txt` beside them that also lists each
file's size and SHA-256. Every file is ASCII with LF line ends, in every checkout (`.gitattributes`).

The files of the first three tests are made up in the test's own temporary folder; the fourth reads
tests/data/ itself. Those four hold text and bytes only.

The tests after them are about the frozen numbers of the real model: where EdgeTAM found each object of
two synthetic clips on the processor (tests/data/edgetam_cpu_positions.csv) and the hash of its weights
(tests/data/edgetam_weights.txt). No model runs here: the table is compared with the clips' true centers,
worked out from the recipes that draw the clips. Positions are in px in Tracker's convention (pixel
centers at +0.5, u to the right, v downward) in the full 1920 x 1080 frame; frames are video frame
numbers. The slow tests that run the model against the two files (tests/slow/test_frozen_reference.py)
leave two decisions to the helper, and both are tested here with made-up headers and positions: on which
machine the limit of 0.01 px is asserted (`same_machine`), and how positions are judged
(`compare_with_frozen`).

The last test is about the golden Tracker-format files (tests/data/tracker_format/): with which decoder
their bytes are asserted (`same_decoder`), on made-up decoder tags. How a file is judged with another
decoder is tested where the three cases are (tests/test_from_tracker_port.py).
"""

import hashlib
import math
import platform
import re
import sys
from datetime import date
from importlib import metadata

import frozen_helpers
import numpy as np
import pytest
from frozen_helpers import (COLUMNS, DATA, GOLDEN, POSITIONS, SIDECAR, SWITCH, WEIGHTS, compare_with_frozen,
                            freeze_asked, frozen_files, frozen_text, header_lines, machine_here, machine_name,
                            machine_of, position_rows, read_frozen, read_positions, read_weights, same_decoder,
                            same_machine, write_frozen, write_listing)
from helpers import HOME_PATH

from outline_tracker import provenance, video

COMMAND = "OUTLINE_TRACKER_FREEZE=1 uv run pytest tests/test_frozen_files.py -q"  # made up, for the headers here
NEEDED = ("made", "commit", "tool", "system", "machine", "libraries", "command")  # every header has these keys
HEADER = "".join(f"# {key}: made up\n" for key in NEEDED).encode("ascii")  # a header that keeps the rule

FRAMES = list(range(0, 40, 2))  # the tracked frames of both clips: 0, 2, ..., 38
# Where the clips' recipes draw each object: (center on frame 0, step per frame), in array coordinates
# (pixel centers at whole numbers), px. The selftest clip: outline_tracker/synthetic.py, `selftest_clip`.
# The three ellipses: `starts` and `steps` of `write_three_ellipse_clip` in
# tests/slow/pipeline_helpers.py.
RECIPES = {
    ("selftest", "selftest"): ((700.0, 500.0), (0.6, 0.2)),
    ("three_ellipses", "A"): ((500.0, 300.0), (0.6, 0.2)),
    ("three_ellipses", "B"): ((1000.0, 600.0), (-0.5, 0.3)),
    ("three_ellipses", "C"): ((1400.0, 350.0), (0.2, -0.6)),
}
TRUTH_LIMIT_PX = 3.0  # the selftest criterion of SPEC 13.4: every found center under 3 px from the true one


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def true_center(clip: str, track_id: str, frame: int) -> tuple[float, float]:
    """(u_px, v_px) of an object's true center on a frame, in Tracker's convention: where the recipe
    draws it, plus 0.5."""
    (x0, y0), (dx, dy) = RECIPES[clip, track_id]
    return x0 + dx * frame + 0.5, y0 + dy * frame + 0.5


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
    # and the two files of the real model's numbers, each with a header of its own
    assert [name for name in names if "/" not in name] == ["edgetam_cpu_positions.csv", "edgetam_weights.txt"]


def test_a_positions_table_is_written_with_4_decimals_and_read_back(tmp_path):
    # made-up positions, px: two frames, two objects; B is lost on frame 2
    nan = float("nan")
    found = [[(700.5, 500.25), (1000.123449, 600.99995)], [(701.7, 500.65), (nan, nan)]]
    rows = position_rows("made_up", [0, 2], ["A", "B"], found)
    assert rows == [
        "made_up,0,A,700.5000,500.2500",
        "made_up,0,B,1000.1234,601.0000",
        "made_up,2,A,701.7000,500.6500",
        "made_up,2,B,,",  # a lost row: two empty cells
    ]
    table = tmp_path / "table.csv"
    table.write_bytes(frozen_text(["made: made up", "device: cpu"], [COLUMNS, *rows]))
    header, read = read_positions(table)
    assert header == {"made": "made up", "device": "cpu"}
    assert read[:3] == [("made_up", 0, "A", 700.5, 500.25), ("made_up", 0, "B", 1000.1234, 601.0),
                        ("made_up", 2, "A", 701.7, 500.65)]
    assert read[3][:3] == ("made_up", 2, "B") and math.isnan(read[3][3]) and math.isnan(read[3][4])
    assert len(read) == 4
    # a table with other columns is not read as this one
    table.write_bytes(frozen_text(["made: made up"], ["clip,frame,track_id,v_px,u_px", *rows]))
    with pytest.raises(ValueError, match="clip,frame,track_id,u_px,v_px"):
        read_positions(table)


def test_the_frozen_positions_are_within_3_px_of_the_clips_true_centers():
    header, rows = read_positions()
    # 20 rows of the selftest clip (one object), then 60 of the three ellipses (A, B, C on every frame)
    assert [row[:3] for row in rows] == ([("selftest", frame, "selftest") for frame in FRAMES]
                                         + [("three_ellipses", frame, name) for frame in FRAMES for name in "ABC"])
    farthest = {"selftest": 0.0, "three_ellipses": 0.0}  # px from the true centers
    for clip, frame, track_id, u_px, v_px in rows:
        assert not math.isnan(u_px) and not math.isnan(v_px), f"{clip}, {track_id}, frame {frame}: a lost row"
        true_u, true_v = true_center(clip, track_id, frame)
        distance = math.hypot(u_px - true_u, v_px - true_v)
        assert distance < TRUTH_LIMIT_PX, f"{clip}, {track_id}, frame {frame}: {distance:.3f} px from the true center"
        farthest[clip] = max(farthest[clip], distance)

    # the text: the column line, then numbers with 4 decimals
    text = read_frozen(POSITIONS)[1]
    assert text[0] == COLUMNS == "clip,frame,track_id,u_px,v_px" and len(text) == 1 + 20 + 60
    assert [row for row in text[1:] if not re.fullmatch(r"\w+,\d+,\w+,\d+\.\d{4},\d+\.\d{4}", row)] == []

    # the header says how the numbers were made, and what it says about the truth is what the table gives
    assert list(header) == [*NEEDED[:5], "decoder", *NEEDED[5:], "device", "torch threads", "weights sha256",
                            "agreement with the template", "largest distance from the true centers"]
    assert header["device"] == "cpu" and int(header["torch threads"]) >= 1
    assert float(header["agreement with the template"].partition(" px")[0]) <= 0.01
    said = dict(re.findall(r"(\w+) (\d+\.\d+) px", header["largest distance from the true centers"]))
    assert sorted(said) == sorted(farthest)
    for clip, distance in farthest.items():  # the table has 4 decimals and the line 3: equal within 0.001 px
        assert abs(float(said[clip]) - distance) < 0.001, clip


def test_the_frozen_weights_file_names_the_weights_of_the_frozen_positions():
    header, facts = read_weights()
    assert [key for key in NEEDED if key not in header] == []
    assert list(facts)[:2] == ["sha256", "bytes"]
    assert re.fullmatch(r"[0-9a-f]{64}", facts["sha256"]) and int(facts["bytes"]) > 0
    assert facts["sha256"] == read_positions()[0]["weights sha256"]  # one run froze both files


def test_the_limit_is_asserted_only_on_the_machine_that_froze_the_numbers():
    # a made-up header, and the machine that made it as `machine_here` would describe it
    header = {"system": "macOS-27.0.1-arm64-arm-64bit; Python 3.12.15", "machine": "Apple M1 Max",
              "decoder": "opencv-5.0.0/darwin/arm64",
              "libraries": "torch 2.14.1; torchvision 0.29.1; transformers 5.18.0; numpy 2.5.3",
              "weights sha256": "ab" * 32}
    here = {"platform": "darwin", "architecture": "arm64", "chip": "Apple M1 Max", "torch": "2.14.1",
            "weights sha256": "ab" * 32}
    assert machine_of(header) == here
    assert same_machine(header, here) is True

    # the same machine after an update of the system, of Python or of another library: the rule names none of them
    updated = {**header, "system": "macOS-27.1-arm64-arm-64bit; Python 3.12.16", "decoder": "opencv-5.1.0/darwin/arm64",
               "libraries": "torch 2.14.1; torchvision 0.30.0; transformers 5.19.0; numpy 2.6.0"}
    assert same_machine(updated, here) is True

    # another machine, one fact at a time
    others = {
        "another chip": {"machine": "Apple M3 Pro"},
        "another torch version": {"libraries": "torch 2.15.0; torchvision 0.29.1; transformers 5.18.0; numpy 2.5.3"},
        "another hash": {"weights sha256": "cd" * 32},
        "another operating system": {"decoder": "opencv-5.0.0/linux/arm64"},
        "another architecture": {"decoder": "opencv-5.0.0/darwin/x86_64"},
    }
    for what, lines in others.items():
        assert same_machine({**header, **lines}, here) is False, what
    # a header without one of the lines names no machine, and neither does an empty fact on both sides
    for line in ("machine", "decoder", "libraries", "weights sha256"):
        assert same_machine({key: value for key, value in header.items() if key != line}, here) is False, line
    assert same_machine({**header, "machine": ""}, {**here, "chip": ""}) is False

    # asked without the hash, as the test of the weights file asks: the hash plays no part, the machine does
    no_hash = {key: value for key, value in here.items() if key != "weights sha256"}
    assert same_machine({**header, "weights sha256": "cd" * 32}, no_hash) is True
    assert same_machine({**header, "machine": "Apple M3 Pro"}, no_hash) is False

    # this machine, described directly and read back from a header made here: the same facts
    assert machine_here() == {"platform": sys.platform, "architecture": platform.machine(), "chip": machine_name(),
                              "torch": metadata.version("torch")}
    assert machine_here("ab" * 32) == {**machine_here(), "weights sha256": "ab" * 32}
    made_here = dict(line.split(": ", 1) for line in header_lines(COMMAND, {"weights sha256": "ab" * 32}))
    assert machine_of(made_here) == machine_here("ab" * 32)
    assert same_machine(made_here, machine_here("ab" * 32)) is True and same_machine(made_here, machine_here()) is True
    assert same_machine(made_here, machine_here("cd" * 32)) is False

    # the two frozen files hold every fact that the rule asks them for
    assert [key for key, value in machine_of(read_positions()[0]).items() if not value] == []
    assert [key for key, value in machine_of(read_frozen(WEIGHTS)[0]).items() if not value] == ["weights sha256"]


def test_positions_are_judged_by_the_frozen_numbers_here_and_by_the_truth_elsewhere(capsys):
    # made-up positions, px: frozen[i][k] = (u_px, v_px) of one object on three frames; every frozen row
    # is 0.5 px from its true center
    frozen = np.array([[[700.5, 500.5]], [[701.7, 500.9]], [[702.9, 501.3]]])
    true = frozen + (0.3, -0.4)

    def judge(found, strict, frozen=frozen):
        compare_with_frozen("made up", found, frozen, true, strict)

    near = frozen + (0.0048, 0.0064)  # every row 0.008 px from its frozen one
    moved = frozen.copy()
    moved[1, 0] += (0.012, 0.016)  # one row 0.02 px from its frozen one
    far = frozen.copy()
    far[2, 0] = true[2, 0] + (3.1, 0.0)  # one row 3.1 px from its true center
    lost = frozen.copy()
    lost[0, 0] = np.nan  # one row not found

    # on the machine that froze the numbers: 0.01 px from the frozen ones, and the same rows lost
    judge(near, strict=True)
    line = capsys.readouterr().out
    assert line.startswith("VALIDATION frozen numbers, made up: ") and line.count("\n") == 1
    assert "frozen positions 0.0080 px over 3 rows" in line and "true centers 0.502 px" in line
    assert "lost rows: frozen 0, found 0" in line
    with pytest.raises(AssertionError, match=r"0\.0200 px from the frozen positions"):
        judge(moved, strict=True)
    with pytest.raises(AssertionError, match="lost"):
        judge(lost, strict=True)
    judge(lost, strict=True, frozen=lost)  # the rows that the frozen table has lost

    # on another machine: the limit is measured and printed, not asserted; the truth and "no row lost" are
    capsys.readouterr()
    judge(moved, strict=False)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and all(line.startswith("VALIDATION frozen numbers, made up: ") for line in lines)
    assert "not measured" in lines[1] and "0.0200 px" in lines[1]
    with pytest.raises(AssertionError, match=r"3\.100 px from the true centers"):
        judge(far, strict=False)
    with pytest.raises(AssertionError, match="lost"):
        judge(lost, strict=False)
    with pytest.raises(AssertionError, match="lost"):
        judge(lost, strict=False, frozen=lost)  # the same rows as the frozen table, but a row is lost


def test_the_golden_bytes_are_asserted_only_with_the_decoder_that_froze_them():
    # a made-up header; a tag is what `video.decoder_tag()` gives: OpenCV's version, the system, the architecture
    header = {"machine": "Apple M1 Max", "decoder": "opencv-5.0.0/darwin/arm64"}
    assert same_decoder(header, "opencv-5.0.0/darwin/arm64") is True
    others = {
        "another OpenCV version": "opencv-5.1.0/darwin/arm64",
        "another system": "opencv-5.0.0/linux/arm64",
        "another architecture": "opencv-5.0.0/darwin/x86_64",
    }
    for what, tag in others.items():
        assert same_decoder(header, tag) is False, what
    # the rule asks for the decoder and for nothing else: another chip with the same decoder asserts the bytes
    assert same_decoder({**header, "machine": "Apple M3 Pro"}, "opencv-5.0.0/darwin/arm64") is True
    # a header without the line names no decoder, and neither does an empty tag on both sides
    assert same_decoder({"machine": "Apple M1 Max"}, "opencv-5.0.0/darwin/arm64") is False
    assert same_decoder({**header, "decoder": ""}, "") is False

    # a header made here names this machine's decoder
    made_here = dict(line.split(": ", 1) for line in header_lines(COMMAND))
    assert same_decoder(made_here, video.decoder_tag()) is True
    # the header of the golden files has the line that the rule asks for, and it is a tag
    assert re.fullmatch(r"opencv-[\d.]+/\w+/\w+", read_frozen(GOLDEN / SIDECAR)[0]["decoder"])
