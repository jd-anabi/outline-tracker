"""The frozen numbers of the real model, and the machine and decoder rules that decide where they are asserted.

The tests are about the frozen numbers of the real model: where EdgeTAM found each object of
two synthetic clips on the processor (tests/data/edgetam_cpu_positions.csv) and the hash of its weights
(tests/data/edgetam_weights.txt). No model runs here: the table is compared with the clips' true centers,
worked out from the recipes that draw the clips. Positions are in px in Tracker's convention (pixel
centers at +0.5, u to the right, v downward) in the full 1920 x 1080 frame; frames are video frame
numbers. The slow tests that run the model against the two files (tests/slow/test_frozen_reference.py)
leave two decisions to the helper (tests/support/frozen_numbers_helpers.py), and both are tested here with
made-up headers and positions: on which machine the limit of 0.01 px is asserted (`same_machine`), and how
positions are judged (`compare_with_frozen`). So is the reason that one of those tests is skipped with where
the limit was not asserted (`why_the_limit_was_not_asserted`).

The test after those is about the golden Tracker-format files (tests/data/tracker_format/): with which
decoder their bytes are asserted (`same_decoder`), on made-up decoder tags. How a file is judged with
another decoder is tested where the three cases are (tests/workflows/test_from_tracker_port.py).

The last two tests hold the installed torch and OpenCV to the versions that the headers name. The strict
checks (the limit of 0.01 px, the weights' hash, the golden bytes) are made only with those versions, so
with others no computer would make them and every run would stay green: the first of the two fails
instead, and the second shows that it does.

The helper that writes a frozen file and the rules that every file under tests/data/ keeps are tested in
tests/repo/test_frozen_files.py.
"""

import math
import platform
import re
import sys
from importlib import metadata

import cv2
import numpy as np
import pytest
from frozen_helpers import (COMMAND, GOLDEN, NEEDED, SIDECAR, frozen_text, header_lines, machine_name, read_frozen,
                            same_decoder)
from frozen_numbers_helpers import (COLUMNS, POSITIONS, WEIGHTS, compare_with_frozen, machine_here, machine_of,
                                    position_rows, read_positions, read_weights, same_machine,
                                    why_the_limit_was_not_asserted)

from outline_tracker.video import reader as video

FRAMES = list(range(0, 40, 2))  # the tracked frames of both clips: 0, 2, ..., 38
# Where the clips' recipes draw each object: (center on frame 0, step per frame), in array coordinates
# (pixel centers at whole numbers), px. The selftest clip: outline_tracker/synthetic/clips.py, `selftest_clip`.
# The three ellipses: `starts` and `steps` of `write_three_ellipse_clip` in
# tests/support/pipeline_helpers.py.
RECIPES = {
    ("selftest", "selftest"): ((700.0, 500.0), (0.6, 0.2)),
    ("three_ellipses", "A"): ((500.0, 300.0), (0.6, 0.2)),
    ("three_ellipses", "B"): ((1000.0, 600.0), (-0.5, 0.3)),
    ("three_ellipses", "C"): ((1400.0, 350.0), (0.2, -0.6)),
}
TRUTH_LIMIT_PX = 3.0  # the selftest criterion of SPEC 13.4: every found center under 3 px from the true one


def true_center(clip: str, track_id: str, frame: int) -> tuple[float, float]:
    """(u_px, v_px) of an object's true center on a frame, in Tracker's convention: where the recipe
    draws it, plus 0.5."""
    (x0, y0), (dx, dy) = RECIPES[clip, track_id]
    return x0 + dx * frame + 0.5, y0 + dy * frame + 0.5


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


def test_the_reason_names_each_fact_in_which_this_machine_is_not_the_headers():
    # a made-up header, and machines as `machine_here` would describe them
    header = {"machine": "Apple M1 Max", "decoder": "opencv-5.0.0/darwin/arm64",
              "libraries": "torch 2.14.1; torchvision 0.29.1; numpy 2.5.3", "weights sha256": "ab" * 32}
    here = {"platform": "darwin", "architecture": "arm64", "chip": "Apple M1 Max", "torch": "2.14.1",
            "weights sha256": "ab" * 32}
    # the machine that froze the numbers: the limit was asserted, and there is no reason to give
    assert same_machine(header, here) is True
    assert why_the_limit_was_not_asserted(header, here) == ""

    # another machine, one fact at a time: the whole reason for the first, typed by hand
    assert why_the_limit_was_not_asserted(header, {**here, "chip": "Apple M3 Pro"}) == (
        "The limit of 0.01 px was not asserted: this is not the machine that froze the numbers (chip: Apple M3 Pro "
        "here, Apple M1 Max in the header). The position tests asserted instead that no row is lost and that every "
        "position is under 3 px from its true center, and printed the distance from the frozen positions (shown "
        "with -rP).")
    others = {
        "chip": ("Apple M3 Pro", "chip: Apple M3 Pro here, Apple M1 Max in the header"),
        "torch": ("2.15.0", "torch: 2.15.0 here, 2.14.1 in the header"),
        "weights sha256": ("cd" * 32, f"hash of the weights: {'cd' * 32} here, {'ab' * 32} in the header"),
        "platform": ("linux", "operating system: linux here, darwin in the header"),
        "architecture": ("x86_64", "architecture: x86_64 here, arm64 in the header"),
    }
    for key, (value, named) in others.items():
        reason = why_the_limit_was_not_asserted(header, {**here, key: value})
        assert f"({named})" in reason, key  # this fact and no other
        assert reason.startswith("The limit of 0.01 px was not asserted: ") and "under 3 px" in reason, key
        assert same_machine(header, {**here, key: value}) is False, key

    # a test machine of CI: every fact differs, and each is named, in the order of `machine_here`
    ci = {"platform": "linux", "architecture": "x86_64", "chip": "AMD EPYC 7763 64-Core Processor",
          "torch": "2.14.1+cpu", "weights sha256": "cd" * 32}
    reason = why_the_limit_was_not_asserted(header, ci)
    assert "(operating system: linux here, darwin in the header; architecture: x86_64 here, arm64 in the header; " in reason
    assert "; chip: AMD EPYC 7763 64-Core Processor here, Apple M1 Max in the header; " in reason
    assert f"; torch: 2.14.1+cpu here, 2.14.1 in the header; hash of the weights: {'cd' * 32} here, " in reason

    # a fact that the header does not hold, and one that this machine could not read: named as such, never passed over
    no_hash = {key: value for key, value in header.items() if key != "weights sha256"}
    assert f"(hash of the weights: {'ab' * 32} here, not named in the header)" in why_the_limit_was_not_asserted(no_hash, here)
    assert "(chip: not known here, Apple M1 Max in the header)" in why_the_limit_was_not_asserted(header, {**here, "chip": ""})
    # asked without the hash, as the rule may be: the hash is no reason
    without = {key: value for key, value in here.items() if key != "weights sha256"}
    assert why_the_limit_was_not_asserted({**header, "weights sha256": "cd" * 32}, without) == ""

    # a reason is given exactly where the rule says "another machine"
    for machine in (here, ci, without, {**here, "chip": ""}, *({**here, key: value} for key, (value, _) in others.items())):
        for made in (header, no_hash):
            assert (why_the_limit_was_not_asserted(made, machine) == "") is same_machine(made, machine)


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


def frozen_torch(path) -> str:
    """The torch version that the header of a frozen file names, as the machine rule reads it."""
    return machine_of(read_frozen(path)[0])["torch"]


def frozen_opencv() -> str:
    """OpenCV's version in the header of the golden files: the first part of its `decoder` line
    (`opencv-5.0.0/darwin/arm64`: 5.0.0), the line that the decoder rule compares."""
    return read_frozen(GOLDEN / SIDECAR)[0]["decoder"].partition("/")[0].removeprefix("opencv-")


def test_the_installed_torch_and_opencv_are_those_of_the_frozen_files_headers():
    # The strict checks are made only where the libraries are those of the headers: the limit of 0.01 px
    # and the weights' hash with the header's torch version (`same_machine`), the golden bytes with the
    # header's OpenCV version (`same_decoder`). What a build adds is no other version: a local part of
    # torch's (`2.14.1+cpu`, CI's on Linux), and parts of OpenCV's beyond those that the header holds.
    torch_here = metadata.version("torch").partition("+")[0]  # read without importing torch
    ended = [f"- {path.name}, {check}: frozen with torch {frozen_torch(path)}, installed is torch {torch_here}"
             for path, check in ((POSITIONS, "the limit of 0.01 px for the positions of the real model"),
                                 (WEIGHTS, "the hash of the weights"))
             if frozen_torch(path) != torch_here]
    opencv_frozen = frozen_opencv()
    if cv2.__version__.split(".")[:opencv_frozen.count(".") + 1] != opencv_frozen.split("."):
        ended.append(f"- {GOLDEN.name}/{SIDECAR}, the bytes of the golden Tracker-format files: frozen with OpenCV "
                     f"{opencv_frozen}, installed is OpenCV {cv2.__version__}")
    if ended:
        pytest.fail("\n".join([
            "With the installed versions these strict checks are asserted on no computer, and every run would stay "
            "green without them:", *ended,
            "Do not edit the header of a frozen file: it says how the file was made. The owner decides what "
            "happens (docs/ROADMAP.md, section 3: the decision on a change of the torch version or of the "
            "reference machine)."]))


def test_with_another_torch_or_opencv_version_that_test_fails_and_names_the_checks_that_end(monkeypatch):
    check = test_the_installed_torch_and_opencv_are_those_of_the_frozen_files_headers
    version_of = metadata.version

    def install(torch_version, opencv_version):
        """Make this version of torch the installed one, and this one what OpenCV says of itself."""
        monkeypatch.setattr(metadata, "version", lambda name: torch_version if name == "torch" else version_of(name))
        monkeypatch.setattr(cv2, "__version__", opencv_version)

    torch_frozen, opencv_frozen = frozen_torch(POSITIONS), frozen_opencv()
    assert torch_frozen == frozen_torch(WEIGHTS)  # one run froze both files
    assert re.fullmatch(r"\d+\.\d+\.\d+", torch_frozen) and re.fullmatch(r"\d+\.\d+\.\d+", opencv_frozen)

    install(torch_frozen, opencv_frozen)
    check()
    install(torch_frozen + "+cpu", opencv_frozen + ".93")  # what a build adds
    check()

    # another torch version, also one that begins like the header's (2.14.10 for 2.14.1): the two checks of
    # the real model end, and the golden bytes do not
    for other in ("0.0.1", torch_frozen + "0", torch_frozen.rpartition(".")[0]):
        install(other, opencv_frozen)
        with pytest.raises(pytest.fail.Exception) as failed:
            check()
        message = str(failed.value)
        assert "asserted on no computer" in message and "Do not edit the header" in message, other
        assert "The owner decides" in message and "docs/ROADMAP.md, section 3" in message, other
        assert [line.partition(",")[0] for line in message.splitlines() if line.startswith("- ")] == [
            "- edgetam_cpu_positions.csv", "- edgetam_weights.txt"], other
        assert f"frozen with torch {torch_frozen}, installed is torch {other}" in message, other
        assert "the limit of 0.01 px" in message and "the hash of the weights" in message, other

    # another OpenCV version: the golden bytes end, and the checks of the real model do not
    for other in ("0.0.1", opencv_frozen + "0", opencv_frozen.rpartition(".")[0]):
        install(torch_frozen, other)
        with pytest.raises(pytest.fail.Exception) as failed:
            check()
        message = str(failed.value)
        assert [line.partition(",")[0] for line in message.splitlines() if line.startswith("- ")] == [
            "- tracker_format/HEADER.txt"], other
        assert "the bytes of the golden Tracker-format files" in message, other
        assert f"frozen with OpenCV {opencv_frozen}, installed is OpenCV {other}" in message, other

    install("0.0.1", "0.0.1")  # both: all three
    with pytest.raises(pytest.fail.Exception) as failed:
        check()
    assert len([line for line in str(failed.value).splitlines() if line.startswith("- ")]) == 3
