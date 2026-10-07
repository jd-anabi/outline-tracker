"""Tests for the hidden `compare-tracks` command (decision X10): what it prints and its exit code.

The folders, the positions and where the expected values come from are described in
tests/cli_compare_helpers.py: straight tracks written with the ported `write_tracker_file`, shifted by a
known offset ((0.18, 0.24) px is 0.3 px) or moved on one frame (RMS = d / sqrt(n)). Differences are px
in Tracker's image coordinates (SPEC 3.1); frames are the video's own frame numbers. The table of
`tracker_io.compare_tracks` itself is tested in tests/test_cli_compare_table.py.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cli_compare_helpers import (B_START, FRAMES, OK_AT_0_3, SHIFT, cells, compare, folders, jump_on_136, line_of,
                                 no_ok, path_px, printed, write_track)

from outline_tracker import cli

REPO = Path(__file__).resolve().parents[1]
HEADER = ["track", "old_track", "common", "first", "last", "rms_px", "max_px", "at_frame"]


# --------------------------------------------------------------------------- pairs by name

def test_a_known_offset_of_0_3_px_is_reported_as_rms_0_300(tmp_path, capsys):
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    bx, by = path_px(FRAMES, B_START)
    write_track(old, "A", FRAMES, ax, ay)
    write_track(new, "A", FRAMES, ax, ay, shift=SHIFT)
    write_track(old, "B", FRAMES, bx, by)
    write_track(new, "B", FRAMES, bx, by, shift=(0.0, -0.1))
    assert compare(new, old) == 0
    lines = printed(capsys)
    assert lines[0].split() == HEADER
    assert cells(lines, "A") == ["A", "A", "25", "100", "148", "0.300", "0.300"]
    assert cells(lines, "B") == ["B", "B", "25", "100", "148", "0.100", "0.100"]
    assert int(line_of(lines, "A").split()[7]) in FRAMES  # the same difference on every frame
    assert lines[-1] == OK_AT_0_3
    assert len(lines) == 4  # the header, one line per pair, the verdict
    assert len({len(line) for line in lines[:3]}) == 1  # columns of fixed width


def test_a_pair_above_the_limit_exits_1_and_names_the_frame_of_the_largest_difference(tmp_path, capsys):
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    bx, by = path_px(FRAMES, B_START)
    write_track(old, "A", FRAMES, ax, ay)
    write_track(new, "A", FRAMES, ax, ay)
    write_track(old, "B", FRAMES, bx, by)
    write_track(new, "B", FRAMES, *jump_on_136(bx, by))
    assert compare(new, old) == 1
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "A", "25", "100", "148", "0.000", "0.000"]
    assert line_of(lines, "B").split() == ["B", "B", "25", "100", "148", "10.000", "50.000", "136"]  # 50 / sqrt(25)
    assert lines[-1].startswith("CHECK: B: ") and "frame 136" in lines[-1]
    assert no_ok(lines)


def test_limit_0_01_passes_for_identical_folders(tmp_path, capsys):
    new, old = folders(tmp_path)
    for name, start in (("A", (100.5, 80.5)), ("B", B_START)):
        for folder in (new, old):
            write_track(folder, name, FRAMES, *path_px(FRAMES, start))
    assert compare(new, old, "--limit", "0.01") == 0
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "A", "25", "100", "148", "0.000", "0.000"]
    assert cells(lines, "B") == ["B", "B", "25", "100", "148", "0.000", "0.000"]
    assert lines[-1] == "OK: worst RMS 0.000 px (limit 0.01 px)"


@pytest.mark.parametrize("limit, status", [("0.31", 0), ("0.29", 1), ("nan", 1)])
def test_an_rms_above_the_limit_fails_and_one_below_it_passes(limit, status, tmp_path, capsys):
    new, old = folders(tmp_path)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    assert compare(new, old, "--limit", limit) == status
    lines = printed(capsys)
    if status == 0:
        assert lines[-1] == "OK: worst RMS 0.300 px (limit 0.31 px)"
    else:
        assert lines[-1].startswith("CHECK: A: ") and no_ok(lines)


def test_only_the_frames_both_tracks_have_are_compared(tmp_path, capsys):
    # The new track starts at frame 106 and was lost on 110 and 112 (rows with empty cells, which the
    # reader drops); the old one ends at 140. Both have 106, 108, 114, 116, ..., 140: 16 frames.
    new, old = folders(tmp_path)
    new_frames, old_frames = FRAMES[FRAMES >= 106], FRAMES[FRAMES <= 140]
    nx, ny = path_px(new_frames)
    lost = np.isin(new_frames, (110, 112))
    nx[lost] = ny[lost] = np.nan
    write_track(new, "A", new_frames, nx, ny, shift=SHIFT)
    write_track(old, "A", old_frames, *path_px(old_frames))
    assert compare(new, old) == 0
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "A", "16", "106", "140", "0.300", "0.300"]
    assert lines[-1] == OK_AT_0_3


def test_odd_frames_against_even_frames_have_no_common_frames(tmp_path, capsys):
    new, old = folders(tmp_path)
    even, odd = np.arange(0, 50, 2), np.arange(1, 50, 2)  # 0 to 48, 1 to 49
    write_track(old, "A", even, *path_px(even))
    write_track(new, "A", odd, *path_px(odd))
    write_track(old, "B", even, *path_px(even, B_START))  # this pair agrees on every frame
    write_track(new, "B", even, *path_px(even, B_START))
    assert compare(new, old) == 1
    lines = printed(capsys)
    line = line_of(lines, "A")
    assert "no common frames" in line and "1 to 49" in line and "0 to 48" in line  # both frame ranges
    assert cells(lines, "B") == ["B", "B", "25", "0", "48", "0.000", "0.000"]
    assert lines[-1].startswith("CHECK: A: ") and "no common frames" in lines[-1]
    assert no_ok(lines)


@pytest.mark.parametrize("option, reason", [((), "no common frames: new none, old 100 to 148"),
                                            (("--by-position",), "no partner")])
def test_a_new_track_that_was_lost_on_every_frame_has_nothing_to_compare(option, reason, tmp_path, capsys):
    # an object that was never found (review focus 4): every row has empty cells, so the reader keeps none
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, np.full(len(FRAMES), np.nan), np.full(len(FRAMES), np.nan))
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    assert compare(new, old, *option) == 1
    lines = printed(capsys)
    assert reason in line_of(lines, "A")
    assert lines[-1].startswith("CHECK: A: ") and "nothing was compared" in lines[-1]
    assert no_ok(lines)


@pytest.mark.parametrize("old_tracks", [("A",), ()])
def test_an_empty_new_folder_exits_1_and_prints_no_ok(old_tracks, tmp_path, capsys):
    new, old = folders(tmp_path)
    new.mkdir(parents=True)
    old.mkdir(parents=True)
    for name in old_tracks:
        write_track(old, name, FRAMES, *path_px(FRAMES))
    assert compare(new, old) == 1
    lines = printed(capsys)
    assert lines[-1].startswith("CHECK:") and "nothing was compared" in lines[-1]
    assert no_ok(lines)


def test_a_new_track_without_a_partner_exits_1(tmp_path, capsys):
    new, old = folders(tmp_path)
    for name, start in (("A", (100.5, 80.5)), ("C", B_START)):
        write_track(new, name, FRAMES, *path_px(FRAMES, start))
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    assert compare(new, old) == 1
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "A", "25", "100", "148", "0.000", "0.000"]
    assert "no partner" in line_of(lines, "C")
    assert lines[-1].startswith("CHECK: C: ")
    assert no_ok(lines)


def test_an_old_track_without_a_new_one_is_named_and_is_not_an_error(tmp_path, capsys):
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    write_track(old, "C", FRAMES, *path_px(FRAMES, B_START))
    assert compare(new, old) == 0
    lines = printed(capsys)
    assert lines[-2:] == ["note: old tracks paired with no new track: C", OK_AT_0_3]


# --------------------------------------------------------------------------- pairs by position

def test_by_position_pairs_the_same_tracks_under_other_file_names(tmp_path, capsys):
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    bx, by = path_px(FRAMES, B_START)
    write_track(new, "A", FRAMES, ax, ay, shift=SHIFT)
    write_track(new, "B", FRAMES, bx, by, shift=(0.3, 0.4))  # 0.5 px away
    write_track(old, "shrimp2", FRAMES, ax, ay)  # sorted by name, last week's files come in the other order
    write_track(old, "shrimp1", FRAMES, bx, by)
    assert compare(new, old, "--by-position") == 0
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "shrimp2", "25", "100", "148", "0.300", "0.300"]
    assert cells(lines, "B") == ["B", "shrimp1", "25", "100", "148", "0.500", "0.500"]
    assert lines[-1] == "OK: worst RMS 0.500 px (limit 2 px)"
    assert len(lines) == 4 and len({len(line) for line in lines[:3]}) == 1

    assert compare(new, old) == 1  # by name, no file of one folder is in the other
    lines = printed(capsys)
    assert "no partner" in line_of(lines, "A") and "no partner" in line_of(lines, "B")
    assert lines[-1].startswith("CHECK: A: ") and no_ok(lines)


def test_by_position_looks_at_the_first_frame_the_two_tracks_share(tmp_path, capsys):
    # Last week's A runs from frame 100 to 148. This week it was ended at 122 and continued as A2 from
    # 124, where the object is 24 px to the right of and 12 px below its place on frame 100: more than
    # 15 px from where the old track starts, and 0.3 px from where the old track is on frame 124.
    new, old = folders(tmp_path)
    ax, ay = path_px(FRAMES)
    early, late = FRAMES <= 122, FRAMES >= 124
    write_track(old, "A", FRAMES, ax, ay)
    write_track(old, "B", FRAMES, *path_px(FRAMES, B_START))
    write_track(new, "A", FRAMES[early], ax[early], ay[early])
    write_track(new, "A2", FRAMES[late], ax[late], ay[late], shift=SHIFT)
    assert compare(new, old, "--by-position") == 0
    lines = printed(capsys)
    assert cells(lines, "A") == ["A", "A", "12", "100", "122", "0.000", "0.000"]
    assert cells(lines, "A2") == ["A2", "A", "13", "124", "148", "0.300", "0.300"]
    assert lines[-1] == OK_AT_0_3


@pytest.mark.parametrize("shift, paired", [((8.4, 11.2), True), ((9.6, 12.8), False)])  # 14 px and 16 px away
def test_by_position_pairs_only_tracks_at_most_15_px_apart(shift, paired, tmp_path, capsys):
    new, old = folders(tmp_path)
    write_track(old, "last_week", FRAMES, *path_px(FRAMES))
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=shift)
    assert compare(new, old, "--by-position", "--limit", "100") == (0 if paired else 1)
    lines = printed(capsys)
    if paired:
        assert cells(lines, "A") == ["A", "last_week", "25", "100", "148", "14.000", "14.000"]
        assert lines[-1] == "OK: worst RMS 14.000 px (limit 100 px)"
    else:
        assert "no partner" in line_of(lines, "A")
        assert lines[-1].startswith("CHECK: A: ") and no_ok(lines)


def test_a_file_name_that_is_not_ascii_is_printed_in_ascii(tmp_path, capsys):
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    write_track(old, "camarón", FRAMES, *path_px(FRAMES))
    assert compare(new, old, "--by-position") == 0
    partner = line_of(printed(capsys), "A").split()[1]  # `printed` asserts plain ASCII
    assert partner.startswith("camar") and partner.endswith("n") and "\\" in partner


# --------------------------------------------------------------------------- folders and files

def test_folders_named_with_spaces_and_non_ascii_characters(tmp_path, capsys):
    new, old = tmp_path / "vidéo test ü" / "edgetam", tmp_path / "la semaine dernière" / "edgetam"
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    assert compare(new, old) == 0
    assert printed(capsys)[-1] == OK_AT_0_3


def test_hidden_files_are_not_tracks(tmp_path, capsys):
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    (new / "._A.csv").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X")  # macOS writes these on USB sticks
    assert compare(new, old) == 0
    lines = printed(capsys)
    assert len(lines) == 3 and lines[-1] == OK_AT_0_3


def test_a_missing_folder_is_one_error_line(tmp_path, capsys):
    _, old = folders(tmp_path)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    missing = tmp_path / "nowhere" / "edgetam"
    for pair in ((missing, old), (old, missing)):
        assert compare(*pair) == 1
        shown = capsys.readouterr()
        assert shown.err == f"ERROR: No such folder: {missing}\n"
        assert shown.out == ""


def test_a_csv_that_is_not_a_tracker_file_is_one_error_line(tmp_path, capsys):
    # the run folder given instead of its edgetam folder: positions.csv is not in Tracker's format
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES))
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    (new / "positions.csv").write_text("frame,t_s,track_id,x_mm,y_mm\n100,0.4166667,A,1.0,2.0\n", encoding="utf-8")
    assert compare(new, old) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith("ERROR: positions.csv: ") and shown.err.count("\n") == 1
    assert shown.out == ""  # no table and no verdict


# --------------------------------------------------------------------------- the program

def test_compare_tracks_loads_neither_the_model_nor_qt(tmp_path):
    # in a fresh process, so that what this test session has already imported does not count
    new, old = folders(tmp_path)
    write_track(new, "A", FRAMES, *path_px(FRAMES), shift=SHIFT)
    write_track(old, "A", FRAMES, *path_px(FRAMES))
    code = ("import sys\n"
            "from outline_tracker import cli\n"
            "status = cli.main(sys.argv[1:])\n"
            "print('HEAVY', sorted(m for m in ('torch', 'transformers', 'PySide6', 'pyqtgraph') if m in sys.modules))\n"
            "raise SystemExit(status)\n")
    done = subprocess.run(
        [sys.executable, "-c", code, "compare-tracks", str(new), str(old)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-2:] == [OK_AT_0_3, "HEAVY []"]


def test_compare_tracks_is_not_shown_in_the_help(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "convert" in shown and "compare-tracks" not in shown
    assert cli.main([]) == 0
    assert "compare-tracks" not in capsys.readouterr().out


def test_its_own_help_shows_the_folders_and_the_options(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["compare-tracks", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    for word in ("NEW_DIR", "OLD_DIR", "--limit", "--by-position"):
        assert word in shown


def test_the_limit_is_2_px_and_pairs_go_by_name_unless_told_otherwise():
    args = cli.build_parser().parse_args(["compare-tracks", "new", "old"])
    assert (args.new_dir, args.old_dir, args.limit, args.by_position) == ("new", "old", 2.0, False)


def test_compare_tracks_needs_both_folders(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["compare-tracks", "only_one"])
    assert stopped.value.code == 2
    assert "OLD_DIR" in capsys.readouterr().err
