"""Shared by the tests of the hidden `compare-tracks` command (tests/test_cli_compare.py: what the command
prints and returns; tests/test_cli_compare_table.py: the table of `tracker_io.compare_tracks` behind it).

Where the expected values come from: every folder is written here with the ported `write_tracker_file`,
at positions chosen so that the differences are known by construction. The test object moves on a
straight line, 1 px to the right and 0.5 px down per frame (`path_px`), so two tracks of it agree on
every frame both have, whatever frames each one holds. The other folder holds the same line shifted by
(0.18, 0.24) px, which is 0.3 px away (a 3-4-5 triangle), or moved on one frame only: for n frames that
differ by d on one frame and by 0 on the others, RMS = d / sqrt(n).

Units: positions and differences are px in Tracker's image coordinates (SPEC 3.1: from the top-left
corner, pixelx to the right, pixely down); frames are the video's own frame numbers. The x and y
columns (mm) are written, as in every Tracker-format file, and never compared.
"""

import numpy as np

from outline_tracker import cli, tracker_io

FPS = 240.0  # only for the t column, which is not compared
FRAMES = np.arange(100, 150, 2)  # 25 frames: 100, 102, ..., 148
SHIFT = (0.18, 0.24)  # px; 0.3 px long
OK_AT_0_3 = "OK: worst RMS 0.300 px (limit 2 px)"
B_START = (200.5, 40.5)  # a second object: 100 px to the right of the first and 40 px above it, on every frame


def folders(tmp_path):
    """(NEW_DIR, OLD_DIR) of a test, each named with a space; `write_track` creates them."""
    return tmp_path / "new run" / "edgetam", tmp_path / "last week" / "edgetam"


def path_px(frames, start=(100.5, 80.5)):
    """(pixelx, pixely) of the test object on each frame, px: `start` on frame 100, then 1 px to the
    right and 0.5 px down per frame."""
    steps = np.asarray(frames, float) - 100
    return start[0] + steps, start[1] + 0.5 * steps


def jump_on_136(px, py):
    """The positions of `FRAMES` with frame 136 alone moved by (30, -40) px: 50 px away there."""
    px, py = px.copy(), py.copy()
    px[FRAMES == 136] += 30.0
    py[FRAMES == 136] -= 40.0
    return px, py


def write_track(folder, name, frames, px, py, shift=(0.0, 0.0)):
    """Write `<folder>/<name>.csv` as the tool writes a track (the ported `write_tracker_file`), at
    (px, py) + shift in px. A NaN position is a lost frame: its row has empty cells."""
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    px, py = np.asarray(px, float) + shift[0], np.asarray(py, float) + shift[1]
    tracker_io.write_tracker_file(folder / f"{name}.csv", name, frames, frames / FPS, 0.05 * px, -0.05 * py, px, py)


def compare(*args) -> int:
    """Run `outline-tracker compare-tracks ARGS...` in this process and return its exit code."""
    return cli.main(["compare-tracks", *(str(arg) for arg in args)])


def printed(capsys) -> list[str]:
    """The lines the command printed: plain ASCII, and nothing on stderr."""
    shown = capsys.readouterr()
    assert shown.err == ""
    assert shown.out.isascii()
    return shown.out.splitlines()


def line_of(lines, track) -> str:
    """The one line of the table that starts with the new track `track`."""
    found = [line for line in lines[1:] if line.split()[:1] == [track]]
    assert len(found) == 1, lines
    return found[0]


def cells(lines, track) -> list[str]:
    """The first seven cells of the table line of `track`: everything but at_frame."""
    return line_of(lines, track).split()[:7]


def no_ok(lines) -> bool:
    """True when no line the command printed holds `OK`."""
    return "OK" not in "\n".join(lines)
