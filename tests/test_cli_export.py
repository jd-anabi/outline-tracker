"""`outline-tracker export SESSION.json [--overlay]`: every output again from the session and results.npz
(SPEC 8, 11; review focus 1 and 5).

The run folders are tracked by `tracking.run_job` with `ExactFake` on the synthetic dish clip
(tests/export_helpers.py: a stick of 300 px = 30 mm, so K = 0.1 mm per px, origin (100.5, 60.25) px, axis
angle 0), so every expected number comes from the scene: the true centroids of the masks and the stick. The
files themselves are tested in tests/test_export*.py; here it is what the command prints and returns, what it
refuses, and what it leaves alone. Errors are one `ERROR: ...` line on stderr and exit code 1, and nothing is
written; warnings and notes go to stdout, after the list of files, and the exit code stays 0.

Coordinates: px in Tracker's convention (pixel centers at +0.5), world mm with y up. Sizes are printed in
SI units (1 kB = 1000 bytes).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cli_export_helpers import edit_session, export, listing, one_error, within_rounding, written
from cli_probe_helpers import decode
from export_helpers import K, MODEL, coarse_run, load, table, world
from helpers import ODD_FOLDER
from tracking_helpers import DISH_BOX, box_truth

from outline_tracker import cli

REPO = Path(__file__).resolve().parents[1]
GRID = list(range(0, 120, 2))  # the dish clip has 120 frames; the sessions take every 2nd
FILES = {"positions.csv", "shapes.csv", "radial.csv", "outlines.npz", "README.txt", "run.log",
         f"{MODEL}/A.csv", f"{MODEL}/B.csv", f"{MODEL}/C.csv"}  # what an export lists for the tracks A, B, C
HEAVY = ("torch", "transformers", "PySide6", "pyqtgraph")
DEFERRED = (*HEAVY, "numpy", "cv2", "pandas", "scipy", "skimage", "imageio_ffmpeg")


@pytest.fixture(scope="module")
def tracked(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse inside the dish crop; nothing exported yet."""
    folder = tmp_path_factory.mktemp("cli_export") / "run"
    coarse_run(dish_clip, folder, ["A", "B", "C"])
    return folder


@pytest.fixture
def run_folder(tracked, tmp_path):
    """This test's own copy of the tracked run folder (its video is found through the absolute path)."""
    return shutil.copytree(tracked, tmp_path / "run")


@pytest.fixture
def run_without_video(clip_in_odd_folder):
    """B tracked in a run folder inside a folder named with a space and non-ASCII characters, next to a
    video that is then deleted: the session names a video that is no longer anywhere."""
    folder = clip_in_odd_folder.path.parent / "run"
    coarse_run(clip_in_odd_folder, folder, ["B"])
    clip_in_odd_folder.path.unlink()
    return folder


def lines_of(capsys) -> list[str]:
    """The lines the command printed on stdout since the last call."""
    return capsys.readouterr().out.splitlines()


def after_the_first_line(out: str) -> str:
    """The output without its first line, which names the run folder: pytest names that folder after the
    test, so a word searched for must not be looked for there."""
    return out.split("\n", 1)[1]


def after_the_list(lines, line) -> bool:
    """Whether `line` comes after the last `wrote ...` line: notes and warnings close the output."""
    return lines.index(line) > max(i for i, text in enumerate(lines) if text.startswith("wrote "))


# ---------------------------------------------------------------------------------------------
# The command line


def test_help_shows_the_session_file_and_the_option_and_the_command_is_listed(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["export", "--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "SESSION.json" in shown and "--overlay" in shown
    assert cli.main([]) == 0
    assert any(line.split()[:1] == ["export"] for line in capsys.readouterr().out.splitlines())  # in the list


def test_export_needs_a_session_file(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["export"])
    assert stopped.value.code == 2
    assert "SESSION.json" in capsys.readouterr().err


def test_building_the_parser_loads_nothing_heavy():
    # cli.py imports cli_export for every command: numpy, OpenCV and the rest come only when `export` runs
    code = ("import sys\n"
            "from outline_tracker import cli\n"
            "args = cli.build_parser().parse_args(['export', 'x.json', '--overlay'])\n"
            "print(args.session, args.overlay, sorted(m for m in sys.argv[1:] if m in sys.modules))\n")
    done = subprocess.run([sys.executable, "-c", code, *DEFERRED], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "x.json True []"


@pytest.mark.parametrize(("size", "text"), [
    (0, "0 B"), (1, "1 B"), (999, "999 B"), (1000, "1.0 kB"), (12_345, "12.3 kB"), (999_949, "999.9 kB"),
    (999_999, "1.0 MB"), (1_000_000, "1.0 MB"), (2_500_000, "2.5 MB"), (999_999_999, "1.0 GB"),
    (1_234_567_890, "1.2 GB"), (12_000_000_000, "12.0 GB"),
])
def test_sizes_are_printed_in_plain_units_with_one_decimal_from_kb_up(size, text):
    from outline_tracker import cli_export

    assert cli_export.size_text(size) == text  # 1 kB = 1000 bytes, as a file manager shows it


# ---------------------------------------------------------------------------------------------
# What it prints


def test_export_writes_every_file_and_lists_each_with_its_size(run_folder, capsys):
    assert export(run_folder / "session.json") == 0
    shown = capsys.readouterr()
    assert shown.err == ""
    assert shown.out.splitlines()[0] == f"Exporting the run folder {run_folder}"
    listed = written(shown.out)
    assert set(listed) == FILES
    for name, size in listed.items():
        assert within_rounding(size, (run_folder / name).stat().st_size), (name, size)
    rest = after_the_first_line(shown.out)
    assert "WARNING" not in rest and "partial" not in rest and "session.new.json" not in rest
    assert shown.out.replace(str(run_folder), "").isascii()  # ASCII apart from file names
    assert {path.name for path in run_folder.iterdir()} == {"session.json", "results.npz", "positions.csv",
                                                             "shapes.csv", "radial.csv", "outlines.npz", "run.log",
                                                             "README.txt", MODEL}


def test_a_relative_session_path_names_the_run_folder_in_full(run_folder, monkeypatch, capsys):
    monkeypatch.chdir(run_folder)
    assert export("session.json") == 0
    assert lines_of(capsys)[0] == f"Exporting the run folder {run_folder}"
    assert (run_folder / "positions.csv").is_file()


def test_a_second_export_lists_the_same_files_and_writes_the_same_bytes(run_folder, capsys):
    assert export(run_folder / "session.json") == 0
    names = ("positions.csv", "shapes.csv", "radial.csv", f"{MODEL}/A.csv", "README.txt")
    first = {name: (run_folder / name).read_bytes() for name in names}
    capsys.readouterr()
    assert export(run_folder / "session.json") == 0
    out = capsys.readouterr().out
    assert set(written(out)) == FILES and "WARNING" not in after_the_first_line(out)
    assert {name: (run_folder / name).read_bytes() for name in names} == first


def test_a_leftover_in_the_tracker_folder_is_removed_and_the_removal_is_shown(run_folder, capsys):
    (run_folder / MODEL).mkdir()
    (run_folder / MODEL / "A.csv.new").write_text("an export while A.csv was open elsewhere")
    assert export(run_folder / "session.json") == 0
    lines = lines_of(capsys)
    assert any("A.csv.new" in line and "removed" in line for line in lines), lines
    assert not (run_folder / MODEL / "A.csv.new").exists()


# ---------------------------------------------------------------------------------------------
# A new calibration needs no tracking, no torch and no Qt (SPEC 13.2)


def test_after_editing_the_stick_length_the_command_rescales_x_and_y_and_loads_neither_torch_nor_qt(
        run_folder, dish_clip):
    session_file = run_folder / "session.json"
    assert export(session_file) == 0  # in this process: the files made with the stick of 30 mm
    old = table(run_folder / "positions.csv")
    edit_session(run_folder, lambda data: data["calibration"]["stick"].update(length_mm=29.0))
    # in a process of its own: pytest-qt has already loaded PySide6 into this one
    code = ("import sys\n"
            "from outline_tracker import cli\n"
            "status = cli.main(sys.argv[1:])\n"
            f"print('HEAVY', sorted(m for m in {HEAVY!r} if m in sys.modules))\n"
            "raise SystemExit(status)\n")
    done = subprocess.run([sys.executable, "-c", code, "export", str(session_file)], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=180)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "HEAVY []"
    new = table(run_folder / "positions.csv")

    ratio = 29.0 / 30.0
    for column in ("x_mm", "y_mm"):  # two cells rounded to 6 decimals: within 1e-6 mm
        np.testing.assert_allclose(new[column], ratio * old[column], rtol=0, atol=1e-6, err_msg=column)
    np.testing.assert_allclose(new.area_mm2, ratio ** 2 * old.area_mm2, rtol=0, atol=1e-6)
    for column in ("track_id", "frame", "t_s", "u_px", "v_px"):
        assert list(new[column]) == list(old[column]), column
    for track_id in "ABC":  # and the numbers themselves: the true centroids, with the stick of 29 mm
        u, v, _, _ = box_truth(dish_clip, track_id, GRID, DISH_BOX)
        x, y = world(u, v)  # in the sessions of export_helpers, with the stick of 30 mm
        mine = new[new.track_id == track_id]
        # 0.01 px is SPEC 13.2's tolerance with ExactFake; the cells are rounded to 6 decimals
        np.testing.assert_allclose(mine.x_mm, ratio * x, rtol=0, atol=ratio * K * 0.01 + 1e-6)
        np.testing.assert_allclose(mine.y_mm, ratio * y, rtol=0, atol=ratio * K * 0.01 + 1e-6)


# ---------------------------------------------------------------------------------------------
# The overlay (SPEC 8.8)


def test_with_the_option_the_overlay_is_made_from_the_video(run_folder, capsys):
    assert export(run_folder / "session.json", "--overlay") == 0
    out = capsys.readouterr().out
    listed = written(out)
    assert set(listed) == FILES | {"overlay.mp4"}
    assert within_rounding(listed["overlay.mp4"], (run_folder / "overlay.mp4").stat().st_size)
    assert len(decode(run_folder / "overlay.mp4")) == len(GRID)  # one overlay frame per tracked frame
    rest = after_the_first_line(out)
    assert "not regenerated" not in rest and "not made" not in rest and "skipped" not in rest


def test_without_the_option_an_existing_overlay_is_left_alone_and_the_command_says_so(run_folder, capsys):
    overlay = run_folder / "overlay.mp4"
    overlay.write_bytes(b"the overlay of an earlier export")
    assert export(run_folder / "session.json") == 0
    out = capsys.readouterr().out
    assert overlay.read_bytes() == b"the overlay of an earlier export"
    said = [line for line in out.splitlines() if "overlay.mp4 was not regenerated" in line]
    assert len(said) == 1 and said[0].startswith("NOTE: ") and "--overlay" in said[0]
    rest = after_the_first_line(out)
    assert "overlay.mp4" not in written(out) and "not made" not in rest and "skipped" not in rest


def test_without_the_option_and_without_an_overlay_the_command_says_none_was_made(run_folder, capsys):
    assert export(run_folder / "session.json") == 0
    said = [line for line in lines_of(capsys) if "overlay.mp4 was not made" in line]
    assert len(said) == 1 and said[0].startswith("NOTE: ") and "--overlay" in said[0]
    assert not (run_folder / "overlay.mp4").exists()


def test_without_the_video_every_file_is_still_written_and_the_overlay_is_said_to_be_skipped(run_without_video,
                                                                                            capsys):
    assert export(run_without_video / "session.json", "--overlay") == 0
    out = capsys.readouterr().out
    assert set(written(out)) == {"positions.csv", "shapes.csv", "radial.csv", "outlines.npz", "README.txt",
                                 "run.log", f"{MODEL}/B.csv"}
    skipped = [line for line in out.splitlines() if line.startswith("WARNING: ") and "overlay.mp4 was skipped" in line]
    assert len(skipped) == 1 and "dish_tracker.mp4" in skipped[0] and "not found" in skipped[0]
    assert not (run_without_video / "overlay.mp4").exists()
    rows = table(run_without_video / "positions.csv")
    assert set(rows.track_id) == {"B"} and list(rows.frame) == GRID
    assert out.replace(ODD_FOLDER, "").isascii()  # ASCII apart from file names: here the folder's name


def test_without_the_video_and_without_the_option_nothing_is_said_to_be_skipped(run_without_video, capsys):
    assert export(run_without_video / "session.json") == 0
    out = capsys.readouterr().out
    assert len(written(out)) == 7 and "skipped" not in after_the_first_line(out) and "WARNING" not in out
    assert any("overlay.mp4 was not made" in line for line in out.splitlines())


# ---------------------------------------------------------------------------------------------
# Notes and warnings: printed after the list of files, and the exit code stays 0


def test_a_partial_session_exports_the_frames_it_has_and_says_it_is_partial(run_folder, capsys):
    _, store = load(run_folder)
    store.replace_from("B", 40)  # as after a run that was stopped: B has frames 0 to 38
    store.save(run_folder / "results.npz")
    edit_session(run_folder, lambda data: data.update(complete=False))
    assert export(run_folder / "session.json") == 0
    lines = lines_of(capsys)
    said = [line for line in lines[1:] if "partial" in line]
    assert len(said) == 1 and said[0].startswith("NOTE: ") and after_the_list(lines, said[0])
    rows = table(run_folder / "positions.csv")
    assert list(rows[rows.track_id == "B"].frame) == list(range(0, 40, 2))
    assert list(rows[rows.track_id == "A"].frame) == GRID and list(rows[rows.track_id == "C"].frame) == GRID
    assert json.loads((run_folder / "session.json").read_text(encoding="utf-8"))["complete"] is False


def test_a_session_new_json_beside_the_session_is_named_with_the_session_and_not_used(run_folder, capsys, dish_clip):
    newer = run_folder / "session.new.json"
    data = json.loads((run_folder / "session.json").read_text(encoding="utf-8"))
    data["calibration"]["stick"]["length_mm"] = 29.0  # what a later save wrote while session.json was locked
    newer.write_text(json.dumps(data), encoding="utf-8")
    before = newer.read_bytes()
    assert export(run_folder / "session.json") == 0
    lines = lines_of(capsys)
    said = [line for line in lines[1:] if "session.new.json" in line]
    assert len(said) == 1 and said[0].startswith("NOTE: ") and after_the_list(lines, said[0])
    assert "session.json" in said[0].replace("session.new.json", "")  # both files are named
    assert newer.read_bytes() == before
    rows = table(run_folder / "positions.csv")  # made with session.json: the stick of 30 mm
    u, v, _, _ = box_truth(dish_clip, "A", GRID, DISH_BOX)
    np.testing.assert_allclose(rows[rows.track_id == "A"].x_mm, world(u, v)[0], rtol=0, atol=K * 0.01 + 1e-6)


def test_a_locked_file_gets_its_new_data_next_to_it_and_one_warning_after_the_list(run_folder, lock_file, capsys):
    assert export(run_folder / "session.json") == 0
    capsys.readouterr()
    old = (run_folder / "positions.csv").read_bytes()
    edit_session(run_folder, lambda data: data["calibration"]["stick"].update(length_mm=29.0))
    lock_file("positions.csv", waits="none")  # another program holds it open; one try, then the fallback
    assert export(run_folder / "session.json") == 0
    lines = lines_of(capsys)
    assert (run_folder / "positions.csv").read_bytes() == old
    listed = written("\n".join(lines))
    assert "positions.new.csv" in listed and "positions.csv" not in listed
    assert within_rounding(listed["positions.new.csv"], (run_folder / "positions.new.csv").stat().st_size)
    warnings = [line for line in lines if line.startswith("WARNING: ")]
    assert len(warnings) == 1 and "positions.csv" in warnings[0] and "positions.new.csv" in warnings[0]
    assert after_the_list(lines, warnings[0])


def test_a_track_left_out_is_reported_once(run_folder, capsys):
    edit_session(run_folder, lambda data: data.update(tracks=[track for track in data["tracks"] if track["id"] != "C"]))
    assert export(run_folder / "session.json") == 0
    said = [line for line in lines_of(capsys) if "does not list" in line]
    assert len(said) == 1 and said[0].startswith("WARNING: ") and "'C'" in said[0]
    assert set(table(run_folder / "positions.csv").track_id) == {"A", "B"}


# ---------------------------------------------------------------------------------------------
# What is refused: one ERROR line on stderr, exit code 1, nothing written


@pytest.mark.parametrize(("version", "written_by"), [(2, "a newer outline-tracker"), (0, "an older outline-tracker")])
def test_a_session_of_another_schema_version_gives_the_plain_message(run_folder, capsys, version, written_by):
    edit_session(run_folder, lambda data: data.update(schema_version=version))
    before = listing(run_folder)
    assert export(run_folder / "session.json") == 1
    shown = capsys.readouterr()
    assert shown.out == ""
    assert shown.err.startswith("ERROR: ") and shown.err.endswith("\n") and shown.err.count("\n") == 1, shown.err
    assert f"session format version {version}" in shown.err and "reads only version 1" in shown.err
    assert f"written by {written_by}" in shown.err
    assert listing(run_folder) == before


def test_a_missing_file_gives_exit_code_1(tmp_path, capsys):
    assert export(tmp_path / "nowhere" / "session.json") == 1
    error = one_error(capsys)
    assert "No such file" in error and "nowhere" in error
    assert listing(tmp_path) == []


def test_a_folder_instead_of_the_session_file_is_refused_and_named(run_folder, capsys):
    before = listing(run_folder)
    assert export(run_folder) == 1
    error = one_error(capsys)
    assert "folder" in error and "session.json" in error
    assert listing(run_folder) == before


def test_a_file_with_another_name_is_refused_because_export_reads_session_json(run_folder, capsys):
    other = run_folder / "ana copy.json"
    shutil.copyfile(run_folder / "session.json", other)
    before = listing(run_folder)
    assert export(other) == 1
    error = one_error(capsys)
    assert "ana copy.json" in error and "session.json" in error
    assert listing(run_folder) == before


def test_a_file_that_is_not_a_session_is_refused(tmp_path, capsys):
    (tmp_path / "session.json").write_text("this is not JSON\n")
    assert export(tmp_path / "session.json") == 1
    assert "is not a session file" in one_error(capsys)
    assert listing(tmp_path) == ["session.json"]


def test_a_folder_without_results_is_refused_with_a_plain_message(run_folder, capsys):
    (run_folder / "results.npz").unlink()
    assert export(run_folder / "session.json") == 1
    assert "results.npz" in one_error(capsys)
    assert not (run_folder / "positions.csv").exists()


def test_a_radial_step_other_than_5_degrees_is_refused_and_nothing_is_written(run_folder, capsys):
    edit_session(run_folder, lambda data: data["processing"].update(radial_step_deg=10))
    before = listing(run_folder)
    assert export(run_folder / "session.json") == 1
    error = one_error(capsys)
    assert "radial_step_deg" in error and "10" in error and "5" in error
    assert listing(run_folder) == before
