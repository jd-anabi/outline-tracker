"""Export: the Tracker-format folder `<model>/<id>.csv`, locked files, and odd folder names (SPEC 8.1,
8.3, 13.2; review focus 1 and 5).

Students' own loaders read every .csv and .txt file of that folder as a track, so it may hold
nothing but `<id>.csv`: no temporary file and no fallback file there ends in .csv or .txt, and the
files of tracks that no longer exist are removed. The files are read back with last week's
unmodified `shrimp.segment.read_tracker_export` (tests/reference).

Coordinates: x, y in mm in the user's axes (y up), pixelx, pixely in px (Tracker's convention).
"""

import errno
import os
import shutil
import time

import numpy as np
import pytest
from export_helpers import MODEL, RUN_FILES, coarse_run, load, store_run, table
from overlay_helpers import record
from shrimp import segment as reference

from outline_tracker import export, export_tables, fileio, schema, tracker_io

GRID = list(range(0, 120, 2))


@pytest.fixture(scope="module")
def tracked(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse inside the dish crop; nothing exported yet."""
    folder = tmp_path_factory.mktemp("tracker_folder") / "run"
    coarse_run(dish_clip, folder, ["A", "B", "C"])
    return folder


@pytest.fixture
def run_folder(tracked, tmp_path):
    """This test's own copy of the tracked run folder."""
    return shutil.copytree(tracked, tmp_path / "run")


def silent(_line):
    """A log that keeps nothing."""


def _lock(monkeypatch, *names):
    """Make every rename onto a file of one of these names fail as Windows does while another
    program holds the file open, and do not wait between the tries."""
    real = os.replace

    def replace(src, dst, **kwargs):
        if os.path.basename(dst) in names:
            raise PermissionError(13, "The process cannot access the file", str(dst))
        return real(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(fileio, "RETRY_DELAYS_S", ())


# ---------------------------------------------------------------------------------------------
# The files (SPEC 8.3)


def test_tracker_files_parse_with_last_weeks_reader_and_agree_with_positions(run_folder):
    export.export_all(run_folder, log=silent)
    positions = table(run_folder / "positions.csv")
    for track_id in "ABC":
        path = run_folder / MODEL / f"{track_id}.csv"
        lines = path.read_text().splitlines()
        assert lines[:2] == [f",{track_id},,,,,", "t,frame,x,y,pixelx,pixely"]
        (name, read), = reference.read_tracker_export(path).items()
        mine = positions[(positions.track_id == track_id) & (positions.visible == 1)]
        assert name == track_id and list(read.frame) == list(mine.frame) == GRID
        # the same numbers written with the same decimals: the same values, not merely close ones
        for theirs, ours in (("x", "x_mm"), ("y", "y_mm"), ("pixelx", "u_px"), ("pixely", "v_px"), ("t", "t_s")):
            np.testing.assert_array_equal(read[theirs].to_numpy(), mine[ours].to_numpy(), err_msg=theirs)


def test_each_track_is_written_by_the_ported_writer_to_a_name_no_loader_reads(run_folder, monkeypatch):
    calls = []
    real = tracker_io.write_tracker_file

    def spy(path, name, *columns):
        calls.append((os.fspath(path), name))
        return real(path, name, *columns)

    monkeypatch.setattr(tracker_io, "write_tracker_file", spy)
    export.export_all(run_folder, log=silent)
    assert [name for _, name in calls] == ["A", "B", "C"]
    for path, name in calls:
        assert os.path.dirname(path) == str(run_folder / MODEL)  # renamed in place, on one drive
        assert os.path.basename(path).startswith(f"{name}.csv")
        assert os.path.splitext(path)[1].lower() not in (".csv", ".txt")
        assert not os.path.exists(path)  # the temporary file is gone
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "B.csv", "C.csv"}


def test_the_folder_is_named_after_the_sessions_model(dish_clip, tmp_path):
    coarse_run(dish_clip, tmp_path / "run", ["B"], model="stand-in")
    export.export_all(tmp_path / "run", log=silent)
    assert {path.name for path in (tmp_path / "run" / "stand-in").iterdir()} == {"B.csv"}
    assert not (tmp_path / "run" / "edgetam").exists()


def test_a_track_id_is_made_safe_for_the_file_name_as_last_week(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    ids = ["A", "tail fin/1"]  # ids of a Tracker export may be any text
    store_run(run_folder, {track_id: [record(dish_clip, "B", frame) for frame in (0, 2)] for track_id in ids})
    export.export_all(run_folder, log=silent)
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "tail_fin_1.csv"}
    assert (run_folder / MODEL / "tail_fin_1.csv").read_text().splitlines()[0] == ",tail fin/1,,,,,"
    assert list(table(run_folder / "positions.csv").track_id) == ["A", "A", "tail fin/1", "tail fin/1"]


def test_two_ids_that_give_one_file_name_are_refused_before_anything_is_written(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    ids = ["tail fin", "tail_fin"]
    store_run(run_folder, {track_id: [record(dish_clip, "B", frame) for frame in (0, 2)] for track_id in ids})
    with pytest.raises(ValueError, match="tail_fin.csv"):
        export.export_all(run_folder, log=silent)
    assert not (run_folder / "positions.csv").exists() and not (run_folder / MODEL).exists()


def test_files_of_tracks_that_are_gone_and_leftovers_of_a_crash_are_removed_and_logged(run_folder):
    export.export_all(run_folder, log=silent)
    session, store = load(run_folder)
    session.tracks = [track for track in session.tracks if track.id != "C"]
    store.replace_from("C", 0)  # every record of C
    store.save(run_folder / "results.npz")
    session.save(run_folder / "session.json")
    (run_folder / MODEL / "B.csv.tmp").write_text("half a file")  # an export that was killed
    (run_folder / MODEL / "A.csv.new").write_text("an export while A.csv was open elsewhere")
    lines = []
    report = export.export_all(run_folder, log=lines.append)
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "B.csv"}
    assert any("C.csv" in line and "removed" in line for line in lines)
    assert set(table(run_folder / "positions.csv").track_id) == {"A", "B"}
    assert report.warnings == []


def test_results_of_a_track_the_session_does_not_list_are_not_exported_and_reported(run_folder):
    session, _ = load(run_folder)
    session.tracks = [track for track in session.tracks if track.id != "C"]
    session.save(run_folder / "session.json")
    report = export.export_all(run_folder, log=silent)
    assert set(table(run_folder / "positions.csv").track_id) == {"A", "B"}
    assert not (run_folder / MODEL / "C.csv").exists()
    assert any("C" in warning and "session" in warning for warning in report.warnings)


# ---------------------------------------------------------------------------------------------
# Locked files (SPEC 8.1)


def test_locked_files_get_the_new_data_next_to_them_and_a_warning(run_folder, monkeypatch):
    export.export_all(run_folder, log=silent)
    old = {name: (run_folder / name).read_bytes() for name in ("positions.csv", f"{MODEL}/A.csv")}
    session, _ = load(run_folder)
    session.calibration.stick["length_mm"] = 29.0  # so that the new files differ from the old ones
    session.save(run_folder / "session.json")
    _lock(monkeypatch, "positions.csv", "A.csv")
    lines = []
    report = export.export_all(run_folder, log=lines.append)
    assert {name: (run_folder / name).read_bytes() for name in old} == old  # the locked files are as they were
    new_positions = table(run_folder / "positions.new.csv")
    np.testing.assert_allclose(new_positions.x_mm, 29 / 30 * table(run_folder / "positions.csv").x_mm, atol=1e-6)
    # in the Tracker-format folder the fallback must not be a .csv or .txt: a loader would read it as a track
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "A.csv.new", "B.csv", "C.csv"}
    (name, read), = reference.read_tracker_export(_as_csv(run_folder / MODEL / "A.csv.new")).items()
    np.testing.assert_array_equal(read.x.to_numpy(), new_positions[new_positions.track_id == "A"].x_mm.to_numpy())
    for locked, fallback in (("positions.csv", "positions.new.csv"), ("A.csv", "A.csv.new")):
        assert any(locked in warning and fallback in warning for warning in report.warnings), report.warnings
        assert any(fallback in line for line in lines)
    assert run_folder / "positions.new.csv" in report.files and run_folder / MODEL / "A.csv.new" in report.files

    monkeypatch.undo()  # the other program has closed the files: the next export replaces them and tidies up
    report = export.export_all(run_folder, log=lines.append)
    assert report.warnings == []
    assert not (run_folder / "positions.new.csv").exists()
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "B.csv", "C.csv"}
    np.testing.assert_array_equal(table(run_folder / "positions.csv").x_mm.to_numpy(), new_positions.x_mm.to_numpy())
    assert any("positions.new.csv" in line and "removed" in line for line in lines)


def _as_csv(path):
    """A copy of a file under the name A.csv in a folder of its own (the reader names a track by its file)."""
    target = path.parent.parent / "copy" / "A.csv"
    target.parent.mkdir()
    shutil.copyfile(path, target)
    return target


# One rule for a locked file. `fileio.atomic_write` cannot write into the Tracker-format folder (its
# temporary file and its fallback end in .csv), so that folder names its own two files and renames
# with `fileio.replace_with_retry`, as `atomic_write` does: it must wait, give up and complain
# exactly as `atomic_write`. Each case: the error of a rename onto A.csv, how many times it comes,
# and whether the fallback is locked as well.
ALWAYS = 10 ** 6
LOCKS = [
    pytest.param(PermissionError(13, "in use"), 0, False, id="free"),
    pytest.param(PermissionError(13, "in use"), 1, False, id="locked at the first try"),
    pytest.param(PermissionError(13, "in use"), len(fileio.RETRY_DELAYS_S), False, id="free at the last try"),
    pytest.param(PermissionError(13, "in use"), ALWAYS, False, id="stays locked"),
    pytest.param(PermissionError(13, "in use"), ALWAYS, True, id="the fallback is locked as well"),
    pytest.param(OSError(errno.EBUSY, "busy"), ALWAYS, False, id="busy, which is not locked"),
    pytest.param(BlockingIOError(errno.EAGAIN, "try again"), ALWAYS, False, id="try again, which is not locked"),
    pytest.param(FileNotFoundError(errno.ENOENT, "no such folder"), ALWAYS, False, id="another error"),
]


def _what_happens(monkeypatch, folder, write, error, times, fallback_locked):
    """Write A.csv in `folder`, where it exists already, with `write(folder)`, which returns the path
    written. Returns what was done about the lock, in order: each rename tried, each wait in s, how
    it ended, then what each file of the folder holds. The fallback's name and the folder's are left
    out of the texts, since the two writers differ in them by design."""
    folder.mkdir()
    (folder / "A.csv").write_text("old")
    events, fallback, real = [], [], os.replace

    def replace(src, dst, **kwargs):
        onto_file = os.path.basename(dst) == "A.csv"
        events.append("rename onto A.csv" if onto_file else "rename onto the fallback")
        if not onto_file:
            fallback.append(os.path.basename(dst))
            if fallback_locked:
                raise PermissionError(13, "in use", str(dst))
        elif events.count("rename onto A.csv") <= times:
            raise type(error)(*error.args)
        return real(src, dst, **kwargs)

    def plain(text):
        for name in fallback[:1]:
            text = text.replace(name, "<fallback>")
        return text.replace(str(folder), "<folder>")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace)
        patch.setattr(time, "sleep", lambda seconds: events.append(f"wait {seconds:g} s"))
        try:
            events.append(plain(f"returned {write(folder).name}"))
        except OSError as err:
            events.append(plain(f"{type(err).__name__}: {err}"))
    events += [plain(f"{path.name} holds the {'old' if path.read_text() == 'old' else 'new'} data")
               for path in sorted(folder.iterdir())]
    return events


@pytest.mark.parametrize(("error", "times", "fallback_locked"), LOCKS)
def test_a_locked_tracker_file_is_handled_exactly_as_atomic_write_handles_one(tracked, tmp_path, monkeypatch, error,
                                                                             times, fallback_locked):
    track_a = export.derive_all(*load(tracked)).derived["A"]
    ours = _what_happens(monkeypatch, tmp_path / MODEL, lambda folder: export_tables.write_tracker_folder(
        folder, {"A": track_a}, {"A": "A.csv"}, silent)[0][0], error, times, fallback_locked)
    theirs = _what_happens(monkeypatch, tmp_path / "plain", lambda folder: fileio.atomic_write(
        folder / "A.csv", lambda tmp: tmp.write_text("new")), error, times, fallback_locked)
    assert ours == theirs
    assert ours[0] == "rename onto A.csv" and len(ours) >= 3  # a rename, an end, and A.csv itself


# ---------------------------------------------------------------------------------------------
# Odd folder names, the overlay, a video that is gone (review focus 1 and 5)


def test_export_with_the_overlay_into_a_folder_with_a_space_and_non_ascii_letters(clip_in_odd_folder):
    run_folder = clip_in_odd_folder.path.parent / "dish_tracker_outline_Zoë K"
    coarse_run(clip_in_odd_folder, run_folder, ["B", "C"])
    lines = []
    report = export.export_all(run_folder, overlay=True, log=lines.append)
    assert report.warnings == []
    assert {path.name for path in run_folder.iterdir()} == RUN_FILES | {schema.OVERLAY_MP4}
    assert run_folder / "overlay.mp4" in report.files and (run_folder / "overlay.mp4").stat().st_size > 1000
    assert list(table(run_folder / "positions.csv").track_id) == ["B"] * 60 + ["C"] * 60
    assert "overlay.mp4" in (run_folder / "run.log").read_text(encoding="utf-8")


def test_without_the_video_every_other_file_is_written_and_the_overlay_is_reported_as_skipped(clip_in_odd_folder):
    run_folder = clip_in_odd_folder.path.parent / "run"
    coarse_run(clip_in_odd_folder, run_folder, ["B"])
    clip_in_odd_folder.path.unlink()
    lines = []
    report = export.export_all(run_folder, overlay=True, log=lines.append)
    assert {path.name for path in run_folder.iterdir()} == RUN_FILES
    assert any("overlay" in warning and "skipped" in warning for warning in report.warnings)
    assert any("overlay" in line and "skipped" in line for line in lines)
    assert "overlay" in (run_folder / "run.log").read_text(encoding="utf-8")


def test_without_overlay_an_existing_overlay_is_left_alone(run_folder):
    (run_folder / "overlay.mp4").write_bytes(b"an earlier overlay")
    report = export.export_all(run_folder, log=silent)
    assert (run_folder / "overlay.mp4").read_bytes() == b"an earlier overlay"
    assert run_folder / "overlay.mp4" not in report.files
