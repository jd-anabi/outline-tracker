"""Export of a results.npz that has a gap inside a track (SPEC 8.2: every grid frame from a track's
first to its last, lost frames as rows with empty positions).

Tracking and the edit functions leave no gap (tests/test_export_pipeline.py,
tests/test_corrections_gaps.py). A results.npz made by hand or by an older version may have one:
then positions.csv and shapes.csv get a lost row for every frame of the clip's grid between the
track's first and last record that has no record, written as the row of a frame on which the
object was not found; the Tracker-format file (SPEC 8.3: one row per tracked frame), radial.csv and
outlines.npz hold the frames with a record only; and a warning names the track.

The run folders are made by hand (`export_helpers.store_run`) from the ground truth of the dish
clip: fps_true = 239.6 frames per s, so frame 4 is at 4 / 239.6 = 0.0166945 s and frame 6 at
0.0250417 s. "As a lost frame is written" is checked against a second folder that has, on the
same frames, the records tracking stores for an object that was not found. Frames are video frame
numbers; positions are px in Tracker's convention and mm in the user's axes (y up).
"""

import numpy as np
import pytest
from export_helpers import MODEL, cells, frames_of, log_section, store_run
from overlay_helpers import lost_record, record

from outline_tracker import export

# what a lost row keeps: every other cell of it is empty
KEPT = {"positions.csv": {"track_id", "frame", "t_s", "visible", "mode", "flags"},
        "shapes.csv": {"track_id", "frame", "t_s", "n_components", "shape_ok", "flags"}}


def silent(_line):
    """A log that keeps nothing."""


def gap_run(clip, run_folder, lost=False):
    """A run folder made by hand, with a results.npz the tool does not make: of the clip's frames
    0, 2, ..., 10, track A has no record on frames 4 and 6 (B has every frame). With `lost`, A has
    there what tracking stores for an object that was not found. Radial and outline rows are
    written for these coarse tracks. Returns the session; fps_true is 239.6."""
    between = [lost_record(clip, frame) for frame in (4, 6)] if lost else []
    records = {"A": [record(clip, "A", 0), record(clip, "A", 2), *between, record(clip, "A", 8), record(clip, "A", 10)],
               "B": [record(clip, "B", frame) for frame in range(0, 11, 2)]}
    session = store_run(run_folder, records)
    assert (session.clip.start, session.clip.step) == (0, 2)
    session.processing.shape_files_for_coarse = True
    session.save(run_folder / "session.json")
    return session


def test_a_gap_in_results_npz_is_written_as_lost_rows_in_positions_and_shapes(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    gap_run(dish_clip, run_folder)
    report = export.export_all(run_folder, log=silent)
    for name, kept in KEPT.items():
        header, rows = cells(run_folder / name)
        assert [row["frame"] for row in rows] == ["0", "2", "4", "6", "8", "10"] * 2
        assert [row["track_id"] for row in rows] == ["A"] * 6 + ["B"] * 6
        for row, t_s in ((rows[2], "0.0166945"), (rows[3], "0.0250417")):  # 4 / 239.6 and 6 / 239.6 s
            assert row["t_s"] == t_s
            assert [row[column] for column in header if column not in kept] == [""] * (len(header) - len(kept))
            assert row["flags"] == "LOST;HEADGUESS"  # as a lost row of a track without a head click
            if name == "positions.csv":
                assert (row["visible"], row["mode"]) == ("0", "coarse")
            else:
                assert (row["n_components"], row["shape_ok"]) == ("0", "0")
        for row in rows[:2] + rows[4:]:  # the frames that have a record are there as before
            assert row["u_px" if name == "positions.csv" else "major_mm"] != "" and "LOST" not in row["flags"]
    # the user is told, with the track and the first frame without a record
    (warning,) = report.warnings
    assert "track A" in warning and "frame 4" in warning and "2 frames" in warning
    assert "A: LOST in 2 of 6 frames (first at frame 4, t = 0.017 s)" in log_section(run_folder, "qc summary:")


def test_a_frame_without_a_record_is_a_lost_row_in_positions_and_shapes_and_no_row_in_the_tracker_file(dish_clip,
                                                                                                      tmp_path):
    gap_run(dish_clip, tmp_path / "gap")
    gap_run(dish_clip, tmp_path / "lost", lost=True)
    reports = {name: export.export_all(tmp_path / name, log=silent) for name in ("gap", "lost")}
    # positions.csv and shapes.csv: as if tracking had stored "not found" there. B has every frame.
    for name in ("positions.csv", "shapes.csv", f"{MODEL}/B.csv"):
        assert (tmp_path / "gap" / name).read_bytes() == (tmp_path / "lost" / name).read_bytes(), name
    # the Tracker-format file has one row per tracked frame (SPEC 8.3): none for frames 4 and 6
    lost = (tmp_path / "lost" / MODEL / "A.csv").read_text().splitlines()
    assert lost[4:6] == ["0.0166945,4,,,,", "0.0250417,6,,,,"]  # a frame tracked and not found keeps its row
    gap = (tmp_path / "gap" / MODEL / "A.csv").read_text().splitlines()
    assert gap[:2] == [",A,,,,,", "t,frame,x,y,pixelx,pixely"]
    assert [line.split(",")[1] for line in gap[2:]] == ["0", "2", "8", "10"]
    assert gap == lost[:4] + lost[6:]  # and the rows of the frames with a record are what they are without a gap
    # the warning names the two files that show the frames as lost, and not the Tracker-format file
    (warning,) = reports["gap"].warnings
    assert "positions.csv" in warning and "shapes.csv" in warning and MODEL not in warning
    assert reports["lost"].warnings == []


def test_a_frame_without_a_record_gets_no_row_in_radial_csv_and_outlines_npz(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    gap_run(dish_clip, run_folder)
    export.export_all(run_folder, log=silent)
    assert frames_of(run_folder, "radial.csv", "A") == [0, 2, 8, 10]
    assert frames_of(run_folder, "radial.csv", "B") == [0, 2, 4, 6, 8, 10]
    with np.load(run_folder / "outlines.npz") as outlines:
        assert list(outlines["A__frames"]) == [0, 2, 8, 10] and outlines["A__xy_mm"].shape == (4, 128, 2)
        assert np.isfinite(outlines["A__xy_mm"]).all() and list(outlines["B__frames"]) == [0, 2, 4, 6, 8, 10]


def test_the_lost_rows_are_on_the_clips_grid_and_the_tracker_file_has_the_frames_with_a_record(dish_clip, tmp_path):
    run_folder = tmp_path / "run"
    # records on frames 5, 10 and 17; the clip takes frames 1, 5, 9, 13, 17, ...: 9 and 13 have none
    session = store_run(run_folder, {"B": [record(dish_clip, "B", frame) for frame in (5, 10, 17)]})
    session.clip.start, session.clip.step, session.clip.end = 1, 4, 21
    session.save(run_folder / "session.json")
    report = export.export_all(run_folder, log=silent)
    for name in ("positions.csv", "shapes.csv"):
        _, rows = cells(run_folder / name)
        # frame 10 is not a frame of the clip, but it has a record: its row stays
        assert [(row["frame"], "LOST" in row["flags"]) for row in rows] == [
            ("5", False), ("9", True), ("10", False), ("13", True), ("17", False)]
    # the Tracker-format file: the three frames with a record, no row for 9 and 13
    assert [line.split(",")[1] for line in (run_folder / MODEL / "B.csv").read_text().splitlines()[2:]] == [
        "5", "10", "17"]
    (warning,) = report.warnings
    assert "track B" in warning and "frame 9" in warning and "2 frames" in warning


@pytest.mark.parametrize("step", [0, -2, 2.5, None])
def test_a_clip_step_that_gives_no_grid_is_refused_and_nothing_is_written(dish_clip, tmp_path, step):
    run_folder = tmp_path / "run"
    session = store_run(run_folder, {"A": [record(dish_clip, "A", frame) for frame in (0, 2)]})
    session.clip.step = step
    session.save(run_folder / "session.json")
    before = sorted(path.name for path in run_folder.iterdir())
    with pytest.raises(ValueError, match="step"):
        export.export_all(run_folder, log=silent)
    assert sorted(path.name for path in run_folder.iterdir()) == before
