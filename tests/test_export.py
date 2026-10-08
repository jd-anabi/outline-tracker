"""Export: positions.csv, shapes.csv and the file set of a run folder (SPEC 8.1, 8.2, 8.4, 8.11, 13.2).

The run folders are tracked by `tracking.run_job` with `ExactFake` on the synthetic dish clip, so
every expected number comes from the scene: the true masks' centroids and pixel counts
(tests/tracking_helpers.py), the stick of tests/export_helpers.py (K = 0.1 mm per px, origin
(100.5, 60.25) px, axis angle 0) and t_s = frame / 240. The Tracker-format folder is in
tests/test_export_tracker_folder.py, the shape numbers in tests/test_export_shapes.py, corrections
in tests/test_export_corrections.py and run.log in tests/test_export_log.py.

Coordinates: px in Tracker's convention (pixel centers at +0.5), world mm with y up.
"""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from export_helpers import K, MODEL, RUN_FILES, cells, coarse_run, load, silent, store_run, table, true_pieces, world
from overlay_helpers import lost_record, record
from tracking_helpers import DISH_BOX, box_truth

from outline_tracker import export, schema, schema_docs

GRID = list(range(0, 120, 2))  # the dish clip has 120 frames; the sessions take every 2nd
MEASURED = {  # the columns that hold a measured number: an empty cell on a lost row
    schema.POSITIONS_CSV: ["x_mm", "y_mm", "u_px", "v_px", "area_mm2"],
    schema.SHAPES_CSV: ["area_mm2", "perimeter_mm", "major_mm", "minor_mm", "eccentricity", "theta_rad", "core_x_mm",
                        "core_y_mm", "core_frac", "solidity", "circularity", "feret_max_mm", "largest_fraction",
                        "px_along_major", "cells_along_major", "wall_dist_centroid_mm", "wall_dist_min_mm"],
}


@pytest.fixture(scope="module")
def tracked(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse inside the dish crop; nothing exported yet."""
    folder = tmp_path_factory.mktemp("export") / "run"
    coarse_run(dish_clip, folder, ["A", "B", "C"])
    return folder


@pytest.fixture
def run_folder(tracked, tmp_path):
    """This test's own copy of the tracked run folder."""
    return shutil.copytree(tracked, tmp_path / "run")


# ---------------------------------------------------------------------------------------------
# The files (SPEC 8.1)


def test_a_coarse_only_run_gets_every_file_with_empty_shape_files(run_folder):
    report = export.export_all(run_folder, log=silent)
    assert {path.name for path in run_folder.iterdir()} == RUN_FILES
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "B.csv", "C.csv"}
    assert report.warnings == []
    written = {path.relative_to(run_folder).as_posix() for path in report.files}
    assert written == {"positions.csv", "shapes.csv", "radial.csv", "outlines.npz", "README.txt", "run.log",
                       "edgetam/A.csv", "edgetam/B.csv", "edgetam/C.csv"}
    # no fine track: the header alone, and `meta` alone (decision X12)
    assert (run_folder / "radial.csv").read_bytes() == (schema.header_line(schema.RADIAL) + "\n").encode()
    with np.load(run_folder / "outlines.npz") as outlines:
        assert outlines.files == ["meta"]
        meta = json.loads(str(outlines["meta"]))
    assert meta["n_points"] == 128 and meta["tracks"] == []
    assert (run_folder / "README.txt").read_bytes() == schema_docs.readme_text().encode("utf-8")


def test_the_new_tables_are_utf8_text_with_lf_line_ends_and_the_schema_headers(run_folder):
    export.export_all(run_folder, log=silent)
    for name, columns in ((schema.POSITIONS_CSV, schema.POSITIONS), (schema.SHAPES_CSV, schema.SHAPES)):
        data = (run_folder / name).read_bytes()
        assert b"\r" not in data and data.endswith(b"\n")
        assert data.decode("utf-8").split("\n")[0] == ",".join(column.name for column in columns)


# ---------------------------------------------------------------------------------------------
# Rows and numbers (SPEC 8.2, 8.4)


def test_every_track_has_every_grid_frame_sorted_by_track_then_frame(run_folder):
    export.export_all(run_folder, log=silent)
    for name in (schema.POSITIONS_CSV, schema.SHAPES_CSV):
        rows = table(run_folder / name)
        assert list(rows.track_id) == ["A"] * 60 + ["B"] * 60 + ["C"] * 60
        assert list(rows.frame) == GRID * 3


def test_positions_are_the_true_centroids_in_px_and_in_the_users_axes(run_folder, dish_clip):
    export.export_all(run_folder, log=silent)
    rows = table(run_folder / "positions.csv")
    for track_id in "ABC":
        mine = rows[rows.track_id == track_id]
        u, v, area, _ = box_truth(dish_clip, track_id, GRID, DISH_BOX)
        x, y = world(u, v)
        # 0.01 px is SPEC 13.2's tolerance with ExactFake; the cells are rounded to 3 and 6 decimals
        np.testing.assert_allclose(mine.u_px, u, rtol=0, atol=0.01 + 0.0005)
        np.testing.assert_allclose(mine.v_px, v, rtol=0, atol=0.01 + 0.0005)
        np.testing.assert_allclose(mine.x_mm, x, rtol=0, atol=K * 0.01 + 1e-6)
        np.testing.assert_allclose(mine.y_mm, y, rtol=0, atol=K * 0.01 + 1e-6)
        np.testing.assert_allclose(mine.area_mm2, K * K * area, rtol=0, atol=1e-6)
        np.testing.assert_allclose(mine.t_s, np.array(GRID) / 240.0, rtol=0, atol=5e-8)
        assert set(mine.visible) == {1} and set(mine["mode"]) == {"coarse"}


def test_shapes_has_the_rows_and_the_flags_of_positions(run_folder, dish_clip):
    export.export_all(run_folder, log=silent)
    positions, shapes = table(run_folder / "positions.csv"), table(run_folder / "shapes.csv")
    for column in ("track_id", "frame", "t_s", "area_mm2", "flags"):
        assert list(positions[column]) == list(shapes[column]), column
    assert set(shapes[shapes.track_id != "A"].n_components) == {1}  # B and C are plain ellipses: one piece
    # every track, frame by frame: the pieces of the true mask inside the dish crop, counted with scipy
    pieces = {track_id: [true_pieces(dish_clip, track_id, frame, DISH_BOX) for frame in GRID] for track_id in "ABC"}
    for track_id in "ABC":
        assert list(shapes[shapes.track_id == track_id].n_components) == pieces[track_id], track_id
    # A is not one piece on every frame: its antennae are thinner than a pixel, so on some frames
    # the true mask itself is a body and a piece of antenna that does not touch it
    assert dish_clip.scene.objects[0].shape.antenna_width_px < 1.0 and max(pieces["A"]) > 1


def test_a_lost_frame_keeps_its_row_with_empty_cells_and_zero_counts(dish_clip, tmp_path):
    # fps_true = 239.6 in these store-level sessions: 2 / 239.6 = 0.00834724...
    records = {"A": [record(dish_clip, "A", 0), lost_record(dish_clip, 2), record(dish_clip, "A", 4)]}
    store_run(tmp_path / "run", records)
    export.export_all(tmp_path / "run", log=silent)
    for name, counts in ((schema.POSITIONS_CSV, ["visible"]), (schema.SHAPES_CSV, ["n_components", "shape_ok"])):
        header, rows = cells(tmp_path / "run" / name)
        assert [row["frame"] for row in rows] == ["0", "2", "4"]
        lost = rows[1]
        assert lost["track_id"] == "A" and lost["t_s"] == "0.0083472"
        assert [lost[column] for column in MEASURED[name]] == [""] * len(MEASURED[name])
        assert [lost[column] for column in counts] == ["0"] * len(counts)
        assert lost["flags"] == "LOST;HEADGUESS"  # no head click: HEADGUESS is on every row of the track
        if name == schema.POSITIONS_CSV:
            assert lost["mode"] == "coarse"
        assert all(rows[0][column] != "" for column in MEASURED[name] if not column.startswith("wall_"))
    assert (tmp_path / "run" / MODEL / "A.csv").read_text().splitlines()[3] == "0.0083472,2,,,,"


def test_tracks_are_sorted_by_plain_text_order_of_their_ids(dish_clip, tmp_path):
    ids = ["b", "A2", "A", "10"]
    store_run(tmp_path / "run", {track_id: [record(dish_clip, "B", frame) for frame in (0, 2)] for track_id in ids})
    export.export_all(tmp_path / "run", log=silent)
    for name in (schema.POSITIONS_CSV, schema.SHAPES_CSV):
        rows = table(tmp_path / "run" / name)
        assert list(rows.track_id) == ["10", "10", "A", "A", "A2", "A2", "b", "b"]
        assert list(rows.frame) == [0, 2] * 4


def test_a_partial_store_exports_the_frames_it_has(run_folder):
    session, store = load(run_folder)
    store.replace_from("B", 40)  # as after a run that was stopped: B has frames 0 to 38
    store.save(run_folder / "results.npz")
    session.complete = False
    session.save(run_folder / "session.json")
    export.export_all(run_folder, log=silent)
    rows = table(run_folder / "positions.csv")
    assert list(rows[rows.track_id == "B"].frame) == list(range(0, 40, 2))
    assert list(rows[rows.track_id == "A"].frame) == GRID and list(rows[rows.track_id == "C"].frame) == GRID
    assert load(run_folder)[0].complete is False  # export does not decide that
    assert "partial" in (run_folder / "run.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# A new calibration needs no tracking (SPEC 7, 13.2)


def test_a_stick_of_29_mm_scales_every_x_and_y_by_29_over_30_without_tracking(run_folder, monkeypatch):
    from outline_tracker import measure, tracking

    export.export_all(run_folder, log=silent)
    before = {name: table(run_folder / name) for name in ("positions.csv", "shapes.csv")}
    session30, store = load(run_folder)
    results = (run_folder / "results.npz").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("export must not track or measure a mask")

    monkeypatch.setattr(tracking, "run_job", forbidden)
    monkeypatch.setattr(measure, "measure_mask", forbidden)
    session29, _ = load(run_folder)
    session29.calibration.stick["length_mm"] = 29.0
    session29.save(run_folder / "session.json")
    export.export_all(run_folder, log=silent)
    after = {name: table(run_folder / name) for name in ("positions.csv", "shapes.csv")}
    assert (run_folder / "results.npz").read_bytes() == results

    ratio = 29.0 / 30.0
    old, new = before["positions.csv"], after["positions.csv"]
    for column in ("x_mm", "y_mm"):  # two cells rounded to 6 decimals: within 1e-6 mm
        np.testing.assert_allclose(new[column], ratio * old[column], rtol=0, atol=1e-6)
    np.testing.assert_allclose(new.area_mm2, ratio ** 2 * old.area_mm2, rtol=0, atol=1e-6)
    for column in ("track_id", "frame", "t_s", "u_px", "v_px", "visible", "flags"):
        assert list(new[column]) == list(old[column]), column
    old, new = before["shapes.csv"], after["shapes.csv"]
    for column in ("perimeter_mm", "major_mm", "minor_mm", "core_x_mm", "core_y_mm", "feret_max_mm",
                   "wall_dist_centroid_mm", "wall_dist_min_mm"):
        np.testing.assert_allclose(new[column], ratio * old[column], rtol=0, atol=1e-6, err_msg=column)
    for column in ("theta_rad", "eccentricity", "solidity", "circularity", "core_frac", "px_along_major"):
        np.testing.assert_allclose(new[column], old[column], rtol=0, atol=1e-6, err_msg=column)

    # in memory, before any rounding: 1e-12 relative
    data30, data29 = export.derive_all(session30, store), export.derive_all(session29, store)
    assert list(data30.derived) == ["A", "B", "C"]
    for track_id in "ABC":
        old, new = data30.derived[track_id], data29.derived[track_id]
        np.testing.assert_allclose(new.x_mm, ratio * old.x_mm, rtol=1e-12, atol=0)
        np.testing.assert_allclose(new.y_mm, ratio * old.y_mm, rtol=1e-12, atol=0)
        np.testing.assert_allclose(new.area_mm2, ratio ** 2 * old.area_mm2, rtol=1e-12, atol=0)


def test_export_loads_neither_torch_nor_qt(run_folder):
    # in a process of its own: pytest-qt has already loaded PySide6 into this one
    code = ("import sys; from outline_tracker import export; export.export_all(sys.argv[1], log=lambda line: None); "
            "print(sorted(name for name in sys.argv[2:] if name in sys.modules))")
    heavy = ["torch", "torchvision", "transformers", "timm", "PySide6", "pyqtgraph"]
    done = subprocess.run([sys.executable, "-c", code, str(run_folder), *heavy], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
    assert (run_folder / "positions.csv").is_file() and (run_folder / "run.log").is_file()


# ---------------------------------------------------------------------------------------------
# What is refused, before anything is written


def _files(run_folder):
    return sorted(path.relative_to(run_folder).as_posix() for path in run_folder.rglob("*"))


def test_a_radial_step_other_than_5_degrees_is_refused_and_nothing_is_written(run_folder):
    session, _ = load(run_folder)
    session.processing.radial_step_deg = 10
    session.save(run_folder / "session.json")
    before = _files(run_folder)
    with pytest.raises(ValueError, match=r"radial_step_deg.*\b10\b.*\b5\b"):
        export.export_all(run_folder, log=silent)
    assert _files(run_folder) == before


def test_a_session_without_a_scale_is_refused_and_nothing_is_written(run_folder):
    session, _ = load(run_folder)
    session.calibration.stick = None
    session.save(run_folder / "session.json")
    before = _files(run_folder)
    with pytest.raises(ValueError, match="no scale"):
        export.export_all(run_folder, log=silent)
    assert _files(run_folder) == before


def test_a_folder_without_results_is_refused_with_a_plain_message(run_folder):
    (run_folder / "results.npz").unlink()
    with pytest.raises(FileNotFoundError, match="results.npz"):
        export.export_all(run_folder, log=silent)
    assert not (run_folder / "positions.csv").exists()
