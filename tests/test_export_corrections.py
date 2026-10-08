"""Export after corrections (SPEC 6.6, 13.2): "Re-track from here", "End track here" and "Continue
as new track", at the level of the results store.

The correction functions are written in another task; here a test does to results.npz and
session.json what they do (`ResultsStore.replace_from`, `truncate_after`, `put`, and the session's
tracks), and export must then show exactly that. New records are the ground truth of another
object of the dish scene, so they differ from the old ones on every frame.

Coordinates: px in Tracker's convention (pixel centers at +0.5), world mm with y up.
"""

import shutil

import numpy as np
import pytest
from export_helpers import K, MODEL, cells, coarse_run, load, table, world
from overlay_helpers import record
from tracking_helpers import table_truth

from outline_tracker import export
from outline_tracker.session import Correction, Prompt, Track

GRID = list(range(0, 120, 2))
POSITION_COLUMNS = ["u_px", "v_px", "x_mm", "y_mm", "area_mm2", "visible"]


@pytest.fixture(scope="module")
def exported(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse and exported once."""
    folder = tmp_path_factory.mktemp("corrections") / "run"
    coarse_run(dish_clip, folder, ["A", "B", "C"])
    export.export_all(folder, log=lambda line: None)
    return folder


# Overrides the shared `run_folder` of tests/conftest.py: it copies a run that was also exported once (`exported`).
@pytest.fixture
def run_folder(exported, tmp_path):
    """This test's own copy of the exported run folder."""
    return shutil.copytree(exported, tmp_path / "run")


def test_after_a_retrack_only_that_tracks_rows_from_frame_k_on_change(run_folder, dish_clip):
    _, before = cells(run_folder / "positions.csv")
    session, store = load(run_folder)
    store.replace_from("B", 60)
    for frame in range(60, 120, 2):  # B now follows the scene's A: another position on every frame
        store.put("B", record(dish_clip, "A", frame))
    store.save(run_folder / "results.npz")
    session.corrections.append(Correction(time="2026-10-07T03:10:00-07:00", tracks=["B"], action="retrack", frame=60))
    session.save(run_folder / "session.json")
    export.export_all(run_folder, log=lambda line: None)
    header, after = cells(run_folder / "positions.csv")
    assert [(row["track_id"], row["frame"]) for row in after] == [(row["track_id"], row["frame"]) for row in before]
    for old, new in zip(before, after, strict=True):
        same = [old[column] == new[column] for column in POSITION_COLUMNS]
        if new["track_id"] == "B" and int(new["frame"]) >= 60:
            assert not any(same[:5]), new  # u, v, x, y and the area all moved
        else:
            assert all(same), new
    mine = table(run_folder / "positions.csv")
    mine = mine[(mine.track_id == "B") & (mine.frame >= 60)]
    u, v, _ = table_truth(dish_clip, "A", range(60, 120, 2))
    np.testing.assert_allclose(mine.x_mm, world(u, v)[0], rtol=0, atol=K * 0.01 + 1e-6)


def test_after_end_track_and_new_piece_the_export_has_both_a_and_a2(run_folder, dish_clip):
    session, store = load(run_folder)
    store.truncate_after("A", 58)
    for frame in range(60, 120, 2):
        store.put("A2", record(dish_clip, "A", frame))
    store.save(run_folder / "results.npz")
    session.tracks[0].ended_at = 58
    click = Prompt(frame=60, points_px=[[10.0, 10.0]], labels=[1])
    session.tracks.insert(1, Track(id="A2", color="#FFFF00", start_frame=60, prompts=[click]))
    session.corrections += [Correction(time="2026-10-07T03:10:00-07:00", tracks=["A"], action="end", frame=58),
                            Correction(time="2026-10-07T03:11:00-07:00", tracks=["A2"], action="new_piece", frame=60,
                                       prompts=[click])]
    session.save(run_folder / "session.json")
    report = export.export_all(run_folder, log=lambda line: None)
    assert report.warnings == []
    for name in ("positions.csv", "shapes.csv"):
        rows = table(run_folder / name)
        assert list(rows.track_id) == ["A"] * 30 + ["A2"] * 30 + ["B"] * 60 + ["C"] * 60
        assert list(rows[rows.track_id == "A"].frame) == GRID[:30]
        assert list(rows[rows.track_id == "A2"].frame) == GRID[30:]
    assert {path.name for path in (run_folder / MODEL).iterdir()} == {"A.csv", "A2.csv", "B.csv", "C.csv"}
    piece = table(run_folder / "positions.csv")
    piece = piece[piece.track_id == "A2"]
    u, v, _ = table_truth(dish_clip, "A", GRID[30:])
    np.testing.assert_allclose(piece.u_px, u, rtol=0, atol=0.01 + 0.0005)
    np.testing.assert_allclose(piece.v_px, v, rtol=0, atol=0.01 + 0.0005)
    log = (run_folder / "run.log").read_text(encoding="utf-8")
    assert "end" in log and "new_piece" in log
