"""A run folder of version 1 opens and exports (W3 step 1, "Done when"; rule 5 of docs/ROADMAP.md section 2;
decisions 37 to 39; docs/design/formats.md, "Writing over a version 1 file").

One run folder of each kind that 0.1.0 made: in the window (formats.md's example 1) and by from-tracker
(example 2), built from literal values beside a copy of the dish clip, with the results of its tracks as the
tool tracks them with `ExactFake` (tests/support/session_v1_helpers.py, `old_run`). Opening it in the window
and closing it writes nothing; `outline-tracker export` writes the outputs and leaves session.json as it was;
Export all in the window saves first, so the old file is then kept as session.v1.json, byte for byte.

Expected positions come from the scene: the centroids of the true masks (inside the dish crop for the run made
in the window, which has the dish circle; on the whole frame for the other), mapped to mm by SPEC 3.2 with
k = 0.1 mm per px, the origin (100.5, 60.25) px and the axis angle of the kind:
x = k[(u - u0) cos a - (v - v0) sin a], y = -k[(u - u0) sin a + (v - v0) cos a]; t_s = frame / 240.

Coordinates: px in Tracker's convention (pixel centers at +0.5), world mm with y up; frames of the video.
"""

import json
import math

import numpy as np
import pytest

from cli_export_helpers import export
from export_helpers import table
from export_panel_helpers import export_panel, export_to_end
from gui_helpers import record_dialogs
from session_helpers import body, read_json, settle
from session_v1_helpers import ANGLE_DEG, IDS, K, OLD_NS, ORIGIN, old_run, snapshot
from tracking_helpers import DISH_BOX, box_truth, table_truth

pytestmark = pytest.mark.usefixtures("own_settings")  # every test here has a settings folder of its own

KINDS = ["window", "from-tracker"]
GRID = list(range(0, 119, 2))  # the clip of both kinds: frames 0 to 118 in steps of 2


def assert_true_positions(run_folder, clip, kind):
    """positions.csv of the run folder holds the scene's centroids of each track on GRID, in px and in mm."""
    rows = table(run_folder / "positions.csv")
    alpha = math.radians(ANGLE_DEG[kind])
    assert sorted(set(rows.track_id)) == list(IDS[kind])
    for track_id in IDS[kind]:
        mine = rows[rows.track_id == track_id]
        if kind == "window":  # the dish circle: tracked inside its crop
            u, v, _, _ = box_truth(clip, track_id, GRID, DISH_BOX)
        else:
            u, v, _ = table_truth(clip, track_id, GRID)
        du, dv = u - ORIGIN[0], v - ORIGIN[1]
        x = K * (du * math.cos(alpha) - dv * math.sin(alpha))
        y = -K * (du * math.sin(alpha) + dv * math.cos(alpha))
        assert list(mine.frame) == GRID
        # 0.01 px is SPEC 13.2's tolerance with ExactFake; the cells are rounded to 3 and 6 decimals
        np.testing.assert_allclose(mine.u_px, u, rtol=0, atol=0.01 + 0.0005)
        np.testing.assert_allclose(mine.v_px, v, rtol=0, atol=0.01 + 0.0005)
        np.testing.assert_allclose(mine.x_mm, x, rtol=0, atol=K * 0.01 + 1e-6)
        np.testing.assert_allclose(mine.y_mm, y, rtol=0, atol=K * 0.01 + 1e-6)
        np.testing.assert_allclose(mine.t_s, np.array(GRID) / 240.0, rtol=0, atol=5e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_the_command_exports_a_version_1_run_folder_and_leaves_its_session_json(clip_in_odd_folder, tmp_path, kind):
    run_folder = old_run(clip_in_odd_folder, kind, tmp_path / "scratch")
    before = snapshot(run_folder)

    assert export(run_folder / "session.json") == 0
    assert_true_positions(run_folder, clip_in_odd_folder, kind)
    after = snapshot(run_folder)
    for name in ("session.json", "results.npz"):
        assert after[name] == before[name]  # the same bytes and the same time (decision 38)
    assert "session.v1.json" not in after


@pytest.mark.parametrize("kind", KINDS)
def test_a_version_1_run_folder_opens_in_the_window_and_closing_it_writes_nothing(window, qtbot, monkeypatch,
                                                                                  clip_in_odd_folder, tmp_path,
                                                                                  kind):
    asked = record_dialogs(monkeypatch)
    run_folder = old_run(clip_in_odd_folder, kind, tmp_path / "scratch")
    before = snapshot(clip_in_odd_folder.path.parent)

    window.open_path(run_folder / "session.json")
    controller = window.controller
    assert controller.run_folder == run_folder and controller.video_path == clip_in_odd_folder.path
    assert [track.id for track in controller.session.tracks] == list(IDS[kind])
    assert body(window, 1).name_edit.text() == "ana"
    assert export_panel(window).refusal() is None  # results, fps_true and a scale: ready to export
    settle(qtbot, window)
    assert window.close()

    assert snapshot(clip_in_odd_folder.path.parent) == before  # the same files, bytes and times (rule 5)
    assert asked.messages == []


@pytest.mark.parametrize("kind", KINDS)
def test_export_all_in_the_window_keeps_the_version_1_file_and_writes_the_true_positions(window, qtbot,
                                                                                         monkeypatch,
                                                                                         clip_in_odd_folder,
                                                                                         tmp_path, kind):
    asked = record_dialogs(monkeypatch)
    run_folder = old_run(clip_in_odd_folder, kind, tmp_path / "scratch")
    old = (run_folder / "session.json").read_bytes()
    window.open_path(run_folder / "session.json")
    settle(qtbot, window)

    panel = export_panel(window)
    export_to_end(qtbot, panel)
    assert panel.message.kind == "success", panel.message.text()
    kept = run_folder / "session.v1.json"
    assert kept.read_bytes() == old and kept.stat().st_mtime_ns == OLD_NS  # the save before the export kept it
    saved = read_json(run_folder / "session.json")
    assert saved["schema_version"] == 2 and saved["label"] == "ana"
    assert json.loads(old)["student"] == "ana" and "label" not in json.loads(old)
    assert_true_positions(run_folder, clip_in_odd_folder, kind)
    (_, kind_of_message, text), = asked.messages
    assert kind_of_message == "info" and "session.v1.json" in text
    assert window.close()
    assert kept.read_bytes() == old
