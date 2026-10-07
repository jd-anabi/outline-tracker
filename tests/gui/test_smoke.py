"""The smoke test of the whole flow in the window (SPEC 13.5, P0; task C6): the window opens, a
synthetic clip loads, calibration, circle and origin are set through the panels' slots, one object
is added by a simulated click, a run with `ExactFake` completes, Export all writes every P0 file
of SPEC 8, the session reopens in a new window with everything restored, and no error was recorded.

Expected values come from the scene and from geometry, not from the window:
- the dish clip is 320 x 240 px with 120 frames at 240 frames per s; cut to frames 0 to 20 at the
  default step of 2 it has 11 tracked frames: 0, 2, ..., 20, at t = frame / 240 s;
- the stick is 300 px long for 30 mm, so the scale is 0.1 mm per px (100 µm per px);
- eight points on the scene's dish wall give its circle back: center (160.3, 119.8) px, R = 108 px,
  so 10.8 mm; Origin to Center puts the origin there, and with the axes angle 0 a point (u, v) px
  is x = 0.1 (u - 160.3) mm, y = -0.1 (v - 119.8) mm (y up);
- `ExactFake` answers with the scene's true mask, so A's position is the true mask's centroid
  (the scene's own table), to 0.01 px;
- SPEC 8.1 lists the files of a run folder (tests/gui/export_panel_helpers.py, `P0_FILES`).
Nothing needs the real model, a dialog that the user must answer, or a file outside the test's own
folder; nothing is read from a pixel of text; no test waits with a delay.

Coordinates: px in Tracker's convention (SPEC 3.1); mm in the session's axes, y up. Frames are
video frame numbers.
"""

import numpy as np
import pytest

from export_helpers import table
from export_panel_helpers import (END, FRAMES, P0_FILES, STICK_ENDS, STICK_MM, export_panel,  # noqa: F401 (fixture)
                                  no_error_recorded, second_window, walk_through)
from finish_helpers import record_every_dialog
from gui_helpers import show
from outline_tracker import schema
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.tracker_io import read_tracker_export
from outline_tracker.video import probe
from session_helpers import body, read_json
from track_helpers import NAME, own_copy, results_of, track_panel
from tracking_helpers import center, table_truth

K = 0.1  # mm per px: 30 mm on 300 px


def reopen(second, qtbot, clip, run_folder):
    """Open the run folder's session in the second window, with a model, and show it."""
    second.segmenter_factory = lambda model, device: ExactFake(clip)
    second.open_path(run_folder / schema.SESSION_JSON)
    show(second, qtbot)
    qtbot.waitUntil(lambda: worker_of(second).ready)
    return second


def test_the_whole_flow_from_a_clip_to_the_exported_files_and_back(window, second_window, qtbot, dish_clip,
                                                                   tmp_path, monkeypatch, caplog):
    asked = record_every_dialog(monkeypatch)
    clip = own_copy(dish_clip, tmp_path / "videos")
    cu, cv, radius = clip.scene.dish
    run_folder = walk_through(window, qtbot, clip)
    assert run_folder == clip.path.parent / "dish_tracker_outline_Ada"  # SPEC 8.1's default

    # what the panels' slots and the one click put into the session
    session = window.controller.session
    assert (session.student, session.clip.start, session.clip.end, session.clip.step) == (NAME, 0, END, 2)
    assert (session.time.fps_true, session.time.source) == (240.0, "typed")
    assert session.calibration.stick == {"p1_px": list(STICK_ENDS[0]), "p2_px": list(STICK_ENDS[1]),
                                         "length_mm": STICK_MM}
    assert session.world_frame().k_mm_per_px == pytest.approx(K)
    assert session.circle.center_px == pytest.approx([cu, cv], abs=1e-6)
    assert session.circle.radius_px == pytest.approx(radius, abs=1e-6)
    assert session.axes.origin_px == session.circle.center_px and session.axes.angle_deg == 0.0
    (track,) = session.tracks
    (prompt,) = track.prompts
    (point,), u0, v0 = prompt.points_px, *center(clip, "A", 0)
    assert (track.id, prompt.frame, prompt.labels) == ("A", 0, [1])
    assert abs(point[0] - u0) < 1.0 and abs(point[1] - v0) < 1.0  # the click fell on a whole screen px
    assert session.complete and [(run.tracks, run.frames_done) for run in session.runs] == [(["A"], 11)]
    assert [panel.state for panel in window.panels[1:4]] == ["done", "done", "done"]  # time, calibration, dish
    assert [window.panels[number - 1].state for number in (6, 7, 9)] == ["done", "done", "done"]

    # the run tracked A on the 11 frames, where the scene has it
    arrays = results_of(window).arrays("A")
    u, v, _ = table_truth(clip, "A", FRAMES)
    assert arrays.frames.tolist() == FRAMES
    assert abs(arrays.u - u).max() < 0.01 and abs(arrays.v - v).max() < 0.01

    # Export all wrote every P0 file of SPEC 8.1, and nothing else
    assert {entry.name for entry in run_folder.iterdir()} == P0_FILES
    assert [entry.name for entry in (run_folder / "edgetam").iterdir()] == ["A.csv"]
    positions = table(run_folder / schema.POSITIONS_CSV)
    assert positions.track_id.tolist() == ["A"] * 11 and positions.frame.tolist() == FRAMES
    np.testing.assert_allclose(positions.t_s, np.array(FRAMES) / 240.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(positions.x_mm, K * (u - cu), rtol=0, atol=K * 0.01 + 1e-6)
    np.testing.assert_allclose(positions.y_mm, -K * (v - cv), rtol=0, atol=K * 0.01 + 1e-6)
    np.testing.assert_allclose(positions.u_px, u, rtol=0, atol=0.011)
    assert positions.visible.tolist() == [1] * 11 and positions["mode"].tolist() == ["coarse"] * 11
    shapes = table(run_folder / schema.SHAPES_CSV)
    assert shapes.frame.tolist() == FRAMES and shapes.area_mm2.gt(0).all()
    assert table(run_folder / schema.RADIAL_CSV).empty  # A is a coarse track: the header alone
    with np.load(run_folder / schema.OUTLINES_NPZ) as outlines:
        assert outlines.files == [schema.OUTLINES_META_KEY]
    (name, in_tracker_format), = read_tracker_export(run_folder / "edgetam" / "A.csv").items()
    assert name == "A" and in_tracker_format.frame.tolist() == FRAMES  # as last week's reader reads it
    np.testing.assert_allclose(in_tracker_format.pixelx, u, rtol=0, atol=0.011)
    overlay = probe(run_folder / schema.OVERLAY_MP4)
    assert overlay.n_frames == 11 and overlay.width % 2 == 0
    assert (run_folder / schema.README_TXT).read_text(encoding="utf-8").strip()
    log = (run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert "==== export, " in log and "overlay.mp4: " in log and "skipped" not in log
    assert no_error_recorded(window, asked, caplog)

    # the session reopens in a new window with everything restored
    window.close()
    on_disk = read_json(run_folder / schema.SESSION_JSON)
    second = reopen(second_window, qtbot, clip, run_folder)
    again = second.controller
    assert again.run_folder == run_folder and again.video_path == clip.path
    assert again.session.to_json() == on_disk
    first, time, calibration, dish = (body(second, number) for number in (1, 2, 3, 4))
    assert first.name_edit.text() == NAME and first.file_label.full_text == "dish_tracker.mp4"
    assert (first.start_box.value(), first.end_box.value(), first.step_box.value()) == (0, END, 2)
    assert time.fps_edit.text() == "240" and time.source_label.text() == "typed in"
    assert calibration.length_box.value() == STICK_MM
    assert calibration.stick_tool.scale().k_mm_per_px == pytest.approx(K)
    assert calibration.scale_value.text().startswith("100.00 µm/px")
    assert dish.center_value.text() == "160.3, 119.8 px" and dish.radius_value.text() == "10.80 mm (108.0 px)"
    assert dish.origin_value.text() == "160.3, 119.8 px"
    assert [panel.state for panel in second.panels[1:4]] == ["done", "done", "done"]
    objects, tracking, export = body(second, 6), track_panel(second), export_panel(second)
    assert objects.table.rowCount() == 1
    assert [objects.table.item(0, column).text() for column in (0, 1, 4)] == ["A", "coarse", "tracked"]
    assert second.panels[6].state == "done"
    assert second.panels[6].hint.text() == "Tracking is complete: 1 object, 11 frames."
    assert not tracking.track_button.isEnabled()  # nothing is left to track
    # the results are drawn on the frame shown, the clip's first
    assert second.view.frame == 0 and list(tracking.overlays.shown) == ["A"]
    assert tracking.overlays.shown["A"].center == pytest.approx((u[0], v[0]), abs=0.01)
    assert second.panels[8].state == "done" and second.panels[8].hint.text().startswith("Exported at ")
    assert export.export_button.isEnabled() and export.open_button.isEnabled()
    assert export.folder_label.full_text == "dish_tracker_outline_Ada"
    assert no_error_recorded(second, asked, caplog)
    assert {entry.name for entry in run_folder.iterdir()} == P0_FILES  # opening it wrote no other file


def test_the_flow_in_a_folder_with_a_space_and_accented_letters(window, second_window, qtbot, clip_in_odd_folder,
                                                                monkeypatch, caplog):
    asked = record_every_dialog(monkeypatch)
    clip = clip_in_odd_folder
    assert clip.path.parent.name == "vidéo test ü"
    run_folder = walk_through(window, qtbot, clip)
    assert run_folder == clip.path.parent / "dish_tracker_outline_Ada"
    assert {entry.name for entry in run_folder.iterdir()} == P0_FILES
    assert table(run_folder / schema.POSITIONS_CSV).frame.tolist() == FRAMES
    assert probe(run_folder / schema.OVERLAY_MP4).n_frames == 11
    assert export_panel(window).folder_label.toolTip() == str(run_folder)
    assert no_error_recorded(window, asked, caplog)

    window.close()
    second = reopen(second_window, qtbot, clip, run_folder)
    assert second.controller.video_path == clip.path and second.controller.run_folder == run_folder
    assert [track.id for track in second.controller.session.tracks] == ["A"]
    assert [second.panels[number - 1].state for number in (7, 9)] == ["done", "done"]
    assert no_error_recorded(second, asked, caplog)
