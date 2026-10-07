"""Panel 7, Track (SPEC 6.1, 6.3, 6.4, 10.1; task C5): the estimate before a run, Track, the
progress bar with s/frame and the time left, Cancel, and one line that says how the run ended.

Expected values are worked out by hand:
- the estimate is s per frame of one object x frames x (1 + 0.5 per extra object), for each run:
  3 coarse objects on 60 frames at 2.0 s are 2.0 x 60 x 2 = 240 s, "about 4 min"; a coarse run of 2
  objects and a fine run of 1, 60 frames each, are 2.0 x 60 x 1.5 + 2.0 x 60 = 300 s, "about 5 min";
- a run of 3 objects that took 3.0 s per frame means 3.0 / 2 = 1.5 s for one object;
- the dish clip has 120 frames: a clip at step 2 has 60, and one that ends on frame 20 has 11;
- the close-up scene's shrimp A is 64.2 px across on frame 0 (tests/test_tracking_fine.py), so its
  fine window is ceil(3 x 64.2) = 193 px, and on every later frame the window is centered on the
  shrimp's centroid of the tracked frame before, its corner rounded to whole px (SPEC 6.3).
The texts are the design note's. Nothing is read from a pixel of text, and no test waits with a
delay: a stand-in is parked on a gate (tests/gui/prompt_helpers.py).

Coordinates: px in Tracker's convention (SPEC 3.1). Frames are video frame numbers.
"""

import pytest
from PySide6.QtWidgets import QProgressBar, QPushButton

from export_helpers import calibrate, table
from gui_helpers import record_dialogs, show
from outline_tracker import schema
from outline_tracker.export import export_all
from outline_tracker.gui import estimate
from outline_tracker.gui.panels.track_panel import TrackPanel
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from outline_tracker.session import RunRecord
from outline_tracker.tracking_plan import RunPlan
from prompt_helpers import Gate
from session_helpers import body
from track_helpers import NAME, Tracked, own_copy, ready_to_track, results_of, run_to_end, track_panel
from tracking_helpers import assert_centered, center, table_truth

FULL = (0, 0, 320, 240)


def hint(window) -> str:
    return window.panels[6].hint.text()


def plan(ids, mode="coarse", frames=range(0, 120, 2)) -> RunPlan:
    return RunPlan(tuple(ids), frames[0], mode, frames, FULL)


# ---------------------------------------------------------------------------------------------
# The estimate, by hand


def test_the_estimate_is_time_per_frame_times_frames_times_half_more_per_extra_object():
    assert estimate.estimate_seconds(2.0, [plan("ABC")]) == pytest.approx(240.0)
    assert estimate.estimate_seconds(2.0, [plan("A")]) == pytest.approx(120.0)
    assert estimate.estimate_seconds(2.0, [plan("AB"), plan("C", "fine")]) == pytest.approx(300.0)
    assert estimate.estimate_seconds(0.5, [plan("A", frames=range(0, 20, 2))]) == pytest.approx(5.0)
    assert estimate.frames_and_objects([plan("AB"), plan("C", "fine")]) == (120, 3)
    assert estimate.frames_and_objects([]) == (0, 0)


def test_the_time_of_one_object_from_a_frame_that_held_several():
    assert estimate.seconds_per_object_frame(3.0, 3) == pytest.approx(1.5)
    assert estimate.seconds_per_object_frame(0.8, 1) == pytest.approx(0.8)
    assert estimate.seconds_per_object_frame(4.5, 10) == pytest.approx(4.5 / 5.5)


def test_the_last_run_of_a_session_gives_the_time_per_frame():
    runs = [RunRecord(tracks=["A"], frames_done=60, seconds_per_frame=0.4),
            RunRecord(tracks=["A", "B", "C"], frames_done=60, seconds_per_frame=3.0),
            RunRecord(tracks=["D"], frames_done=0, seconds_per_frame=0.0)]  # stopped before its first frame
    assert estimate.last_run_seconds(runs) == pytest.approx(1.5)
    assert estimate.last_run_seconds(runs[:1]) == pytest.approx(0.4)
    assert estimate.last_run_seconds([]) is None and estimate.last_run_seconds(runs[2:]) is None


@pytest.mark.parametrize("seconds, text", [
    (240.0, "about 4 min"), (300.0, "about 5 min"), (89.0, "about 1 min"), (91.0, "about 2 min"),
    (29.0, "under 1 min"), (5.0, "under 1 min"), (7200.0, "about 120 min"),
])
def test_the_estimate_in_words(seconds, text):
    assert estimate.about(seconds) == text


@pytest.mark.parametrize("seconds, text", [
    (150.0, "2 min 30 s"), (59.4, "59 s"), (60.0, "1 min 0 s"), (0.0, "0 s"), (3725.0, "62 min 5 s"),
])
def test_the_time_left_in_words(seconds, text):
    assert estimate.time_left(seconds) == text


def test_the_progress_line_is_the_design_notes():
    assert estimate.progress_text(1, 2, 240, 600, 0.42, 150.0) == (
        "Run 1 of 2 · frame 240 of 600 · 0.42 s/frame · time left (ETA) 2 min 30 s")


# ---------------------------------------------------------------------------------------------
# The panel before a run


def test_the_panels_controls_are_in_panel_7_under_its_hint(window):
    panel = track_panel(window)
    assert isinstance(panel, TrackPanel) and body(window, 7) is panel
    assert isinstance(panel.track_button, QPushButton) and panel.track_button.text() == "Track"
    assert isinstance(panel.cancel_button, QPushButton) and panel.cancel_button.text() == "Cancel"
    assert isinstance(panel.progress_bar, QProgressBar) and not panel.progress_bar.isTextVisible()
    assert panel.track_button.property("kind") == "primary" and panel.cancel_button.property("kind") is None
    assert not panel.track_button.autoDefault() and not panel.cancel_button.autoDefault()
    assert not panel.cancel_button.isEnabled()


def test_track_is_off_with_the_reason_as_its_hint_until_everything_is_there(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel = track_panel(window)

    def off_because(reason):
        return (not panel.track_button.isEnabled() and hint(window) == reason
                and panel.track_button.toolTip() == reason)

    panel.refresh()
    assert off_because("Type your name in panel 1 first.")
    window.controller.set_student(NAME)
    panel.refresh()
    assert off_because("Open your video in panel 1 first.")
    with Gate() as gate:
        def factory(model, device):
            gate.park()
            return Tracked(ExactFake(clip))

        window.segmenter_factory = factory
        window.open_path(clip.path)
        show(window, qtbot)
        assert off_because("Type the true frame rate (fps_true) in panel 2 (Time) first.")
        window.controller.session.time.fps_true = clip.scene.fps
        window.controller.touch()
        assert off_because("Add an object in panel 6 first.")
        objects = body(window, 6)
        objects.add_object()
        assert off_because("Add an object in panel 6 first.")  # an object without a point is not one to track
        assert objects.prompts.add_point(*center(clip, "A", 0), 1)
        qtbot.waitUntil(gate.parked.is_set)
        assert off_because("The model is loading. Track is ready when the model is ready.")
        gate.open()
        qtbot.waitUntil(lambda: panel.worker.ready)
    qtbot.waitUntil(panel.track_button.isEnabled)
    assert panel.track_button.toolTip() == "Track the objects that have points (panel 6)"
    assert window.panels[6].state == "todo"


def test_the_estimate_is_shown_before_the_run_starts(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    segmenter = Tracked(ExactFake(clip))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter, ids="ABC")
    panel.seconds_per_frame = 2.0
    panel.refresh()
    assert hint(window) == "Estimated time: about 4 min for 60 frames and 3 objects."
    assert panel.track_button.isEnabled() and not panel.cancel_button.isEnabled()
    assert not panel.jobs.running and segmenter.tracked == 0  # nothing has started
    # one object, 11 frames, 0.5 s each: 5.5 s
    window.controller.session.tracks[1:] = []
    window.controller.session.clip.end = 20
    panel.seconds_per_frame = 0.5
    window.controller.touch()
    assert hint(window) == "Estimated time: under 1 min for 11 frames and 1 object."


def test_without_a_time_per_frame_the_hint_still_says_what_would_be_tracked(window, qtbot, clip_in_odd_folder):
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder, Tracked(ExactFake(clip_in_odd_folder)), ids="AB")
    panel.seconds_per_frame = None
    panel.refresh()
    assert hint(window) == "No time estimate yet: 60 frames and 2 objects are ready to track."
    assert panel.track_button.isEnabled()


def test_the_time_an_outline_took_becomes_the_time_per_frame(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    now = [100.0]
    with Gate() as gate:
        segmenter = Tracked(ExactFake(clip))
        panel, objects = ready_to_track(window, qtbot, clip, segmenter)
        panel.clock = lambda: now[0]
        preview = segmenter.inner.preview

        def parked_preview(image, prompts):
            gate.park()
            return preview(image, prompts)

        segmenter.inner.preview = parked_preview
        objects.add_object()
        assert objects.prompts.add_point(*center(clip, "B", 0), 1)  # asked at 100.0 s, for A and B
        qtbot.waitUntil(gate.parked.is_set)
        now[0] = 103.0
        gate.open()
        qtbot.waitUntil(lambda: not objects.prompts.busy)
    # 3.0 s for a frame with two objects: 3.0 / 1.5 = 2.0 s for one; 60 frames of both: 2.0 x 60 x 1.5 = 180 s
    assert panel.seconds_per_frame == pytest.approx(2.0)
    assert hint(window) == "Estimated time: about 3 min for 60 frames and 2 objects."


# ---------------------------------------------------------------------------------------------
# During a run, and after it


def test_during_a_run_the_bar_and_the_line_show_the_progress_and_cancel_can_be_pressed(window, qtbot,
                                                                                     clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:
        segmenter = Tracked(ExactFake(clip), gate=gate, park_at={6})
        panel, _ = ready_to_track(window, qtbot, clip, segmenter)
        assert panel.progress_bar.isHidden() and panel.progress_label.isHidden()
        panel.track_button.click()
        qtbot.waitUntil(gate.parked.is_set)
        qtbot.waitUntil(lambda: panel.progress_bar.value() == 5)  # five frames are done, the 6th is with the model
        assert (panel.progress_bar.minimum(), panel.progress_bar.maximum()) == (0, 60)
        assert not panel.progress_bar.isHidden() and not panel.progress_label.isHidden()
        qtbot.waitUntil(lambda: panel.progress_label.text().startswith("Run 1 of 1 · frame 5 of 60 · "))
        assert " s/frame · time left (ETA) " in panel.progress_label.text()
        # Track stays where it is, off; Cancel is on
        assert not panel.track_button.isEnabled() and panel.track_button.toolTip() == "Tracking is running."
        assert panel.cancel_button.isEnabled()
        assert hint(window) == "Tracking is running. You can look at other frames meanwhile."
        panel.cancel_button.click()
        assert not panel.cancel_button.isEnabled()
        assert panel.progress_label.text() == "Stopping after the current frame."
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    # frames 0 to 10 were tracked; frame 12 is the first that was not
    said = "Tracking was cancelled at frame 12. The 6 tracked frames were kept."
    assert (panel.message.kind, panel.message.text()) == ("warning", said)
    assert window.statusBar().currentMessage() == said
    assert panel.progress_bar.isHidden() and panel.progress_label.isHidden()
    assert not panel.cancel_button.isEnabled()
    assert window.panels[6].state == "attention"
    assert hint(window) == "Tracking is not complete: A stops before the end of the clip."
    assert not panel.track_button.isEnabled()  # nothing is left that Track could start
    assert results_of(window).arrays("A").frames.tolist() == [0, 2, 4, 6, 8, 10]


def test_after_a_run_that_reached_the_end_the_panel_is_done(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="ABC", end=20)
    run_to_end(qtbot, panel)
    said = "Tracking is complete: 3 objects, 11 frames."
    assert (panel.message.kind, panel.message.text()) == ("success", said)
    assert window.panels[6].state == "done" and hint(window) == said
    assert not panel.track_button.isEnabled() and not panel.cancel_button.isEnabled()
    assert panel.progress_bar.isHidden()
    # the run's own time per frame is the next estimate's: one coarse run of three objects
    (run,) = window.controller.session.runs
    assert panel.seconds_per_frame == pytest.approx(run.seconds_per_frame / 2)
    # the object table of panel 6 follows
    statuses = [body(window, 6).table.item(row, 4).text() for row in range(3)]
    assert statuses == ["tracked"] * 3


def test_a_failure_is_one_plain_line_in_the_panel_and_a_dialog(window, qtbot, clip_in_odd_folder, monkeypatch):
    shown = record_dialogs(monkeypatch)
    clip = clip_in_odd_folder
    segmenter = Tracked(ExactFake(clip), fail_at=4, error=RuntimeError("The model ran out of memory."))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter)
    run_to_end(qtbot, panel)
    assert panel.message.kind == "problem"
    said = panel.message.text()  # the 4th tracked frame is frame 6
    assert said.startswith("Tracking stopped with an error. ") and "\n" not in said
    assert "frame 6" in said and "The model ran out of memory." in said and "Traceback" not in said
    ((parent, kind, text),) = shown.messages
    assert parent is window and kind == "problem"
    heading, _, rest = text.partition("\n")
    assert heading == "Tracking stopped with an error"
    assert "The model ran out of memory." in rest and "run.log" in rest and "Traceback" not in rest
    assert window.panels[6].state == "attention" and panel.progress_bar.isHidden()
    assert results_of(window).arrays("A").frames.tolist() == [0, 2, 4]


def test_a_model_that_cannot_be_loaded_is_said_in_the_panel_and_track_stays_off(window, qtbot, clip_in_odd_folder,
                                                                              monkeypatch):
    shown = record_dialogs(monkeypatch)

    def factory(model, device):
        raise OSError("The model files could not be downloaded.")

    window.segmenter_factory = factory
    window.controller.set_student(NAME)
    window.open_path(clip_in_odd_folder.path)
    show(window, qtbot)
    panel = track_panel(window)
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    window.controller.session.time.fps_true = 240.0
    objects = body(window, 6)
    objects.add_object()
    assert objects.prompts.add_point(*center(clip_in_odd_folder, "A", 0), 1)
    reason = "The model could not be loaded. The model files could not be downloaded."
    assert hint(window) == reason and not panel.track_button.isEnabled()
    assert [(kind, text.partition("\n")[0]) for _, kind, text in shown.messages] == [
        ("problem", "The model could not be loaded")]
    assert "The model files could not be downloaded." in shown.messages[0][2]


def test_a_window_made_without_a_model_shows_no_dialog(window, qtbot, monkeypatch):
    shown = record_dialogs(monkeypatch)
    window.segmenter_factory = None  # as `MainWindow()` has it
    show(window, qtbot)
    panel = track_panel(window)
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    assert shown.messages == [] and not panel.track_button.isEnabled()


# ---------------------------------------------------------------------------------------------
# A fine object


def test_an_object_switched_to_fine_runs_through_the_fine_runner(window, qtbot, closeup_clip, tmp_path):
    clip = own_copy(closeup_clip, tmp_path / "close up")
    segmenter = Tracked(ExactFake(clip))
    panel, objects = ready_to_track(window, qtbot, clip, segmenter, end=20)
    objects.mode_box.setCurrentText("fine")  # the Mode box under the object table
    assert window.controller.session.tracks[0].mode == "fine"
    assert body(window, 6).table.item(0, 1).text() == "fine"
    calibrate(window.controller.session)
    window.controller.touch()
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete"

    frames = list(range(0, 21, 2))
    arrays = results_of(window).arrays("A")
    assert arrays.frames.tolist() == frames and arrays.mode.tolist() == ["fine"] * 11
    # the model was shown the 193 px window on every tracked frame, and one grid cell is 193 / 256 px
    tracked = [shape for what, _, _, shape in segmenter.calls if what in ("start", "step")]
    assert tracked == [(193, 193, 3)] * 11
    assert arrays.cell_px.tolist() == [193 / 256] * 11
    u, v, _ = table_truth(clip, "A", frames)
    assert abs(arrays.u - u).max() < 0.01 and abs(arrays.v - v).max() < 0.01
    # the window the job chose crossed to the session in the window, and is on disk
    (track,) = window.controller.session.tracks
    assert track.fine_window_px == 193
    (run,) = window.controller.session.runs
    assert (run.tracks, run.mode, run.frames_done) == (["A"], "fine", 11)

    # the square the model saw on frame 10 is drawn: 193 px, around the centroid of frame 8
    window.show_frame(10)
    drawn = panel.overlays.shown["A"]
    assert_centered((10, drawn.box[:2], drawn.box[2:]), 193, (u[4], v[4]))
    c0, r0, side, _ = drawn.box
    corners = [(c0, r0), (c0 + side, r0), (c0 + side, r0 + side), (c0, r0 + side), (c0, r0)]
    assert [tuple(point) for point in zip(*drawn.window.getData())] == corners
    # on the track's first frame, around the object where it was found there
    window.show_frame(0)
    first = panel.overlays.shown["A"]
    assert_centered((0, first.box[:2], first.box[2:]), 193, (u[0], v[0]))

    # Export gives the fine track its rows in radial.csv
    window.controller.save_now()
    export_all(window.controller.run_folder, log=lambda text: None)
    radial = table(window.controller.run_folder / schema.RADIAL_CSV)
    assert radial[radial.track_id == "A"].frame.tolist() == frames
    assert radial.filter(like="r_").notna().any(axis=1).all()  # every row holds radii


def test_a_coarse_object_has_no_window_square(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), end=10)
    run_to_end(qtbot, panel)
    window.show_frame(4)
    drawn = panel.overlays.shown["A"]
    assert drawn.box is None and drawn.window is None


def test_a_stand_in_without_ground_truth_tracks_too(window, qtbot, disk_clip, tmp_path):
    clip = own_copy(disk_clip, tmp_path / "disks")
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ThresholdFake()), ids="AB", end=10)
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete"
    store = results_of(window)
    for track_id in "AB":
        arrays = store.arrays(track_id)
        for row, frame in enumerate(range(0, 11, 2)):
            cu, cv = center(clip, track_id, frame)
            assert abs(arrays.u[row] - cu) < 0.25 and abs(arrays.v[row] - cv) < 0.25
