"""Panel 7, Track (SPEC 6.1, 6.3, 6.4, 10.1; task C5): a fine object, and the one line that says how the
run ended.

The estimate, Track before a run, the progress bar and Cancel are in tests/gui/test_tracking_panel.py.

Expected values are worked out by hand:
- the dish clip has 120 frames: a clip at step 2 has 60, and one that ends on frame 20 has 11; a
  clip that asks for frames 0 to 200 at step 2 has 101, of which the video has 0, 2, ..., 118: 60;
  an object clicked on frame 100 is tracked on 100, 102, ..., 118: 10 frames;
- the close-up scene's shrimp A is 64.2 px across on frame 0 (tests/tracking/test_tracking_fine.py), so its
  fine window is ceil(3 x 64.2) = 193 px, and on every later frame the window is centered on the
  shrimp's centroid of the tracked frame before, its corner rounded to whole px (SPEC 6.3).
The texts are the design note's. Nothing is read from a pixel of text, and no test waits with a
delay: a stand-in is parked on a gate (tests/support/prompt_helpers.py).

Coordinates: px in Tracker's convention (SPEC 3.1). Frames are video frame numbers.
"""

from export_helpers import calibrate, table
from gui_helpers import record_dialogs
from outline_tracker.core import schema
from outline_tracker.export.pipeline import export_all
from outline_tracker.core.fileio import new_name
from outline_tracker.tracking import job as tracking
from outline_tracker.measure.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import Gate, panel_with
from session_helpers import body
from track_helpers import (NAME, Saying, Tracked, hint, own_copy, ready_to_track, results_of, run_log, run_to_end,
                           session_on_disk, track_panel)
from tracking_helpers import assert_centered, center, table_truth

NOTE = "The stand-in changed its device on frame 4."  # said to the job's log under the run's line


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


# ---------------------------------------------------------------------------------------------
# What a job had to say is in the line about the run; a video that ends before the clip does


def test_a_video_that_ends_before_the_clip_is_tracked_to_its_last_frame_and_the_panel_says_so(window, qtbot,
                                                                                            clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), end=200)
    panel.seconds_per_frame = None
    panel.refresh()
    assert hint(window) == "No time estimate yet: 101 frames and 1 object are ready to track."
    run_to_end(qtbot, panel)

    assert panel.jobs.status == "complete"
    assert results_of(window).arrays("A").frames.tolist() == list(range(0, 120, 2))  # what the video has
    assert window.controller.session.complete is True and session_on_disk(window)["complete"] is True
    # the line about the run: complete, with the frames that were tracked, and where the video ended
    said = panel.message.text()
    assert panel.message.kind == "warning" and "\n" not in said
    assert said.startswith("Tracking is complete: 1 object, 60 frames. The video ended after frame 118")
    assert window.statusBar().currentMessage() == said
    assert "The video ended after frame 118" in run_log(window.controller.run_folder)
    # the panel: done, with nothing left for Track, and the bar ended full
    assert hint(window) == "Tracking is complete: 1 object, 60 frames."
    assert window.panels[6].state == "done"
    assert not panel.track_button.isEnabled() and panel.track_button.toolTip() == hint(window)
    assert (panel.progress_bar.value(), panel.progress_bar.maximum()) == (60, 60) and panel.progress_bar.isHidden()
    assert body(window, 6).table.item(0, 4).text() == "tracked"


def test_a_track_that_was_left_partial_is_still_named_after_a_job_that_met_the_videos_end(window, qtbot,
                                                                                        clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:  # A: cancelled in its 6th frame, so it has frames 0 to 10
        segmenter = Tracked(ExactFake(clip), gate=gate, park_at={6})
        panel, objects = ready_to_track(window, qtbot, clip, segmenter, end=200)
        panel.track()
        qtbot.waitUntil(gate.parked.is_set)
        panel.cancel()
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert hint(window) == "Tracking is not complete: A stops before the end of the clip."
    objects.add_object()  # B: tracked as far as the video goes, frame 118
    assert objects.prompts.add_point(*center(clip, "B", 0), 1)
    qtbot.waitUntil(lambda: "B" in objects.prompts.outlines)
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete" and panel.jobs.video_end == 118
    assert results_of(window).arrays("B").frames.tolist() == list(range(0, 120, 2))
    assert window.controller.session.complete is False  # A is still partial
    # B has every frame the video has: only A is named, and the line about the run does not say "complete"
    assert hint(window) == "Tracking is not complete: A stops before the end of the clip."
    said = panel.message.text()
    assert panel.message.kind == "warning" and "The video ended after frame 118" in said
    assert "Tracking is complete" not in said
    assert window.panels[6].state == "attention" and not panel.track_button.isEnabled()


def test_the_complete_line_counts_the_frames_that_were_tracked_not_the_frames_of_the_clip(window, qtbot,
                                                                                        clip_in_odd_folder):
    clip = clip_in_odd_folder
    window.controller.set_student(NAME)
    objects = panel_with(window, qtbot, clip, Tracked(ExactFake(clip)))
    window.controller.session.time.fps_true = clip.scene.fps
    window.controller.touch()
    window.show_frame(100)  # the object is clicked on frame 100: its track starts there
    objects.add_object()
    assert objects.prompts.add_point(*center(clip, "A", 100), 1)
    qtbot.waitUntil(lambda: "A" in objects.prompts.outlines)
    panel = track_panel(window)
    run_to_end(qtbot, panel)
    assert results_of(window).arrays("A").frames.tolist() == list(range(100, 120, 2))
    said = "Tracking is complete: 1 object, 10 frames."  # the clip has 60
    assert (panel.message.kind, panel.message.text()) == ("success", said)
    assert hint(window) == said and window.panels[6].state == "done"


def test_results_that_went_to_another_file_are_said_once_and_the_run_is_not_called_complete(window, qtbot,
                                                                                          clip_in_odd_folder,
                                                                                          monkeypatch):
    save = ResultsStore.save

    def held(self, path):  # as on Windows while another program has results.npz open (`fileio.atomic_write`)
        return save(self, new_name(path))

    monkeypatch.setattr(ResultsStore, "save", held)
    monkeypatch.setattr(tracking, "AUTOSAVE_EVERY", 4)  # 11 frames: saved after 4 and 8 frames, and at the end
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), end=20)
    run_to_end(qtbot, panel)
    folder = window.controller.run_folder
    assert panel.jobs.status == "complete"
    assert (folder / "results.new.npz").is_file() and not (folder / schema.RESULTS_NPZ).exists()
    said = panel.message.text()
    assert panel.message.kind == "warning" and "Tracking is complete" not in said
    assert said.count("results.npz is open in another program") == 1 and said.count("results.new.npz") == 1
    assert window.statusBar().currentMessage() == said
    assert run_log(folder).count("results.new.npz") == 3  # the log has every time it was said
    assert window.panels[6].state == "attention"  # the window cannot read these results yet


def test_a_note_of_a_cancelled_run_follows_what_was_kept(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:
        segmenter = Saying({3: f"  {NOTE}", 5: f"  {NOTE}"}, inner=ExactFake(clip), gate=gate, park_at={6})
        panel, _ = ready_to_track(window, qtbot, clip, segmenter)
        panel.track()
        qtbot.waitUntil(gate.parked.is_set)
        panel.cancel()
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    said = f"Tracking was cancelled at frame 12. The 6 tracked frames were kept. {NOTE}"  # said twice, shown once
    assert (panel.message.kind, panel.message.text()) == ("warning", said)
    assert run_log(window.controller.run_folder).count(NOTE) == 2


def test_a_note_of_a_failed_run_follows_the_reason_in_the_panel_and_in_the_dialog(window, qtbot, clip_in_odd_folder,
                                                                                monkeypatch):
    shown = record_dialogs(monkeypatch)
    clip = clip_in_odd_folder
    segmenter = Saying({2: f"  {NOTE}"}, inner=ExactFake(clip), fail_at=4,
                       error=RuntimeError("The model ran out of memory."))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter)
    run_to_end(qtbot, panel)
    said = panel.message.text()
    assert panel.message.kind == "problem" and said.startswith("Tracking stopped with an error. ")
    assert said.endswith(f" {NOTE}") and said.count("The model ran out of memory.") == 1
    assert " s per frame" not in said and "Run 1 of 1" not in said and "Traceback" not in said
    ((_, kind, text),) = shown.messages
    assert kind == "problem" and NOTE in text and text.count("The model ran out of memory.") == 1


def test_a_run_without_a_note_says_nothing_more_than_how_it_ended(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Saying({}, inner=ExactFake(clip)), end=20)
    run_to_end(qtbot, panel)
    assert panel.jobs.notes == []  # the run's line and its time are for run.log, not for the panel
    assert (panel.message.kind, panel.message.text()) == ("success", "Tracking is complete: 1 object, 11 frames.")
    assert " s per frame" in run_log(window.controller.run_folder)
