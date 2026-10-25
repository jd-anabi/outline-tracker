"""Tracking in the worker thread (SPEC 6.4, 10.2; decision X7; task C5): a job is
`tracking.run_job` in the window's worker, with the model the worker loaded.

What is checked, with stand-in segmenters and no torch:
- a job tracks every frame of the clip, reports its progress, and leaves results.npz and
  session.json as a command-line run leaves them;
- while the model works on a frame the window's event loop runs, and the model's call is in
  another thread than the GUI's; what a job reports arrives in the GUI thread;
- Cancel stops after the frame the model is in and keeps what was tracked; closing the window does
  the same and waits for the thread;
- session.json is written by the GUI thread only, results.npz by the worker only;
- every image the model is given can be written to.

What does not start, what fails, and what a job says (run.log) are in tests/gui/test_worker_jobs_log.py.

Expected values: the dish scene's clip has 120 frames, so a clip at step 2 has the 60 frames 0, 2,
..., 118; `ExactFake` answers with the scene's own masks, so a stored center is the centroid of
the true mask (tests/support/tracking_helpers.py, `table_truth`). A stand-in parked in its call for the
6th tracked frame has returned 5 frames; when the gate opens the 6th is stored, and a job that was
cancelled meanwhile stops before the 7th: frames 0, 2, 4, 6, 8, 10. A gate is a `threading.Event`
(tests/support/prompt_helpers.py); no test waits with a delay. In the two tests that close the window
the GUI thread waits in `close()`, so another thread opens the gate, once the worker has been told
to stop (its `stopping`).

Coordinates: px in Tracker's convention (SPEC 3.1). Frames are video frame numbers.
"""

import numpy as np
from PySide6.QtCore import QTimer

from outline_tracker.core import schema
from outline_tracker.tracking import job as tracking
from outline_tracker.gui.worker import worker_of
from outline_tracker.measure.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.core.session import Session
from prompt_helpers import Gate, gui_thread, this_thread
from session_helpers import body
from track_helpers import (NAME, Heard, Tracked, off_the_gui_thread, own_copy, parked_in_frame_6, ready_to_track,
                           results_of, run_to_end, session_on_disk)
from tracking_helpers import center, table_truth

GRID = list(range(0, 120, 2))  # the clips have 120 frames; a new session takes every 2nd
FIRST_SIX = [0, 2, 4, 6, 8, 10]


# ---------------------------------------------------------------------------------------------
# A job runs to its end


def test_a_job_tracks_every_frame_reports_progress_and_leaves_the_run_folder_complete(window, qtbot,
                                                                                    clip_in_odd_folder):
    clip = clip_in_odd_folder  # a folder with a space and letters that are not ASCII
    segmenter = Tracked(ExactFake(clip))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter, ids="ABC")
    heard = Heard(panel.jobs, window.controller)
    run_to_end(qtbot, panel)

    assert heard.finished == [("complete", "")]
    done = [call[0] for call in heard.progress]
    assert done == sorted(set(done)) and done[-1] == 60  # rising, and the last report is the last frame
    assert {call[1] for call in heard.progress} == {60}
    assert all(s_per_frame > 0 and eta_s >= 0 for _, _, s_per_frame, eta_s in heard.progress)
    assert heard.progress[-1][3] == 0  # nothing is left
    assert set(heard.threads) == {gui_thread()}  # every report arrived in the GUI thread
    assert off_the_gui_thread(segmenter)  # and the model worked in another one
    assert [what for what, *_ in segmenter.calls if what != "preview"] == ["start"] + ["step"] * 59

    assert window.controller.run_folder == clip.path.parent / f"dish_tracker_outline_{NAME}"
    store = results_of(window)
    assert store.track_ids == ["A", "B", "C"]
    for track_id in "ABC":
        arrays = store.arrays(track_id)
        u, v, area = table_truth(clip, track_id, GRID)
        assert arrays.frames.tolist() == GRID and arrays.visible.all()
        assert np.abs(arrays.u - u).max() < 0.01 and np.abs(arrays.v - v).max() < 0.01
        assert arrays.area_px.tolist() == area.tolist()

    # the session in the window is the one on disk: the run's record crossed to the GUI thread
    session, on_disk = window.controller.session, session_on_disk(window)
    assert session.complete is True and on_disk["complete"] is True
    (run,) = on_disk["runs"]
    assert (run["tracks"], run["start_frame"], run["mode"], run["frames_done"]) == (["A", "B", "C"], 0, "coarse", 60)
    assert run["finished"] is not None and run["seconds_per_frame"] > 0
    assert [(one.tracks, one.frames_done) for one in session.runs] == [(["A", "B", "C"], 60)]
    assert (panel.jobs.running, panel.jobs.status) == (False, "complete")


def test_the_session_is_saved_before_the_job_starts(window, qtbot, clip_in_odd_folder):
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder, Tracked(ExactFake(clip_in_odd_folder)), end=10)
    heard = Heard(panel.jobs, window.controller)
    run_to_end(qtbot, panel)
    assert heard.order[:2] == ["session saved", "started"]
    assert heard.order[-1] == "finished" and heard.order[-2] == "session saved"  # and when it has ended


# ---------------------------------------------------------------------------------------------
# The window stays usable; Cancel; closing the window


def test_while_the_model_is_in_a_frame_the_event_loop_runs_and_the_model_is_off_the_gui_thread(window, qtbot,
                                                                                             clip_in_odd_folder):
    with Gate() as gate:
        panel, segmenter, heard = parked_in_frame_6(window, qtbot, clip_in_odd_folder, gate)
        fired = []
        QTimer.singleShot(0, lambda: fired.append(this_thread()))  # posted now, while the worker is parked
        qtbot.waitUntil(lambda: fired != [])
        assert fired == [gui_thread()] and panel.jobs.running
        assert segmenter.tracked == 6 and gate.parked.is_set()  # still inside the call for the 6th frame
        what, thread, _, _ = segmenter.calls[-1]
        assert what == "step" and thread != gui_thread()
        # the reports of the five frames before it have arrived meanwhile
        qtbot.waitUntil(lambda: panel.jobs.done == 5)
        assert (panel.jobs.done, panel.jobs.total) == (5, 60)
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert heard.finished == [("complete", "")] and off_the_gui_thread(segmenter)


def test_cancel_while_the_model_is_in_the_6th_frame_keeps_exactly_six_frames(window, qtbot, clip_in_odd_folder):
    with Gate() as gate:
        panel, segmenter, heard = parked_in_frame_6(window, qtbot, clip_in_odd_folder, gate)
        panel.cancel_button.click()
        assert panel.jobs.running and panel.jobs.cancelling  # it stops after the frame the model is in
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert heard.finished == [("cancelled", "")]
    assert segmenter.tracked == 6
    assert results_of(window).arrays("A").frames.tolist() == FIRST_SIX
    on_disk = session_on_disk(window)
    assert on_disk["complete"] is False
    assert [(run["tracks"], run["frames_done"]) for run in on_disk["runs"]] == [(["A"], 6)]
    assert window.controller.session.complete is False
    assert worker_of(window).traces == []  # a cancelled job is no failure


def test_closing_the_window_during_a_run_cancels_it_and_ends_the_worker_thread(window, qtbot, clip_in_odd_folder):
    with Gate() as gate:
        panel, segmenter, heard = parked_in_frame_6(window, qtbot, clip_in_odd_folder, gate)
        worker, run_folder = worker_of(window), window.controller.run_folder
        opener = gate.open_when(worker.stopping)  # the GUI thread waits in close(): another thread opens the gate
        window.close()
        opener.join()
    assert not worker.is_running() and worker.state == "stopped"
    assert worker.traces == [] and segmenter.tracked == 6  # no error, and no 7th frame
    assert not panel.jobs.running and heard.finished == [("cancelled", "")]
    # what was tracked is kept, and the session says that it is not all
    assert ResultsStore.load(run_folder / schema.RESULTS_NPZ).arrays("A").frames.tolist() == FIRST_SIX
    saved = Session.load(run_folder / schema.SESSION_JSON)
    assert saved.complete is False and [(run.tracks, run.frames_done) for run in saved.runs] == [(["A"], 6)]


# ---------------------------------------------------------------------------------------------
# Who writes what; what the model is given


def test_session_json_is_written_by_the_gui_thread_and_results_npz_by_the_worker(window, qtbot, clip_in_odd_folder,
                                                                               monkeypatch):
    writers = {"session": [], "results": []}
    save_session, save_results = Session.save, ResultsStore.save

    def session_saved(self, path):
        writers["session"].append(this_thread())
        return save_session(self, path)

    def results_saved(self, path):
        writers["results"].append(this_thread())
        return save_results(self, path)

    monkeypatch.setattr(Session, "save", session_saved)
    monkeypatch.setattr(ResultsStore, "save", results_saved)
    monkeypatch.setattr(tracking, "AUTOSAVE_EVERY", 4)  # 11 frames: saved after 4 and 8 frames, and at the end
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder, Tracked(ExactFake(clip_in_odd_folder)), end=20)
    run_to_end(qtbot, panel)
    assert len(writers["results"]) == 3 and gui_thread() not in writers["results"]
    assert writers["session"] and set(writers["session"]) == {gui_thread()}
    assert session_on_disk(window)["runs"][0]["frames_done"] == 11


def test_every_image_the_model_is_given_can_be_written_to(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    segmenter = Tracked(ExactFake(clip))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter, end=10)
    assert not window.controller.source.get(0).flags.writeable  # the premise: the view's frames are read-only
    run_to_end(qtbot, panel)
    kinds = [what for what, *_ in segmenter.calls]
    assert kinds == ["preview", "start"] + ["step"] * 5
    assert all(writeable for _, _, writeable, _ in segmenter.calls)
    assert {shape for *_, shape in segmenter.calls} == {(240, 320, 3)}  # the whole frame each time


# ---------------------------------------------------------------------------------------------
# Where a video ended before its clip


def test_a_job_keeps_where_the_video_ended_and_it_says_nothing_about_the_video_opened_next(window, qtbot,
                                                                                         clip_in_odd_folder,
                                                                                         dish_clip, tmp_path):
    # the clip asks for frames 0 to 200 every 2; the video has frames 0 to 119, so the last tracked frame is 118
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), end=200)
    assert panel.jobs.video_end is None  # nobody has read the video to its end yet
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete" and panel.jobs.video_end == 118
    assert [(run.tracks, run.frames_done) for run in window.controller.session.runs] == [(["A"], 60)]
    window.open_path(own_copy(dish_clip, tmp_path / "another video").path)
    assert panel.jobs.video_end is None


def test_a_job_that_tracks_every_frame_of_its_clip_saw_no_end_of_the_video(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    with Gate() as gate:  # neither a job that ran to the clip's end, nor one that was cancelled before it
        segmenter = Tracked(ExactFake(clip), gate=gate, park_at={17})
        panel, _ = ready_to_track(window, qtbot, clip, segmenter, end=20)
        run_to_end(qtbot, panel)
        assert panel.jobs.status == "complete" and panel.jobs.video_end is None
        body(window, 6).add_object()
        assert body(window, 6).prompts.add_point(*center(clip, "B", 0), 1)
        qtbot.waitUntil(lambda: "B" in body(window, 6).prompts.outlines)
        panel.track()  # B: the 17th tracked frame of the stand-in is B's 6th
        qtbot.waitUntil(gate.parked.is_set)
        panel.cancel()
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert panel.jobs.status == "cancelled" and results_of(window).arrays("B").frames.tolist() == FIRST_SIX
    assert panel.jobs.video_end is None  # a run that was stopped says nothing about the video
