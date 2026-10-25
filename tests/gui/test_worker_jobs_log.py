"""Tracking in the worker thread (SPEC 6.4, 10.2; decision X7; task C5): what fails, what a job says.

What is checked, with stand-in segmenters and no torch:
- no job starts without a name or while another one runs, and a plain sentence says so;
- a failure is reported in one plain line, and its trace goes to run.log, never to the window;
- what a job says while it tracks (`Callbacks.log`) is in run.log of the job's run folder once,
  however the job ended, written by the GUI thread; the notes among it are kept for the panel.

A job that runs to its end, Cancel, closing the window, who writes what, and where a video ended before
its clip are in tests/gui/test_worker_jobs.py.

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

from gui_helpers import record_dialogs
from outline_tracker.core import schema
from outline_tracker.gui import worker_jobs
from outline_tracker.gui.worker import worker_of
from outline_tracker.gui.worker_jobs import jobs_of
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import Gate, gui_thread, this_thread
from session_helpers import body
from track_helpers import (NAME, Heard, Saying, Tracked, own_copy, parked_in_frame_6, ready_to_track, results_of,
                           run_log, run_to_end, session_on_disk)
from tracking_helpers import center

NOTE = "  The stand-in changed its device on frame 4."  # two blanks first: a note under a run's line


# ---------------------------------------------------------------------------------------------
# What does not start, and what fails


def test_without_a_name_nothing_starts_and_the_sentence_says_what_to_do(window, qtbot, clip_in_odd_folder):
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder)
    window.controller.session.student = ""
    window.controller.touch()
    heard = Heard(panel.jobs, window.controller)
    assert panel.jobs.start() == "Type your name in panel 1 first."
    assert not panel.jobs.running and "started" not in heard.order
    assert not list(clip_in_odd_folder.path.parent.glob("*/results.npz"))


def test_a_model_that_fails_on_a_frame_keeps_what_was_tracked_and_its_trace_goes_to_run_log(window, qtbot,
                                                                                         clip_in_odd_folder,
                                                                                         monkeypatch):
    shown = record_dialogs(monkeypatch)
    clip = clip_in_odd_folder
    error = RuntimeError("The model ran out of memory.\n  File \"somewhere.py\", line 1")
    segmenter = Tracked(ExactFake(clip), fail_at=3, error=error)
    panel, _ = ready_to_track(window, qtbot, clip, segmenter)
    heard = Heard(panel.jobs, window.controller)
    run_to_end(qtbot, panel)

    ((status, reason),) = heard.finished
    assert status == "failed" and "The model ran out of memory." in reason
    assert "Traceback" not in reason and "somewhere.py" not in reason and "\n" not in reason
    assert results_of(window).arrays("A").frames.tolist() == [0, 2]  # the two frames before it are kept
    assert session_on_disk(window)["complete"] is False
    # the trace: kept by the worker and written to run.log, not shown
    (trace,) = worker_of(window).traces
    assert "Traceback (most recent call last):" in trace and "RuntimeError: The model ran out of memory." in trace
    log = (window.controller.run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert trace in log and log.count("Traceback (most recent call last):") == 1
    (_, kind, text), = shown.messages
    assert kind == "problem" and "The model ran out of memory." in text
    for visible in (text, panel.message.text(), window.panels[6].hint.text(), window.statusBar().currentMessage()):
        assert "Traceback" not in visible and "somewhere.py" not in visible
    assert panel.message.kind == "problem" and "The model ran out of memory." in panel.message.text()


def test_a_job_that_tracking_refuses_is_reported_in_its_own_words(window, qtbot, clip_in_odd_folder, monkeypatch):
    record_dialogs(monkeypatch)
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder)
    window.controller.session.processing.core_open_frac = -1.0  # `run_job` refuses it before it tracks
    heard = Heard(panel.jobs, window.controller)
    assert panel.jobs.start() is None
    qtbot.waitUntil(lambda: not panel.jobs.running)
    ((status, reason),) = heard.finished
    assert status == "failed" and reason.startswith("core_open_frac must be a number that is 0 or more")
    assert not (window.controller.run_folder / schema.RESULTS_NPZ).exists()
    log = (window.controller.run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert "ValueError: core_open_frac must be a number" in log


def test_the_trace_of_a_model_that_cannot_be_loaded_goes_to_run_log_once_there_is_a_run_folder(window, qtbot,
                                                                                            clip_in_odd_folder,
                                                                                            monkeypatch):
    from gui_helpers import show

    shown = record_dialogs(monkeypatch)

    def factory(model, device):
        raise OSError("The model files could not be downloaded.")

    window.segmenter_factory = factory
    worker, jobs = worker_of(window), jobs_of(window)
    window.open_path(clip_in_odd_folder.path)  # no name yet: no run folder
    show(window, qtbot)
    qtbot.waitUntil(lambda: worker.state == "failed")
    assert window.controller.run_folder is None and len(worker.traces) == 1
    (_, kind, text), = shown.messages
    assert kind == "problem" and "The model files could not be downloaded." in text and "Traceback" not in text
    assert jobs.start() == "Type your name in panel 1 first."

    body(window, 1).name_edit.setText(NAME)
    window.controller.save_now()  # takes the name as typed; the run folder is made
    log = (window.controller.run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert worker.traces[0] in log and "OSError: The model files could not be downloaded." in log
    window.controller.save_now()
    again = (window.controller.run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert again == log  # written once
    assert jobs.start() == "The model could not be loaded. The model files could not be downloaded."


def test_the_trace_of_an_outline_that_could_not_be_made_goes_to_run_log(window, qtbot, clip_in_odd_folder,
                                                                      monkeypatch):
    from outline_tracker.segmenter.fake import ThresholdFake
    from prompt_helpers import panel_with

    shown = record_dialogs(monkeypatch)

    class Broken(ThresholdFake):
        def preview(self, image, prompts):
            raise RuntimeError("No memory for the outline.")

    window.controller.set_student(NAME)
    objects = panel_with(window, qtbot, clip_in_odd_folder, Broken())
    window.controller.save_now()
    objects.add_object()
    assert objects.prompts.add_point(*center(clip_in_odd_folder, "A", 0), 1)
    qtbot.waitUntil(lambda: not objects.prompts.busy)
    (_, kind, text), = shown.messages
    assert kind == "problem" and "No memory for the outline." in text and "Traceback" not in text
    log = (window.controller.run_folder / schema.RUN_LOG).read_text(encoding="utf-8")
    assert "RuntimeError: No memory for the outline." in log and "Traceback (most recent call last):" in log


def test_one_job_at_a_time(window, qtbot, clip_in_odd_folder):
    with Gate() as gate:
        panel, segmenter, heard = parked_in_frame_6(window, qtbot, clip_in_odd_folder, gate)
        assert panel.jobs.start() == "Tracking is running."
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert heard.order.count("started") == 1 and segmenter.tracked == 60


def test_a_copy_of_the_clip_is_what_the_helpers_track(tmp_path, dish_clip):
    copy = own_copy(dish_clip, tmp_path / "videos")
    assert copy.path.parent == tmp_path / "videos" and copy.path.read_bytes() == dish_clip.path.read_bytes()
    assert copy.scene is dish_clip.scene


# ---------------------------------------------------------------------------------------------
# What a job says goes to run.log, however the job ended


def parked_with_a_note(window, qtbot, clip, gate):
    """A job on the dish clip's object A whose stand-in said `NOTE` in its 3rd tracked frame (frame
    4) and is parked in its call for the 6th. Returns the panel."""
    segmenter = Saying({3: NOTE}, inner=ExactFake(clip), gate=gate, park_at={6})
    panel, _ = ready_to_track(window, qtbot, clip, segmenter)
    panel.track()
    qtbot.waitUntil(gate.parked.is_set)
    return panel


def test_what_a_job_says_is_in_run_log_when_it_completes_written_by_the_gui_thread(window, qtbot, clip_in_odd_folder,
                                                                                  monkeypatch):
    said, writers = [], []
    run_job, append_block = worker_jobs.run_job, worker_jobs.append_block

    def listening(job, callbacks):  # every line `run_job` gives its log, as it gives it
        log = callbacks.log
        callbacks.log = lambda text: (said.append(text), log(text))[1]
        return run_job(job, callbacks)

    def appended(path, lines):
        writers.append(this_thread())
        return append_block(path, lines)

    monkeypatch.setattr(worker_jobs, "run_job", listening)
    monkeypatch.setattr(worker_jobs, "append_block", appended)
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Saying({3: NOTE}, inner=ExactFake(clip)), end=20)
    folder = window.controller.run_folder
    assert run_log(folder) == ""  # nothing was said yet
    run_to_end(qtbot, panel)

    assert panel.jobs.status == "complete" and worker_of(window).traces == []
    assert NOTE in said and len(said) >= 3  # the premise: the run's line, the note, the run's time
    lines = run_log(folder).splitlines()
    (heading,) = [line for line in lines if line.startswith("Tracking in the window, ")]
    assert heading.endswith(" (complete):")
    assert lines[lines.index(heading) + 1:] == said  # every line, in its order, and nothing else
    assert "Traceback" not in run_log(folder)
    assert writers and set(writers) == {gui_thread()}
    assert panel.jobs.notes == [NOTE.strip()]  # not the run's line, and not its time


def test_what_a_cancelled_job_said_is_in_run_log_too(window, qtbot, clip_in_odd_folder):
    with Gate() as gate:
        panel = parked_with_a_note(window, qtbot, clip_in_odd_folder, gate)
        panel.cancel()
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert panel.jobs.status == "cancelled" and worker_of(window).traces == []
    log = run_log(window.controller.run_folder)
    assert log.count("Tracking in the window, ") == 1 and " (cancelled):\n" in log
    assert log.count(NOTE) == 1 and "Traceback" not in log
    assert panel.jobs.notes == [NOTE.strip()]


def test_closing_the_window_during_a_run_leaves_what_the_job_said_in_run_log(window, qtbot, clip_in_odd_folder):
    with Gate() as gate:
        panel = parked_with_a_note(window, qtbot, clip_in_odd_folder, gate)
        worker, run_folder = worker_of(window), window.controller.run_folder
        opener = gate.open_when(worker.stopping)  # the GUI thread waits in close(): another thread opens the gate
        window.close()
        opener.join()
    assert not panel.jobs.running and panel.jobs.status == "cancelled"
    log = run_log(run_folder)
    assert " (cancelled):\n" in log and log.count(NOTE) == 1


def test_what_a_failed_job_said_is_in_run_log_once_with_its_trace(window, qtbot, clip_in_odd_folder, monkeypatch):
    record_dialogs(monkeypatch)
    clip = clip_in_odd_folder
    segmenter = Saying({2: NOTE}, inner=ExactFake(clip), fail_at=4, error=RuntimeError("The model ran out of memory."))
    panel, _ = ready_to_track(window, qtbot, clip, segmenter)
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "failed" and "The model ran out of memory." in panel.jobs.reason
    log = run_log(window.controller.run_folder)
    assert log.count(NOTE) == 1 and log.count("Traceback (most recent call last):") == 1
    assert log.index(NOTE) < log.index("Traceback (most recent call last):")  # in the order they were said
    assert panel.jobs.notes == [NOTE.strip()]  # the failure's own line is the reason, not one more note


def test_a_run_log_that_cannot_be_written_when_the_job_ends_is_written_at_the_next_save(window, qtbot,
                                                                                      clip_in_odd_folder,
                                                                                      monkeypatch):
    held = [True]
    append_block = worker_jobs.append_block

    def appended(path, lines):
        if held[0]:
            raise PermissionError(f"{path} is open in another program.")
        return append_block(path, lines)

    monkeypatch.setattr(worker_jobs, "append_block", appended)
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Saying({3: NOTE}, inner=ExactFake(clip)), end=20)
    run_to_end(qtbot, panel)
    folder = window.controller.run_folder
    assert panel.jobs.status == "complete" and run_log(folder) == ""
    assert panel.jobs.notes == [NOTE.strip()]  # the panel has the note whatever happens to the file
    held[0] = False
    window.controller.save_now()
    log = run_log(folder)
    assert log.count(NOTE) == 1 and " (complete):\n" in log
    window.controller.save_now()
    assert run_log(folder) == log  # written once


def test_what_a_job_said_goes_to_its_own_run_folder_when_another_video_was_opened_meanwhile(window, qtbot,
                                                                                          clip_in_odd_folder,
                                                                                          dish_clip, tmp_path):
    other = own_copy(dish_clip, tmp_path / "another video")
    with Gate() as gate:
        panel = parked_with_a_note(window, qtbot, clip_in_odd_folder, gate)
        first = window.controller.run_folder
        window.open_path(other.path)  # this cancels the run
        assert window.controller.run_folder not in (None, first)
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert panel.jobs.status == "cancelled"
    assert run_log(first).count(NOTE) == 1 and " (cancelled):\n" in run_log(first)
    assert NOTE not in run_log(window.controller.run_folder)
